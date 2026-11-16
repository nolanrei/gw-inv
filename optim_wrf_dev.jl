using LinearAlgebra
using SparseArrays
using StaticArrays
using Random
using Plots
using NetCDF
using Profile
using CSV
using DataFrames
using Printf
using Optimization, OptimizationOptimJL, OptimizationIpopt
using Zygote, ReverseDiff

######################################################################################
##                  Dataset Stuff
######################################################################################
include("waccm-utils.jl")
include("wrf-utils.jl")
include("L81-2.jl")

cases = [8]#[1,2,3,4,5,6,7,8,9,10]
files = wrf_get_fileset(cases)
vars = wrf_load(files)
wrf_postprocess!(vars)

vars_to_be_avgd = ["Up", "Vp", "DX", "DY", "UW", "VW"]
tstamps,idx2case,case_order = wrf_tstamps_cases(files["U"])
order = wrf_order(tstamps,idx2case,case_order)
for var in vars_to_be_avgd
    vars[var] = wrf_knit(vars[var],order)
end
drag_means!(vars)

cmin = -100
cmax = 100
cstride = 5
ndirs = 2
srcs = [4]#, 32, 48]   # ~18km
wavespds = Array(cmin:cstride:cmax)
kw = 2pi/5e5

waves = make_waveset(wavespds, ndirs, srcs; ref_flux=1e-3, u=nothing,v=nothing,N2=nothing)
nw = length(waves);

z = vars["oz"];

wts = (vcat(vars["rho"],vars["rho"]))*1.0
sw = sum(wts)
wts[1:16] .= 0.0
wts[111:116] .= 0.0
wts ./= (sw/length(wts));

######################################################################################
##                  Loading a particular column
######################################################################################
nx,ny,nz,nt = size(vars["Up"][1])

xi = rand(1:nx)
yi = rand(1:ny)
ti = rand(1:nt)
fi = 1

nlev = size(vars["DX"][fi],3)
nz = length(z)
dz = 500*ones(nz)       # each cell is 500km wide
col = ColumnProfile(vars["Up"][fi][xi,yi,:,ti], vars["Vp"][fi][xi,yi,:,ti], vars["N"], vars["rho"], nz, dz)

momdep_col = cat(dims=1,vars["DX"][fi][xi,yi,:,ti], vars["DY"][fi][xi,yi,:,ti]) # concatenate x- and y-fluxes

ref_flux = 1e-3
dir = [1.0;0.0]
dir2 = [0.0;1.0]
wavspd = 10.0
k = (2pi)/5e4    # 50km wavelength
usrc = [col.U[srcs[1]], col.V[srcs[1]]]
testwave = c_to_wave(wavspd*dir[1], wavspd*dir[2], dir, k, usrc, col.N[srcs[1]]^2)
testwave.flux = ref_flux
testwave.src = srcs[1]

momdep_tst = zeros(2nz)
L81_wp!(testwave,col,momdep_tst)
err = randn(2nz)

######################################################################################
##                  Optimization Stuff
######################################################################################

function ls_obj(x,p)
    A,b,lambda = p
    tmp = b .- A*x
    return tmp'*tmp + lambda*sum(x)     # L1 regularization
end

function make_GX(propag!, waves, col)
    # construct the matrix
    nz = 2*size(col.U,1)         # size of propag output vectors
    nw = length(waves)          # number of waves we're considering

    sparsity_factor = 0.35
    nnz = Int(ceil(sparsity_factor*nz*nw))  # bound on Number of NonZeros in GX -- sparsity factor * total numel
    
    densecol = zeros(nz)
    J = zeros(Int64,nnz)        # column indices of nonzeros
    I = zeros(Int64,nnz)         # row indices of nonzeros
    V = zeros(Float64,nnz)       # values of nonzeros
    
    nz_true = 0
    for idx = 1:nw
        densecol .= 0.0
        propag!(waves[idx], col, densecol)
        nz_true = append_col_to_spmat!(I,J,V,idx,nz_true,densecol)
    end
    sGX = sparse(I[1:nz_true],J[1:nz_true],V[1:nz_true],nz,nw)
    return sGX
end

function make_dense_GX(propag!, waves, col)
    # construct the matrix
    nz = 2*size(col.U,1)         # size of propag output vectors
    nw = length(waves)          # number of waves we're considering

    GX = zeros(nz,nw)
    densecol = zeros(nz)
    
    for idx = 1:nw
        densecol .= 0.0
        propag!(waves[idx], col, densecol)
        GX[:,idx] .= densecol
    end

    return GX
end

######################################################################################
##                  Problem Setup (WRF)
######################################################################################

sGX = make_GX(L81_wp!, waves,col)

sigma = 0.1*maximum(abs.(momdep_tst))    # magnitude of Gaussian error = 0.1*momdep magnitude

lambda = 1e-8
momdep = momdep_tst.+sigma*err
p = (sGX, momdep, lambda)
x0 = lsqr(sGX, momdep; damp=lambda) # has negative numbers (out of domain) by default
# let's get rid of those o-o-d's
eps = 1e-10
function eps_relu(x,eps)
    return (x > eps ? x : eps)
end
for i = 1:length(x0)
    x0[i] = eps_relu(x0[i],eps)
end

cons(res,x,p) = (res.=x)       # constraint function -- here just x since we want x \geq 0

optf = OptimizationFunction(ls_obj, Optimization.AutoReverseDiff(compile=true), cons=cons)
prob = OptimizationProblem(optf, x0, p, lcons = zeros(nw), ucons = Inf*ones(nw))    # lower bound 0, upper bound Inf

sol = solve(prob, IpoptOptimizer(
           hessian_approximation = "limited-memory",
           limited_memory_max_history = 10); verbose=5);

plot(sol.u,label="recon"); savefig("testfig.png")