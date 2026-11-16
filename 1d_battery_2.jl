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
#using Optimization, OptimizationMOI, OptimizationOptimJL, Ipopt
#using ReverseDiff, Zygote
#using ModelingToolkit      # currently malfunctioning

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
cstride = 1
ndirs = 2       ####################### 1D tests == disjoint X and Y problems
srcs = [4]#, 32, 48]   # ~12km
wavespds = Array(cmin:cstride:cmax)
waves = make_waveset(wavespds, ndirs, srcs)
nws = length(wavespds)
nw = length(waves);

z = vars["oz"];

wts = (vcat(vars["rho"],vars["rho"]))*1.0
sw = sum(wts)
wts[1:16] .= 0.0
wts[111:116] .= 0.0
wts ./= (sw/length(wts));

######################################################################################
##                  Lovading a particular column
######################################################################################
nx,ny,nz,nt = size(vars["Up"][1])

xi = rand(1:nx)
yi = rand(1:ny)
ti = rand(1:nt)
fi = 1

nlev = size(vars["DX"][fi],3)
nz = length(z)
col = ColumnProfile(vars["Up"][fi][xi,yi,:,ti], vars["Vp"][fi][xi,yi,:,ti], vars["N"], vars["rho"], nz)

momdep_col = cat(dims=1,vars["DX"][fi][xi,yi,:,ti], vars["DY"][fi][xi,yi,:,ti]) # concatenate x- and y-fluxes

ref_flux = 1e-3
dir = [1.0;0.0]
dir2 = [0.0;1.0]
wavspd = 10.0
k = 1e6/(2pi)
c = SVector(wavspd*dir[1], wavspd*dir[2], 10.0)
testwave = WaveProfile(2pi/3000, 0.0, 0.0, 0.0, c, ref_flux, srcs[1])

momdep_tst = zeros(2nz)
L81_wp!(testwave,col,momdep_tst)
err = randn(2nz)
sigma = 0.1*maximum(abs.(momdep_tst))    # magnitude of Gaussian error = 0.1*momdep magnitude

######################################################################################
##                  Testing reconstruction and making figures
######################################################################################

############################ test wave, no err ###################################
spectrum = lin_invert(L81_wp!, waves, momdep_tst, col; lam=1e-4, reg="l2", refine=1);

recalc_momdep = spectrum_to_momdep(L81_wp!, waves, col, spectrum)
plot(momdep_tst[1:nz],z,xlabel="zonal drag (m/s^2)",ylabel="z (km)",label="truth")
plot!(recalc_momdep[1:nz],z,xlabel="zonal drag (m/s^2)",ylabel="z (km)",label="reconstruction")
savefig("1d_pm_drag.png")

testspec = zeros(nws)
idx = Int(div(nws,2)+1+wavspd)
testspec[idx] = ref_flux/waves[idx].flux
plot(wavespds,testspec,label="original",title="X spectrum",xlabel="phase speed (m/s)",ylabel="flux (mPa)")
plot!(wavespds,spectrum[1:nws],label="reconstructed")
savefig("1d_pm_spec.png")

############################ test wave, additive err ###############################
momdep_w_err = momdep_tst.+sigma*err
spec_w_err = lin_invert(L81_wp!, waves, momdep_w_err, col; lam=1e-2, reg="l2", refine=1);

rmom_w_err = spectrum_to_momdep(L81_wp!, waves, col, spec_w_err)
plot(momdep_w_err[1:nz],z,xlabel="zonal drag (m/s^2)",ylabel="z (km)",label="truth")
plot!(rmom_w_err[1:nz],z,xlabel="zonal drag (m/s^2)",ylabel="z (km)",label="reconstruction")
savefig("1d_pm_drag_err2.png")

testspec = zeros(nws)
idx = Int(div(nws,2)+1+wavspd)
testspec[idx] = ref_flux/waves[idx].flux
plot(wavespds,testspec,label="original",title="X spectrum, additive error",xlabel="phase speed (m/s)",ylabel="flux (mPa)")
plot!(wavespds,spec_w_err[1:nws],label="reconstructed")
savefig("1d_pm_spec_err.png")

############################ test wave, additive err, L1 reg ########################
momdep_w_err = momdep_tst.+sigma*err
specl1_w_err = lin_invert(L81_wp!, waves, momdep_w_err, col; reg="l1", wts);

rmoml1_w_err = spectrum_to_momdep(L81_wp!, waves, col, spec_w_err)
plot(momdep_w_err[1:nz],z,xlabel="zonal drag (m/s^2)",ylabel="z (km)",label="truth")
plot!(rmoml1_w_err[1:nz],z,xlabel="zonal drag (m/s^2)",ylabel="z (km)",label="l1 recon")
savefig("1d_pm_drag_err_l1.png")

testspec = zeros(nws)
idx = Int(div(nws,2)+1+wavspd)
testspec[idx] = ref_flux/waves[idx].flux
plot(wavespds,testspec,label="original",title="X spectrum, additive error",xlabel="phase speed (m/s)",ylabel="flux (mPa)")
plot!(wavespds,specl1_w_err[1:nws],label="l1 recon")
savefig("1d_pm_spec_err_l1.png")

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
        clear!(densecol)
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

sGX = make_GX(L81_wp!, waves,col)
GX = make_dense_GX(L81_wp!, waves,col)

#lambda = 1e-2
#p = (sGX, momdep_tst.+sigma*err, lambda)
#x0 = 1e-7*ones(nw)

#cons(res,x,p) = (res.=x)       # constraint function -- here just x since we want x \geq 0

#optf = OptimizationFunction(ls_obj, Optimization.AutoReverseDiff(), cons=cons)
#prob = OptimizationProblem(optf, x0, p, lcons = zeros(nw), ucons = Inf*ones(nw))    # lower bound 0, upper bound Inf

#sol = solve(prob, IPNewton());

