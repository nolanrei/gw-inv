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

include("waccm-utils.jl")
include("wrf-utils.jl")
include("L81-2.jl")
include("L81-autograd-compatible.jl")

cmin = -60
cmax = 60
cstride = 1
ndirs = 1
srcs = [4]#, 32, 48]   # ~18km
wavespds = Array(cmin:cstride:cmax)
kw = 2pi/5e5

z = 10.25:0.5:64.75
nz = length(z)

#N = ncread("/scratch/nr2489/work/loose_spectra/waccm_rho_and_N.nc","N")
#rho = ncread("/scratch/nr2489/work/loose_spectra/waccm_rho_and_N.nc","rho")
N = 2e-2*ones(nz)
rho = 1.22*exp.(-z/7)  #scale height set to 7km

waves = make_waveset(wavespds, ndirs, srcs; ref_flux=1e-3)
nw = length(waves);

######################################################################################
##
######################################################################################

dz = 500*ones(nz)       # each cell is 500km wide
V0 = zeros(nz)
U1 = 10.0 .- 50/(z[end]-z[1])*(-z[1].+z)
col1 = ColumnProfile(U1, V0, N, rho, nz, dz)

ref_flux = 1e-3
dir = [1.0;0.0]
th = pi/3
dir2 = [cos(th);sin(th)]
wavspd = -12.0
k = (2pi)/5e4    # 50km wavelength
usrc = col1.U[srcs[1]]*dir[1] + col1.V[srcs[1]]*dir[2]
testflux = 0.5*ref_flux
testsrc = srcs[1]
testwave = c_to_wave(wavspd, dir, k, testflux, testsrc)

momdep_tst1 = zeros(2nz)
L81_flux!(testwave,col1,testflux,momdep_tst1)

err = randn(2nz)
sigma = 0.05*maximum(abs.(momdep_tst1))   # error scales with momdep_tst1
momdep1 = momdep_tst1 .+ sigma.*err

######################################################################################
##
######################################################################################

function nl_obj(x,p)
    propag!,waves,col,b,lambda,BIG,Gi,J = p
    Gi .= 0.0
    J .= 0.0
    for i in eachindex(waves)
        # we don't use the Jacobian here but might as well make it accurate
        propag!(waves[i],col,x[i],Gi,view(J,:,i))
        #println(Gi)
    end
    Gi .-= b
    # optimization needs the numbers to be larger
    return BIG*(Gi'*Gi + lambda*sum(exp.(x))) - 1e-9*sum(x)     # L1 regularization and a log-barrier to get it to stay away from 0
end

function nl_obj_grad!(G,x,p)
    # computes gradient of nl_obj w/ respect to x in G
    # J must be size nz x nw
    propag!,waves,col,b,lambda,BIG,Gi,J = p
    # need to clean up Gi and J storage arrays
    Gi .= 0.0
    J .= 0.0
    for i in eachindex(waves)
        # additively builds up Gi into g(S;X) while building up jacobian
        v = view(J,:,i)
        propag!(waves[i],col,x[i],Gi,v)
        #v .*= exp.(x[i])
    end
    Gi .-= b

    G .= -1e-9 .+ BIG*(lambda*exp.(x) .+ 2*J'*Gi)
    return nothing
end

function nl_obj_hess!(H,x,p)
    # computes Hessian of nl_obj w/ respect to x in H
    # J must be size nz x nw
    propag!,waves,col,b,lambda,BIG,Gi,J = p
    # need to clean up Gi and J storage arrays
    Gi .= 0.0
    J .= 0.0
    nw = length(waves)
    for i = 1:nw
        # additively builds up Gi into g(S;X) while building up jacobian
        # but we don't need g(S;X) here
        propag!(waves[i],col,x[i],Gi,view(J,:,i))
    end
    H .= 2*BIG*(J'*J)
    for i = 1:nw
        H[i,i] += BIG*lambda*exp(x[i])
    end
    return nothing
end

lambda = 1e-4
BIG = 1e6
nw = length(waves)
nz = length(col1.U)
J = zeros(2nz,nw)
H = zeros(nw,nw)
gi= zeros(2nz)
p = (L81_exp_grad!,waves,col1,momdep1,lambda,BIG,gi,J)

################ First plot: objective function as a function of
## wave flux
fx = -15:0.2:3
nfs = length(fx)
Is = zeros(nfs,nfs)
Js = zeros(nw,nfs,nfs)
tmp = zeros(nw)
for j = 1:nfs
    for i = 1:nfs
        tmp[1] = fx[i]   # wavespeed -10
        tmp[2] = fx[j]   # wavespeed 10
        Is[i,j] = nl_obj(tmp, p)
        nl_obj_grad!(Js[:,i,j], tmp, p)
    end
end

heatmap(fx, fx, log.(Is)', xlabel="β1, c = 0", ylabel="β2, c = 10")
savefig("Figures/obj_v_2flux.png")
#rescaled = [real(Complex(j)^0.2)*sign(j) for j in Js];
#quiver(fx, fx, quiver=(Js[1,:,:], Js[2,:,:]), xlabel="β1, c = 0", ylabel="β2, c = 10")
#savefig("Figures/obj_v_2grad.png")

dx = zeros(nw)
xnew = zeros(nw)
df = zeros(nw)   # gradient
x0 = -10*ones(nw)

include("mini-newton.jl")
x = 0.0 .+ x0
steps = newton_solve!(x, nl_obj, nl_obj_grad!, nl_obj_hess!, df, H, dx, xnew, p; reltol=1e-5, abstol=1e-10, max_newton_steps=20, keep_steps=true)

# plot reconstructed solution
recon = spectrum_to_momdep(L81_flux!,waves,col1,exp.(x))
plot(1e3*momdep1[1:nz],z,xlabel="flux (mPa)",ylabel="z (km)",label="truth")
plot!(1e3*recon[1:nz],z,label="x")
savefig("Figures/1d_nw_test.png")

heatmap(fx, fx, log.(Is)', xlabel="β1, c = 0", ylabel="β2, c = 10")
#I_steps = [nl_obj(steps[:,i],p) for i = 1:size(steps,2)]
plot!(steps[1,:],steps[2,:],label="optimization path")
savefig("Figures/2d_optim_path.png")