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

include("waccm-utils.jl")
include("wrf-utils.jl")
include("L81-2.jl")
include("L81-autograd-compatible.jl")

cmin = -100
cmax = 100
cstride = 5
ndirs = 1
srcs = [4]#, 32, 48]   # ~18km
wavespds = [-5.0, 5.0] #Array(cmin:cstride:cmax)
kw = 2pi/5e5

z = 10.25:0.5:64.75
nz = length(z)

#N = ncread("/scratch/nr2489/work/loose_spectra/waccm_rho_and_N.nc","N")
#rho = ncread("/scratch/nr2489/work/loose_spectra/waccm_rho_and_N.nc","rho")
N = 2e-2*ones(nz)
rho = 1.22*exp.(-z/7)  #scale height set to 7km

waves = make_waveset(wavespds, ndirs, srcs; ref_flux=1e-3)
nw = length(waves);

wts = (vcat(rho,rho))*1.0
sw = sum(wts)
wts[1:16] .= 0.0
wts[111:116] .= 0.0
wts ./= (sw/length(wts));

######################################################################################
##
######################################################################################

dz = 500*ones(nz)       # each cell is 500km wide
U0 = 5*ones(nz)
U1 = -10.0 .+ 50/(z[end]-z[1])*(-z[1].+z)
U2 = 39/(z[end]-z[1])^2 * (-z[1].+z).*(z[end].-z)
U3 = 30.0 .- 50/(z[end]-z[1])*(-z[1].+z)
V0 = zeros(nz)
col0 = ColumnProfile(U0, V0, N, rho, nz, dz)
col1 = ColumnProfile(U1, V0, N, rho, nz, dz)
col2 = ColumnProfile(U2, V0, N, rho, nz, dz)
col3 = ColumnProfile(U3, V0, N, rho, nz, dz)
# col4 is col1 rotated by an angle th = 60deg
th = pi/3
col4 = ColumnProfile(U1*cos(th), U1*sin(th), N, rho, nz, dz)

ref_flux = 1e-3
dir = [1.0;0.0]
dir2 = [cos(th);sin(th)]
wavspd = 5.0
k = (2pi)/5e4    # 50km wavelength
usrc = col0.U[srcs[1]]*dir[1] + col0.V[srcs[1]]*dir[2]
testflux = ref_flux
testsrc = srcs[1]
testwave = c_to_wave(wavspd, dir, k, testflux, testsrc)
testwave2 = c_to_wave(wavspd, dir2, k, testflux, testsrc)
prettify_wave!(testwave,usrc,N[testsrc])

function split_waves(wave0, Dc, nwaves)
    k = hypot(wave0.k,wave0.l)
    dir = [wave0.k/k, wave0.l/k]
    wavspd = dir'*wave0.c[1:2]
    dc = Dc/(nwaves-1)
    Dc2 = 0.5*Dc
    out = Vector{WaveProfile}(undef,nwaves)
    for i = 1:nwaves
        out[i] = c_to_wave(wavspd-Dc2+(i-1)*dc, dir, k, wave0.flux/nwaves, wave0.src)
    end
    f = wave0.flux/nwaves*ones(nwaves)
    return out,f
end
spec2,f2 = split_waves(testwave,2,2)
spec4,f4 = split_waves(testwave,2,4)

err = randn(2nz)

z2 = [z z]

momdep_tst0 = zeros(2nz)
L81_flux!(testwave,col0,ref_flux,momdep_tst0)
plot(col0.U,z,label="U",xaxis="m/s or 0.1 mPa",yaxis="z (km)")
plot!(testwave.c[1]*ones(nz),z,label="wave speed")
plot!(1e4*momdep_tst0[1:nz],z,label="fluxdep")
savefig("testfig0.png")

momdep_tst1 = zeros(2nz)
L81_flux!(testwave,col1,ref_flux,momdep_tst1)
fd2_1 = spectrum_to_momdep(L81_flux!,spec2,col1,f2)
fd4_1 = spectrum_to_momdep(L81_flux!,spec4,col1,f4)
plot(col1.U,z,label="U")
plot!(testwave.c[1]*ones(nz),z,label="wave speed",xaxis="m/s or 0.1 mPa",yaxis="z (km)")
plot!(1e4*momdep_tst1[1:nz],z,label="fluxdep")
#plot!(1e4*fd2_1[1:nz],z,label="fluxdep 2w")
#plot!(1e4*fd4_1[1:nz],z,label="fluxdep 4w")
savefig("testfig1.png")

momdep_tst2 = zeros(2nz)
L81_flux!(testwave,col2,ref_flux,momdep_tst2)
plot(col2.U,z,label="U")
plot!(testwave.c[1]*ones(nz),z,label="wave speed",xaxis="m/s or 0.1 mPa",yaxis="z (km)")
plot!(1e4*momdep_tst2[1:nz],z,label="fluxdep")
savefig("testfig2.png")

momdep_tst3 = zeros(2nz)
L81_flux!(testwave,col3,ref_flux,momdep_tst3)
fd2_3 = spectrum_to_momdep(L81_flux!,spec2,col3,f2)
fd4_3 = spectrum_to_momdep(L81_flux!,spec4,col3,f4)
plot(col3.U,z,label="U")
plot!(testwave.c[1]*ones(nz),z,label="wave speed",xaxis="m/s or 0.1 mPa",yaxis="z (km)")
plot!(1e4*momdep_tst3[1:nz],z,label="fluxdep")
#plot!(1e4*fd2_3[1:nz],z,label="fluxdep 2w")
#plot!(1e4*fd4_3[1:nz],z,label="fluxdep 4w")
savefig("testfig3.png")

momdep_tst4 = zeros(2nz)
L81_flux!(testwave2,col4,ref_flux,momdep_tst4)
plot([col4.U  col4.V],z2,label=["U" "V"],layout=2)
plot!(testwave.c[1]*ones(nz,2),z2,label="wave speed",xaxis="m/s or 0.1 mPa",yaxis="z (km)",layout=2)
plot!(1e4*reshape(momdep_tst4,(nz,2)),z2,label=["fluxdep X" "fluxdep Y"],layout=2)
savefig("testfig4.png")

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

dx = zeros(nw)
xnew = zeros(nw)
df = zeros(nw)   # gradient
x0 = -10*ones(nw)

include("mini-newton.jl")
x = 0.0 .+ x0
steps = newton_solve!(x, nl_obj, nl_obj_grad!, nl_obj_hess!, df, H, dx, xnew, p; reltol=1e-5, abstol=1e-10, max_newton_steps=20, keep_steps=true)