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

cmin = -60
cmax = 60
cstride = 2
ndirs = 1
srcs = [4]#, 32, 48]   # ~18km
wavespds = Array(cmin:cstride:cmax)
kw = 2pi/5e5
nws = length(wavespds)

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
V0 = zeros(nz)
#U1 = -10.0 .+ 50/(z[end]-z[1])*(-z[1].+z)
#U1 = -30.0 .+ 60/(0.25*(z[end]-z[1])^2)*(-z[1].+z).*(z[end].-z)
U1 = 30*zeros(nz)
z_ref = 2pi*(0:1/nz:1-1/nz)
Uc = 50*sin.(z_ref)
Vc = 50*cos.(z_ref)
# f_ushape = x -> 50*sin(x)
f_ushape = x -> 20*exp(x/(2*pi))*sin(3*x)
Uwack = f_ushape.(z_ref)
col1 = ColumnProfile(Uwack, V0, N, rho, nz, dz)

ref_flux = 1e-3
dir = [1.0;0.0]
th = 2pi*rand()
dir2 = [cos(th);sin(th)]
wavspd = 20
k = (2pi)/5e4    # 50km wavelength
testflux = 0.5*ref_flux
testsrc = srcs[1]
testwave = c_to_wave(wavspd, dir2, k, testflux, testsrc)

momdep_tst1 = zeros(2nz)
L81_flux!(testwave,col1,testflux,momdep_tst1)

#####################################
# five random waves
#####################################

ntst = 5
test_wvsp = -40 .+(40-(-40))*rand(ntst)
test_fxs = 0.8*ref_flux*rand(ntst)
test_waves = Vector{WaveProfile}(undef,ntst)
for i = 1:ntst
    th = 0 #2pi*rand()  # going back to 1d for a sec
    test_waves[i] = c_to_wave(test_wvsp[i], [cos(th);sin(th)], k, test_fxs[i], testsrc)
end
recon_5waves = spectrum_to_momdep(L81_flux!,test_waves,col1,test_fxs)

err = randn(2nz)
sigma = 0.05*maximum(abs.(momdep_tst1))   # error scales with momdep_tst1
momdep1 = momdep_tst1 .+ sigma.*err
recon_5w_err = recon_5waves .+ sigma*err

#####################################
# Gaussian wavepacket
#####################################
wdth = 10+3*randn()
mn = 30*randn()
fluxsum = 1e-3
gauss_fn = c -> fluxsum/sqrt(pi)/wdth*exp( -((c-mn)/wdth)^2 )
wp_fxs = gauss_fn.(wavespds)
recon_wp = spectrum_to_momdep(L81_flux!,waves,col1,wp_fxs)

s2 = 0.05*maximum(abs.(recon_wp))
recon_wp_err = recon_wp .+ s2*err

#=
#####################################
plot!(ones(nz)*test_wvsp',z,label=["wave 1" "wave 2" "wave 3" "wave 4" "wave 5"])
savefig("Figures/quadratic_plus_waves.png")
=#
plot([col1.U col1.V],z,xlabel=["U (m/s)" "V (m/s)"],ylabel="z (km)",label="mean flow",layout=2)
#plot!(ones(nz,1)*wavspd*dir2',z,label="wave speed")
savefig("Figures/U_V.png")

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
    return BIG*(Gi'*Gi + lambda*sum(exp.(x))) #- 1e-9*sum(x)
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

    G .= BIG*(lambda*exp.(x) .+ 2*J'*Gi)
    return nothing
end

function nl_obj_hess!(H,x,p)
    # computes Hessian of nl_obj w/ respect to x in H
    # J must be size nz x nw
    propag!,waves,col,b,lambda,BIG,Gi,J = p
    # need to clean up Gi and J storage arrays
    eps = 1e-10
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
        H[i,i] += BIG*lambda*exp(x[i]) + eps
    end
    return nothing
end

lambda = 1e-5
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
#=
steps = newton_solve!(x, nl_obj, nl_obj_grad!, nl_obj_hess!, df, H, dx, xnew, p; reltol=1e-5, abstol=1e-10, max_newton_steps=30, error_on_max_steps=false, keep_steps=true)

# plot reconstruction
recon = spectrum_to_momdep(L81_flux!,waves,col1,exp.(x))
plot(1e3*reshape(momdep_tst1,(nz,2)), z, xlabel="flux (mPa)", ylabel="z (km)", label="truth", layout=2)
plot!(1e3*reshape(momdep1,(nz,2)), z, label="data",layout=2)
plot!(1e3*reshape(recon,(nz,2)),z,label="recon",layout=2)
savefig("Figures/spiral_recon.png")

# plot spectrum
lgi = zeros(Int64,0)
for i = 1:nw
    if x[i] > -10
        push!(lgi,i)
    end
end
lgx = [waves[i].c[1] for i in lgi]
lgy = [waves[i].c[2] for i in lgi]
scatter(lgx,lgy,xlabel="cx (m/s)",ylabel="cy (m/s)",label="reconstructed waves")
plot!([0.0, testwave.c[1]],[0.0, testwave.c[2]],label="original wave")
savefig("Figures/1w_2dir_scatter.png")

nlg = length(lgi)
wbw = zeros(2nz,nlg)
for i = 1:nlg
    L81_flux!(waves[lgi[i]],col1,exp(x[lgi[i]]),view(wbw,:,i))
end
plot(1e3*wbw[1:nz,:],z,xlabel="x flux (mPa)",ylabel="z (km)",label=["w1" "w2" "w3"])
savefig("Figures/recon_wbw_bkdn_x.png")
plot(1e3*wbw[nz+1:2nz,:],z,xlabel="y flux (mPa)",ylabel="z (km)",label=["w1" "w2" "w3"])
savefig("Figures/recon_wbw_bkdn_y.png")

la_1d_spec = zeros(5,2)
wx = wavspd*dir2[1]
wy = wavspd*dir2[2]
la_1d_spec_x = [wavespds[1] wx-1e-3 wx wx+1e-3 wavespds[end];
                wavespds[1] wy-1e-3 wy wy+1e-3 wavespds[end]]'
la_1d_spec[3,1:2] = testflux*dir2
plot(la_1d_spec_x,1e3*la_1d_spec,label="true spectrum",xlabel="wave speed (m/s)",ylabel="flux (mPa)",layout=2)
plot!(wavespds,1e3*reshape(exp.(x),(nws,2)),label="reconstructed spectrum", layout=2)
savefig("Figures/spiral_spec.png")
=#

# and then again for the 5 waves

x2 = 0.0 .+ x0
p2 = (L81_exp_grad!,waves,col1,recon_5w_err,lambda,BIG,gi,J)
newton_solve!(x2, nl_obj, nl_obj_grad!, nl_obj_hess!, df, H, dx, xnew, p2; reltol=1e-5, abstol=1e-10, max_newton_steps=60, error_on_max_steps=false)

recon2 = spectrum_to_momdep(L81_flux!,waves,col1,exp.(x2))
plot(1e3*reshape(recon_5waves,(nz,2)),z,xlabel="flux (mPa)", ylabel="z (km)", label="truth", layout=2)
plot!(1e3*reshape(recon_5w_err,(nz,2)),z,label="data",layout=2)
plot!(1e3*reshape(recon2,(nz,2)),z,label="recon",layout=2)
savefig("Figures/1d_5wtest.png")

#### Compare test spectrum to reconstructed spectrum
# Construct an array that will look like test_spectrum when plotted
lookalike_vx = zeros(3*ntst)
lookalike_hx = zeros(3*ntst)
tiny = 1e-3
test_cx = [test_waves[i].c[1] for i=1:ntst]
px = sortperm(test_cx)    # sorting permutation of test wavespeeds
fx = [test_waves[px[i]].flux for i=1:ntst]  # permuted fluxes
lookalike_vx[2:3:end-1] .= 1e3*fx
lookalike_hx[2:3:end-1] .= test_cx[px]
lookalike_hx[1:3:end-2] .= -tiny .+ test_cx[px]
lookalike_hx[3:3:end]   .=  tiny .+ test_cx[px]
plot(lookalike_hx,lookalike_vx,xlabel="wavespeed (m/s)",ylabel="flux (mPa)",label="test spectrum")
plot!(wavespds,1e3*exp.(x2),label="reconstructed spectrum")
savefig("Figures/1d_5w_spectrum.png")

#########################################################
# wavepacket
#########################################################
x3 = 0.0 .+ x0
p3 = (L81_exp_grad!,waves,col1,recon_wp_err,lambda,BIG,gi,J)
newton_solve!(x3, nl_obj, nl_obj_grad!, nl_obj_hess!, df, H, dx, xnew, p3; reltol=1e-5, abstol=1e-10, max_newton_steps=60, error_on_max_steps=false)

recon3 = spectrum_to_momdep(L81_flux!,waves,col1,exp.(x3))
plot(1e3*reshape(recon_wp,(nz,2)),z,xlabel="flux (mPa)", ylabel="z (km)", label="truth", layout=2)
plot!(1e3*reshape(recon_wp_err,(nz,2)),z,label="data",layout=2)
plot!(1e3*reshape(recon3,(nz,2)),z,label="recon",layout=2)
savefig("Figures/1d_wavepacket_test.png")

# wavepacket plots
plot(wavespds,1e3*wp_fxs,xlabel="wavespeed (m/s)",ylabel="flux (mPa)",label="test spectrum")
plot!(wavespds,1e3*exp.(x3),label="reconstructed spectrum")
savefig("Figures/1d_wp_spectrum.png")

#=
# 2d stuff
lookalike_vx = zeros(3*ntst)
lookalike_hx = zeros(3*ntst)
lookalike_vy = zeros(3*ntst)
lookalike_hy = zeros(3*ntst)
tiny = 1e-3
test_cx = [test_waves[i].c[1] for i=1:ntst]
test_cy = [test_waves[i].c[2] for i=1:ntst]
px = sortperm(test_cx)    # sorting permutation of test wavespeeds
py = sortperm(test_cy)
fx = [test_fxs[j]*test_waves[j].k/hypot(test_waves[j].k,test_waves[j].l) for j in px]
fy = [test_fxs[j]*test_waves[j].l/hypot(test_waves[j].k,test_waves[j].l) for j in py]
lookalike_vx[2:3:end-1] .= 1e3*fx
lookalike_hx[2:3:end-1] .= test_cx[px]
lookalike_hx[1:3:end-2] .= -tiny .+ test_cx[px]
lookalike_hx[3:3:end]   .=  tiny .+ test_cx[px]
lookalike_vy[2:3:end-1] .= 1e3*fy
lookalike_hy[2:3:end-1] .= test_cy[py]
lookalike_hy[1:3:end-2] .= -tiny .+ test_cy[py]
lookalike_hy[3:3:end]   .=  tiny .+ test_cy[py]
plot([lookalike_hx lookalike_hy],[lookalike_vx lookalike_vy],xlabel="wavespeed (m/s)",ylabel="flux (mPa)",label="test spectrum",layout=2)
proj = project_onto_XY(wavespds, waves, exp.(x2))
plot!(wavespds,proj,label="reconstructed spectrum",layout=2)
savefig("Figures/2d_5w_spectrum.png")
=#