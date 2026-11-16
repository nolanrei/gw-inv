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
include("mini-newton.jl")
include("inversion.jl")

######################################################################################
##
######################################################################################

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

lambda = 1e-5
BIG = 1e6
nw = length(waves)
J = zeros(2nz,nw)
H = zeros(nw,nw)
gi= zeros(2nz)

######################################################################################
##
######################################################################################

dz = 500*ones(nz)       # each cell is 500km wide
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

ref_flux = 1e-3
k = (2pi)/5e4    # 50km wavelength

dx = zeros(nw)
xnew = zeros(nw)
df = zeros(nw)     # gradient
x0 = -10*ones(nw)  # starting condition

# want to have col accessible outside the loop for debugging
col = ColumnProfile(vars["Up"][1][1,1,:,1], vars["Vp"][1][1,1,:,1], vars["N"], vars["rho"], nz, dz)

#####################################
# five random waves -- fixed seed
#####################################

seed = -8854870600175104528
ntst = 5
test_wvsp = -40 .+(40-(-40))*rand(Xoshiro(seed),ntst)
test_fxs = 0.8*ref_flux*rand(Xoshiro(seed+1),ntst)
test_waves = Vector{WaveProfile}(undef,ntst)
for i = 1:ntst
    th = 0 #2pi*rand()  # going back to 1d for a sec
    test_waves[i] = c_to_wave(test_wvsp[i], [cos(th);sin(th)], k, test_fxs[i], srcs[1])
end

nprof = 1000       ### number of profiles to test statistics with
x2 = zeros(nw)    # starting guess
sols2 = zeros(nw,nprof)    # where the solutions go for statistics-ing with
recon_5waves = zeros(2nz)

z_err = zeros(2nz)          # sum over idx of (recon_5waves(z) .- reconstructed flux(z)).^2
z_rms = zeros(2nz)          # sum of recon_5waves(z).^2
tmp = zeros(2nz)            # space to work in

for idx = 1:nprof
    sz = size(vars["Up"][1])
    xi = rand(1:sz[1])
    yi = rand(1:sz[2])
    ti = rand(1:sz[4])
    col.U .= vars["Up"][1][xi,yi,:,ti]
    col.V .= vars["Vp"][1][xi,yi,:,ti]
    recon_5waves .= spectrum_to_momdep(L81_flux!,test_waves,col,test_fxs)
    
    err = randn(2nz)
    sigma = 0.05*maximum(abs.(recon_5waves))   # error scales with signal
    recon_5w_err = recon_5waves .+ sigma*err
    
    x2 .= x0    # reinitialize to a good starting guess
    p2 = (L81_exp_grad!,waves,col,recon_5w_err,lambda,BIG,gi,J)
    newton_solve!(x2, nl_obj, nl_obj_grad!, nl_obj_hess!, df, H, dx, xnew, p2; 
        reltol=1e-5, abstol=1e-10, max_newton_steps=60, error_on_max_steps=false,
        verbose=false, armijo_max_steps_no_error=true)
    sols2[:,idx] .= x2

    tmp .= spectrum_to_momdep(L81_flux!,waves,col,exp.(x2))
    z_err .+= (tmp.-recon_5waves).^2
    z_rms .+= (recon_5w_err).^2
end

# now we store the mean of the solutions in x2
x2 .= mean(exp.(sols2),dims=2)

z_err .= sqrt.(z_err/nprof)
z_rms .= sqrt.(z_rms/nprof)

#### Compare test spectrum to reconstructed spectrum
# Construct an array that will look like test_spectrum when plotted
#begin
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
plot!(wavespds,1e3*x2,label="reconstructed spectrum")
savefig("Figures/statistics/1d_5w_spectrum.png")

plot(reshape(z_err./z_rms,(nz,2)),z,xlabel="rms flux error / rms flux",ylabel="z (km)",label=["X" "Y"],layout=2)
savefig("Figures/statistics/1d_5w_err.png")
#end

# Debuggleshooting
#=
begin
recon2 = spectrum_to_momdep(L81_flux!,waves,col,exp.(x2))
plot(1e3*reshape(recon_5waves,(nz,2)),z,xlabel="flux (mPa)", ylabel="z (km)", label="truth", layout=2)
plot!(1e3*reshape(recon2,(nz,2)),z,label="recon",layout=2)
savefig("Figures/1d_5wtest.png")
end
=#

#####################################
# five random waves drawn from Gaussian distribution -- change from iteration to iteration
#####################################

ntst = 5
test_waves = Vector{WaveProfile}(undef,ntst)

# hyperparameters for Gaussian distribution
sd = 4.0         # stdev of distrib
mn = 0.0         # mean of distrib

nprof = 800       ### number of profiles to test statistics with
x1 = zeros(nw)    # starting guess
sols1 = zeros(nw,nprof)    # where the solutions go for statistics-ing with
recon_5waves = zeros(2nz)

for idx = 1:nprof
    test_wvsp = mn .+ sd*randn(ntst)
    test_fxs = 0.4e-3*rand(ntst)     # this should make the sum of all five flux=1mPa on average
    for i = 1:ntst
        th = 0 #2pi*rand()  # going back to 1d for a sec
        test_waves[i] = c_to_wave(test_wvsp[i], [cos(th);sin(th)], k, test_fxs[i], srcs[1])
    end
    
    sz = size(vars["Up"][1])
    xi = rand(1:sz[1])
    yi = rand(1:sz[2])
    ti = rand(1:sz[4])
    col.U .= vars["Up"][1][xi,yi,:,ti]
    col.V .= vars["Vp"][1][xi,yi,:,ti]
    recon_5waves .= spectrum_to_momdep(L81_flux!,test_waves,col,test_fxs)
    
    err = randn(2nz)
    sigma = 0.05*maximum(abs.(recon_5waves))   # error scales with signal
    recon_5w_err = recon_5waves .+ sigma*err
    
    x1 .= x0    # reinitialize to a good starting guess
    p1 = (L81_exp_grad!,waves,col,recon_5w_err,lambda,BIG,gi,J)
    newton_solve!(x1, nl_obj, nl_obj_grad!, nl_obj_hess!, df, H, dx, xnew, p1; 
        reltol=1e-5, abstol=1e-10, max_newton_steps=60, error_on_max_steps=false,
        verbose=false, armijo_max_steps_no_error=true)
    sols1[:,idx] .= x1
end

# now we store the mean of the solutions in x2
x1 .= mean(exp.(sols1),dims=2)

#### Compare test spectrum to reconstructed spectrum
# Construct an array that will look like test_spectrum when plotted
#begin
fluxsum = 1e-3
gauss_fn = c -> 2*fluxsum/sqrt(pi)/sd*exp( -((c-mn)/sd)^2 )
og_distrib = 1e3*gauss_fn.(wavespds)
plot(wavespds,og_distrib,xlabel="wavespeed (m/s)",ylabel="flux (mPa)",label="test spectrum")
plot!(wavespds,1e3*x1,label="reconstructed spectrum")
savefig("Figures/1d_5w_distrib.png")
#end

#####################################
# Gaussian wavepacket
#####################################
seed2 = 5923522080718240123

wdth = 10+3*randn(Xoshiro(seed2))
mn = 30*randn(Xoshiro(seed2+1))
fluxsum = 1e-3
gauss_fn = c -> fluxsum/sqrt(pi)/wdth*exp( -((c-mn)/wdth)^2 )
wp_fxs = gauss_fn.(wavespds)

nprof2 = 1       ### number of profiles to test statistics with
x3 = zeros(nw)    # starting guess
sols3 = zeros(nw,nprof2)    # where the solutions go for statistics-ing with
recon_wp = zeros(2nz)

for idx = 1:nprof2
    sz = size(vars["Up"][1])
    xi = rand(1:sz[1])
    yi = rand(1:sz[2])
    ti = rand(1:sz[4])
    col.U .= vars["Up"][1][xi,yi,:,ti]
    col.V .= vars["Vp"][1][xi,yi,:,ti]
    recon_wp .= spectrum_to_momdep(L81_flux!,waves,col,wp_fxs)
    
    err = randn(2nz)
    sigma = 0.05*maximum(abs.(recon_wp))   # error scales with signal
    recon_wp_err = recon_wp .+ sigma*err
    
    x3 .= x0    # reinitialize to a good starting guess
    p3 = (L81_exp_grad!,waves,col,recon_wp_err,lambda,BIG,gi,J)
    newton_solve!(x3, nl_obj, nl_obj_grad!, nl_obj_hess!, df, H, dx, xnew, p3; 
        reltol=1e-5, abstol=1e-10, max_newton_steps=60, error_on_max_steps=false, 
        verbose=false, armijo_max_steps_no_error=true)
    sols3[:,idx] .= x3
end

# now we store the mean of the solutions in x3
x3 .= mean(exp.(sols3),dims=2)

#=
recon3 = spectrum_to_momdep(L81_flux!,waves,col,exp.(x3))
plot(1e3*reshape(recon_wp,(nz,2)),z,xlabel="flux (mPa)", ylabel="z (km)", label="truth", layout=2)
plot!(1e3*reshape(recon_wp_err,(nz,2)),z,label="data",layout=2)
plot!(1e3*reshape(recon3,(nz,2)),z,label="recon",layout=2)
savefig("Figures/1d_wavepacket_test.png")
=#

# wavepacket plots
plot(wavespds,1e3*wp_fxs,xlabel="wavespeed (m/s)",ylabel="flux (mPa)",label="test spectrum")
plot!(wavespds,1e3*x3,label="reconstructed spectrum")
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