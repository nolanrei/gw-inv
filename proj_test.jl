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


cmin = -100
cmax = 100
cstride = 2.5
ndirs = 2
srcs = [4]#, 32, 48]   # ~18km
wavespds = Array(cmin:cstride:cmax)
kw = 2pi/5e5
nws = length(wavespds)

z = 10.25:0.5:64.75
nz = length(z)

waves = make_waveset(wavespds, ndirs, srcs; ref_flux=1e-3)
nw = length(waves);
wlen = length(wavespds)

lambda = 1e-5
BIG = 1e6

#########################################################
##
#########################################################

dz = 500*ones(nz)       # each cell is 500m thick in z
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

# want to have col and fluxvec accessible outside the loop for debugging
col = ColumnProfile(vars["Up"][1][1,1,:,1], vars["Vp"][1][1,1,:,1], vars["N"], vars["rho"], nz, dz)
fluxvec = zeros(2nz)

#########################################################
##
#########################################################
seed = -5986149189942985559
ntst = 1
test_wvsp = -40 .+(40-(-40))*rand(Xoshiro(seed),ntst)
test_fxs = 0.8*ref_flux*rand(Xoshiro(seed+1),ntst)
test_waves = Vector{WaveProfile}(undef,ntst)
for i = 1:ntst
    th = 2pi*rand(Xoshiro(seed+1+i))
    test_waves[i] = c_to_wave(test_wvsp[i], [cos(th);sin(th)], k, test_fxs[i], srcs[1])
end

proj_test_waves = project_onto_XY(wavespds,test_waves,test_fxs)

sz = size(vars["Up"][1])
xi = rand(Xoshiro(seed+10000),1:sz[1])
yi = rand(Xoshiro(seed+10001),1:sz[2])
ti = rand(Xoshiro(seed+10002),1:sz[4])
col.U .= vars["Up"][1][xi,yi,:,ti]
col.V .= vars["Vp"][1][xi,yi,:,ti]
recon_5waves = spectrum_to_momdep(L81_flux!,test_waves,col,test_fxs)
recon_proj5w = spectrum_to_momdep(L81_flux!,waves,col,proj_test_waves[:])

plot(reshape(1e3*recon_5waves,(nz,2)),z,xlabel="flux (mPa)",ylabel="z (km)",label=["X orig" "Y orig"],layout=2)
plot!(reshape(1e3*recon_proj5w,(nz,2)),z,label=["X proj" "Y proj"],layout=2)
savefig("Figures/proj_testing.png")