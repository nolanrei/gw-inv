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
include("inversion.jl")

cmin = -10
cmax = 10
cstride = 5
ndirs = 1
srcs = [4]#, 32, 48]   # ~18km
wavespds = [10.0] #Array(cmin:cstride:cmax)
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
U1 = -10.0 .+ 50/(z[end]-z[1])*(-z[1].+z)
col1 = ColumnProfile(U1, V0, N, rho, nz, dz)

ref_flux = 1e-3
dir = [1.0;0.0]
th = pi/3
dir2 = [cos(th);sin(th)]
wavspd = 10.0
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

########## First plot:  momdep_tst1 as a function of flux
nfluxes = 5
reference_fluxes = [3e-4, 5.5e-4, 1e-3, 1.7e-3, 3e-3]
reference_wavefluxes = zeros(2nz,nfluxes)
for i = 1:nfluxes
    fx = reference_fluxes[i]
    L81_flux!(testwave,col1,fx,view(reference_wavefluxes,:,i))
end
slabel = ["3e-4" "5.5e-4" "1e-3" "1.7e-3" "3e-3"]
plot(1e3*reference_wavefluxes[1:nz,:],z,xlabel="flux (mPa)",ylabel="z (km)",label=slabel)
savefig("Figures/5-amplitudes.png")

plot(1e3*momdep_tst1[1:nz],z,xlabel="flux (mPa)",ylabel="z (km)",label="original")
plot!(momdep1[1:nz],z,label="with error")
savefig("Figures/1-amplitude.png")

######################################################################################
##
######################################################################################

