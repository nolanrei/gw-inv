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

cmin = -100
cmax = 100
cstride = 5
ndirs = 1
srcs = [4]#, 32, 48]   # ~18km
wavespds = Array(cmin:cstride:cmax)
kw = 2pi/5e5

N = ncread("/scratch/nr2489/work/loose_spectra/waccm_rho_and_N.nc","N")
rho = ncread("/scratch/nr2489/work/loose_spectra/waccm_rho_and_N.nc","rho")

waves = make_waveset(wavespds, ndirs, srcs; ref_flux=1e-3)
nw = length(waves);

z = 10.25:0.5:64.75

wts = (vcat(rho,rho))*1.0
sw = sum(wts)
wts[1:16] .= 0.0
wts[111:116] .= 0.0
wts ./= (sw/length(wts));

######################################################################################
##
######################################################################################

nz = length(z)
dz = 500*ones(nz)       # each cell is 500km wide
U0 = -10*ones(nz)
U1 = -10.0 .+ 50/(z[end]-z[1])*(-z[1].+z)
U2 = 39/(z[end]-z[1])^2 * (-z[1].+z).*(z[end].-z)
U3 = 30.0 .- 50/(z[end]-z[1])*(-z[1].+z)
V0 = zeros(nz)
col0 = ColumnProfile(U0, V0, N, rho, nz, dz)
col1 = ColumnProfile(U1, V0, N, rho, nz, dz)
col2 = ColumnProfile(U2, V0, N, rho, nz, dz)
col3 = ColumnProfile(U3, V0, N, rho, nz, dz)

ref_flux = 1e-3
dir = [1.0;0.0]
wavspd = 10.0
k = (2pi)/5e4    # 50km wavelength
usrc = col0.U[srcs[1]]*dir[1] + col0.V[srcs[1]]*dir[2]
testflux = 0.1*ref_flux
testsrc = srcs[1]
testwave = c_to_wave(wavspd, dir, k, testflux, testsrc)
prettify_wave!(testwave,usrc,N[testsrc])

objective = zeros(2nz)
L81_wp!(testwave,col1,objective)
sigma = 8e-3
objective .+= sigma*randn(2nz)

fluxes = [0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 30.0, 50.0, 100.0, 300.0, 1000.0]*ref_flux
errs = zeros(length(fluxes))
momdep = zeros(2nz)
plot()
for (i,f) = enumerate(fluxes)
    testwave.flux = f
    momdep .= 0.0
    L81_wp!(testwave,col1,momdep)
    momdep .*= 86400
    plot!(momdep[1:nz],z,label=nothing,xaxis="acceleration (m/s/day)",yaxis="z (km)")
    errs[i] = (momdep .- objective)'*(momdep .- objective)
end
savefig("Figures/L81-amplitude-test.png")

plot(fluxes,errs,xaxis="flux",yaxis="objective function",label=nothing,xscale=:log10,yscale=:log10)
savefig("Figures/objectivefn-convexity-test.png")