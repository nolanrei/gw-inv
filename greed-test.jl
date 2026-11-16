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

momdep_tst1 = zeros(2nz)
L81_flux!(testwave,col1,ref_flux,momdep_tst1)

######################################################################################
##
######################################################################################

lambda = 1e-4

nw = length(waves)
nz = length(col1.U)
gi= zeros(2nz)
mu= 1e-10

nobj = 3
dx = zeros(nobj)
xnew = zeros(nobj)
df = zeros(nobj)   # gradient
H = zeros(nobj,nobj)
x0 = [-10.0, 0.0, 0.0]

p = (L81_flux!,col1,momdep_tst1,k,srcs[1],lambda,mu,gi)

include("greed.jl")
out = obj_3param(x0,p)
println(out)
obj_grad!(df,x0,p)
obj_hess!(H,x0,p)
