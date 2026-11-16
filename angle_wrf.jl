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
using HypothesisTests
using Distributions
using FFTW

include("waccm-utils.jl")
include("wrf-utils.jl")
include("L81-2.jl")
include("L81-autograd-compatible.jl")
include("mini-newton.jl")
include("inversion.jl")

include("find_angle.jl")

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

z = 10:0.5:65
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
cases = [rand(1:10)]#[1,2,3,4,5,6,7,8,9,10]
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

#########################################################
##
#########################################################

sz = size(vars["Up"][1])
xi = rand(1:sz[1])
yi = rand(1:sz[2])
ti = rand(1:sz[4])

uw = vars["UW"][1][xi,yi,:,ti].*vars["rho"]
vw = vars["VW"][1][xi,yi,:,ti].*vars["rho"]
th = zeros(nz)
r = zeros(nz)
thprev = 0.0
# want to make a plot without the branch cuts by allowing 
# angle to be outside (-pi,pi)
for i = 1:nz
    r[i] = hypot(uw[i], vw[i])
    th[i] = atan(vw[i], uw[i])
    delta = th[i]-thprev
    if abs(delta) > pi   
        # we assume a branch cut happened
        winding_num = round(delta/(2pi))
        th[i] -= 2*pi*winding_num
    end
    global thprev = th[i]
end

plot(th,z,xlabel="flux angle (radians)",ylabel="z (km)",label=nothing)
savefig("Figures/wrf_flux_theta.png")
plot(r,z,xlabel="flux magnitude (mPa)",ylabel="z (km)",label=nothing)
savefig("Figures/wrf_flux_r.png")
fxs = zeros(nz,2);
fxs[:,1] .= uw;
fxs[:,2] .= vw;
plot(fxs,z,xlabel="flux (mPa)",ylabel="z (km)",label=["x flux" "y flux"])
savefig("Figures/wrf_flux_xy.png")

vvec = zeros(nz,2);
vvec[:,1] .= vars["Up"][1][xi,yi,:,ti];
vvec[:,2] .= vars["Vp"][1][xi,yi,:,ti];
plot(vvec,z,xlabel="wind velocity (m/s)",ylabel="z (km)",label=["U" "V"])
savefig("Figures/wrf_UV.png")

drxy = zeros(nz-1,2);
drxy[:,1] .= 86400*vars["DX"][1][xi,yi,:,ti];  # it's already in mPa
drxy[:,2] .= 86400*vars["DY"][1][xi,yi,:,ti];
plot(drxy,vars["oz_dx"],xlabel="drag (m/s/day)",ylabel="z (km)",label=["Dx" "Dy"])
savefig("Figures/wrf_DXY.png")

th_and_mag_dr = zeros(nz-1,2);
angle_no_brcuts!(view(th_and_mag_dr,:,1), drxy[:,1], drxy[:,2])
th_and_mag_dr[:,2] .= hypot.(drxy[:,1], drxy[:,2]);
plot(th_and_mag_dr,vars["oz_dx"],xlabel=["drag angle (radians)" "drag magnitude (m/s/day)"],
    ylabel="z (km)",label=nothing,layout=2,size=(800,400))
savefig("Figures/wrf_drag_thmag.png")

th_cut = atan.(drxy[:,2], drxy[:,1])
histogram(th_cut, nbins = 20)
savefig("Figures/wrf_drag_th_hist.png")