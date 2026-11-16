### Run this before run_synth_ma.jl (only needs to be done once)
# this loads packages and data so the runner can be fast

using LinearAlgebra
using SparseArrays
using StaticArrays
using Random
using Plots
using NetCDF

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

######################################################################################
##
######################################################################################

dz = 500*ones(nz)       # each cell is 500m thick in z
cases = [5]#[1,2,3,4,5,6,7,8,9,10]
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
