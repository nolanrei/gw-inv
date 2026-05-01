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
using Distributions

include("waccm-utils.jl")
include("wrf-utils.jl")
include("L81-2.jl")

######################################################################################
##
######################################################################################
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

z = vars["oz"]
nz = length(z)
dz = 500*ones(nz)       # each cell is 500m thick in z

ref_flux = 1e-3
k = (2pi)/5e4    # 50km wavelength

# want to have col and fluxvec accessible outside the loop for debugging
col = ColumnProfile(vars["Up"][1][1,1,:,1], vars["Vp"][1][1,1,:,1], vars["N"], vars["rho"], nz, dz)
fluxvec = zeros(2nz);