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
using DSP

include("waccm-utils.jl")
include("wrf-utils.jl")
include("L81.jl")

cases = [1,2,3,4,5,6,7,8,9,10]
files = wrf_get_fileset(cases)
vars = wrf_load(files)
wrf_postprocess!(vars)
vars_to_be_avgd = ["Up", "Vp", "DX", "DY", "UW", "VW"]
tstamps, idx2case, case_order = wrf_tstamps_cases(files["U"])
order = wrf_order(tstamps,idx2case,case_order)
for var in vars_to_be_avgd
    vars[var] = wrf_knit(vars[var],order)
end
drag_means!(vars)

cmin = -100
cmax = 100
cstride = 1
ndirs = 4
srcs = [16]   # ~18km
wavespds = Array(cmin:cstride:cmax)
waves = make_waveset(wavespds, ndirs, srcs)

ref_flux = 1e-3
istrt = srcs[1]
iend = 110
obs_wndw = [istrt:iend; istrt:iend]

wts = (vcat(vars["rho"],vars["rho"]))*1.0
sw = sum(wts)
wts[1:16] .= 0.0
wts[111:116] .= 0.0
wts ./= (sw/length(wts))
err_val,col_err,col_dev,recon,truth,spectra,recons = err_in_mean_wrf(vars,waves;spectra_sv=true,recons_sv=false,reg="l1",wts)

ncol,nw = size(spectra)
attribs = Dict("units" => "m/s^2")

fpath = "/scratch/nr2489/work/loose_spectra/wrf_18km_l1_spectra.nc"
isfile(fpath) && rm(fpath)  # checks if the file exists, and if it does, removes it
nccreate(fpath, "spectra", "cols",ncol, "waves",nw; atts=attribs)

ncwrite(spectra, fpath, "spectra")