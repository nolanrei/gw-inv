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

include("omp.jl")
include("ensemble_accuracy.jl")
include("reconstruct_dataset.jl")

case = 10
println("case = $case")
wrf_ds = load_wrf_case(case)
z = wrf_ds[:z]
nz = length(z)
dz = 500*ones(nz)

nx,ny,nz,nt = size(wrf_ds[:UW])

width=6    # defines averaging width for retrieve_fluxvec!
tr = (width+1):(2*width):(nt-width)

xs, ys, ts = column_indices(wrf_ds; xr=1:2:nx, yr=1:2:ny, tr=tr)  # xr,yr,tr are subset ranges

#N = 300
N = length(xs)

outfile = "~/work/gw-inv/Testing/c$(case)/full_recon.nc"

kw = 2pi/1e4
src = 13   # 16km
reg = 1e-8
max_nw = 20
bic_opts=BICOptions()


retrieve_fluxvec! = retrieve_fluxvec_avg!
retrieve_args=(dt=width)

reconstruct_dataset(wrf_ds, retrieve_fluxvec!, xs[1:N], ys[1:N], ts[1:N], outfile;
        kw, src, reg, max_nw,
        bic_opts, retrieve_args,
        sharpen=sharpen_thresh!, sharpen_params=(1e-5,),
        project=true, col_op=nothing, zis=nothing,
        save_fit=false, T=Float32,
        block=500, resume=false, zero_tol=0.0)