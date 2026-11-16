# Usage: julia run_recon.jl <task_id> <nshards>
#   task_id in 0 : (10*nshards - 1)  ->  case = task_id ÷ nshards + 1,  shard = task_id % nshards
# Each task reconstructs every column of its case whose index ≡ shard (mod nshards).

using LinearAlgebra
using SparseArrays
using StaticArrays
using Random
using NetCDF
using CSV
using Printf
using Distributions

include("waccm-utils.jl")
include("wrf-utils.jl")
include("L81-2.jl")

include("omp.jl")
include("ensemble_accuracy.jl")
include("reconstruct_dataset.jl")

task    = parse(Int, ARGS[1])
nshards = parse(Int, ARGS[2])
case    = task ÷ nshards + 1
shard   = task % nshards
println("task $task -> case $case, shard $shard of $nshards"); flush(stdout)

wrf_ds = load_wrf_case(case)
nx, ny, nz, nt = size(wrf_ds[:UW])

width = 6                                   # averaging half-width for retrieve_fluxvec_avg!
tr = (width+1):(2*width):(nt-width)         # non-overlapping time windows
xs, ys, ts = column_indices(wrf_ds; tr=tr)  # all x, all y

# Interleaved slice: spreads cheap/expensive columns evenly across shards
r = (shard+1):nshards:length(xs)

# Anchor output to this script's directory, not to wherever sbatch was run
outdir = joinpath(@__DIR__, "Testing", "c$(case)")
mkpath(outdir)
outfile = joinpath(outdir, "full_recon_s$(shard)of$(nshards).nc")
println("pwd = $(pwd())\nwriting to $outfile"); flush(stdout)

reconstruct_dataset(wrf_ds, retrieve_fluxvec_avg!, xs[r], ys[r], ts[r], outfile;
        kw=2pi/1e4, src=13, reg=1e-8, max_nw=20,
        bic_opts=BICOptions(), retrieve_args=(width,),
        sharpen=sharpen_thresh!, sharpen_params=(1e-5,),
        project=true, col_op=nothing, zis=nothing,
        save_fit=false, T=Float32,
        block=500, resume=true, zero_tol=0.0)

println("done: $outfile")