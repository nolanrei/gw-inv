# Full-dataset spectrum reconstruction: loops over an explicit list of columns
# (no resampling), writes results to NetCDF in blocks, and can resume after a
# crash / wall-time kill.
#
# Requires omp.jl, wrf-utils.jl (sample_existing_dims!, find_tropopause) and
# ensemble_accuracy.jl (project_to_min!) to be loaded.
#
# nxs status codes stored in the file:
#    >= 0  number of parameters (3 * n_waves) recovered for that column
#     -1   not yet processed (used for resuming)
#     -2   solver threw an error (logged with @warn)
#     -3   no tropopause found (only when src is found dynamically)

using NetCDF
using Dates

include("omp.jl")
include("ensemble_accuracy.jl")

"""
    column_indices(ds; xr=nothing, yr=nothing, tr=nothing) -> (xs, ys, ts)

Every (x, y, t) column in `ds` (or in the given sub-ranges), x varying fastest.
Each column appears exactly once.
"""
function column_indices(ds::DimStack; xr=nothing, yr=nothing, tr=nothing)
    sz = size(ds[:U])
    xr = isnothing(xr) ? (1:sz[1]) : xr
    yr = isnothing(yr) ? (1:sz[2]) : yr
    tr = isnothing(tr) ? (1:sz[4]) : tr
    n = length(xr) * length(yr) * length(tr)
    # Int (not Int32): retrieve_fluxvec! functions dispatch on xi::Int.
    # Conversion to Int32 happens only when writing to NetCDF.
    xs = Vector{Int}(undef, n); ys = similar(xs); ts = similar(xs)
    i = 0
    for t in tr, y in yr, x in xr
        i += 1
        xs[i] = x; ys[i] = y; ts[i] = t
    end
    return xs, ys, ts
end

"""
    setup_column!(os, z, ds, xi, yi, ti, src, col_op) -> src_used

Loads N, rho, z, U, V for column (xi, yi, ti) into os, sets the launch level
(fixed `src`, or dynamic tropopause if `src === nothing`), and applies col_op.
Returns the launch level used (-1 if no tropopause was found).
"""
function setup_column!(os::OMPStruct, z::Vector{Float64}, ds::DimStack,
        xi::Integer, yi::Integer, ti::Integer, src, col_op)
    sample_existing_dims!(os.col.N, ds, :N, xi, yi, ti)
    sample_existing_dims!(os.col.rho, ds, :rho, xi, yi, ti)
    sample_existing_dims!(z, ds, :z, xi, yi, ti)
    s = isnothing(src) ?
        find_tropopause(os.col.N, z; z_min=5, threshold=0.015, persistence_km=2) : src
    os.wav.src = s
    os.col.U .= ds[:U][xi, yi, :, ti]
    os.col.V .= ds[:V][xi, yi, :, ti]
    isnothing(col_op) || col_op(os.col)
    return s
end

function init_recon_file(outfile, xs, ys, ts, npar, nz; save_fit, T, gatts)
    ncol = length(xs)
    nccreate(outfile, "xs",  "col", ncol; t=Int32, gatts=gatts)
    nccreate(outfile, "ys",  "col", ncol; t=Int32)
    nccreate(outfile, "ts",  "col", ncol; t=Int32)
    nccreate(outfile, "nxs", "col", ncol; t=Int32)
    nccreate(outfile, "src", "col", ncol; t=Int16)
    # spectra are mostly zero padding -> compress well
    nccreate(outfile, "spectra", "col", ncol, "param", npar; t=T, compress=4)
    if save_fit
        nccreate(outfile, "fit", "col", ncol, "zz", 2nz; t=T, compress=4)
    end
    ncwrite(Int32.(xs), outfile, "xs")
    ncwrite(Int32.(ys), outfile, "ys")
    ncwrite(Int32.(ts), outfile, "ts")
    ncwrite(fill(Int32(-1), ncol), outfile, "nxs")   # -1 = not yet done
    ncclose(outfile)
end

"""
    reconstruct_dataset(ds, retrieve_fluxvec!, xs, ys, ts, outfile; kwargs...)

Reconstructs the spectrum of every column (xs[i], ys[i], ts[i]), in order,
writing to `outfile` every `block` columns. Stores only xs, ys, ts, nxs, src,
spectra, and (optionally) the fitted flux profile.

If `resume=true` and `outfile` exists with the same column list, picks up at
the first unprocessed column.
"""
function reconstruct_dataset(ds::DimStack, retrieve_fluxvec!::Function,
        xs::AbstractVector{<:Integer}, ys::AbstractVector{<:Integer},
        ts::AbstractVector{<:Integer}, outfile::String;
        kw=2pi/5e4, src=nothing, reg=1e-8, max_nw=20,
        bic_opts::BICOptions=BICOptions(), retrieve_args=(),
        sharpen=sharpen_thresh!, sharpen_params=(1e-5,),
        project=false, col_op=nothing, zis=nothing,
        save_fit=false, T::Type{<:AbstractFloat}=Float32,
        block::Int=500, resume::Bool=true, zero_tol::Float64=0.0)

    ncol = length(xs)
    @assert length(ys) == ncol && length(ts) == ncol "xs, ys, ts must have equal length"
    nz = size(ds[:U], 3)
    npar = 3max_nw

    # ---- file setup / resume ----
    start_col = 1
    if resume && isfile(outfile)
        @assert ncread(outfile, "xs") == xs && ncread(outfile, "ys") == ys &&
                ncread(outfile, "ts") == ts "existing $outfile has a different column list"
        nxs_done = ncread(outfile, "nxs")
        start_col = something(findfirst(==(-1), nxs_done), ncol + 1)
        ncclose(outfile)
        println("Resuming $outfile at column $start_col / $ncol")
    else
        isfile(outfile) && rm(outfile)
        gatts = Dict("reg" => reg, "kw" => kw, "max_nw" => max_nw,
                     "src" => (isnothing(src) ? -1 : src),
                     "project" => Int(project), "save_fit" => Int(save_fit),
                     "created" => string(now()),
                     "nxs_codes" => "-1 unprocessed, -2 solver error, -3 no tropopause")
        init_recon_file(outfile, xs, ys, ts, npar, nz; save_fit, T, gatts)
    end
    start_col > ncol && return outfile

    # ---- workspaces (allocated once) ----
    wav_dummy = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, isnothing(src) ? 1 : src)
    col = ColumnProfile(zeros(nz), zeros(nz), zeros(nz), zeros(nz), nz, zeros(nz))
    os  = OMPStruct(col, L81_AD_wrapper!, wav_dummy; max_nw=max_nw)
    ws  = FullHessianOMPWorkspace(nz, max_nw)
    res = get_tmp(os.res, 0.0)
    z   = zeros(nz)
    x   = zeros(npar)

    spec_buf = zeros(T, block, npar)
    fit_buf  = save_fit ? zeros(T, block, 2nz) : nothing
    nx_buf   = zeros(Int32, block)
    src_buf  = zeros(Int16, block)

    t_start = time()
    for c0 = start_col:block:ncol
        c1 = min(c0 + block - 1, ncol)
        nb = c1 - c0 + 1
        fill!(spec_buf, 0)
        save_fit && fill!(fit_buf, 0)

        for (b, i) in enumerate(c0:c1)
            xi, yi, ti = xs[i], ys[i], ts[i]
            nx_buf[b] = -2
            src_buf[b] = -1
            try
                s = setup_column!(os, z, ds, xi, yi, ti, src, col_op)
                src_buf[b] = s
                if s < 1
                    nx_buf[b] = -3
                    continue
                end
                retrieve_fluxvec!(os.fluxvec, ds, xi, yi, ti, os, retrieve_args...)

                fill!(x, 0.0)
                nx = 0
                if maximum(abs, os.fluxvec) > zero_tol   # skip all-zero columns
                    nx = find_measure!(x, reg, os, sharpen, ws, bic_opts;
                                       sharpen_params, zis)
                    if project
                        v = view(x, 1:nx)
                        nx = project_to_min!(v, v, kw, os.col)
                    end
                end

                @views spec_buf[b, 1:nx] .= x[1:nx]
                if save_fit
                    get_res!(x, nx, os)                     # res = fluxvec - model
                    @views fit_buf[b, :] .= os.fluxvec .- res
                end
                nx_buf[b] = nx
            catch err
                @warn "Column $i (x=$xi, y=$yi, t=$ti) failed" exception=(err, catch_backtrace())
            end
        end

        # Write data first, nxs last: nxs acts as the commit marker, so a kill
        # mid-write leaves nxs == -1 and the block is redone on resume.
        ncwrite(spec_buf[1:nb, :], outfile, "spectra"; start=[c0, 1], count=[nb, npar])
        save_fit && ncwrite(fit_buf[1:nb, :], outfile, "fit"; start=[c0, 1], count=[nb, 2nz])
        ncwrite(src_buf[1:nb], outfile, "src"; start=[c0], count=[nb])
        ncwrite(nx_buf[1:nb],  outfile, "nxs"; start=[c0], count=[nb])
        ncclose(outfile)   # flush to disk

        ndone = c1 - start_col + 1
        rate = ndone / (time() - t_start)
        eta_h = (ncol - c1) / rate / 3600
        print("\rColumn $c1 / $ncol  ($(round(rate, digits=2)) col/s, ETA $(round(eta_h, digits=1)) h)")
        flush(stdout)
    end
    println()
    return outfile
end

"""
    merge_shards(infiles, outfile)

Concatenates shard files (from a job array) into one file. Column order is
irrelevant since xs/ys/ts travel with each row. Warns if any shard is unfinished.
"""
function merge_shards(infiles::Vector{String}, outfile::String)
    parts = [load_ensemble_from_netcdf(f) for f in infiles]
    for (f, p) in zip(infiles, parts)
        any(==(-1), p.nxs) && @warn "$f has unprocessed columns (nxs == -1)"
    end
    cat_field(k) = reduce(vcat, (getfield(p, k) for p in parts))
    xs, ys, ts = cat_field(:xs), cat_field(:ys), cat_field(:ts)
    spectra = cat_field(:spectra)
    save_fit = haskey(parts[1], :fit)
    nz = save_fit ? div(size(parts[1].fit, 2), 2) : 0

    nc = NetCDF.open(infiles[1]); gatts = copy(nc.gatts); ncclose(infiles[1])
    gatts["merged_from"] = join(basename.(infiles), ",")

    isfile(outfile) && rm(outfile)
    init_recon_file(outfile, xs, ys, ts, size(spectra, 2), nz;
                    save_fit, T=eltype(spectra), gatts)
    ncwrite(spectra, outfile, "spectra")
    save_fit && ncwrite(cat_field(:fit), outfile, "fit")
    ncwrite(Int16.(cat_field(:src)), outfile, "src")
    ncwrite(Int32.(cat_field(:nxs)), outfile, "nxs")
    ncclose(outfile)
    return outfile
end

"""
    recompute_fit(ds, xs_i, ys_i, ts_i, spec_row, nx, src_i; kw, max_nw, col_op) -> Vector

Recomputes the fitted flux profile for one stored column from its spectrum,
for when the file was written with save_fit=false.
"""
function recompute_fit(ds::DimStack, xi::Integer, yi::Integer, ti::Integer,
        spec_row::AbstractVector, nx::Integer, src_i::Integer;
        kw=2pi/5e4, max_nw=20, col_op=nothing)
    nz = size(ds[:U], 3)
    wav = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, Int(src_i))
    col = ColumnProfile(zeros(nz), zeros(nz), zeros(nz), zeros(nz), nz, zeros(nz))
    os  = OMPStruct(col, L81_AD_wrapper!, wav; max_nw=max_nw)
    setup_column!(os, zeros(nz), ds, xi, yi, ti, Int(src_i), col_op)
    fit = zeros(2nz)
    nx > 0 && fwd!(fit, Float64.(spec_row[1:nx]), Int(nx), os)   # Float64: DiffCache expects it
    return fit
end

"""
    enstructure(spectra, xs, ys, ts)

Takes in the unstructured output of reconstruct_dataset and makes a grid of it again
Mostly intended for plotting
"""
function enstructure(spectra::Array{F,2}, xs::AbstractVector{I}, 
        ys::AbstractVector{I}, ts::AbstractVector{I}) where {F <: AbstractFloat, I <: Integer}
    xmin = minimum(xs)
    xmax = maximum(xs)
    ymin = minimum(ys)
    ymax = maximum(ys)
    tmin = minimum(ts)
    tmax = maximum(ts)
    
    nx = xmax-xmin+1
    ny = ymax-ymin+1
    nt = tmax-tmin+1
    max_nx = size(spectra,2)
    nsamp = size(spectra,1)

    ssq = fill(F(NaN), (nx, ny, max_nx, nt))

    for idx = 1:nsamp
        x = xs[idx]-xmin+1
        y = ys[idx]-ymin+1
        t = ts[idx]-tmin+1

        ssq[x,y,:,t] .= view(spectra, idx, :)
    end

    return ssq
end