# for testing accuracy on a random sample
include("omp.jl")
include("spectrum_metric.jl")

function ensemble_accuracy(ds::DimStack, retrieve_fluxvec!::Function; 
        kw=2pi/5e4, src=nothing, reg=1e-8, max_nw=20, nsamp=1000, calc_dist=true,
        bic_opts::BICOptions=BICOptions(), retrieve_args=(), project=false,
        col_op=nothing, zis=nothing, xs=nothing, ys=nothing, ts=nothing)
    # retrieve_args is where special inputs to the retrieve_fluxvec! go
    sz = size(ds[:U])
    nz = sz[3]
    wav_dummy = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)

    spectra = zeros(nsamp, 3max_nw)
    nxs = zeros(Int64,nsamp)
    if !isnothing(xs)
        idxs = rand(1:length(xs),nsamp)
    end
    x = zeros(3max_nw)
    if isnothing(xs)
        xs = rand(1:sz[1],nsamp)
    else
        xs = xs[idxs]
    end
    if isnothing(ys)
        ys = rand(1:sz[2],nsamp)
    else
        ys = ys[idxs]
    end
    if isnothing(ts)
        ts = rand(1:sz[4],nsamp)
    else
        ts = ts[idxs]
    end
    errors = zeros(nsamp, 2nz)
    fluxes = zeros(nsamp,2nz)
    rms_flux = zeros(nz)    # for normalizing error. fluxvec[i]^2 + fluxvec[i+nz]^2
    known_spectra = []      # if we know the spectra they go in this list of lists
    spec_dist = zeros(nsamp)

    ws = FullHessianOMPWorkspace(nz, max_nw)

    N = zeros(nz)
    rho = zeros(nz)
    z = zeros(nz)
    dz = zeros(nz)   #it's entirely unused
    
    col_dummy = ColumnProfile(ds[:U][1,1,:,1], ds[:V][1,1,:,1], N, rho, nz, dz)
    os = OMPStruct(col_dummy, L81_AD_wrapper!, wav_dummy; max_nw=max_nw)
    res = get_tmp(os.res, 0.0)   # gets the Float64 residual

    # accumulator for earth-mover distance
    mean_spectrum_distance = 0.0
    n_dist_samples = 0

    # if src == nothing, find it dynamically
    if src == nothing
        src_dyn = true
    else
        src_dyn = false
    end

    # progress bar printing
    print_stride=5

    # Loop over samples
    for i = 1:nsamp
        xi,yi,ti = xs[i], ys[i], ts[i]
    
        # Initialize column profile
        sample_existing_dims!(os.col.N, ds, :N, xi, yi, ti)
        sample_existing_dims!(os.col.rho, ds, :rho, xi, yi, ti)
        sample_existing_dims!(z, ds, :z, xi, yi, ti)
        
        if src_dyn
            src = find_tropopause(os.col.N, z; z_min=5, threshold=0.015, persistence_km=2)
        end
        os.wav.src = src
        os.col.U .= ds[:U][xi,yi,:,ti]
        os.col.V .= ds[:V][xi,yi,:,ti]

        # optional operation on col (if you want to flip U and V or remove rho gradient or smth)
        if !isnothing(col_op)
            col_op(os.col)
        end
        
        nx = 0
        fill!(x, 0.0)   # zero it out between runs
        
        # Initialize fluxvec
        known_spec_i = retrieve_fluxvec!(os.fluxvec, ds, xi, yi, ti, os, retrieve_args...)
        if !isnothing(known_spec_i) && project
            nxk = project_to_min!(known_spec_i, known_spec_i, kw, os.col)
            resize!(known_spec_i,nxk)
        end
        push!(known_spectra, known_spec_i)
        fluxes[i,:] .= os.fluxvec
        # update rms_flux
        for j = 1:nz
            rms_flux[j] += os.fluxvec[j]^2 + os.fluxvec[j+nz]^2
        end
        
        # Solve
        sharpen = sharpen_thresh!
        sharpen_params = (1e-5,)
        nx = find_measure!(x,reg,os,sharpen,ws,bic_opts; sharpen_params, zis)
        #fgh_closure = make_full_hessian_fgh(os, ws, reg, os.sc)

        # Store spectrum
        if project
            v = view(x,1:nx)
            nx = project_to_min!(v, v, kw, os.col)
        end
        spectra[i,1:nx] .= view(x,1:nx)
        nxs[i] = nx

        # Accumulate spectrum distance whenever we have a known spectrum to compare against
        if calc_dist && known_spec_i !== nothing
            dist = earth_mover_dist(known_spec_i, view(x, 1:nx))
            total_mass = sum(known_spec_i[3:3:end])
            spec_dist[i] = dist/total_mass
            mean_spectrum_distance += dist
            n_dist_samples += 1
        end

        # Compute error
        get_res!(x,nx,os)
        errors[i,:] .= res

        if i%print_stride == 0
            print("\rProcessing item $i / $nsamp...")
            flush(stdout)
        end
    end

    # Some final statistics
    for i = 1:nz
        rms_flux[i] = sqrt(rms_flux[i]/nsamp)
    end
    abs_err = dropdims(mean(abs2,errors,dims=1),dims=1)
    rms_abs_err_z = sqrt.(abs_err[1:nz] .+ abs_err[nz+1:2nz])  # rms err vs z
    rms_rel_err_z = rms_abs_err_z ./ rms_flux

    # computes mean_spectrum_distance assuming even if some but not all samples
    # do not have a known launching spectrum
    mean_spectrum_distance = [n_dist_samples > 0 ? mean_spectrum_distance / n_dist_samples : NaN]

    known_spectra, ks_offsets = flatten_ragged(known_spectra)

    return (
        rms_abs_err_z = rms_abs_err_z,
        rms_rel_err_z = rms_rel_err_z,
        spectra = spectra,
        known_spectra = known_spectra,
        ks_offsets = ks_offsets,
        errors = errors,
        fluxes = fluxes,
        nxs = nxs,
        xs = xs,
        ys = ys,
        ts = ts,
        mean_spectrum_distance = mean_spectrum_distance,
        spec_dist = spec_dist
    )
end

function save_ensemble_to_netcdf(filename::String, results::NamedTuple)
    if isfile(filename)
        rm(filename)
    end

    # Infer NetCDF dimensions from each array independently
    dims = Dict{Tuple, NcDim}()
    vars = NcVar[]

    for (name, data) in pairs(results)
        data isa AbstractArray || error(
            "Cannot save $name: expected an array, got $(typeof(data))"
        )

        dim_names = Tuple(Symbol(name, "_dim", i) for i in 1:ndims(data))

        dim_objs = NcDim[]
        for (dim_name, dim_size) in zip(dim_names, size(data))
            key = (dim_name, dim_size)

            if !haskey(dims, key)
                dims[key] = NcDim(String(dim_name), dim_size)
            end

            push!(dim_objs, dims[key])
        end

        push!(vars, NcVar(String(name), dim_objs, t=eltype(data)))
    end

    nc = NetCDF.create(filename, vars)

    for (name, data) in pairs(results)
        NetCDF.putvar(nc, String(name), data)
    end

    println("Successfully wrote outputs to $filename")
end

function load_ensemble_from_netcdf(filename::String)
    if !isfile(filename)
        error("File not found: $filename")
    end

    varnames = NetCDF.open(filename) do nc
        collect(keys(nc.vars))
    end

    pairs_vec = Vector{Pair{Symbol,Any}}()
    for name in varnames
        data = ncread(filename, name)
        push!(pairs_vec, Symbol(name) => data)
    end

    return NamedTuple(pairs_vec)
end

function load_all_ensembles(ncases::Int; prefix="ensemble_results_synth")
    all_results = Vector{NamedTuple}(undef, ncases)
    for case = 1:ncases
        filename = "$(prefix)$(case).nc"
        all_results[case] = load_ensemble_from_netcdf(filename)
    end
    return all_results
end

function condense_spectra(spectra, nxs)
    # Takes in spectra and nxs and concatenates all spectra into
    # one long vector
    nnx = length(nxs)
    nx_tot = sum(nxs)
    out = zeros(nx_tot)
    idx = 0
    for i = 1:nnx
        nxi = nxs[i]
        out[idx+1:idx+nxi] .= spectra[i,1:nxi]
        idx += nxi
    end
    #=
    # normalization for plotting
    mass = 0.0
    for i = 3:3:idx
        mass += out[i]
    end
    for i = 3:3:idx
        out[i] /= mass
    end
    =#
    return out
end

function project_to_min!(xout::AbstractVector, xin::AbstractVector, k::Float64, col::ColumnProfile)
    # Breaking waves pass through unchanged (their c is real information).
    # Non-breaking waves are merged into a single equivalent wave: flux
    # vectors summed, then placed at the minimum phase speed that keeps
    # *that* merged wave from breaking anywhere in the column. This is the
    # unique flux-space-invariant representation of "some non-breaking mass
    # in this direction-ish", independent of how many waves the recovery
    # happened to split it into.
    nz  = length(col.U)
    ik2 = 2/k
    nw  = div(length(xin), 3)

    fx_sum = 0.0
    fy_sum = 0.0
    nout   = 0

    for wi = 1:nw
        th = xin[3wi-2]
        c  = xin[3wi-1]
        f0 = xin[3wi]
        ct, st = cos(th), sin(th)

        broke = false
        uprev = NaN
        for zi = 1:nz
            u = ct*col.U[zi] + st*col.V[zi]
            clim = u + (f0*ik2*col.N[zi]/col.rho[zi])^(1/3)
            if clim > c || sign(c-u)*sign(c-uprev) < 0
                broke = true
                break
            end
            uprev = u
        end

        if broke
            nout += 1
            xout[3nout-2] = th
            xout[3nout-1] = c
            xout[3nout]   = f0
        else
            fx_sum += f0*ct
            fy_sum += f0*st
        end
    end

    fmag = hypot(fx_sum, fy_sum)
    if fmag > 0
        th_eq  = atan(fy_sum, fx_sum)
        ct, st = cos(th_eq), sin(th_eq)
        cmax = 100     # MAX_C
        nout += 1
        xout[3nout-2] = th_eq
        xout[3nout-1] = cmax
        xout[3nout]   = fmag
    end

    return 3*nout
end

function project_to_min!(xout, xin::Nothing, k::Float64, col::ColumnProfile)
    # no-op
end

function project_to_min_keepn!(xout::AbstractVector, xin::AbstractVector, k::Float64, col::ColumnProfile)
    # map phase speeds on waves that don't break to the minimum speed 
    # that would produce the same flux profile
    nz = length(col.U)
    ik2 = 2/k

    nx = length(xin)
    nw = div(nx,3)
    cmax = 0.0
    for wi = 1:nw
        cmax = 0.0
        # check if this wave breaks
        broke = false
        th = xin[3wi-2]
        c  = xin[3wi-1]
        f0 = xin[3wi]
        ct = cos(th)
        st = sin(th)
        for zi = 1:nz
            u = ct*col.U[zi] + st*col.V[zi]
            clim = u + (f0*ik2*col.N[zi]/col.rho[zi])^(1/3)
            if clim > c
                broke = true
                break
            end
            if clim > cmax
                cmax = clim
            end
        end
        if broke
            xout[3wi-2] = xin[3wi-2]
            xout[3wi-1] = cmax
            xout[3wi]   = xin[3wi]
        else
            xout[3wi-2] = xin[3wi-2]
            xout[3wi-1] = cmax
            xout[3wi]   = xin[3wi]
        end
    end
    return nothing
end

function visualize_given_sample(idx::Int, results, z::AbstractVector; 
        outdir::String=".", prefix::String="sample")
    truth = 1e3*reshape(view(results.fluxes, idx, :), (nz, 2))
    recon = 1e3*reshape(view(results.fluxes, idx, :) .- view(results.errors, idx, :), (nz, 2))

    p1 = plot(truth, z, xlabel="fluxes (mPa)", ylabel="z (km)", layout=2,
              label=["truth X" "truth Y"])
    plot!(p1, recon, z, label=["recon X" "recon Y"])
    savefig(p1, joinpath(outdir, "$(prefix)_flux_$idx.png"))
end

function visualize_random_sample(results, wrf_ds, z::AbstractVector, src::Int;
        cmax::Float64=100.0, thresh::Float64=12.0, cutoff_ang::Float64=pi/6,
        min_flux2::Float64=1e-9, max_attempts::Int=1000,
        outdir::String=".", prefix::String="sample")

    nsamp = size(results.fluxes, 1)
    nz = div(size(results.fluxes, 2), 2)

    if length(z) != nz
        error("length(z)=$(length(z)) doesn't match nz=$nz inferred from results.fluxes")
    end

    # Pick a random sample with nonzero flux, bounded so we can't spin forever
    # on a results set where every sample happens to be near-zero
    i = 0
    attempt = 0
    while attempt < max_attempts
        cand = rand(1:nsamp)
        attempt += 1
        if sum(abs2, view(results.fluxes, cand, :)) >= min_flux2
            i = cand
            break
        end
    end
    if i == 0
        error("No sample with flux^2 >= $min_flux2 found in $max_attempts attempts")
    end

    xi, yi, ti = results.xs[i], results.ys[i], results.ts[i]

    # Pull just this sample's known spectrum straight out of the flattened
    # storage via its offsets -- no need to reconstruct the whole ragged array
    known_i = view(results.known_spectra, results.ks_offsets[i]+1 : results.ks_offsets[i+1])

    # --- Flux profile: truth vs. reconstruction ---
    truth = 1e3*reshape(view(results.fluxes, i, :), (nz, 2))
    recon = 1e3*reshape(view(results.fluxes, i, :) .- view(results.errors, i, :), (nz, 2))

    p1 = plot(truth, z, xlabel="fluxes (mPa)", ylabel="z (km)", layout=2,
              label=["truth X" "truth Y"])
    plot!(p1, recon, z, label=["recon X" "recon Y"])
    savefig(p1, joinpath(outdir, "$(prefix)_flux_$i.png"))

    # --- Rosette: recovered spectrum vs. known spectrum, with mean-flow arrow ---
    U = wrf_ds[:U_raw][xi, yi, :, ti]
    V = wrf_ds[:V_raw][xi, yi, :, ti]

    nx_i = results.nxs[i]
    recovered_i = view(results.spectra, i, 1:nx_i)
    rosmap = rosette_heatmap(recovered_i, nx_i; cmax=cmax)

    r = hypot(U[src], V[src])
    theta = atan(V[src], U[src])
    plot!(rosmap, [0, theta], [0, r], arrow=true)

    if !isnan(known_i[1])
        rosette!(rosmap, known_i, length(known_i))
    end
    savefig(rosmap, joinpath(outdir, "$(prefix)_rosette_$i.png"))

    # --- Source-level diagnostic for the recovered spectrum ---
    srcs, prox = src_check(view(results.spectra, i, :), nx_i,
                            view(results.fluxes, i, :), U, V, wrf_ds[:iz];
                            thresh, cutoff_ang)

    # --- Spectrum-distance diagnostic for this sample ---
    # earth_mover_dist expects flat Float64 vectors; views work fine here,
    # but if it's not type-stable on views for you, collect(...) each first
    spectrum_dist = if nx_i > 0 && length(known_i) > 0
        earth_mover_dist(collect(recovered_i), collect(known_i))
    else
        NaN
    end

    return (index=i, xi=xi, yi=yi, ti=ti, srcs=srcs, prox=prox,
            spectrum_dist=spectrum_dist,
            flux_plot=p1, rosette_plot=rosmap, recovered=recovered_i, known=known_i)
end

# Some possible retrieve_fluxvecs

function create_sample_fluxvec!(fluxvec::Vector{Float64}, ds::DimStack, 
        xi::Int, yi::Int, ti::Int, 
        os::OMPStruct, nxw::Int64;
        alpha=0.9, mn=20.0, beta=0.2, vm=2.0, cscale=4e-4, f0=1e-4
    )
    # we're making random spectra nxw/3 long and synthesizing fluxvec
    waveset = zeros(nxw)   # allocating :(     but it doesn't make sense not to
    src = os.wav.src
    uvec = SA[os.col.U[src], os.col.V[src]]
    sample_gws!(waveset, nxw, uvec; alpha, mn, beta, vm, cscale, f0)
    fwd!(fluxvec, waveset, nxw, os)
    return waveset
end

function retrieve_fluxvec_avg!(fluxvec::Vector{Float64}, ds::DimStack, 
        xi::Int, yi::Int, ti::Int, 
        os::OMPStruct, dt::Int=4)
    nz = div(length(fluxvec),2)
    tlo = max(1, ti-dt)
    thi = min(size(ds[:UW],4), ti+dt)
    trange = tlo:thi
    for zi = 1:nz
        fluxvec[zi]    = mean(ds[:UW][xi,yi,zi,trange])  #UW
        fluxvec[zi+nz] = mean(ds[:VW][xi,yi,zi,trange])  #VW
    end
    return nothing
end

function retrieve_fluxvec_avg_rhomul!(fluxvec::Vector{Float64}, ds::DimStack, 
        xi::Int, yi::Int, ti::Int, 
        os::OMPStruct, dt::Int=4)
    nz = div(length(fluxvec),2)
    sample_existing_dims!(view(fluxvec,1:nz), ds, :rho, xi, yi, ti)
    for zi = 1:nz
        fluxvec[zi+nz] = fluxvec[zi]*mean(ds[:VW][xi,yi,zi,ti-dt:ti+dt],dims=2)  #VW
        fluxvec[zi]    *= mean(ds[:UW][xi,yi,zi,ti-dt:ti+dt],dims=2)             #UW
    end
    return nothing
end

function retrieve_fluxvec_sample_rhomul!(fluxvec::Vector{Float64}, ds::DimStack, 
        xi::Int, yi::Int, ti::Int,
        os::OMPStruct)
    nz = div(length(fluxvec),2)
    sample_existing_dims!(view(fluxvec,1:nz), ds, :rho, xi, yi, ti)
    for zi = 1:nz
        fluxvec[zi+nz] = fluxvec[zi]*ds[:VW][xi,yi,zi,ti]  #VW
        fluxvec[zi]    *= ds[:UW][xi,yi,zi,ti]             #UW
    end
    return nothing
end

function retrieve_fluxvec_sample!(fluxvec::Vector{Float64}, ds::DimStack, 
        xi::Int, yi::Int, ti::Int,
        os::OMPStruct)
    nz = div(length(fluxvec),2)
    for zi = 1:nz
        fluxvec[zi]    = ds[:UW][xi,yi,zi,ti]  #UW
        fluxvec[zi+nz] = ds[:VW][xi,yi,zi,ti]  #VW
    end
    return nothing
end