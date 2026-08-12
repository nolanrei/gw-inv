# for testing accuracy on a random sample
include("omp.jl")

function ensemble_accuracy(ds::DimStack, retrieve_fluxvec!::Function; 
        kw=2pi/5e4, src=nothing, reg=1e-8, max_nw=20, nsamp=1000, retrieve_args=())
    # params is where special inputs to the retrieve_fluxvec! go
    sz = size(ds[:U])
    nz = sz[3]
    wav_dummy = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)

    spectra = zeros(nsamp, 3max_nw)
    nxs = zeros(Int64,nsamp)
    x = zeros(3max_nw)
    xs, ys, ts = rand(1:sz[1],nsamp), rand(1:sz[2],nsamp), rand(1:sz[4], nsamp) #[75],[17],[24]
    errors = zeros(nsamp, 2nz)
    fluxes = zeros(nsamp,2nz)
    rms_flux = zeros(nz)    # for normalizing error. fluxvec[i]^2 + fluxvec[i+nz]^2
    known_spectra = []      # if we know the spectra they go in this list of lists

    ws = FullHessianOMPWorkspace(nz, max_nw)

    N = zeros(nz)
    rho = zeros(nz)
    z = zeros(nz)
    dz = zeros(nz)   #it's entirely unused
    
    col_dummy = ColumnProfile(ds[:U][1,1,:,1], ds[:V][1,1,:,1], N, rho, nz, dz)
    os = OMPStruct(col_dummy, L81_AD_wrapper!, wav_dummy; max_nw=max_nw)
    res = get_tmp(os.res, 0.0)   # gets the Float64 residual

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
        
        nx = 0
        fill!(x, 0.0)   # zero it out between runs
        
        # Initialize fluxvec
        push!(known_spectra, retrieve_fluxvec!(os.fluxvec, ds, xi, yi, ti, os, retrieve_args...))
        fluxes[i,:] .= os.fluxvec              # no NaNs
        # update rms_flux
        for j = 1:nz
            rms_flux[j] += os.fluxvec[j]^2 + os.fluxvec[j+nz]^2
        end
        
        # Solve
        nx = find_measure!(x,reg,os,ws, fast_sharpen=true)
        #fgh_closure = make_full_hessian_fgh(os, ws, reg, os.sc)

        # Store spectrum
        spectra[i,1:nx] .= view(x,1:nx)       # no NaNs
        #println(x[1:nx])
        nxs[i] = nx

        # Compute error
        get_res!(x,nx,os)
        errors[i,:] .= res                    # NaNs -- where did they get here?

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

    NetCDF.close(nc)

    println("Successfully wrote outputs to $filename")
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