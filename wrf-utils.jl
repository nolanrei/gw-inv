using NetCDF
using Dates
using DimensionalData
using Statistics

function rosette(x,nx)
    nw = div(nx,3)
    r = zeros(nw)
    th = zeros(nw)
    z = zeros(nw)
    for i = 0:nw-1
        th[i+1]  = x[3i+1]
        r[i+1] = x[3i+2]
        z[i+1]  = x[3i+3]
    end
    p = scatter(th, r, proj=:polar, marker_z=z)
    return p
end

function rosette!(p, x, nx)
    # can't be polar because it conflicts with things it could be plotted over
    nw = div(nx, 3)
    x_cart = zeros(nw)
    y_cart = zeros(nw)
    z = zeros(nw)
    
    for i = 0:nw-1
        th = x[3i+1]
        r  = x[3i+2]
        z[i+1] = x[3i+3]
        
        x_cart[i+1] = r * cos(th)
        y_cart[i+1] = r * sin(th)
    end
    
    scatter!(p, x_cart, y_cart, marker_z=z, colorbar=false) 
end

function rosette_heatmap(x, nx; ep=0.3, n=200, cmax=nothing)
    nw = div(nx, 3)

    if isnothing(cmax)
        cmax = maximum(x[2:3:nx-1])
    end

    # 1. Extract wave dot coordinates (Cartesian) and weights
    dots = zeros(nw, 2)
    weights = zeros(nw)
    for i = 1:nw
        th = x[3i-2]
        c  = x[3i-1]
        dots[i, 1] = c * cos(th)
        dots[i, 2] = c * sin(th)
        weights[i] = x[3i] # Wave weight/amplitude
    end

    # 2. Build 2D Cartesian grid spanning [-cmax, cmax]
    grid_1d = range(-cmax, cmax, length=n)
    map_2d = zeros(n, n)
    ep2 = ep * ep

    # 3. Evaluate RBF on Cartesian grid
    for i in 1:n        # X index (columns)
        x_val = grid_1d[i]
        for j in 1:n    # Y index (rows)
            y_val = grid_1d[j]
            
            # Mask out points outside the disk radius
            if x_val^2 + y_val^2 > cmax^2
                map_2d[j, i] = NaN
                continue
            end

            # Accumulate RBF sum
            val = 0.0
            for k in 1:nw
                dx = x_val - dots[k, 1]
                dy = y_val - dots[k, 2]
                r2 = dx*dx + dy*dy
                val += weights[k] * exp(-ep2 * r2)
            end
            map_2d[j, i] = val
        end
    end

    # 4. Plot Cartesian heatmap bounded into a clean circle
    p = heatmap(grid_1d, grid_1d, map_2d, 
                aspect_ratio = :equal, 
                color = :viridis, 
                xlims = (-cmax*1.05, cmax*1.05), 
                ylims = (-cmax*1.05, cmax*1.05))

    return p
end

function gaussian_rbf(xx,x,y,ep)
    nxx = size(xx,1)
    ny  = length(y)   # x and y should be the same length
    ep2 = ep*ep

    out = zeros(nxx)
    
    for i = 1:nxx
        xxi = selectdim(xx,1,i)
        for j = 1:ny
            xj = selectdim(x,1,j)
            r2 = sum(abs2, xxi.-xj)
            out[i] += y[j]*exp(-ep2*r2)
        end
    end
    return out
end

function flatten_ragged(xs)
    # turn a ragged array into a flat array and an array of offsets
    offsets = [0; cumsum(length.(xs))]
    data = reduce(vcat, xs; init=eltype(first(xs))[])
    return data, offsets
end

function unflatten_ragged(data, offsets)
    [
        data[offsets[i]+1 : offsets[i+1]]
        for i in 1:length(offsets)-1
    ]
end

function src_check(x,nx,fluxvec,U,V,iz; thresh=12.0, cutoff_ang=pi/6)
    # Hypothesis: waves with direction dir are launched
    # from levels with mean windspeed in direction dir
    # For each wave in x, check if there is launch level below tropopause
    # (maybe this should be below breaking level instead but that's harder)
    nw = div(nx,3)
    nz = length(U)
    
    wavdir = zeros(2)
    srcs = zeros(nw)    # source heights in km. Must be less than thresh
    prox = zeros(nw)    # proximity -- cos(angular difference)
    
    dirs = cat(U,V,dims=2)
    dirs = dirs ./ sqrt.(sum(abs2,dirs,dims=2))   # unit direction vectors

    good_enough = cos(cutoff_ang)  # gotta be within e.g. 30 degrees of wave to accept launch level
    for i = 1:nw
        th = x[3i-2]
        wavdir[1] = cos(th); wavdir[2] = sin(th)

        # find the best match (allow + and - matches since we think that's possible)
        best = 0.0
        zbest = 0.0
        for j = 1:nz
            if iz[j] > thresh
                break
            end
            proj = abs(dot(wavdir, dirs[j,:]))   # + and -
            if proj > best
                best = proj
                zbest = iz[j]
            end
        end
        if best > good_enough
            srcs[i] = zbest
        else
            srcs[i] = NaN
        end
        prox[i] = best
    end
    return srcs, prox
end

function conv(arr, filt; dim=1)
    # in the case of a 2+ dimensional input, dim specifies which
    # (for now single) dimension we should filter over
    sz = size(arr)
    nd = length(sz)
    la = length(arr)

    narr = size(arr,dim)
    narr2 = div(narr,2)
    nfilt = length(filt)
    nfilt2 = div(nfilt,2)

    padarr_c = zeros(ComplexF64, narr+nfilt)
    filt_c = zeros(ComplexF64, narr+nfilt)
    filt_c[narr2.+(1:nfilt)] .= filt
    fft!(filt_c)
    tmp = zeros(ComplexF64,narr+nfilt)
    out = zeros(sz)

    # we can index an arbitrary dimension of an array by using 
    # linear indices and directly constructing the stride
    s = stride(arr,dim)
    for i = 1:s
        idxs = i:s:la

        padarr_c[1:nfilt2] .= arr[i]   # pad left with leftmost element
        padarr_c[nfilt2.+(1:narr)] .= arr[idxs]
        padarr_c[nfilt2+narr+1:end] .= arr[idxs[end]]   # pad right with rightmost element
        
        fft!(padarr_c)
        padarr_c .*= filt_c
        ifft!(padarr_c)
        ifftshift!(tmp,padarr_c)
        
        out[idxs] .= real.(tmp[nfilt2.+(1:narr)])
    end

    return out
end

function spike_plot_formatting(xs, ys)
    # For when you want to plot a set of delta functions
    # f = ∑ y_i δ(x-x_i)
    # Sort i in terms of ascending x_i, then make each spike
    # a thin triangle with height y_i (inaccurate, but spiritually true)
    tiny = 1e-3
    
    lookalike_v = zeros(3*ntst)
    lookalike_h = zeros(3*ntst)
    px = sortperm(xs)    # sorting permutation of test wavespeeds
    lookalike_v[2:3:end-1] .= 1e3*ys[px]
    lookalike_h[2:3:end-1] .= xs[px]
    lookalike_h[1:3:end-2] .= -tiny .+ xs[px]
    lookalike_h[3:3:end]   .=  tiny .+ xs[px]

    return lookalike_h, lookalike_v
end


function wasserstein_distance(pdf1, pdf2)
    # Computes Wasserstein 1-distance of pdf1 and pdf2
    # In a sense, how much mass needs to be moved from pdf1 to make pdf2
    # Corresponds to 1-metric of the discrete cdfs
    cdf1 = cumsum(pdf1)
    cdf2 = cumsum(pdf2)
    return norm(cdf1.-cdf2, 1)
end

function project_onto_XY(wavespds, waves, amps)
    #=
    Store wave.flux[x] under wave.vel[x], ditto for y
    Requires an output spectrum [X wavespds, Y wavespds] 
    =#
    wlen = length(wavespds)
    out = zeros(wlen,2)
    nw = length(waves)
    for i = 1:nw
        w = waves[i]
        dirx = w.k/hypot(w.k,w.l)
        fl = abs(dirx)*amps[i]  #*sign(w.c[1])
        # binary search to find the right place to store w.c[x]
        bi = bsrch(w.c[1],wavespds)
        # blending
        if bi < wlen   # if bi = wlen then the wave is outside wavespds
            v = (w.c[1]-wavespds[bi])/(wavespds[bi+1]-wavespds[bi])
            out[bi,1] += (1-v)*fl
            out[bi+1,1] += v*fl
        else
            # deposit all at the maximum
            out[bi,1] += fl
        end

        diry = w.l/hypot(w.k,w.l)
        fl = abs(diry)*amps[i] #*sign(w.c[2])
        # binary search to find the right place to store w.c[y]
        bi = bsrch(w.c[2],wavespds)
        # blending
        if bi < wlen
            v = (w.c[2]-wavespds[bi])/(wavespds[bi+1]-wavespds[bi])
            out[bi,2] += (1-v)*fl
            out[bi+1,2] += v*fl
        else
            out[bi,2] += fl
        end
    end
    return out
end

function project_posneg(wavespds, posspec, negspec)
    #=
    Store wave.flux[x] under wave.vel[x], ditto for y
    Requires an output spectrum [X wavespds, Y wavespds] 
    
    Specifically designed for positive/negative plots because 
    some directions of wave have negative projections onto X or Y
    and must be handled separately in order to preserve the 
    positivity/negativity of the spectra
    =#
    wlen = length(wavespds)
    posout = zeros(wlen*2)
    negout = zeros(wlen*2)
    slen = length(posspec.waves)
    for i = 1:slen
        w = posspec.waves[i]
        # binary search to find the right place to store w.c[x]
        bix = bsrch(w.c[1],wavespds)
        # binary search to find the right place to store w.c[y]
        biy = bsrch(w.c[2],wavespds)

        # x blending
        vx = (w.c[1]-wavespds[bix])/(wavespds[bix+1]-wavespds[bix])
        # y blending
        vy = (w.c[2]-wavespds[biy])/(wavespds[biy+1]-wavespds[biy])

        # x component
        flx = w.flux[1]
        if flx > 0
            # positive x comp + positive amp ==> positive x momdep
            posout[bix] += (1-vx)*flx*posspec.amplitudes[i]
            posout[bix+1] += vx*flx*posspec.amplitudes[i]
            negout[bix] += (1-vx)*flx*negspec.amplitudes[i]
            negout[bix+1] += vx*flx*negspec.amplitudes[i]
        else
            # if the direction of the wave is negative in x, a positive amplitude
            # will lead to a negative momentum transport in the x direction
            posout[bix] += (1-vx)*flx*negspec.amplitudes[i]
            posout[bix+1] += vx*flx*negspec.amplitudes[i]
            negout[bix] += (1-vx)*flx*posspec.amplitudes[i]
            negout[bix+1] += vx*flx*posspec.amplitudes[i]
        end
        # y component
        fly = w.flux[2]
        if fly > 0
            # positive y comp + positive amp ==> positive y momdep
            posout[biy+wlen] += (1-vy)*fly*posspec.amplitudes[i]
            posout[biy+1+wlen] += vy*fly*posspec.amplitudes[i]
            negout[biy+wlen] += (1-vy)*fly*negspec.amplitudes[i]
            negout[biy+1+wlen] += vy*fly*negspec.amplitudes[i]
        else
            # negative y comp + positive amp ==> negative y momdep
            posout[biy+wlen] += (1-vy)*fly*negspec.amplitudes[i]
            posout[biy+1+wlen] += vy*fly*negspec.amplitudes[i]
            negout[biy+wlen] += (1-vy)*fly*posspec.amplitudes[i]
            negout[biy+1+wlen] += vy*fly*posspec.amplitudes[i]
        end
    end
    return posout, negout
end

function project_posneg(wavespds, spectrum)
    #=
    Store wave.flux[x] under wave.vel[x], ditto for y
    Requires an output spectrum [X wavespds, Y wavespds] 
    
    Specifically designed for positive/negative plots because 
    some directions of wave have negative projections onto X or Y
    and must be handled separately in order to preserve the 
    positivity/negativity of the spectra
    =#
    wlen = length(wavespds)
    posout = zeros(wlen*2)
    negout = zeros(wlen*2)
    slen = length(spectrum.waves)
    for i = 1:slen
        w = spectrum.waves[i]
        # binary search to find the right place to store w.c[x]
        bix = bsrch(w.c[1],wavespds)
        # binary search to find the right place to store w.c[y]
        biy = bsrch(w.c[2],wavespds)

        # x blending
        vx = (w.c[1]-wavespds[bix])/(wavespds[bix+1]-wavespds[bix])
        # y blending
        vy = (w.c[2]-wavespds[biy])/(wavespds[biy+1]-wavespds[biy])

        # x component
        flx = w.flux[1]*spectrum.amplitudes[i]
        if flx > 0
            # positive x comp + positive amp ==> positive x momdep
            posout[bix] += (1-vx)*flx
            posout[bix+1] += vx*flx
        else
            # if the direction of the wave is negative in x, a positive amplitude
            # will lead to a negative momentum transport in the x direction
            negout[bix] += (1-vx)*flx
            negout[bix+1] += vx*flx
        end
        # y component
        fly = w.flux[2]*spectrum.amplitudes[i]
        if fly > 0
            # positive y comp + positive amp ==> positive y momdep
            posout[biy+wlen] += (1-vy)*fly
            posout[biy+1+wlen] += vy*fly
        else
            # negative y comp + positive amp ==> negative y momdep
            negout[biy+wlen] += (1-vy)*fly
            negout[biy+1+wlen] += vy*fly
        end
    end
    return posout, negout
end

function text_filter(pattern::Regex,str_arr::Array{String})
    approved = []
    for str in str_arr
        if occursin(pattern,str)
            push!(approved,str)
        else
            continue
        end
    end
    return approved
end

function bsrch(x::Real, a::AbstractArray{<:Real})
    # binary search
    # outputs i such that a[i] <= x <= a[i+1]
    ln = length(a)
    lo = 1
    hi = ln+1
    md = 0
    MAXITER = 50     # enough to cover arrays up to a quadrillion entries
    for iter = 1:MAXITER
        md = div(lo+hi,2)
        if a[md] >= x
            hi = md
        else
            lo = md
        end
        # if hi = lo+1, then we have x bracketed
        if hi == lo+1
            break
        end
    end
    return lo
end

"""
    parse_wrf_metadata(filepath::String)

Parses the date from the filename and identifies the case number.
"""
function parse_wrf_metadata(filepath::String)
    # Match date format YYYY-MM-DD
    date_match = match(r"(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})\.nc", filepath)
    base_date = DateTime(
        parse(Int, date_match[:year]), 
        parse(Int, date_match[:month]), 
        parse(Int, date_match[:day])
    )
    
    # Match case number from the folder path structure
    case_match = match(r"case(?<case>\d+)", filepath)
    case_num = parse(Int, case_match[:case])
    
    return base_date, case_num
end

"""
    get_wrf_files(case::Int)

Scans and sorts input/output files for a specific case number, respecting the directory hierarchy.
"""
function get_wrf_files(case::Int)
    input_dir  = "/projects/rps/epg2/gerberlab/WRF/INPUT/input_1deg_case$(case)"
    output_dir = "/projects/rps/epg2/gerberlab/WRF/OUTPUT/mf_case$(case)"
    
    # Gather and filter files matching the date regex
    is_nc_file(f) = occursin(r"\d{4}-\d{2}-\d{2}\.nc$", f)
    
    u_files  = isdir(input_dir)  ? filter(is_nc_file, readdir(input_dir, join=true)) : String[]
    uw_files = isdir(output_dir * "/uw/sgs_rey") ? filter(is_nc_file, readdir(output_dir * "/uw/sgs_rey", join=true)) : String[]
    vw_files = isdir(output_dir * "/vw/sgs_rey") ? filter(is_nc_file, readdir(output_dir * "/vw/sgs_rey", join=true)) : String[]
    
    # Chronologically sort files using their filename metadata
    sort_by_date(files) = sort(files, by = f -> parse_wrf_metadata(f)[1])
    
    return sort_by_date(u_files), sort_by_date(uw_files), sort_by_date(vw_files)
end

"""
    load_wrf_case(case::Int) -> DimStack

Loads, interpolates, and stitches WRF files for a single case into a compatible DimStack.
"""
function load_wrf_case(case::Int)
    u_files, uw_files, vw_files = get_wrf_files(case)
    isempty(u_files) && error("No files found for WRF case $case")

    # 1. Chronologically concatenate raw arrays
    u_raw  = cat([ncread(f, "u1deg") for f in u_files]..., dims=4)
    v_raw  = cat([ncread(f, "v1deg") for f in u_files]..., dims=4) # V resides in U files
    uw_raw = cat([ncread(f, "rey_g") for f in uw_files]..., dims=4)
    vw_raw = cat([ncread(f, "rey_g") for f in vw_files]..., dims=4)
    
    # 2. Build time axis
    time_coords = DateTime[]
    for f in u_files
        base_date, _ = parse_wrf_metadata(f)
        nt = size(ncread(f, "u1deg"), 4)
        append!(time_coords, [base_date + Minute(15) * i for i in 0:(nt-1)])
    end
    
    nx, ny, _, nt_total = size(u_raw)
    
    # Define vertical scales
    iz    = 1.0:1.0:65.0
    oz    = 10.0:0.5:65.0
    oz_dx = 10.25:0.5:64.75
    noz   = length(oz)
    
    # 3. Grid interpolation for U and V
    zb, ze = 10, 65
    Up = zeros(nx, ny, noz, nt_total)
    Vp = zeros(nx, ny, noz, nt_total)
    
    Up[:, :, 1:2:end, :] .= u_raw[:, :, zb:ze, :]
    Up[:, :, 2:2:end, :] .= 0.5 .* u_raw[:, :, zb:ze-1, :] .+ 0.5 .* u_raw[:, :, zb+1:ze, :]
    
    Vp[:, :, 1:2:end, :] .= v_raw[:, :, zb:ze, :]
    Vp[:, :, 2:2:end, :] .= 0.5 .* v_raw[:, :, zb:ze-1, :] .+ 0.5 .* v_raw[:, :, zb+1:ze, :]
    
    # 4. Load average background profiles and process DX/DY
    waccm_meta_path = "/scratch/nr2489/work/loose_spectra/waccm_rho_and_N.nc"
    N_meta = ncread(waccm_meta_path, "N")
    rho_meta = ncread(waccm_meta_path, "rho")
    
    rho_3d  = reshape(rho_meta, 1, 1, :, 1)
    rhoi_3d = 0.5 .* (rho_3d[:, :, 2:end, :] .+ rho_3d[:, :, 1:end-1, :])
    
    fv_x = uw_raw .* rho_3d
    DX   = (fv_x[:, :, 2:end, :] .- fv_x[:, :, 1:end-1, :]) ./ 500.0 ./ rhoi_3d
    
    fv_y = vw_raw .* rho_3d
    DY   = (fv_y[:, :, 2:end, :] .- fv_y[:, :, 1:end-1, :]) ./ 500.0 ./ rhoi_3d
    
    # 5. Pack everything into standardized DimensionalData structures
    # Retrieve lat/lon from the first u_file
    lon_raw = ncread(u_files[1], "lon")
    lat_raw = ncread(u_files[1], "lat")

    # Define a flexible spatial_dims function that accepts the Z dimension as an argument
    spatial_dims(Z_dim) = (X(1:nx), Y(1:ny), Z_dim, Ti(time_coords))

    # Drop the redundant time dimension if it exists
    lon_data = (ndims(lon_raw) == 3 ? lon_raw[:, :, 1] : lon_raw)
    lat_data = (ndims(lat_raw) == 3 ? lat_raw[:, :, 1] : lat_raw)

    # 4D Data (Raw, iZ axis, length 65)
    da_Ur = DimArray(u_raw, spatial_dims(Dim{:Z_raw}(iz)); name=:U_raw)
    da_Vr = DimArray(v_raw, spatial_dims(Dim{:Z_raw}(iz)); name=:V_raw)
    
    # 4D Data (Standard Z axis, length 111)
    da_Up = DimArray(Up,    spatial_dims(Z(oz));    name=:U)
    da_Vp = DimArray(Vp,    spatial_dims(Z(oz));    name=:V)
    da_UW = DimArray(fv_x,  spatial_dims(Z(oz));    name=:UW)
    da_VW = DimArray(fv_y,  spatial_dims(Z(oz));    name=:VW)
    
    # 4D Data (Staggered Z_dx axis, length 110)
    da_DX = DimArray(DX,    spatial_dims(Dim{:Z_dx}(oz_dx)); name=:DX)
    da_DY = DimArray(DY,    spatial_dims(Dim{:Z_dx}(oz_dx)); name=:DY)

    # 1D/2D Metadata
    lon_dims = ndims(lon_data) == 1 ? (X(1:nx),) : (X(1:nx), Y(1:ny))
    lat_dims = ndims(lat_data) == 1 ? (Y(1:ny),) : (X(1:nx), Y(1:ny))
    
    da_lon = DimArray(lon_data, lon_dims; name=:lon)
    da_lat = DimArray(lat_data, lat_dims; name=:lat)
    
    # Distinct dimensions for index-based and raw metadata profiles
    da_iz  = DimArray(iz,       (Dim{:Z_iz}(1:65),); name=:iz)
    da_oz  = DimArray(oz,       (Z(oz),);            name=:z)
    da_rho = DimArray(rho_meta, (Dim{:Z_meta}(1:length(rho_meta)),); name=:rho)
    da_N   = DimArray(N_meta,   (Dim{:Z_meta}(1:length(N_meta)),);   name=:N)
    
    return DimStack(da_Up, da_Vp, da_Ur, da_Vr, da_UW, da_VW, da_DX, da_DY, 
                    da_lon, da_lat, da_iz, da_oz, da_rho, da_N; 
                    metadata=Dict("case" => case, "source" => "WRF"))
end

# One last DimStack function -- WRF has N/rho as single columns, WACCM has 4D arrays
# This function lets you get your specific column if it exists, and otherwise the default one
@inline function existing_selectors(A, xi, yi, ti)
    selectors = ()
    if hasdim(A, X)  selectors = (selectors..., X(xi)) end
    if hasdim(A, Y)  selectors = (selectors..., Y(yi)) end
    if hasdim(A, Ti) selectors = (selectors..., Ti(ti)) end
    return selectors
end

function sample_existing_dims!(target_col::AbstractVector{Float64}, ds::DimStack, key, xi::Int, yi::Int, ti::Int)
    A = ds[key]
    selectors = existing_selectors(A, xi, yi, ti)
    
    if isempty(selectors)
        target_col .= parent(A)
    else
        target_col .= vec(parent(view(A, selectors...)))
    end
end

function mean_existing_dims(ds::DimStack, key, dims)
    A = ds[key]
    dims=()
    if hasdim(A, X)  dims = (dims..., X) end
    if hasdim(A, Y)  dims = (dims..., Y) end
    if hasdim(A, Ti) dims = (dims..., Ti) end
    if isempty(dims)
        target_col = parent(A)
    else
        target_col = vec(parent(mean(A;dims)))
    end
    return target_col
end

function find_tropopause(N::Vector{T}, z::Vector{T}; 
                                    z_min=5.0, threshold=0.016, persistence_km=2.0) where T<:AbstractFloat
    #=
        find_tropopause(N, z; z_min=5.0, threshold=0.016, persistence_km=2.0)
    
    Finds the first index where N exceeds a threshold and remains sustained 
    above a buffer value for a given vertical thickness.
    =#
    start_search = searchsortedfirst(z, z_min)
    baseline = 0.01
    do_not_dip_below = 0.7*(threshold - baseline)
    
    for i in start_search:length(z)
        if N[i] >= threshold
            end_z = z[i] + persistence_km
            end_idx = searchsortedlast(z, end_z)
            
            if end_idx > i
                # Check if all points in the next 'persistence_km' stay above a buffer (e.g., 0.014)
                # This protects against a minor oscillatory dip dropping back into tropospheric values
                if all(@view(N[i:end_idx]) .>= do_not_dip_below)
                    return i
                end
            end
        end
    end
    return -1 # Return -1 if no distinct tropopause is found
end

function cos_filter(nf::Number)
    # 10 is a nice resolution
    nf2 = div(nf,2)
    filter = zeros(1,1,1,nf)
    for i = -nf2:nf2-1
        filter[1,1,1,i+nf2+1] = cos(pi*i/nf)^2
    end
    return filter/sum(filter)     # needs to sum up to 1
end

function cos_filter(nf::AbstractArray)
    # nf should have 4 entries
    # 10 is a nice resolution
    nf2 = div.(nf,2)
    filter = zeros(nf...)
    for l = 1:nf[4]
        for k = 1:nf[3]
            for j = 1:nf[2]
                for i = 1:nf[1]
                    idx = [i j k l] .- nf2
                    filter[i,j,k,l] = prod(cos.(pi*idx./nf).^2)
                end
            end
        end
    end
    return filter/sum(filter)     # needs to sum up to 1
end

function wrf_tavg!(vars, varnames, filter)
    # Use FFT to rapidly apply a filter (to the t-dimension, probably)
    # varnames = e.g. ["DX", "DY", "Up", "Vp"]
    # Assumes all of these are knitted by wrf_knit
    szf = size(filter)
    szf2 = CartesianIndex(div.(szf,2)...)
    szf2p1 = CartesianIndex(1 .+div.(szf,2)...)
    szf2m1 = CartesianIndex((-1 .+szf.-div.(szf,2))...)   # this is not szf2-1 b/c of integer division
    nc = length(vars[varnames[1]])
    tmp = zeros(size(vars[varnames[1]]))
    for ci = 1:nc
        for var in varnames
            sz = CartesianIndex(size(vars[var][ci])...)   # allowing for the files to have different dimensions (esp. in t)
            vars[var][ci] .= conv(vars[var][ci],filter)[szf2p1:sz.+szf2]
        end
    end
    return nothing
end

function err_in_mean_wrf(vars, waves;
        obs_wndw=nothing, be=nothing,
        spectra_sv=true, recons_sv=false,
        reg="l2", wts=nothing
    )
    # difference compared to other versions of err_in_mean is that U,V,etc. come as 
    # lists of arrays inside the dictionary vars, so this loops over the outer list
    # of arrays before looping over the dimensions of the internal arrays
    nz = length(vars["oz"])
    nz_d = nz
    if obs_wndw === nothing
        # "observation window" for the column
        # a way of measuring error only over a subset of the column
        # e.g. in the case of launching waves below the troposphere (where gw drag is nonphysical)
        obs_wndw = [1:nz_d; nz.+(1:nz_d)] # entire column except for 
    end
    nobs = length(obs_wndw)
    if reg=="l1" && wts===nothing
        wts=ones(2nz)
    end

    nf = length(vars["Up"])

    # Defining parameters for inversion
    lam = 1e-4                          # lambda, regularization parameter. 1e-3 is usually right

    # find total number of cols
    # loop over files (they're not all necessarily the same size)
    ncol = 0
    for fidx = 1:nf
        sz = size(vars["Up"][fidx])
        nx,ny,_,nt = sz
        ######################### DEBUG
        if !isnothing(be)
            xb,xe,yb,ye = be[fidx]
        else
            xb,xe,yb,ye = 1,nx,1,ny
        end
        #########################
        ncol += (xe-xb+1)*(ye-yb+1)*nt
    end

    ntodate = 0
    err = zeros(ncol, 5)    # [error, xi, yi, ti, fidx]
    column_error = zeros(nobs)
    column_dev = zeros(nobs)
    mean_recon = zeros(2*nz_d)
    mean_truth = zeros(2*nz_d)

    if spectra_sv
        spectra = zeros(ncol, length(waves))
    else
        spectra = nothing
    end
    if recons_sv
        recons = zeros(ncol,2*nz_d)
    else
        recons = nothing
    end
    idx = 1            # count up row by row
    for fidx = 1:nf
        U = vars["Up"][fidx]
        V = vars["Vp"][fidx]
        ugw = vars["DX"][fidx]
        vgw = vars["DY"][fidx]
        N = vars["N"]
        rho = vars["rho"]
        sz = size(U)
        nx,ny,_,nt = sz
        ############## DEBUG
        if !isnothing(be)
            xb,xe,yb,ye = be[fidx]
        else
            xb,xe,yb,ye = 1,nx,1,ny
        end
        ##############
        for ti=1:nt, yi=yb:ye, xi=xb:xe
            #idx = 1 + (xi-1) + nx*((yi-1) + ny*(ti-1)) + ntodate   # ntodate should be the number of cols from previous files
            col = ColumnProfile(U[xi,yi,:,ti], V[xi,yi,:,ti], N, rho, nz_d)
    
            momdep = cat(dims=1,ugw[xi,yi,:,ti], vgw[xi,yi,:,ti]) # concatenate x- and y-fluxes
    
            if norm(momdep) == 0.0
                # if norm(momdep)==0, the wave didn't break and reconstructing it is going to be meaningless
                err[idx,:] .= [NaN, xi, yi, ti, fidx]
                println(err[idx,:])
                continue
            end
            spectrum = lin_invert(L81_wp!, waves, momdep, col; ref_flux=1e-5, lam, reg, wts, refine=1) # propag_grad=L81_grad,
            if spectra_sv
                spectra[idx,:] .= spectrum
            end
            recalc_momdep = spectrum_to_momdep(L81_wp!, waves, col, spectrum)
            if recons_sv
                recons[idx,:] .= recalc_momdep
            end
            
            # test correctness
            # propagate all the waves and add their momfluxes
            mean_truth .+= momdep
            mean_recon .+= recalc_momdep
            tr = momdep[obs_wndw]
            rc = recalc_momdep[obs_wndw]
            @views column_dev .+= ((tr .- rc)) ./ vars["means"][fidx]
            @views column_error .+= (tr .- rc).^2# ./ (vars["std"][fidx].^2)
            error = norm(tr .- rc)/norm(tr)
    
            err[idx,:] .= [error, xi, yi, ti, fidx]
            idx += 1
        end
        #ntodate += nx*ny*nt
    end
    
    mean_recon ./= ncol
    mean_truth ./= ncol
    column_dev .= column_dev/ncol
    column_error .= sqrt.((column_error)./ncol)
    return err, column_error, column_dev, mean_recon, mean_truth, spectra, recons
end
