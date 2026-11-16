using NetCDF
using NCDatasets
using Random
using Dates
using DimensionalData

# (tag, dataset index, zonal var, meridional var)
const WACCM_GW_SOURCES = (
    ('O', 3, "UTGWORO",   "VTGWORO"),    # orography
    ('F', 3, "UTGWSPEC",  "VTGWSPEC"),   # frontogenesis (probably)
    ('C', 4, "BUTGWSPEC", "BVTGWSPEC"),  # Beres deep convection
)

"""
    drag_to_flux(drag, dz, rho)

Integrate drag from the model top downward. Flux at the top of cell k is the
drag deposited in all cells above it (1:k-1). Assumes top-down z ordering.
"""
function drag_to_flux(drag, dz, rho)
    cum = cumsum(drag .* dz .* rho, dims=3)
    flx = zeros(eltype(cum), size(cum))
    flx[:, :, 2:end, :] .= @view cum[:, :, 1:end-1, :]
    return flx
end

"""
    load_waccm_stack(start_idxs, end_idxs; wtype="OFC", separate=false) -> DimStack

Loads raw subsets of WACCM data into a standardized DimStack.
`wtype` selects which sources are summed into UW/VW/DX/DY.
If `separate=true`, each selected source is also stored individually as
UW_<tag>, VW_<tag>, DX_<tag>, DY_<tag> (tag ∈ O, F, C).
"""
function load_waccm_stack(start_idxs::Vector{Int}, end_idxs::Vector{Int};
                          wtype="OFC", separate=false)
    path_to_waccm = "/scratch/nr2489/waccm_1yr/"

    ds1 = NCDataset(joinpath(path_to_waccm, "Input1_year1.nc"), "r")
    ds2 = NCDataset(joinpath(path_to_waccm, "Input2_year1.nc"), "r")
    ds3 = NCDataset(joinpath(path_to_waccm, "Input3_year1.nc"), "r")
    ds4 = NCDataset(joinpath(path_to_waccm, "Input4_year1.nc"), "r")
    dss = (ds1, ds2, ds3, ds4)

    lon_vals  = ds1["lon"][start_idxs[1]:end_idxs[1]]
    lat_vals  = ds1["lat"][start_idxs[2]:end_idxs[2]]
    time_vals = ds1["time"][start_idxs[4]:end_idxs[4]]
    z_indices = start_idxs[3]:end_idxs[3]

    start_CI = CartesianIndex(start_idxs...)
    end_CI   = CartesianIndex(end_idxs...)
    e3 = CartesianIndex(0, 0, 1, 0)

    # Base variables
    U = ds1["U"][start_CI:end_CI]
    V = ds1["V"][start_CI:end_CI]
    Z3 = ds1["Z3"][start_CI:end_CI] / 1e3  # km

    Z_ext = ds1["Z3"][start_CI:(end_CI + e3)]
    dz = abs.(Z_ext[:, :, 1:end-1, :] .- Z_ext[:, :, 2:end, :])

    rhoi = ds2["RHOI"][start_CI:(end_CI + e3)]
    rho  = 0.5 .* (rhoi[:, :, 1:end-1, :] .+ rhoi[:, :, 2:end, :])

    # -----------------------------------------------------------
    # Drags: read only the requested sources, sum them, and
    # optionally keep each one separately
    # -----------------------------------------------------------
    active = filter(s -> occursin(s[1], wtype), WACCM_GW_SOURCES)
    isempty(active) && error("wtype=\"$wtype\" selects no sources (use some of O, F, C)")

    ugw_drag = zeros(eltype(U), size(U))
    vgw_drag = zeros(eltype(V), size(V))
    components = Tuple{Char, Array, Array}[]   # (tag, du, dv), top-down

    for (tag, di, uname, vname) in active
        du = dss[di][uname][start_CI:end_CI]
        dv = dss[di][vname][start_CI:end_CI]
        ugw_drag .+= du
        vgw_drag .+= dv
        separate && push!(components, (tag, du, dv))
    end

    # Drag -> flux (integrate top-down, before flipping)
    UW = drag_to_flux(ugw_drag, dz, rho)
    VW = drag_to_flux(vgw_drag, dz, rho)

    # Background N
    T = ds1["T"][start_CI:end_CI]
    g, R, p0, kappa = 9.81, 287.05, 1e5, 2/7

    p  = @. R * rho * T
    th = @. T * (p0 / p)^kappa

    dp    = p[:, :, 3:end, :] .- p[:, :, 1:end-2, :]
    dlnth = log.(th[:, :, 3:end, :]) .- log.(th[:, :, 1:end-2, :])
    N2_interior = @. -g^2 * rho[:, :, 2:end-1, :] * (dlnth / dp)

    N2 = zeros(eltype(N2_interior), size(T))
    N2[:, :, 2:end-1, :] .= N2_interior
    N2[:, :, 1, :]       .= N2[:, :, 2, :]
    N2[:, :, end, :]     .= N2[:, :, end-1, :]
    N = sqrt.(N2 .* (N2 .> 0))

    close.(dss)

    # -----------------------------------------------------------
    # Flip Z (dim 3) to match WRF (bottom-up)
    # -----------------------------------------------------------
    flipz(A) = reverse(A, dims=3)
    z_indices_flipped = reverse(z_indices)
    dims_4d = (X(lon_vals), Y(lat_vals), Z(z_indices_flipped), Ti(time_vals))

    da_U   = DimArray(flipz(U),        dims_4d; name=:U)
    da_V   = DimArray(flipz(V),        dims_4d; name=:V)
    da_UW  = DimArray(flipz(UW),       dims_4d; name=:UW)
    da_VW  = DimArray(flipz(VW),       dims_4d; name=:VW)
    da_DX  = DimArray(flipz(ugw_drag), dims_4d; name=:DX)
    da_DY  = DimArray(flipz(vgw_drag), dims_4d; name=:DY)
    da_rho = DimArray(flipz(rho),      dims_4d; name=:rho)
    da_N   = DimArray(flipz(N),        dims_4d; name=:N)
    da_p   = DimArray(flipz(p),        dims_4d; name=:p)
    da_th  = DimArray(flipz(th),       dims_4d; name=:th)
    da_z   = DimArray(flipz(Z3),       dims_4d; name=:z)

    # Per-source arrays (empty if separate=false)
    comp_das = DimArray[]
    for (tag, du, dv) in components
        push!(comp_das,
            DimArray(flipz(drag_to_flux(du, dz, rho)), dims_4d; name=Symbol("UW_", tag)),
            DimArray(flipz(drag_to_flux(dv, dz, rho)), dims_4d; name=Symbol("VW_", tag)),
            DimArray(flipz(du), dims_4d; name=Symbol("DX_", tag)),
            DimArray(flipz(dv), dims_4d; name=Symbol("DY_", tag)),
        )
    end

    da_lon = DimArray(lon_vals, (X(lon_vals),); name=:lon)
    da_lat = DimArray(lat_vals, (Y(lat_vals),); name=:lat)

    return DimStack(da_U, da_V, da_UW, da_VW, da_DX, da_DY, da_rho, da_N,
                    da_p, da_th, da_z, da_lon, da_lat, comp_das...;
                    metadata=Dict("source" => "WACCM", "wtype" => wtype,
                                  "separate" => separate))
end

function load_momdep(start_idxs::Vector{Int64}, end_idxs::Vector{Int64}, verbose=true)
    #=
    Loads and computes WACCM data required for looking at gravity wave propagation and sources
    Loads data between start_idxs and end_idxs, as in [start_idxs[1]:end_idxs[1], ..., start_idxs[end]:end_idxs[end]]
    Indices should be in order [x, y, z, t]
    Outputs:
    - ugw (total zonal GW momentum flux)
    - vgw (total meridional GW momentum flux)
    =#
    path_to_waccm = "/scratch/nr2489/waccm_1yr/"
    f1 = "Input1_year1.nc"
    f2 = "Input2_year1.nc"
    f3 = "Input3_year1.nc"
    f4 = "Input4_year1.nc"
    
    # NetCDF package wants these intervals as (start, number of lines to read in this dimension)
    count = 1 .+ end_idxs .- start_idxs
    
    # gravity wave drags
    ugw = ncread(path_to_waccm*f3, "UTGWORO", start_idxs, count) +
      ncread(path_to_waccm*f3, "UTGWSPEC", start_idxs, count) +
      ncread(path_to_waccm*f4, "BUTGWSPEC", start_idxs, count)
    vgw = ncread(path_to_waccm*f3, "VTGWORO", start_idxs, count) +
      ncread(path_to_waccm*f3, "VTGWSPEC", start_idxs, count) +
      ncread(path_to_waccm*f4, "BVTGWSPEC", start_idxs, count)

    momdep = cat(ugw,vgw,dims=3) # concatenate in dim=3 (z)

    return momdep
end

#=
function load_waccm_data(start_pos::Vector{Float64}, end_pos::Vector{Float64}, verbose=true)
    # Functionally identical to the above, but lets you specify start and end in term of 
    # Earth coordinates [lon, lat, p, t] instead of indices [xi, yi, zi, ti]
    path_to_waccm = "/scratch/nr2489/waccm_1yr/"
    f1 = "Input1_year1.nc"
    
    lon = ncread(path_to_waccm*f1, "lon")
    lat = ncread(path_to_waccm*f1, "lat")
    # find the indices using binary search

    g = 9.81      # gravitational acceleration, duh
    R = 287.05    # gas constant of dry air
    p0 = 1e5      # surface pressure in Pa
    rhoi = ncread(path_to_waccm*f2, "RHOI", start+[0,0,-1,0], count+[0,0,2,0])
    rho = 0.5*(rhoi[:,:,1:end-1,:]+rhoi[:,:,2:end,:])
    T = ncread(path_to_waccm*f1, "T", start+[0,0,-1,0], count+[0,0,1,0])
end
=#

function find_big_cols(ugw, threshold=1e-10)
    # data order [x, y, z, t]
    # if there is a large entry (ugw[idxs] > threshold), add all indices but z to a list
    big_cols = Vector{Int64}[]
    sz = size(ugw)

    for l=1:sz[4], k=1:sz[3], j=1:sz[2], i=1:sz[1]
        if abs(ugw[i,j,k,l]) > 1e-10
            push!(big_cols, [i, j, l])
        end
    end
    return big_cols
end

function save_subset(start_idxs::Vector{Int64}, end_idxs::Vector{Int64}, ncfilename::String)
    # Load and save a subset of WACCM data to a new nc4 file
    # Intended for cases where a small subset is needed but loading it
    # requires loading the entire dataset (e.g. small x,y, large t)
    U, V, rho, N, lon, lat, Z, ugw, vgw, p = load_waccm_data(start_idxs, end_idxs)
    ds = NCDataset(ncfilename,"c")       # create a new file
    defDim(ds,"lat",length(lat))
    defDim(ds,"lon",length(lon))
    defDim(ds,"v_lat",size(U,1))
    defDim(ds,"v_lon",size(U,2))
    defDim(ds,"z",size(U,3))
    defDim(ds,"r_z",size(rho,3))
    defDim(ds,"t",size(U,4))

    uu = defVar(ds,"U",Float64,("v_lon","v_lat","z","t"), attrib = Dict("units" => "m/s",))
    vv = defVar(ds,"V",Float64,("v_lon","v_lat","z","t"), attrib = Dict("units" => "m/s",))
    rrho = defVar(ds,"rho",Float64,("v_lon","v_lat","r_z","t"), attrib = Dict("units" => "kg/m^3",))
    nn = defVar(ds,"N",Float64,("v_lon","v_lat","z","t"), attrib = Dict("units" => "s^-1",))
    llon = defVar(ds,"lon",Float64,("lon",), attrib = Dict("units" => "degrees longitude",))
    llat = defVar(ds,"lat",Float64,("lat",), attrib = Dict("units" => "degrees latitude",))
    zz = defVar(ds,"Z",Float64,("v_lon","v_lat","z","t"), attrib = Dict("units" => "m",))
    uugw = defVar(ds,"ugw",Float64,("v_lon","v_lat","z","t"), attrib = Dict("units" => "m/s^2",))
    vvgw = defVar(ds,"vgw",Float64,("v_lon","v_lat","z","t"), attrib = Dict("units" => "m/s^2",))
    pp = defVar(ds,"Z",Float64,("v_lon","v_lat","z","t"), attrib = Dict("units" => "Pa",))

    uu .= U
    vv .= V
    rrho .= rho
    nn .= N
    llon .= lon
    llat .= lat
    zz .= Z
    uugw .= ugw
    vvgw .= vgw
    pp .= p
    close(ds)
    
    return U, V, rho, N, lon, lat, Z, ugw, vgw, p
end
