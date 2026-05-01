using NetCDF
using NCDatasets
using Random
using Dates

function load_waccm_data(start_idxs::Vector{Int64}, end_idxs::Vector{Int64}, verbose=false; wtype="OFC")
    #=
    Loads and computes WACCM data required for looking at gravity wave propagation and sources
    Loads data between start_idxs and end_idxs, as in [start_idxs[1]:end_idxs[1], ..., start_idxs[end]:end_idxs[end]]
    Indices should be in order [x, y, z, t]
    Outputs:
    - U (zonal mean-flow velocity)
    - V (meridional mean-flow velocity)
    - rho (density, has to be interpolated from cell edges)
    - N (buoyancy frequency, has to be computed from derivative of potential temperature)
    - lon (longitude)
    - lat (latitude)
    - Z (geopotential)
    - ugw (total zonal GW momentum flux)
    - vgw (total meridional GW momentum flux)
    =#
    path_to_waccm = "/scratch/nr2489/waccm_1yr/"
    f1 = "Input1_year1.nc"
    f2 = "Input2_year1.nc"
    f3 = "Input3_year1.nc"
    f4 = "Input4_year1.nc"

    ds1 = NCDataset(path_to_waccm*f1,"r")
    ds2 = NCDataset(path_to_waccm*f2,"r")
    ds3 = NCDataset(path_to_waccm*f3,"r")
    ds4 = NCDataset(path_to_waccm*f4,"r")

    lon = ds1["lon"]
    lat = ds1["lat"]

    start_CI = CartesianIndex(start_idxs...)
    end_CI = CartesianIndex(end_idxs...)

    # Z is nearly redundant data in the stratosphere -- average in lat/lon to save space?
    Z = ds1["Z3"][start_CI:end_CI]
    if verbose
        println("longitude: $(lon[start_idxs[1]]) -- $(lon[end_idxs[1]]),\t latitude: $(lat[start_idxs[2]]) -- $(lat[end_idxs[2]])")
    end

    # gravity wave drags
    ugw_oro = ds3["UTGWORO"][start_CI:end_CI]
    ugw_frn = ds3["UTGWSPEC"][start_CI:end_CI]
    ugw_con = ds4["BUTGWSPEC"][start_CI:end_CI]
    vgw_oro = ds3["VTGWORO"][start_CI:end_CI]
    vgw_frn = ds3["VTGWSPEC"][start_CI:end_CI]
    vgw_con = ds4["BVTGWSPEC"][start_CI:end_CI]

    # allows specifying
    ugw = zeros(size(ugw_oro))
    vgw = zeros(size(vgw_oro))
    if !isnothing(match(r"O",wtype))
        ugw .+= ugw_oro
        vgw .+= vgw_oro
    end
    if !isnothing(match(r"F",wtype))
        ugw .+= ugw_frn
        vgw .+= vgw_frn
    end
    if !isnothing(match(r"C",wtype))
        ugw .+= ugw_con
        vgw .+= vgw_con
    end

    # Interpolating rho from cell faces to cell centers
    # CAVEAT: to do the derivatives we need a few more points in Z than we're going to use
    # so don't specify start_idxs and end_idxs such that z goes from 1 to end
    # 2 to end-1 is the maximum safe range
    e3 = CartesianIndex(0,0,1,0)
    rhoi = ds2["RHOI"][(start_CI-e3):(end_CI+2*e3)]
    rho = 0.5*(rhoi[:,:,1:end-1,:]+rhoi[:,:,2:end,:])
    T = ds1["T"][(start_CI-e3):(end_CI+e3)]

    # mean-flow velocities
    U = ds1["U"][start_CI:end_CI]
    V = ds1["V"][start_CI:end_CI]

#=    
    # NetCDF package wants these intervals as (start, number of lines to read in this dimension)
    count = 1 .+ end_idxs .- start_idxs
    
    lon = ncread(path_to_waccm*f1, "lon")
    lat = ncread(path_to_waccm*f1, "lat")
    # Z is nearly redundant data in the stratosphere -- average in lat/lon to save space?
    Z = ncread(path_to_waccm*f1,"Z3", start_idxs, count)
    if verbose
        println("longitude: $(lon[start_idxs[1]]) -- $(lon[end_idxs[1]]),\t latitude: $(lat[start_idxs[2]]) -- $(lat[end_idxs[2]])")
    end

    # gravity wave drags
    ugw = ncread(path_to_waccm*f3, "UTGWORO", start_idxs, count) +
      ncread(path_to_waccm*f3, "UTGWSPEC", start_idxs, count) +
      ncread(path_to_waccm*f4, "BUTGWSPEC", start_idxs, count)
    vgw = ncread(path_to_waccm*f3, "VTGWORO", start_idxs, count) +
      ncread(path_to_waccm*f3, "VTGWSPEC", start_idxs, count) +
      ncread(path_to_waccm*f4, "BVTGWSPEC", start_idxs, count)

    # Interpolating rho from cell faces to cell centers
    # CAVEAT: to do the derivatives we need a few more points in Z than we're going to use
    # so don't specify start_idxs and end_idxs such that z goes from 1 to end
    # 2 to end-1 is the maximum safe range
    rhoi = ncread(path_to_waccm*f2, "RHOI", start_idxs+[0,0,-1,0], count+[0,0,2,0])
    rho = 0.5*(rhoi[:,:,1:end-1,:]+rhoi[:,:,2:end,:])
    T = ncread(path_to_waccm*f1, "T", start_idxs+[0,0,-1,0], count+[0,0,1,0])

    # mean-flow velocities
    U = ncread(path_to_waccm*f1, "U", start_idxs, count)
    V = ncread(path_to_waccm*f1, "V", start_idxs, count)
    
=#
    # Computing potential temperature for N
    g = 9.81      # gravitational acceleration, duh
    R = 287.05    # gas constant of dry air
    p0 = 1e5      # surface pressure in Pa
    gamma = 1.4   # c_v/c_p for diatomic gases
    p = R*rho.*T
    th = @. T*(p0/p)^gamma
    # NOTE: the below does (-d/dz log theta)/(-d/dz p). This is not intentional, but the minuses cancel out right.
    N = @. -g*g*rho[:,:,2:end-1,:]*(log(th[:,:,3:end,:])-log(th[:,:,2:end-1,:]))/(p[:,:,3:end,:]-p[:,:,2:end-1,:])
    #N = @. g*g*(rho[:,:,3:end,:]-rho[:,:,2:end-1,:])/(p[:,:,3:end,:]-p[:,:,2:end-1,:])
    N = sqrt.(N.*(N.>0))
    
    return U, V, rho, N, lon, lat, Z, ugw, vgw, p
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
