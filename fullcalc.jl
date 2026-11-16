using LinearAlgebra
using SparseArrays
using StaticArrays

include("waccm-utils.jl")

function fullcalc(waves::Vector{WaveProfile}; 
        lims::Vector{Ti}=[1,288,1,192,25,58,1,2919], 
        tstride::Integer=8, 
        fname::String) where {Ti <: Integer}
    # Runs a function X on all columns of WACCM data, divided into 
    # chunks that fit into memory (defined by tstride)
    xmin,xmax,ymin,ymax,zmin,zmax,tmin,tmax = lims
    nw = length(waves)
    nx = xmax-xmin+1
    ny = ymax-ymin+1
    nz = 2*(zmax-zmin)
    nt = tmax-tmin+1
    ncols = nx*ny*nt

    # stdev for normalization
    stdfull = ncread("mean_and_std.nc","stdfull")
    nz2 = (zmax-zmin+1)
    nh = div(size(stdfull,3),2)
    idxs = vcat(nz2:-1:1,nh.+(nz2:-1:1))
    stdev = view(stdfull, xmin:xmax, ymin:ymax, idxs)

    # parameters for lin_invert
    lam = 1e-3
    reg = "l2"
    refine = 1

    # create destination arrays
    spectrum = zeros(nw)
    spectra_xyt = zeros(nw,nx,ny,tstride)
    col_err = zeros(nx, ny, nz)
    truth = zeros(nx, ny, nz)
    recalc = zeros(nx, ny, nz)
    tmp = zeros(nz)

    # create a new NC4 file to store that data in
    ds = NCDataset(fname, "c")
    defDim(ds,"wav",nw)
    defDim(ds,"lon",nx)
    defDim(ds,"lat",ny)
    defDim(ds,"rho",nz)
    defDim(ds,"time",nt)
    sp = defVar(ds,"spec",Float64,("wav","lon","lat","time"), 
        attrib = Dict("units" => "m^2/s^2","standard_name" => "flux spectrum"))
    ce = defVar(ds,"col_err",Float64,("lon","lat","rho"), 
        attrib = Dict("units" => "m^2/s^2","standard_name" => "column error"))
    tr = defVar(ds,"true_tend",Float64,("lon","lat","rho"), 
        attrib = Dict("units" => "m/s^2","standard_name" => "true time-mean tendency"))
    rc = defVar(ds,"rec_tend",Float64,("lon","lat","rho"), 
        attrib = Dict("units" => "m/s^2","standard_name" => "reconstruction time-mean tendency"))

    nslice = Int(ceil(nt/tstride))
    for idx = 1:nslice
        tb = tmin + (idx-1)*tstride
        te = min(tmin + idx*tstride-1, tmax)
        U, V, rho, N, lon, lat, Z, ugw, vgw = load_waccm_data([xmin,ymin,zmin,tb],[xmax,ymax,zmax,te])
        for ti=1:(te-tb+1), yi=1:(ymax-ymin+1), xi=1:(xmax-xmin+1)
            col = ColumnProfile(U[xi,yi,end:-1:2,ti], V[xi,yi,end:-1:2,ti], N[xi,yi,end:-1:1,ti], rho[xi,yi,end-1:-1:2,ti], nz2-1)
            momdep = cat(dims=1,ugw[xi,yi,end:-1:2,ti], vgw[xi,yi,end:-1:2,ti]) # concatenate x- and y-fluxes

            if norm(momdep) == 0.0
                # if norm(momdep)==0, the wave didn't break and reconstructing it is going to be meaningless
                spectra_xyt[:,xi,yi,ti] .= 0.0
                continue
            end 

            spectrum .= lin_invert(L81_wp!, waves, momdep, col; lam, reg, refine)
            spectra_xyt[:,xi,yi,ti] .= spectrum
            recalc_momdep = spectrum_to_momdep(L81_wp!, waves, col, spectrum)
            
            tmp .= (momdep .- recalc_momdep)
            col_err[xi,yi,:] .+= (tmp.^2)
            truth[xi,yi,:] .+= momdep./ncols
            recalc[xi,yi,:] .+= recalc_momdep./ncols
        end
        #println(idx)
        sp[:,:,:,1:(te-tb+1)] .= spectra_xyt[:,:,:,1:(te-tb+1)]
    end
    ce .= sqrt.(col_err/ncols)./stdev
    tr .= truth
    rc .= recalc
    close(ds)
    return nothing
end

function fullcalc_nospec(waves::Vector{WaveProfile}; 
        lims::Vector{Ti}=[1,288,1,192,25,58,1,2919], 
        tstride::Integer=8, 
        fname::String) where {Ti <: Integer}
    # Runs a function X on all columns of WACCM data, divided into 
    # chunks that fit into memory (defined by tstride)
    xmin,xmax,ymin,ymax,zmin,zmax,tmin,tmax = lims
    nw = length(waves)
    nx = xmax-xmin+1
    ny = ymax-ymin+1
    nz = 2*(zmax-zmin)
    nt = tmax-tmin+1
    ncols = nx*ny*nt

    # stdev for normalization
    stdfull = ncread("mean_and_std.nc","stdfull")
    nz2 = (zmax-zmin+1)
    nh = div(size(stdfull,3),2)
    idxs = vcat(zmin:zmax-1,nh.+(zmin:zmax-1))
    stdev = view(stdfull, xmin:xmax, ymin:ymax, idxs)

    # parameters for lin_invert
    lam = 1e-3
    reg = "l2"
    refine = 1

    # create destination arrays
    spectrum = zeros(nw)
    col_err = zeros(nx, ny, nz)
    truth = zeros(nx, ny, nz)
    recalc = zeros(nx, ny, nz)
    tmp = zeros(nz)

    # create a new NC4 file to store that data in
    ds = NCDataset(fname, "c")
    defDim(ds,"wav",nw)
    defDim(ds,"lon",nx)
    defDim(ds,"lat",ny)
    defDim(ds,"rho",nz)
    defDim(ds,"time",nt)
    ce = defVar(ds,"col_err",Float64,("lon","lat","rho"), 
        attrib = Dict("units" => "m^2/s^2","standard_name" => "column error"))
    tr = defVar(ds,"true_tend",Float64,("lon","lat","rho"), 
        attrib = Dict("units" => "m/s^2","standard_name" => "true time-mean tendency"))
    rc = defVar(ds,"rec_tend",Float64,("lon","lat","rho"), 
        attrib = Dict("units" => "m/s^2","standard_name" => "reconstruction time-mean tendency"))

    nslice = Int(ceil(nt/tstride))
    for idx = 1:nslice
        tb = tmin + (idx-1)*tstride
        te = min(tmin + idx*tstride-1, tmax)
        U, V, rho, N, lon, lat, Z, ugw, vgw = load_waccm_data([xmin,ymin,zmin,tb],[xmax,ymax,zmax,te])
        for ti=1:(te-tb+1), yi=1:(ymax-ymin+1), xi=1:(xmax-xmin+1)
            col = ColumnProfile(U[xi,yi,end:-1:2,ti], V[xi,yi,end:-1:2,ti], N[xi,yi,end:-1:1,ti], rho[xi,yi,end-1:-1:2,ti], nz2-1)
            momdep = cat(dims=1,ugw[xi,yi,end:-1:2,ti], vgw[xi,yi,end:-1:2,ti]) # concatenate x- and y-fluxes

            if norm(momdep) == 0.0
                # if norm(momdep)==0, the wave didn't break and reconstructing it is going to be meaningless
                continue
            end 

            spectrum .= lin_invert(L81_wp!, waves, momdep, col; lam, reg, refine)
            recalc_momdep = spectrum_to_momdep(L81_wp!, waves, col, spectrum)
            
            tmp .= (momdep .- recalc_momdep)
            col_err[xi,yi,:] .+= (tmp.^2)
            truth[xi,yi,:] .+= momdep./ncols
            recalc[xi,yi,:] .+= recalc_momdep./ncols
        end
        #println(idx)
    end
    ce .= sqrt.(col_err/ncols)./stdev
    tr .= truth
    rc .= recalc
    close(ds)
    return nothing
end