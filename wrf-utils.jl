using NetCDF

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

function wrf_get_fileset(cases::Array{<:Integer})
    input_dir  = "/projects/rps/epg2/gerberlab/WRF/INPUT/input_1deg_case"
    output_dir = "/projects/rps/epg2/gerberlab/WRF/OUTPUT/mf_case"
    
    ifiles = Dict("U"=>[], "V"=>[]) #, "Q"=>[])
    ofiles = Dict("UW"=>[], "VW"=>[])
    v2d = Dict("UW"=>"/uw/", "VW"=>"/vw/")
    for ivar in keys(ifiles)
        for case in cases
            idir = input_dir*string(case)
            append!(ifiles[ivar],text_filter(r"\d{4}-\d{2}-\d{2}\Q.nc\E",readdir(idir,join=true)))
        end
    end

    for ovar in keys(ofiles)
        for case in cases
            odir = output_dir*string(case)*v2d[ovar]*"sgs_rey"          # sgs_rey is the Reynolds fluxes; sgs_tau would be the uugs
            append!(ofiles[ovar],text_filter(r"\d{4}-\d{2}-\d{2}\Q.nc\E",readdir(odir,join=true)))
        end
    end

    return merge(ifiles, ofiles)
end

function wrf_load(files::Dict{String,Vector{Any}})
    # Need variables U, V, q, rho, N, Dx, Dy, lon, lat, z, t
    # rho and N need to be falsified
    # z and t need to be inferred from file structure
    # Dx and Dy need to be calculated from UW and VW, resp.
    # output in ncases x nlon x nlat x nz x nt form

    # load in the "meat" of the data
    v2k = Dict("U"=>"u1deg", "V"=>"v1deg", "UW"=>"rey_g", "VW"=>"rey_g") #"Q"=>"q1deg",
    vars = Dict("U"=>[], "V"=>[], "UW"=>[], "VW"=>[]) #"Q"=>[],
    dims = ("lon", "lat")
    for var in keys(files)
        nf = length(files[var])
        vars[var] = Vector{Array{Float64,4}}(undef,nf)
        key = v2k[var]
        for idx = 1:nf
            f = files[var][idx]
            vars[var][idx] = ncread(f,key)
        end
    end

    # Supply z -- all input files share one z scheme, and so do all output files
    vars["iz"] = 1:65
    vars["oz"] = 10:0.5:65
    vars["oz_dx"] = 10.25:0.5:64.75     # originally 10:0.5:65 but then we're averaging it for DX and DY

    # Supply rho and N
    vars["N"] = ncread("/scratch/nr2489/work/loose_spectra/waccm_rho_and_N.nc","N")
    vars["rho"] = ncread("/scratch/nr2489/work/loose_spectra/waccm_rho_and_N.nc","rho")
    nzl = length(vars["rho"])
    nzs = nzl - 1
    rho = zeros(1,1,nzl,1)
    rho[1,1,:,1] .= vars["rho"]
    rhoi = zeros(1,1,nzs,1)
    rhoi[1,1,:,1] .= 0.5*(rho[1,1,2:end,1] .+ rho[1,1,1:end-1,1])

    # Calculate DX and DY
    nX = length(vars["UW"])
    nY = length(vars["VW"])
    vars["DX"] = Vector{Array{Float64,4}}(undef,nX)
    vars["DY"] = Vector{Array{Float64,4}}(undef,nY)
    for idx = 1:nX
        fv = vars["UW"][idx].*rho
        vars["DX"][idx] = (fv[:,:,2:end,:].-fv[:,:,1:end-1,:])/(500)    # d/dz(uw), divided by dz = 500m
        vars["DX"][idx] ./= rhoi
    end
    for idx = 1:nY
        fv = vars["VW"][idx].*rho
        vars["DY"][idx] = (fv[:,:,2:end,:].-fv[:,:,1:end-1,:])/(500)    # d/dz(vw), divided by dz = 500m
        vars["DY"][idx] ./= rhoi
    end

    # Supply t -- julian date
    vars["t"] = Vector{Array{Float64,1}}(undef,nX)
    for idx = 1:nX
        f = files["U"][idx]
        vars["t"][idx] = ncread(f,"time")
    end
   
    return vars
end

function wrf_postprocess!(vars::Dict)
    # U and V are on the wrong grid. Interpolate them from the iz grid onto the oz grid
    nU = length(vars["U"])
    nV = length(vars["V"])
    vars["Up"] = Vector{Array{Float64,4}}(undef,nU)
    vars["Vp"] = Vector{Array{Float64,4}}(undef,nV)
    zb = 10
    ze = 65
    noz = length(vars["oz"])
    for idx = 1:nU
        fv = vars["U"][idx]
        nx,ny,nz,nt = size(fv)
        tmp = zeros(nx,ny,noz,nt)
        tmp[:,:,1:2:end,:] .= fv[:,:,zb:ze,:]  # "interpolate" onto integer mark
        tmp[:,:,2:2:end,:] .= 0.5*fv[:,:,zb:ze-1,:]+0.5*fv[:,:,zb+1:ze,:]  # interpolate onto X.5 mark
        #tmp[:,:,1:2:end,:] .= 0.75*fv[:,:,zb:ze-1,:]+0.25*fv[:,:,zb+1:ze,:]  # interpolate onto X.25 mark
        #tmp[:,:,2:2:end,:] .= 0.25*fv[:,:,zb:ze-1,:]+0.75*fv[:,:,zb+1:ze,:]  # interpolate onto X.75 mark
        vars["Up"][idx] = 0.0.+tmp     # 0.0.+tmp is supposed to make a copy
    end
    for idx = 1:nV
        fv = vars["V"][idx]
        nx,ny,nz,nt = size(fv)
        tmp = zeros(nx,ny,noz,nt)
        tmp[:,:,1:2:end,:] .= fv[:,:,zb:ze,:]  # "interpolate" onto integer mark
        tmp[:,:,2:2:end,:] .= 0.5*fv[:,:,zb:ze-1,:]+0.5*fv[:,:,zb+1:ze,:]  # interpolate onto X.5 mark
        #tmp[:,:,1:2:end,:] .= 0.75*fv[:,:,zb:ze-1,:]+0.25*fv[:,:,zb+1:ze,:]  # interpolate onto X.25 mark
        #tmp[:,:,2:2:end,:] .= 0.25*fv[:,:,zb:ze-1,:]+0.75*fv[:,:,zb+1:ze,:]  # interpolate onto X.75 mark
        vars["Vp"][idx] = 0.0.+tmp     # 0.0.+tmp is supposed to make a copy
    end
    return vars
end

function wrf_load_conv_indics!(vars, files)
    #=
    Load in convection indicators (olr, precip) into vars
    olr + precip are both in the inputs files, so files["U"] will find them
    =#
    nf = length(files["U"])
    vars["precip"] = Vector{Array{Float64,4}}(undef,nf)
    vars["olr"] = Vector{Array{Float64,4}}(undef,nf)
    for idx = 1:nf
        f = files["U"][idx]
        tmp = ncread(f,"precip")
        sz = size(tmp)
        vars["precip"][idx] = reshape(tmp, (sz[1], sz[2], 1, sz[3]))
        tmp = ncread(f,"olr")
        vars["olr"][idx] = reshape(tmp, (sz[1], sz[2], 1, sz[3]))
    end
end

function wrf_tstamps_cases(files)
    dates = []
    cases = Set()
    case_order = []
    case_idxs = Dict()
    idx2case = zeros(Int64,length(files))
    for idx = 1:length(files)
        f = files[idx]
        regmatch = match(r"(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})\Q.nc\E", f)
        date = DateTime(parse(Int,regmatch[:year]), parse(Int,regmatch[:month]), parse(Int,regmatch[:day]))
        push!(dates,date)

        regmatch2 = match(r"\Qcase\E(?<case>\d+)", f)
        case = parse(Int,regmatch2[:case])
        if !(case in cases)
            push!(cases, case)
            push!(case_order,case)       # meant to keep the order of the cases identical to load order
            case_idxs[case] = length(case_order)  # match the cases to their index in load order
        end
        idx2case[idx] = case_idxs[case]
    end
    return dates, idx2case, case_order
end

function wrf_order(tstamps, idx2case, case_order)
    # Two stages -- sort files by case, and then within each case sort by time order
    # First stage groups by case
    nc = length(case_order)
    idxs = [[] for ci = 1:nc]
    for idx = 1:length(tstamps)
        push!(idxs[idx2case[idx]],idx)
    end
    # And now sort by time within cases
    for ci = 1:nc
        sort!(idxs[ci])
    end
    return idxs
end

function wrf_knit(var, order)
    # On loading in, all the variables are in [f1.var, f2.var, f3.var,...] format
    # This function pieces together the different files in time order, 
    # keeping different cases distinct
    nc = length(order)
    vark = Array{Array{Float64,4},1}(undef,nc)
    for ci = 1:nc
        nf = length(order[ci])
        sz = collect(size(var[order[ci][1]]))     # size of all the files concatenated within this case
        for i = 2:nf
            sz[4] += size(var[order[ci][i]],4)    # add up time dimensions
        end
        vark[ci] = zeros(Float64,sz...)
        ti = 1
        for i = 1:nf
            n4 = size(var[order[ci][i]],4)
            vark[ci][:,:,:,ti:(ti+n4-1)] .= var[order[ci][i]]
            ti += n4
        end
    end
    return vark
end

function drag_means!(vars)
    ncases = length(vars["DX"])
    vars["means"] = Vector{Array{Float64,1}}(undef,ncases)
    vars["std"] = Vector{Array{Float64,1}}(undef,ncases)
    for idx = 1:ncases
        mnx = dropdims(mean(vars["DX"][idx],dims=(1,2,4)),dims=(1,2,4))
        mny = dropdims(mean(vars["DY"][idx],dims=(1,2,4)),dims=(1,2,4))
        vars["means"][idx] = vcat(mnx,mny)
        stdx = dropdims(std(vars["DX"][idx],dims=(1,2,4)),dims=(1,2,4))
        stdy = dropdims(std(vars["DY"][idx],dims=(1,2,4)),dims=(1,2,4))
        vars["std"][idx] = vcat(stdx,stdy)
    end
    return nothing
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
