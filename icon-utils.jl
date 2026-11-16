function load_and_flip_data(ncfile)
    Z = ncread(f,"z")[:,end:-1:1]
    T = ncread(f,"time")                       # this doesn't flip
    XI = ncread(f,"xidx")                      # this doesn't flip
    DX = ncread(f,"dx")[:,end:-1:1,:]
    DY = ncread(f,"dy")[:,end:-1:1,:]
    U = ncread(f,"u")[:,end:-1:1,:]
    V = ncread(f,"v")[:,end:-1:1,:]
    RH = ncread(f,"rho")[:,end:-1:1,:]
    return Z,T,XI,DX,DY,U,V,RH
end

function tavg_data(ncfile,tbat::Ti) where {Ti<:Integer}
    Z = ncread(f,"z")[:,end:-1:1]
    T = ncread(f,"time")                       # this doesn't flip
    XI = ncread(f,"xidx")                      # this doesn't flip
    DX = ncread(f,"dx")[:,end:-1:1,:]
    DY = ncread(f,"dy")[:,end:-1:1,:]
    U = ncread(f,"u")[:,end:-1:1,:]
    V = ncread(f,"v")[:,end:-1:1,:]
    RH = ncread(f,"rho")[:,end:-1:1,:]
    nt = length(T)
    nxi = length(XI)
    nz = size(Z,2)
    ntb = div(nt,tbat)
    t = zeros(ntb)
    dx = zeros(nxi,nz,ntb)
    dy = zeros(nxi,nz,ntb)
    u = zeros(nxi,nz,ntb)
    v = zeros(nxi,nz,ntb)
    rh = zeros(nxi,nz,ntb)
    batchmean = (A,i) -> mean(A[:,:,(tbat*(i-1)).+(1:tbat)],dims=3)
    for i = 1:ntb
        dx[:,:,i] .= batchmean(DX,i)
        dy[:,:,i] .= batchmean(DY,i)
        u[:,:,i]  .= batchmean(U,i)
        v[:,:,i]  .= batchmean(V,i)
        rh[:,:,i] .= batchmean(RH,i)
    end
    return Z,t,XI,dx,dy,u,v,rh
end

function err_in_mean_icon(U::AbstractArray{Tv}, V::AbstractArray{Tv}, 
        rho::AbstractArray{Tv}, N::Function, 
        ugw::AbstractArray{Tv}, vgw::AbstractArray{Tv},
        waves::AbstractArray{WaveProfile};
        start_idxs=[1,25,1], end_idxs=nothing, obs_wndw=nothing
    ) where {Tv<:AbstractFloat}

    nxi,nz,nt = size(U)
    nz_d = nz
    if end_idxs === nothing
        end_idxs = start_idxs .+ (-1 .+ size(U))   # if size[1] = 3, then indices 1:(1+3) will have 4 entries, not 3 as desired
    end
    if obs_wndw === nothing
        # "observation window" for the column
        # a way of measuring error only over a subset of the column
        # e.g. in the case of launching waves below the troposphere (where gw drag is nonphysical)
        obs_wndw = [1:nz_d; nz.+(1:nz_d)] # entire column except for 
    end
    nobs = length(obs_wndw)

    # Defining parameters for inversion
    lam = 1e-3                          # lambda, regularization parameter. 1e-3 is usually right

    # read in the stdev of drag over the entire year for use in normalizing error
    #TODO

    ncol = nxi*nt
    err = zeros(ncol, 3)    # [error, xi, ti]
    column_error = zeros(nobs)
    column_dev = zeros(nobs)
    mean_recon = zeros(2*nz_d)
    mean_truth = zeros(2*nz_d)
    
    spectra = zeros(nxi, length(waves), nt)
    
    for ti=1:nt, xi=1:nxi
        idx = 1 + (xi-1) + nxi*(ti-1)
        
        col = ColumnProfile(U[xi,:,ti], V[xi,:,ti], Nfun.(Z[xi,:]), rho[xi,:,ti], nz_d)

        momdep = cat(dims=1,ugw[xi,:,ti], vgw[xi,:,ti]) # concatenate x- and y-fluxes

        if norm(momdep) == 0.0
            # if norm(momdep)==0, the wave didn't break and reconstructing it is going to be meaningless
            err[idx,:] .= [NaN, xi, ti]
            continue
        end

        spectrum = lin_invert(L81_wp!, waves, momdep, col; ref_flux=1e-4, lam, reg="l2", refine=1) # propag_grad=L81_grad,
        spectra[xi, :, ti] .= spectrum
        recalc_momdep = spectrum_to_momdep(L81_wp!, waves, col, spectrum)
        
        # test correctness
        # propagate all the waves and add their momfluxes
        mean_truth .+= momdep
        mean_recon .+= recalc_momdep
        tr = momdep[obs_wndw]
        rc = recalc_momdep[obs_wndw]
        @views column_dev .+= ((tr .- rc)) #./ meanyz[yi,obs_wndw]
        @views column_error .+= (tr .- rc).^2
        error = norm(tr .- rc)/norm(tr)

        err[idx,:] .= [error, xi, ti]
    end

    mean_recon ./= (nt*nxi)
    mean_truth ./= (nt*nxi)
    column_dev .= column_dev/(nt*nxi)
    column_error .= sqrt.((column_error)./(nt*nxi))#./stdev[:,:,obs_wndw]
    return err, column_error, column_dev, mean_recon, mean_truth, spectra
end