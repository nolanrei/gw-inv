#include("waccm-utils.jl")
#include("L81.jl")
using NetCDF

function err_in_mean(U::AbstractArray{Tv}, V::AbstractArray{Tv}, 
        rho::AbstractArray{Tv}, N::AbstractArray{Tv}, 
        ugw::AbstractArray{Tv}, vgw::AbstractArray{Tv},
        waves::AbstractArray{WaveProfile};
        start_idxs=[1,1,25,1], end_idxs=nothing, obs_wndw=nothing
    ) where {Tv<:AbstractFloat}

    nx,ny,nz,nt = size(U)
    nz_d = nz-1
    if end_idxs === nothing
        end_idxs = start_idxs .+ (-1 .+ size(U))   # if size[1] = 3, then indices 1:(1+3) will have 4 entries, not 3 as desired
    end
    if obs_wndw === nothing
        # "observation window" for the column
        # a way of measuring error only over a subset of the column
        # e.g. in the case of launching waves below the troposphere (where gw drag is nonphysical)
        obs_wndw = 1:2(nz_d)#[1:nz_d; nz.+(1:nz_d)] # entire column except for 
    end
    nobs = length(obs_wndw)

    # Defining parameters for inversion
    lam = 1e-3                          # lambda, regularization parameter. 1e-3 is usually right

    # read in the stdev of drag over the entire year for use in normalizing error
    #=
    meanfull = ncread("mean_and_std.nc","meanfull")
    stdfull = ncread("mean_and_std.nc","stdfull")
    idxs = vcat(34:-1:1,68:-1:35)
    meanyz = ones(ny,2*nz_d)
    stdyz = ones(ny,2*nz_d)
    idxs2 = vcat(start_idxs[3].+(0:33), nz_d + start_idxs[3].+(0:33))
    meanyz[:,idxs2] = dropdims(mean(meanfull[start_idxs[1]:end_idxs[1],start_idxs[2]:end_idxs[2],idxs],dims=1),dims=1)
    stdyz[:,idxs2] = dropdims(sqrt.(mean(stdfull[start_idxs[1]:end_idxs[1],start_idxs[2]:end_idxs[2],idxs].^2,dims=1)),dims=1)
    =#
    stdfull = ncread("mean_and_std.nc","stdfull")
    nz2 = (end_idxs[3]-start_idxs[3]+1)
    nh = div(size(stdfull,3),2)
    idxs = vcat(nz2:-1:1,nh.+(nz2:-1:1))
    stdev = stdfull[start_idxs[1]:end_idxs[1], start_idxs[2]:end_idxs[2], idxs]
    # note that this normalizes by the local stdev, not the zonally averaged

    ncol = nx*ny*nt
    err = zeros(ncol, 4)    # [error, xi, yi, ti]
    column_error = zeros(nx,ny,nobs)
    column_dev = zeros(nx,ny,nobs)
    mean_recon = zeros(ny,2*nz_d)
    mean_truth = zeros(ny,2*nz_d)
    
    spectra = zeros(nx, ny, length(waves), nt)
    
    for ti=1:nt, yi=1:ny, xi=1:nx
        idx = 1 + (xi-1) + nx*((yi-1) + ny*(ti-1))
        
        col = ColumnProfile(U[xi,yi,end:-1:2,ti], V[xi,yi,end:-1:2,ti], N[xi,yi,end:-1:1,ti], rho[xi,yi,end-1:-1:2,ti], nz-1)

        momdep = cat(dims=1,ugw[xi,yi,end:-1:2,ti], vgw[xi,yi,end:-1:2,ti]) # concatenate x- and y-fluxes

        if norm(momdep) == 0.0
            # if norm(momdep)==0, the wave didn't break and reconstructing it is going to be meaningless
            err[idx,:] .= [NaN, xi, yi, ti]
            continue
        end

        spectrum = lin_invert(L81_wp!, waves, momdep, col; lam, reg="l2", refine=1) # propag_grad=L81_grad,
        spectra[xi, yi, :, ti] .= spectrum
        recalc_momdep = spectrum_to_momdep(L81_wp!, waves, col, spectrum)
        
        # test correctness
        # propagate all the waves and add their momfluxes
        mean_truth[yi,:] .+= momdep
        mean_recon[yi,:] .+= recalc_momdep
        tr = momdep[obs_wndw]
        rc = recalc_momdep[obs_wndw]
        @views column_dev[xi,yi,:] .+= ((tr .- rc)) #./ meanyz[yi,obs_wndw]
        @views column_error[xi,yi,:] .+= (tr .- rc).^2
        error = norm(tr .- rc)/norm(tr)

        err[idx,:] .= [error, xi, yi, ti]
    end

    mean_recon ./= (nt*nx)
    mean_truth ./= (nt*nx)
    column_dev .= column_dev/(nt*nx)
    println
    column_error .= sqrt.((column_error)./(nt))#./stdev[:,:,obs_wndw]
    return err, column_error, column_dev, mean_recon, mean_truth, spectra
end