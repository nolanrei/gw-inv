include("waccm-utils.jl")
include("L81.jl")
using Plots
using NetCDF

function L81_validate(U::Array{Float64}, V::Array{Float64}, 
        rho::Array{Float64}, N::Array{Float64},
        ugw::Array{Float64}, vgw::Array{Float64};
        wavespds=nothing,ndirs=4,nsrc=4,ref_flux=1e-4,reg="l2",
        use_waccm_drag=false,refine=1,sum_vs_best_src=true)
    #=
    loop over each column in data to test whether lin_invert(L81_wp(wave)) = wave for a new random wave in each column
    =#
    nx,ny,nz,nt = size(U)
    if wavespds == nothing
        wavespds = [i*1.0 for i=-100:100]
    end

    # read in the stdev of drag over the entire year for use in normalizing error
    stdz = ncread("mean_and_std.nc","stdz")
    stdz = stdz[vcat(2:nz,nz+2:2*nz)]

    waves = make_waveset(wavespds, ndirs, nsrc)

    ncol = nx*ny*nt
    err = zeros(ncol, 5)    # [error, cx, cy]
    column_error = zeros(2*(nz-1))
    colnorm = 1e-3*ones(2*(nz-1))
    th = 0.1
    
    suberrors = zeros(nsrc)
    
    for ti=1:nt, yi=1:ny, xi=1:nx
        best_j = -1
        idx = 1 + (xi-1) + nx*((yi-1) + ny*(ti-1))
        
        col = ColumnProfile(U[xi,yi,end:-1:2,ti], V[xi,yi,end:-1:2,ti], N[xi,yi,end:-1:1,ti], rho[xi,yi,end-1:-1:2,ti], nlev)
    
        # multivariate Gaussians of equal variance are isotropic
        wav.c[1:2] .= 50*randn(2)
        #wav.c[1] = 50*randn()
        #wav.c[2] = 0.0
        wav.flux .= 1e-3*exp(randn())*wav.c[1:2]/norm(wav.c[1:2])

        if use_waccm_drag
            momdep = cat(dims=1,ugw[xi,yi,end:-1:2,ti], vgw[xi,yi,end:-1:2,ti]) # concatenate x- and y-fluxes
        else
            momdep = L81_wp(wav,col)
        end

        if norm(momdep) == 0.0
            # if norm(momdep)==0, the wave didn't break and reconstructing it is going to be meaningless
            err[idx,:] .= [NaN, xi, yi, ti, 0]
            continue
        end

        #if sum_vs_best_src
        spectrum = lin_invert(L81_wp!, waves, momdep, col; ref_flux, lam, reg, refine) # propag_grad=L81_grad,
        recalc_momdep = spectrum_to_momdep(L81_wp!, waves, col, spectrum)
            #=
        else
            spectrum = zeros(nwav)
            for j = 1:nsrc
                idxs = ((nk*ndirs)*(j-1)+1):(nk*ndirs)*j
                spectrum[idxs],_ = lin_invert(L81_wp!, waves[idxs], momdep, col; ref_flux, lam, reg, refine)
                recalc_momdep = spectrum_to_momdep(L81_wp!, waves[idxs], col, spectrum)
                recalc_momdep .= ((momdep .- recalc_momdep)).^2 ./ stdz
                suberrors[j] = norm(recalc_momdep)
            end
            best_j = argmax(suberrors)

            # clean out the rest of the spectrum array so the reconstruction is accurate
            spectrum[1:((nk*ndirs)*(best_j-1)+1)] .= 0.0
            spectrum[(nk*ndirs)*best_j+1:end] .= 0.0
        end
            =#
    
        # test correctness
        # propagate all the waves and add their momfluxes
        

        column_error .+= ((momdep .- recalc_momdep)).^2 ./ stdz
    
        error = norm(momdep .- recalc_momdep)/norm(momdep)
        #println(idx)
        err[idx,:] .= [error, xi, yi, ti, best_j]
    end

    column_error = sqrt.(column_error/ncol)
    return err, column_error
end

