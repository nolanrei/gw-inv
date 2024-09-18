include("waccm-utils.jl")
include("L81.jl")
using Plots

function L81_validate(U::Array{Float64}, V::Array{Float64}, 
        rho::Array{Float64}, N::Array{Float64},
        ugw::Array{Float64}, vgw::Array{Float64};
        wavespds=nothing,ndirs=4,ref_flux=1e-4,reg="l2",
        use_waccm_drag=false,refine=false)
    #=
    loop over each column in data to test whether lin_invert(L81_wp(wave)) = wave for a new random wave in each column
    =#
    nx,ny,nz,nt = size(U)
    if wavespds == nothing
        wavespds = [i*1.0 for i=-100:100]
    end

    src = 1
    lam = 1e-4
    nk = length(wavespds)
    thetas = zeros(1, ndirs)
    for i = 1:ndirs
        thetas[i] = pi*(i-1)/ndirs
    end
    dirs = vcat(cos.(thetas), sin.(thetas))

    wav = WaveProfile(2pi/3000, 0.0, 0.0, 0.0, MVector(1.0, 0.0, 10.0), MVector(1e-3, 0.0), src)
    nlev = size(U,3)-1           # number of z points in our columns doesn't change
    
    waves = Array{WaveProfile,1}(undef, nk*ndirs)
    for j = 1:ndirs
        for i = 1:nk
            c = SVector(wavespds[i]*dirs[1,j], wavespds[i]*dirs[2,j], 10.0)
            flux = SVector{2}(ref_flux*dirs[:,j])
            waves[(j-1)*nk+i] = WaveProfile(2pi/3000, 0.0, 0.0, 0.0, c, flux, src);
        end
    end

    ncol = nx*ny*nt
    err = zeros(ncol, 7)    # [error, cx, cy]
    
    for ti=1:nt, yi=1:ny, xi=1:nx
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
            err[idx,:] .= [NaN, wav.c[1], wav.c[2], wav.flux[1], wav.flux[2], xi, yi]
            continue
        end

        spectrum,GX = lin_invert(L81_wp, waves, momdep, col; ref_flux, lam, reg, refine) # propag_grad=L81_grad,
    
        # test correctness
        # propagate all the waves and add their momfluxes
        recalc_momdep = spectrum_to_momdep(L81_wp, waves, col, spectrum)
        #recalc_momdep = GX*spectrum
    
        error = norm(momdep .- recalc_momdep)/norm(momdep)
        #println(idx)
        err[idx,:] .= [error, wav.c[1], wav.c[2], wav.flux[1], wav.flux[2], xi, yi]
    end

    return err
end