using LinearAlgebra
using SparseArrays
using StaticArrays
using PROPACK
using IterativeSolvers
using Lasso

mutable struct WaveProfile
    k::Float64                  # x wavenumber
    l::Float64                  # y wavenumber
    m::Float64                  # z wavenumber
    om::Float64                 # omega; frequency
    c::MVector{3,Float64}       # group velocity
    flux::MVector{2,Float64}    # amount of momentum flux carried
    src::Int64                  # index of vertical level at which waves are launched
end

struct ColumnProfile
    U::Array{Float64,1}         # zonal velocity
    V::Array{Float64,1}         # meridional velocity
    N::Array{Float64,1}         # buoyancy frequency
    rho::Array{Float64,1}       # density
    nlev::Int64                 # length of arrays
end

function Base.copy(wav::WaveProfile)
    # Copy to a new object
    return WaveProfile(wav.k, wav.l, wav.m, wav.om, copy(wav.c), copy(wav.flux), wav.src)
end

function Base.copy!(wav1::WaveProfile, wav2::WaveProfile)
    # Copy wav2 to wav1
    wav1.k = wav2.k
    wav1.l = wav2.l
    wav1.m = wav2.m
    wav1.om = wav2.om
    wav1.c .= wav2.c
    wav1.flux .= wav2.flux
    wav1.src = wav2.src
    return nothing
end

function WaveProfile()
    return WaveProfile(0.0,0.0,0.0,0.0,SVector(0.0,0.0,0.0),SVector(0.0,0.0),0)
end

function make_WaveProfile(k, l, m, om, f, flux, source)
    # Add group velocity and flux calculations
    K = sqrt(k^2 + l^2)
    K3 = sqrt(k^2 + l^2 + m^2)
    hvel = om*m^2/(K*K3^2)      #horizontal velocity
    cx = hvel*k/K
    cy = hvel*l/K
    cz = -om*m/K3^2
    omf = om*om - f*f
    opf = om*om + f*f

    # Flux is calculated as <u'w'>, <v'w'> in terms of the input variable "flux"=<p'p'>
    # Not parallel to group velocity in the presence of rotation
    fx = flux*( (-k/m)*(l^2*f^2 + om^2*k^2)/(omf^2) + (-l/m)*(k*l/omf) )
    fy = flux*( (-k/m)*(k*l/omf) + (-l/m)*(k^2*f^2 + om^2*l^2)/(opf^2) )
    return WaveProfile(k,l,m,om,MVector(cx,cy,cz),MVector(fx,fy),source) 
end

function wavePacket(wav0::WaveProfile, stdev, dm)
    # Traditional wave packets keep direction constant and vary wavespeed
    # This means keeping k/l constant and varying m (for k ~ l ~ m, c ~ m^2)
    m0 = wav0.m
    mmin = m0-4*stdev   # this allows negative m. Should it?
    mmax = m0+4*stdev   # also, this is the wrong cutoff for the Gaussian going to 0, but how to do better?
    ms = LinRange(mmin, mmax, dm)
    nm = length(ms)
    c0 = sqrt(wav0.cx^2 + wav0.cy^2)
    
    out = Array{WaveProfile,1}(undef,nm)
    for idxm = 1:nm
        mi = ms[idxm]
        c = wav0.om*mi^2/(sqrt(wav0.k^2 + wav0.l^2)*(wav0.k^2 + wav0.l^2 + mi^2))
        flux = wav0.flux * exp(((c-c0)/stdev)^2)
        out[idxm] = make_WaveProfile(wav0.k, wav0.l, mi, wav0.om, 2pi/86400, flux, wav0.source)
    end

    return out
end

function isotropic_wave_packet(wav0::WaveProfile, stdev, dk)
    # TODO
    return 0
end

function L81_wp(wav::WaveProfile, col::ColumnProfile)
    # Compute gravity wave flux using the Lindzen '81 propagation scheme
    # "Lindzen '81 WavePacket formulation" to differentiate from wavespeed formulation below
    # Outputs a vector of dimensions (2*nlev, 1) with [Dx; Dy]

    momdep = zeros(2*col.nlev)               # array for momentum deposited by wave (solution array)
    flux = norm(wav.flux)                    # amount of momentum transported
    momdir = (flux > 0 ? wav.flux/flux : [1 0])      # direction of mom transport
    c = norm(wav.c[1:2])                     # wavespeed in the horizontal
    cdir = (c > 0 ? wav.c/c : [1 0 0])       # direction of wave propagation
    
    v = col.U[wav.src]*cdir[1] + col.V[wav.src]*cdir[2]        # mean flow speed in direction of wave at src

    b_sign_src = sign(c-v)
    # Loop from bottom to top of column, depositing momentum where it exceeds maximum stable transport
    for lvl = wav.src:col.nlev
        if flux == 0
            # if the wave has no momentum we don't need to keep going
            break
        end
        v = col.U[lvl]*cdir[1] + col.V[lvl]*cdir[2]                   # mean flow speed in direction dir

        b = col.rho[lvl]/col.rho[wav.src] * 0.5*abs(wav.k)*(c-v)^3/col.N[lvl]      # breaking condition -- maximum stable momentum transport
        #println(b)

        # Set b to 0 above 1st sign change; since sign(b) = sign(c-v), this corresponds to detecting critical layers
	    # that occur between levels of the discretization
    	if b*b_sign_src <= 0
            b = 0
            b_sign_src = 0   # cuts off any further deposition
        end
        #println("$(b)\t$(flux)")
        #println("$(v)\t$(c)")

        # If |flux| > b, set flux to b and deposit extra at this level
        b = abs(b)              # we don't care about the sign
        if flux > b
            fb = flux-b
            # we assume that x-momentum gets deposited proportional to the amount of x-momflux
            # and similarly with y
            momdep[lvl] = fb*momdir[1]                    # zonal
            momdep[lvl+col.nlev] = fb*momdir[2]           # meridional (all the ys at the end)
            flux = b
        end
    end

    return momdep
end

function L81_grad(wav::WaveProfile, col::ColumnProfile)
    # Compute gravity wave flux using the Lindzen '81 propagation scheme
    # Piggybacks on the flux calculation to compute the derivative w.r.t. wave amplitude
    # Outputs two vectors of dimensions (2*nlev, 1) with [Dx; Dy]

    momdep = zeros(2*col.nlev)               # array for momentum deposited by wave (solution array)
    deriv  = spzeros(2*col.nlev)             # derivative array (solution array)
    flux = norm(wav.flux)                    # amount of momentum transported
    momdir = (flux > 0 ? wav.flux/flux : [1 0])      # direction of mom transport
    c = norm(wav.c[1:2])                     # wavespeed in the horizontal
    cdir = (c > 0 ? wav.c/c : [1 0 0])       # direction of wave propagation
    first_deposition = false
    
    v = col.U[wav.src]*cdir[1] + col.V[wav.src]*cdir[2]        # mean flow speed in direction of wave at src

    b_sign_src = sign(c-v)
    # Loop from bottom to top of column, depositing momentum where it exceeds maximum stable transport
    for lvl = wav.src:col.nlev
        if flux == 0
            # if the wave has no momentum we don't need to keep going
            break
        end
        v = col.U[lvl]*cdir[1] + col.V[lvl]*cdir[2]                   # mean flow speed in direction dir

        b = col.rho[lvl]/col.rho[wav.src] * 0.5*abs(wav.k)*(c-v)^3/col.N[lvl]      # breaking condition -- maximum stable momentum transport
        #println(b)

        # Set b to 0 above 1st sign change; since sign(b) = sign(c-v), this corresponds to detecting critical layers
	    # that occur between levels of the discretization
    	if b*b_sign_src <= 0
            b = 0
            b_sign_src = 0   # cuts off any further deposition
        end

        # If |flux| > b, set flux to b and deposit extra at this level
        b = abs(b)              # we don't care about the sign
        #println("flux\t b")
        if flux > b
            if !first_deposition
                # any additional momentum gets dropped at the first opportunity
                # so the derivative is only nonzero there
                deriv[lvl] = momdir[1]
                deriv[lvl+col.nlev] = momdir[2]
                first_deposition = true
            end
            fb = flux-b
            # we assume that x-momentum gets deposited proportional to the amount of x-momflux
            # and similarly with y
            momdep[lvl] = fb*momdir[1]                    # zonal
            momdep[lvl+col.nlev] = fb*momdir[2]           # meridional (all the ys at the end)
            flux = b
        end
    end

    return momdep,deriv
end

function L81_ws(c::Float64, k::Float64, flux::Float64, dir::Vector{Number}, src_lvl::Int64, col::ColumnProfile)
    # Compute gravity wave flux using the Lindzen '81 propagation scheme
    # "Lindzen '81 WaveSpeed formulation" to differentiate from wavepacket formulation above
    vel = col.U*dir[1] + col.V*dir[2]       # mean flow speed in direction of wave
    momdep = zeros(col.nlev)                # array for momentum deposited by wave (solution array)
    flux_z = flux                           # flux at a height z (for internal use)

    b_sign_src = sign(c-vel[src])
    # Loop from bottom to top of column, depositing momentum where it exceeds maximum stable transport
    for lvl = src_lvl:col.nlev
        if flux_z == 0
            # if the wave has no momentum we don't need to keep going
            break
        end
        v = col.U[lvl]*dir[1] + col.V[lvl]*dir[2]                   # mean flow speed in direction dir

        b = col.rho[lvl]/col.rho[src_lvl] * 0.5*abs(k)*(c-vel[lvl])^3/col.N[lvl]      # breaking condition -- maximum stable momentum transport
        # note that we don't care about sign(b) as long as it doesn't switch signs, so later we'll be taking abs(b)

        # Set b to 0 above 1st sign change; since sign(b) = sign(c-v), this corresponds to detecting critical layers
	    # that occur between levels of the discretization
    	if b*b_sign_src <= 0
            b = 0
            b_sign_src = 0   # cuts off any further deposition
        end
        #println("$(b)\t$(flux_z)")

        # If |flux| > b, set flux to b and deposit extra at this level
        b = abs(b)              # we don't care about the sign
        if flux_z > b
            momdep[lvl] = flux_z - b
            flux_z = b
        end
    end

    return momdep
    
end

function lin_invert(propag::Function, 
        waves::Vector{WaveProfile},
        momdep::Vector{Float64}, 
        col::ColumnProfile; 
        ref_flux::Float64=1e-3, lam::Float64=1e-3, 
        propag_grad=nothing,
        reg="l2", refine=false)
    #=
    To a close approximation, gravity wave propagation and breaking is linear:
    parameterizations are completely noninteracting between wavenumbers (except for MSGWaM), and
    variations in amplitude are slightly nonlinear when critical layers are not full-occlusion.
    Thus, we can approximate the wave propagation problem
                      g(X, S) = D
    by the linear problem
                      G_X*S = D.
    G_X has dimensions (nz x nk), where nz is #(pts in the vertical) and nk is # wavespeeds.
    If these are small enough we can construct G_X explicitly as G_X[:,i] = g(X,ref_flux*e_i)/ref_flux
    (rescaling to not pick up too much detail. g(X,ref_flux*e_i) ~ G_X*ref_flux, so remember to divide by it at the end)
    In principle, the solve could be done by Krylov space methods if forming the matrix is impractical.
    =#

    # construct the matrix
    nz = size(momdep,1)     # size of propag output vectors
    nc = length(waves)      # number of waves we're considering
    
    GX = zeros((nz,nc))
    for i = 1:nc
        GX[:,i] .= propag(waves[i], col)
    end
    sGX = sparse(GX)

    # solve!
    if reg == "l2"
        # since GX is super sparse, we're applying a CG algorithm to the normal eqns
        spectrum = lsqr(sGX, momdep; damp=lam)
        if refine
            # this is a few steps of Newton iteration with the starting point of the linear solve
            DG = GX
            recalc_momentum = zeros(nz)
            tmp = zeros(2)
            maxit = 1
            for n_it = 1:maxit
                # evaluate around new solution and construct derivatives
                if !isnothing(propag_grad)
                    # use the efficient propagator+gradient calculator provided
                    for i = 1:nc
                        flag = false
                        if abs(spectrum[i]) > 1e-7
                            flag = true
                            waves[i].flux *= spectrum[i]
                        end
                        g,dg = propag_grad(waves[i], col)
                        if flag
                            # we need to reset the wave fluxes back to ref_flux
                            waves[i].flux /= spectrum[i]
                        else
                            g .*= spectrum[i]
                        end
                        recalc_momentum .+= g
                        DG[:,i] .= dg
                    end
                else
                    # Finite difference approach
                    epsi = 1e-8
                    for i = 1:nc
                        tmp .= waves[i].flux
                        waves[i].flux .= spectrum[i]*tmp
                        recalc_momentum .+= L81_wp(waves[i], col)
                        waves[i].flux .= (spectrum[i]+epsi)*tmp
                        DG[:,i] .= L81_wp(waves[i],col)
                        waves[i].flux .= (spectrum[i]-epsi)*tmp
                        DG[:,i] .-= L81_wp(waves[i],col)
                        DG[:,i] ./= 2epsi
                        waves[i].flux .= tmp
                    end
                end
            end
            sDG = sparse(DG)
            # Newton step
            #println(momdep.-recalc_momentum)
            spectrum .+= lsqr(sDG, momdep.-recalc_momentum)
        end
    elseif reg == "l1"
        
    else
        println("Unrecognized regularization keyword argument '$reg'")
    end
    
    return spectrum, sGX
end

function spectrum_to_momdep(propag::Function, 
        waves::Vector{WaveProfile}, 
        col::ColumnProfile)
    # A convenience wrapper for running a propagator on a vector of waves
    momdep = zeros(2*size(col.U,1))
    for wave in waves
        momdep .+= propag(wave, col)
    end
    return momdep        
end

function spectrum_to_momdep(propag::Function, 
        waves::Vector{WaveProfile}, 
        col::ColumnProfile,
        weights::Vector{Float64})
    # A convenience wrapper for running a propagator on a vector of waves
    # This one allows specifying a vector of weights to modulate the wavefluxes
    # Reason: this is optimized to not allocate an entire array yet avoid 
    # overwriting the original wave vector
    momdep = zeros(2*size(col.U,1))
    tmp_wave = WaveProfile()
    for idx in eachindex(waves)
        copy!(tmp_wave, waves[idx])
        tmp_wave.flux .*= weights[idx]
        momdep .+= propag(tmp_wave, col)
    end
    return momdep        
end
