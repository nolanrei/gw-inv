using LinearAlgebra
using SparseArrays
using StaticArrays
using IterativeSolvers
using Lasso

mutable struct WaveProfile
    k::Float64                  # x wavenumber
    l::Float64                  # y wavenumber
    m::Float64                  # z wavenumber
    om::Float64                 # omega; frequency
    c::MVector{3,Float64}       # phase velocity
    flux::Float64               # amount of momentum flux carried -- scalar
    src::Int64                  # index of vertical level at which waves are launched
end

struct ColumnProfile
    U::Array{Float64,1}         # zonal velocity
    V::Array{Float64,1}         # meridional velocity
    N::Array{Float64,1}         # buoyancy frequency
    rho::Array{Float64,1}       # density
    nlev::Int64                 # length of arrays
    dz::Array{Float64,1}        # z extent of each cell
end

mutable struct Spectrum
    waves::Vector{WaveProfile}
    amplitudes::Vector{Float64}
end

function Base.copy(wav::WaveProfile)
    # Copy to a new object
    return WaveProfile(wav.k, wav.l, wav.m, wav.om, copy(wav.c), wav.flux, wav.src)
end

function Base.copyto!(wav1::WaveProfile, wav2::WaveProfile)
    wav2.k = wav1.k
    wav2.l = wav1.l
    wav2.m = wav1.m
    wav2.om = wav1.om
    wav2.c .= wav1.c
    wav2.flux = wav1.flux
    wav2.src = wav1.src
    return nothing
end

function Base.copy(wav::ColumnProfile)
    # Copy to a new object
    return ColumnProfile(col.U, col.V, col.N, col.rho, col.nlev, col.dz)
end

function Base.copy!(wav1::WaveProfile, wav2::WaveProfile)
    # Copy wav2 to wav1
    wav1.k = wav2.k
    wav1.l = wav2.l
    wav1.m = wav2.m
    wav1.om = wav2.om
    wav1.c .= wav2.c
    wav1.flux = wav2.flux
    wav1.src = wav2.src
    return nothing
end

function c_to_wave(c::Real, dir::AbstractArray{Float64}, K::Float64, flux::Float64, src::Int64)
    # c is signed magnitude of phase speed
    # dir is the direction of (k,l) -- should be equal to direction of c,
    #      but can be nonzero while c is 0 (lee waves)
    # K^2 = k^2 + l^2 magnitude of wavenumber vector
    K2 = K*K
    sc = (c >= 0 ? 1.0 : -1.0)  # need to specify nonzero sign at c == 0

    k = K*sc*dir[1]
    l = K*sc*dir[2]

    wav = WaveProfile(k,l,0.0,0.0,MVector{3}([c*dir[1],c*dir[2],0.0]),flux,src)
    return wav
end

function prettify_wave!(wav::WaveProfile, usrc::Float64, N::Float64)
    # usrc is the meanflow velocity in direction dir at the wave source level
    # N is buoyancy frequency at source level
    # Neglecting f and 1/4H^2 in the dispersion relation
    K2 = wav.k^2 + wav.l^2
    K = sqrt(K2)
    c = hypot(wav.c[1],wav.c[2])
    m = N*K/abs(K*c - K*usrc)    # L81 WKB m evaluated at source level
    kl2 = K2+m*m
    
    omint = N*K/sqrt(kl2)    # intrinsic freq
    omabs = omint + K*usrc   # absolute frequency
    cz = omabs*m/kl2         # phase speed z
    wav.m = m
    wav.om = omabs
    wav.c[3] = cz
    return nothing
end

function make_waveset(wavespds::AbstractVector{Tv}, ndirs::Ti, srcs::AbstractVector{Ti}; 
        kw::Float64=2pi/5e4, ref_flux::Float64=1e-3, th0=0.0) where {Tv<:Real, Ti<:Integer}
    nk = length(wavespds)
    thetas = zeros(1, ndirs)
    for i = 1:ndirs
        thetas[i] = pi*(i-1)/ndirs + th0  # th0 is an optional offset for the angles
    end
    dirs = vcat(cos.(thetas), sin.(thetas))
    nsrc = length(srcs)

    nwav = nk*ndirs*nsrc
    waves = Array{WaveProfile,1}(undef, nwav)
    for k = 1:nsrc
        for j = 1:ndirs
            for i = 1:nk
                waves[i+nk*(j-1+ndirs*(k-1))] = c_to_wave(wavespds[i],dirs[:,j],kw,ref_flux,srcs[k])
            end
        end
    end

    return waves
end

function make_waveset(cmin::Tv, cmax::Tv, cstride::Tv, ndirs::Ti, nsrc::Ti, nlev::Ti) where {Tv<:Real,Ti<:Integer}
    wavespds = Array(cmin:cstride:cmax)
    srcs = [nmin+Int(floor((nmax-nmin) * i/nsrc)) for i = 0:nsrc-1]
    return make_waveset(wavespds, ndirs, nsrc, nlev, srcs)
end

function make_stochastic_waveset(wavespds::Vector{Tv}, ndirs::Ti, srcs::Vector{Tv}; 
        ref_flux::Float64=1e-3) where {Tv<:Real, Ti<:Integer}
    lam = 1e-4
    nk = length(wavespds)
    thetas = zeros(1, ndirs)
    for i = 1:ndirs
        thetas[i] = pi*(i-1)/ndirs
    end
    dirs = vcat(cos.(thetas), sin.(thetas))
    nsrc = length(srcs)
    wav = WaveProfile(2pi/1e5, 0.0, 0.0, 0.0, MVector(1.0, 0.0, 10.0), 1e-3, 1)

    nwav = nk*ndirs*nsrc
    waves = Array{WaveProfile,1}(undef, nwav)
    for k = 1:nsrc
        for j = 1:ndirs
            for i = 1:nk
                c = SVector(wavespds[i]*dirs[1,j], wavespds[i]*dirs[2,j], 10.0)
                flux = ref_flux
                waves[i+nk*(j-1+ndirs*(k-1))] = WaveProfile(2pi/1e5, 0.0, 0.0, 0.0, c, flux, srcs[k]);
            end
        end
    end

    return waves
end

function append_col_to_spmat!(I::Vector{Ti}, 
        J::Vector{Ti}, 
        V::Vector{Tv}, 
        colidx::Ti,
        numel::Ti,
        spcol::SparseVector{Tv,Ti}) where {Tv,Ti<:Integer}
    # Appends spcol to I,J,V used to construct a sparse matrix
    # I = row indices of nonzeros
    # J = col indices
    # V = values
    # colidx = column index of column being added
    # numel = number of elements already in I,J,V
    # spcol is a sparse vector
    # returns new number of nonzeros
    numel_new = length(spcol.nzind)    # number of elements in sparse column
    for i = 1:numel_new
        idx = numel + i
        I[idx] = spcol.nzind[i]
        J[idx] = colidx
        V[idx] = spcol.nzval[i]
    end
    return numel + numel_new
end

function append_col_to_spmat!(I::Vector{Ti}, 
        J::Vector{Ti}, 
        V::Vector{Tv}, 
        colidx::Ti,
        numel::Ti,
        col::Vector{Tv}) where {Tv,Ti<:Integer}
    # Appends col to I,J,V used to construct a sparse matrix
    # I = row indices of nonzeros
    # J = col indices
    # V = values
    # colidx = column index of column being added
    # numel = number of elements already in I,J,V
    # col is a dense vector
    # returns new number of nonzeros
    numel_new = numel    # number of elements in sparse column
    for i = 1:length(col)
        if col[i] == 0
            continue
        end
        numel_new += 1
        I[numel_new] = i
        J[numel_new] = colidx
        V[numel_new] = col[i]
    end
    return numel_new
end

function clear!(arr)
    for i in eachindex(arr)
        arr[i] = 0
    end
    return nothing
end        

function WaveProfile()
    return WaveProfile(0.0,0.0,0.0,0.0,SVector(0.0,0.0,0.0),0.0,0)
end

function set_flux!(wav::WaveProfile,new_flux)
    wav.flux .= new_flux
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

function L81_wp!(wav::WaveProfile, col::ColumnProfile, flux::Float64, momdep::AbstractArray{<:Number})
    # Compute gravity wave flux using the Lindzen '81 propagation scheme
    # "Lindzen '81 WavePacket formulation" to differentiate from wavespeed formulation below
    # Output is a vector of dimensions (2*nlev, 1) with [Dx; Dy]
    # Nonallocating version -- ADDS ON TO provided momdep vector

    #flux = wav.flux
    absk = hypot(wav.k,wav.l)
    c = norm(wav.c[1:2])                  # wavespeed in the horizontal
    cdir = [wav.k, wav.l]/absk            # direction of wave propagation
    vb = col.U[wav.src]*cdir[1] + col.V[wav.src]*cdir[2]        # mean flow speed in direction of wave at src
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/col.N[wav.src]
    # Loop from bottom to top of column, depositing momentum where it exceeds maximum stable transport
    for lvl = wav.src:col.nlev
        if flux == 0
            # if the wave has no momentum we don't need to keep going
            break
        end
        if lvl == col.nlev
            vt = vb
            bt = bb
        else
            vt = col.U[lvl+1]*cdir[1] + col.V[lvl+1]*cdir[2]       # flow speed in direction cdir at top of cell
            bt = col.rho[lvl+1] * 0.5*absk*(c-vt)^3/col.N[lvl+1]   # breaking condition -- maximum stable momentum transport
        end

        # If c-v at bottom and top of cell have different signs, there's a 
        # critical layer somewhere in here and we need to break and dump all momentum
    	if bb*bt <= 0
            bb = 0
            bt = 0   # cuts off any further deposition
        end

        # If |flux| > b, set flux to b and deposit extra at this level
        bb = sign(flux)*abs(bb)   # we need it to match the sign on flux
    	momsign = sign(c-vb)      # super important -- should break below crit layers and not above (vb vs vt)
        if abs(flux) > abs(bb)    # this should probably only work if flux is positive
            fb = flux-bb
            # we assume that momentum gets deposited in the direction of phase speed
            momdep[lvl] += momsign*fb*cdir[1]/col.rho[lvl]/col.dz[lvl]                    # zonal
            momdep[lvl+col.nlev] += momsign*fb*cdir[2]/col.rho[lvl]/col.dz[lvl]           # meridional (all the ys at the end)
            flux = bb
        end
    #    println(lvl,"\t",flux,"\t",bb,"\t",bt,"\t",momdep[lvl]) ###################### DEBUG MODE ############
        vb = vt
        bb = bt
    end

    return nothing
end

function L81!(wav::WaveProfile, col::ColumnProfile, flux_in, momdep)
    # Compute gravity wave flux using the Lindzen '81 propagation scheme
    # Nonallocating version -- ADDS ON TO provided momdep vector

    flux = flux_in    #### DIFFERENCE FROM L81_wp! -- allows accepting a flux value not part of a WaveProfile
    absk = hypot(wav.k,wav.l)
    c = norm(wav.c[1:2])                  # wavespeed in the horizontal
    cdir = [wav.k, wav.l]/absk            # direction of wave propagation
    vb = col.U[wav.src]*cdir[1] + col.V[wav.src]*cdir[2]        # mean flow speed in direction of wave at src
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/col.N[wav.src]
    # Loop from bottom to top of column, depositing momentum where it exceeds maximum stable transport
    for lvl = wav.src:col.nlev
        if flux == 0
            # if the wave has no momentum we don't need to keep going
            break
        end
        if lvl == col.nlev
            vt = vb
            bt = bb
        else
            vt = col.U[lvl+1]*cdir[1] + col.V[lvl+1]*cdir[2]       # flow speed in direction cdir at top of cell
            bt = col.rho[lvl+1] * 0.5*absk*(c-vt)^3/col.N[lvl+1]   # breaking condition -- maximum stable momentum transport
        end

        # If c-v at bottom and top of cell have different signs, there's a 
        # critical layer somewhere in here and we need to break and dump all momentum
    	if bb*bt <= 0
            bb = 0
            bt = 0   # cuts off any further deposition
        end

        # If |flux| > b, set flux to b and deposit extra at this level
        bb = sign(flux)*abs(bb)   # we need it to match the sign on flux
    	momsign = sign(c-vb)      # super important -- should break below crit layers and not above (vb vs vt)
        if abs(flux) > abs(bb)    # this should probably only work if flux is positive
            fb = flux-bb
            # we assume that momentum gets deposited in the direction of phase speed
            momdep[lvl] += momsign*fb*cdir[1]/col.rho[lvl]/dz[lvl]                    # zonal
            momdep[lvl+col.nlev] += momsign*fb*cdir[2]/col.rho[lvl]/dz[lvl]           # meridional (all the ys at the end)
            flux = bb
        end
    #    println(lvl,"\t",flux,"\t",bb,"\t",bt,"\t",momdep[lvl]) ###################### DEBUG MODE ############
        vb = vt
        bb = bt
    end

    return nothing
end

function L81_flux!(wav::WaveProfile, col::ColumnProfile, flux_in, flux_out)
    # Compute gravity wave flux using the Lindzen '81 propagation scheme
    # Nonallocating version -- ADDS ON TO provided momdep vector

    flux = flux_in    #### DIFFERENCE FROM L81_wp! -- allows accepting a flux value not part of a WaveProfile
    absk = hypot(wav.k,wav.l)
    c = norm(wav.c[1:2])                  # wavespeed in the horizontal
    cdir = [wav.k, wav.l]/absk            # direction of wave propagation
    vb = col.U[wav.src]*cdir[1] + col.V[wav.src]*cdir[2]        # mean flow speed in direction of wave at src
    momsign = sign(c-vb)         # flux has to be in direction c-v_src
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/col.N[wav.src]
    # Loop from bottom to top of column, depositing momentum where it exceeds maximum stable transport
    for lvl = wav.src:col.nlev
        if flux == 0
            # if the wave has no momentum we don't need to keep going
            break
        end
        if lvl == col.nlev
            vt = vb
            bt = bb
        else
            vt = col.U[lvl+1]*cdir[1] + col.V[lvl+1]*cdir[2]       # flow speed in direction cdir at top of cell
            bt = col.rho[lvl+1] * 0.5*absk*(c-vt)^3/col.N[lvl+1]   # breaking condition -- maximum stable momentum transport
        end
        
        # If c-v at bottom and top of cell have different signs, there's a 
        # critical layer somewhere in here and we need to break and dump all momentum
    	if bb*bt <= 0
            bb = 0
            bt = 0   # cuts off any further deposition
        end

        # If |flux| > b, set flux to b and deposit extra at this level
        bb = sign(flux)*abs(bb)   # we need it to match the sign on flux
        if abs(flux) > abs(bb)    # this should probably only work if flux is positive
            flux = bb
        end
        # flux is in direction of c^hat * sgn(c-v_src)
        flux_out[lvl] += momsign*flux*cdir[1]                    # zonal
        flux_out[lvl+col.nlev] += momsign*flux*cdir[2]           # meridional (all the ys at the end)
    #    println(lvl,"\t",flux,"\t",bb,"\t",bt,"\t",momdep[lvl]) ###################### DEBUG MODE ############
        vb = vt
        bb = bt
    end

    return nothing
end

function dupcatch!(propag!::Function, 
        col::ColumnProfile,
        waves::Vector{WaveProfile}, 
        eqclass::Vector{Integer}, 
        densecols::Matrix{Tr},
        I::Vector{Ti}, 
        J::Vector{Ti}, 
        V::Vector{Tr}, 
        startcol::Integer) where {Tr,Ti<:Integer}
    # When trying to save memory and matsolve time by catching duplicate columns,
    # this function replaces the inner loop which populates GX
    # What this is doing:
    # Usually (though not always), I'm selecting a set of waves to launch at different
    # heights in the column. Many of these break in exactly the same places as each other,
    # due to the geometry of the velocity profile. This function tries to catch those
    # duplicates; that is, where wav(src=z1) breaks in exactly the same places as wav(src=z2).
    #   Functionally, it does the same as the GX-populating loop, except that
    # it groups waves in equivalence classes "eqclass" -- usually, this eqclass will be
    # "everything the same but the source level"
    ncols = length(eqclass)
    nz = 2*col.nlev
    clear!(densecols)
    for i = 1:ncols
        propag!(waves[eqclass[i]], col, densecol)
        # add to the set
        nz_true = append_col_to_spmat!(I,J,V,col_idx,nz_true,densecol)
    end
end

function L81_wp(wav::WaveProfile, col::ColumnProfile)
    # Wrapper for L81_wp! that creates a new array
    # Outputs a vector of dimensions (2*nlev, 1) with [Dx; Dy]
    momdep = zeros(2*col.nlev)               # array for momentum deposited by wave (solution array)
    L81_wp!(wav,col,momdep)
    return momdep
end

function miss_check(waves,col)
    ## labour-saving device
    
    # find max and min mean flow speed in each direction, using a dict vbounds[dir] = (minv,maxv)
    vbounds = Dict()
    for w in waves
        c = norm(w.c[1:2])
        dir = (c > 0 ? w.c[1:2]/c : [1.0 0.0])
        vbounds[dir] = [0.0,0.0]    # initialization
    end
    for dir in keys(vbounds)
        vmin = 1e8       # minimum speed of mean flow along this direction
        vmax = -1e8      # maximum speed -- waves with c > this we know will not deposit momentum
        for z = 1:div(nz,2)
            vtmp = dir[1]*col.U[z] + dir[2]*col.V[z]
            if vtmp > vmax
                vmax = vtmp
            end
            if vtmp < vmin
                vmin = vtmp
            end
            vbounds[dir] .= [vmin,vmax]
        end
    end
    # now, check each wave to see if it will miss the mean flow
    dep = [false for i = 1:nw]
    for i = 1:nw
        c = norm(waves[i].c[1:2])
        dir = (c > 0 ? waves[i].c[1:2]/c : [1.0 0.0])
        vmin,vmax = vbounds[dir]
        if c > vmax || c < vmin
            continue
        end
        dep[i] = true
    end
    nc = sum(dep .== true)    # we only want the waves that break somewhere
    used = zeros(Int64,nc)    # indices in the original array of the waves we're using
    ctr = 1
    for i = 1:nw
        if dep[i] == true
            used[ctr] = i
            ctr += 1
        end
    end
    return used
end

function lin_invert(propag!::Function, 
        waves::Vector{WaveProfile},
        momdep::Vector{Float64}, 
        col::ColumnProfile; 
        lam::Float64=1e-4, reg="l2", 
        wts=ones(length(momdep)), refine::Int64=1,
        used=nothing, sparsity_factor=0.2)
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

    nz = size(momdep,1)         # size of propag output vectors
    nw = length(waves)          # number of waves we're considering

    if isnothing(used)
        # used is a set of indices for the subset of waves that we're really using
        # the most likely source of this set of indices is used = miss_check(wav,col)
        # if it's unset, we want to use the whole set of waves
        nc = nw
        used = 1:nw
    else
        nc = length(used)
        sparsity_factor = 0.35   # since we're precutting some of the zeros the matrix will be denser
    end

    # if momdep == 0, should just output 0 instead of running through these delicate internals
    # necessary for testing with single waves (L1 reg usually breaks with 0 momdep)
    if all(abs.(momdep) .<= 1e-11)
        return zeros(nw)
    end

    # construct the matrix
    nnz = Int(ceil(sparsity_factor*nz*nc))  # bound on Number of NonZeros in GX -- sparsity factor * total numel
    
    densecol = zeros(nz)
    J = zeros(Int64,nnz)        # column indices of nonzeros
    I = zeros(Int64,nnz)         # row indices of nonzeros
    V = zeros(Float64,nnz)       # values of nonzeros
    
    nz_true = 0
    for i = 1:nc
        # used[i] is the index of the ith wave that wouldn't have been 0
        idx = used[i]
        clear!(densecol)
        propag!(waves[idx], col, densecol)
        nz_true = append_col_to_spmat!(I,J,V,i,nz_true,densecol)
    end
    sGX = sparse(I[1:nz_true],J[1:nz_true],V[1:nz_true],nz,nc)

    if reg == "l2"
        spectrum = lsqr(sGX, momdep; damp=lam)
        if refine > 0
            # this is a few steps of Newton iteration with the starting point of the linear solve
            recalc_momentum = zeros(nz)
            tmp = zeros(2)
            for n_it = 1:refine
                # evaluate around new solution and construct derivatives
                # Finite difference approach
                clear!(recalc_momentum)
                epsi = 1e-8
                nz_true = 0
                for i = 1:nc
                    idx = used[i]
                    tmp = waves[idx].flux
                    waves[idx].flux = spectrum[i]*tmp
                    propag!(waves[idx], col, recalc_momentum) # not clearing beforehand because we want to add
                    waves[idx].flux = (spectrum[i]+epsi)*tmp
                    clear!(densecol)
                    propag!(waves[idx],col,densecol)
                    waves[idx].flux = (-spectrum[i]+epsi)*tmp       # propag! only adds, so to subtract we need to propagate negative flux
                    propag!(waves[idx],col,densecol)
                    densecol ./= 2epsi
                    waves[idx].flux = tmp
                    # append to sparse matrix
                    nz_true = append_col_to_spmat!(I,J,V,i,nz_true,densecol)
                end
            end
            sDG = sparse(I[1:nz_true],J[1:nz_true],V[1:nz_true],nz,nc)
            # Newton step
            spectrum .+= lsqr(sDG, momdep.-recalc_momentum)
        end
    elseif reg == "l1"
        α = 0.9
        # needs some weighting -- I've been using wts ~ (rho)^1/4
        x = fit(LassoPath,sGX,momdep,Normal(),IdentityLink(); α,wts,λminratio=1e-2,intercept=false,standardize=false,maxncoef=size(sGX,2))
        spectrum=x.coefs[:,end-1]
    else
        println("Unrecognized regularization keyword argument '$reg'")
    end
    full_spectrum = zeros(length(waves))
    for i = 1:nc
        full_spectrum[used[i]] = spectrum[i]
    end
    return full_spectrum
end

function spectrum_to_momdep!(propag!::Function, 
        spectrum::Spectrum, 
        col::ColumnProfile,
        momdep::AbstractVector{Float64})
    # A convenience wrapper for running a propagator on a vector of waves
    # This one allows specifying a vector of weights to modulate the wavefluxes
    # Reason: this is optimized to not allocate an entire array yet avoid 
    # overwriting the original wave vector
    tmp_wave = WaveProfile()
    for idx in eachindex(spectrum.waves)
        copy!(tmp_wave, spectrum.waves[idx])
        tmp_wave.flux *= spectrum.amplitudes[idx]
        propag!(tmp_wave, col, momdep)
    end
    return nothing        
end

function spectrum_to_momdep(propag!::Function, 
        waves::AbstractArray{WaveProfile}, 
        col::ColumnProfile, 
        fluxes)
    nz = length(col.U)
    momdep = zeros(2nz)
    for i in eachindex(waves)
        propag!(waves[i],col,fluxes[i],momdep)
    end
    return momdep
end

function spectrum_to_momdep!(propag!::Function, 
        waves::AbstractArray{WaveProfile}, 
        col::ColumnProfile, 
        fluxes,
        momdep
    )
    for i in eachindex(waves)
        propag!(waves[i],col,fluxes[i],momdep)
    end
    return nothing
end