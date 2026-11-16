using FFTW
include("L81-2.jl")
include("inversion.jl")

function angle_no_brcuts!(th, x, y)
    # calculates angle of vector with no branch cuts
    # want to see angle without the branch cuts by allowing 
    # angle to be outside (-pi,pi)
    n = length(th)
    thprev = 0.0
    for i = 1:n
        th[i] = atan(y[i], x[i])
        delta = th[i]-thprev
        if abs(delta) > pi   
            # we assume a branch cut happened
            winding_num = round(delta/(2pi))
            th[i] -= 2*pi*winding_num
        end
        thprev = th[i]
    end
end

function div_by_angle(fluxvec)
    # returns breakpoints that divide fluxvec into 
    # very roughly constant-direction pieces
    nz = div(length(fluxvec),2)
    th = zeros(nz)

    angle_no_brcuts!(th,view(fluxvec,1:nz),view(fluxvec,nz+1:2nz))

    ## Smooth it to isolate large-scale trends
    thk = zeros(ComplexF64,nz)  # FFT coefficients for convolution
    smk = zeros(ComplexF64,nz)  # coeffs of the smoothing operator
    s = 3   # characteristic length scale I want to smooth over, in GRID POINTS
    ffi = fftshift(-div(nz,2):div(nz+1,2)-1)
    for i = 1:nz
        smk[i] = exp(-(ffi[i]/s)^2)
    end
    fft!(smk)
    smk ./= smk[1]
    
    thk .= th
    fft!(thk)
    thk .*= smk
    ifft!(thk)

    # Break down the smoothed curve into segments increasing and decreasing
    # then for each one pick a dividing line somewhere along it
    clumps = Vector{Vector{Int64}}(undef,0)
    sign_dth = 0  
    for i = 1:nz-1
        #println(real.(thk[i+1]-thk[i]), "\t",sign_dth)
        sgn_i = sign(real.(thk[i+1]-thk[i]))
        if sgn_i == sign_dth
            # add new index to active clump
            push!(clumps[end],i)
        else
            # make a new clump with the new index
            push!(clumps,[i])
            sign_dth = sgn_i
        end
    end
    # the above misses the last breakpoint (almost always going to be nz)
    push!(clumps,[nz])

    abridged_brkpts = zeros(Int64,length(clumps))
    for i = 1:length(clumps)
        lc = length(clumps[i])
        # special logic: we want to keep 1, and round the rest of the clump down to 1
        if clumps[i][1] == 1
            abridged_brkpts[i] = 1
        else
            idx = max(1, Int(round(2*lc/3)) ) # make sure it doesn't index 1-long clumps at idx 2/3 = 0
            abridged_brkpts[i] = clumps[i][idx]  # replace all these indices with one 2/3 of the way along the clump
        end
    end
    return th, abridged_brkpts, real.(thk)
end

function find_angle(fluxvec, th, brk)
    # Important -- assumes that fluxvec is rho u'w' and rho v'w'
    nz = div(length(fluxvec),2)
    
    angles = []
    flx = fluxvec .+ 0.0  # copy of fluxvec that we can modify
    th0 = th .+ 0.0  # copy of th that we can modify
    
    nb = length(brk)
    for i = nb:-1:2        
        nj = brk[i]-brk[i-1]  # number of points in this segment
        
        # estimate the angle and magnitude of the wave in this segment
        angle_no_brcuts!( view(th0,brk[i-1]:brk[i]), 
            view(flx,brk[i-1]:brk[i]), 
            view(flx,(brk[i-1]+nz):(brk[i]+nz)) )
         
        angle = mean( th0[brk[i-1]:(brk[i-1]+div(3*nj,4))] ) 
        
        # actually estimate flux from first half, since flux can get deposited
        # as you go up
        flux = 0.0
        nj2 = div(nj+1,2)
        for j = brk[i-1]:(brk[i-1]+nj2)
            flux += hypot(flx[j],flx[j+nz])
        end
        flux /= nj2  # mean
        
        # and now that we know the strength of the wave, subtract this
        # flux from the column below to get the angle there unpolluted
        # by flux that's just passing through
        #fx = flux*cos(angle)
        #fy = flux*sin(angle)
        #for j = 1:brk[i-1]
        #    flx[j] -= fx
        #    flx[j+nz] -= fy
        #end
        
        push!( angles, (brk[i-1], brk[i], angle, flux) )
    end
    return angles[end:-1:1], flx, th0
end

function rot_mat(th)
    c = cos(th)
    s = sin(th)
    return [c -s; s c]
end

function fit_subset!(x, fluxvec, col, angle, window, mem, p)
    # fit a part of the profile that we believe is 
    # dominated by one wave or set of waves
    # x is the output vector of amplitudes
    # fluxvec is the observed flux
    # col is the column profile
    # angle is the mean angle of the flux in the chunk
    # window is the range of indices we're looking at
    # mem is some dedicated available memory; (dx, xnew, df, H)
    # p -- (L81_exp_grad!,waves,col,fluxvec,lambda,BIG,gi,J,wts,wndw,wndw_flag)

    # We need to reinitialize waves to refer to waves at the right angle
    # -- want x,y axes tilted by "angle"
    waves = p[2]
    nw = length(waves)
    # unsafe approach -- simply rotates everything by angle,
    # so if your starting point isn't 0 you're off
    R = rot_mat(angle)
    for i=1:nw
        waves[i].c .= R*waves[i].c
    end

    # Windowing -- need to multiply fluxvec and spectrum_to_momdep(x)
    # by 0 outside the area we're interested in, or optimizer will try
    # to fit stuff we can't see
    wlo,whi = window
    p[11] = true   # yes we are windowing
    p[10] .= 0.0
    p[10][wlo:whi] .= 1.0
    p[10][(wlo+nz):(whi+nz)] .= 1.0
    p[4] .= fluxvec
    niter = wave_invert_mid!(x, p, mem; max_newton_steps=100, 
        error_on_max_steps=false, armijo_max_steps_no_error=true) # safeties off :(

    # Compute flux carried by these waves lower in the column
    # and subtract it so we get a more accurate reading on 
    # the flux carried by waves that break lower
    p[7] .= spectrum_to_momdep(L81_exp_flux!,waves,col,x)
    fluxvec[1:wlo-1] .-= p[7][1:wlo-1]
    ##TODO: unsupervised -- if returns flux that is nonphysical below 
    # window, then could ruin reconstructions below

    # Rotate the waves back so the next section isn't messed up
    # R is orthogonal so rot(-angle) = rot(angle)'
    for i=1:nw
        waves[i].c .= R'*waves[i].c
    end
    return nothing
end

function divide_et_impera(col::ColumnProfile, fluxvec::AbstractVector{Float64},
    waves::AbstractVector{WaveProfile}, z::AbstractVector{Float64}, mem, p)
    # Runner function for inverting propagation of gravity waves
    # via estimating that we have relatively few waves breaking at once
    nw = length(waves)

    # Divide and invert from the middle down, then top to middle
    # I do not trust the fluxes at the top and don't want to
    # propagate that down
    thvec,brkpts = div_by_angle(fluxvec)
    angles = find_angle(fluxvec,thvec,brkpts)

    nth = length(angles)
    xs = zeros(nw,nth)
    window = zeros(2nz)

    # Start closest to 40km and work down
    first = bsrch(40,z[breakpoints])
    for i = first:-1:2
        window .= 0.0
        window[angles[i-1]:angles[i]] .= 1.0
        fit_subset!(xs[i], fluxvec, col, angles[i], window, mem, p)
    end
end