include("L81-2.jl")

function L81_grad_soft!(wav::WaveProfile, col::ColumnProfile, flux_in, flux_out, grad)
    # Computes L81(flux*wave) and d/dflux L81(flux*wave)
    
    flux = flux_in    #### DIFFERENCE FROM L81_wp! -- allows accepting a flux value not part of WaveProfile

    th = 0.2          ### Lindzen hard flux cutoff is replaced by geometric decay to stability criterion, 
                      ### flux = (1-th)*b + th*flux

    nz = length(col.U)
    
    absk = hypot(wav.k,wav.l)
    c = norm(wav.c[1:2])                  # wavespeed in the horizontal
    cdir = [wav.k, wav.l]/absk            # direction of wave propagation
    vb = col.U[wav.src]*cdir[1] + col.V[wav.src]*cdir[2]        # mean flow speed in direction of wave at src
    momsign = sign(c-vb)         # flux has to be in direction c-v_src
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/col.N[wav.src]

    # tracks decay of flux_in's influence on flux_out
    g = 1.0
    
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
            flux = (1-th)*bb + th*flux
            g *= th
        end
        # we assume that momentum gets deposited in the direction of phase speed
        flux_out[lvl] += momsign*flux*cdir[1]                   # zonal
        flux_out[lvl+col.nlev] += momsign*flux*cdir[2]          # meridional (all the ys at the end)
        # as flux approaches b, g --> 0
        grad[lvl] += g*momsign*cdir[1]                          # zonal
        grad[lvl+col.nlev] += g*momsign*cdir[2]
        #println(lvl,"\t",flux,"\t",bb,"\t",bt,"\t",flux_out[lvl]) ###################### DEBUG MODE ############
        vb = vt
        bb = bt
    end

    return nothing
end

function L81_grad!(wav::WaveProfile, col::ColumnProfile, flux_in, flux_out, grad)
    # Computes L81(flux*wave) and d/dflux L81(flux*wave)
    
    flux = flux_in    #### DIFFERENCE FROM L81_wp! -- allows accepting a flux value not part of WaveProfile

    nz = length(col.U)
    
    absk = hypot(wav.k,wav.l)
    c = norm(wav.c[1:2])                  # wavespeed in the horizontal
    cdir = [wav.k, wav.l]/absk            # direction of wave propagation
    vb = col.U[wav.src]*cdir[1] + col.V[wav.src]*cdir[2]        # mean flow speed in direction of wave at src
    momsign = sign(c-vb)         # flux has to be in direction c-v_src
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/col.N[wav.src]

    # df/da is only nonzero before flux has been deposited
    has_deposited = false
    
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
            if has_deposited == false
                has_deposited = true
            end
            flux = bb
            # we assume that momentum gets deposited in the direction of phase speed
            flux_out[lvl] += momsign*flux*cdir[1]                   # zonal
            flux_out[lvl+col.nlev] += momsign*flux*cdir[2]          # meridional (all the ys at the end)
        elseif !has_deposited
            # f = min(min(a(z) for z<z_current), b(z_current))
            # So df/da is only nonzero before a(z) has decreased at all
            grad[lvl] += momsign*cdir[1]                   # zonal
            grad[lvl+col.nlev] += momsign*cdir[2]
        end
        #println(lvl,"\t",flux,"\t",bb,"\t",bt,"\t",flux_out[lvl]) ###################### DEBUG MODE ############
        vb = vt
        bb = bt
    end

    return nothing
end

function L81_exp_grad!(wav::WaveProfile, col::ColumnProfile, alpha, flux_out, grad)
    # Computes L81(flux*wave) and d/dflux L81(flux*wave)
    # where flux is defined as exp(alpha)
    
    flux = exp(alpha)    ####  DIFFERENCE -- exponentiating
    f0 = exp(alpha)      # this one is for the derivative
    
    th = 0.0          ### Lindzen hard flux cutoff is replaced by geometric decay to stability criterion, 
                      ### flux = (1-th)*b + th*flux

    # further softening the discontinuities in the gradient by replacing
    # if statements with sigmoids
    ksig = 20   # sigmoid sensitivity
    tiny = 1e-15  # for scaling

    nz = length(col.U)
    
    absk = hypot(wav.k,wav.l)
    c = norm(wav.c[1:2])                  # wavespeed in the horizontal
    cdir = [wav.k, wav.l]/absk            # direction of wave propagation
    vb = col.U[wav.src]*cdir[1] + col.V[wav.src]*cdir[2]        # mean flow speed in direction of wave at src
    momsign = sign(c-vb)         # flux has to be in direction c-v_src
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/col.N[wav.src]

    # tracks decay of flux_in's influence on flux_out
    g = 1.0
    
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
        ## if statement now dubious, perhaps, when we're trying to soften the creases
    	if bb*bt <= 0
            bb = 0.0
            bt = 0.0   # cuts off any further deposition
        end
        
        # If |flux| > b, set flux to b and deposit extra at this level
        ######################
        # There is some ambiguity in what the output flux should represent
        # In particular, is flux[lvl] the flux at the bottom or top of the cell?
        # This code chooses flux[lvl] = top-of-cell flux
        ######################
        sgnfl = sign(flux)
        bb = sgnfl*abs(bb)        # we need it to match the sign on flux

        # sigmoid replaces "if abs(flux) > abs(bb)"
        diff = ( abs(flux) - abs(bb) )/(abs(bb) + tiny)
        s = 0.5*(1 + tanh(ksig * diff))  # if flux > bb, s will go to 1
        th_eff = (1-s) + s*th     # for flux << bb, th_eff = 1; for flux >> bb, th_eff = th
        fi = flux
        flux = (1-th_eff)*bb + th_eff*flux
        #println(i,"\t",fi,"\t",bb,"\t",s,"\t",th_eff,"\t",flux)

        dsdf = 0.5*ksig*(sech(ksig*diff))^2/(abs(bb) + tiny) * g * sgnfl
        g = th_eff*g + (th-1)*dsdf*(fi-bb)     
        
        # we assume that momentum gets deposited in the direction of phase speed
        flux_out[lvl] += momsign*flux*cdir[1]                   # zonal
        flux_out[lvl+col.nlev] += momsign*flux*cdir[2]          # meridional (all the ys at the end)
        # as flux approaches b, g --> 0
        grad[lvl] += g*momsign*cdir[1]*f0        ### DIFFERENCE -- chain rule gets us an extra factor of flux
        grad[lvl+col.nlev] += g*momsign*cdir[2]*f0
        
        vb = vt
        bb = bt
    end
end

function L81_exp_hess!(wav::WaveProfile, col::ColumnProfile, alpha, d2G_dxi2)
    # Computes d^2/dalpha^2 L81(flux*wave)
    # where flux is defined as exp(alpha)
    
    flux = exp(alpha)    
    f0 = exp(alpha)      
    
    th = 0.0             
    ksig = 20   
    tiny = 1e-15 

    nz = length(col.U)
    
    absk = hypot(wav.k, wav.l)
    c = norm(wav.c[1:2])                                
    cdir = [wav.k, wav.l] / absk                        
    vb = col.U[wav.src]*cdir[1] + col.V[wav.src]*cdir[2]
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/col.N[wav.src]

    # momsign shouldn't change as long as wave doesn't cross a critical layer
    momsign = sign(c-vb)

    # g tracks d(flux)/df0
    g = 1.0
    # h tracks d^2(flux)/df0^2
    h = 0.0 
    
    for lvl = wav.src:col.nlev
        if flux == 0
            break
        end
        if lvl == col.nlev
            vt = vb
            bt = bb
        else
            vt = col.U[lvl+1]*cdir[1] + col.V[lvl+1]*cdir[2]       
            bt = col.rho[lvl+1] * 0.5*absk*(c-vt)^3/col.N[lvl+1]   
        end

        if bb*bt <= 0
            bb = 0.0
            bt = 0.0   
        end
        
        sgnfl = sign(flux)
        bb = sgnfl * abs(bb)            

        diff = (abs(flux) - abs(bb)) / (abs(bb) + tiny)
        
        # Precompute transcendentals to save CPU time
        tanh_kd = tanh(ksig * diff)
        sech2_kd = sech(ksig * diff)^2
        
        s = 0.5 * (1 + tanh_kd)  
        th_eff = (1 - s) + s * th      
        fi = flux
        flux = (1 - th_eff) * bb + th_eff * flux

        # First and second derivatives of the Sigmoid with respect to local flux
        S_prime = 0.5 * ksig * sech2_kd / (abs(bb) + tiny) * sgnfl
        S_prime_prime = - (ksig^2) * sech2_kd * tanh_kd / (abs(bb) + tiny)^2
        
        # Chain rule derivatives with respect to source flux (F0)
        dsdf = S_prime * g
        d2sdf2 = S_prime_prime * (g^2) + S_prime * h
        
        # 1. Update h BEFORE overwriting g
        h = th_eff * h + 2 * (th - 1) * g * dsdf + (th - 1) * (fi - bb) * d2sdf2
        
        # 2. Update g
        g = th_eff * g + (th - 1) * dsdf * (fi - bb)      
        
        # Deposit the exact second derivative with respect to alpha
        # d2M/dalpha2 = momsign * cdir * (h * e^(2a) + g * e^a)
        val = momsign * (h * f0^2 + g * f0)
        
        d2G_dxi2[lvl] += val * cdir[1]                   
        d2G_dxi2[lvl+col.nlev] += val * cdir[2]          
        
        vb = vt
        bb = bt
    end
end