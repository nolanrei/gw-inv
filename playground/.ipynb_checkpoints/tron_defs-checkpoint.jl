using NLPModels
using JSOSolvers
using LinearAlgebra: dot

mutable struct GWFit{T, S} <: AbstractNLPModel{T, S}
    meta :: NLPModelMeta{T, S}
    counters :: Counters

    propag! :: Function
    propag_2der! :: Function
    waves :: AbstractVector{WaveProfile}
    col :: ColumnProfile
    b :: AbstractVector{T}
    lambda :: T
    Gi :: AbstractVector{T}
    J :: AbstractArray{T}
    Jv :: AbstractVector{T}
    der2 :: AbstractArray{T}
    wts :: AbstractVector{T}
    wndw :: Union{AbstractVector{T}, Nothing}
end

function GWFit(waves, col, fluxvec; lambda=1e-5, wndw=nothing)
    nz = length(col.U)
    nw = length(waves)
    
    J = zeros(2nz, nw)
    Jv = zeros(2nz)
    der2 = zeros(2nz, nw)
    gi = zeros(2nz)

    x0 = 1e-7 .+ 1e-6 .* rand(nw)   # starting condition

    # makes high-speed waves more expensive in the optimization
    wts = zeros(nw)
    for i = 1:nw
        wts[i] = 1.0 + 0.0002 * sum(waves[i].c.^2)
    end

    # Positivity constraints
    lvar = zeros(nw)
    uvar = fill(Inf, nw)

    # Number of Non-Zeros in the lower triangle of the Hessian
    nnzh = div(nw * (nw + 1), 2)

    meta = NLPModelMeta(
        nw,
        x0 = x0,
        lvar = lvar,
        uvar = uvar,
        nnzh = nnzh,
        name = "Gravity Wave Inverse Propagation"
    )
    
    return GWFit(
        meta, Counters(), 
        L81_grad!, L81_hess!, # Assumes these are defined elsewhere
        waves, col, fluxvec, lambda, 
        gi, J, Jv, der2, wts, wndw
    )
end

# ---------------------------------------------------------
# Required NLPModels Interface Methods
# ---------------------------------------------------------
import NLPModels: obj, grad!, hess_structure!, hess_coord!

function obj(nlp::GWFit, x::AbstractVector{Float64})
    nlp.Gi .= 0.0
    nlp.J .= 0.0
    for i in eachindex(nlp.waves)
        nlp.propag!(nlp.waves[i], nlp.col, x[i], nlp.Gi, view(nlp.J, :, i))
    end
    
    nlp.Gi .-= nlp.b
    
    if !isnothing(nlp.wndw)
        nlp.Gi .*= nlp.wndw
    end
    
    return 0.5 * dot(nlp.Gi, nlp.Gi) + nlp.lambda * sum(nlp.wts .* x)
end

function grad!(nlp::GWFit, 
        x::AbstractVector{Float64}, 
        G::AbstractVector{Float64})
    nlp.Gi .= 0.0
    nlp.J .= 0.0
    
    for i in eachindex(nlp.waves)
        nlp.propag!(nlp.waves[i], nlp.col, x[i], nlp.Gi, view(nlp.J, :, i))
    end
    
    nlp.Gi .-= nlp.b
    
    if !isnothing(nlp.wndw)
        nlp.Gi .*= nlp.wndw
        nlp.J .*= nlp.wndw
    end

    # J' * Gi + lambda * wts
    mul!(G, nlp.J', nlp.Gi) 
    G .+= nlp.lambda .* nlp.wts
    
    return G
end

function hess_structure!(nlp::GWFit, 
        rows::AbstractVector{<:Integer}, 
        cols::AbstractVector{<:Integer})
    nw = length(nlp.meta.x0)
    idx = 1
    for j in 1:nw
        for i in j:nw # STRICTLY LOWER TRIANGLE
            rows[idx] = i
            cols[idx] = j
            idx += 1
        end
    end
    return rows, cols
end

function NLPModels.hess_coord!(nlp::GWFit, 
        x::AbstractVector{Float64}, 
        y::AbstractVector{Float64}, 
        hvals::AbstractVector{Float64}; 
        obj_weight::Float64=1.0)

    # y is Lagrange multipliers for bounds, 
    # but the second derivatives of my bounds are 0
    # so that term disappears
    nlp.Gi .= 0.0
    nlp.J .= 0.0
    nlp.der2 .= 0.0
    nw = length(nlp.waves)
    
    for i = 1:nw
        nlp.propag!(nlp.waves[i], nlp.col, x[i], nlp.Gi, view(nlp.J, :, i))
        nlp.propag_2der!(nlp.waves[i], nlp.col, x[i], view(nlp.der2, :, i))
    end
    
    nlp.Gi .-= nlp.b
    
    if !isnothing(nlp.wndw)
        nlp.Gi .*= nlp.wndw
        for i = 1:nw
            nlp.J[:, i] .*= nlp.wndw
            nlp.der2[:, i] .*= nlp.wndw
        end
    end

    # Extract exact Hessian values straight into the 1D hvals vector
    idx = 1
    for j = 1:nw
        for i = j:nw # strictly lower triangle
            # Gauss-Newton term: (J^T J)_ij
            val = dot(view(nlp.J, :, i), view(nlp.J, :, j))
            
            # Second derivative correction (only applies to diagonal for NLLS)
            if i == j
                val += dot(nlp.Gi, view(nlp.der2, :, i))
            end
            
            # Apply obj_weight (required by solver API)
            hvals[idx] = val * obj_weight
            idx += 1
        end
    end
    
    return hvals
end

function NLPModels.hprod!(nlp::GWFit, 
        x::AbstractVector{Float64}, 
        y::AbstractVector{Float64}, 
        v::AbstractVector{Float64}, 
        Hv::AbstractVector{Float64}; 
        obj_weight::Float64=1.0)
    
    # y is Lagrange multipliers for bounds, 
    # but the second derivatives of my bounds are 0
    # so that term disappears
    
    # 1. Evaluate the system at x
    nlp.Gi .= 0.0
    nlp.J .= 0.0
    nlp.der2 .= 0.0
    nw = length(nlp.waves)
    
    for i = 1:nw
        nlp.propag!(nlp.waves[i], nlp.col, x[i], nlp.Gi, view(nlp.J, :, i))
        nlp.propag_2der!(nlp.waves[i], nlp.col, x[i], view(nlp.der2, :, i))
    end
    
    nlp.Gi .-= nlp.b
    
    if !isnothing(nlp.wndw)
        nlp.Gi .*= nlp.wndw
        for i = 1:nw
            nlp.J[:, i] .*= nlp.wndw
            nlp.der2[:, i] .*= nlp.wndw
        end
    end

    # 2. Compute the Hessian-vector product: Hv = J^T * (J * v) + D * v
    
    # Step A: Compute J * v
    mul!(nlp.Jv, nlp.J, v)
    
    # Step B: Compute J^T * (Jv) and store it directly in Hv
    mul!(Hv, nlp.J', nlp.Jv)
    
    # Step C: Add the diagonal second-derivative correction (D * v)
    for i = 1:nw
        correction = dot(nlp.Gi, view(nlp.der2, :, i))
        Hv[i] += correction * v[i]
    end
    
    # 3. Apply the solver's objective weight
    Hv .*= obj_weight
    
    return Hv
end

function L81_grad!(wav::WaveProfile, col::ColumnProfile, alpha, flux_out, grad)
    # Computes L81(flux*wave) and d/dflux L81(flux*wave)
    # NON-exponentiating version (flux = alpha)
    
    flux = alpha
    
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
        grad[lvl] += g*momsign*cdir[1]
        grad[lvl+col.nlev] += g*momsign*cdir[2]
        
        vb = vt
        bb = bt
    end
end

function L81_hess!(wav::WaveProfile, col::ColumnProfile, alpha, d2G_dxi2)
    # Computes d^2/dalpha^2 L81(flux*wave)
    # this is the NON-exponentiating version (flux = alpha)
    
    flux = alpha    
    
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

        # First and second derivatives of the sigmoid with respect to local flux
        S_prime = 0.5 * ksig * sech2_kd / (abs(bb) + tiny) * sgnfl
        S_prime_prime = - (ksig^2) * sech2_kd * tanh_kd / (abs(bb) + tiny)^2
        
        # Chain rule derivatives with respect to source flux (F0)
        dsdf = S_prime * g
        d2sdf2 = S_prime_prime * (g^2) + S_prime * h
        
        # 1. Update h before overwriting g
        h = th_eff * h + 2 * (th - 1) * g * dsdf + (th - 1) * (fi - bb) * d2sdf2
        
        # 2. Update g
        g = th_eff * g + (th - 1) * dsdf * (fi - bb)      
        
        # Deposit the exact second derivative with respect to alpha
        # d2M/dalpha2 = momsign * cdir * h
        val = momsign * h
        
        d2G_dxi2[lvl] += val * cdir[1]                   
        d2G_dxi2[lvl+col.nlev] += val * cdir[2]          
        
        vb = vt
        bb = bt
    end
end

