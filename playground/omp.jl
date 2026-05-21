# Despite the name, it implements an algorithm somewhat closer to
# Sliding Frank-Wolfe by Denoyelle et al. 2019

using Optim
using ForwardDiff
using LinearAlgebra
using PreallocationTools
using ADTypes
using Sobol
using LineSearches
using LeastSquaresOptim

include(expanduser("~/work/gw-inv/L81-2.jl"))

# T is the element type of the residual, F is the function, C is the Cache type
mutable struct OMPStruct{T, F1, F2, C}
    col         :: ColumnProfile
    wav         :: WaveProfile
    prop_AD!    :: F1
    cdf         :: F2
    fluxvec     :: Vector{T}
    
    res         :: C
    res_cache   :: C
    tmp_cache   :: C        # holds a DiffCache instead of a Vector
    tmp_cache2  :: C

    backup_buf  :: Vector{T}
    scaling_buf :: Vector{T}
    lo_buf      :: Vector{T}
    up_buf      :: Vector{T}
    maxwaves    :: Int64
end

function OMPStruct(col, prop_AD!::F1, my_cdf::F2, wav; max_nw=200, fluxvec=nothing) where {F1, F2}
    nz = length(col.U)

    if isnothing(fluxvec)
        fluxvec = zeros(2nz)
    end
    
    # Initialize the DiffCache with a base Float64 array of the correct size.
    # Automatically builds the Dual array behind the scenes.
    res = DiffCache(zeros(2nz))
    cache = DiffCache(zeros(2nz))
    res_cache = DiffCache(zeros(2nz))   # special one just for get_res!
    cache2 = DiffCache(zeros(3max_nw))

    backup_buf = zeros(3max_nw)
    scaling_buf = zeros(3max_nw)
    lo_buf = zeros(3max_nw)
    up_buf = zeros(3max_nw)
    
    return OMPStruct{Float64, F1, F2, typeof(cache)}(col, wav, prop_AD!, my_cdf, 
        fluxvec, res, res_cache, cache, cache2,
        backup_buf, scaling_buf, lo_buf, up_buf, max_nw)
end

function get_res!(x::AbstractVector, nx::Int64, os::OMPStruct)
    nz = length(os.col.U)
    tmp = get_tmp(os.res_cache, x) 

    res = get_tmp(os.res, x)
    res .= os.fluxvec

    nw = div(nx, 3)
    for i = 1:nw
        # Non-allocating view for parameters
        p_i = view(x, (3*(i-1)+1):(3*i))
        
        # propagate wave i
        fill!(tmp, 0)
        os.prop_AD!(tmp, p_i, os.wav, os.col)
        
        @inbounds for j = 1:2nz
            res[j] -= tmp[j]
        end
    end
    return nothing
end

function cost(x::AbstractVector, nx::Int64, reg::Float64, os::OMPStruct)
    nz = length(os.col.U)
    get_res!(x,nx,os)
    out = 0.0

    res = get_tmp(os.res, x)
    
    for i = 1:2nz
        out += res[i]^2
    end
    out *= 0.5
    for i = 3:3:nx
        out += reg*x[i]
    end
    return out
end

function get_smart_p0(os, n_samples=50)
    s = skip(SobolSeq([-π, 0.0], [π, 100.0]), n_samples)
    best_p = [0.0, 0.0]
    best_val = Inf
    
    for i in 1:n_samples
        p = next!(s)
        val = get_score(p, os)
        if val < best_val
            best_val = val
            best_p = p
        end
    end
    return best_p
end

function find_next_wave!(x::AbstractVector, nx::Int64, os::OMPStruct)
    # In the OMP/sliding Frank-Wolfe formulation, finds the next 
    # wave to add to our support

    ## 1. Find the best (th, c) wave

    # Get residual
    get_res!(x,nx,os)

    # Define an anonymous function for the score at x
    sc = view(os.scaling_buf, 1:2)
    sc[1], sc[2] = pi, 100.0
    lo = view(os.lo_buf, 1:2)
    up = view(os.up_buf, 1:2)
    score = params -> get_score(params.*sc, os)

    # Starting wave guess
    n_sobol_samples = 40
    p0 = get_smart_p0(os, n_sobol_samples)./sc

    # Run the optimization
    lo[1], lo[2] = -pi/sc[1], 0.0/sc[2]     # Min angle, Min phase speed
    up[1], up[2] = pi/sc[1], 100.0/sc[2]    # Max angle, Max phase speed
    
    df = OnceDifferentiable(score, p0; autodiff=AutoForwardDiff())
    p_results = optimize(df, lo, up, p0, Fminbox(LBFGS()))
    
    # new wave (th,c)
    pw0 = Optim.minimizer(p_results).*sc

    if !Optim.converged(p_results)
        @warn "Optimization failed to converge on a new wave."
    end

    ## 2. Now minimize residual norm over new wave flux for x + new wave

    # Define new anonymous function to minimize
    # Brent expects a scalar variable
    resnorm = f -> begin
        tmp = get_tmp(os.tmp_cache, f) # dual or float based on type of f
        res = get_tmp(os.res, f)
        os.prop_AD!(tmp, [pw0[1], pw0[2], f], os.wav, os.col)

        val = 0.0
        for i in eachindex(tmp)
            val += (res[i] - tmp[i])^2 # res = (fluxvec - sum(g(xi))) - g(xnew)
        end
        return 0.5 * val
    end

    # Starting flux guess -- only important that it be big enough
    f0 = [3e-3] # start at 3mPa and work down

    # Minimize
    l = 0.0
    u = 15e-3   # maximum of 15 mPa (still stupid large)
    f_results = optimize(resnorm, l, u, Brent())

    if !Optim.converged(f_results)
        @warn "Optimization failed to converge on flux for new wave."
    end

    # Add new wave to the end of preallocated x vector
    # nx is length of x
    x[nx+1], x[nx+2], x[nx+3] = pw0[1], pw0[2], Optim.minimizer(f_results)
    nw = div(length(x), 3)
    nx += 3
    return nx
end

const SHARPEN_OPTIONS = Optim.Options(
    iterations = 20,
    outer_iterations = 5,
    show_trace = true,
    extended_trace = true,
    f_reltol = 1e-6,
    g_tol = 1e-6,
    allow_f_increases = true # Helpful for noisy ridges
)

function sharpen!(x::Vector{Float64}, nx::Int64, reg::Float64, os::OMPStruct)
    
    # Fill buffers
    nw = div(nx,3)
    sc = view(os.scaling_buf, 1:nx)
    lo = view(os.lo_buf, 1:nx)
    up = view(os.up_buf, 1:nx)
    for i = 0:nw-1
        idx = 3i
        sc[idx+1], sc[idx+2], sc[idx+3] = π, 100.0, 1e-3
        lo[idx+1], lo[idx+2], lo[idx+3] = -Inf, -Inf, 0.0
        up[idx+1], up[idx+2], up[idx+3] = Inf, Inf, Inf
    end

    # Objective function
    # let block freezes variables in scope so compiler is happier
    Iw = let sc=sc, reg=reg, os=os, nx=nx
        p_scaled -> begin
            tmp = view(get_tmp(os.tmp_cache2, p_scaled), 1:nx)

            res = get_tmp(os.res, p_scaled)
            
            # In-place scaling: tmp = p_scaled .* sc
            @inbounds for i in 1:nx
                tmp[i] = p_scaled[i] * sc[i]
            end
            
            get_res!(tmp, nx, os)
            
            val = 0.5*sum(abs2, res)

            # L1 Regularization on fluxes (every 3rd element)
            @inbounds for i = 3:3:nx
                val += reg * p_scaled[i]
            end
            return val
        end
    end

    # Optimize
    # need to move x to scaled space
    for i = 1:nx
        x[i] /= sc[i]
    end
    # pointer-wrapped flat vectors avoid dynamic sizing issues (but are uglier than the devil's ass)
    x_flat_view = unsafe_wrap(Vector{Float64}, pointer(x), nx)
    lo_flat_view = unsafe_wrap(Vector{Float64}, pointer(os.lo_buf), nx)
    up_flat_view = unsafe_wrap(Vector{Float64}, pointer(os.up_buf), nx)

    df = OnceDifferentiable(Iw, x_flat_view; autodiff=AutoForwardDiff())
    res = optimize(df, lo_flat_view, up_flat_view, x_flat_view, 
        Fminbox(LBFGS(linesearch = LineSearches.HagerZhang())), SHARPEN_OPTIONS)
    view(x,1:nx) .= Optim.minimizer(res)
    # ...and back from scaled space
    for i = 1:nx
        x[i] *= sc[i]
    end

    # convenience -- map angles and speeds back to [-pi, pi], [0,Inf]
    for i = 0:nw-1
        if x[i+2] < 0
            x[i+2] *= -1
            x[i+1] += pi
        end
        x[i+1] = rem2pi(x[i+1], RoundNearest)
    end
    
    return nothing
end

function find_measure!(x::Vector{Float64}, reg::Float64, os::OMPStruct)
    nz2 = 2 * length(os.col.U)
    nx = 0  
    fill!(x, 0.0)

    MAXITER = os.maxwaves
    BIC_prev = Inf
    nx_best = 0 

    # Track the baseline error of the *current* model before adding anything
    err_baseline = cost(x, nx, 0.0, os)

    for iter = 1:MAXITER
        # Back up state
        @inbounds os.backup_buf[1:nx] .= view(x, 1:nx)
        nx_prev = nx

        # 1. Find new wave candidate (greedy step)
        nx = find_next_wave!(x, nx, os)
        #### DEBUG
        println("Iteration $iter: new x start \n", x[1:nx])

        # 2. Cheap gatekeeper: Calculate approximate, unsharpened SSE
        err_approx = cost(x, nx, 0.0, os)

        # Drop out immediately if the greedy placement didn't even dent the residual.
        # Choose a conservative threshold (e.g., 0.999 means at least 0.1% improvement)
        if err_approx > 0.999 * err_baseline
            nx = nx_prev  # Roll back the candidate length
            break
        end

        # 3. Expensive optimization step: Only run if the candidate passes the gatekeeper
        sharpen!(x, nx, reg, os)

        # 4. True BIC: Evaluate the fully optimized joint model
        err_sharpened = cost(x, nx, 0.0, os)
        BIC = nz2*log(err_sharpened / nz2) + nx*log(nz2)

        if BIC > BIC_prev
            @inbounds x[1:nx_prev] .= view(os.backup_buf, 1:nx_prev)
            nx = nx_prev
            break
        else
            BIC_prev = BIC
            nx_best = nx
            err_baseline = err_sharpened # Update baseline for the next loop
        end
    end

    if nx_best < length(x)
        @inbounds fill!(view(x, (nx_best + 1):length(x)), 0.0)
    end

    return nx_best
end

# score function -- propagation-agnostic but nonsmooth
function get_score_ugly(params::AbstractVector, os::OMPStruct)
    # params = [theta, c]
    target_flux = [1e-9]
    
    f_to_diff = (p_flux) -> begin
        p_full = [params[1], params[2], p_flux[1]]
        
        # get_tmp looks at p_full. If p_full is Duals, local_tmp is the Dual array.
        # If p_full is Float64, local_tmp is the Float64 array.
        local_tmp = get_tmp(os.tmp_cache, p_full)
        
        # We must manually zero it out because it's recycled memory from the last pass
        fill!(local_tmp, 0) 
        
        # Run forward model
        os.prop_AD!(local_tmp, p_full, os.wav, os.col)
        
        return dot(local_tmp, os.res)
    end

    # ForwardDiff passes Dual numbers into f_to_diff
    score_val = ForwardDiff.gradient(f_to_diff, target_flux)
    
    return score_val[1]
end

# Lindzen-specific hack to get_score that makes it substantially nicer
function b_calc!(b_out::AbstractVector, wave_params::AbstractVector, wav::WaveProfile, col::ColumnProfile)
    th = wave_params[1]
    c = wave_params[2]
    # we don't use flux at all and want to optimize over only th and c

    nz = length(col.U)

    # We keep wav around for the wave parameters that we usually hold constant (absk, src)
    absk = hypot(wav.k, wav.l)
    cdir_1, cdir_2 = cos(th), sin(th)            # direction of wave propagation
    vb = col.U[wav.src]*cdir_1 + col.V[wav.src]*cdir_2        # mean flow speed in direction of wave at src
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/col.N[wav.src]
    sb = sign(c-vb)

    # Loop from bottom to top of column, depositing momentum where it exceeds maximum stable transport
    for lvl = wav.src:col.nlev
        if lvl == col.nlev
            vt = vb
            bt = bb
        else
            vt = col.U[lvl+1]*cdir_1 + col.V[lvl+1]*cdir_2       # flow speed in direction cdir at top of cell
            bt = col.rho[lvl+1] * 0.5*absk*(c-vt)^3/col.N[lvl+1]   # breaking condition -- maximum stable momentum transport
        end

        # flux_out depends on bb, so we want bb from this function
        b_out[lvl] = bb

        # everything below this actually modulates bt
        
        # If c-v at bottom and top of cell have different signs, there's a 
        # critical layer somewhere in here and we need to break and dump all momentum
    	vscale = 0.4  # should scale things such that 1m/s difference is tanh(2.5) = 0.986 
    	sig_crit = 0.5*(1 + tanh(sb*(c-vt)/vscale))
        bb *= sig_crit
        bt *= sig_crit
        
        # In flux deposition mode, we deposit flux if flux > bb
        # Here we want bb, so we need to keep a rolling min of bb
        if abs(bb) < abs(bt)
            # Ensure bt retains its sign, but is capped at the magnitude of bb
            bt = sign(bt) * abs(bb)
        end
        
        vb = vt
        bb = bt
    end
end
#=
function get_score(params::AbstractVector, os::OMPStruct)
    # Uses the Lindzen hack to get global gradient information by
    # tracking b instead of the gradient
    # cdf is the CDF of the probability distribution we apply to different
    # fluxes when integrating the score over them

    # this should pull the right tmp to match params
    tmp = get_tmp(os.tmp_cache, params)
    fill!(tmp,0)
    nz = div(length(tmp),2)

    # calculate b (only fills first half of tmp)
    b_calc!(tmp, params, os.wav, os.col)

    # replace b with cdf(b) -- this is the integral of dg/df over pdf(f)
    # The expected gradient carries the sign of the wave flux, 
    # but the CDF is evaluated on the magnitude of the breaking limit.
    @inbounds for i in eachindex(tmp)
        tmp[i] = sign(tmp[i]) * os.cdf(abs(tmp[i]))
    end

    # Need something here to penalize waves that don't break until the top
    # Because otherwise they pretend to project equally well onto all localized residuals
    res = get_tmp(os.res, params)
    tmp ./= (0.1*length(tmp) + norm(tmp,1))
    # tmp ./= ((params[2]/100.0)^2 + 1)

    # And one last piece of the puzzle. So far we've only used th and c
    # together as one variable, which collapses it to one DOF
    # To reintroduce the second, we need to use the angle of the flux
    th = params[1]
    for i = 1:nz
        tmp[nz+i] = sin(th)*tmp[i]
        tmp[i]    = cos(th)*tmp[i]
    end

    # score is abs(dot(cdf(b), res))
    return dot(tmp, res)
end
=#

function get_score(params::AbstractVector, os::OMPStruct)
    # Estimates a finite change in residual on adding a wave defined by params
    # with a nonzero expected amplitude

    # this should pull the right tmp to match params
    tmp = get_tmp(os.tmp_cache, params)
    fill!(tmp,0)

    # fill tmp with finite-amplitude wave
    f_exp = 0.5e-3
    os.prop_AD!(tmp, [params[1], params[2], f_exp], os.wav, os.col)

    # Compute change in residual
    res = get_tmp(os.res, params)
    out = -dot(tmp, res)
    out += 0.5*sum(abs2, tmp)
    return out
end

function L81_AD_wrapper!(flux_out::AbstractVector, wave_params::AbstractVector, wav::WaveProfile, col::ColumnProfile)
    # Computes L81(flux*wave) and d/dflux L81(flux*wave)
    # Reformats inputs so automatic differentiation can do its thing

    th = wave_params[1]
    c = wave_params[2]
    flux = wave_params[3]

    # softening the discontinuities in the gradient by replacing
    # if statements with sigmoids
    ksig = 20   # sigmoid sensitivity
    tiny = 1e-15  # for scaling

    nz = length(col.U)

    # We keep wav around for the wave parameters that we usually hold constant (absk, src)
    absk = hypot(wav.k, wav.l)
    cdir_1, cdir_2 = cos(th), sin(th)            # direction of wave propagation
    vb = col.U[wav.src]*cdir_1 + col.V[wav.src]*cdir_2        # mean flow speed in direction of wave at src
    momsign = sign(c-vb)         # flux has to be in direction c-v_src
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/col.N[wav.src]
    sb = sign(c-vb)
    
    # Loop from bottom to top of column, depositing momentum where it exceeds maximum stable transport
    for lvl = wav.src:col.nlev
        #println(flux, "\t", bb)
        if flux == 0
            # if the wave has no momentum we don't need to keep going
            break
        end
        if lvl == col.nlev
            vt = vb
            bt = bb
        else
            vt = col.U[lvl+1]*cdir_1 + col.V[lvl+1]*cdir_2       # flow speed in direction cdir at top of cell
            bt = col.rho[lvl+1] * 0.5*absk*(c-vt)^3/col.N[lvl+1]   # breaking condition -- maximum stable momentum transport
        end
        # If c-v at bottom and top of cell have different signs, there's a 
        # critical layer somewhere in here and we need to break and dump all momentum
        vscale = 0.4  # should scale things such that 1m/s difference is tanh(2.5) = 0.986 
    	sig_crit = 0.5*(1 + tanh(sb*(c-vt)/vscale))
        bb *= sig_crit
        bt *= sig_crit
        
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
        th_eff = (1-s)     # for flux << bb, th_eff = 1; for flux >> bb, th_eff = 0
        fi = flux
        flux = (1-th_eff)*bb + th_eff*flux  
        
        # we assume that momentum gets deposited in the direction of phase speed
        flux_out[lvl] = momsign*flux*cdir_1                   # zonal
        flux_out[lvl+col.nlev] = momsign*flux*cdir_2          # meridional (all the ys at the end)
        
        vb = vt
        bb = bt
    end
end
