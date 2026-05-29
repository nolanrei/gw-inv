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
using BlackBoxOptim

include(expanduser("~/work/gw-inv/L81-2.jl"))

# T is the element type of the residual, F is the function, C is the Cache type
mutable struct OMPStruct{T, F, C}
    col         :: ColumnProfile
    wav         :: WaveProfile
    prop_AD!    :: F
    fluxvec     :: Vector{T}
    ksig        :: T
    
    res         :: C
    res_cache   :: C
    tmp_cache   :: C        # holds a DiffCache instead of a Vector
    tmp_cache2  :: C

    backup_buf  :: Vector{T}
    sc_buf      :: Vector{T}
    lo_buf      :: Vector{T}
    up_buf      :: Vector{T}
    maxwaves    :: Int64
end

function OMPStruct(col, prop_AD!::F, wav; max_nw=20, fluxvec=nothing) where F
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
    sc_buf = zeros(3max_nw)
    lo_buf = zeros(3max_nw)
    up_buf = zeros(3max_nw)
    
    return OMPStruct{Float64, F, typeof(cache)}(col, wav, prop_AD!, 
        fluxvec, 20.0, res, res_cache, cache, cache2,
        backup_buf, sc_buf, lo_buf, up_buf, max_nw)
end

function get_res!(x::AbstractVector, nx::Int64, os::OMPStruct; ksig::Float64=os.ksig)
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
        os.prop_AD!(tmp, p_i, os.wav, os.col; ksig)
        
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

#=
function find_next_wave!(x::AbstractVector, nx::Int64, reg::Float64, os::OMPStruct)
    # In the OMP/sliding Frank-Wolfe formulation, finds the next 
    # wave to add to our support
    tmp = get_tmp(os.tmp_cache, x)
    params = zeros(3)

    # Get residual
    get_res!(x,nx,os)
    res = get_tmp(os.res, x)

    # lower and upper bounds
    lo = view(get_tmp(os.lo_buf,x),1:3)
    up = view(get_tmp(os.up_buf,x),1:3)
    sc = view(get_tmp(os.sc_buf,x),1:3)
    sc[1],sc[2],sc[3] = 2pi,100.0,1e-3
    fmax = log(1e-1/sc[3])
    fmin = log(1e-7/sc[3])
    nlsc3 = fmax-fmin
    lo[1],lo[2],lo[3] = -0.5,0.0,fmin/nlsc3
    up[1],up[2],up[3] = 0.5,1.0,fmax/nlsc3

    resnorm = let os=os, params=params, tmp=tmp, res=res, reg=reg, sc=sc
        p -> begin
            params[1] = sc[1]*p[1]
            params[2] = sc[2]*p[2]
            params[3] = sc[3]*exp(nlsc3*p[3])  # log flux space
            os.prop_AD!(tmp, params, os.wav, os.col; ksig=os.ksig)
            out = 0.0
            for i = eachindex(tmp)
                out += (res[i] - tmp[i])^2
            end
            return 0.5*out + reg*params[3]
        end
    end

    result = bboptimize(resnorm, SearchRange=[(lo[1],up[1]),(lo[2],up[2]),(lo[3],up[3])],
                        Method=:adaptive_de_rand_1_bin, MaxFuncEvals=10000, PopulationSize=100)

    x[nx+1:nx+3] .= best_candidate(result)
    x[nx+1:nx+2] .*= sc[1:2]
    x[nx+3] = sc[3]*exp(nlsc3*x[nx+3])
    nx += 3
    return nx
end
=#

function find_next_wave!(x::AbstractVector, nx::Int64, reg::Float64, os::OMPStruct)
    # In the OMP/sliding Frank-Wolfe formulation, finds the next 
    # wave to add to our support
    tmp = get_tmp(os.tmp_cache, x)
    p_cur = zeros(3)

    # Get residual
    ksig_smooth = 10.0        # low value here smooths reconstruction
    get_res!(x,nx,os; ksig=ksig_smooth)
    res = get_tmp(os.res, x)
    nz = length(os.col.U)

    # Starting wave guess
    n_sobol_samples = 100
    s = skip(SobolSeq([-π, 0.0], [π, 100.0]), n_sobol_samples)
    p0 = view(p_cur,1:2)
    pbest = view(x,(nx+1):(nx+3))
    cbest = Inf
    for iter = 1:n_sobol_samples
        next!(s,p0)
        resnorm = let os=os, tmp=tmp, res=res, reg=reg, p_cur=p_cur
            f -> begin
                p_cur[3] = exp(f)     # p0 is the first two elements of tmp2 already
                fill!(tmp,0)
                os.prop_AD!(tmp, p_cur, os.wav, os.col; ksig=ksig_smooth)
                out = 0.0
                for i = eachindex(tmp)
                    out += (res[i] - tmp[i])^2
                end
                return 0.5*out + reg*p_cur[3]
            end
        end
    
        # Minimize
        l = log(1e-8)
        u = log(5e-2)   # maximum of 50 mPa (still stupid large)
        f_results = optimize(resnorm, l, u, Brent())

        fmin = Optim.minimum(f_results)
        if fmin < cbest
            pbest .= p0[1], p0[2], exp(Optim.minimizer(f_results))
            cbest = fmin
        end
    end
     
    nx += 3
    return nx
end

function sharpen!(x::Vector{Float64}, nx::Int64, reg::Float64, os::OMPStruct)
    
    nw = div(nx, 3)
    nz2 = 2 * length(os.col.U)
    sc = view(os.sc_buf, 1:nx)
    lo = view(os.lo_buf, 1:nx)
    up = view(os.up_buf, 1:nx)
    
    for i = 0:nw-1
        idx = 3i
        sc[idx+1], sc[idx+2], sc[idx+3] = π, 100.0, 1e-3
        lo[idx+1], lo[idx+2], lo[idx+3] = -Inf, -150/sc[idx+2], 0.0
        up[idx+1], up[idx+2], up[idx+3] = Inf, 150/sc[idx+2], Inf
    end

    small = 1e-7
    x[nx] = (x[nx] < small ? x[nx] : small)

    n_res = nz2 + 2nw
    
    # Trackers for closure
    current_reg = Ref(reg)
    i_of = Ref(0)

    f_lsq! = let sc=sc, current_reg=current_reg, i_of=i_of, os=os, nx=nx, nz2=nz2, nw=nw, x=x
        (out, p_scaled) -> begin
            n_p = length(p_scaled)
            ci_of = i_of[]
            c_sm = sm[]

            tmp = view(get_tmp(os.tmp_cache2, p_scaled), 1:nx)

            # fill all elements with constant baseline
            @inbounds for i in 1:nx
                tmp[i] = x[i] * sc[i]
            end
            
            # overwrite only the active window
            @inbounds for i in 1:n_p
                idx = ci_of + i
                tmp[idx] = p_scaled[i] * sc[idx]
            end
            
            get_res!(tmp, nx, os; ksig=os.ksig)
            res_phys = get_tmp(os.res, p_scaled)
            
            @inbounds for i in 1:nz2
                out[i] = res_phys[i]
            end

            reg_val = current_reg[]

            # 3. Regularization routing
            @inbounds for i = 1:nw
                idx_flux = 3i
                idx_c = 3i - 1
                
                # Check if this wave falls inside the active optimization window
                if ci_of < idx_flux <= ci_of + n_p
                    flux_scaled = p_scaled[idx_flux - ci_of]
                    c_scaled = p_scaled[idx_c - ci_of]
                else
                    flux_scaled = x[idx_flux]
                    c_scaled = x[idx_c]
                end

                out[nz2 + 2i-1] = sqrt(2.0 * reg_val * max(0.0, flux_scaled) + 1e-12)
                out[nz2 + 2i]   = 4e-7 * c_scaled
            end
            
            return out
        end
    end

    # Scale to solver space
    for i = 1:nx
        x[i] /= sc[i]
    end

    lenx = length(x)
    
    # =========================================================================
    # PHASE 1: Smooth Optimization Landscape (Active Window Only)
    # =========================================================================
    os.ksig = 5.0
    current_reg[] = 0.0

    # Determine window size and offset
    nx_a = nw == 1 ? 3 : 6
    i_of[] = nx - nx_a
    
    # Use unsafe_wrap to trick the solver into accepting a standard Vector
    # pointer(x, idx) is 1-based, so we add 1 to the offset
    x_active = unsafe_wrap(Vector{Float64}, pointer(x, i_of[] + 1), nx_a)
    
    prob_smooth = LeastSquaresProblem(
        x = x_active, 
        f! = f_lsq!, 
        output_length = n_res, 
        autodiff = :forward
    )
    
    res_smooth = LeastSquaresOptim.optimize!(prob_smooth, LevenbergMarquardt(), 
                    lower = view(os.lo_buf, (i_of[] + 1):nx), 
                    upper = view(os.up_buf, (i_of[] + 1):nx),
                    #iterations = 5, 
                    x_tol = 1e-6, 
                    f_tol = 1e-6)

    # Write back the results from the active window
    x_active .= res_smooth.minimizer

    # =========================================================================
    # PHASE 2: Sharp/Physical Optimization Landscape (All Waves)
    # =========================================================================
    os.ksig = 20.0
    current_reg[] = reg
    i_of[] = 0  # Reset offset to 0 so the active window is the whole array

    resize!(x, nx)
    
    prob_sharp = LeastSquaresProblem(
        x = x, 
        f! = f_lsq!, 
        output_length = n_res, 
        autodiff = :forward
    )

    res_sharp = LeastSquaresOptim.optimize!(prob_sharp, LevenbergMarquardt(), 
                    lower = view(os.lo_buf, 1:nx), 
                    upper = view(os.up_buf, 1:nx),
                    x_tol = 1e-6, 
                    f_tol = 1e-6)

    view(x, 1:nx) .= res_sharp.minimizer
    resize!(x, lenx)

    # Back from scaled space
    for i = 1:nx
        x[i] *= sc[i]
    end

    for i = 0:nw-1
        if x[3i+2] < 0
            x[3i+2] *= -1
            x[3i+1] += pi
        end
        x[3i+1] = rem2pi(x[3i+1], RoundNearest)
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

    # Keep track of our guess for the largest remaining wave's amplitude
    f_guess = 0.5e-3   # 0.5 mPa at start
    f_shrink = 0.1

    for iter = 1:MAXITER
        # Back up state
        @inbounds os.backup_buf[1:nx] .= view(x, 1:nx)
        nx_prev = nx

        # 1. Find new wave candidate (greedy step)
        nx = find_next_wave!(x, nx, reg, os)
        f_guess *= f_shrink
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

function get_score(params::AbstractVector, f_guess::Float64, os::OMPStruct)
    # Estimates a finite change in residual on adding a wave defined by params
    # with a nonzero expected amplitude

    # this should pull the right tmp to match params
    tmp = get_tmp(os.tmp_cache, params)
    fill!(tmp,0)

    # fill tmp with finite-amplitude wave
    os.prop_AD!(tmp, [params[1], params[2], f_guess], os.wav, os.col)

    # Compute change in residual
    res = get_tmp(os.res, params)
    out = -dot(tmp, res)
    out += 0.5*sum(abs2, tmp)
    return out
end

function L81_AD_wrapper!(flux_out::AbstractVector, wave_params::AbstractVector, 
        wav::WaveProfile, col::ColumnProfile; ksig::Float64=20.0)
    # Computes L81(flux*wave) and d/dflux L81(flux*wave)
    # Reformats inputs so automatic differentiation can do its thing

    # Get the underlying type (will be Float64 normally, or Dual during AD)
    T = eltype(wave_params)

    th = wave_params[1]
    c = wave_params[2]
    flux = wave_params[3]

    # softening the discontinuities in the gradient by replacing
    # if statements with sigmoids
    #ksig = 20   # sigmoid sensitivity
    tiny = T(1e-15)  # for scaling

    nz = length(col.U)

    # We keep wav around for the wave parameters that we usually hold constant (absk, src)
    absk = hypot(wav.k, wav.l)
    cdir_1, cdir_2 = cos(th), sin(th)            # direction of wave propagation
    vb = col.U[wav.src]*cdir_1 + col.V[wav.src]*cdir_2        # mean flow speed in direction of wave at src
    momsign = sign(c-vb)         # flux has to be in direction c-v_src
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/col.N[wav.src]
    sb = sign(c-vb)

    bmin = T(Inf)
    
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

        # Compute b(z) instead of b-tilde(z)
        if abs(bb) < bmin
            bmin = abs(bb)
        else
            bb = sign(bb)*bmin
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
