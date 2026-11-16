# Despite the name, it implements an algorithm somewhat closer to
# Sliding Frank-Wolfe by Denoyelle et al. 2019

using Optim
using StaticArrays
using ForwardDiff
using LinearAlgebra
using PreallocationTools
using ADTypes
using Sobol
using LineSearches
using LeastSquaresOptim
using BlackBoxOptim

include(expanduser("~/work/gw-inv/L81-2.jl"))

const MAX_C = 150.0

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
    
    xcopy       :: Vector{T}
    xbest       :: Vector{T}
    backup_buf  :: Vector{T}
    sc          :: SVector{3, T}
    lo          :: SVector{3, T}
    up          :: SVector{3, T}
    maxwaves    :: Int64
end

# Define explicit types to bypass runtime closure compilation bottlenecks
struct L81HessTag end
const TagType = ForwardDiff.Tag{L81HessTag, Float64}
const D1 = ForwardDiff.Dual{TagType, Float64, 3}
const D2 = ForwardDiff.Dual{TagType, D1, 3}

struct FullHessianOMPWorkspace
    nz2::Int64
    max_waves::Int64
    
    # Pre-allocated nested dual profiles (one vector per active wave)
    profile_buffers::Vector{Vector{D2}}
    p_seeded::Vector{D2}
    
    # Real-valued scratch pads
    r::Vector{Float64}
    sum_W::Vector{Float64}
    
    function FullHessianOMPWorkspace(nz::Int64, max_waves::Int64)
        nz2 = 2 * nz
        profile_buffers = [zeros(D2, nz2) for _ in 1:max_waves]
        p_seeded = zeros(D2, 3)
        return new(nz2, max_waves, profile_buffers, p_seeded, zeros(nz2), zeros(nz2))
    end
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

    xcopy = zeros(3max_nw)
    xbest = zeros(3max_nw)
    backup_buf = zeros(3max_nw)
    sc = SA[2π, 100.0, 4e-3]
    lo = SA[-Inf, -150/sc[2], 0.0]
    up = SA[Inf, 150/sc[2], Inf]
    
    return OMPStruct{Float64, F, typeof(cache)}(col, wav, prop_AD!, 
        fluxvec, 20.0, res, res_cache, cache, cache2, xcopy, xbest,
        backup_buf, sc, lo, up, max_nw)
end

function fwd!(out::AbstractVector, x::AbstractVector, nx::Int64, os::OMPStruct; ksig::Float64=os.ksig)
    nz = length(os.col.U)
    tmp = get_tmp(os.res_cache, x)
    
    fill!(out, 0)
    nw = div(nx, 3)
    for i = 1:nw
        # Non-allocating view for parameters
        p_i = view(x, (3*(i-1)+1):(3*i))
        
        # propagate wave i
        fill!(tmp, 0)
        os.prop_AD!(tmp, p_i, os.wav, os.col; ksig)
        
        @inbounds for j = 1:2nz
            out[j] += tmp[j]
        end
    end
    return nothing
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
    
    for i = os.wav.src:nz
        out += res[i]^2
        out += res[i+nz]^2
    end
    out *= 0.5
    for i = 3:3:nx
        out += reg*x[i]
    end
    return out
end


function find_next_wave!(x::AbstractVector, nx::Int64, reg::Float64, os::OMPStruct; nwnew::Int64=1, max_f_calls=10000, pop_size=30, stride=2)
    # In the OMP/sliding Frank-Wolfe formulation, finds the next 
    # wave to add to our support
    nz = length(os.col.U)
    tmp = get_tmp(os.tmp_cache, x)
    nxnew = 3nwnew
    params = zeros(nxnew)

    # Get residual
    get_res!(x,nx,os)
    res = get_tmp(os.res, x)

    # scaling
    sc = os.sc

    # 1. Define the coarse objective
    resnorm_coarse = let os=os, params=params, tmp=tmp, res=res, reg=reg, sc=sc, 
        nwnew=nwnew, nz2=2nz, stride=stride
        p -> begin
            resc = get_tmp(os.res_cache, p)
            fill!(tmp, 0.0)
            # map scaled parameters to physical space
            y_to_x!(params,p,sc,nwnew)
            
            for i = 0:nwnew-1
                idx = 3i
                # stride>1 skips calculating a bunch of the points
                os.prop_AD!(resc, view(params, idx+1:idx+3), os.wav, os.col; ksig=os.ksig, stride)
                
                # COARSE: Only accumulate every nth grid point
                @inbounds for j = 1:stride:nz2
                    tmp[j] += resc[j]
                end
            end
            
            out = 0.0
            # COARSE: Only evaluate residual on every nth grid point
            @inbounds for i = 1:stride:nz2
                out += (res[i] - tmp[i])^2
            end
            out *= 0.5
            
            for i = 0:nwnew-1
                idx = 3i
                out += reg * params[idx+3]
            end
            
            return out
        end
    end

    # 2. Coarse shotgun search
    best_coarse_val = Inf
    guess_x = zeros(Float64, nxnew)
    temp_x = zeros(Float64, nxnew)
    
    # Rapidly evaluate a couple thousand starting guesses
    n_coarse = 5000
    lo = 0.0
    up = 1.0
    sob = skip(SobolSeq(zeros(nxnew), ones(nxnew)), n_coarse)
    for _ in 1:n_coarse
        next!(sob, temp_x)
        
        val = resnorm_coarse(temp_x)
        if val < best_coarse_val
            best_coarse_val = val
            guess_x .= temp_x
        end
    end

    # 3. Full-resolution resnorm
    resnorm = let os=os, params=params, tmp=tmp, res=res, reg=reg, sc=sc, nwnew=nwnew, nz2=2nz
        p -> begin
            resc = get_tmp(os.res_cache, p)
            fill!(tmp,0.0)
            y_to_x!(params,p,sc,nwnew)
            for i = 0:nwnew-1
                idx=3i
                os.prop_AD!(resc, view(params,idx+1:idx+3), os.wav, os.col; ksig=os.ksig)
                @inbounds for j = 1:nz2
                    tmp[j] += resc[j]
                end
            end
            
            out = 0.0
            for i = eachindex(tmp)
                out += (res[i] - tmp[i])^2
            end
            out *= 0.5
            for i = 0:nwnew-1
                idx=3i
                out += reg*params[idx+3]
            end
            return out
        end
    end

    # SearchRange=(0.0, 1.0) corresponds to [(0,2pi), (0,100), (0,4e-3)]
    result = bboptimize(resnorm, SearchRange=(0.0, 1.0), NumDimensions=nxnew,
        Method=:adaptive_de_rand_1_bin, MaxFuncEvals=max_f_calls, 
        PopulationSize=pop_size, initial_x=guess_x, TraceMode = :silent)

    y_to_x!(view(x,nx+1:nx+nxnew), best_candidate(result), sc, nwnew)
    nx += nxnew
    return nx
end

function pack_surviving_parameters!(dest::AbstractVector, source::AbstractVector, n_waves::Int64, drop_idx::Int64)
    # this doesn't really need to be implemented but it's more elegant
    curr_dst = 1
    for i = 0:n_waves-1
        if i == drop_idx; continue; end
        @inbounds dest[curr_dst]   = source[3i+1]
        @inbounds dest[curr_dst+1] = source[3i+2]
        @inbounds dest[curr_dst+2] = source[3i+3]
        curr_dst += 3
    end
end

struct BICOptions
    min_improvement::Float64      # for sharpen!, how much flat improvement we need to accept new solution
    complexity_weight::Float64    # 1.0 is standard BIC, </> puts more/less emphasis on fit quality
    noise_variance::Float64       # expected variance of additive flux noise
end

BICOptions(;
    min_improvement=0.0,
    complexity_weight=1.0,
    noise_variance=0.0,
) = BICOptions(min_improvement, complexity_weight, noise_variance)

function evaluate_bic(x::AbstractVector, nx::Int64, os::OMPStruct, opts::BICOptions)
    nz2 = 2 * length(os.col.U)
    err = cost(x, nx, 0.0, os)
    BIC = nz2*log(err / nz2 + opts.noise_variance) + 
            opts.complexity_weight*nx*log(nz2)
end

function x_to_y!(y, x, sc, nw; iw_start=0)
    # Real space --> solution space
    for i = iw_start:nw-1
        idx = 3i
        y[idx+1] = x[idx+1]/sc[1]
        y[idx+2] = x[idx+2]/sc[2]
        y[idx+3] = sqrt(x[idx+3]/sc[3])
    end
    return nothing
end

function y_to_x!(x, y, sc, nw; iw_start=0)
    # Solution space --> real space
    for i = iw_start:nw-1
        idx = 3i
        x[idx+1] = sc[1]*y[idx+1]
        x[idx+2] = sc[2]*y[idx+2]
        x[idx+3] = sc[3]*(y[idx+3]^2)
    end
    return nothing
end

# Source-level saturation bound
function source_bound(th, c, wav::WaveProfile, col::ColumnProfile)
    absk = hypot(wav.k, wav.l)
    s  = wav.src
    vb = col.U[s]*cos(th) + col.V[s]*sin(th)
    return col.rho[s] * 0.5 * absk * abs(c - vb)^3 / (col.N[s] + 1e-15)
end

# Clamp launched flux to what the forward model can actually see
function clamp_to_source!(x::AbstractVector, nx::Int, os::OMPStruct)
    for i = 1:div(nx, 3)
        b = source_bound(x[3i-2], x[3i-1], os.wav, os.col)
        x[3i] = min(x[3i], b)
    end
    return nothing
end

function sharpen_thresh!(x::AbstractVector, nx::Int64, os::OMPStruct, 
        bic_opts::BICOptions, fgh_closure; verbose=false, params)
    # Simple sharpen step that drops waves with amplitudes below a threshold

    thresh = params[1]     # we only need one thing in params
    
    nw = div(nx, 3)
    nz2 = 2 * length(os.col.U)
    sc = os.sc

    x_to_y!(x, x, sc, nw)

    # sharpen

    options = Optim.Options(
        iterations=50,
        show_trace = false,
    )
    
    lenx = length(x)
    resize!(x,nx)
    res = Optim.optimize(Optim.only_fgh!(fgh_closure), x, NewtonTrustRegion(), options)
    if verbose
        println(res.minimizer)
    end
    x[1:nx] .= res.minimizer
    resize!(x,lenx)

    y_to_x!(x,x,sc,nw)
    clamp_to_source!(x, nx, os)

    # threshold
    idx = 1
    nw_t = nw
    for i = 1:nw
        # kill any waves with flux less than 0.01 mPa
        if x[3i] <= thresh
            x[3i-2] = x[3i-1] = x[3i] = 0.0
            nw_t -= 1
            continue
        end
        # contract to fill space
        x[3idx-2] = x[3i-2]
        x[3idx-1] = x[3i-1]
        x[3idx]   = x[3i]
        idx += 1
    end

    # Map th and c onto [0, 2pi) and [0.0, MAX_C]
    for i = 0:nw-1
        if x[3i+2] < 0
            x[3i+2] *= -1
            x[3i+1] += pi
        end
        x[3i+1] = mod(x[3i+1], 2pi)
        x[3i+2] = min(x[3i+2], MAX_C)
    end
    return 3*nw_t
end

function fb_sharpen_adaptive!(x::AbstractVector, nx::Int64, os::OMPStruct, 
        bic_opts::BICOptions, fgh_closure; verbose=false, params)
    # Forward-Backward step
    # Start with a solution then remove each wave in turn, getting rid of the ones
    # that don't sufficiently improve the solution

    fast = params[1]  # params only needs to hold fast

    # NOTE: reg is baked into fgh_closure
    nw = div(nx, 3)
    nz2 = 2 * length(os.col.U)
    sc = os.sc
    lo = os.lo
    up = os.up

    noise = 1e-7
    bic_x = evaluate_bic(x,nx,os, bic_opts)

    x_to_y!(x, x, sc, nw)
    
    options = Optim.Options(
        iterations=50,
        show_trace = false,
    )
    
    lenx = length(x)
    resize!(x,nx)
    res = Optim.optimize(Optim.only_fgh!(fgh_closure), x, NewtonTrustRegion(), options)
    if verbose
        println(res.minimizer)
    end
    x[1:nx] .= res.minimizer
    resize!(x,lenx)
    
    
    keep_dropping = true

    xcopy_init_len = length(os.xcopy)
    
    while keep_dropping && nw > 1
        best_drop_score = Inf
        target_dim = nx - 3
        resize!(os.xcopy,target_dim)
        
        # Test dropping each individual wave
        for drop_idx = 0:nw-1
            # Pack all waves except the dropped one into a pre-allocated buffer
            pack_surviving_parameters!(os.xcopy, x, nw, drop_idx)

            if fast
                # cost needs to be evaluated in x space
                y_to_x!(os.xcopy, os.xcopy, sc, nw-1)
                score = cost(os.xcopy, target_dim, 1e-8, os)  # Not too happy about this but we don't have reg
                x_to_y!(os.xcopy, os.xcopy, sc, nw-1)
            else
                res = Optim.optimize(Optim.only_fgh!(fgh_closure), os.xcopy, 
                                        NewtonTrustRegion(), options)
                score = Optim.minimum(res)
            end

            if verbose
                y_to_x!(os.xcopy, os.xcopy, sc, nw-1)
                println("Best sol dropping wave $drop_idx: ", os.xcopy)
                x_to_y!(os.xcopy, os.xcopy, sc, nw-1)
                println("score: $score")
            end

            # Drop each wave, re-evaluate how good solution is
            # Choose wave to drop by which solution was the best
            if score < best_drop_score
                best_drop_score = score
                # keep track of best wave to drop
                if fast
                    # fast optimizes later so we don't want to rescale
                    os.xbest[1:target_dim] .= os.xcopy
                else
                    # slow can scale back
                    y_to_x!(os.xbest, res.minimizer, sc, nw-1)
                end
            end
        end

        if fast
            # Fast optimizes once after dropping a wave
            res = Optim.optimize(Optim.only_fgh!(fgh_closure), os.xbest, 
                                NewtonTrustRegion(), options)
            # now we scale back
            y_to_x!(os.xbest, res.minimizer, sc, nw-1)
        end
        
        # Evaluate BIC / pruning condition...
        bic_xbest = evaluate_bic(os.xbest, target_dim, os, bic_opts)
        if verbose
            println("Trial x: ", os.xbest[1:target_dim])
            println(bic_xbest,"\t",bic_x)
        end
        
        if bic_xbest <= bic_x - bic_opts.min_improvement
            x_to_y!(x, view(os.xbest, 1:target_dim), sc, nw-1)
            nx = target_dim
            nw -= 1
            bic_x = bic_xbest
        else
            keep_dropping = false
        end
    end
    y_to_x!(x,x,sc,nw)

    # Map th and c onto [0, 2pi) and [0.0, MAX_C]
    for i = 0:nw-1
        if x[3i+2] < 0
            x[3i+2] *= -1
            x[3i+1] += pi
        end
        x[3i+1] = mod(x[3i+1], 2pi)
        x[3i+2] = min(x[3i+2], MAX_C)
    end
    
    resize!(os.xcopy,xcopy_init_len)
    return nx
end

function find_measure!(x::Vector{Float64}, reg::Float64, os::OMPStruct,
        sharpen!::Function, ws::Union{FullHessianOMPWorkspace,Nothing}, 
        bic_opts::BICOptions; sharpen_params,zis=nothing)
    nz = length(os.col.U)
    nz2 = 2 * nz
    nx = 0  
    fill!(x, 0.0)

    max_nx = 3*os.maxwaves
    BIC_prev = evaluate_bic(x, nx, os, bic_opts)
    nx_best = 0 

    if isnothing(ws)
        ws = FullHessianOMPWorkspace(nz, os.maxwaves)
    end

    fgh_closure = make_full_hessian_fgh(os, ws, reg, os.sc, zis=zis)

    nwnew = 2
    while nx + 3*nwnew <= max_nx
        # Back up state
        @inbounds os.backup_buf[1:nx] .= view(x, 1:nx)
        nx_prev = nx

        # 1. Find new wave candidate (greedy step)   -- O(50ms)
        nx = find_next_wave!(x, nx, reg, os; nwnew, max_f_calls=5000, pop_size=50, stride=1)
        clamp_to_source!(x, nx, os)
        #### DEBUG
        #println("find_measure!: new x start, nx=$nx \n", x[1:nx])

        # 2. Optimize over all parameters      -- O(1ms * (nw/4)^2 * nw)
        nx = sharpen!(x, nx, os, bic_opts, fgh_closure; params=sharpen_params)

        #println("find_measure!: post-sharpen, nx=$nx \n", x[1:nx])

        # termination criterion -- BIC isn't decreasing any more
        BIC = evaluate_bic(x, nx, os, bic_opts)
        if BIC >= BIC_prev - bic_opts.min_improvement
            @inbounds for i = 1:nx_prev
                x[i] = os.backup_buf[i]
            end
            nx = nx_prev
            break
        else
            BIC_prev = BIC
            nx_prev = nx
        end
    end
    #println("find_measure!: post-termination, nx=$nx \n", x[1:nx])

    if nx < length(x)
        @inbounds fill!(view(x, (nx + 1):length(x)), 0.0)
    end

    return nx
end

function get_score(params::AbstractVector, f_guess::Float64, os::OMPStruct)
    # Estimates a finite change in residual on adding a wave defined by params
    # with a nonzero expected amplitude

    # this should pull the right tmp to match params
    tmp = get_tmp(os.tmp_cache, params)
    fill!(tmp,0)

    # fill tmp with finite-amplitude wave
    os.prop_AD!(tmp, SA[params[1], params[2], f_guess], os.wav, os.col)

    # Compute change in residual
    res = get_tmp(os.res, params)
    out = -dot(tmp, res)
    out += 0.5*sum(abs2, tmp)
    return out
end

function make_full_hessian_fgh(os::OMPStruct, ws::FullHessianOMPWorkspace, reg::Float64, sc::SVector; zis=nothing)
    nz2 = ws.nz2
    nz = div(nz2,2)
    window = isnothing(zis) ? collect(1:nz2) : vcat(collect(zis), collect(zis) .+ nz)
    
    return function fgh!(f, g, H, x)
        n_waves = div(length(x), 3)
        
        # --- 1. FORWARD PASS (Seeding & Vector Hyper-AD) ---
        fill!(ws.sum_W, 0.0)
        
        for k in 1:n_waves
            # Extract raw parameters for wave k
            p1 = x[3*k-2]
            p2 = x[3*k-1]
            p3 = x[3*k]
            
            # Manually seed the hyper-duals for a 3-parameter Hessian matrix
            for i in 1:3
                val = x[3*(k-1)+i]
                
                # Level 1 Partials (Inner)
                # Dual is structured as (value, derivatives)
                # e.g. p1 --> (p1, (dp1/dp1, dp1/dp2, dp1/dp3)) = (p1, (1.0, 0.0, 0.0))
                inner_p = ForwardDiff.Partials((i==1 ? 1.0 : 0.0, i==2 ? 1.0 : 0.0, i==3 ? 1.0 : 0.0))
                inner_dual = D1(val, inner_p)
                
                # Level 2 Partials (Outer)
                # Now holds ((p1, (1.0, 0.0, 0.0, 0.0)), (Hessian matrix = 0.0 initially))
                outer_p1 = D1(i==1 ? 1.0 : 0.0, ForwardDiff.Partials((0.0, 0.0, 0.0)))
                outer_p2 = D1(i==2 ? 1.0 : 0.0, ForwardDiff.Partials((0.0, 0.0, 0.0)))
                outer_p3 = D1(i==3 ? 1.0 : 0.0, ForwardDiff.Partials((0.0, 0.0, 0.0)))
                outer_partials = ForwardDiff.Partials((outer_p1, outer_p2, outer_p3))
                
                ws.p_seeded[i] = D2(inner_dual, outer_partials)
            end
            
            # Rescale our chunk
            p_seeded_scaled = SA[
                sc[1] * ws.p_seeded[1],
                sc[2] * ws.p_seeded[2],
                sc[3] * ws.p_seeded[3]^2
            ]
            
            # Run the physics engine.
            prof_buf = ws.profile_buffers[k]
            os.prop_AD!(prof_buf, p_seeded_scaled, os.wav, os.col; ksig=os.ksig)
            
            # Accumulate the real values into the total model profile
            @inbounds for i in window#1:nz2
                ws.sum_W[i] += prof_buf[i].value.value
            end
        end
        
        # Compute master residual vector: r = target - model
        @inbounds for i in window#1:nz2
            ws.r[i] = os.fluxvec[i] - ws.sum_W[i]
        end
        
        # --- 2. COMPUTE COST FUNCTION ---
        if f !== nothing
            cost = 0.0
            @inbounds for i in window#1:nz2
                cost += ws.r[i]^2
            end
            cost *= 0.5
            
            for k in 1:n_waves
                flux_unscaled = sc[3] * x[3k] * x[3k]
                cost += reg * flux_unscaled
            end
            f = cost     # f is a float so this doesn't do anything
        end
        
        # --- 3. COMPUTE GRADIENT ---
        if g !== nothing
            fill!(g, 0.0)
            for k in 1:n_waves
                g_k = view(g, (3*k-2):(3*k))
                prof_k = ws.profile_buffers[k]
                
                # g_k = -J_k' * r
                for col in 1:3
                    val = 0.0
                    @inbounds for row in window#1:nz2
                        val -= prof_k[row].value.partials[col] * ws.r[row]
                    end
                    g_k[col] = val
                end
                
                # Regularization gradient contribution
                g_k[3] += reg * 2sc[3] * x[3*k]
            end
        end
        
        # --- 4. Compute Hessian Matrix ---
        if H !== nothing
            fill!(H, 0.0)
            
            for k in 1:n_waves
                prof_k = ws.profile_buffers[k]
                row_range = (3*k-2):(3*k)
                
                for m in 1:n_waves
                    prof_m = ws.profile_buffers[m]
                    
                    # A. Gauss-Newton Core: J_k' * J_m
                    for c in 1:3
                        c_idx = 3*m - 3 + c
                        for r in 1:3
                            r_idx = 3*k - 3 + r
                            v = 0.0
                            @inbounds for i in window#1:nz2
                                v += prof_k[i].value.partials[r] * prof_m[i].value.partials[c]
                            end
                            H[r_idx, c_idx] = v
                        end
                    end
                    
                    # B. Non-Convex Curvature Correction: -sum( r * d²W/dp² )
                    # This applies ONLY to the 3x3 diagonal blocks (k == m)
                    if k == m
                        for c in 1:3
                            c_idx = 3*k - 3 + c
                            for r in 1:3
                                r_idx = 3*k - 3 + r
                                v_corr = 0.0
                                @inbounds for i in window#1:nz2
                                    v_corr -= ws.r[i] * prof_k[i].partials[r].partials[c]
                                end
                                H[r_idx, c_idx] += v_corr
                            end
                        end
                        
                        # Add analytical regularization curvature to the flux coordinate
                        H[3k, 3k] += reg * 2*sc[3]^2
                    end
                end
            end
        end
        
        return f
    end
end
#=
function make_lm_resjac!(os::OMPStruct, ws::LMWorkspace, sc::SVector)

    return function resjac!(r, J, x)

        nw  = div(length(x), 3)
        nz2 = ws.nz2

        fill!(ws.sum_W, 0.0)
        fill!(r, 0.0)

        for k in 1:nw

            # Seed the 3 parameters of this wave
            for i in 1:3
                val = x[3*(k-1)+i]

                ws.p_seeded[i] = D1(
                    val,
                    ForwardDiff.Partials((
                        i == 1 ? 1.0 : 0.0,
                        i == 2 ? 1.0 : 0.0,
                        i == 3 ? 1.0 : 0.0
                    ))
                )
            end

            p_scaled = SA[
                sc[1] * ws.p_seeded[1],
                sc[2] * ws.p_seeded[2],
                sc[3] * ws.p_seeded[3]^2
            ]

            prof = ws.profile_buffers[k]

            os.prop_AD!(
                prof,
                p_scaled,
                os.wav,
                os.col;
                ksig=os.ksig
            )

            @inbounds for i in 1:nz2
                ws.sum_W[i] += prof[i].value
            end
        end

        # residual
        @inbounds for i in 1:nz2
            r[i] = os.fluxvec[i] - ws.sum_W[i]
        end

        # Jacobian of residual
        fill!(J, 0.0)

        for k in 1:nw
            prof = ws.profile_buffers[k]

            cols = (3k-2):(3k)

            @inbounds for i in 1:nz2
                for j in 1:3
                    J[i, cols[j]] = -prof[i].partials[j]
                end
            end
        end

        return nothing
    end
end
=#
function L81_AD_wrapper!(flux_out::AbstractVector, wave_params::AbstractVector, 
                         wav::WaveProfile, col::ColumnProfile; 
                         ksig::Float64=20.0, stride::Int64=1,
                         mom_leak=false
    )
    # Computes L81(flux*wave) and d/dflux L81(flux*wave)
    # Reformats inputs so automatic differentiation can do its thing

    # Get the underlying type (will be Float64 normally, or Dual during AD)
    T = eltype(wave_params)

    th = wave_params[1]
    c = wave_params[2]
    flux = wave_params[3]

    # softening the discontinuities in the gradient by replacing
    # if statements with sigmoids
    tiny = T(1e-15)  # for scaling

    nz = length(col.U)

    # We keep wav around for the wave parameters that we usually hold constant (absk, src)
    absk = hypot(wav.k, wav.l)
    cdir_1, cdir_2 = cos(th), sin(th)            # direction of wave propagation
    vb = col.U[wav.src]*cdir_1 + col.V[wav.src]*cdir_2        # mean flow speed in direction of wave at src
    momsign = sign(c-vb)         # flux has to be in direction c-v_src
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/(col.N[wav.src] + tiny)
    sb = sign(c-vb)

    bmin = T(Inf)
    
    # Loop from bottom to top of column, jumping by stride
    for lvl = wav.src:stride:col.nlev        
        if lvl == col.nlev
            vt = vb
            bt = bb
        else
            # Look ahead by stride, capping at the top of the atmosphere
            next_lvl = min(lvl + stride, col.nlev)
            vt = col.U[next_lvl]*cdir_1 + col.V[next_lvl]*cdir_2       # flow speed at top of coarse cell
            bt = col.rho[next_lvl] * 0.5*absk*(c-vt)^3/(col.N[next_lvl] + tiny)   # breaking condition
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
        
        # sigmoid replaces "if abs(flux) > abs(bb)"
        sgnfl = sign(flux)
        bb = sgnfl*abs(bb)        # we need it to match the sign on flux
        diff = ( abs(flux) - abs(bb) )/(abs(bb) + tiny)
        s = 0.5*(1 + tanh(ksig * diff))  # if flux > bb, s will go to 1
        th_eff = (1-s)     # for flux << bb, th_eff = 1; for flux >> bb, th_eff = 0
        fi = flux
        flux = (1-th_eff)*bb + th_eff*flux  
        
        # we assume that momentum gets deposited in the direction of phase speed
        if mom_leak
            # Setting mimicking WACCM which will give flux the wrong sign on the last 
            # cell about half the time
            momsign = sign(c - 0.5*(vb+vt))  # WACCM does midpoint; this is close enough
        end
        flux_out[lvl] = momsign*flux*cdir_1                    # zonal
        flux_out[lvl+col.nlev] = momsign*flux*cdir_2           # meridional (all the ys at the end)
        
        vb = vt
        bb = bt
    end
end

# copying the whole thing to make it easier to dispatch into WACCM runs
function L81_AD_wrapper_leak!(flux_out::AbstractVector, wave_params::AbstractVector, 
                         wav::WaveProfile, col::ColumnProfile; 
                         ksig::Float64=20.0, stride::Int64=1
    )
    # Computes L81(flux*wave) and d/dflux L81(flux*wave)
    # Reformats inputs so automatic differentiation can do its thing

    # Get the underlying type (will be Float64 normally, or Dual during AD)
    T = eltype(wave_params)

    th = wave_params[1]
    c = wave_params[2]
    flux = wave_params[3]

    # softening the discontinuities in the gradient by replacing
    # if statements with sigmoids
    tiny = T(1e-15)  # for scaling

    nz = length(col.U)

    # We keep wav around for the wave parameters that we usually hold constant (absk, src)
    absk = hypot(wav.k, wav.l)
    cdir_1, cdir_2 = cos(th), sin(th)            # direction of wave propagation
    vb = col.U[wav.src]*cdir_1 + col.V[wav.src]*cdir_2        # mean flow speed in direction of wave at src
    momsign = sign(c-vb)         # flux has to be in direction c-v_src
    bb = col.rho[wav.src]*0.5*absk*(c-vb)^3/(col.N[wav.src] + tiny)
    sb = sign(c-vb)

    bmin = T(Inf)
    
    # Loop from bottom to top of column, jumping by stride
    for lvl = wav.src:stride:col.nlev        
        if lvl == col.nlev
            vt = vb
            bt = bb
        else
            # Look ahead by stride, capping at the top of the atmosphere
            next_lvl = min(lvl + stride, col.nlev)
            vt = col.U[next_lvl]*cdir_1 + col.V[next_lvl]*cdir_2       # flow speed at top of coarse cell
            bt = col.rho[next_lvl] * 0.5*absk*(c-vt)^3/(col.N[next_lvl] + tiny)   # breaking condition
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
        
        # sigmoid replaces "if abs(flux) > abs(bb)"
        sgnfl = sign(flux)
        bb = sgnfl*abs(bb)        # we need it to match the sign on flux
        diff = ( abs(flux) - abs(bb) )/(abs(bb) + tiny)
        s = 0.5*(1 + tanh(ksig * diff))  # if flux > bb, s will go to 1
        th_eff = (1-s)     # for flux << bb, th_eff = 1; for flux >> bb, th_eff = 0
        fi = flux
        flux = (1-th_eff)*bb + th_eff*flux  
        
        # we assume that momentum gets deposited in the direction of phase speed
        
        # Setting mimicking WACCM which will give flux the wrong sign on the last 
        # cell about half the time
        momsign = sign(c - 0.5*(vb+vt))  # WACCM does midpoint; this is close enough
        flux_out[lvl] = momsign*flux*cdir_1                    # zonal
        flux_out[lvl+col.nlev] = momsign*flux*cdir_2           # meridional (all the ys at the end)
        
        vb = vt
        bb = bt
    end
end

struct MK99Workspace
    m::AbstractVector                    # workspace for m values
    m1::AbstractVector                   # last step's m workspace
    uu::AbstractVector                   # workspace for rho uu values
    uu1::AbstractVector                  # last step's rho uu values
    idxs::Vector{Int64}                  # index vector for keeping track of sorting
    idxs1::Vector{Int64}                 # last step's index vector
    beta1::Vector{Float64}               # last step's betas
end

function MK99Workspace(max_nw::Int64)
    mw = zeros(max_nw)
    m1w = zeros(max_nw)
    uuw = zeros(max_nw)
    uu1w = zeros(max_nw)
    idxsw = zeros(Int64, max_nw)
    idxs1w = zeros(Int64, max_nw)
    betas1w = zeros(max_nw)
    return MK99Workspace(mw, m1w, uuw, uu1w, idxsw, idxs1w, betas1w)
end

function MK99!(flux_out::AbstractVector,             # output fluxes. 2nz tall
               x::AbstractVector,                    # input waves, used length nx
               nx::Int64,                            # length of x
               col::ColumnProfile,                   # column parameters (U, V, N, rho)
               dummy_wav::WaveProfile,               # stores parameters of waves we keep constant
               z::AbstractVector,                    # vertical grid (km)
               wk::MK99Workspace                     # stores all the memory needed
    )
    # Implements the Medvedev-Klaasen 1999 method for gravity wave parameterization
    # A critical component is that it evolves all waves together
    fill!(flux_out,0)
    
    nz2 = length(flux_out)
    nz = div(nz2,2)
    nw = div(nx,3)
    sq2 = sqrt(2)
    sq2pi = sqrt(2pi)

    K = dummy_wav.k
    src = dummy_wav.src
    tiny = 1e-20

    Uz = col.U[src]
    Vz = col.V[src]
    rhoz = col.rho[src]
    Nz = col.N[src]

    lenm = length(wk.m)   # we allow our workspace to be longer than the number of waves it currently holds
    # but we need to logically resize wk.m and wk.idxs to prevent the sorting from
    # accessing stuff out of bounds
    resize!(wk.m,nw)
    resize!(wk.idxs,nw)

    # initialize workspace
    for j = 1:nw
        th = x[3j-2]
        c = x[3j-1]
        f = x[3j]
        u = cos(th)*Uz + sin(th)*Vz   # initialized above
        wk.m[j] = Nz/(c-u)
        wk.uu[j] = abs(wk.m[j])/K * f    # rho uu = m/k rho uw  (we are evolving rho uu instead of uu)
        # abs(m) instead of m because <uu> needs to be positive physically (also gives right flux direction)

        flux_out[src]    += cos(th)*K/wk.m[j]*wk.uu[j]   # we need to record the flux here too
        flux_out[src+nz] += sin(th)*K/wk.m[j]*wk.uu[j]
    end
    sortperm!(wk.idxs, wk.m)
    # we need to compute beta at source level
    for j = nw:-1:1
        ij = wk.idxs[j]
        m = wk.m[ij]
        sigma = 0.0
        for n = j:nw
            sigma  += wk.uu[wk.idxs[n]]
        end
        sigma  = sqrt(sigma/rhoz)
        alpha = Nz/(sq2 * m * sigma)
        wk.beta1[j] = sq2pi * Nz/sigma * exp(-alpha*alpha)   # damping strength
    end
    
    for zi = src+1:nz     # src is where the values are prescribed, so we start computing one level above
        Uz1 = Uz; Vz1 = Vz; rhoz1 = rhoz; Nz1 = Nz
        Uz = col.U[zi]
        Vz = col.V[zi]
        rhoz = col.rho[zi]
        Nz = col.N[zi]
        dz = (z[zi] - z[zi-1])*1000    # turn it into meters
        # compute m values
        for j = 1:nw
            # move over last step's values
            wk.m1[j] = wk.m[j]
            wk.idxs1[j] = wk.idxs[j]
            wk.uu1[j] = wk.uu[j]
            
            th = x[3j-2]
            c = x[3j-1]
            u = cos(th)*Uz + sin(th)*Vz
            wk.m[j] = Nz/(c-u)
        end

        # sort m values
        sortperm!(wk.idxs, wk.m)

        # compute nonlinear effects and update from zi-1 to zi
        for j = nw:-1:1         # compute largest-m to smallest-m to get the correct trapezoid rule on beta
            ij = wk.idxs[j]
            m = wk.m[ij]
            m1 = wk.m1[ij]    # needs to refer to the same wave

            # this is our nonlinear coupling
            sigma = 0.0
            for n = j:nw   # yes we do count the wave itself in the background field
                # because we're computing uu(zi) from biggest m to smallest 
                # we have sigma except for at n==j, where we default to uu(z_i-1)
                sigma  += wk.uu[wk.idxs[n]]
            end
            sigma  = sqrt(sigma/rhoz + tiny)

            alpha = Nz/(sq2 * m * sigma)

            # update step
            if alpha > 1/sq2
                beta = sq2pi * Nz/sigma * exp(-alpha*alpha)   # damping strength
                beta1 = wk.beta1[ij]
                # uu1[ij] b/c it needs to be the same wave, not the same rank
                wk.uu[ij] = wk.uu1[ij] * m/m1 * exp(-0.5*(beta+beta1)*dz)
                wk.beta1[ij] = beta
            else
                # exp(-alpha^2) is an approximation to erfi(alpha)[erfi(alpha)^2-1]^-0.5
                # which goes violently to 0 around alpha = 1/sqrt(2)
                wk.uu[ij] = 0.0
                wk.beta1[ij] = Inf
            end

            # update flux in the column
            th = x[3ij-2]
            flux_out[zi]    += cos(th)*K/m*wk.uu[ij]
            flux_out[zi+nz] += sin(th)*K/m*wk.uu[ij]
        end
    end
    resize!(wk.m, lenm)
    resize!(wk.idxs, lenm)
    return nothing
end