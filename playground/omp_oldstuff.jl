
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

function fb_sharpen!(x::Vector{Float64}, nx::Int64, reg::Float64, os::OMPStruct)
    # Forward-Backward step
    # Start with two new waves, then remove each wave in turn (including previous waves)
    # and keep best solution (ending at nw+1 waves)
    nw = div(nx, 3)
    nz2 = 2 * length(os.col.U)
    sc = view(os.sc_buf, 1:nx)
    lo = view(os.lo_buf, 1:nx)
    up = view(os.up_buf, 1:nx)
    xc = get_tmp(os.xcache, x)
    
    for i = 0:nw-1
        idx = 3i
        sc[idx+1], sc[idx+2], sc[idx+3] = π, 100.0, 1e-3
        lo[idx+1], lo[idx+2], lo[idx+3] = -Inf, -150/sc[idx+2], 0.0
        up[idx+1], up[idx+2], up[idx+3] = Inf, 150/sc[idx+2], Inf
    end

    # Rescale and transform x to solution space
    for i = 0:nw-1
        idx = 3i
        x[idx+1] /= sc[idx+1]
        x[idx+2] /= sc[idx+2]
        x[idx+3] = sqrt(x[idx+3]/sc[idx+3])
    end

    fill!(xc,0)

    ## We assume that any sharpening on the nx solution has already been done
    ## Now we do only the sharpening on subproblems

    # For reference: 
    bic0 = Inf
    bic1 = Inf

    ## Declare optimization problem (unconstrained, solving for beta = sqrt(f0) such that beta^2 always >0)
    x_out = zeros(Float64, nx-3)
    td = let nx=nx-3, nw=nw-1, os=os, reg=reg   # looking at a problem with nw-1 waves
        obj = x -> begin 
            tmp = get_tmp(os.tmp_cache2, x)
            # scale x back
            for j = 0:nw-1
                idx = 3j
                tmp[idx+1] = sc[idx+1]*x[idx+1]
                tmp[idx+2] = sc[idx+2]*x[idx+2]
                tmp[idx+3] = sc[idx+3]*(x[idx+3]^2)
            end
            return cost(tmp, nx, reg, os)
        end
        TwiceDifferentiable(obj, x_out, autodiff=:forward)
    end

    ## Loop over waves, dropping one at a time and optimizing the remainder
    options = Optim.Options()
    for i = 0:nw-1
        # copy the rest into x_out
        for j = 1:3i
            x_out[j] = x[j]
        end
        for j = 3i+4:nx
            x_out[j-3] = x[j]
        end

        # Optimize
        res = Optim.optimize(td, x_out, NewtonTrustRegion(), options)

        # scale res.minimizer back
        for j = 0:nw-2   # this array is only 3*(nw-1) long
            idx = 3j
            idx2 = 3*(j>i ? j-1 : j)
            x_out[idx+1] = sc[idx2+1]*res.minimizer[idx+1]
            x_out[idx+2] = sc[idx2+2]*res.minimizer[idx+2]
            x_out[idx+3] = sc[idx2+3]*(res.minimizer[idx+3]^2)
        end

        # Test if BIC(res.minimizer) is lower than BIC(x_out)
        bic1 = nz2*log(cost(x_out,nx-3,0.0,os) / nz2) + (nx-3)*log(nz2)
        if bic1 < bic0
            for j = 1:nx-3
                xc[j] = x_out[j]
            end
            bic0 = bic1
        end
    end

    ## Store the best solution (we already rescaled)
    nx = nx-3
    for j = 1:nx
        x[j] = xc[j]
    end
    return nx
end

function test_jump!(x::Vector{Float64}, nx::Int64, dx::Float64, f0::Float64, reg::Float64, os::OMPStruct)
    ### Randomly perturb, sharpen, accept/reject based on relative quality
    ### x, nx -- solution so far
    ### dx    -- magnitude of perturbations to try
    ### f0    -- objective function at x,nx (to avoid recomputing it each test_jump!)
    ### reg   -- regularization parameter
    ### os    -- the struct holding everything defining a problem
    nw = div(nx, 3)
    nz2 = 2 * length(os.col.U)
    sc = view(os.sc_buf, 1:nx)
    lo = view(os.lo_buf, 1:nx)
    up = view(os.up_buf, 1:nx)
    xc = get_tmp(os.xcache, x)
    tmp = view(get_tmp(os.tmp_cache2, x), 1:nx)
    
    for i = 0:nw-1
        idx = 3i
        sc[idx+1], sc[idx+2], sc[idx+3] = π, 100.0, 1e-3
        lo[idx+1], lo[idx+2], lo[idx+3] = -Inf, -150/sc[idx+2], 0.0
        up[idx+1], up[idx+2], up[idx+3] = Inf, 150/sc[idx+2], Inf
    end

    ## Copy x to xc and add perturbation
    for i = 1:nx
        xc[i] = x[i]
    end
    # perturbation
    for i = 1:nx
        tmp[i] = dx*randn()
    end
    tmp .*= sc      # scale perturbation
    # add perturbation and clamp so xc is inside domain
    for i = 1:nx
        xc[i] = min(up[i]*sc[i], max(lo[i]*sc[i], xc[i] + tmp[i]))
    end

    ## Sharpen the perturbed solution
    sharpen!(xc, nx, reg, os)

    ## Check to see which solution is better
    f1 = cost(xc, nx, reg, os)

    # if f1 < f0, we accept the new solution
    if f1 < f0
        for i = 1:nx
            x[i] = xc[i]
        end
    else
        f1 = f0   # otherwise reject test
    end
    return f1  # so the next test_jump! can use the new cost without recomputing it
end