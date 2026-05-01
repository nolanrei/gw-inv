using PositiveFactorizations

function newton_update!(dx, H, hess!, J, grad!, x, p)
    hess!(H,x,p)
    grad!(J,x,p)
    # cholesky(Positive,H) invokes PositiveFactorizations
    # to compute the modified cholesky factorization, approximating 
    # non-positive-definite Hessians minimally and efficiently
    F = lu!(H)#cholesky!(Positive,H)
    dx .= -(F\J)
    return nothing
end

function gauss_newton_update!(dx, J, grad!, r, dfk, x, p)
    grad!(J,x,p)   # needs to construct dfk and also residual r=f(x)-b
    # these will be stored in p somewhere
    dx .= -dfk\r
    return nothing
end

function armijo_line_srch(dx, min_improvement, shrink, 
        obj, grad!, x, xnew, J, p; 
        max_line_search_steps=30, verbose=true,
        max_steps_no_error=false
    )
    # generally if you're hitting max steps something's wrong
    # If you're REALLY sure you need to not know when this happens,
    # max_steps_no_error = true will return the value quietly
    # with a flag that tells newton_solve to stop immediately
    # (since the step size is effectively 0, further steps are useless)
    fold = obj(x,p)
    grad!(J,x,p)
    
    alpha = 1.0
    iter = 1
    while iter <= max_line_search_steps
        xnew .= x .+ alpha*dx
        fnew = obj(xnew,p)
        # we require the objective function to decrease at least an amount 
        # proportional to the step length taken dot the gradient
        # (remember that gradient goes away from minimum so J'*dx always <0
        # for good steps)
        #println("f(xi) = $fold \t f(x_{i+1}) = $fnew \t criterion = $(min_improvement*alpha*(J'*dx))")
        if fnew - fold <= min_improvement*alpha*(J'*dx)
            break
        else
            # shrink should be a number between 0 and 1
            alpha *= shrink
        end
        iter += 1
    end
    if verbose
        println("# line search steps: $iter")
    end
    if iter > max_line_search_steps
        if !max_steps_no_error
            throw("Maximum number of iterations ($(max_line_search_steps)) reached in line search")
        else
            # flag 1 tells newton_solve to stop iterating
            return alpha,1
        end
    end
    # flag 0 says everything is fine
    return alpha,0
end

function newton_solve!(x, obj, grad!, hess!, J, H, dx, xnew, p; 
        min_improvement = 0.3, shrink = 0.5, 
        reltol = 1e-6, abstol = 1e-9, max_newton_steps = 10, 
        keep_steps = false, error_on_max_steps = true, verbose = true,
        max_line_search_steps=30, armijo_max_steps_no_error=false)  
    # Minimizes f to tolerance |∇f(x)| < reltol*|∇f(x0)| + abstol
    grad!(J,x,p)
    normJ0 = norm(J)  # magnitude of gradient at start, for breaking condition
    if keep_steps
        nx = length(x)
        steps = zeros(nx,max_newton_steps)
    end
    iter = 1
    while iter <= max_newton_steps
        if keep_steps
            steps[:,iter] .= x
        end
        # find search direction
        newton_update!(dx, H, hess!, J, grad!, x, p)
        # find step size with backtracking line search
        alpha,flag = armijo_line_srch(dx, min_improvement, shrink, obj, 
            grad!, x, xnew, J, p; max_line_search_steps, verbose,
            max_steps_no_error=armijo_max_steps_no_error)
        #println("\t",alpha)
        if flag == 1
            # armijo_line_srch is sad -- we're not finding a good step.
            # Stop iterating; this is as good as we're going to get

            ## NOTE: this quiet error behavior is NOT RECOMMENDED
            # and comes from setting armijo_max_steps_no_error = true
            # Use at your own peril
            break
        end
        # update x
        x .+= alpha*dx
        # test for convergence
        grad!(J,x,p)
        nJ = norm(J)
        if verbose
            println("x = $x")
            println("|∇f(x)| = $nJ \t f(x) = $(obj(x,p))")
        end
        if nJ <= reltol*normJ0 + abstol
            break
        end
        iter += 1
    end
    println(norm(J),"\t",reltol*normJ0,"\t",abstol)
    if error_on_max_steps && (iter > max_newton_steps)
        throw("Maximum number of iterations ($(max_newton_steps)) reached in newton solve")
    end
    if keep_steps
        return steps[:,1:iter-1]
    end
    return iter-1   # (let me know the number of iterations)
end

