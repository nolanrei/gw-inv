

function barrier(x, mu)
    bar = 0.0
    for i = 1:length(x)
        bar .+= -mu*log(x[i])
    end
    return bar
end

function barrier_grad!(J, x, mu)
    for i = 1:length(x)
        J[i] .+= -mu/x[i]
    end
    return nothing
end

function barrier_hess!(H, x, mu)
    for i = 1:length(x)
        H[i,i] .+= mu/(x[i]*x[i])
    end
    return nothing
end

function newton_update!(dx, H, hess!, J, grad!, x, mu, p)
    hess!(H,x,p)
    barrier_hess!(H, x, mu)
    grad!(J,x,p)
    barrier_grad!(J,x,mu)
    dx .= -H\J
    return nothing
end

function armijo_line_srch(dx, min_improvement, shrink, 
        obj, grad!, x, xnew, J, mu, p; 
        max_line_search_steps=30)
    fnow = obj(x,p)
    fnow += barrier(x,mu)
    grad!(J,x,p)
    barrier_grad!(J,x,mu)
    
    alpha = 1.0
    for i = 1:max_line_search_steps
        xnew .= x .+ alpha*dx
        fspace = obj(xnew,p)
        fspace += barrier(xnew,mu)
        # we require the objective function to decrease at least an amount 
        # proportional to the step length taken dot the gradient
        if fnow - fspace >= min_improvement*alpha*(J'*dx)
            break
        else
            # shrink should be a number between 0 and 1
            alpha *= shrink
        end
    else
        throw("Maximum number of iterations ($(max_line_search_steps)) reached in line search")
    end
    return alpha
end

function newton_solve!(x, obj, grad!, hess!, J, H, dx, xnew, mu, p; 
        min_improvement = 0.5, shrink = 0.5, 
        abstol = 1e-6, max_newton_steps = 10)
    # Minimizes f - mu sum_i log(x_i) to tolerance ∇f(x) < abstol
    for iter = 1:max_newton_steps
        # find search direction
        newton_update!(dx, H, hess!, J, grad!, x, mu, p)
        # find step size with backtracking line search
        alpha = armijo_line_srch(dx, min_improvement, shrink, obj, 
            grad!, x, xnew, J, mu, p; max_line_search_steps=30)
        # update x
        x .+= alpha*dx
        # test for convergence
        f = obj(x,p)
        f += barrier(x,mu)
        if norm(f) <= abstol
            break
        end
    else
        throw("Maximum number of iterations ($(max_newton_steps)) reached in newton solve")
    end
    return nothing
end

function interior_point(obj!, grad!, hess!, x0; max_mu_steps = 50, max_newton_steps=10, max_line_search_steps = 30)
    # runner function for interior point method with nonnegativity constraints
    
end