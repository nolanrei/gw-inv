include("mini-newton.jl")

function nl_obj(x,p)
    propag!,propag_2der!,waves,col,b,lambda,BIG,Gi,J,der2,w,wndw,wndw_flag = p
    Gi .= 0.0
    J .= 0.0
    for i in eachindex(waves)
        # we don't use the Jacobian here but might as well make it accurate
        propag!(waves[i],col,x[i],Gi,view(J,:,i))
        #println(Gi)
    end
    Gi .-= b
    if wndw_flag
        # window Gi to only look at a subsection of the column
        Gi .*= wndw
    end
    # optimization needs the numbers to be larger
    return BIG*(Gi'*Gi + lambda*sum(w.*exp.(x)))
end

function nl_obj_grad!(G,x,p)
    # computes gradient of nl_obj w/ respect to x in G
    # J must be size nz x nw
    propag!,propag_2der!,waves,col,b,lambda,BIG,Gi,J,der2,w,wndw,wndw_flag = p
    # need to clean up Gi and J storage arrays
    Gi .= 0.0
    J .= 0.0
    for i in eachindex(waves)
        # additively builds up Gi into g(S;X) while building up jacobian
        v = view(J,:,i)
        propag!(waves[i],col,x[i],Gi,v)
        #v .*= exp.(x[i])
    end
    Gi .-= b
    if wndw_flag
        # window Gi to only look at a subsection of the column
        Gi .*= wndw
        for i = 1:nw
            J[1:nz,i] .*= wndw
            J[nz+1:2nz,i] .*= wndw
        end
    end

    G .= BIG*(lambda*w.*exp.(x) .+ 2*J'*Gi)
    return nothing
end

function nl_obj_hess!(H,x,p)
    # computes Hessian of nl_obj w/ respect to x in H
    # J must be size nz x nw
    propag!,propag_2der!,waves,col,b,lambda,BIG,Gi,J,der2,w,wndw,wndw_flag = p
    @assert typeof(propag_2der!) <: Function "p[2] must be 2nd deriv function"
    # need to clean up Gi and J storage arrays
    eps = 1e-12
    Gi .= 0.0
    J .= 0.0
    der2 .= 0.0
    nw = length(waves)
    for i = 1:nw
        # additively builds up Gi into g(S;X) while building up jacobian
        propag!(waves[i],col,x[i],Gi,view(J,:,i))
        # fills in second derivative
        propag_2der!(waves[i],col,x[i],view(der2,:,i))
    end
    Gi .-= b
    if wndw_flag
        # window Gi to only look at a subsection of the column
        Gi .*= wndw
        for i = 1:nw
            J[1:nz,i] .*= wndw
            J[nz+1:2nz,i] .*= wndw
            der2[1:nz,i] .*= wndw
            der2[nz+1:2nz,i] .*= wndw
        end
    end
    H .= 2*BIG*(J'*J)
    for i = 1:nw
        # second derivative correction
        correction = dot(Gi,view(der2,:,i))
        # cuts off negative corrections since they can lead to 
        # the Newton step not being a descent direction
        corr_throttle = (correction > 0.0 ? 1.0 : 0.0) 
        H[i,i] += 2*BIG*corr_throttle*correction
        # +eps is undocumented regularization preventing small xi from crashing inversion
        H[i,i] += BIG*lambda*w[i]*exp(x[i]) + eps
    end
    return nothing
end

# Setup function for an inversion
function init_inv(waves, col, fluxvec; lambda=1e-5, BIG=1e6, wndw_flag=false, wndw=zeros(0))
    nz = length(col.U)
    nw = length(waves)
    
    J = zeros(2nz,nw)
    der2 = zeros(2nz,nw)
    gi = zeros(2nz)

    H = zeros(nw,nw)
    dx = zeros(nw)
    xnew = zeros(nw)
    df = zeros(nw)     # gradient
    x0 = -8*ones(nw)   # starting condition
    # Let's look at a starting condition that downvalues large fluxes
    #x0 = zeros(nw)
    #for i = 1:nw
    #    x0[i] = -8.0 + log(50/( 50+0.01*sum(waves[i].c.^2) ))
    #end

    # z weights are interesting but not what I intended to write
    #=
    z = 10.25:0.5:64.75
    z_wts = exp.(vcat(z,z)/28)   # very roughly rho^0.25
    sw = sum(wts)
    wts ./= (sw/length(wts));
    =#

    # makes high-speed waves more expensive in the optimization
    wts = zeros(nw)
    for i = 1:nw
        wts[i] = (( 50+0.01*sum(waves[i].c.^2) )/50)
    end

    p = (L81_exp_grad!,L81_exp_hess!,waves,col,fluxvec,lambda,BIG,gi,J,der2,wts,wndw,wndw_flag)  # params
    mem = (dx, xnew, df, H)    # memory
    return x0, p, mem
end

# Mid-level interface
function wave_invert_mid!(x, p, mem; reltol=1e-5, abstol=1e-10, 
        max_newton_steps=60, error_on_max_steps=true,
        verbose=false, armijo_max_steps_no_error=false, 
        max_line_search_steps=30, keep_steps=false)
    dx, xnew, df, H = mem
    niter = newton_solve!(x, nl_obj, nl_obj_grad!, nl_obj_hess!, df, H, dx, xnew, p; 
        reltol, abstol, max_newton_steps, error_on_max_steps, max_line_search_steps, 
        verbose, armijo_max_steps_no_error, keep_steps)
    return niter
end

# high-level interface
function wave_invert_hi(waves, col, fluxvec)
    x0, p, mem = init_inv(waves, col, fluxvec; lambda=1e-5, BIG=1e6)
    wave_invert_mid!(x0,p,mem)
    return x0
end