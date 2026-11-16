include("mini-newton.jl")

########################################################################
## Idea: find the "best" one wave to encapsulate the signal, then iterate
##    A form of greedy descent. Initial conditions are important here.

# Probably makes sense to use finite differences here since the analytic
# derivatives don't play nice

# Objective function
function obj_3param(x,p)
    propag!,col,b,K,src,lambda,mu,Gi = p
    # x = (beta, theta, c)
    flux = exp(x[1])
    th = x[2]
    dir = [cos(th) sin(th) 0.0]
    k = dir[1]*K
    l = dir[2]*K
    wave = WaveProfile(k,l,0.0,0.0,x[3]*dir,flux,src)
    Gi .= 0.0
    # we don't use the Jacobian here but might as well make it accurate
    propag!(wave,col,flux,Gi)
    Gi .-= b
    # lambda and mu are regularization for amplitude and wavespeed, resp.
    return Gi'*Gi + lambda*flux + mu*x[2]^2
end

function obj_grad!(J,x,p)
    db = 1e-5      # larger than usual for a finite diff. but I'm worried about the exponential
    dth = 2e-8     # theta is O(1)
    dc = 1e-1      # c is O(10)
    dx = [db dth dc]
    I = obj_3param(x,p)
    for i = 1:length(x)
        xd = x .+ 0.0
        xd[i] += dx[i]
        tmp = obj_3param(xd,p)
        J[i] = (tmp-I)/dx[i]
    end
    return nothing
end

function obj_hess!(H,x,p)
    # going to try a somewhat clever version of the fin.diff. 
    # to reduce function evaluations (19->13)
    db = 1e-5      # larger than usual for a finite diff. but I'm worried about the exponential
    dth = 2e-8     # theta is O(1)
    dc = 1e-1      # c is O(10)
    dx = [db dth dc]
    I = obj_3param(x,p)
    Ix = zeros(3,2)  # holds single perturbations + and - for each var
    Ixy = zeros(3,2) # holds ++ and -- double perturbations; e.g. I(b+db,t+dt,c) or I(b-db,t-dt,c)
    # Ixy holds them in the order given by pairs, defined below 
    xd = zeros(length(x))
    # diagonal terms
    for i = 1:length(x)
        xd .= x
        xd[i] += dx[i]
        Ix[i,1] = obj_3param(xd,p)
        xd[i] = x[i] - dx[i]
        Ix[i,2] = obj_3param(xd,p)
        H[i,i] = (Ix[i,1] - 2I + Ix[i,2])/dx[i]^2
    end
    ## cross terms
    pairs = [(1,2),(1,3),(2,3)]
    for idx = length(pairs)
        i,j = pairs[idx]

        # calculating double perturbation terms
        # I don't really need to save them but it saves me
        # calling things tmp and tmp2, idk
        xd .= x
        xd[i] += dx[i]
        xd[j] += dx[j]
        Ixy[idx,1] = obj_3param(xd,p)
        xd[i] = x[i] - dx[i]
        xd[j] = x[j] - dx[j]
        Ixy[idx,2] = obj_3param(xd,p)

        H[i,j] = Ixy[idx,1] - Ix[i,1] - Ix[j,1] + 2I - Ix[i,2] - Ix[j,2] + Ixy[idx,2]
        H[i,j] /= (2*dx[i]*dx[j])
        H[j,i] = H[i,j]  # Hessian is symmetric
    end
    return nothing
end