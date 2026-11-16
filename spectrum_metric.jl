using OptimalTransport
using Distributions
using Clustering

function energy_distance(X::AbstractVector, Y::AbstractVector, dist::Function)
    # Measure distance between 
    # - X: a collection of samples from an empirical distribution
    # - Y: a collection of samples from a target distribution
    # Both X and Y should be collections of vectors describing objects in the same space
    # dist is a metric for ||X - Y||
    m = length(X)
    n = length(Y)

    XY = 0.0
    XX = 0.0
    YY = 0.0
    for i = 1:m
        x = X[i]
        for j = 1:n
            y = Y[j]
            XY += dist(x,y)
        end

        for j = 1:m
            x2 = X[j]
            XX += dist(x,x2)
        end
    end

    for i = 1:n
        for j = 1:n
            y = Y[i]
            y2 = Y[j]
            YY += dist(y,y2)
        end
    end

    XY /= (m*n)
    XX /= (m*m)
    YY /= (n*n)

    return 2XY - XX -YY
end

function polar_dist(pol1, pol2)
    θ1, r1 = pol1
    θ2, r2 = pol2
    s1, c1 = sincos(θ1)
    s2, c2 = sincos(θ2)
    return hypot(r1*c1 - r2*c2, r1*s1 - r2*s2)
end

# TV-based proxdiv, replacing the default KL-based one
# From Chizat et al. 2016: for F(p|q) = λ|p-q|, the proxdiv operator clips
# the ratio p/s to [exp(-λ/ε), exp(λ/ε)]
function proxdiv_TV(s, p, lam, eps)
    lo = exp(-lam/eps)
    hi = exp(lam/eps)
    return clamp.(p ./ s, lo, hi)
end

function earth_mover_dist(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    # Compute the Wasserstein distance between x and y point-mass spectra
    # x and y must be in (th, c, flux) flat vector form (th1, c1, f1, th2, c2, f2, ...)
    nx = length(x)
    nwx = div(nx,3)
    ny = length(y)
    nwy = div(ny,3)
    
    masses_x = view(x,3:3:nx)
    masses_y = view(y,3:3:ny)

    Cxy = zeros(nwx, nwy)
    Cx  = zeros(nwx, nwx)
    Cy  = zeros(nwy, nwy)
    for i = 1:nwy
        for j = 1:nwx
            Cxy[j,i] = polar_dist(view(y,3i-2:3i-1), view(x,3j-2:3j-1))
        end
    end

    for i = 1:nwx
        for j = 1:nwx
            Cx[j,i] = polar_dist(view(x,3i-2:3i-1), view(x,3j-2:3j-1))
        end
    end

    for i = 1:nwy
        for j = 1:nwy
            Cy[j,i] = polar_dist(view(y,3i-2:3i-1), view(y,3j-2:3j-1))
        end
    end

    # lam sets the cost of creating mass in x and y
    # Inf recovers balanced Wasserstein distance.
    lam = 80.0    # has units of distance. In our case, (m/s).

    ϵ = 10.0       # has units of distance. length scale at which points are "more or less the same".

    cost = sinkhorn_divergence_unbalanced(masses_x, masses_y, Cxy, Cx, Cy, lam, ϵ)
    return cost
end

function earth_mover_plan(x::AbstractVector{Float64}, y::AbstractVector{Float64};
        lam::Float64=80.0, epsilon::Float64=10.0)
    nx = length(x)
    nwx = div(nx,3)
    ny = length(y)
    nwy = div(ny,3)
    
    masses_x = view(x,3:3:nx)
    masses_y = view(y,3:3:ny)

    Cxy = zeros(nwx, nwy)
    for i = 1:nwy
        for j = 1:nwx
            Cxy[j,i] = polar_dist(view(y,3i-2:3i-1), view(x,3j-2:3j-1))
        end
    end

    plan = sinkhorn_unbalanced(masses_x, masses_y, Cxy, lam, lam, epsilon)
    return plan
end

function pairwise_dist(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    nx = length(x)
    nwx = div(nx,3)
    ny = length(y)
    nwy = div(ny,3)
    
    Cxy = zeros(nwx, nwy)
    for i = 1:nwy
        for j = 1:nwx
            Cxy[j,i] = polar_dist(view(y,3i-2:3i-1), view(x,3j-2:3j-1))
        end
    end
    return Cxy
end

function earth_mover_dist_tv(x::AbstractVector{Float64}, y::AbstractVector{Float64};
        lam::Float64=40.0, eps::Float64=5.0)
    nx = length(x); nwx = div(nx,3)
    ny = length(y); nwy = div(ny,3)

    masses_x = view(x,3:3:nx)
    masses_y = view(y,3:3:ny)

    Cxy = zeros(nwx, nwy)
    Cx  = zeros(nwx, nwx)
    Cy  = zeros(nwy, nwy)
    for i = 1:nwy, j = 1:nwx
        Cxy[j,i] = polar_dist(view(y,3i-2:3i-1), view(x,3j-2:3j-1))
    end
    for i = 1:nwx, j = 1:nwx
        Cx[j,i] = polar_dist(view(x,3i-2:3i-1), view(x,3j-2:3j-1))
    end
    for i = 1:nwy, j = 1:nwy
        Cy[j,i] = polar_dist(view(y,3i-2:3i-1), view(y,3j-2:3j-1))
    end

    F = (s,p) -> proxdiv_TV(s,p,lam,eps)

    OT_xy = sinkhorn_unbalanced2(masses_x, masses_y, Cxy, lam, lam, eps; proxdiv_F1=F, proxdiv_F2=F)
    OT_xx = sinkhorn_unbalanced2(masses_x, masses_x, Cx,  lam, lam, eps; proxdiv_F1=F, proxdiv_F2=F)
    OT_yy = sinkhorn_unbalanced2(masses_y, masses_y, Cy,  lam, lam, eps; proxdiv_F1=F, proxdiv_F2=F)

    return OT_xy - 0.5*(OT_xx + OT_yy)
end