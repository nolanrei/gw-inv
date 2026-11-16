using LinearAlgebra
using SparseArrays
using StaticArrays
using IterativeSolvers
using Random
using Plots
using Printf
using Optimization, OptimizationOptimJL, OptimizationIpopt
using Zygote, ReverseDiff

function ls_obj(x,p)
    A,b,lambda = p
    tmp = b .- A*x
    return tmp'*tmp + lambda*sum(x)     # L1 regularization
end

function nnls(A::AbstractArray{Tv,2}, b::AbstractVector{Tv}, lambda::Tv=1e-7) where {Tv <: Real}
    m,n = size(A)
    p = (A, b, lambda)
    x0 = lsqr(A, b; damp=lambda) # has negative numbers (out of domain) by default
    # let's get rid of those o-o-d's
    eps = 1e-10
    for i = 1:length(x0)
        x0[i] = (x0[i] > eps ? x0[i] : eps)
    end
    
    cons(res,x,p) = (res.=x)       # constraint function -- here just x since we want x \geq 0
    
    optf = OptimizationFunction(ls_obj, Optimization.AutoZygote(), cons=cons)
    prob = OptimizationProblem(optf, x0, p, lcons = zeros(n), ucons = Inf*ones(n))    # lower bound 0, upper bound Inf
    
    sol = solve(prob, IpoptOptimizer(
               hessian_approximation = "limited-memory",
               limited_memory_max_history = 10); verbose=true)

    return sol.u
end

A = 1e-7*[1.0 2.0 -2.0  1.0;
     2.0 1.0  1.0 -2.0]
b = A*[1.0, 1.0, 0.0, 0.0]
err = 1e-10*randn(2)
be = b .+ err
lambda = 1e-11
x0 = lsqr(A, be; damp=lambda)
x = nnls(A,be,lambda)