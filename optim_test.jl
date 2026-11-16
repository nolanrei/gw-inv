using LinearAlgebra
using SparseArrays
using StaticArrays
using IterativeSolvers
using Random
using Plots
using NetCDF
using Profile
using CSV
using DataFrames
using Printf
using Optimization, OptimizationMOI, OptimizationOptimJL, OptimizationIpopt
using Zygote

function surrogate_A(m,n,density,decay,min_val=1e-2)
    # makes a matrix A to resemble lin_invert sGX
    # (m,n) is size of A
    # density is fraction of columns of A that are nonzero
    # nonzero columns have a randomly determined row with value 1 and 
    #     log(min_val)/log(decay) decaying rows scattered throughout
    #     to provide cross-linking
    n_decaying_rows = Int(log(min_val)/log(decay))
    n_nonzero_cols = Int(density*n)
    randcols = randperm(n)
    nonzero_col_idxs = sort(randcols[1:n_nonzero_cols]) # cols that will have nonzero entries

    nnz = n_nonzero_cols*n_decaying_rows    # since each column has same nnz we know total nnz
    I = zeros(Int64,nnz)
    J = zeros(Int64,nnz)
    V = zeros(nnz)

    row_set = Set(1:m)     # set of all row indices
    miniset = Set()        # going to hold row indices we pop out of row_set when choosing nonzero entries

    idx = 1                # master index for sparse matrix
    for (j,cj) = enumerate(nonzero_col_idxs)
        for i = 1:n_decaying_rows
            ri = rand(row_set)
            pop!(row_set,ri)  # don't want to have different row entries clobber each other
            push!(miniset,ri) # and we want to keep track of which ones we got rid of so we can add them back in
            val = decay^(i-1)     # starts at 1 and decays by a factor of (decay) until it goes below (min_val)
            I[idx] = ri
            J[idx] = cj
            V[idx] = val
            idx +=1
        end
        row_set = union(row_set,miniset)
        miniset = Set()
    end
    return sparse(I,J,V,m,n)
end

# We want to solve min ||Ax-b||^2 s.t. xi > 0
m = 50
n = 100
d = 0.1
density = 0.5   # half the columns have entries
decay = 0.1     # secondary entries decay by a factor of 10 each time
min_val = 1e-2  # smallest secondary is 0.01
A = surrogate_A(m,n,density,decay,min_val)#sprandn(m,n,d)
dx = 1.0/n
a = 5/(m*dx)  # we're making x a permuted exponentially decaying vector. this is the decay rate.
x = [exp(-a*dx*i) for i = 1:n]
x = x[randperm(n)]   # scrombles
sigma = 1e-2
b = A*x .+ sigma*randn(m);

function ls_obj(x,p)
    # somewhat concerned calculating the square of A squares the condition number
    A,b,lambda = p
    tmp = b .- A*x
    return tmp'*tmp + lambda*x'*x     # L2 regularization
end

lambda = 1e-3
p = (A, b, lambda)
x0 = lsqr(A, b; damp=lambda)   # has negative numbers (out of domain) by default
# let's get rid of those o-o-d's
eps = 1e-10
function eps_relu(x,eps)
    return (x > eps ? x : eps)
end
for i = 1:length(x0)
    x0[i] = eps_relu(x0[i],eps)
end

cons(res,x,p) = (res.=x)       # constraint function -- here just x since we want x \geq 0

optf = OptimizationFunction(ls_obj, Optimization.AutoReverseDiff(compile=true), cons=cons)
prob = OptimizationProblem(optf, x0, p, lcons = zeros(n), ucons = Inf*ones(n))    # lower bound 0, upper bound Inf

sol = solve(prob, IpoptOptimizer(), verbose=5);

plot(x,label="truth"); plot!(sol.u,label="recon"); savefig("testfig.png")