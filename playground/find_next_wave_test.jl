using Test
using Optim
using LinearAlgebra

include("omp.jl")

# --- GLOBAL SETUP FOR THE TEST FILE ---
Random.seed!(86) 

sz = size(vars["Up"][1])
xi, yi, ti = rand(1:sz[1]), rand(1:sz[2]), rand(1:sz[4])

# Distribution CDF parameterization
λ = 1 / 1e-3  
my_cdf = x -> 1.0 - exp(-λ * x)

# Initialize Column Profile
col = ColumnProfile(vars["Up"][1][xi,yi,:,ti], vars["Vp"][1][xi,yi,:,ti], vars["N"], vars["rho"], nz, dz)
kw = 2pi/5e5
src = 4
wav = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)

# MAXITER limits how high the loop can search
max_nw = 20
os = OMPStruct(col, L81_AD_wrapper!, my_cdf, wav; max_nw=max_nw)

# Synthetic Truth Generation
ntst = 6
cm = 0.0
cp = 60.0
test_cs = cm .+ (cp - cm) * rand(ntst)

alpha = 2
ref_flux = 1e-3        # 1 mPa
mnfx = 0.1 * ref_flux  # mean flux
test_fxs = rand(Gamma(alpha, mnfx / alpha), ntst)
test_ths = 2pi * rand(ntst) .- pi

# Build RHS (Clean synthetic signal)
target_flux = zeros(2nz)
clean_signal = zeros(2nz)
for i = 1:ntst
    fill!(target_flux, 0.0)
    os.prop_AD!(target_flux, [test_ths[i], test_cs[i], test_fxs[i]], os.wav, os.col)
    clean_signal .+= target_flux
end

# Add noise and assign directly to preallocated struct buffer
err = randn(2nz)
sigma = 0.05 * maximum(abs.(clean_signal)) 
os.fluxvec .= clean_signal .+ (sigma .* err)

# Allocate pre-sized storage vector for parameters as required by find_measure!
x_test = zeros(3 * max_nw)
nx = 0

println("Testing wave recovery...")

function find_next_wave_debug!(x::AbstractVector, nx::Int64, reg::Float64, os::OMPStruct)
    # In the OMP/sliding Frank-Wolfe formulation, finds the next 
    # wave to add to our support
    tmp = get_tmp(os.tmp_cache, x)
    p_cur = zeros(3)

    # Get residual
    get_res!(x,nx,os)
    res = get_tmp(os.res, x)

    # Starting wave guess
    n_sobol_samples = 100
    s = skip(SobolSeq([-π, 0.0], [π, 100.0]), n_sobol_samples)
    p0 = view(p_cur,1:2)
    pbest = view(x,(nx+1):(nx+3))
    cbest = Inf
    sobol_pts = zeros(n_sobol_samples,3)
    sobol_vals = zeros(n_sobol_samples)
    for iter = 1:n_sobol_samples
        next!(s,p0)
        sobol_pts[iter,1:2] .= p0
        resnorm = let os=os, tmp=tmp, res=res, reg=reg, p_cur=p_cur
            f -> begin
                p_cur[3] = exp(f)     # p0 is the first two elements of tmp2 already
                fill!(tmp,0)
                os.prop_AD!(tmp, p_cur, os.wav, os.col)
                out = 0.0
                for i = eachindex(tmp)
                    out += (res[i] - tmp[i])^2
                end
                return 0.5*out + reg*p_cur[3]
            end
        end
    
        # Minimize
        l = log(1e-8)
        u = log(1e-2)   # maximum of 10 mPa (still stupid large)
        f_results = optimize(resnorm, l, u, Brent())

        sobol_vals[iter] = Optim.minimum(f_results)
        sobol_pts[iter,3] = Optim.minimizer(f_results)

        fmin = Optim.minimum(f_results)
        if fmin < cbest
            pbest .= p0[1], p0[2], exp(Optim.minimizer(f_results))
            cbest = fmin
        end
    end
     
    nx += 3
    return nx, sobol_pts, sobol_vals
end

nx,s_p_d,s_v_d = find_next_wave_debug!(x_test, nx, reg_param, os)

# Sort the waves in order of flux magnitude so we can rank them by importance
ix = sortperm(test_fxs,rev=true)
test_fxs = test_fxs[ix]
test_cs = test_cs[ix]
test_ths = test_ths[ix]

#----------------------------------------------------------------------
# Debug plotting
# 1. Define the grid
th_range = range(-π, π, length=100)
c_range  = range(0.0, 100.0, length=100)

# 2. Compute the score on the grid
# We use the positive score (maximize) for the visualization
p_cur = zeros(3)
resnorm = let os=os, tmp=get_tmp(os.tmp_cache,1.0), res=get_tmp(os.res,1.0), reg=reg_param, p_cur=p_cur
    f -> begin
        p_cur[3] = exp(f)     # p0 is the first two elements of tmp2 already
        fill!(tmp,0)
        os.prop_AD!(tmp, p_cur, os.wav, os.col)
        out = 0.0
        for i = eachindex(tmp)
            out += (res[i] - tmp[i])^2
        end
        return 0.5*out + reg*p_cur[3]
    end
end
score = zeros(length(th_range),length(c_range))
l = log(1e-8)
u = log(1e-2)   # maximum of 10 mPa (still stupid large)
for (i,th) in enumerate(th_range)
    for (j,c) in enumerate(c_range)
        p_cur[1],p_cur[2] = th,c
        f_results = optimize(resnorm, l, u, Brent())
        score[i,j] = Optim.minimum(f_results)
    end
end

# 3. Create the plot
p1 = surface(th_range, c_range, score', 
             xlabel="Angle (rad)", ylabel="Phase Speed (m/s)", zlabel="Score",
             title="Score Surface around Truth", color=:viridis)

p2 = heatmap(th_range, c_range, score', 
             xlabel="Angle (rad)", ylabel="Phase Speed (m/s)",
             title="Score Heatmap (Top View)", color=:viridis)

# 4. Sobol points
n_sobol_samples = 100
s = skip(SobolSeq([-π, 0.0], [π, 100.0]), n_sobol_samples)
sobol_pts = zeros(n_sobol_samples,3)
sobol_vals = zeros(n_sobol_samples)
for i = 1:n_sobol_samples
    next!(s,view(sobol_pts,i,1:2))
    p_cur[1:2] .= sobol_pts[i,1:2]
    f_results = optimize(resnorm, l, u, Brent())
    sobol_vals[i] = Optim.minimum(f_results)
    sobol_pts[i,3] = Optim.minimizer(f_results)
end

scatter(sobol_pts[:,1],sobol_pts[:,2],sobol_vals,zcolor=sobol_vals);
savefig("scatter_diagnostic.png")
             
# Mark the truth and the found solution
scatter!(p2, test_ths, test_cs, color=:red, label="Truth", markersize=5)
scatter!(p2, sobol_pts[:,1], sobol_pts[:,2], color=:black, label="Sobol pts", markersize=5)
scatter!(p2, [x_test[nx-2]], [x_test[nx-1]], color=:white, label="found wave", markersize=5)

plot(p1, p2, layout=(1,2), size=(900, 400))
savefig("diagnostic_plot.png")
#-----------------------------------------------------------------------
n_1d_samp = 100
score_samp = zeros(n_1d_samp)
f_samp = 10.0.^range(-7,-2.8,n_1d_samp)
p_cur[1:2] .= x_test[1:2]
for i = 1:n_1d_samp
    score_samp[i] = resnorm(f_samp[i])
end
plot(1e3*f_samp, score_samp,xlabel="flux (mPa)",ylabel="cost (Pa^2)",label=nothing)
savefig("score_v_f.png")