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

# Add noise and assign directly to your preallocated struct buffer
err = randn(2nz)
sigma = 0.05 * maximum(abs.(clean_signal)) 
os.fluxvec .= clean_signal .+ (sigma .* err)

# Allocate pre-sized storage vector for parameters as required by find_measure!
x_test = zeros(3 * max_nw)
nx = 0

println("Testing wave recovery...")
nx = find_next_wave!(x_test, nx, os)

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
score = [get_score([th, c], os) for th in th_range, c in c_range]

# 3. Create the plot
p1 = surface(th_range, c_range, score', 
             xlabel="Angle (rad)", ylabel="Phase Speed (m/s)", zlabel="Score",
             title="Score Surface around Truth", color=:viridis)

p2 = heatmap(th_range, c_range, score', 
             xlabel="Angle (rad)", ylabel="Phase Speed (m/s)",
             title="Score Heatmap (Top View)", color=:viridis)
             
# Mark the truth and the found solution
scatter!(p2, test_ths, test_cs, color=:red, label="Truth", markersize=5)
scatter!(p2, [x_test[1]], [x_test[2]], color=:white, label="found wave", markersize=5)

plot(p1, p2, layout=(1,2), size=(900, 400))
savefig("diagnostic_plot.png")
#-----------------------------------------------------------------------



# --- START TESTS ---
#=
@testset "OMP find_next_wave Recovery Test" begin
    # Create a fresh copy or reset residual for the test
    x_empty = Float64[]
    get_res!(x_empty, os)
    
    println("Testing wave recovery...")
    found_params = find_next_wave(x_empty, os, my_cdf)
    println(found_params)
    
    #@test found_params[1] ≈ true_th atol=0.05
    #@test found_params[2] ≈ true_c  atol=1.0
    #@test found_params[3] ≈ true_f  rtol=0.1
end
=#