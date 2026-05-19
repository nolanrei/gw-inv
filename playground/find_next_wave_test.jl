using Test
using Optim
using LinearAlgebra

include("omp.jl")

# --- GLOBAL SETUP FOR THE TEST FILE ---
# Seed the RNG so the profile is the same every time you run the test
#Random.seed!(42) 

sz = size(vars["Up"][1])
xi, yi, ti = rand(1:sz[1]), rand(1:sz[2]), rand(1:sz[4])

# Use the exact lambda that worked in your manual test
λ = 1/1e-3  
my_cdf = x -> 1.0 - exp(-λ * x)

# Ensure nz and dz are defined/accessible here
col = ColumnProfile(vars["Up"][1][xi,yi,:,ti], vars["Vp"][1][xi,yi,:,ti], vars["N"], vars["rho"], nz, dz)
kw = 2pi/5e5
src = 4
wav = c_to_wave(0.0, [1.0;0.0], kw, 0.0, src)
os = OMPStruct(col, L81_AD_wrapper!, wav)

# Synthetic Truth
ntst = 6
cm = 0.0
cp = 60.0
test_cs = cm .+(cp-cm)*rand(ntst)

alpha=2
ref_flux = 1e-3        # 1 mPa
mnfx = 0.1*ref_flux    # mean flux
test_fxs = rand(Gamma(alpha,mnfx/alpha),ntst)

test_ths = 2pi*rand(ntst).-pi

kw = 2pi/5e5
src = 4

# make rhs: all of the test waves, plus error
target_flux = zeros(2nz)
for i = 1:ntst
    os.prop_AD!(target_flux,[test_ths[i], test_cs[i], test_fxs[i]],os.wav,os.col)
    os.fluxvec .+= target_flux
end
err = randn(2nz)
sigma = 0.05*maximum(abs.(fluxvec))   # error scales with signal
os.fluxvec .+= sigma*err

x_empty = Float64[]
get_res!(x_empty, os)

println("Testing wave recovery...")
found_params = find_next_wave(x_empty, os, my_cdf)
println(found_params)

# Sort the waves in order of flux magnitude so we can rank them by importance
ix = sortperm(test_fxs)
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
z = [get_score([th, c], os, my_cdf) for th in th_range, c in c_range]
#z = [get_score_ugly([th, c], os) for th in th_range, c in c_range]

# 3. Create the plot
p1 = surface(th_range, c_range, z', 
             xlabel="Angle (rad)", ylabel="Phase Speed (m/s)", zlabel="Score",
             title="Score Surface around Truth", color=:viridis)

p2 = heatmap(th_range, c_range, z', 
             xlabel="Angle (rad)", ylabel="Phase Speed (m/s)",
             title="Score Heatmap (Top View)", color=:viridis)
             
# Mark the truth and the found solution
scatter!(p2, test_ths, test_cs, color=:red, label="Truth", markersize=5)
scatter!(p2, [found_params[1]], [found_params[2]], color=:white, label="found wave", markersize=5)

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