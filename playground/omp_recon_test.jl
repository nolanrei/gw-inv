using Test
using Optim
using LinearAlgebra
using Distributions # Needed for Gamma distribution

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


println("Testing sharpen!")
reg_param = 1e-8 # Choose a small physical penalty for testing
nx = 0  
fill!(x_test, 0.0)

for iter = 1:2
    global nx = find_next_wave!(x_test, nx, os)
    #### DEBUG
    println("New x start \n", x_test[1:nx])
    
    # plot this test wave alongside the residual
    get_res!(x_test,nx,os)
    res = get_tmp(os.res,1.0)  # give me the float
    plot(reshape(os.fluxvec,(nz,2)),z,xlabel=["X residual (Pa)" "Y residual (Pa)"],ylabel="z (km)",label=nothing,layout=2)
    savefig("fluxvec$(iter).png")
    # somewhat hackily get forward propagated vec from recon = res - os.fluxvec
    recon = os.fluxvec .- res
    plot(reshape(recon,(nz,2)),z,xlabel=["X recon (Pa)" "Y recon (Pa)"],ylabel="z (km)",label=nothing,layout=2)
    savefig("recon$(iter).png")
    
    # 3. Expensive optimization step: Only run if the candidate passes the gatekeeper
    sharpen!(x_test, nx, reg_param, os)
end
nw_found = div(nx, 3)

#=
println("Running full multi-wave reconstruction...")
reg_param = 1e-8 # Choose a small physical penalty for testing
nx = find_measure!(x_test, reg_param, os)
nw_found = div(nx_found, 3)
=#

println("Reconstruction complete. Found $nw_found waves out of $ntst true waves.")

# Extract optimized parameter view
active_params = view(x_test, 1:nx)
println("Optimized Parameters Layout: ", active_params)

# --- RECONSTRUCT MODEL FOR PERFORMANCE METRICS ---
reconstructed_signal = zeros(2nz)
tmp_buf = zeros(2nz)
for i = 1:nw_found
    fill!(tmp_buf, 0.0)
    p_i = view(x_test, (3*(i-1)+1):(3*i))
    os.prop_AD!(tmp_buf, p_i, os.wav, os.col)
    reconstructed_signal .+= tmp_buf
end

# Calculate explained variance (R²) of the clean structural signal
residual_vs_truth = clean_signal .- reconstructed_signal
r2_score = 1.0 - (sum(abs2, residual_vs_truth) / sum(abs2, clean_signal))
println("Explained Variance (R² score relative to clean truth): $(round(r2_score * 100, digits=2))%")

plot(reshape(clean_signal,(nz,2)),z,layout=2);savefig("signal.png")
plot(reshape(reconstructed_signal,(nz,2)),z,layout=2);savefig("recon.png")
plot(reshape(residual_vs_truth,(nz,2)),z,layout=2);savefig("residual.png")

# --- START TESTS ---
@testset "Sliding Frank-Wolfe Architecture Recovery Test" begin
    # Check that it found a reasonable number of waves without running wild
    @test nw_found > 0
    @test nw_found <= ntst + 2 

    # Verify that the reconstructed parameters structurally explain the true profiles
    # R² > 0.80 ensures the optimizer actually tracked down the physical ridges
    @test r2_score > 0.80
end