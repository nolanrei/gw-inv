using Test
using Optim
using LinearAlgebra
using Distributions # Needed for Gamma distribution
using LaTeXStrings

include("omp.jl")
include("ensemble_accuracy.jl")

Random.seed!(501)
sz = size(wrf_ds[:U])
xi, yi, ti = rand(1:sz[1]), rand(1:sz[2]), rand(1:sz[4])

# Initialize Column Profile
col = ColumnProfile(wrf_ds[:U][xi,yi,:,ti], wrf_ds[:V][xi,yi,:,ti], wrf_ds[:N], wrf_ds[:rho], nz, dz)
kw = 2pi/50e3
src = 1
wav = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)

# plot windspeed
plot(cat(col.U, col.V, dims=2), z, xlabel=["U (m/s)" "V(m/s)"], ylabel="z (km)", label=nothing, layout=2)
savefig("windspeed.png")

# MAXITER limits how high the loop can search
max_nw = 20
x = zeros(3*max_nw)
nx = 0
os = OMPStruct(col, L81_AD_wrapper!, wav; max_nw=max_nw)

# Initialize fluxvec

# alternative -- manually specified waves
true_spec = [0.0, -35.0, 1e-3, 0.5pi, 11.5, 1e-3]
fwd!(os.fluxvec, true_spec, length(true_spec), os)

nxw = 15    # 5 synthetic waves
#true_spec = create_sample_fluxvec!(os.fluxvec, wrf_ds, xi, yi, ti, os, nxw; 
#    alpha=0.9, mn=50.0, beta=0.2, vm=2.0, cscale=4e-4, f0=1e-4)
#true_spec = retrieve_fluxvec_sample!(os.fluxvec, wrf_ds, xi, yi, ti, os)
if isnothing(true_spec)
    true_spec = [NaN NaN NaN]
end

## Choose these with care
# min_improvement -- increase to require a minimum improvement to accept. O(10)?
# complexity_weight -- 1.0 ==> solution length and recon quality equal value. Higher --> more emphasis on length
# noise variance -- variance of noise model. O(sigma^2) in additive noise, TODO -- more nuanced?
bic_opts = BICOptions(0.0, 1.0, 0.0)

# Solve
reg = 1e-8
ws = FullHessianOMPWorkspace(nz, max_nw)
fgh_closure = make_full_hessian_fgh(os, ws, reg, os.sc)

# Keep track of our guess for max remaining wave flux
f_guess = 0.25e-3
f_shrink = 0.5

# Automated solution to compare cutoffs
xau = zeros(3*max_nw)
#nxau = find_measure!(xau, reg, os, bic_opts, ws, fast_sharpen=true)

nwnew=2
nxnew=3*nwnew
for iter = 1:3
    global nx = find_next_wave!(x, nx, reg, os; nwnew=2, max_f_calls=10000, pop_size=50, stride=2)
    #nx = find_next_wave!(x, nx, reg, os; nwnew=1,max_f_calls=50000, pop_size=30)
    #### DEBUG
    println("New x start \n", x[1:nx])
    
    # plot this test wave alongside the residual
    get_res!(x,nx,os)
    res = get_tmp(os.res,1.0)  # give me the float
    # somewhat hackily get forward propagated vec from recon = res - os.fluxvec
    recon = os.fluxvec .- res
    plot(1e3*reshape(os.fluxvec,(nz,2)),z,xlabel=["X recon (mPa)" "Y recon (mPa)"],ylabel="z (km)",label="flux",layout=2)
    plot!(1e3*reshape(recon,(nz,2)),z,label="recon")
    savefig("recon$(iter)_t.png")

    #plot score landscape
# Debug plotting
get_res!(x,nx-3,os)
# 1. Define the grid
th_range = range(-π, π, length=100)
c_range  = range(0.0, 100.0, length=100)

# 2. Compute the score on the grid
# We use the positive score (maximize) for the visualization
score = [get_score([th, c], f_guess, os) for th in th_range, c in c_range]

# 3. Create the plot
p1 = surface(th_range, c_range, score', 
             xlabel="Angle (rad)", ylabel="Phase Speed (m/s)", zlabel="Score",
             title="Score Surface around Truth", color=:viridis)

p2 = heatmap(th_range, c_range, score', 
             xlabel="Angle (rad)", ylabel="Phase Speed (m/s)",
             title="Score Heatmap (Top View)", color=:viridis)
             
# Mark the found solution
scatter!(p2, x[1:3:nx-nxnew], x[2:3:nx-nxnew], color=:black, label="previous waves", markersize=5)
scatter!(p2, x[nx-nxnew+1:3:nx], x[nx-nxnew+2:3:nx], color=:white, label="found wave", markersize=5)

plot(p1, p2, layout=(1,2), size=(900, 400))
savefig("diagnostic_plot$(iter).png")

    # Update max wave flux guess
    global f_guess *= f_shrink
    
    # forward-backward sharpen
    nx = fb_sharpen_adaptive!(x, nx, os, bic_opts, fgh_closure; verbose=true, fast=true)
    
    # plot it again after sharpening
    get_res!(x,nx,os)
    res = get_tmp(os.res,1.0)  # give me the float
    # somewhat hackily get forward propagated vec from recon = res - os.fluxvec
    recon = os.fluxvec .- res
    plot(1e3*reshape(os.fluxvec,(nz,2)),z,xlabel=["X recon (mPa)" "Y recon (mPa)"],ylabel="z (km)",label="flux",layout=2)
    plot!(1e3*reshape(recon,(nz,2)),z,label="recon")
    savefig("recon$(iter)_s.png")

    rosmap = rosette_heatmap(x, nx, cmax=60)
    rosette!(rosmap, true_spec, length(true_spec))
    savefig("spectrum$(iter).png")
end

# Plot
reconstructed_signal = zeros(2nz)
tmp_buf = zeros(2nz)
for i = 1:div(nx,3)
    fill!(tmp_buf, 0.0)
    p_i = view(x, (3*(i-1)+1):(3*i))
    os.prop_AD!(tmp_buf, p_i, os.wav, os.col)
    reconstructed_signal .+= tmp_buf
end

plot(reshape(os.fluxvec,(nz,2)),z,
    xlabel=["X flux (Pa)" "Y flux (Pa)"],ylabel="z (km)",label="truth",layout=2);
plot!(reshape(reconstructed_signal,(nz,2)),z,label="recon",layout=2)
savefig("wrf_recon.png")

scatter(x[1:3:nx-2], x[2:3:nx-1], zcolor=x[3:3:nx], 
    xlabel="theta (radians)", ylabel="c (m/s)", 
    label="Recon", markersize=5)
savefig("wrf_spectrum.png")