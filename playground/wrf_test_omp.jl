using Test
using Optim
using LinearAlgebra
using Distributions # Needed for Gamma distribution
using LaTeXStrings

include("omp.jl")

sz = size(vars["Up"][1])
xi, yi, ti = rand(1:sz[1]), rand(1:sz[2]), rand(1:sz[4])

# Initialize Column Profile
col = ColumnProfile(vars["Up"][1][xi,yi,:,ti], vars["Vp"][1][xi,yi,:,ti], vars["N"], vars["rho"], nz, dz)
kw = 2pi/5e4
src = 1
wav = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)

# MAXITER limits how high the loop can search
max_nw = 20
x = zeros(3*max_nw)
os = OMPStruct(col, L81_AD_wrapper!, wav; max_nw=max_nw)

# Initialize fluxvec
dt = 4
os.fluxvec = dropdims(cat(mean(vars["UW"][1][xi,yi,:,ti-dt:ti+dt],dims=2).*vars["rho"], mean(vars["VW"][1][xi,yi,:,ti-dt:ti+dt],dims=2).*vars["rho"], dims=1),dims=2)

# Solve
reg = 1e-6
nx = 0
#nx = find_measure!(x,reg,os)

# Keep track of our guess for max remaining wave flux
f_guess = 0.25e-3
f_shrink = 0.5
for iter = 1:5
    global nx = find_next_wave!(x, nx, reg, os)
    #### DEBUG
    println("New x start \n", x[1:nx])
    
    # plot this test wave alongside the residual
    get_res!(x,nx,os)
    res = get_tmp(os.res,1.0)  # give me the float
    # somewhat hackily get forward propagated vec from recon = res - os.fluxvec
    recon = os.fluxvec .- res
    plot(reshape(os.fluxvec,(nz,2)),z,xlabel=["X recon (Pa)" "Y recon (Pa)"],ylabel="z (km)",label="flux",layout=2)
    plot!(reshape(recon,(nz,2)),z,label="recon")
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
scatter!(p2, x[1:3:3*iter-5], x[2:3:3*iter-4], color=:black, label="previous waves", markersize=5)
scatter!(p2, [x[3*iter-2]], [x[3*iter-1]], color=:white, label="found wave", markersize=5)

plot(p1, p2, layout=(1,2), size=(900, 400))
savefig("diagnostic_plot$(iter).png")

    # Update max wave flux guess
    global f_guess *= f_shrink
    # 3. Expensive optimization step: Only run if the candidate passes the gatekeeper
    sharpen!(x, nx, reg, os)

    # plot it again after sharpening
    get_res!(x,nx,os)
    res = get_tmp(os.res,1.0)  # give me the float
    # somewhat hackily get forward propagated vec from recon = res - os.fluxvec
    recon = os.fluxvec .- res
    plot(reshape(os.fluxvec,(nz,2)),z,xlabel=["X recon (Pa)" "Y recon (Pa)"],ylabel="z (km)",label="flux",layout=2)
    plot!(reshape(recon,(nz,2)),z,label="recon")
    savefig("recon$(iter)_s.png")

    scatter(x[1:3:3*iter-2], x[2:3:3*iter-1], color=:black, label="sharpened soln", markersize=5)
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