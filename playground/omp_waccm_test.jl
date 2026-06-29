using Distributions # Needed for Gamma distribution
using LaTeXStrings

include("omp.jl")
closeall()

sz = size(U)
xi = 94 #rand(1:sz[1])   # x,y,t = 94,7,21 for known hard sideswipe case (need tb=800 in waccm_setup)
yi = 7 #rand(1:sz[2])
ti = 1
te = sz[4]
zb = 16    # 67km   (24    # 45km)
ze = 50    # 10km
nz = ze-zb+1

u = U[xi,yi,ze:-1:zb,ti:te]
v = V[xi,yi,ze:-1:zb,ti:te]
dx = ugw[xi,yi,ze:-1:zb,ti:te]
dy = vgw[xi,yi,ze:-1:zb,ti:te]
T = 3*(1:te-ti+1)
z = 1e-3*Z[xi,yi,ze:-1:zb,ti]  #put it into km

# find which times have nonzero drag and select one
drag_presence = dropdims(sum(abs.(dx),dims=1) .+ sum(abs.(dy),dims=1), dims=1)
dragyes = findall(drag_presence .> 1e-8)
t = 21 #rand(dragyes)

dz = Z[xi,yi,ze:-1:zb,t] .- Z[xi,yi,ze+1:-1:zb+1,t]
col = ColumnProfile(u[:,t],v[:,t],N[xi,yi,ze:-1:zb,t],rho[xi,yi,ze:-1:zb,t],ze-zb+1,dz)

## Turn flux into drag
# fluxvec[lvl] = flux at *top* of cell
fluxvec = zeros(2nz)
dflxx = dx[:,t].*dz.*rho[xi,yi,ze:-1:zb,t] # amount of x flux deposited in this cell
dflxy = dy[:,t].*dz.*rho[xi,yi,ze:-1:zb,t]
fluxvec[nz] = 0.0
fluxvec[2nz] = 0.0
for i = nz-1:-1:1
    fluxvec[i]    = fluxvec[i+1]    + dflxx[i+1]  # minus sign from -1/rho df/dz cancels out minus sign from integrating down
    fluxvec[i+nz] = fluxvec[i+1+nz] + dflxy[i+1]
end

## Find the prevailing direction of the drag -- this is for plotting only
# With high probability the mean drag is not exactly 0, so find mean direction and then
# take the absolute value along this direction, then take the mean again -- this should
# minimize cancellation for mostly 1D data
dir_tmp = zeros(2)
dir_tmp[1] = mean(dx[:,t]); dir_tmp[2] = mean(dy[:,t])
dir_tmp ./= norm(dir_tmp)
dir = zeros(2)
for i = 1:size(dx,1)
    sgn = sign(dir_tmp[1]*dx[i,t] + dir_tmp[2]*dy[i,t])  # is this level in same direction as dir_tmp?
    dir[1] += sgn*dx[i,t]
    dir[2] += sgn*dy[i,t]
    # should have the effect of an absolute value along dir
end
dir ./= norm(dir)

## Set up OMPStruct
max_nw = 20
x = zeros(3*max_nw)
nx = 0

kw = 2pi/5e3
src = 1
wav = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)
os = OMPStruct(col, L81_AD_wrapper!, wav; max_nw, fluxvec)
#=
#############################################################################################
## Special Testing Code -- launch two known half-breaking waves in a known sideswipe profile
#############################################################################################
ntst = 2
th = atan(dir[2],dir[1])
test_ths = [th,th+pi]
test_cs = [20.0, 20.0]
test_fxs = [1e-2, 1e-2] # 10mPa

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
#############################################################################################
=#
## Reconstruct
reg = 1e-8
#nx = find_measure!(x,reg,os)
ws = FullHessianOMPWorkspace(nz, max_nw)
fgh_closure = make_full_hessian_fgh(os, ws, reg, os.sc)

# Keep track of our guess for max remaining wave flux
f_guess = 0.25e-3
f_shrink = 0.5

nwnew=2
nxnew=3*nwnew
for iter = 1:5
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
    nx = fb_sharpen_adaptive!(x, nx, os, fgh_closure; reltol=0.0)
    
    # plot it again after sharpening
    get_res!(x,nx,os)
    res = get_tmp(os.res,1.0)  # give me the float
    # somewhat hackily get forward propagated vec from recon = res - os.fluxvec
    recon = os.fluxvec .- res
    plot(1e3*reshape(os.fluxvec,(nz,2)),z,xlabel=["X recon (mPa)" "Y recon (mPa)"],ylabel="z (km)",label="flux",layout=2)
    plot!(1e3*reshape(recon,(nz,2)),z,label="recon")
    savefig("recon$(iter)_s.png")

    scatter(x[1:3:nx-2], x[2:3:nx-1], xlims=[0.0,2pi],ylims=[0.0,100.0], zcolor=x[3:3:nx],
        label="sharpened soln", markersize=5)
    savefig("spectrum$(iter).png")
end


## Plot
plot(1e3*reshape(fluxvec,(nz,2)),z,layout=2,label=["fx" "fy"])
savefig("~/work/gw-inv/Figures/WACCM/Reconstructions/fluxvec.png")

plot(hcat(dx[:,t],dy[:,t]),z,xlabel="drag (m/s^2)",ylabel="z (km)",label=["X" "Y"],layout=2)
savefig("~/work/gw-inv/Figures/WACCM/Reconstructions/drags.png")

plot((dir[1]*u[:,t]).+(dir[2]*v[:,t]),z,xlabel="windspeed (m/s)",ylabel="z (km)",label=nothing,title="windspeed in direction of GWs")
savefig("~/work/gw-inv/Figures/WACCM/Reconstructions/windspeed.png")

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
savefig("waccm_recon.png")

scatter(x[1:3:nx-2], x[2:3:nx-1], zcolor=x[3:3:nx], 
    xlabel="theta (radians)", ylabel="c (m/s)", 
    label="Recon", markersize=5)
savefig("waccm_spectrum.png")
savefig(@sprintf("~/work/gw-inv/Figures/WACCM/Reconstructions/OMP/lat=%.2f,lon=%.2f,t=%d_flux.png",lat[yb+yi],lon[xi],tb+t))