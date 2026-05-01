# Meant to be run after loading in data with waccm_setup
include("tron_defs.jl")

sz = size(U)
xi = rand(1:sz[1]) #150
yi = rand(1:sz[2]) #9
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

# find which times have nonzero drag and select one to plot the angle of
drag_presence = dropdims(sum(abs.(dx),dims=1) .+ sum(abs.(dy),dims=1), dims=1)
dragyes = findall(drag_presence .> 0.0)
t = rand(dragyes)

dz = Z[xi,yi,ze:-1:zb,t] .- Z[xi,yi,ze+1:-1:zb+1,t]
col = ColumnProfile(u[:,t],v[:,t],N[xi,yi,ze:-1:zb,t],rho[xi,yi,ze:-1:zb,t],ze-zb+1,dz)

## Find the prevailing direction of the drag -- this is the direction we should use waves in
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

## Set up waves
cmin = -60
cmax = 60
cstride = 1
ndirs = 1   
src = 1         # source level = 10km
wavespds = Array(cmin:cstride:cmax)
kw = 2pi/5e5
nws = length(wavespds)
ref_flux = 1e-3
nw = nws*ndirs

waves = Array{WaveProfile,1}(undef, nw)
for i = 1:length(wavespds)
    waves[i] = c_to_wave(wavespds[i],dir,kw,ref_flux,src)
end

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

plot(1e3*reshape(fluxvec,(nz,2)),z,layout=2,label=["fx" "fy"])
savefig("~/work/minimization/Figures/WACCM/Reconstructions/fluxvec.png")

begin
plot(hcat(dx[:,t],dy[:,t]),z,xlabel="drag (m/s^2)",ylabel="z (km)",label=["X" "Y"],layout=2)
savefig("~/work/minimization/Figures/WACCM/Reconstructions/drags.png")
end

begin
plot((dir[1]*u[:,t]).+(dir[2]*v[:,t]),z,xlabel="windspeed (m/s)",ylabel="z (km)",label=nothing,title="windspeed in direction of GWs")
savefig("~/work/minimization/Figures/WACCM/Reconstructions/windspeed.png")
end

### Solve using JSOSolvers.jl
nlp = GWFit(waves, col, fluxvec; lambda=1e-6, wndw=nothing)
solver = TronSolver(nlp);
stats = solve!(solver, nlp; verbose=5)

## Plot
plot(wavespds, 1e3*stats.solution, xlabel="phase speed (m/s)", ylabel="flux (mPa)", label=nothing)
savefig(@sprintf("~/work/minimization/Figures/WACCM/Reconstructions/lat=%.2f,lon=%.2f,t=%d.png",lat[yb+yi],lon[xi],tb+t))

begin
recon_momdep = spectrum_to_momdep(L81_flux!,waves,col,stats.solution)
plot(1e3*reshape(fluxvec,(nz,2)),z,layout=2,xlabel="flux (mPa)",ylabel="z (km)",label=["fx" "fy"])
plot!(1e3*reshape(recon_momdep,(nz,2)),z,layout=2,label=["x recon" "y recon"])
savefig(@sprintf("~/work/minimization/Figures/WACCM/Reconstructions/lat=%.2f,lon=%.2f,t=%d_flux.png",lat[yb+yi],lon[xi],tb+t))
end