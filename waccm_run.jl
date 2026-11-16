
sz = size(U)
xi = rand(1:sz[1])
yi = rand(1:sz[2])
ti = 1
te = sz[4]
zb = 24    # 10km
ze = 51    # 45km

u = U[xi,yi,ze:-1:zb,ti:te]
v = V[xi,yi,ze:-1:zb,ti:te]
dx = 86400*ugw[xi,yi,ze:-1:zb,ti:te]    # put it into m/s/day
dy = 86400*vgw[xi,yi,ze:-1:zb,ti:te]
T = 3*(1:te-ti+1)
z = 1e-3*Z[xi,yi,ze:-1:zb,ti]  #put it into km

# find which times have nonzero drag and select one to plot the angle of
drag_presence = dropdims(sum(abs.(dx),dims=1) .+ sum(abs.(dy),dims=1), dims=1)
trand = rand(findall(drag_presence .> 0.0))

plot(atan.(dy[:,trand],dx[:,trand]),z,xlabel="angle (radians)",ylabel="z (km)",title="angle of WACCM drag",label=nothing)
savefig("Figures/waccm_angles.png")

heatmap(T,z,u,xlabel="time (hrs)",ylabel="z (km)",title="U (m/s)")
savefig("Figures/hovmoller_u_waccm.png")

heatmap(T,z,dx,xlabel="time (hrs)",ylabel="z (km)",title="X drag (m/s/day)")
savefig("Figures/hovmoller_dx_waccm.png")

heatmap(T,z,dy,xlabel="time (hrs)",ylabel="z (km)",title="Y drag (m/s/day)")
savefig("Figures/hovmoller_dy_waccm.png")