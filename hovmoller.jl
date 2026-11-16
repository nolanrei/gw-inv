sz = size(vars["Up"][1])
nT = 96  # one day in 15min chunks
xi = rand(1:sz[1])
yi = rand(1:sz[2])
ti = 1 #rand(1:sz[4]-nT+1)
te = sz[4] #ti+nT-1

dT = 16   # 24*15min = 6hrs, a lower bound for GW propagation time over col

nfilt2 = 3*dT
xx = (-nfilt2+1:nfilt2)/dT
filter = exp.(-xx.^2)
filter ./= sum(filter)

z = 10:0.5:65
zc = 70

u = vars["Up"][1][xi,yi,:,ti:te]
v = vars["Vp"][1][xi,yi,:,ti:te]
T = 0.25*(1:te-ti+1)

heatmap(T,z[1:zc],u[1:zc,:],xlabel="time (hrs)",ylabel="z (km)",title="U (m/s)")
savefig("Figures/hovmoller_u.png")

uw = vars["UW"][1][xi,yi,:,ti:te].*vars["rho"]
vw = vars["VW"][1][xi,yi,:,ti:te].*vars["rho"]

drx = 86400*vars["DX"][1][xi,yi,:,ti:te];
dry = 86400*vars["DY"][1][xi,yi,:,ti:te];

drx_f = conv(drx,filter;dim=2)
dry_f = conv(dry,filter;dim=2)

heatmap(T,z[1:zc],uw[1:zc,:],xlabel="time (hrs)",ylabel="z (km)",title="UW (m^2/s^2)")
savefig("Figures/hovmoller_uw.png")

heatmap(T,z[1:zc],vw[1:zc,:],xlabel="time (hrs)",ylabel="z (km)",title="VW (m^2/s^2)")
savefig("Figures/hovmoller_vw.png")

heatmap(T[1:dT:end],vars["oz_dx"][1:zc],drx_f[1:zc,1:dT:end],xlabel="time (hrs)",ylabel="z (km)",title="X drag (m/s/day)")
savefig("Figures/hovmoller_dx.png")

heatmap(T[1:dT:end],vars["oz_dx"][1:zc],dry_f[1:zc,1:dT:end],xlabel="time (hrs)",ylabel="z (km)",title="Y drag (m/s/day)")
savefig("Figures/hovmoller_dy.png")