## Tests whether I'm mapping drag to flux correctly

# Generate flux and drag from L81_flux and L81_drag, respectively
# May well NOT be the same (there's an inconsistency in the formulas used for breaking)
fluxes = spectrum_to_momdep(L81_flux!, waves, col, exp.(x))
drags = spectrum_to_momdep(L81!, waves, col, exp.(x))

# fluxvec[lvl] = flux at *top* of cell
fluxvec = zeros(2nz)
dflxx = dx[:,t].*dz.*rho[xi,yi,ze:-1:zb,t] # amount of x flux deposited in this cell
dflxy = dy[:,t].*dz.*rho[xi,yi,ze:-1:zb,t]
fluxvec[nz] = 0.0
fluxvec[2nz] = 0.0
for i = nz-1:-1:1
    fluxvec[i]    = fluxvec[i+1]    - dflxx[i+1]
    fluxvec[i+nz] = fluxvec[i+1+nz] - dflxy[i+1]
end

plot(1e3*reshape(fluxvec,(nz,2)),z,layout=2,label=["fx" "fy"])
savefig("Figures/WACCM/Reconstructions/fluxvec.png")

begin
plot(hcat(dx[:,t],dy[:,t]),z,xlabel="drag (m/s^2)",ylabel="z (km)",label=["X" "Y"],layout=2)
savefig("Figures/WACCM/Reconstructions/drags.png")
end