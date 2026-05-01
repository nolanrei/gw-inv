## Tests how different L81_flux! and L81_exp_grad! are

recon_L81 = zeros(2nz)
recon_soft = zeros(2nz)
gradz = zeros(2nz)

for i = 1:nw
    L81_flux!(waves[i],col,exp(x[i]),recon_L81)
    L81_exp_grad!(waves[i],col,x[i],recon_soft,gradz)
end

plot(reshape(recon_L81,(nz,2)),z,xlabel="flux (mPa)",ylabel="z (km)",title=["X" "Y"],label="exact L81",layout=2)
plot!(reshape(recon_soft,(nz,2)),z,label="smoothed L81",layout=2)
savefig("Figures/L81_v_soft.png")