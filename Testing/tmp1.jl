cnv = load_all_ensembles(2,prefix="Testing/ensemble_results_conv_c$(case)_");

idxs_near = findall(abs.(wrf_ds[:conv_dist_st]) .< 300)
# lazy (x,y,t,z) view, no copy
Up = PermutedDimsArray(wrf_ds[:U], (1, 2, 4, 3))
Vp = PermutedDimsArray(wrf_ds[:V], (1, 2, 4, 3))

umn = vec(mean(Up[idxs_near, :]; dims=1))   # length nz
vmn = vec(mean(Vp[idxs_near, :]; dims=1))

plot(umn,vmn,xlabel="U (m/s)",ylabel="V (m/s)",zlabel="z (km)",line_z=z);
savefig("Testing/c$(case)_winds_2d.png")

snr = condense_spectra(cnv[1].spectra,cnv[1].nxs);
sfr = condense_spectra(cnv[2].spectra,cnv[2].nxs);

rnr = rosette_heatmap(snr,length(snr), cmax=40);
savefig(rnr,"Testing/ros_c$(case)_unrot_near.png")
rfr = rosette_heatmap(sfr,length(sfr), cmax=40);
savefig(rfr,"Testing/ros_c$(case)_unrot_far.png")