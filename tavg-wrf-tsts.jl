include("omp.jl")
include("spectrum_recovery.jl")
include("ensemble_accuracy.jl")

# You need to load in a dataset before
ds = wrf_ds
z = mean_existing_dims(ds,:z,:z)
nz = length(z)

src = 12  # (16km launch level)

nsamp = 1500

ncases = 0
case_mats = []

dts = [2 6 12]

for i = 1:length(dts)
    push!(case_mats, (retrieve_fluxvec_avg!, (dt=dts[i])))
    global ncases += 1
end

zis = src:nz

### Run all test cases and save data
bic_opts = BICOptions(0.0,1.0,0.0)
for case = 1:ncases
    retrieve_func!, retrieve_args = case_mats[case]
    results = ensemble_accuracy(ds, retrieve_func!; 
        src, reg=1e-8, nsamp, kw=2pi/1e4, bic_opts, retrieve_args, project=true, zis=zis)

    # measure distance from empirical reconstructed distribution to samples drawn from true distrib
    if any(.!isnan.(results.known_spectra))
        emp_truth = [view(results.known_spectra, results.ks_offsets[i]+1 : results.ks_offsets[i+1]) 
                    for i = 1:nsamp if results.ks_offsets[i+1] > results.ks_offsets[i]]
        emp_recon = [view(results.spectra, i, 1:results.nxs[i]) for i = 1:nsamp if results.nxs[i] > 0]
        if length(results.known_spectra) < 1e5
            energy = energy_distance(emp_recon, emp_truth, earth_mover_dist)
            results = merge(results, (distribution_distance = [energy],))
        end
    end
    save_ensemble_to_netcdf("wrf_tavg_ens$(case).nc", results)
end