include("omp.jl")
include("spectrum_recovery.jl")
include("ensemble_accuracy.jl")

# You need to load in a dataset before
ds = wrf_ds
z = mean_existing_dims(ds,:z,:z)
nz = length(z)

src = 1

nsamp = 1500

ncases = 0
case_mats = []

# Noise model
z0 = -7 / (5/4)  # approximation of flux ~ rho ^ 5/4
zdep = exp.(z/z0)
sx = 20 .* zdep
sy = 5  .* zdep
noise_model = GPNoiseModel(z, sx, sy; l=1.5, lam=5.0, rxy=0.0, jitter=1e-8)

## Case 1: Null hypothesis

# ProportionalFLuxModel lets the distribution determine the mean amplitude so f0 is unused
f0 = (th, c) -> 1.0
total_flux = 1e-3      # 1mPa
spec_model1 = NWaveNoResample([])  # no waves -- no flux, just noise

# add to case_mats
push!(case_mats, (retrieve_fluxvec_from_model!, (spec_model1, noise_model)))
ncases += 1

### Run all test cases and save data
bic_opts = BICOptions(0.0,1.0,0.0)
for case = 1:ncases
    retrieve_func!, retrieve_args = case_mats[case]
    results = ensemble_accuracy(ds, retrieve_func!; calc_dist=false,
        src, reg=1e-8, nsamp, kw=2pi/1e4, bic_opts, retrieve_args, project=true)

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
    save_ensemble_to_netcdf("ensemble_null$(case).nc", results)
end