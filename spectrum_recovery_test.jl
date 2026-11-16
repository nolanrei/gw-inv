include("omp.jl")
include("spectrum_recovery.jl")
include("ensemble_accuracy.jl")


# You need to load in a dataset before
ds = wrf_ds
z = mean_existing_dims(ds,:z,:z)
nz = length(z)

src = 4

nsamp = 400
function make_gaussian_deltas(nds)
    deltas = zeros(3*nds)
    cdev = 30
    massmn = 1/nds
    masstot = 0.0
    for i = 1:nds
        cvec = cdev*randn(2)
        th = atan(cvec[2], cvec[1])
        c = hypot(cvec[1], cvec[2])
        mass = massmn*rand()
        deltas[3i-2] = th
        deltas[3i-1] = c
        deltas[3i]   = masstot
        masstot += mass       # important here. needs to be (0, mass1, mass1+mass2, ...)
    end
    # normalize the cdf to add up to 1
    deltas[3:3:3nds] ./= masstot
    return deltas
end

function stdev_spectrum(spec,nxs)
    # I think we ignore the fluxes?
    ns,mnx = size(spec)
    nw = div(sum(nxs),3)     # total number of waves
    x = zeros(nw)
    y = zeros(nw)
    idx = 1
    for i = 1:ns
        for j = 1:div(nxs[i],3)
            th = spec[i,3j-2]
            c = min(200, spec[i,3j-1])
            x[idx] = c*cos(th)
            y[idx] = c*sin(th)
            idx += 1
        end
    end
    return (std(x), std(y))
end

### Case 1 -- Deltas
#=
nds = 6     # number of spikes in test spectrum
deltas = make_gaussian_deltas(nds)
locP = DeltaLocProposal(deltas)

num_wav_dist = Dirac(nds)     # there will always be nds waves but they won't always be the same nds waves like this
#num_wav_dist = Poisson(nds)   # a Poisson distrib over number of waves with mean nds

# ProportionalFLuxModel lets the distribution determine the mean amplitude so f0 is unused
f0 = (th, c) -> 1.0
total_flux = 1e-3      # 1mPa
spec_model = ProportionalFluxModel(locP, f0, total_flux, num_wav_dist)
=#

### Case 2 -- Gaussian

nwav = 5
ctr = (10.0, 0.0)
sdev = (60.0, 60.0)
gx = Normal(ctr[1], sdev[1])
gy = Normal(ctr[2], sdev[2])
locP = IndependentCartesianLocProposal(gx, gy)
num_wav_dist = Poisson(nwav)   # a Poisson distrib over number of waves with mean nwav

# ProportionalFLuxModel lets the distribution determine the mean amplitude so f0 is unused
f0 = (th, c) -> 1.0
total_flux = 1e-3      # 1mPa
spec_model = ProportionalFluxModel(locP, f0, total_flux, num_wav_dist)


noise_model = WhiteNoiseModel(0.05)         # 0.05 ==> 5% additive flux error

retrieve_args = (spec_model, noise_model)
bic_opts = BICOptions(0.0,0.0,0.0)    # don't care about number of waves
results = ensemble_accuracy(ds, retrieve_fluxvec_from_model!; src, reg=1e-8, nsamp, kw=2pi/5e3, bic_opts, retrieve_args);

save_ensemble_to_netcdf("ensemble_results_synth.nc", results)


rms_abs_err_z = results.rms_abs_err_z
rms_rel_err_z = results.rms_rel_err_z
spectra = results.spectra
known_spectra = results.known_spectra
ks_offsets = results.ks_offsets
errors = results.errors
fluxes = results.fluxes
nxs = results.nxs
xs = results.xs
ys = results.ys
ts = results.ts

ks_sq, nxks = unflatten_square(known_spectra, ks_offsets)

flux_abs = dropdims(mean(abs2,fluxes,dims=(1)),dims=1)
flux_abs = sqrt.(flux_abs[1:nz] .+ flux_abs[nz+1:2nz])

begin
plot(rms_abs_err_z[src:end],z[src:end], xlabel="flux abs. error", ylabel="z (km)", label="error")
plot!(flux_abs[src:end], z[src:end],label="flux")
savefig("relerr.png")
end

grid = -100.0:1.0:100.0

# if your flux model has a specified continuous mean f0 plot it here
#target_density = [target_density_cx_cy(spec_model.f0, cx, cy) for cx, cy in grid, grid]
begin
avg_known = ensemble_mean_density(ks_sq, nxks, grid; ep=0.3)
pk = heatmap(grid, grid, avg_known, xlabel="cx (m/s)", ylabel="cy (m/s)")
savefig(pk, "spectrum_recon_known.png")

avg_recov = ensemble_mean_density(spectra, nxs, grid; ep=0.3)
pk = heatmap(grid, grid, avg_recov, xlabel="cx (m/s)", ylabel="cy (m/s)")
savefig(pk, "spectrum_recon_invert.png")
end