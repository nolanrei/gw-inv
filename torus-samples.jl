include("omp.jl")
include("spectrum_recovery.jl")
include("ensemble_accuracy.jl")

# You need to load in a dataset before
ds = wrf_ds
z = mean_existing_dims(ds,:z,:z)
nz = length(z)

src = 1

nsamp = 1000
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

function flux_isotropic(k, th, u, v; k0 = 2pi/5e3, f0 = 1e-3)
    return f0/(pi*k0*k0)*exp(-(k/k0)^2)
end
function flux_skew(k, th, u, v; k0 = 2pi/5e3, f0 = 1e-3, th0=pi/6)
    #s = cos(th)*u[1] + sin(th)*v[1]
    s = 0.5*(1+cos(th-th0))
    return f0/(pi*k0*k0)*exp(-(k/k0)^2)*s^2
end
omega = 2pi/(300)

ncases = 0
case_mats = []

# Noise model
noise_model = WhiteNoiseModel(0.05)         # 0.05 ==> 5% additive flux error

# Number of waves
nws = [32]#[4 8 16 32]

###
# --- ProportionalFluxModel with the same mean measure as cont_spec ---
flux_fn  = flux_skew          # or flux_skew
cmin, cmax = 1.0, 100.0            # must match the ContinuousMap constructor (defaults here)

# (θ,c)-space density, 4-arg so target_density_cxcy can use it for plotting
g_θc = (θ, c, U, V) -> flux_fn(omega/c, θ, U, V) * omega^2 / c^3

# flux_isotropic / flux_skew ignore U,V, so dummy columns are fine
Udum = zeros(1); Vdum = zeros(1)
g2 = (θ, c) -> g_θc(θ, c, Udum, Vdum)

# Total flux on the box. abstol must be 0: the integral itself is O(1e-3)
prob = IntegralProblem((u, p) -> g2(u[1], u[2]), ([0.0, cmin], [2pi, cmax]))
total_flux_cm = solve(prob, HCubatureJL(); reltol=1e-6, abstol=0.0).u

# Rejection sampler on an O(1)-rescaled shape (sampling doesn't care about scale)
gmax  = maximum(g2(θ, c) for θ in range(0, 2pi, 361), c in range(cmin, cmax, 2000))
shape = (θ, c) -> g2(θ, c) / gmax
locP_cm = RejectionLocProposal(shape, 0.0, 2pi, cmin, cmax, 1.05)   # 5% safety on fmax

for i = 1:length(nws)
    nwav = nws[i]
    prop_spec = ProportionalFluxModel(locP_cm, g_θc, total_flux_cm, Poisson(nwav))
    push!(case_mats, (retrieve_fluxvec_from_model!, (prop_spec, noise_model)))
    global ncases += 1
end

function rotate_winds!(col::ColumnProfile)
    x,y = 0.0,0.0
    nz = length(col.U)
    for i = 1:nz
        x,y = col.U[i],col.V[i]
        col.U[i] = -y
        col.V[i] = x
    end
    return nothing
end


### Run all test cases and save data
bic_opts = BICOptions(0.0,1.0,0.0)
for case = 1:ncases
    retrieve_func!, retrieve_args = case_mats[case]
    results = ensemble_accuracy(ds, retrieve_func!; 
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
    save_ensemble_to_netcdf("torus_results_skew$(case).nc", results)
end