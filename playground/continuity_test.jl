# run after wrf_setup.jl
include("omp.jl")

# load in a random profile
sz = size(vars["Up"][1])
xi = rand(1:sz[1])
yi = rand(1:sz[2])
ti = rand(1:sz[4])
col = ColumnProfile(vars["Up"][1][xi,yi,:,ti], vars["Vp"][1][xi,yi,:,ti], vars["N"], vars["rho"], nz, dz)

# pick x vector (half of these waves)
ntst = 6
cm = -50.0
cp = 50.0
test_wvsp = cm .+(cp-cm)*rand(ntst)

alpha=2
ref_flux = 1e-3        # 1 mPa
mnfx = 0.1*ref_flux    # mean flux
theta = mnfx/alpha
test_fxs = rand(Gamma(alpha,theta),ntst)

kw = 2pi/5e5
src = 4

test_waves = Vector{WaveProfile}(undef,ntst)
for i = 1:ntst
    c = randn(2)
    c ./= norm(c)
    test_waves[i] = c_to_wave(test_wvsp[i], c, kw, test_fxs[i], src)
end

# make our OMPStruct
os = OMPStruct(col, L81_AD_wrapper!, c_to_wave(0.0, [1.0;0.0], kw, 0.0, src))

# make rhs: all of the test waves, plus error
fluxvec = spectrum_to_momdep(L81_flux!,test_waves,col,test_fxs)  
err = randn(2nz)
sigma = 0.05*maximum(abs.(fluxvec))   # error scales with signal
fluxvec .+= sigma*err

# compute residual
# remember only half of the waves are being used because that's what we chose x to be
current = spectrum_to_momdep(L81_flux!,test_waves[1:div(ntst,2)],col,test_fxs[1:div(ntst,2)])
os.res = fluxvec .- current

# our central wave is wave ntst/2 + 1
wav = test_waves[div(ntst,2)+1]
wave_params = [atan(wav.l,wav.k), norm(wav.c)]

# pick a random direction in parameter space to go along
dw = randn(2).*[0.2*pi, 0.2*cm]

# pick a distribution to average our gradients over
pf_lam = 2e-4    # units of Pa to match b
pf = Exponential(pf_lam)
cf = f -> cdf(pf,f)

# plot score along this line
ns = 101
s = range(-1,1,ns)
scores = zeros(ns)
for i = 1:ns
    scores[i] = get_score(wave_params .+ s[i]*dw, os, cf)
end
plot(s, scores, xlabel="perturbation scaling", ylabel="score", label=nothing)
savefig("continuity_test.png")