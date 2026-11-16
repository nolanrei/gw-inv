# assumes you've already run test cases and loaded them in by load_all_ensembles
include("spectrum_metric.jl")

data = resultses[5]

nsamp = length(data.spec_dist)

idx = argmax(data.spec_dist)#rand(1:nsamp)

bi = data.ks_offsets[idx]+1
ei = data.ks_offsets[idx+1]
truth = data.known_spectra[bi:ei]
nx = data.nxs[idx]
recon = data.spectra[idx,1:nx]

costs = pairwise_dist(recon, truth)
plan = earth_mover_plan(recon, truth)