include("../omp.jl")
include("../ensemble_accuracy.jl")

# Do wrf_setup first

function cart2xyt(idxs::Vector{CartesianIndex{3}})
    return (getindex.(idxs, d) for d in 1:3)
end

idxs_near = findall(abs.(wrf_ds[:conv_dist_st]) .< 300)
idxs_far  = findall(abs.(wrf_ds[:conv_dist_st]) .> 300)
xs,ys,ts = cart2xyt(idxs_near)

nsamp = 1500

kw = 2pi/1e4
src = 13   # 16km
reg = 1e-8
results = ensemble_accuracy(wrf_ds, retrieve_fluxvec_avg!; 
        kw, src, reg, max_nw=20, nsamp, bic_opts=BICOptions(), 
        retrieve_args=(dt=4), project=true, xs, ys, ts, zis=src:nz)

save_ensemble_to_netcdf("Testing/ensemble_results_conv_c$(case)_1.nc", results)

xs,ys,ts = cart2xyt(idxs_far)
results = ensemble_accuracy(wrf_ds, retrieve_fluxvec_avg!; 
        kw, src, reg, max_nw=20, nsamp, bic_opts=BICOptions(), 
        retrieve_args=(dt=4), project=true, xs, ys, ts, zis=src:nz)

save_ensemble_to_netcdf("Testing/ensemble_results_conv_c$(case)_2.nc", results)