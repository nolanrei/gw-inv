include("../../omp.jl")
include("../../ensemble_accuracy.jl")

function cart2xyt(idxs::Vector{CartesianIndex{3}})
    return (getindex.(idxs, d) for d in 1:3)
end

function neighborhood(A::AbstractArray, center::CartesianIndex{M}, r::Real;
                      dims = ntuple(identity, M)) where {M}
    length(dims) == M || throw(ArgumentError("center needs one coordinate per entry in dims"))
    ri = floor(Int, r)
    lo = CartesianIndex(ntuple(k -> max(first(axes(A, dims[k])), center[k] - ri), M))
    hi = CartesianIndex(ntuple(k -> min(last(axes(A, dims[k])),  center[k] + ri), M))
    return [I for I in lo:hi if sum(abs2, Tuple(I - center)) <= r^2]
end

idxs_ghost = neighborhood(parent(wrf_ds[:U]),CartesianIndex(30,16,150),3,dims=(1,2,4))
idxs_conv = neighborhood(parent(wrf_ds[:U]),CartesianIndex(37,13,170),3,dims=(1,2,4))
idxs_bkgd = neighborhood(parent(wrf_ds[:U]),CartesianIndex(23,31,70),3,dims=(1,2,4))

kw = 2pi/1e4
src = 13   # 16km
reg = 1e-8

xs,ys,ts = cart2xyt(idxs_ghost)
nsamp = length(idxs_ghost)
results = ensemble_accuracy(wrf_ds, retrieve_fluxvec_avg!; 
        kw, src, reg, max_nw=20, nsamp, bic_opts=BICOptions(), 
        retrieve_args=(dt=4), project=true, xs, ys, ts, zis=src:nz)

save_ensemble_to_netcdf("Testing/c10/ensemble_ghost.nc", results)

xs,ys,ts = cart2xyt(idxs_conv)
nsamp = length(idxs_conv)
results = ensemble_accuracy(wrf_ds, retrieve_fluxvec_avg!; 
        kw, src, reg, max_nw=20, nsamp, bic_opts=BICOptions(), 
        retrieve_args=(dt=4), project=true, xs, ys, ts, zis=src:nz)

save_ensemble_to_netcdf("Testing/c10/ensemble_conv.nc", results)

xs,ys,ts = cart2xyt(idxs_bkgd)
nsamp = length(idxs_bkgd)
results = ensemble_accuracy(wrf_ds, retrieve_fluxvec_avg!; 
        kw, src, reg, max_nw=20, nsamp, bic_opts=BICOptions(), 
        retrieve_args=(dt=4), project=true, xs, ys, ts, zis=src:nz)

save_ensemble_to_netcdf("Testing/c10/ensemble_bkgd.nc", results)

bkg = load_ensemble_from_netcdf("Testing/c10/ensemble_bkgd.nc")
sbk = condense_spectra(bkg.spectra, bkg.nxs)
sbk[3:3:end] ./= length(sbk/3)   # normalize for plotting the mean rather the sum
rbk = rosette_heatmap(sbk, length(sbk))
savefig(rbk,"Testing/c10/ros_bkgd.png")
