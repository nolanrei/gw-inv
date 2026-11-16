include("omp.jl")
include("ensemble_accuracy.jl")

function meanfluxlow(fluxU::AbstractVector, fluxV::AbstractVector, z::AbstractVector)
    lowidxs = findall(z .> 18. .&& z .< 25.)
    return mean(sqrt.(fluxU[lowidxs].^2 .+ fluxV[lowidxs].^2))
end

function getfluxes(ds::DimStack, idxs)
    fluxes = Vector{Float64}(undef, length(idxs))
    for (n, I) in enumerate(idxs)
        i, j, k = Tuple(I)
        fluxes[n] = meanfluxlow(view(ds[:UW], i, j, :, k), view(ds[:VW], i, j, :, k), ds[:z])
    end
    return fluxes
end

function get_recon_flux(x,nx)
    return sum(x[3:3:nx])
end
#=
rainidxs = findall(wrf_ds[:conv_dist_km] .< 100.0)
norainidxs = findall(wrf_ds[:conv_dist_km] .> 250.0)
rainfluxes   = getfluxes(wrf_ds, rainidxs)
norainfluxes = getfluxes(wrf_ds, norainidxs)
println("mean rain fluxes = ", mean(rainfluxes))
println("mean no-rain fluxes = ", mean(norainfluxes))
=#

cidxs = CartesianIndices(wrf_ds[:precip])
allidxs = vec(collect(cidxs))          # flatten to a Vector{CartesianIndex}
sampleidxs = allidxs[randperm(length(allidxs))[1:300]]

max_nw = 20
sz = size(wrf_ds[:U])
nz = sz[3]
kw = 2pi/1e4
src = 1
reg = 1e-8
wav_dummy = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)
ws = FullHessianOMPWorkspace(nz, max_nw)
bic_opts=BICOptions()

N = zeros(nz)
rho = zeros(nz)
z = zeros(nz)
dz = zeros(nz)   #it's entirely unused

col_dummy = ColumnProfile(wrf_ds[:U][1,1,:,1], wrf_ds[:V][1,1,:,1], N, rho, nz, dz)
os = OMPStruct(col_dummy, L81_AD_wrapper!, wav_dummy; max_nw=max_nw)

sharpen = sharpen_thresh!
sharpen_params = (1e-5,)
retrieve_fluxvec! = retrieve_fluxvec_avg!
dt=4
retrieve_args = (dt,)
x = zeros(3max_nw)

N = 300
allfluxes = zeros(N)
allrecon = zeros(N)
alldists = zeros(N)
allstrs  = zeros(N)
for (i, idx) in enumerate(sampleidxs)
    xi,yi,ti = Tuple(idx)
    allfluxes[i] = getfluxes(wrf_ds, [idx])[1]
    fill!(x,0)

    # Initialize column profile
    sample_existing_dims!(os.col.N, wrf_ds, :N, xi, yi, ti)
    sample_existing_dims!(os.col.rho, wrf_ds, :rho, xi, yi, ti)
    sample_existing_dims!(z, wrf_ds, :z, xi, yi, ti)
    
    os.col.U .= wrf_ds[:U][xi,yi,:,ti]
    os.col.V .= wrf_ds[:V][xi,yi,:,ti]
    retrieve_fluxvec!(os.fluxvec, wrf_ds, xi, yi, ti, os, retrieve_args...)
    nx = find_measure!(x,reg,os,sharpen,ws,bic_opts;sharpen_params)
    v = view(x,1:nx)
    nx = project_to_min!(v, v, kw, os.col)
    allrecon[i] = get_recon_flux(x,nx)
    
    alldists[i] = wrf_ds[:conv_dist_km][idx]
    allstrs[i]  = wrf_ds[:conv_strength][idx]
end
scatter(allstrs, alldists, 1e3*allfluxes, xlabel="precipitation rate (mm/hr)", ylabel="mean 18-25km flux (mPa)")
savefig("str_v_dist_v_flux_short.png")
scatter(alldists, 1e3*allfluxes, xlabel="distance from convection (km)", ylabel="mean 18-25km flux (mPa)")
savefig("dist_v_flux_short.png")
scatter(alldists, 1e3*allrecon, xlabel="distance from convection (km)", ylabel="mean 18-25km flux (mPa)")
savefig("dist_v_flux_recon_short.png")