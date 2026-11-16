include("wrf-utils.jl")
include("ensemble_accuracy.jl")

# You need to load in a dataset before
ds = wrf_ds
z = mean_existing_dims(ds,:z,:z)
nz = length(z)

src = 4

nsamp = 200
nxw = 15
retrieve_args = (nxw)
results = ensemble_accuracy(ds, create_sample_fluxvec!; src, reg=1e-8, nsamp, kw=2pi/5e3, retrieve_args);

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

ks_rag = unflatten_ragged(known_spectra, ks_offsets)


flux_abs = dropdims(mean(abs2,fluxes,dims=(1)),dims=1)
flux_abs = sqrt.(flux_abs[1:nz] .+ flux_abs[nz+1:2nz])

begin
i = rand(1:nsamp)
while true
    if sum(abs2,fluxes[i,:]) < 1e-9
        global i = rand(1:nsamp)
    else
        break
    end
end
display = reshape((fluxes[i,:].-errors[i,:]),(nz,2))
plot(reshape((fluxes[i,:]),(nz,2)),z,xlabel="fluxes (mPa)",ylabel="z (km)",layout=2,label=["truth X" "truth Y"])
plot!(display,z,label=["recon X" "recon Y"])
savefig("tst1.png")
U = wrf_ds[:U_raw][xs[i],ys[i],:,ts[i]]
V = wrf_ds[:V_raw][xs[i],ys[i],:,ts[i]]
rosmap = rosette_heatmap(spectra[i,1:nxs[i]],nxs[i],cmax=100)
r = hypot(U[src],V[src])
theta = atan(V[src],U[src])
plot!(rosmap,[0,theta],[0,r],arrow=true)
rosette!(rosmap, ks_rag[i], length(ks_rag[i]))
savefig(rosmap,"rosette.png")
srcs,prox = src_check(spectra[i,:],nxs[i],fluxes[i,:],U,V,wrf_ds[:iz]; thresh=12.0, cutoff_ang=pi/6)
end



begin
plot(rms_abs_err_z[src:end],z[src:end], xlabel="flux abs. error", ylabel="z (km)", label=["err X" "err Y"])
plot!(flux_abs[src:end], z[src:end],label=["F X" "F Y"])
savefig("relerr.png")
end

function adjust_by_vsrc(spectra::Array{Float64}, nxs::Vector{Int64}, vel::Array{Float64})
    nsamp = size(spectra,1)
    nxg = sum(nxs)
    gesamt = zeros(nxg)
    # Rotate each spectrum such that th=0 is direction of wind at src
    idx = 0
    for i = 1:nsamp
        th_src = atan(vel[i,2], vel[i,1])
        nw = div(nxs[i],3)
        for j = 1:nw
            gesamt[idx+1] = spectra[i,3j-2] - th_src  # aligns windspeed to 0
            gesamt[idx+2] = spectra[i,3j-1]           # other two are unchanged
            gesamt[idx+3] = spectra[i,3j]
            idx += 3
        end
    end
    return gesamt
end

function get_src_u(ds::DimStack, src::Int64, xs::Vector{Int64}, ys::Vector{Int64}, ts::Vector{Int64})
    nsamp = length(xs)
    vel = zeros(nsamp,2)
    for i = 1:nsamp
        xi=xs[i]
        yi=ys[i]
        ti=ts[i]
        vel[i,1] = ds[:U][xi,yi,src,ti]
        vel[i,2] = ds[:V][xi,yi,src,ti]
    end
    return vel
end

function rotated_c_minus_u(spectra, nxs, vel)
    nsamp = size(spectra, 1)
    nxg = sum(nxs)
    
    gesamt = zeros(eltype(spectra), nxg)
    
    idx = 0
    for i in 1:nsamp
        u = vel[i, 1]
        v = vel[i, 2]
        th_src = atan(v, u)
        
        nw = div(nxs[i], 3)
        for j in 1:nw
            th   = spectra[i, 3j - 2]
            c    = spectra[i, 3j - 1]
            flux = spectra[i, 3j]
            
            # Intrinsic phase speed vector in Cartesian coords
            cx = c * cos(th) - u
            cy = c * sin(th) - v
            
            # Convert back to polar and rotate reference frame
            # rem2pi keeps the angle inside [-π, π]
            gesamt[idx + 1] = rem2pi(atan(cy, cx) - th_src, RoundNearest)
            gesamt[idx + 2] = hypot(cx, cy)
            gesamt[idx + 3] = flux
            
            idx += 3
        end
    end
    
    return gesamt
end

begin
    vel = get_src_u(ds,src,xs,ys,ts)
    gesamt = adjust_by_vsrc(spectra, nxs, vel)  # rotated_c_minus_u
    
    nx = length(gesamt)   # we added nxs all together
    #p = rosette(gesamt, nx)
    p = rosette_heatmap(gesamt, nx, ep=0.3, n=200, cmax=25)
    savefig(p, "rosette_wrf.png")
    #savefig(p, "rosette_wrf_cminusu.png")
end