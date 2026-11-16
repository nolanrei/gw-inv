include("wrf-utils.jl")
include("playground/ensemble_accuracy.jl")

# You need to load in a dataset before
ds = wrf_ds
z = mean_existing_dims(ds,:z,:z)
nz = length(z)

src = 4
n_avg = 25
na2 = div(n_avg,2)

function spectrum_averaging(ds::DimStack, n_avg::Integer, 
        xi::Integer, yi::Integer, t_mid::Integer; 
        kw=2pi/5e4, src, reg=1e-8, max_nw=20)
    # compares spectrum(time_average(data)) to time_average(spectrum(data))
    sz = size(ds[:U])
    nz = sz[3]
    wav_dummy = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)

    n_tot = n_avg + 1
    spectra = zeros(n_tot, 3max_nw)
    nxs = zeros(Int64,n_tot)
    x = zeros(3max_nw)
    fluxes = zeros(n_tot,2nz)
    fits = zeros(n_tot,2nz)

    ws = FullHessianOMPWorkspace(nz, max_nw)

    N = zeros(nz)
    rho = zeros(nz)
    z = zeros(nz)
    dz = zeros(nz)   #it's entirely unused
    
    col_dummy = ColumnProfile(ds[:U][1,1,:,1], ds[:V][1,1,:,1], N, rho, nz, dz)
    os = OMPStruct(col_dummy, L81_AD_wrapper!, wav_dummy; max_nw=max_nw)
    res = get_tmp(os.res, 0.0)   # gets the Float64 residual

    # if src == nothing, find it dynamically
    if src == nothing
        src_dyn = true
    else
        src_dyn = false
    end

    # Loop over samples
    n_avg_div2 = div(n_avg,2)
    for i = 1:n_tot

        if i <= n_avg
            ti = t_mid+i-1-n_avg_div2
        else
            ti = t_mid
        end
        
        # Initialize column profile
        sample_existing_dims!(os.col.N, ds, :N, xi, yi, ti)
        sample_existing_dims!(os.col.rho, ds, :rho, xi, yi, ti)
        sample_existing_dims!(z, ds, :z, xi, yi, ti)
        
        if src_dyn
            src = find_tropopause(os.col.N, z; z_min=5, threshold=0.015, persistence_km=2)
        end
        os.wav.src = src
        os.col.U .= ds[:U][xi,yi,:,ti]
        os.col.V .= ds[:V][xi,yi,:,ti]
        
        nx = 0
        fill!(x, 0.0)   # zero it out between runs
        
        # Initialize fluxvec
        if i <= n_avg
            retrieve_fluxvec_sample!(os.fluxvec, ds, xi, yi, ti)
        else
            retrieve_fluxvec_avg!(os.fluxvec, ds, xi, yi, ti; dt=n_avg_div2)
        end
        fluxes[i,:] .= os.fluxvec              # no NaNs
        
        # Solve
        nx = find_measure!(x,reg,os,ws)

        # Store spectrum
        spectra[i,1:nx] .= view(x,1:nx)       # no NaNs
        nxs[i] = nx

        # Store reconstruction
        get_res!(x, nx, os)
        res = get_tmp(os.res, x)
        fits[i,:] .= os.fluxvec .- res

    end

    return spectra, fits, fluxes, nxs
end

sz = size(ds[:U])
xi = rand(1:sz[1])
yi = rand(1:sz[2])
ti = rand(1+na2:sz[4]-na2)
spectra, fits, fluxes, nxs = spectrum_averaging(ds, n_avg, xi, yi, ti;
    kw=2pi/5e4, src, reg=1e-8, max_nw=20)

@show sum(abs2,fluxes .- fits, dims=2)

begin
    aflux    = 1e3*reshape(fluxes[1,:],(nz,2))
    afit     = 1e3*reshape(fits[1,:],(nz,2))
    bot_row  = plot(aflux,z,layout=(1,2),title="$(nhrs)hr recon",label="flux", 
        xlabel=["X flux (mPa)" "Y flux (mPa)"], ylabel="z (km)")
    plot!(bot_row, afit, z, layout=(1,2), label="recon")
    savefig("tmp.png")
end

## Plotting
# We want to compare the fits
nhrs = div(n_avg,4)
meanflux = 1e3*reshape(mean(fluxes[1:n_avg,:],dims=1),(nz,2))
meanfit  = 1e3*reshape(mean(fits[1:n_avg,:],dims=1),(nz,2))
top_row  = plot(meanflux,z,layout=(1,2),title="mean(15min recons)",label="flux", 
    xlabel=["X flux (mPa)" "Y flux (mPa)"], ylabel="z (km)")
plot!(top_row, meanfit, z, layout=(1,2), label="recon")
aflux    = 1e3*reshape(fluxes[n_avg+1,:],(nz,2))
afit     = 1e3*reshape(fits[n_avg+1,:],(nz,2))
bot_row  = plot(aflux,z,layout=(1,2),title="$(nhrs)hr recon",label="flux", 
    xlabel=["X flux (mPa)" "Y flux (mPa)"], ylabel="z (km)")
plot!(bot_row, afit, z, layout=(1,2), label="recon")
final_plot = plot(top_row, bot_row, layout = (2, 1))
savefig(final_plot, "t_avg_test.png")

# And also compare the spectra
begin
nx_tot = sum(nxs[1:n_avg])
spec_mean = zeros(nx_tot)
idx = 1
for i = 1:n_avg
    global idx
    nx = nxs[i]
    spec_mean[idx:idx+nx-1] .= spectra[i,1:nx]
    idx += nx
end
spec_mean[3:3:end] ./= n_avg    # averaging reduces the amplitudes of waves but not th or c
ros_mean = rosette_heatmap(spec_mean, nx_tot; ep=0.3, n=200, cmax=75)
title!(ros_mean, "mean(15min recons)")
ros_ctr  = rosette_heatmap(spectra[n_avg+1,:], nxs[n_avg+1]; ep=0.3, n=200, cmax=75)
title!(ros_ctr, "$(nhrs)hr recon")
savefig(ros_mean,"t_avg_ros_mean.png")
savefig(ros_ctr,"t_avg_ros_ctr.png")
end