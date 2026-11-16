function find_nonzero_col(ens; seed=nothing)
    # pick elements of ensemble until finding one that's not trivial
    maxnx,nsamp = size(ens.spectra)
    # guaranteed to find a nonzero col if one exists
    scram = randperm(Xoshiro(seed),nsamp)
    for j in scram
        if ens.spectra[j,3] > 0.0
            return j
        end
    end
    println("No nonzero columns exist in dataset.")
    return nothing
end

function populate_col(idx,ds::DimStack,ens)
    nz = size(ds[:U],3)
    
    xi = ens.xs[idx]
    yi = ens.ys[idx]
    ti = ens.ts[idx]
    U = ds[:U][xi,yi,:,ti]
    V = ds[:V][xi,yi,:,ti]
    
    N = zeros(nz)
    rho = zeros(nz)
    dz = ones(nz)
    
    sample_existing_dims!(N, ds, :N, xi, yi, ti)
    sample_existing_dims!(rho, ds, :rho, xi, yi, ti)

    return ColumnProfile(U, V, N, rho, nz, dz)
end

function consolidate_waves(x,nx)
    f0 = zeros(2)
    nw = div(nx,3)
    # finds a mean direction of flux
    for i = 1:nw
        th = x[3i-2]
        f = x[3i]
        f0[1] += f*cos(th)
        f0[2] += f*sin(th)
    end
    thc = atan(f0[2], f0[1])
    f0 ./= norm(f0)
    # projects each wave onto mean direction before adding together
    fc = 0.0
    for i = 1:nw
        th = x[3i-2]
        f = x[3i]
        fc += f*(f0[1]*cos(th) + f0[2]*sin(th))
    end
    return thc, fc
end

case = 1
ens = wacres[case]
idx = find_nonzero_col(ens)
col = populate_col(idx, waccm_ds, ens)
kw = 2pi/1e4
src = 1
dummy_wav = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)

nxi = ens.nxs[idx]
recovered_spectrum = ens.spectra[idx,1:nxi]
println("recovered spectrum = ",recovered_spectrum)

# see why orographic waves aren't reconstructing to c = 0
fluxi = ens.fluxes[idx,:]
thc, fc = consolidate_waves(recovered_spectrum, nxi)
# wave in same direction and with same mean aggregate flux but at c = 0
comp_spec = [thc, 0.0, fc]
comp_flux = zeros(size(fluxi))
L81_AD_wrapper!(comp_flux, comp_spec, dummy_wav, col; mom_leak=true)
plot(1e3*reshape(fluxi,(nz,2)), z, xlabel=["U flux (mPa)" "V flux (mPa)"], ylabel="z (km)", label="true flux", layout=2)
plot!(1e3*reshape(comp_flux,(nz,2)), z, layout=2, label="theoretical")
savefig("oro_offset.png")