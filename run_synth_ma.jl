### Run setup_synth_ma.jl before running this or it will break
# this is the wave generation part which is intended to be fast

# Pick a WRF column
sz = size(vars["Up"][1])
xi = rand(1:sz[1])
yi = rand(1:sz[2])
ti = rand(1:sz[4])
col = ColumnProfile(vars["Up"][1][xi,yi,:,ti], vars["Vp"][1][xi,yi,:,ti], vars["N"], vars["rho"], nz, dz)

ndirs = 2       # maximum number of unique directions for waves
nwaves = 40      # number of waves generated
cmax = 50      # maximum wave phase speed (m/s)
charflux = 2e-3/nwaves   # flux e-folding amplitude (Pa)

# Generate the waves
dirs = zeros(2,ndirs)
fluxes = zeros(nwaves)
swaves = Vector{WaveProfile}(undef,nwaves) # Sample waves
for wi = 1:nwaves
    c = cmax/3*randn()  # OK so it's not exactly a maximum. 3 sigma is close enough
    if wi <= ndirs
        # choose a new direction randomly
        dir = randn(2)
        dir ./= norm(dir)
        dirs[:,wi] .= dir
    else
        # choose from among the existent directions
        di = rand(1:ndirs)
        dir = dirs[:,di]
    end
    flux = charflux*(7/sqrt(1+c*c))*randexp()
    fluxes[wi] = flux
    swaves[wi] = c_to_wave(c, dir, k, flux, srcs[1])
end

# Get fluxes for these waves using L81
fluxvec = spectrum_to_momdep(L81_flux!,swaves,col,fluxes)

# Plot the fluxes as angle and magnitude
th_and_mag = zeros(nz,2)
angle_no_brcuts!(view(th_and_mag,:,1), fluxvec[1:nz], fluxvec[nz+1:2nz])
th_and_mag[:,2] .= 1e3*hypot.(fluxvec[1:nz], fluxvec[nz+1:2nz])  # cast to mPa
plot(th_and_mag,z,xlabel=["flux angle (radians)" "flux magnitude (mPa)"],
    ylabel="z (km)",label=nothing,layout=2,size=(800,400))
ri = rand(UInt)
savefig("Figures/synth_fxs/nd$(ndirs)nw$(nwaves)cm$(cmax)_$(ri).png")

# Plot the drag as angle (not seamless) and magnitude
dragvec = spectrum_to_momdep(L81_wp!,swaves,col,fluxes)
th_and_mag_dr = zeros(nz,2)
th_and_mag_dr[:,1] .= atan.(dragvec[nz+1:2nz], dragvec[1:nz])
th_and_mag_dr[:,2] .= 86400*hypot.(dragvec[1:nz], dragvec[nz+1:2nz])
plot(th_and_mag_dr,z,xlabel=["drag angle (radians)" "drag magnitude (m/s/day)"],
    ylabel="z (km)",label=nothing,layout=2,size=(800,400))
savefig("Figures/synth_fxs/dr_nd$(ndirs)nw$(nwaves)cm$(cmax)_$(ri).png")