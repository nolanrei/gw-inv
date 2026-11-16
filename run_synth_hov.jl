### Run setup_synth_ma.jl before running this or it will break
# this is the wave generation part which is intended to be fast

# Pick a WRF column
sz = size(vars["Up"][1])
xi = rand(1:sz[1])
yi = rand(1:sz[2])
nT = 96
ti = rand(1:sz[4]-nT+1)
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
    flux = charflux*randexp()  #(7/sqrt(1+c*c))*
    fluxes[wi] = flux
    swaves[wi] = c_to_wave(c, dir, k, flux, srcs[1])
end

# do the Hovmoller thing for synthetic drag
dx = zeros(nz, nT)
dy = zeros(nz, nT)

dragvec = zeros(2nz)
for idx = 1:nT
    dragvec .= 0.0
    col.U .= vars["Up"][1][xi,yi,:,ti+idx]
    col.V .= vars["Vp"][1][xi,yi,:,ti+idx]
    spectrum_to_momdep!(L81_wp!,swaves,col,fluxes,dragvec)
    dx[:,idx] .= dragvec[1:nz]
    dy[:,idx] .= dragvec[nz+1:2nz]
end

heatmap(0.25*(1:nT),z,86400*dx,xlabel="time (hrs)",ylabel="z (km)")
ri = rand(UInt)
savefig("Figures/synth_fxs/hovmoller_synth_dx.png") #_$(ri)

heatmap(0.25*(1:nT),z,86400*dy,xlabel="time (hrs)",ylabel="z (km)")
savefig("Figures/synth_fxs/hovmoller_synth_dy.png") #_$(ri)

te = ti+nT
uw = vars["UW"][1][xi,yi,:,ti+1:te].*vars["rho"]
vw = vars["VW"][1][xi,yi,:,ti+1:te].*vars["rho"]

drx = 86400*vars["DX"][1][xi,yi,:,ti+1:te];
dry = 86400*vars["DY"][1][xi,yi,:,ti+1:te];

heatmap(0.25*(1:nT),z,uw,xlabel="time (hrs)",ylabel="z (km)")
savefig("Figures/hovmoller_uw.png")

heatmap(0.25*(1:nT),z,vw,xlabel="time (hrs)",ylabel="z (km)")
savefig("Figures/hovmoller_vw.png")

heatmap(0.25*(1:nT),vars["oz_dx"],drx,xlabel="time (hrs)",ylabel="z (km)")
savefig("Figures/hovmoller_dx.png")

heatmap(0.25*(1:nT),vars["oz_dx"],dry,xlabel="time (hrs)",ylabel="z (km)")
savefig("Figures/hovmoller_dy.png")