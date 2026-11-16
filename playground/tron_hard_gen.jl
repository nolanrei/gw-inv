# Meant to be run after loading in data with wrf_setup
include("tron_defs.jl")

# --- GLOBAL SETUP FOR THE TEST FILE ---
Random.seed!(86) 

sz = size(vars["Up"][1])
xi, yi, ti = rand(1:sz[1]), rand(1:sz[2]), rand(1:sz[4])

# Distribution CDF parameterization
λ = 1 / 1e-3  
my_cdf = x -> 1.0 - exp(-λ * x)

# Initialize Column Profile
col = ColumnProfile(vars["Up"][1][xi,yi,:,ti], vars["Vp"][1][xi,yi,:,ti], vars["N"], vars["rho"], nz, dz)
kw = 2pi/5e5
src = 4
wav = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)

# Synthetic Truth Generation
ntst = 6
cm = 0.0
cp = 60.0
test_cs = cm .+ (cp - cm) * rand(ntst)

alpha = 2
ref_flux = 1e-3        # 1 mPa
mnfx = 0.1 * ref_flux  # mean flux
test_fxs = rand(Gamma(alpha, mnfx / alpha), ntst)
test_ths = 2pi * rand(ntst) .- pi

# Build RHS (Clean synthetic signal)
target_flux = zeros(2nz)
clean_signal = zeros(2nz)
for i = 1:ntst
    fill!(target_flux, 0.0)
    L81_AD_wrapper!(target_flux, [test_ths[i], test_cs[i], test_fxs[i]], wav, col)
    clean_signal .+= target_flux
end

# Add noise and assign directly to preallocated struct buffer
err = randn(2nz)
sigma = 0.05 * maximum(abs.(clean_signal)) 
fluxvec .= clean_signal .+ (sigma .* err)

## Set up waves
cmin = -60
cmax = 60
cstride = 1
ndirs = 2   
src = 4         # source level = 12km
wavespds = Array(cmin:cstride:cmax)
kw = 2pi/5e5
nws = length(wavespds)
ref_flux = 1e-3
nw = nws*ndirs

# sets up x and y directions
waves = make_waveset(wavespds, ndirs, [src]; kw, ref_flux)

### Solve using JSOSolvers.jl
nlp = GWFit(waves, col, fluxvec; lambda=1e-6, wndw=nothing)
solver = TronSolver(nlp);
stats = solve!(solver, nlp; verbose=5)

## Plot
plot(wavespds, 1e3*reshape(stats.solution,(nws,2)), xlabel="phase speed (m/s)", ylabel="flux (mPa)", label=["X" "Y"],layout=2)
savefig("tron_hard_spec_xy.png")

begin
recon_momdep = spectrum_to_momdep(L81_flux!,waves,col,stats.solution)
plot(1e3*reshape(fluxvec,(nz,2)),z,layout=2,xlabel="flux (mPa)",ylabel="z (km)",label=["fx" "fy"])
plot!(1e3*reshape(recon_momdep,(nz,2)),z,layout=2,label=["x recon" "y recon"])
savefig("tron_hard_recon_xy.png")
end

# convert to [thetas], [cs], [fxs] for comparison to OMP
begin
    scatter(test_ths, test_cs, zcolor=test_fxs, 
    xlabel="theta (radians)", ylabel="c (m/s)", 
    label="Truth", markersize=5,color=:viridis)
    
    tr_ths = zeros(nw)
    tr_cs = zeros(nw)
    tr_fxs = zeros(nw)
    for i = 1:nw
        tr_ths[i] = atan(waves[i].l,waves[i].k)
        tr_cs[i] = hypot(waves[i].c[1], waves[i].c[2])
        tr_fxs[i] = stats.solution[i]
    end
    nzos = (tr_fxs .!== 0.0)
    tr_ths = tr_ths[nzos]
    tr_cs = tr_cs[nzos]
    tr_fxs = tr_fxs[nzos]
    scatter!(tr_ths, tr_cs, zcolor = tr_fxs, label="TRON recon", markersize=5, color=:inferno)
    savefig("tron_hard_xy_2d_signal.png")
end


## 2D grid solve

# make waves for 2d grid
wsp_g = -50:2:50
nws_g = length(wsp_g)
nw_g = nws_g^2
waves_g = Array{WaveProfile,1}(undef, nw_g)
for j = 1:nws_g
    for i = 1:nws_g
        dir = [wsp_g[i], wsp_g[j]]
        c = norm(dir)
        dir /= c
        waves_g[i + nws_g*(j-1)] = c_to_wave(c,dir,kw,ref_flux,src)
    end
end

# Solve
nlp_g = GWFit(waves_g, col, fluxvec; lambda=1e-6, wndw=nothing)
solver_g = TronSolver(nlp_g);
stats_g = solve!(solver_g, nlp_g; verbose=5)

## Plot
heatmap(wsp_g, wsp_g, 1e3*reshape(stats.solution,(nws_g,nws_g)), 
    xlabel="x phase speed (m/s)", ylabel="y phase speed (m/s)")
savefig("tron_hard_spec_grid.png")

begin
recon_g = spectrum_to_momdep(L81_flux!,waves_g,col,stats_g.solution)
plot(1e3*reshape(fluxvec,(nz,2)),z,layout=2,xlabel="flux (mPa)",ylabel="z (km)",label=["fx" "fy"])
plot!(1e3*reshape(recon_g,(nz,2)),z,layout=2,label=["x recon" "y recon"])
savefig("tron_hard_recon_grid.png")
end