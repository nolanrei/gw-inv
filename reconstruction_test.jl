cmin = -80
cmax = 80
cstride = 2
ndirs = 1
srcs = [4]#, 32, 48]   # ~18km
wavespds = Array(cmin:cstride:cmax)
kw = 2pi/5e5
nws = length(wavespds)

th0 = 2*pi*rand()   # random offset for waves
waves = make_waveset(wavespds, ndirs, srcs; ref_flux=1e-3, th0=th0)
nw = length(waves);
wlen = length(wavespds)

lambda = 1e-7
BIG = 1e6

########################################################################
### Pick ntst waves
########################################################################

ntst = 2
cm = -50.0
cp = 50.0
test_wvsp = cm .+(cp-cm)*rand(ntst)
println(test_wvsp)

alpha=2
mnfx = 0.1*ref_flux    # mean flux
theta = mnfx/alpha
test_fxs = rand(Gamma(alpha,theta),ntst)

test_waves = Vector{WaveProfile}(undef,ntst)
for i = 1:ntst
    test_waves[i] = c_to_wave(test_wvsp[i], [cos(th0);sin(th0)], kw, test_fxs[i], srcs[1])
end

########################################################################
### Test on nprof profiles
########################################################################

fluxvec = zeros(2nz)

x0, p, mem = init_inv(waves, col, fluxvec; lambda, BIG, wndw_flag=false, wndw=zeros(0))
x0 .= -12.0

nprof = 1       ### number of profiles to test statistics with
x2 = zeros(nw)    # starting guess
sols2 = zeros(nw,nprof)    # where the solutions go for statistics-ing with
recon = zeros(2nz)

z_err = zeros(2nz)          # sum over idx of (recon_5waves(z) .- reconstructed flux(z)).^2
z_rms = zeros(2nz)          # sum of recon_5waves(z).^2
tmp = zeros(2nz)            # space to work in

for idx = 1:nprof
    sz = size(vars["Up"][1])
    xi = rand(1:sz[1])
    yi = rand(1:sz[2])
    ti = rand(1:sz[4])
    col.U .= vars["Up"][1][xi,yi,:,ti]
    col.V .= vars["Vp"][1][xi,yi,:,ti]
    recon .= spectrum_to_momdep(L81_flux!,test_waves,col,test_fxs)
    
    err = randn(2nz)
    sigma = 0.05*maximum(abs.(recon))   # error scales with signal
    recon_err = recon .+ sigma*err
    fluxvec .= recon_err
    
    x2 .= x0    # reinitialize to a good starting guess
    wave_invert_mid!(x2, p, mem; reltol=1e-5, abstol=1e-10, 
        max_newton_steps=50, error_on_max_steps=false,
        verbose=true, armijo_max_steps_no_error=false, 
        max_line_search_steps=30, keep_steps=false)
    sols2[:,idx] .= x2

    tmp .= spectrum_to_momdep(L81_flux!,waves,col,exp.(x2))
    z_err .+= (tmp.-recon).^2
    z_rms .+= (recon_err).^2
end

# now we store the mean of the solutions in x2
x2 .= mean(exp.(sols2),dims=2)

z_err .= sqrt.(z_err/nprof)
z_rms .= sqrt.(z_rms/nprof)

#### Compare test spectrum to reconstructed spectrum
# Construct an array that will look like test_spectrum when plotted
lookalike_hx, lookalike_vx = spike_plot_formatting(test_wvsp,test_fxs)

plot(lookalike_hx,lookalike_vx,xlabel="wavespeed (m/s)",ylabel="flux (mPa)",label="test spectrum")
plot!(wavespds,1e3*x2,label="reconstructed spectrum")
savefig("Figures/statistics/1d_5w_spectrum.png")

plot(reshape(z_rms,(nz,2)),z,xlabel="rms flux",ylabel="z (km)",title=["X" "Y"],label="rms flux",layout=2)
plot!(reshape(z_err,(nz,2)),z,label="rms error",layout=2)
savefig("Figures/statistics/1d_5w_err.png")