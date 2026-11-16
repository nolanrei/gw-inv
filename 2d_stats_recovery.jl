using LinearAlgebra
using SparseArrays
using StaticArrays
using Random
using Plots
using NetCDF
using Profile
using CSV
using DataFrames
using Printf

include("waccm-utils.jl")
include("wrf-utils.jl")
include("L81-2.jl")
include("L81-autograd-compatible.jl")
include("mini-newton.jl")
include("inversion.jl")

######################################################################################
##
######################################################################################


cmin = -100
cmax = 100
cstride = 2.5
ndirs = 2
srcs = [4]#, 32, 48]   # ~18km
wavespds = Array(cmin:cstride:cmax)
kw = 2pi/5e5
nws = length(wavespds)

z = 10:0.5:65
nz = length(z)

waves = make_waveset(wavespds, ndirs, srcs; ref_flux=1e-3)
nw = length(waves);
wlen = length(wavespds)

# tilted sets of waves
nth = 4
th0s = [(pi/ndirs)*(i/nth) for i = 0:nth-1]
wave_multiset = Array{WaveProfile,2}(undef,nw,nth)
for i = 1:nth
    wave_multiset[:,i] .= make_waveset(wavespds, ndirs, srcs; ref_flux=1e-3, th0=th0s[i])
end

lambda = 1e-5
BIG = 1e6

######################################################################################
##
######################################################################################

dz = 500*ones(nz)       # each cell is 500m thick in z
cases = [8]#[1,2,3,4,5,6,7,8,9,10]
files = wrf_get_fileset(cases)
vars = wrf_load(files)
wrf_postprocess!(vars)    ### DEPRECATED due to change from drag to flux

vars_to_be_avgd = ["Up", "Vp", "DX", "DY", "UW", "VW"]
tstamps,idx2case,case_order = wrf_tstamps_cases(files["U"])
order = wrf_order(tstamps,idx2case,case_order)
for var in vars_to_be_avgd
    vars[var] = wrf_knit(vars[var],order)
end
drag_means!(vars)

ref_flux = 1e-3
k = (2pi)/5e4    # 50km wavelength

# want to have col and fluxvec accessible outside the loop for debugging
col = ColumnProfile(vars["Up"][1][1,1,:,1], vars["Vp"][1][1,1,:,1], vars["N"], vars["rho"], nz, dz)
fluxvec = zeros(2nz)

# dummy initialization so the memory is allocated
x0, p1, mem = init_inv(wave_multiset[:,1], col, fluxvec; lambda, BIG)
xs = zeros(nw, nth)  # solutions for differently tilted axes
p = []
for i = 1:nth
    push!(p, (p1[1],wave_multiset[:,i],col,fluxvec,lambda,BIG,p1[7],p1[8],p1[9]) )
end

#####################################
# five random waves -- fixed seed
#####################################
#=
seed = -8854870601175104528
ntst = 5
test_wvsp = -40 .+(40-(-40))*rand(Xoshiro(seed),ntst)
test_fxs = 0.8*ref_flux*rand(Xoshiro(seed+1),ntst)
test_waves = Vector{WaveProfile}(undef,ntst)
for i = 1:ntst
    th = 2pi*rand(Xoshiro(seed+1+i))
    test_waves[i] = c_to_wave(test_wvsp[i], [cos(th);sin(th)], k, test_fxs[i], srcs[1])
end

nprof = 200                 ### number of profiles to test statistics with
x = zeros(nw)               # starting guess
sols2 = zeros(nw,nprof)     # where the solutions go for statistics-ing with
recon_5waves = zeros(2nz)
z_err = zeros(2nz)          # sum over idx of recon_5waves(z) .- reconstructed flux(z)
z_rms = zeros(2nz)          # sum of recon_5waves(z).^2
tmp = zeros(2nz)            # space to work in

unconverged = []

for idx = 1:nprof
    sz = size(vars["Up"][1])
    xi = rand(1:sz[1])
    yi = rand(1:sz[2])
    ti = rand(1:sz[4])
    col.U .= vars["Up"][1][xi,yi,:,ti]
    col.V .= vars["Vp"][1][xi,yi,:,ti]
    recon_5waves .= spectrum_to_momdep(L81_flux!,test_waves,col,test_fxs)
    
    err = randn(2nz)
    sigma = 0.05*maximum(abs.(recon_5waves))   # error scales with signal
    fluxvec .= recon_5waves .+ sigma*err
    
    x .= x0    # reinitialize to a good starting guess
    niter = wave_invert_mid!(x, p1, mem; reltol=1e-5, abstol=1e-10, 
        max_newton_steps=35, error_on_max_steps=false,
        verbose=false, armijo_max_steps_no_error=true)
    #println(niter)
    #if niter > 27
    #    push!(unconverged,(xi,yi,ti))
    #end
    sols2[:,idx] .= x

    tmp .= spectrum_to_momdep(L81_flux!,waves,col,exp.(x))
    z_err .+= (tmp.-recon_5waves).^2
    z_rms .+= (fluxvec).^2
end

# now we store the mean of the solutions in x
x .= mean(exp.(sols2),dims=2)

z_err .= sqrt.(z_err/nprof)
z_rms .= sqrt.(z_rms/nprof)

#### Compare test spectrum to reconstructed spectrum
# Construct an array that will look like test_spectrum when plotted
#begin
test_cx = [test_waves[i].c[1] for i=1:ntst]
test_cy = [test_waves[i].c[2] for i=1:ntst]
fx = [test_fxs[j]*test_waves[j].k/hypot(test_waves[j].k,test_waves[j].l) for j=1:ntst]
fy = [test_fxs[j]*test_waves[j].l/hypot(test_waves[j].k,test_waves[j].l) for j=1:ntst]
lookalike_hx, lookalike_vx = spike_plot_formatting(test_cx, fx)
lookalike_hy, lookalike_vy = spike_plot_formatting(test_cy, fy)
plot([lookalike_hx lookalike_hy],[lookalike_vx lookalike_vy],xlabel="wavespeed (m/s)",ylabel="flux (mPa)",label="test spectrum",layout=2)
plot!([wavespds wavespds],1e3*[x[1:wlen] x[wlen+1:2wlen]],label="reconstructed spectrum")
savefig("Figures/2d_5w_spectrum.png")

plot(reshape(z_err./z_rms,(nz,2)),z,xlabel="rms flux error / rms flux",ylabel="z (km)",label=["X" "Y"],layout=2)
savefig("Figures/statistics/2d_5w_err.png")
#end

# Debuggleshooting
#=
begin
recon2 = spectrum_to_momdep(L81_flux!,waves,col,exp.(x2))
plot(1e3*reshape(recon_5waves,(nz,2)),z,xlabel="flux (mPa)", ylabel="z (km)", label="truth", layout=2)
plot!(1e3*reshape(recon2,(nz,2)),z,label="recon",layout=2)
savefig("Figures/1d_5wtest.png")
end
=#
=#
#####################################
# five random waves drawn from Gaussian distribution -- change from iteration to iteration
#####################################
#=
ntst = 5
test_waves = Vector{WaveProfile}(undef,ntst)

# hyperparameters for Gaussian distribution
sd = 4.0         # stdev of distrib
mn = 0.0         # mean of distrib

nprof = 800       ### number of profiles to test statistics with
x1 = zeros(nw)    # starting guess
sols1 = zeros(nw,nprof)    # where the solutions go for statistics-ing with
recon_5waves = zeros(2nz)

for idx = 1:nprof
    test_wvsp = mn .+ sd*randn(ntst)
    test_fxs = 0.4e-3*rand(ntst)     # this should make the sum of all five flux=1mPa on average
    for i = 1:ntst
        th = 0 #2pi*rand()  # going back to 1d for a sec
        test_waves[i] = c_to_wave(test_wvsp[i], [cos(th);sin(th)], k, test_fxs[i], srcs[1])
    end
    
    sz = size(vars["Up"][1])
    xi = rand(1:sz[1])
    yi = rand(1:sz[2])
    ti = rand(1:sz[4])
    col.U .= vars["Up"][1][xi,yi,:,ti]
    col.V .= vars["Vp"][1][xi,yi,:,ti]
    recon_5waves .= spectrum_to_momdep(L81_flux!,test_waves,col,test_fxs)
    
    err = randn(2nz)
    sigma = 0.05*maximum(abs.(recon_5waves))   # error scales with signal
    recon_5w_err = recon_5waves .+ sigma*err
    
    x1 .= x0    # reinitialize to a good starting guess
    p1 = (L81_exp_grad!,waves,col,recon_5w_err,lambda,BIG,gi,J)
    newton_solve!(x1, nl_obj, nl_obj_grad!, nl_obj_hess!, df, H, dx, xnew, p1; 
        reltol=1e-5, abstol=1e-10, max_newton_steps=60, error_on_max_steps=false,
        verbose=false, armijo_max_steps_no_error=true)
    sols1[:,idx] .= x1
end

# now we store the mean of the solutions in x2
x1 .= mean(exp.(sols1),dims=2)

#### Compare test spectrum to reconstructed spectrum
# Construct an array that will look like test_spectrum when plotted
#begin
fluxsum = 1e-3
gauss_fn = c -> 2*fluxsum/sqrt(pi)/sd*exp( -((c-mn)/sd)^2 )
og_distrib = 1e3*gauss_fn.(wavespds)
plot(wavespds,og_distrib,xlabel="wavespeed (m/s)",ylabel="flux (mPa)",label="test spectrum")
plot!(wavespds,1e3*x1,label="reconstructed spectrum")
savefig("Figures/1d_5w_distrib.png")
#end
=#
#####################################
# Gaussian wavepacket
#####################################

seed2 = 5913522180718240123

# we need to draw a covariance matrix from a th-isotropic distribution
# Doing this by constructing the SVD
mn = zeros(2) #20*randn(Xoshiro(seed2),2)
ma = 25   # major axis for distribution
mi = 10   # minor axis
V = zeros(2,2)
V[:,1] .= ma*randn(Xoshiro(seed2+2),2)
l1 = sqrt.(sum(V[:,1].^2))
V[:,1] ./= l1
l2 = mi*randn(Xoshiro(seed2+4))
V[:,2] .= [-V[2,1]; V[1,1]]
C = V'*diagm([l1^2; l2^2])*V
Ci = inv(C)

fluxsum = 1e-3
gauss_fn = c -> fluxsum/(pi*abs(l1*l2))*exp( -(c.-mn)'*Ci*(c.-mn) )

# we need a new set of waves to sample this 2d gaussian wavepacket
nx = 50
ny = 50
wdth = zeros(2)
wdth[1] = max(abs(V[1,1]*l1), abs(V[2,1]*l2)) # max of dot(v1,xhat), dot(v2,xhat)
wdth[2] = max(abs(V[1,2]*l1), abs(V[2,2]*l2)) # max of dot(v1,yhat), dot(v2,yhat)
cxs = range((mn[1]-3*wdth[1]), (mn[1]+3*wdth[1]), nx)   #3 sigma spread
cys = range((mn[2]-3*wdth[2]), (mn[2]+3*wdth[2]), ny)
wavesquare = Array{WaveProfile}(undef,nx*ny)
for ix = 1:nx
    for iy = 1:ny
        cx = cxs[ix]
        cy = cys[iy]
        clen = hypot(cx,cy)
        cdir = [cx; cy]/clen
        wavesquare[(iy-1)*nx+ix] = c_to_wave(clen, cdir, k, gauss_fn([cx;cy])*(26*prod(wdth))/(nx*ny), srcs[1])
    end
end
wp_fxs = [wavesquare[i].flux for i = 1:length(wavesquare)]
wp_fxs_square = reshape(wp_fxs,(nx,ny))
heatmap(cxs,cys,wp_fxs_square)
savefig("testfig.png")

nprof2 = 200       ### number of profiles to test statistics with
x3 = zeros(nw)    # starting guess
sols3 = zeros(nw,nprof2)    # where the solutions go for statistics-ing with
recon_wp = zeros(2nz)

z_err = zeros(2nz,nth)          # sum over idx of recon_5waves(z) .- reconstructed flux(z)
z_rms = zeros(2nz,nth)          # sum of recon_5waves(z).^2
tmp = zeros(2nz)                # space to work in

for i = 1:nth
    for idx = 1:nprof2
        sz = size(vars["Up"][1])
        xi = rand(1:sz[1])
        yi = rand(1:sz[2])
        ti = rand(1:sz[4])
        col.U .= vars["Up"][1][xi,yi,:,ti]
        col.V .= vars["Vp"][1][xi,yi,:,ti]
        recon_wp .= spectrum_to_momdep(L81_flux!,wavesquare,col,wp_fxs)
        
        err = randn(2nz)
        sigma = 0.05*maximum(abs.(recon_wp))   # error scales with signal
        fluxvec .= recon_wp .+ sigma*err
        
        xs[:,i] .= x0    # reinitialize to a good starting guess
        wave_invert_mid!(view(xs,:,i), p[i], mem; reltol=1e-5, abstol=1e-10, 
            max_newton_steps=60, error_on_max_steps=false,
            verbose=false, armijo_max_steps_no_error=true)
        sols3[:,idx] .= xs[:,i]

        tmp .= spectrum_to_momdep(L81_flux!,waves,col,exp.(xs[:,i]))
        z_err[:,i] .+= (tmp.-recon_wp).^2
        z_rms[:,i] .+= (fluxvec).^2
    end
    # now we store the mean of the solutions in xs
    xs[:,i] .= mean(exp.(sols3),dims=2)
end

# wavepacket plots
spec_fxy = project_onto_XY(wavespds,wavesquare,wp_fxs)
plot([wavespds wavespds],1e3*spec_fxy,xlabel="wavespeed (m/s)",ylabel="flux (mPa)",label="test spectrum",layout=2)
for i = 1:nth
    plot!([wavespds wavespds],1e3*[xs[:,i][1:wlen] xs[:,i][wlen+1:2wlen]],label="reconstructed spectrum")
end
savefig("Figures/statistics/2d_wp_spectrum.png")

plot(sqrt.(z_err./z_rms)[1:nz,:],z,xlabel="RMS X error (m^2/s^2)",ylabel="z (km)",label=["0" "pi/16" "pi/8" "3pi/16"],layout=nth)
savefig("Figures/statistics/error_rot.png")