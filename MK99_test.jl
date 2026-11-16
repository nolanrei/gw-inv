using SpecialFunctions
using Plots

include("omp.jl")

# make sure you've already loaded in data in a DimStack ds

function sample_wave!(x, nx; c0=30.0, f0=0.5e-3)
    cx,cy = c0*randn(2)
    x[nx+1] = atan(cy,cx)
    x[nx+2] = hypot(cx,cy)
    x[nx+3] = f0
    return nx + 3
end

nw = 100
nx = 3nw
x = zeros(nx)

# initialize nw random waves
ix = 0
f0 = 2e-3/nw
for i = 1:nw
    global ix = sample_wave!(x,ix;f0)
end

# define column
sz = size(ds[:U])
xi = rand(1:sz[1])
yi = rand(1:sz[2])
ti = rand(1:sz[4])
z = parent(ds[:z])
nz = sz[3]
col = ColumnProfile(ds[:U][xi,yi,:,ti], ds[:V][xi,yi,:,ti], ds[:N], ds[:rho], nz, dz)

# dummy wave
max_nw = 150
kw = 2pi/5e4
src = 1
wav_dummy = c_to_wave(0.0, [1.0; 0.0], kw, 0.0, src)

# L81 forward model
os = OMPStruct(col, L81_AD_wrapper!, wav_dummy; max_nw=max_nw)
flux_l81 = zeros(2nz)
fwd!(flux_l81, x, nx, os)

# MK99 forward model
m9w = MK99Workspace(max_nw)
flux_mk99 = zeros(2nz)
MK99!(flux_mk99, x, nx, col, wav_dummy, z, m9w)

# plots
plot(reshape(flux_l81,(nz,2)),z,xlabel="flux (mPa)",ylabel="z (km)",layout=2,label=["L81 X" "L81 Y"])
plot!(reshape(flux_mk99,(nz,2)),z,label=["MK99 X" "MK99 Y"])
savefig("L81_vs_MK99.png")