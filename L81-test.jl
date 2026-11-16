using LinearAlgebra
using SparseArrays
using StaticArrays
using Random
using Plots
using NetCDF
using Profile
using CSV
using DataFrames
using ProfileVega

include("waccm-utils.jl")
include("L81.jl")
include("L81-validation.jl")
zb = 25      # 0.5 hPa
ze = 58      # 500 hPa
yb = 1
ye = 192
xb = 1
xe = 288
ref_flux = 1e-3
tbatch = 8

err_val = zeros((ye-yb+1)*(xe-xb+1)*tbatch,4)

ndays = 91
for it = 1:(ndays*8/tbatch)
    tb = Int((it-1)*tbatch + 1)
    te = Int(it*tbatch)
    U, V, rho, N, lon, lat, Z, ugw, vgw = load_waccm_data([xb,yb,zb,tb],[xe,ye,ze,te])
    
    err_val .= L81_validate(U,V,rho,N,ugw,vgw; ref_flux, ndirs=4, reg="l2", use_waccm_drag=true, refine=1)
    CSV.write("1dtestruns.csv", DataFrame(err_val,:auto), header=false, append=true)
end
err_df = CSV.read("1dtestruns.csv", DataFrame)
b_range = range(0.0, 2.0, length=21)
histogram(err_val[:,1],bins=b_range,normalize=:probability,title="Error probability for WACCM waves")
xlabel!("normalized error")
ylabel!("bin probability")
savefig("test-wave-hist.png")