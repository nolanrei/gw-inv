using LinearAlgebra
using SparseArrays
using StaticArrays
using Random
using Plots
using NetCDF
using CSV
using DSP
using Printf

include("waccm-utils.jl")
include("wrf-utils.jl")
include("L81-2.jl")

nt = 32
tb = rand(1:(2920-nt))
te = tb + nt
zb = 5      
ze = 65      
yb = 96-8      # close to +15 degrees    # 33 for 60N
ye = 96+8     # close to -15 degrees    # 66 for 30N
xb = 1
xe = 288

U, V, rho, N, lon, lat, Z, ugw, vgw, p = load_waccm_data([xb,yb,zb,tb],[xe,ye,ze,te],wtype="OFC");  # all GW is "OFC"