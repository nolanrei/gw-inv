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
tb = 30 #rand(1:(2920-nt))  #800
te = tb + nt
zb = 5      
ze = 65      
yb = 96-8      # close to +15 degrees    # 33 for 60N
ye = 96+8     # close to -15 degrees    # 66 for 30N
xb = 1
xe = 288

waccm_ds = load_waccm_stack([xb, yb, zb, tb], [xe, ye, ze, te]; wtype="O");  # all GW is "OFC"