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
using DSP

include("waccm-utils.jl")
include("wrf-utils.jl")
include("L81.jl")
include("err_in_mean.jl")

tb = 1
te = 2900
zb = 5      
ze = 65      
yb = 96      # close to +15 degrees    # 33 for 60N
ye = 96     # close to -15 degrees    # 66 for 30N
xb = 1
xe = 288

waccm_ds = load_waccm_stack([xb, yb, zb, tb], [xe, ye, ze, te]; wtype="OFC");  # all GW is "OFC"

nx = xe-xb+1
ny = ye-yb+1
nz = ze-zb+1
nt = te-tb+1
nzp = nz+2

fpath = "/home/nr2489/work/gw-inv/equator_tall_1yr.nc"
isfile(fpath) && rm(fpath)  # checks if the file exists, and if it does, removes it
nccreate(fpath, "U", "x",nx, "y",ny, "z",nz, "t",nt; atts=Dict("units" => "m/s"))
nccreate(fpath, "V", "x",nx, "y",ny, "z",nz, "t",nt; atts=Dict("units" => "m/s"))
nccreate(fpath, "rho", "x",nx, "y",ny, "zp",nzp, "t",nt; atts=Dict("units" => "kg/m^3"))
nccreate(fpath, "N", "x",nx, "y",ny, "z",nz, "t",nt; atts=Dict("units" => "s^-1"))
nccreate(fpath, "Z", "x",nx, "y",ny, "z",nz, "t",nt; atts=Dict("units" => "m"))
nccreate(fpath, "uw", "x",nx, "y",ny, "z",nz, "t",nt; atts=Dict("units" => "Pa"))
nccreate(fpath, "vw", "x",nx, "y",ny, "z",nz, "t",nt; atts=Dict("units" => "Pa"))
#nccreate(fpath, "p", "x",nx, "y",ny, "zp",nzp, "t",nt; atts=Dict("units" => "N/m^2"))
nccreate(fpath, "lat", "y",ny; atts=Dict("units" => "deg"))
nccreate(fpath, "lon", "x",nx; atts=Dict("units" => "deg"))

ncwrite(parent(waccm_ds[:U]), fpath, "U")
ncwrite(parent(waccm_ds[:V]), fpath, "V")
ncwrite(parent(waccm_ds[:rho]), fpath, "rho")
ncwrite(parent(waccm_ds[:N]), fpath, "N")
ncwrite(parent(waccm_ds[:z]), fpath, "Z")
ncwrite(parent(waccm_ds[:UW]), fpath, "uw")
ncwrite(parent(waccm_ds[:VW]), fpath, "vw")
#ncwrite(parent(waccm_ds[:p]), fpath, "p")
ncwrite(parent(waccm_ds[:lat][1:ye-yb+1]), fpath, "lat")
ncwrite(parent(waccm_ds[:lon][1:xe-xb+1]), fpath, "lon")

#=
U, V, rho, N, lon, lat, Z, ugw, vgw, p = load_waccm_data([xb,yb,zb,tb],[xe,ye,ze,te],wtype="O");  # all GW is "OFC"
nccreate(fpath, "ugw_oro", "x",nx, "y",ny, "z",nz, "t",nt; atts=Dict("units" => "m/s^2"))
nccreate(fpath, "vgw_oro", "x",nx, "y",ny, "z",nz, "t",nt; atts=Dict("units" => "m/s^2"))
ncwrite(ugw, fpath, "ugw_oro")
ncwrite(vgw, fpath, "vgw_oro")
=#