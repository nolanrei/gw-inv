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
using Distributions

include("waccm-utils.jl")
include("wrf-utils.jl")
include("L81-2.jl")

######################################################################################
##
######################################################################################
case = 10#rand(1:10)
println("case = $case")
wrf_ds = load_wrf_case(case)
z = wrf_ds[:z]
nz = length(z)
dz = 500*ones(nz)       # each cell is 500m thick in z

wrf_ds
