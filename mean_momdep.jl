using NetCDF
using CSV
using DataFrames

include("waccm-utils.jl")

## OBJECTIVE: loop over all WACCM data and compile mean(x,y,z) and stdev(x,y,z) and mean(z), stdev(z)
function calc_mean_and_var(nt, tstride=64)
    zb = 1
    ze = 68
    yb = 1
    ye = 192
    xb = 1
    xe = 288

    nx = (xe-xb+1)
    ny = (ye-yb+1)
    nz = 2*(ze-zb+1)
    
    EX  = zeros(nx, ny, nz)        # accumulator for E[X]
    var = zeros(nx, ny, nz)        # accumulator for var
    momdep = zeros(nx, ny, nz, tstride)
    
    for i = 1:Int(floor(nt/tstride))
        momdep .= load_momdep([xb,yb,zb,tstride*(i-1)+1], [xe,ye,ze,tstride*i])
        EX .+= dropdims(sum(momdep,dims=4),dims=4)
    end
    EX ./= nt
    
    for i = 1:Int(floor(nt/tstride))
        momdep .= load_momdep([xb,yb,zb,tstride*(i-1)+1], [xe,ye,ze,tstride*i])
        var .+= dropdims(sum((momdep.-EX).^2,dims=4),dims=4)
    end
    var ./= (nt-1)
    
    meanz = dropdims(sum(EX,dims=[1,2]),dims=(1,2))/(nx*ny)
    varz = dropdims(sum(var,dims=[1,2]),dims=(1,2))/(nx*ny)

    return EX, sqrt.(var), meanz, sqrt.(varz)
end


tstride = 73
nt = 2920

meanfull, stdfull, meanz, stdz = calc_mean_and_var(nt, tstride)
println(size(meanfull))
println(size(meanz))
nx, ny, nz = size(meanfull)
attribs = Dict("units" => "m/s^2")

isfile("mean_and_std.nc") && rm("mean_and_std.nc")  # checks if the file exists, and if it does, removes it
nccreate("mean_and_std.nc", "meanfull", "lon", nx, "lat", ny, "lev", nz; atts=attribs)
nccreate("mean_and_std.nc", "stdfull", "lon", nx, "lat", ny, "lev", nz; atts=attribs)
nccreate("mean_and_std.nc", "meanz", "lev", nz; atts=attribs)
nccreate("mean_and_std.nc", "stdz", "lev", nz; atts=attribs)
ncwrite(meanfull, "mean_and_std.nc", "meanfull")
ncwrite(stdfull, "mean_and_std.nc", "stdfull")
ncwrite(meanz, "mean_and_std.nc", "meanz")
ncwrite(stdz, "mean_and_std.nc", "stdz")