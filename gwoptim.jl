using NetCDF

include("L81.jl")
include("mcmc.jl")

input_dir = "/scratch/projects/gerberlab/WRF/INPUT/"
case = "input_1deg_case1/"
dayfile = "2016-08-05.nc"

nc = NetCDF.open(input_dir*case*dayfile)

x = 20
y = 20
t = 1
nlev = size(nc["u1deg"],3)
N = "N somehow"     #TODO
rho = "rho somehow" #TODO
#col = ColumnProfile(nc["u1deg"][x,y,:,t], nc["v1deg"][x,y,:,t], N, rho, nlev)
