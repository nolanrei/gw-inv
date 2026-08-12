using NetCDF
using Interpolations
using Statistics

# save what I'm doing to get the rho and N data for WRF
rho_eq = ncread("equator_tall_1yr.nc","rho")       # kg/m^3
z_eq = ncread("equator_tall_1yr.nc","Z")          # m
N_eq = ncread("equator_tall_1yr.nc","N")          # s^-1

N_eq_mn = dropdims(mean(N_eq,dims=(1,2,4)),dims=(1,2,4))
z_eq_mn = dropdims(mean(z_eq,dims=(1,2,4)),dims=(1,2,4))
rho_eq_mn = dropdims(mean(rho_eq,dims=(1,2,4)),dims=(1,2,4))
rho_eq_mn = rho_eq_mn[1:end-2]

# target z grid
zt = (10:0.5:65)
nz = length(zt)

# interpolate onto target grid
# unfortunately only linear interpolation is available on 
itp_rho = interpolate((z_eq_mn,),rho_eq_mn,Gridded(Linear()))
itp_N = interpolate((z_eq_mn,),N_eq_mn,Gridded(Linear()))
rho = itp_rho.(zt)
N = itp_N.(zt)

fpath = "/scratch/nr2489/work/loose_spectra/waccm_rho_and_N.nc"
isfile(fpath) && rm(fpath)  # checks if the file exists, and if it does, removes it
nccreate(fpath, "rho", "z",nz; atts=Dict("units" => "kg/m^3"))
nccreate(fpath, "N", "z",nz; atts=Dict("units" => "s^-1"))

ncwrite(rho, fpath, "rho")
ncwrite(N, fpath, "N")