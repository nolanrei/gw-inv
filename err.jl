include("L81.jl")
include("fullcalc.jl")

Ntasks = 80
task_id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])

tmin = 1
tmax = 2919  # full year
zb = 2       # bottom
ze = 68      # top
yb = 1
ye = 192
xb = 1
xe = 288

nt = div((tmax-tmin+1),Ntasks)
tb = nt*task_id+1
te = nt*task_id+nt

wavespds = Array(-100:100)
nw = length(wavespds)
ndirs = 4
#srcs = [5] # ~800hPa
#srcs = [8] # ~600hPa
#srcs = [9] # ~500hPa
#srcs = [11] # ~370hPa
srcs = [15] # ~200hPa
waves = make_waveset(wavespds, ndirs, srcs)
fname = "/scratch/nr2489/work/spectra/spectra200_$(task_id).nc"
fullcalc_nospec(waves; lims=[xb,xe,yb,ye,zb,ze,tb,te], tstride=10, fname=fname)