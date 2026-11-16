# Requires running angle_wrf.jl first

include("find_angle.jl")
#fluxvec = vcat(uw, vw)    # but what if instead we ran run_synth_ma.jl first?

th, brk, thsm = div_by_angle(fluxvec)
angles,flx0,th0 = find_angle(fluxvec,thsm,[1 47 111])

pcws = zeros(nz)
flx = zeros(nz,2)
na = length(angles)
for i = na:-1:1
    a = angles[i][3]
    f = angles[i][4]
    j = angles[i][2]
    flx[1:j,1:2] .+= f*[cos(a) sin(a)]
end
# this doesn't quite work since the angles are now discontinuous
angle_no_brcuts!(pcws,flx[:,1],flx[:,2])
# for plotting purposes only calculate right winding number by comparing to th
for i = 1:na
    bp = angles[i][1]
    bn = angles[i][2]
    thtrue = mean(th[bp:bn])
    delta = thtrue - pcws[bp+1]
    winding_num = round(delta/(2pi))
    pcws[bp+1:bn] .+= 2*pi*winding_num
end

plot(th,z,xlabel="flux angle (radians)",ylabel="z (km)",label="theta")
plot!(thsm,z,label="theta smooth")
plot!(pcws,z,label="theta approx")
savefig("Figures/theta_chunked.png")

plot(reshape(fluxvec,(nz,2)),z,layout=2)
plot!(flx,z,layout=2)
savefig("testfig.png")