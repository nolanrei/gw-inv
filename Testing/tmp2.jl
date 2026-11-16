case=10
wrf_ds = load_wrf_case(case)
z = wrf_ds[:z]
zr = wrf_ds[:iz]
cnv = load_all_ensembles(2,prefix="Testing/ensemble_results_conv_c$(case)_");

function get_launch_z(dat,ds)
    TINY = 1e-20
    xs,ys,ts = dat.xs,dat.ys,dat.ts
    nsamp = length(xs)
    
    pi = zeros(nz)
    sth = zeros(2)
    for i = 1:nsamp
        xi,yi,ti = xs[i],ys[i],ts[i]
        nw = div(dat.nxs[i],3)
        fill!(sth,0.0)
        for j = 1:nw
            th = dat.spectra[i,3j-2]
            f  = dat.spectra[i,3j]
            sth[1] += f*cos(th)
            sth[2] += f*cos(th)
        end
        sf = sum(dat.spectra[i,3:3:3nw]) + TINY
        sth /= sf
        for j = 1:nz
            u = ds[:U][xi,yi,j,ti]
            v = ds[:V][xi,yi,j,ti]
            vv = hypot(u,v) + TINY
            u /= vv
            v /= vv
            pi[j] += u*sth[1] + v*sth[2]  # should be proportional to flux*cos(angle)
        end
    end
    return pi/nsamp
end

pi = get_launch_z(cnv[2],wrf_ds)
plot(pi,z,xlabel="dot(wind angle, wave angle)",ylabel="z (km)");
savefig("Testing/pi_$(case)_far.png")