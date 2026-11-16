cases_sampled = [1 2 3 6 10] #5 

function source_n_cases(cases_sampled)
    TINY = 1e-16
    nc = length(cases_sampled)
    cnv = load_all_ensembles(1,prefix="Testing/ensemble_results_conv_c1_")
    nsamp = size(cnv[1].spectra,1)
    zr = wrf_ds[:iz]
    nzr = length(zr)
    
    vecs = zeros(nc*nsamp, nzr)
    for (ci,case) in enumerate(cases_sampled)
        println("case = ", case)
        dat = load_all_ensembles(2,prefix="Testing/ensemble_results_conv_c$(case)_")[2]
        ds = load_wrf_case(case)
        z = wrf_ds[:z]
        zr = wrf_ds[:iz]
        nz = length(z)
        nzr = length(zr)

        sth = zeros(2)
        for i = 1:nsamp
            xi,yi,ti = dat.xs[i],dat.ys[i],dat.ts[i]
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
            for j = 1:nzr
                u = ds[:U_raw][xi,yi,j,ti]
                v = ds[:V_raw][xi,yi,j,ti]
                vv = hypot(u,v) + TINY
                u /= vv
                v /= vv
                vecs[i+(ci-1)*nsamp, j] = u*sth[1] + v*sth[2]  # should be proportional to flux*cos(angle)
            end
        end
    end
    mnpi = mean(vecs,dims=1)
    stpi = std(vecs,dims=1)
    return mnpi,stpi
end

mnpi,stpi = source_n_cases(cases_sampled)

begin
zr = vec(collect(wrf_ds[:iz]))   # plain Vector, no Dimensions
μ  = vec(collect(mnpi))          # flattens 1×nz or nz×1 to a Vector
σ  = vec(collect(stpi))

band = Shape(vcat(μ .- 2σ, reverse(μ .+ 2σ)),
             vcat(zr,      reverse(zr)))

plot(band; fillalpha = 0.3, linealpha = 0, color = :blue, label = "±2σ")
plot!(μ, zr; color = :blue, label = "mean")
savefig("Testing/pi_all_far.png")
end