using Plots

c1f = load_ensemble_from_netcdf("full_recon.nc")

ssq = enstructure(c1f.spectra, c1f.xs, c1f.ys, c1f.ts)
ssq3d = dropdims(sum(ssq[:,:,3:3:end,1:12:end],dims=3),dims=3)

width = 6

nx, ny, _, nt = size(wrf_ds[:U])
sk = 2                                  # plot every sk-th grid point so arrows stay legible
ix, iy = 1:sk:nx, 1:sk:ny
lon2 = Float64.(Array(wrf_ds[:lon]))      # (nx, ny) if lon/lat are 2D
lat2 = Float64.(Array(wrf_ds[:lat]))
xx = vec(lon2[ix, iy])                    # same ix, iy, and order as get2d, so they line up with u and v
yy = vec(lat2[ix, iy])

lev = 11
get2d(A, ti) = Float64.(Array(A[ix, iy, lev, ti]))     # plain Matrix, no DimArray baggage

# fixed arrow scale across frames, so changes in arrow length are real changes in wind
umax = maximum(hypot.(get2d(wrf_ds[:U], 1), get2d(wrf_ds[:V], 1)))
dlon = abs(lon2[2, 1] - lon2[1, 1])       # approximate spacing, assumes a near-regular grid
scale = 1.5 * sk * dlon / umax            # strongest arrow is roughly 1.5 plotted cells long

#getp(ti) = Float64.(Array(wrf_ds[:precip][:, :, ti]))     # (x, y) matrix, plain Array
getp(ti) = 1e3*ssq3d[:,:,(ti-width-1)÷(2width)+1]

#getrain(ti) = dropdims(maximum(Float64.(Array(wrf_ds[:precip][:, :, max(1,ti-width):min(nt,ti+width)])), dims=3), dims=3)
getolr(ti) = dropdims(minimum(Float64.(Array(wrf_ds[:olr][:, :, max(1,ti-width):min(nt,ti+width)])), dims=3), dims=3)

olr_lo, olr_hi = 100.0, 350.0          # anything bracketing the levels you want
to_c(o) = pmax * (clamp(o, olr_lo, olr_hi) - olr_lo) / (olr_hi - olr_lo)

lon = parent(wrf_ds[:lon])[:, 1]      # lon varies along X (dim 1)
lat = parent(wrf_ds[:lat])[1, :]      # lat varies along Y (dim 2)

# fixed color scale across frames, clipped so a few extreme cells don't wash everything out
#pall = Array(wrf_ds[:precip])
pall = 1e3*ssq3d
pmax = quantile(vec(pall), 0.995)
pall = nothing

minx = minimum(xx)
maxx = maximum(xx)
miny = minimum(yy)
maxy = maximum(yy)

anim = @animate for ti in 1:2width:nt
    u = vec(get2d(wrf_ds[:U], ti))
    v = vec(get2d(wrf_ds[:V], ti))
    #r = getrain(ti)'
    r = to_c.(getolr(ti))'

    # heatmap wants rows = y, columns = x, so transpose
    heatmap(lon, lat, getp(ti)';
            color = :Blues, clims = (0, pmax),
            aspect_ratio = 1, xlims=(minx,maxx), ylims=(miny,maxy),
            colorbar_title = "flux (mPa)", title = "t = $ti")

    contour!(lon, lat, r; levels = [to_c(200)], color = :red,   linewidth = 1.5,
             colorbar_entry = false, label = "")
    contour!(lon, lat, r; levels = [to_c(170)], color = :orange, linewidth = 1.5,
             colorbar_entry = false, label = "")

    quiver!(xx, yy; quiver = (scale .* u, scale .* v), color = :black, label = "")
end
gif(anim, "winds_flux_recon.gif"; fps = 8)