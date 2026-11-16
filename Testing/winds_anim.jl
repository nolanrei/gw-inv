using Plots

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
getp(ti) = 1e3*Float64.(Array(sqrt.(wrf_ds[:UW][:, :, 17, ti].^2 .+ wrf_ds[:VW][:, :, 17, ti].^2)))     # (x, y) matrix, plain Array

lon = parent(wrf_ds[:lon])[:, 1]      # lon varies along X (dim 1)
lat = parent(wrf_ds[:lat])[1, :]      # lat varies along Y (dim 2)

# fixed color scale across frames, clipped so a few extreme cells don't wash everything out
#pall = Array(wrf_ds[:precip])
pall = 1e3*Float64.(Array(sqrt.(wrf_ds[:UW][:, :, 17, :].^2 .+ wrf_ds[:VW][:, :, 17, :].^2)))
pmax = quantile(vec(pall), 0.995)
pall = nothing

minx = minimum(xx)
maxx = maximum(xx)
miny = minimum(yy)
maxy = maximum(yy)

anim = @animate for ti in 1:10:nt
    u = vec(get2d(wrf_ds[:U], ti))
    v = vec(get2d(wrf_ds[:V], ti))

    # heatmap wants rows = y, columns = x, so transpose
    heatmap(lon, lat, getp(ti)';
            color = :Blues, clims = (0, pmax),
            aspect_ratio = 1, xlims=(minx,maxx), ylims=(miny,maxy),
            colorbar_title = "flux (mPa)", title = "t = $ti")

    quiver!(xx, yy; quiver = (scale .* u, scale .* v), color = :black, label = "")
end
gif(anim, "Testing/c$(case)/winds_flux.gif"; fps = 8)