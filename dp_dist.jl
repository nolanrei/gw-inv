include("wrf-utils.jl")

function brute_force_dist(rate::AbstractMatrix{Float64}, x_km::AbstractMatrix{Float64}, y_km::AbstractVector{Float64}; thresh=4.0)
    nx, ny = size(rate)
    sources = [(i,j) for i in 1:nx, j in 1:ny if rate[i,j] > thresh]
    dist, strength = fill(NaN, nx, ny), fill(NaN, nx, ny)
    for i in 1:nx, j in 1:ny
        best, bestval = Inf, NaN
        for (si, sj) in sources
            d = abs(x_km[i,j]-x_km[si,j]) + abs(y_km[j]-y_km[sj])
            if d < best
                best, bestval = d, rate[si,sj]
            end
        end
        isfinite(best) && ((dist[i,j], strength[i,j]) = (best, bestval))
    end
    return dist, strength
end

#=
sz = size(wrf_ds[:precip])
precip = zeros(sz[1], sz[2],1)
precip[20,25] = 8.0
lon_data = wrf_ds[:lon]
lat_data = wrf_ds[:lat]
=#
sz = (10,10)
precip = zeros(sz[1],sz[2],1)
precip[5,5,1] = 8.0
lon_data = Array(1.0:1.0:10.0)*ones(sz[2])'
lat_data = ones(sz[1])*Array(-5.0:1.0:4.0)'


"""
    dt1d_l1_labeled!(f, pos, label)

In-place 1D L1 (taxicab) distance transform. `f[p]=0.0` at sources, `Inf`
elsewhere going in. Forward pass propagates the best cost rightward
(cost to reach q from the best candidate at or to its left); backward
pass does the same leftward. Combined, every q ends up with the true
min over ALL candidates -- min-plus arithmetic, so Inf can never turn
into NaN the way it did with the parabola/division approach.
"""
function dt1d_l1_labeled!(f::AbstractVector{Float64}, pos::AbstractVector{Float64}, label::AbstractVector{T}) where T
     @assert issorted(pos) "dt1d_l1_labeled!: pos must be sorted increasing (got non-monotonic input -- check your lon/lat axis convention)"
    n = length(f)
    @inbounds for q = 2:n
        cand = f[q-1] + (pos[q]-pos[q-1])
        if cand < f[q]
            f[q] = cand
            label[q] = label[q-1]
        end
    end
    @inbounds for q = n-1:-1:1
        cand = f[q+1] + (pos[q+1]-pos[q])
        if cand < f[q]
            f[q] = cand
            label[q] = label[q+1]
        end
    end
    return f, label
end

function convective_distance_strength_slice!(dist::AbstractMatrix{Float64}, strength::AbstractMatrix{Float64},
        rate::AbstractMatrix{Float64}, x_km::AbstractMatrix{Float64}, y_km::AbstractVector{Float64};
        thresh::Float64=4.0)
    nx, ny = size(rate)
    fcol, lcol = zeros(ny), zeros(ny)
    g, gval = zeros(nx, ny), zeros(nx, ny)

    for i = 1:nx
        for j = 1:ny
            fcol[j], lcol[j] = rate[i,j] > thresh ? (0.0, rate[i,j]) : (Inf, NaN)
        end
        dt1d_l1_labeled!(fcol, y_km, lcol)
        g[i, :] .= fcol
        gval[i, :] .= lcol
    end

    frow, lrow = zeros(nx), zeros(nx)
    for j = 1:ny
        frow .= view(g, :, j); lrow .= view(gval, :, j)
        dt1d_l1_labeled!(frow, view(x_km, :, j), lrow)
        for i = 1:nx
            dist[i,j], strength[i,j] = isnan(lrow[i]) ? (NaN, NaN) : (frow[i], lrow[i])
        end
    end
    return nothing
end

# conv_dist
conv_dist, conv_strength = convective_distance_strength(precip, lon_data, lat_data; thresh=4.0)