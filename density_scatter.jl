using LinearAlgebra
using Plots

"""
    rbf_density(x, y; radius, weights=nothing, n=200, xlims=nothing, ylims=nothing,
                normalize=false, chunk=4096) -> (gx, gy, dens)

Sum of Gaussian RBFs centred on the points (x[k], y[k]), evaluated on a regular grid:

    dens[j, i] = sum_k w[k] * exp(-(gx[i]-x[k])^2 / (2rx^2) - (gy[j]-y[k])^2 / (2ry^2))

- `radius`: the Gaussian standard deviation σ, in the data's units. Pass a number for an
  isotropic kernel, or a tuple `(rx, ry)` when x and y have different units/scales.
  (For comparison, rosette_heatmap's `exp(-ep^2 r^2)` corresponds to σ = 1/(ep*√2),
  so ep = 0.3 ≈ σ = 2.36.)
- `weights`: optional per-point masses (e.g. fluxes). Defaults to 1.
- `n`: grid resolution; an Int, or `(nx, ny)`.
- `xlims`, `ylims`: grid extent; default is the data range padded by 3σ.
- `normalize=true`: divide by total mass × 2π rx ry, so `dens` integrates to 1 (a KDE).

`dens` is ny × nx (rows = y), which is what `heatmap(gx, gy, dens)` expects.

The Gaussian is separable, so the grid evaluation is a matrix product
`dens = Ey * (w .* Ex)'`, costing O(nx*ny*N) flops in BLAS rather than a
quadruple loop. Points are processed in chunks so memory stays O((nx+ny)*chunk).
"""
function rbf_density(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
        radius, weights=nothing, n=200, xlims=nothing, ylims=nothing,
        normalize::Bool=false, chunk::Int=4096)

    N = length(x)
    @assert length(y) == N "x and y must have the same length"
    isnothing(weights) || @assert length(weights) == N "weights must match length of x"

    rx, ry = radius isa Real ? (radius, radius) : (radius[1], radius[2])
    nx, ny = n isa Integer ? (n, n) : (n[1], n[2])

    xlo, xhi = isnothing(xlims) ? (minimum(x) - 3rx, maximum(x) + 3rx) : xlims
    ylo, yhi = isnothing(ylims) ? (minimum(y) - 3ry, maximum(y) + 3ry) : ylims
    gx = range(xlo, xhi, length=nx)
    gy = range(ylo, yhi, length=ny)

    sx = -0.5 / rx^2
    sy = -0.5 / ry^2

    dens = zeros(ny, nx)
    Ex = zeros(nx, min(chunk, N))
    Ey = zeros(ny, min(chunk, N))

    for c0 in 1:chunk:N
        c1 = min(c0 + chunk - 1, N)
        m = c1 - c0 + 1
        Exv = view(Ex, :, 1:m)
        Eyv = view(Ey, :, 1:m)
        @inbounds for (j, k) in enumerate(c0:c1)
            wk = isnothing(weights) ? 1.0 : Float64(weights[k])
            xk, yk = x[k], y[k]
            for i in 1:nx
                Exv[i, j] = wk * exp(sx * (gx[i] - xk)^2)
            end
            for i in 1:ny
                Eyv[i, j] = exp(sy * (gy[i] - yk)^2)
            end
        end
        mul!(dens, Eyv, transpose(Exv), 1.0, 1.0)   # dens += Ey * Ex'
    end

    if normalize
        mass = isnothing(weights) ? N : sum(weights)
        dens ./= mass * 2pi * rx * ry
    end

    return gx, gy, dens
end

"""
    density_scatter(x, y; radius, weights=nothing, n=200, xlims=nothing, ylims=nothing,
                    normalize=false, kwargs...)

Drop-in replacement for `scatter(x, y)` when points overplot: draws the RBF-summed
density as a heatmap. Extra `kwargs` go to `heatmap` (color, clims, xlabel, ...).
"""
function density_scatter(x, y; radius, weights=nothing, n=200,
        xlims=nothing, ylims=nothing, normalize=false, kwargs...)
    gx, gy, dens = rbf_density(x, y; radius, weights, n, xlims, ylims, normalize)
    return heatmap(gx, gy, dens; color=:viridis, kwargs...)
end

"""
    density_scatter!(p, x, y; ...)

Same as `density_scatter`, drawn onto an existing plot `p`. If xlims/ylims aren't
given, the grid matches `p`'s current axis limits so the layers line up.
"""
function density_scatter!(p, x, y; radius, weights=nothing, n=200,
        xlims=nothing, ylims=nothing, normalize=false, kwargs...)
    xlims = isnothing(xlims) ? Plots.xlims(p) : xlims
    ylims = isnothing(ylims) ? Plots.ylims(p) : ylims
    gx, gy, dens = rbf_density(x, y; radius, weights, n, xlims, ylims, normalize)
    return heatmap!(p, gx, gy, dens; color=:viridis, kwargs...)
end