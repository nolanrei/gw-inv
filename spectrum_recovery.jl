# Framework for sampling test spectra from a "master distribution" over
# gravity-wave space (theta, c, flux), generating noisy fluxvecs, and
# comparing the ensemble-mean reconstructed spectrum against the analytic
# target density f0(cx, cy).
#
# Designed to plug straight into ensemble_accuracy(ds, retrieve_fluxvec!; ...)
# from ensemble_accuracy.jl, via retrieve_args = (spec_model, noise_model).
#
# Requires omp.jl / L81-2.jl to already be loaded (for OMPStruct, fwd!).

using Distributions
using LinearAlgebra
using Statistics
using Integrals

# ---------------------------------------------------------------------
# 1. Location proposals: how (theta, c) get sampled
# ---------------------------------------------------------------------

abstract type AbstractLocProposal end

"""
    rand_loc(p::AbstractLocProposal, U, V) -> (θ, c)

Draw one (angle, phase speed) pair from the proposal. Implement this for
any new proposal type.
"""
function rand_loc(p::AbstractLocProposal, U::AbstractVector, V::AbstractVector) end

"""
    pdf_loc(p::AbstractLocProposal, θ, c, U, V) -> Float64

Density of the proposal at (θ, c), with respect to dθ dc (NOT dcx dcy).
Only required for proposals used inside an ImportanceFluxModel.
"""
function pdf_loc(p::AbstractLocProposal, θ, c, U::AbstractVector, V::AbstractVector) end
"""
    DeltaLocProposal(x)

Delta function proposal -- N waves stored in (theta, c, cdf) order in x,
the ith sampled with probability cdf[i]-(0, cdf[i-1]).
NOTE: cdf needs to stored as cdf[0:n-1] rather than cdf[1:n] so that bsrch works properly
"""
struct DeltaLocProposal <: AbstractLocProposal
    x::Vector{Float64}
end
function rand_loc(p::DeltaLocProposal, U::AbstractVector, V::AbstractVector)
    r = rand()
    nx = length(p.x)
    i = bsrch(r, view(p.x,3:3:nx))
    return (p.x[3i-2], p.x[3i-1])
end
# only valid when called on a point actually produced by rand_loc(p) --
# returns the atom's probability MASS, not a density w.r.t. dθdc
function pdf_loc(p::DeltaLocProposal, θ, c, U::AbstractVector, V::AbstractVector)
    i = findfirst(k -> p.x[3k-2]==θ && p.x[3k-1]==c, 1:div(length(p.x),3))
    lo = i == 1 ? 0.0 : p.x[3i]
    hi = i == div(length(p.x),3) ? 1.0 : p.x[3(i+1)]
    return hi - lo
end
"""
    IndependentLocProposal(θdist, cdist)

Independent product of a Distributions.jl angle distribution and a
Distributions.jl speed distribution. 
"""
struct IndependentLocProposal{Dθ,Dc} <: AbstractLocProposal
    θdist::Dθ
    cdist::Dc
end
rand_loc(p::IndependentLocProposal, U, V) = (rand(p.θdist), rand(p.cdist))
pdf_loc(p::IndependentLocProposal, θ, c, U, V) = pdf(p.θdist, θ) * pdf(p.cdist, c)
"""
    IndependentCartesianLocProposal(xdist, ydist)

Independent product of Distributions.jl cx and cy distributions.
"""
struct IndependentCartesianLocProposal{Dx,Dy} <: AbstractLocProposal
    xdist::Dx
    ydist::Dy
end
function rand_loc(p::IndependentCartesianLocProposal, U::AbstractVector, V::AbstractVector)
    # doesn't use U/V
    x = rand(p.xdist)
    y = rand(p.ydist)
    theta = atan(y,x)
    c = hypot(x,y)
    return (theta, c)
end
function pdf_loc(p::IndependentCartesianLocProposal, th, c, U::AbstractVector, V::AbstractVector)
    # doesn't use U/V
    x = c*cos(th)
    y = c*sin(th)
    pxy = pdf(p.xdist,x) * pdf(p.ydist,y)
    return pxy * c    # Jacobian correction
end

"""
    RejectionLocProposal(f0, θlo, θhi, clo, chi, fmax)

Rejection sampler that draws (θ, c) directly proportional to a
user-supplied density f0(θ,c), given a bounding box and an upper bound
fmax ≥ max(f0) on that box. This is the turnkey way to build a
ProportionalFluxModel from an arbitrary f0 without knowing its
normalizing constant.

DOES NOT WORK with f0 that depends on (U,V) as well -- look
at RejectionLocProposalFamily for that.
"""
struct RejectionLocProposal{F} <: AbstractLocProposal
    f0::F
    θlo::Float64; θhi::Float64
    clo::Float64; chi::Float64
    fmax::Float64
    fsum::Float64              # normalization constant
end
function RejectionLocProposal(f0::F,
        θlo::Float64, θhi::Float64,
        clo::Float64, chi::Float64,
        fmax::Float64) where F
    # f0 is a 2D function -- we can integrate it to get the normalization constant
    integrand = (u, p) -> f0(u[1], u[2])
    domain = ([θlo, clo], [θhi, chi])
    prob = IntegralProblem(integrand, domain)
    sol = solve(prob, HCubatureJL(), reltol = 1e-3, abstol = 1e-3)
    return RejectionLocProposal{F}(f0,θlo,θhi,clo,chi,fmax,sol.u)
end
function rand_loc(p::RejectionLocProposal, U::AbstractVector, V::AbstractVector)
    while true
        θ = p.θlo + (p.θhi - p.θlo) * rand()
        c = p.clo + (p.chi - p.clo) * rand()
        if rand() * p.fmax <= p.f0(θ, c)
            return θ, c
        end
    end
end
pdf_loc(p::RejectionLocProposal, θ, c, U, V) = p.f0(θ, c) / p.fsum

abstract type AbstractLocProposalFamily end

"""
    condition(family_or_proposal, col::ColumnProfile) -> AbstractLocProposal

Instantiate a column-specific proposal. For ordinary (column-independent)
proposals this is just the identity -- they're already usable as-is.
"""
condition(p::AbstractLocProposal, U::AbstractVector, V::AbstractVector) = p

"""
    ColumnDependentRejectionFamily(f0, θlo,θhi,clo,chi,fmax)

Like RejectionLocProposal, but f0(θ,c,U,V) depends on the column's
velocity profile too (e.g. a Beres 2004/2005-style convective source
spectrum, which depends on source-level wind). Since the normalization
constant changes with every column, this type is NOT itself an
AbstractLocProposal -- call `condition(family, col)` once per column to
get back an ordinary, immutable RejectionLocProposal for that column.
"""
struct ColumnDependentRejectionFamily{F} <: AbstractLocProposalFamily
    f0::F     # f0(θ, c, U, V) -- full column vectors; f0 itself decides what part of U,V it needs (e.g. just U[src], V[src])
    θlo::Float64; θhi::Float64
    clo::Float64; chi::Float64
    fmax::Float64
end

function condition(fam::ColumnDependentRejectionFamily, U::AbstractVector, V::AbstractVector)
    f0_col = (θ, c) -> fam.f0(θ, c, U, V)
    integrand = (u, p) -> f0_col(u[1], u[2])
    domain = ([fam.θlo, fam.clo], [fam.θhi, fam.chi])
    prob = IntegralProblem(integrand, domain)
    sol = solve(prob, HCubatureJL(), reltol=1e-3, abstol=1e-3)
    # calls the raw 7-arg struct constructor directly -- bypasses the
    # integrating outer constructor since we already have fsum
    return RejectionLocProposal(f0_col, fam.θlo, fam.θhi, fam.clo, fam.chi, fam.fmax, sol.u)
end

# ---------------------------------------------------------------------
# 2. Spectrum models
# ---------------------------------------------------------------------

abstract type SpectrumModel end

"""
    ProportionalFluxModel(loc_sampler, f0, total_flux, n_dist)

"Constant weight" scheme: wave locations are drawn with probability
density proportional to f0(θ,c); every drawn wave gets the same 
flux amplitude w = total_flux / mean(n_dist).

`f0` is kept only for plotting/validation via target_density_cxcy — it is
NOT used to compute weights here. `loc_sampler` is responsible for
actually sampling proportionally to it (RejectionLocProposal is a
turnkey option).

n_dist can be any Distributions.jl object supporting rand/mean
(Poisson(λ) for a random count, Dirac(n) for a fixed count, etc).
"""
struct ProportionalFluxModel{P,F,ND} <: SpectrumModel   # P: AbstractLocProposal or ...Family
    loc_sampler::P
    f0::F
    total_flux::Float64
    n_dist::ND
end

"""
    ImportanceFluxModel(proposal, f0, n_dist)

"Importance weighted" scheme: wave locations are drawn from an 
arbitrary proposal g(θ,c) (must support pdf_loc); each wave's 
flux is set to f0(θ,c)/g(θ,c), renormalized by mean(n_dist).

By the importance-sampling identity this converges to (a smoothed
version of) f0 for ANY choice of proposal — use this when it's
inconvenient to sample directly proportional to f0 (multimodal / very
peaked targets), or when you want one fixed proposal reused across many
different f0's.
"""
struct ImportanceFluxModel{P,F,ND} <: SpectrumModel     # P: AbstractLocProposal or ...Family
    proposal::P
    f0::F
    n_dist::ND
end

"""
    sample_spectrum(model) -> (n, x)

Draws a full spectrum. x is a length-3n vector of (θ,c,flux) triples,
directly usable by fwd!/get_res!/find_measure!. n is the wave count.
"""
function sample_spectrum(m::ProportionalFluxModel, U::AbstractVector, V::AbstractVector)
    prop = condition(m.loc_sampler, U, V)   # no-op for ordinary proposals
    n = rand(m.n_dist)
    nbar = mean(m.n_dist)
    w = m.total_flux / nbar
    x = zeros(3n)
    for i in 1:n
        θ, c = rand_loc(prop, U, V)
        x[3i-2], x[3i-1], x[3i] = θ, c, w
    end
    return n, x
end

function sample_spectrum(m::ImportanceFluxModel, U::AbstractVector, V::AbstractVector)
    prop = condition(m.proposal, U, V)
    n = rand(m.n_dist)
    nbar = mean(m.n_dist)
    x = zeros(3n)
    for i in 1:n
        θ, c = rand_loc(prop, U, V)
        q = pdf_loc(prop, θ, c, U, V)
        x[3i-2] = θ; x[3i-1] = c
        x[3i]   = m.f0(θ, c, U, V) / q / nbar
    end
    return n, x
end

"""
    target_density_cxcy(f0, cx, cy) -> Float64

Converts the (θ,c)-space target density f0 (with ∫∫ f0 dθ dc = mean total
flux) into (cx,cy)-space via the polar Jacobian, so it can be overlaid on
the same grid as the RBF-smoothed ensemble mean:

    f0_cxcy(cx,cy) = f0(θ,c) / c,   θ = atan(cy,cx),  c = hypot(cx,cy)
"""
function target_density_cxcy(f0, cx, cy, U::AbstractVector, V::AbstractVector)
    c = hypot(cx, cy)
    c == 0 && return 0.0
    θ = atan(cy, cx)
    return f0(θ, c, U, V) / c
end

"""
    NWaveNoResample(x)

Simplest possible dependent spectrum model -- constant spectrum,
always returns the exact same waves.
"""
struct NWaveNoResample <: SpectrumModel
    x::Vector{Float64}
end

function sample_spectrum(m::NWaveNoResample, U::AbstractVector, V::AbstractVector)
    nx = length(m.x)
    nw = div(nx,3)
    return (nw, m.x)
end

"""
    PlusMinusWaves(th_distrib, c, f)

Two waves at (th, c, f) and (th+pi, c, f). Selects th randomly from th_distrib.
"""
struct PlusMinusWaves{D} <: SpectrumModel
    th_distrib::D
    c::Float64
    f::Float64
end

function sample_spectrum(m::PlusMinusWaves, U::AbstractVector, V::AbstractVector)
    th = rand(th_distrib)
    return (2, [th, c, f, th+pi, c, f])
end

"""
    WavesInALine(th_distrib, cf_distrib, n)

Colinear spectrum. Select a direction with th_distrib, then select 
c and f for each of n points with cf_distrib.
"""
struct WavesInALine{D1,D2} <: SpectrumModel
    th_distrib::D1
    cf_distrib::D2
    n::Int64
end

function sample_spectrum(m::WavesInALine, U::AbstractVector, V::AbstractVector)
    th = rand(th_distrib)
    n = m.n

    x = zeros(3n)
    for i = 1:n
        c,f = rand(cf_distrib)
        x[3i-2] = (c > 0 ? th : th+pi)   # Keeps convention that c is positive
        x[3i-1] = abs(c)
        x[3i]   = f
    end
    return (n, x)
end

"""
    ContinuousMap(f, omega, grid)

Returns a discretization of a continuous map. Completely deterministic. 

Evaluates on a roughly 3m/s-square grid. If there is worry about behavior
of spectra when discretized finely, see MK99 tests (or MK99 itself).
"""
struct ContinuousMap{F} <: SpectrumModel
    f::F                                 # source density f(K, th, U, V) -> flux "mass" per unit area of (k,l)-space
    omega::Float64                       # single ground-frame frequency assumed for the whole spectrum (rad/s)
    grid::Vector{NTuple{3,Float64}}      # (K_i, th_i, dA_i): polar grid IN WAVENUMBER SPACE, dA_i = cell area in (k,l)
end

function ContinuousMap(f::F, omega::Float64;
        cmin::Float64 = 1.0,     # m/s -- smallest phase speed we resolve (sets the large-K cutoff)
        cmax::Float64 = 80.0,    # m/s -- largest phase speed we resolve (sets the small-K cutoff)
        nK::Int  = 25,
        nth::Int = 40,
        p::Float64 = 2.0         # >1 clusters K-resolution near Kmin, i.e. near the source's actual power
    ) where F

    @assert 0 < cmin < cmax "need 0 < cmin < cmax"
    Kmin = omega / cmax    # smallest K resolved -> maps to c = cmax
    Kmax = omega / cmin    # largest K resolved  -> maps to c = cmin

    xs = range(0.0, 1.0, nK)
    Ks = Kmin .+ (Kmax - Kmin) .* xs.^p     # clustered near Kmin (large c) -- see note below

    Kedges = Vector{Float64}(undef, nK+1)
    Kedges[1] = 0.0        # innermost cell's area reaches the true center K=0 (the source's actual peak),
                            # even though we never *evaluate* f closer than Ks[1]=Kmin
    for i in 2:nK
        Kedges[i] = 0.5*(Ks[i-1] + Ks[i])
    end
    Kedges[end] = Kmax      # fixed outer cutoff, same philosophy as the old cmax cutoff -- no attempt
                             # to capture residual mass in the truncated K > Kmax tail

    dth = 2pi/nth
    ths = range(0.0, 2pi - dth, nth)

    grid = Vector{NTuple{3,Float64}}(undef, nK*nth)
    idx = 1
    for i in 1:nK
        # d^2k = K dK dth  =>  cell area = (1/2)(K_hi^2 - K_lo^2) dth
        dA_radial = 0.5*(Kedges[i+1]^2 - Kedges[i]^2)
        for j in 1:nth
            grid[idx] = (Ks[i], ths[j], dA_radial*dth)
            idx += 1
        end
    end

    return ContinuousMap(f, omega, grid)
end

function sample_spectrum(m::ContinuousMap, U::AbstractVector, V::AbstractVector)
    n = length(m.grid)
    out = zeros(3n)
    for (i, (K, th, dA)) in enumerate(m.grid)
        mass = m.f(K, th, U, V) * dA         # evaluated where f is smooth, no coordinate singularity
        out[3i-2] = th                       # propagation direction survives the K -> c inversion unchanged
        out[3i-1] = m.omega / K              # c = omega/K, computed once, at the very end
        out[3i]   = mass
    end
    return (n,out)
end

# ---------------------------------------------------------------------
# 3. Noise models
# ---------------------------------------------------------------------

abstract type NoiseModel end

"""
    sample_noise(model, fluxvec) -> Vector{Float64}

Draws a noise vector the same length as fluxvec. Implement this for any
new noise model.
"""
function sample_noise(m::NoiseModel, fluxvec::AbstractVector) end

"""
    WhiteNoiseModel(frac=0.05)

Default/placeholder noise model: iid Gaussian noise scaled so its
*expected* 2-norm is `frac * norm(fluxvec)`. Since
E[‖σ·randn(n)‖] ≈ σ√n, we set σ = frac*‖fluxvec‖/√n.
"""
struct WhiteNoiseModel <: NoiseModel
    frac::Float64
    abs::Float64
end
WhiteNoiseModel() = WhiteNoiseModel(0.05,0.0)

function sample_noise(m::WhiteNoiseModel, fluxvec::AbstractVector)
    n = length(fluxvec)
    σ = m.frac * norm(fluxvec) / sqrt(n)
    return σ .* randn(n) .+ m.abs .* randn(n)
end

"""
    GPNoiseModel

    Red noise model using Matern kernel to mimic WRF data
"""

qp(tau; l, lam) = (1 + √3*abs(tau)/l) * exp(-√3*abs(tau)/l) * cos(2π*tau/lam)

struct GPNoiseModel <: NoiseModel
    L::LowerTriangular{Float64, Matrix{Float64}}
end

function GPNoiseModel(z, sx, sy; l=1.5, lam=3.0, rxy=0.0, jitter=1e-8)
    C = [qp(zi - zj; l, lam) for zi in z, zj in z]
    Σ = kron([1.0 rxy; rxy 1.0], C)          # matches your [x(z); y(z)] layout
    s = [sx; sy]
    Σ .= s .* Σ .* s'
    return GPNoiseModel(cholesky(Symmetric(Σ + jitter*maximum(s)^2*I)).L)
end

# absolute amplitude: ignores fluxvec
sample_noise(m::GPNoiseModel, fluxvec::AbstractVector) = m.L * randn(size(m.L, 1))

# ---------------------------------------------------------------------
# 4. retrieve_fluxvec! glue for ensemble_accuracy
# ---------------------------------------------------------------------

"""
    retrieve_fluxvec_from_model!(fluxvec, ds, xi, yi, ti, os, spec_model, noise_model)

Matches the retrieve_fluxvec! signature expected by ensemble_accuracy
(pass retrieve_args = (spec_model, noise_model)). Samples a spectrum from
spec_model, propagates it through os's current column via fwd!, adds
noise from noise_model, and returns the sampled (θ,c,flux) triples so
ensemble_accuracy can stash the ground truth in known_spectra.
"""
function retrieve_fluxvec_from_model!(fluxvec::Vector{Float64}, ds,
        xi::Int, yi::Int, ti::Int,
        os::OMPStruct,
        spec_model::SpectrumModel,
        noise_model::NoiseModel)
    U = view(ds[:U], xi, yi, :, ti)
    V = view(ds[:V], xi, yi, :, ti)
    n, waveset = sample_spectrum(spec_model, U, V)
    fill!(fluxvec, 0.0)
    fwd!(fluxvec, waveset, 3n, os)
    fluxvec .+= sample_noise(noise_model, fluxvec)
    return waveset
end

# ---------------------------------------------------------------------
# 5. Ensemble-mean RBF density (post-processing, cx-cy space)
# ---------------------------------------------------------------------

"""
    ensemble_mean_density(spectra, nxs, grid; ep=0.3)

`spectra`: nsamp x 3*max_nw matrix of recovered (θ,c,flux) triples, as
produced by ensemble_accuracy's `spectra` output.
`nxs`: length-nsamp vector of how many parameters (3*nwaves) each row uses.
`grid`: range for both cx and cy (assumed square/symmetric domain).

For each ensemble member, projects its waves into (cx,cy), smooths with
a Gaussian RBF of width `ep`, then averages the resulting maps across the
ensemble. This is the direct empirical analogue of target_density_cxcy,
and by construction (see the identity in the module docstring) it should
converge to a smoothed version of the true f0 as nsamp -> ∞, no matter
which SpectrumModel subtype generated the ground truth.

Note: this is an O(nsamp * grid^2 * nwaves) triple loop -- fine for
exploratory plotting, but worth vectorizing/binning if grid resolution or
nsamp grows large.
"""
function ensemble_mean_density(spectra::AbstractMatrix, nxs::AbstractVector{<:Integer},
        grid::AbstractRange; ep::Float64=0.3)
    n = length(grid)
    acc = zeros(n, n)
    nsamp = size(spectra, 1)
    ep2 = ep * ep

    for s in 1:nsamp
        nx = nxs[s]
        nw = div(nx, 3)
        nw == 0 && continue
        for i in 1:n, j in 1:n
            x_val, y_val = grid[i], grid[j]
            val = 0.0
            for k in 1:nw
                θ = spectra[s, 3k-2]
                c = spectra[s, 3k-1]
                f = spectra[s, 3k]
                dx = x_val - c * cos(θ)
                dy = y_val - c * sin(θ)
                val += f * exp(-ep2 * (dx^2 + dy^2))
            end
            acc[j, i] += val
        end
    end
    acc ./= nsamp
    return acc
end