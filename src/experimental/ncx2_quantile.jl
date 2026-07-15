# =============================================================================
# E5: scaled noncentral chi-square quantile  X = c * χ²(d, λ)
# =============================================================================
# Standalone experimental module. NOT included in the QuantileExpansions main
# module; it is loaded directly (include) by test/bench scripts.
#
# GOAL: test whether the single scaled noncentral chi-square (a CIR increment
# marginal) fits the repo's "regime seed + Householder(HH-4)" framework, whose
# core trick is that BS/IG/gamma/beta have a RATIONAL log-density derivative,
# making f''/f', f'''/f' near-free (pure arithmetic, no special functions).
#
# THE §8 TRAP.  The standard NoncentralChisq(d,λ) density is
#     f(y) ∝ e^{-(y+λ)/2} y^{d/4-1/2} I_{d/2-1}(√(λ y))
# so the log-density derivative M'(y) = d/dy log f is
#     M'(y) = -1/2 + (d/2 - 1)/y + (√λ/(2√y)) * R(z),   z = √(λ y),
#     R(z)  = I_{d/2}(z) / I_{d/2-1}(z)   (a MODIFIED BESSEL RATIO, not rational).
# (Derivation note: the density's y^{d/4-1/2} prefactor contributes (d/4-1/2)/y
#  and d/dy log I_{ν}(√(λy)), ν=d/2-1, contributes ANOTHER (d/4-1/2)/y plus the
#  Bessel-ratio term, so the 1/y coefficient is (d/2-1), i.e. 2*(d/4-1/2). This
#  matches the Poisson-mixture ground truth to ~1e-14; see the noncentral
#  chi-square marginal section of RESULTS.md.)
#
# Because R(z) is transcendental, the "rational log-derivative => near-free
# higher derivatives" property BREAKS: every Householder step needs a Bessel
# ratio. Higher derivatives of R are, however, free once R is known, via the
# Riccati equation R'(z) = 1 - R² - ((2ν+1)/z) R (derived from the modified
# Bessel ODE + recurrences), so the cost is exactly ONE Bessel-ratio eval per
# iterate, not one per derivative order.
# =============================================================================
module NCX2Quantile

using Distributions, SpecialFunctions

export ncx2_quantile, patnaik_seed, sankaran_seed, bessel_logderivs,
       quantile_newton, quantile_hh4

# Normal quantile for the seeds. Distributions' is allocation-free and ~6 ns;
# the task allows it for the seed. This is the ONLY special-function eval the
# seeds need (both seeds are otherwise closed-form rational/pow expressions).
@inline _norminv(u::Float64) = quantile(Normal(), u)

# -----------------------------------------------------------------------------
# SEED 1: Patnaik moment-matched central chi-square.
#   χ²(d,λ) ≈ a·χ²(f),  f = (d+λ)²/(d+2λ),  a = (d+2λ)/(d+λ),
#   central χ²(f) quantile via Wilson-Hilferty:  f*(1 - 2/(9f) + z*√(2/(9f)))³.
# When the WH cube base goes non-positive (deep lower tail / small f), fall back
# to the central-χ² power-law tail  Q ≈ (u·2^{f/2}·Γ(f/2+1))^{2/f}  so the seed
# stays finite and positive (mirrors src/dists/gamma.jl).
# -----------------------------------------------------------------------------
@inline function patnaik_seed(d::Float64, λ::Float64, u::Float64)
    f = (d + λ)^2 / (d + 2λ)
    a = (d + 2λ) / (d + λ)
    z = _norminv(u)
    t = 1.0 - 2.0 / (9f) + z * sqrt(2.0 / (9f))
    if t > 0.0
        return a * f * t * t * t
    else
        # power-law lower tail of the central χ²(f)
        lq = (log(u) + (f / 2) * log(2.0) + loggamma(f / 2 + 1.0)) * (2.0 / f)
        return a * exp(lq)
    end
end

# -----------------------------------------------------------------------------
# SEED 2: Sankaran cube-root-normal.
#   h = 1 - (2/3)(d+λ)(d+3λ)/(d+2λ)²,   p = (d+2λ)/(d+λ)²,
#   x = (d+λ) * ( 1 + h(h-1)p - h(h-1)(2-h)(1-3h)p²/2
#                  + z h √(2p) (1 + (h-1)(1-3h)p/2) )^(1/h),  z = Φ⁻¹(u).
# Guard the power base against going non-positive in the deep lower tail.
# -----------------------------------------------------------------------------
@inline function sankaran_seed(d::Float64, λ::Float64, u::Float64)
    dl = d + λ
    h  = 1.0 - (2.0 / 3.0) * dl * (d + 3λ) / (d + 2λ)^2
    p  = (d + 2λ) / (dl * dl)
    z  = _norminv(u)
    hm = h * (h - 1.0)
    base = 1.0 + hm * p - hm * (2.0 - h) * (1.0 - 3h) * p * p / 2.0 +
           z * h * sqrt(2p) * (1.0 + (h - 1.0) * (1.0 - 3h) * p / 2.0)
    # Deep lower tail / small-h: the cube-root base goes non-positive and the
    # power blows up or underflows. Fall back to the Patnaik power-law tail,
    # which is well-behaved and O(1)-accurate there.
    base <= 0.0 && return patnaik_seed(d, λ, u)
    return dl * base^(1.0 / h)
end

# -----------------------------------------------------------------------------
# ANALYTIC log-density derivatives via the Bessel ratio (THE special branch).
# Returns (Mp, X2) where
#   Mp = ρ'/ρ  = M'(y)              (φ2 in the HH-4 interface)
#   X2 = ρ''/ρ = M''(y) + M'(y)²    (ξ  in the HH-4 interface)
# using besselix (exponentially scaled Iν) so the ratio is overflow-safe:
#   R = I_{ν+1}(z)/I_ν(z) = besselix(ν+1,z)/besselix(ν,z).
# Cost: ONE Bessel ratio (two besselix calls); R', needed for M'', is free from
# the Riccati R'(z) = 1 - R² - ((2ν+1)/z) R.
# -----------------------------------------------------------------------------
# R(z) = I_{ν+1}(z)/I_ν(z). besselix is exact and overflow-safe up to z~1e9;
# beyond that AMOS loses accuracy, so use the large-z asymptotic ratio (error
# < 1e-11 for z > 1e4, all ν used here). Legitimate solution iterates keep
# z <~ 1e3; this branch only shields runaway safeguard steps.
@inline function _bessel_ratio(ν::Float64, z::Float64)
    if z < 1.0e5
        return besselix(ν + 1.0, z) / besselix(ν, z)
    end
    μ = 4.0 * ν * ν; μ1 = 4.0 * (ν + 1.0)^2
    a1 = (μ - 1.0) / 8.0;  a2 = (μ - 1.0) * (μ - 9.0) / 128.0
    a1p = (μ1 - 1.0) / 8.0; a2p = (μ1 - 1.0) * (μ1 - 9.0) / 128.0
    return 1.0 + (a1 - a1p) / z + (a2p - a1p * a1 + a1 * a1 - a2) / (z * z)
end

@inline function bessel_logderivs(d::Float64, λ::Float64, y::Float64)
    ν = d / 2 - 1.0
    if λ == 0.0
        Mp = -0.5 + ν / y
        return Mp, (-ν / (y * y) + Mp * Mp)   # rational central-χ² limit
    end
    z  = sqrt(λ * y)
    R  = _bessel_ratio(ν, z)
    b  = sqrt(λ) / (2.0 * sqrt(y))            # = √λ/(2√y) = dz/dy
    Rp = 1.0 - R * R - ((2ν + 1.0) / z) * R   # Riccati (free)
    Mp  = -0.5 + ν / y + b * R
    Mpp = -ν / (y * y) - b * R / (2y) + b * b * Rp
    return Mp, Mpp + Mp * Mp
end

# -----------------------------------------------------------------------------
# Convergence.  We drive relative-x convergence using the Newton step |r/fp| as
# an estimate of the remaining x error: stop when |r/fp| <= xtol*|x|. An absolute
# cdf residual |r| < tol is too loose in high-density regions (a 1e-13 residual
# there still leaves ~1e-7 relative x error), so it is NOT used as the stop; it
# is kept only as a machine-noise floor break. Both polishers are
# bracket-safeguarded: a step leaving the maintained (lo,hi) bracket is replaced
# by bisection (or geometric growth while hi = Inf), so they converge even from
# the asymptotic seeds' deep-tail failures.
# -----------------------------------------------------------------------------

# (a) NEWTON on the Distributions residual. Counts iterations (= cdf+pdf evals).
function quantile_newton(D::NoncentralChisq, u::Float64, x0::Float64;
                         xtol::Float64 = 1e-13, floor_r::Float64 = 1e-300, maxit::Int = 100)
    x = x0 > 0.0 ? x0 : 1e-300
    lo = 0.0; hi = Inf
    iters = 0
    @inbounds for it in 1:maxit
        iters = it
        r = cdf(D, x) - u
        r > 0.0 ? (hi = x) : (lo = x)
        abs(r) < floor_r && break
        (lo > 0.0 && hi < Inf && (hi - lo) <= xtol * hi) && break   # bracket pinned
        fp = pdf(D, x)
        step = fp > 0.0 ? r / fp : (hi == Inf ? -x : x - 0.5 * (lo + hi))
        abs(step) <= xtol * max(x, 1e-300) && break        # relative-x converged
        xn = x - step
        if !(isfinite(xn) && xn > lo && xn < hi)
            xn = hi == Inf ? 2.0 * x : 0.5 * (lo + hi)
        end
        x = xn
    end
    return x, iters
end

# (b) One-step-per-iterate HOUSEHOLDER (order 3, the repo's HH-4) using the
# ANALYTIC Bessel-ratio derivatives for φ2, ξ. Residual/density from
# Distributions (Marcum-Q cdf, pdf). Counts iterations; each iterate additionally
# pays one Bessel ratio (bessel_logderivs) beyond Newton's cdf+pdf.
function quantile_hh4(D::NoncentralChisq, d::Float64, λ::Float64, u::Float64, x0::Float64;
                      xtol::Float64 = 1e-13, floor_r::Float64 = 1e-300, maxit::Int = 100)
    x = x0 > 0.0 ? x0 : 1e-300
    lo = 0.0; hi = Inf
    iters = 0
    @inbounds for it in 1:maxit
        iters = it
        r = cdf(D, x) - u
        r > 0.0 ? (hi = x) : (lo = x)
        abs(r) < floor_r && break
        (lo > 0.0 && hi < Inf && (hi - lo) <= xtol * hi) && break   # bracket pinned
        fp = pdf(D, x)
        if fp > 0.0
            ρ = r / fp
            abs(ρ) <= xtol * max(x, 1e-300) && break       # relative-x converged
            φ2, ξ = bessel_logderivs(d, λ, x)              # THE Bessel branch
            denom = -6.0 + ρ * (6.0 * φ2 - ρ * ξ)
            xn = abs(denom) < 1e-20 ? x - ρ : x + 3.0 * ρ * (2.0 - ρ * φ2) / denom
            if !(isfinite(xn) && xn > lo && xn < hi)
                xn = x - ρ                                 # Newton safeguard
            end
        else
            xn = NaN                                       # no derivative info; use bracket
        end
        if !(isfinite(xn) && xn > lo && xn < hi)
            xn = hi == Inf ? 2.0 * x : 0.5 * (lo + hi)
        end
        x = xn
    end
    return x, iters
end

# -----------------------------------------------------------------------------
# Public entry: quantile of X = c·χ²(d,λ).  Solve standard NoncentralChisq(d,λ)
# then multiply by the scale c (X is just a scale of the standard variable).
# -----------------------------------------------------------------------------
function ncx2_quantile(d::Float64, λ::Float64, u::Float64; c::Float64 = 1.0,
                       seed::Symbol = :sankaran, method::Symbol = :newton,
                       xtol::Float64 = 1e-13)
    if seed === :patnaik
        x0 = patnaik_seed(d, λ, u)
    elseif seed === :sankaran
        x0 = sankaran_seed(d, λ, u)
    else
        throw(ArgumentError("seed must be :sankaran or :patnaik"))
    end
    D  = NoncentralChisq(d, λ)
    if method === :hh4
        x, _ = quantile_hh4(D, d, λ, u, x0; xtol = xtol)
    elseif method === :newton
        x, _ = quantile_newton(D, u, x0; xtol = xtol)
    else
        throw(ArgumentError("method must be :newton or :hh4"))
    end
    return c * x
end

end # module NCX2Quantile
