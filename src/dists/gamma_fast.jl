# Fast semi-analytic gamma quantile: port of the "semianalytic" mode of
# A. Hekimoglu's combined engine (gamma_quant_final_combined_engine.c,
# gamma_quant_semianalytic_one). Trades the full log-HH4 solver's machine
# precision for speed by returning a regime seed directly (ZERO incomplete-gamma
# evaluations) in certified seed-only regions, one cheap update elsewhere, and
# falling through to the exact GammaLogQ path in the deep tails.
#
# This is NOT a machine-precision replacement for gamma_quantile_log; it is the
# reference engine's fast mode for pricing/QMC tolerances. Measured accuracy per
# region is in RESULTS.md ("Gamma large-a ODE5 seed ...").
#
# Region map (a = shape, z = Phi^-1(u), q = 1 - u):
#   seed-only   a >= 10 central (|z| <= 2.5): return the ODE5 (CF5) seed x0
#               a < 2 lower tail: return the order-3 series seed x0
#   one update  2 <= a < 10 central/upper, a >= 10 moderate tail: x0 + 1 log-HH4
#               2 <= a < 10 lower (u < 0.08): series seed + 1 log-HH4
#   two update  a < 2 mid-u transition: series seed + 2 log-Newton steps
#   fallback    deep upper (q <= 1e-5): exact solve(GammaLogQ)

const _GF_SEED   = 0   # seed-only, zero incomplete-gamma evaluations
const _GF_ONE    = 1   # one log-HH4 update, one incomplete-gamma evaluation
const _GF_TWO    = 2   # two log-Newton updates, two incomplete-gamma evaluations
const _GF_FALL   = 3   # falls through to the exact GammaLogQ solver

struct GammaFast
    D::GammaLogQ
end
GammaFast(a::Float64) = GammaFast(GammaLogQ(a))

# one log-HH4 (Householder-3) update in y = ln x, reusing the generic rational
# step so a fast one-update point is bit-identical to the exact solver's first
# iterate. dP/dy = x*pdf, F''/F' = a - x, F'''/F' = (a-x)^2 - x.
@inline function _gamma_fast_hh4(a::Float64, u::Float64, x::Float64, lga::Float64)
    (x > 0.0 && isfinite(x)) || return x
    y = log(x)
    P, _ = gamma_inc(a, x, 0)
    f = P - u
    fp = exp(a * y - x - lga)
    (fp > 0.0 && isfinite(fp)) || return x
    A = a - x
    ξ = A * A - x
    r = f / fp
    denom = -6.0 + r * (6.0 * A - r * ξ)
    dy = abs(denom) < 1e-20 ? -r : 3.0 * r * (2.0 - r * A) / denom
    isfinite(dy) || return x
    dy = clamp(dy, -1.0, 1.0)
    xn = x * exp(dy)
    return (xn > 0.0 && isfinite(xn)) ? xn : x
end

# one log-space Newton step, F'(y) = x*pdf (used in the a < 2 transition band)
@inline function _gamma_fast_newton(a::Float64, u::Float64, x::Float64, lga::Float64)
    (x > 0.0 && isfinite(x)) || return x
    P, _ = gamma_inc(a, x, 0)
    f = P - u
    scale = exp(a * log(x) - x - lga)
    (scale > 0.0 && isfinite(scale)) || return x
    dy = -f / scale
    isfinite(dy) || return x
    dy = clamp(dy, -1.0, 1.0)
    xn = x * exp(dy)
    return (xn > 0.0 && isfinite(xn)) ? xn : x
end

# region classifier, kept in lockstep with _gamma_fast_one (measurement/tests)
@inline function _gamma_fast_region(a::Float64, u::Float64)
    (u <= 0.0 || u >= 1.0 || a == 1.0 || a == 0.5) && return _GF_SEED
    q = 1.0 - u
    z = norminv(u)
    if a >= 10.0
        return (abs(z) <= 2.5 && q > 1e-5) ? _GF_SEED : (q <= 1e-5 ? _GF_FALL : _GF_ONE)
    end
    q <= 1e-5 && return _GF_FALL
    if a <= 0.35
        u < 0.25 && return _GF_SEED
        u < 0.75 && return _GF_TWO
    elseif a <= 0.50
        u < 0.25 && return _GF_SEED
        u < 0.55 && return _GF_TWO
    elseif a < 2.0
        u < 0.15 && return _GF_SEED
        u < 0.75 && return _GF_TWO
    else
        u < 0.08 && return _GF_ONE
    end
    return _GF_ONE
end

@inline function _gamma_fast_one(F::GammaFast, u::Float64)
    D = F.D
    a = D.a
    u <= 0.0 && return 0.0
    u >= 1.0 && return Inf
    a == 1.0 && return -log1p(-u)
    if a == 0.5
        z = norminv(0.5 * (1.0 + u))
        return 0.5 * z * z
    end
    q = 1.0 - u
    z = norminv(u)

    # Large shape: the CF5/ODE5 Gaussian-quantile deformation is accurate.
    if a >= 10.0
        x = _gamma_seed_cf5(a, z)
        (abs(z) <= 2.5 && q > 1e-5) && return x
        q <= 1e-5 && return gamma_quantile_log(a, u)
        return _gamma_fast_hh4(a, u, x, D.lga)
    end

    # Deep upper: hand back to the exact log-survival path.
    q <= 1e-5 && return gamma_quantile_log(a, u)

    # Certified lower seed-only and mid-u log-Newton transition regions.
    if a <= 0.35
        u < 0.25 && return _gamma_seed_lower(u, a, D.lga1)
        if u < 0.75
            x = _gamma_seed_lower(u, a, D.lga1)
            x = _gamma_fast_newton(a, u, x, D.lga)
            return _gamma_fast_newton(a, u, x, D.lga)
        end
    elseif a <= 0.50
        u < 0.25 && return _gamma_seed_lower(u, a, D.lga1)
        if u < 0.55
            x = _gamma_seed_lower(u, a, D.lga1)
            x = _gamma_fast_newton(a, u, x, D.lga)
            return _gamma_fast_newton(a, u, x, D.lga)
        end
    elseif a < 2.0
        u < 0.15 && return _gamma_seed_lower(u, a, D.lga1)
        if u < 0.75
            x = _gamma_seed_lower(u, a, D.lga1)
            x = _gamma_fast_newton(a, u, x, D.lga)
            return _gamma_fast_newton(a, u, x, D.lga)
        end
    else
        if u < 0.08
            x = _gamma_seed_lower(u, a, D.lga1)
            return _gamma_fast_hh4(a, u, x, D.lga)
        end
    end

    # Moderate-shape central / ordinary upper: one ODE5/WH log-HH4 update.
    x = (a >= 2.0 && abs(z) <= 2.5) ? _gamma_seed_cf5(a, z) : _gamma_seed_wh(a, z)
    return _gamma_fast_hh4(a, u, x, D.lga)
end

"""
    gamma_quantile_fast(a, u)

Fast semi-analytic Gamma(shape `a`, scale 1) quantile (port of the reference
engine's "semianalytic" mode). Returns a certified regime seed directly with
zero incomplete-gamma evaluations in the large-a central and lower-tail regions,
one cheap update in moderate regions, and the exact [`gamma_quantile_log`] result
in the deep upper tail. NOT machine precision: max `|Δ ln x|` ranges from ~1e-5
(a = 10) to ~1e-8 (a = 500) in the seed-only region; see RESULTS.md. Use
[`gamma_quantile_log`] when full accuracy is required.
"""
@inline gamma_quantile_fast(a::Float64, u::Float64) = _gamma_fast_one(GammaFast(a), u)

"""
    gamma_quantile_fast_batch!(out, a, us)

Amortized batch of [`gamma_quantile_fast`]: the `GammaFast(a)` per-shape setup
is done ONCE and reused across `us`. Bit-identical to the scalar function;
allocation-free given a preallocated `out`.
"""
function gamma_quantile_fast_batch!(out::Vector{Float64}, a::Float64, us::Vector{Float64})
    length(out) == length(us) || throw(DimensionMismatch("out and us must have equal length"))
    F = GammaFast(a)
    @inbounds for i in eachindex(us)
        out[i] = _gamma_fast_one(F, us[i])
    end
    return out
end
