# Black–Scholes implied total-volatility inversion (Hekimoglu & Gökgöz template).
# Given log-moneyness k = log(K/F) and normalized undiscounted call price c,
# solve for v = σ√T such that  c = Φ(d1) - e^k Φ(d2),  d1 = -k/v + v/2, d2 = d1 - v.
#
# Strategy: reduce to an OTM-equivalent (κ = |k| ≥ 0, cstar), pick a closed-form
# seed by regime, then polish with one universal HH-4 step (quartic convergence).

# Regime boundaries (Definition 2.9)
const Κ1 = 0.001
const Κ2 = 0.81
const Κ3 = 1.155
const Κ4 = 1.347
const CSTAR_TAIL = 0.02128
const KSTAR_TAIL = 0.5
const A1 = INV_SQRT2PI   # 1/√(2π)

# --- forward price (OTM call form, κ ≥ 0) -----------------------------------
@inline function bs_price(κ::Float64, v::Float64)
    invv = 1.0 / v
    d1 = -κ * invv + 0.5 * v
    d2 = d1 - v
    return normcdf(d1) - exp(κ) * normcdf(d2)
end

# --- ATM seed: invert c_tv = erf-series in v (4th order) ---------------------
@inline function seed_atm(cstar::Float64)
    s = SQRT2PI * cstar
    s2 = s * s
    return s * (1.0 + s2 * (1.0/24.0 + s2 * (7.0/1920.0 + s2 * (127.0/322560.0))))
end

# --- Mild-OTM seed: linear-CDF quadratic (P1) --------------------------------
@inline function seed_p1(κ::Float64, cstar::Float64)
    φ = κ * (1.0 + κ * (0.5 + κ * (1.0/6.0 + κ * (1.0/24.0))))   # ≈ e^κ - 1
    twocφ = 2.0 * cstar + φ
    N = twocφ * twocφ - 8.0 * A1 * A1 * κ * φ * (2.0 + φ)
    sq = N > 0.0 ? sqrt(N) : 0.0
    return (twocφ + sq) / (2.0 * A1 * (2.0 + φ))
end

# Odd-polynomial CDF surrogate and its v-derivative, evaluated at v.
# P_m(x) = ½ + a1·S(x);  S(x) = x - x³/6 + x⁵/40 - x⁷/336  (truncate at order m)
# S'(x)   = 1 - x²/2 + x⁴/8 - x⁶/48                          (partial e^{-x²/2})
@inline function _poly_S(x::Float64, ::Val{M}) where {M}
    x2 = x * x
    if M == 1
        return x
    elseif M == 3
        return x * (1.0 - x2 * (1.0/6.0))
    else # M == 7
        return x * (1.0 - x2 * (1.0/6.0 - x2 * (1.0/40.0 - x2 * (1.0/336.0))))
    end
end
@inline function _poly_dS(x::Float64, ::Val{M}) where {M}
    x2 = x * x
    if M == 1
        return 1.0
    elseif M == 3
        return 1.0 - x2 * 0.5
    else # M == 7
        return 1.0 - x2 * (0.5 - x2 * (1.0/8.0 - x2 * (1.0/48.0)))
    end
end

# One Newton step on the polynomial-CDF price surrogate starting from v0.
@inline function seed_poly(κ::Float64, cstar::Float64, v0::Float64, E::Float64, m::Val{M}) where {M}
    invv = 1.0 / v0
    d1 = -κ * invv + 0.5 * v0
    d2 = d1 - v0
    cP = 0.5 * (1.0 - E) + A1 * (_poly_S(d1, m) - E * _poly_S(d2, m))
    # dcP/dv = a1[S'(d1) d1' - E S'(d2) d2'],  d1' = -d2/v, d2' = -d1/v
    dcP = A1 * (_poly_dS(d1, m) * (-d2 * invv) - E * _poly_dS(d2, m) * (-d1 * invv))
    return v0 - (cP - cstar) / dcP
end

# --- Deep-OTM seed: quadratic + Mills correction -----------------------------
@inline function seed_deep(κ::Float64, cstar::Float64)
    z = norminv(cstar)
    vq = z + sqrt(z * z + 2.0 * κ)
    ρ = vq * vq / (κ + 0.5 * vq * vq)
    arg = cstar / ρ
    arg = arg < 0.999999 ? arg : 0.999999
    zq = norminv(arg)
    return zq + sqrt(zq * zq + 2.0 * κ)
end

# --- regime dispatch: returns the seed v0 ------------------------------------
@inline function bs_seed(κ::Float64, cstar::Float64, E::Float64)
    vatm = seed_atm(cstar)
    # tail-filter override
    if cstar < CSTAR_TAIL && κ > KSTAR_TAIL
        return seed_deep(κ, cstar)
    end
    if κ < Κ1
        return vatm
    elseif κ <= Κ2
        v1 = seed_p1(κ, cstar)
        v7 = seed_poly(κ, cstar, v1, E, Val(7))
        return max(v7, vatm)
    elseif κ <= Κ3
        v1 = seed_p1(κ, cstar)
        v3 = seed_poly(κ, cstar, v1, E, Val(3))
        v7 = seed_poly(κ, cstar, v1, E, Val(7))
        return max(0.5 * (v3 + v7), vatm)
    elseif κ <= Κ4
        v1 = seed_p1(κ, cstar)
        v3 = seed_poly(κ, cstar, v1, E, Val(3))
        return max(v3, vatm)
    else
        return max(seed_deep(κ, cstar), vatm)
    end
end

# --- HH-4 polish step (Householder order 3 / quartic) ------------------------
# f = c_BS - cstar,  f' = φ(d1),  φ2 = f''/f' = d1 d2 / v,
# ξ = f'''/f' = ((d1 d2)² - (d1²+d2²) - d1 d2)/v².
#
# Exact identity, exploited below to save one `exp` per iteration:
#   d2 = d1 - v  and  d1·v - v²/2 = -κ   ⇒   exp(-d2²/2) = exp(-d1²/2)·e^{-κ}
# so the gaussian factor for d2 is free once we have the one for d1.
@inline function hh4_step(κ::Float64, cstar::Float64, v::Float64, E::Float64, invE::Float64)
    invv = 1.0 / v
    d1 = -κ * invv + 0.5 * v
    d2 = d1 - v
    Φ1, fp = normcdf_pdf(d1)        # Φ(d1) and φ(d1) share one exp
    g2 = fp * SQRT2PI * invE        # = exp(-d2²/2), no second exp
    f = Φ1 - E * normcdf_withg(d2, g2) - cstar
    r = f / fp
    d1d2 = d1 * d2
    φ2 = d1d2 * invv
    ξ = (d1d2 * d1d2 - (d1 * d1 + d2 * d2) - d1d2) * invv * invv
    denom = -6.0 + r * (6.0 * φ2 - r * ξ)
    if abs(denom) < 1e-20
        return v - r, abs(f)
    end
    return v + 3.0 * r * (2.0 - r * φ2) / denom, abs(f)
end

# --- public scalar solver ----------------------------------------------------
# --- generic-interface adapter (monomorphized by the shared solver) ----------
struct BSCall <: QuantileProblem
    κ::Float64
    E::Float64       # e^{|k|}
    invE::Float64    # e^{-|k|}
end
@inline xlo(::BSCall) = 1e-10
@inline xhi(::BSCall) = 5.0
@inline seed(D::BSCall, cstar::Float64) = bs_seed(D.κ, cstar, D.E)
@inline function hh_terms(D::BSCall, v::Float64, cstar::Float64)
    κ = D.κ; E = D.E
    invv = 1.0 / v
    d1 = -κ * invv + 0.5 * v
    d2 = d1 - v
    Φ1, fp = normcdf_pdf(d1)
    g2 = fp * SQRT2PI * D.invE      # = exp(-d2²/2), no second exp
    f = Φ1 - E * normcdf_withg(d2, g2) - cstar
    d1d2 = d1 * d2
    φ2 = d1d2 * invv
    ξ = (d1d2 * d1d2 - (d1 * d1 + d2 * d2) - d1d2) * invv * invv
    return f, fp, φ2, ξ
end

# Build the problem and call the generic solver.
@inline function bs_implied_vol_generic(k::Float64, c::Float64; tol::Float64 = 1e-14)
    κ = abs(k); ek = exp(k)
    if k >= 0.0
        cstar = c; E = ek; invE = 1.0 / ek
    else
        invek = 1.0 / ek
        cstar = invek * (c - 1.0 + ek); E = invek; invE = ek
    end
    return solve(BSCall(κ, E, invE), cstar; tol = tol)
end

@inline function bs_implied_vol(k::Float64, c::Float64; tol::Float64 = 1e-14, maxiter::Int = 8)
    κ = abs(k)
    ek = exp(k)
    # OTM-equivalent price and prefactor E = exp(κ) = e^{|k|}
    if k >= 0.0
        cstar = c
        E = ek
        invE = 1.0 / ek
    else
        invek = 1.0 / ek
        cstar = invek * (c - 1.0 + ek)
        E = invek
        invE = ek
    end
    v = bs_seed(κ, cstar, E)
    @inbounds for _ in 1:maxiter
        v, af = hh4_step(κ, cstar, v, E, invE)
        af < tol && break
    end
    return v
end

# --- branch-free fixed-step kernel -------------------------------------------
# Runs exactly N HH-4 updates with no residual test, so every input follows the
# same instruction path: no iteration-count divergence. That uniformity is the
# property SIMD needs (the adaptive solver cannot vectorize across lanes that
# converge at different steps). It is also simply faster in the common case,
# since the adaptive loop spends ~0.8 of its ~2.8 residual evaluations merely
# *proving* convergence.
#
# Accuracy follows from quartic convergence alone: a seed with relative error δ
# lands at δ^(4^N). The worst seed on the reference grid is δ ≈ 0.211, so
#     N = 2  ->  δ^16 ≈ 1.5e-11   (measured 2.9e-11)
#     N = 3  ->  δ^64 ≈ 0         (measured 1.3e-15, the machine floor)
# Use Val(2) as a fast mode (~1e-11) and Val(3) for full precision (~1e-15).
#
# NOTE: the *iteration* is branch-free, but `bs_seed` still branches on regime
# and the Cody `erfc` still branches on |d| range. Both must be bucketed or
# blended before this can actually be vectorized.
@inline function bs_implied_vol_fixed(k::Float64, c::Float64, ::Val{N} = Val(3)) where {N}
    κ = abs(k)
    ek = exp(k)
    if k >= 0.0
        cstar = c; E = ek; invE = 1.0 / ek
    else
        invek = 1.0 / ek
        cstar = invek * (c - 1.0 + ek); E = invek; invE = ek
    end
    v = bs_seed(κ, cstar, E)
    @inbounds for _ in 1:N
        invv = 1.0 / v
        d1 = -κ * invv + 0.5 * v
        d2 = d1 - v
        Φ1, fp = normcdf_pdf(d1)
        g2 = fp * SQRT2PI * invE          # exp(-d2²/2), free via the identity
        f = Φ1 - E * normcdf_withg(d2, g2) - cstar
        r = f / fp
        d1d2 = d1 * d2
        φ2 = d1d2 * invv
        ξ = (d1d2 * d1d2 - (d1 * d1 + d2 * d2) - d1d2) * invv * invv
        denom = -6.0 + r * (6.0 * φ2 - r * ξ)
        vn = v + 3.0 * r * (2.0 - r * φ2) / denom
        vn = ifelse(abs(denom) < 1e-20, v - r, vn)   # Newton fallback (branchless)
        vn = ifelse(isfinite(vn), vn, v)             # NaN/Inf guard (branchless)
        v = clamp(vn, 1e-10, 5.0)                    # bounds (min/max, branchless)
    end
    return v
end
