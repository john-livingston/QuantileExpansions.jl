# Gamma quantile in log space — Julia port of A. Hekimoglu's regime-split solver
# (gamma_quant_full_c5_dynamic_boundary_widesafe_engine.c, "full" mode).
#
# The unknown is y = ln x, which keeps x > 0 with no clamping and collapses the
# HH-4 ratios to polynomials: with F(y) = P(a, e^y) - u,
#   F'   = exp(a·y - e^y - lnΓ(a))          (= x·pdf)
#   F''/F'  = a - x
#   F'''/F' = (a - x)² - x
# so his log-HH4 is exactly our generic solve() on a different QuantileProblem —
# the port is the seeds + regime map; the iteration is the shared one.
#
# Seeds (per regime, all closed-form):
#   exact       a = 1 (exponential), a = ½ (χ²₁ via the normal bridge)
#   lower tail  series reversion of P(a,x) near 0:  x = s(1 + c2 s + c3 s² + c4 s³),
#               s = (u Γ(a+1))^{1/a}, with an ANALYTIC validity boundary from the
#               first omitted coefficient c5 (certifies seed→1e-13 after polish)
#   upper tail  gamma-Mills survival series solved in log space
#   large a     5th-order Cornish–Fisher ("ODE5") in ε = 1/√a
#   fallback    Wilson–Hilferty cube
import SpecialFunctions: gamma_inc, loggamma

struct GammaLogQ <: QuantileProblem
    a::Float64
    lga::Float64       # lnΓ(a)
    lga1::Float64      # lnΓ(a+1)
    uL::Float64        # dynamic lower-tail boundary (c5 certificate)
end
function GammaLogQ(a::Float64)
    lga = loggamma(a)
    lga1 = lga + log(a)
    GammaLogQ(a, lga, lga1, _gamma_lower_uL_c5(a, lga1))
end

@inline xlo(::GammaLogQ) = -746.0     # bounds on y = ln x
@inline xhi(::GammaLogQ) = 710.0

# analytic c5 dynamic boundary: largest u for which the order-4 series seed,
# after the polish, is certified below τ = 1e-13
@inline function _gamma_lower_uL_c5(a::Float64, lga1::Float64)
    c5 = (((125.0*a + 1179.0)*a + 3971.0)*a*a + 5661.0*a + 2888.0) /
         (24.0 * (a+1.0)^4 * (a+2.0)^2 * (a+3.0) * (a+4.0))
    τ = 1e-13
    s = (64.0 * τ / (a^7 * c5^8))^(1.0/32.0)
    uL = exp(a * log(s) - lga1)
    return min(uL, 0.25)
end

# lower-tail series seed: x = s(1 + c2 s + c3 s² + c4 s³)
@inline function _gamma_seed_lower(u::Float64, a::Float64, lga1::Float64)
    s = exp((log(u) + lga1) / a)
    c2 = 1.0 / (a + 1.0)
    c3 = (3.0*a + 5.0) / (2.0 * (a+1.0)^2 * (a+2.0))
    c4 = ((8.0*a + 33.0)*a + 31.0) / (3.0 * (a+1.0)^3 * (a+2.0) * (a+3.0))
    return s * (1.0 + s * (c2 + s * (c3 + s * c4)))
end

# upper-tail Mills/survival seed: solve x - (a-1)·ln x - ln S(x) = y_t,
# y_t = -ln q - lnΓ(a), S = 1 + (a-1)/x + (a-1)(a-2)/x² (asymptotic survival)
@inline function _gamma_seed_upper(q::Float64, a::Float64, lga::Float64)
    A = a - 1.0
    yt = -log(q) - lga
    yt = max(yt, 1.5)
    x = yt + A * log(yt)
    x = max(x, 1e-3)
    # 2 Newton steps on x - A·ln x = yt
    for _ in 1:2
        g = x - A * log(x) - yt
        gp = 1.0 - A / x
        abs(gp) > 1e-12 && (x -= g / gp)
        x = max(x, 1e-3)
    end
    # 3 damped Newton steps including the survival series S
    b1 = A; b2 = A * (a - 2.0)
    for _ in 1:3
        invx = 1.0 / x
        S = 1.0 + invx * (b1 + b2 * invx)
        Sp = -invx * invx * (b1 + 2.0 * b2 * invx)
        g = x - A * log(x) - log(max(S, 1e-300)) - yt
        gp = 1.0 - A * invx - Sp / S
        dx = -g / max(abs(gp), 1e-12) * sign(gp)
        dx = clamp(dx, -0.5 * x, 0.5 * x)
        x += dx
        x = max(x, 1e-3)
    end
    return x
end

# large-a 5th-order Cornish–Fisher ("ODE5") seed
@inline function _gamma_seed_cf5(a::Float64, z::Float64)
    ε = 1.0 / sqrt(a)
    z2 = z * z
    p1 = (z2 - 1.0) / 3.0
    p2 = z * (z2 - 7.0) / 36.0
    p3 = -(3.0*z2*z2 + 7.0*z2 - 16.0) / 810.0
    p4 = z * (9.0*z2*z2 + 256.0*z2 - 433.0) / 38880.0
    p5 = (12.0*z2*z2*z2 - 243.0*z2*z2 - 923.0*z2 + 1472.0) / 204120.0
    y = z + ε * (p1 + ε * (p2 + ε * (p3 + ε * (p4 + ε * p5))))
    x = a + sqrt(a) * y
    return (isfinite(x) && x > 0.0) ? x : _gamma_seed_wh(a, z)
end

@inline function _gamma_seed_wh(a::Float64, z::Float64)
    t = 1.0 - 1.0/(9.0*a) + z/(3.0*sqrt(a))
    x = a * t * t * t
    return x > 0.0 ? x : 1e-8          # cube went negative: tiny positive start
end

# regime-mapped seed, returned in y = ln x
@inline function seed(D::GammaLogQ, u::Float64)
    a = D.a
    q = 1.0 - u
    x = 0.0
    if u < D.uL
        x = _gamma_seed_lower(u, a, D.lga1)
    else
        # Mills survival seed only where its asymptotics hold (x ≫ a - 1, i.e.
        # yt dominates); for large a the upper quantile is a + O(√a) and the
        # Cornish–Fisher / WH seeds are the right ones there.
        yt = -log(max(q, 1e-300)) - D.lga
        if ((a < 2.0 && q <= 5e-4) || (a >= 2.0 && q <= 1e-4)) &&
           yt > 2.0 * max(a - 1.0, 1.0)
            x = _gamma_seed_upper(q, a, D.lga)
        else
            z = norminv(u)
            x = (a >= 2.0 && abs(z) <= 2.5) ? _gamma_seed_cf5(a, z) : _gamma_seed_wh(a, z)
        end
    end
    return log(max(x, 1e-300))
end

@inline function hh_terms(D::GammaLogQ, y::Float64, u::Float64)
    a = D.a
    x = exp(y)
    P, _ = gamma_inc(a, x, 0)
    f = P - u
    fp = exp(a * y - x - D.lga)        # F' in log space = x·pdf
    A = a - x
    return f, fp, A, A * A - x         # φ2 = F''/F', ξ = F'''/F'
end

"""
    gamma_quantile_log(a, u; tol=1e-14)

Gamma(shape `a`, scale 1) quantile via the log-space regime-split solver
(port of A. Hekimoglu's C engine). Exact closed forms for `a = 1` and `a = ½`.
"""
@inline function gamma_quantile_log(a::Float64, u::Float64; tol::Float64 = 1e-14)
    u <= 0.0 && return 0.0
    u >= 1.0 && return Inf
    a == 1.0 && return -log1p(-u)
    if a == 0.5
        z = norminv(0.5 * (1.0 + u))
        return 0.5 * z * z
    end
    y = solve(GammaLogQ(a), u; tol = tol)
    return exp(y)
end

# density-scaled convergence (Hekimoglu's criterion): residual small AND the
# log-Newton step |f/fp| = |Δ ln x| negligible — required for tail accuracy.
@inline converged(::GammaLogQ, f, fp, tol) = abs(f) < tol && abs(f) <= 2e-14 * fp
