# Generic regime-split quantile solver.
#
# Every target is "solve F(x) = p for x".  Each distribution D implements a tiny
# interface; the compiler monomorphizes solve(D, p) into specialized, inlined,
# allocation-free code.  Interface:
#
#   seed(D, p)        -> Float64        closed-form regime seed (no iteration)
#   hh_terms(D, x, p) -> (f, fp, φ2, ξ) residual & HH-4 ratios at x
#       f   = F(x) - p          (residual; CDF/price space)
#       fp  = f' = ρ(x)         (density / vega)
#       φ2  = f''/f'  = L'(x)        where L = log ρ  (rational ⇒ cheap)
#       ξ   = f'''/f' = L''(x) + L'(x)²
#   xlo(D), xhi(D)    -> clamp bounds for the iterate (optional; defaults ±Inf)
#
# The HH-4 (Householder order 3, quartic) update and admissibility/stopping are
# written once here, generic over the interface.

abstract type QuantileProblem end

@inline xlo(::QuantileProblem) = 1e-300
@inline xhi(::QuantileProblem) = Inf

# δ_v* = tol^(1/16): relative seed error admissible for 2 net HH-4 steps.
@inline admissible_seed_error(tol) = tol^(1.0 / 16.0)

# Convergence test, overridable per distribution. Default: absolute CDF/price
# residual. Distributions solved deep in the tails (density fp → 0, so |f| < tol
# fires before x has converged) should additionally require the Newton step
# |f/fp| to be negligible — a density-scaled criterion.
@inline converged(::QuantileProblem, f, fp, tol) = abs(f) < tol

# Fourth derivative ratio f''''/f' = L''' + 3L'L'' + L'^3 (rational, like the
# others). Distributions that implement it enable the certified fast path.
function hh4_c4 end
@inline has_c4(::QuantileProblem) = false

# Certified solve (generalization of Hekimoglu's y6/K4 acceptance certificates):
# evaluate hh_terms ONCE at the seed, apply the HH-4 update, and exit without a
# confirmation evaluation when the classical Householder-3 error model
#     e_next ≈ K4 · e^4,   K4 = |5c2³ - 5c2c3 + c4|,
#     c2 = φ2/2, c3 = ξ/6, c4 = (f''''/f')/24,  e ≈ r = f/f'
# certifies the post-update error below τ (in x units — for log/logit problems
# that is relative-x / logit accuracy). Uncertified points fall through to the
# adaptive loop, so this is never *less* accurate than solve(); it only skips
# work where the skip is provably safe (up to the asymptotic error model, hence
# the safety factor).
@inline function solve_certified(D::QuantileProblem, p::Float64; tol::Float64 = 1e-14,
                                 τ::Float64 = 1e-14, safety::Float64 = 16.0, maxiter::Int = 8)
    x = seed(D, p)
    lo = xlo(D); hi = xhi(D)
    f, fp, φ2, ξ = hh_terms(D, x, p)
    converged(D, f, fp, tol) && return x
    r = f / fp
    denom = -6.0 + r * (6.0 * φ2 - r * ξ)
    xn = abs(denom) < 1e-20 ? x - r : x + 3.0 * r * (2.0 - r * φ2) / denom
    if isfinite(xn) && xn > lo && xn < hi
        if has_c4(D)
            c2 = 0.5 * φ2
            c3 = ξ / 6.0
            c4 = hh4_c4(D, x) / 24.0
            K4 = abs(5.0 * c2 * (c2 * c2 - c3) + c4)
            r2 = r * r
            safety * K4 * r2 * r2 <= τ && return xn        # certified: done
        end
        x = xn
    end
    @inbounds for _ in 1:maxiter
        f, fp, φ2, ξ = hh_terms(D, x, p)
        converged(D, f, fp, tol) && break
        r = f / fp
        denom = -6.0 + r * (6.0 * φ2 - r * ξ)
        xn = abs(denom) < 1e-20 ? x - r : x + 3.0 * r * (2.0 - r * φ2) / denom
        if !(isfinite(xn) && xn > lo && xn < hi)
            xn = x - r
            if !(isfinite(xn) && xn > lo && xn < hi)
                xn = f > 0.0 ? 0.5 * (lo + x) : (isfinite(hi) ? 0.5 * (x + hi) : 2.0 * x)
            end
        end
        x = xn
    end
    return x
end

@inline function solve(D::QuantileProblem, p::Float64; tol::Float64 = 1e-14, maxiter::Int = 8)
    x = seed(D, p)
    lo = xlo(D); hi = xhi(D)
    @inbounds for _ in 1:maxiter
        f, fp, φ2, ξ = hh_terms(D, x, p)
        converged(D, f, fp, tol) && break
        r = f / fp
        denom = -6.0 + r * (6.0 * φ2 - r * ξ)
        xn = abs(denom) < 1e-20 ? x - r : x + 3.0 * r * (2.0 - r * φ2) / denom
        if !(isfinite(xn) && xn > lo && xn < hi)
            # Safeguard 1: plain Newton (always steps toward the root for a
            # monotone CDF, fp > 0). Cheap; off the hot path for good seeds.
            xn = x - r
            if !(isfinite(xn) && xn > lo && xn < hi)
                # Safeguard 2: damped bracket step. f>0 ⇒ x too big ⇒ shrink
                # toward lo; f<0 ⇒ x too small ⇒ grow toward hi.
                xn = f > 0.0 ? 0.5 * (lo + x) : (isfinite(hi) ? 0.5 * (x + hi) : 2.0 * x)
            end
        end
        x = xn
    end
    return x
end
