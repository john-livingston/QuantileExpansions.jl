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

@inline function solve(D::QuantileProblem, p::Float64; tol::Float64 = 1e-14, maxiter::Int = 8)
    x = seed(D, p)
    lo = xlo(D); hi = xhi(D)
    @inbounds for _ in 1:maxiter
        f, fp, φ2, ξ = hh_terms(D, x, p)
        abs(f) < tol && break
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
