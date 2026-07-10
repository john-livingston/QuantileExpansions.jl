# Beta quantile in logit space — the coordinate change from A. Hekimoglu's beta engine.
#
# Unknown y = logit(x); x = σ(y) stays in (0,1) intrinsically, and both x and
# 1-x are computed stably from y (no cancellation at either endpoint). The HH-4
# ratios collapse to polynomials in x: with F(y) = I_{σ(y)}(a,b) - u,
#   F'      = pdf(x)·x(1-x) = exp(a·ln x + b·ln(1-x) - ln B(a,b))
#   F''/F'  = a - (a+b)x
#   F'''/F' = (a - (a+b)x)² - (a+b)·x(1-x)
# Seeds are the same regime seeds as BetaQ (Cornish–Fisher bulk, power-law
# tails), mapped through logit. Density-scaled convergence for tail accuracy.
import SpecialFunctions: beta_inc, logbeta

struct BetaLogitQ <: QuantileProblem
    a::Float64
    b::Float64
    lb::Float64        # ln B(a,b)
end
BetaLogitQ(a::Float64, b::Float64) = BetaLogitQ(a, b, logbeta(a, b))

@inline xlo(::BetaLogitQ) = -700.0     # bounds on y = logit(x)
@inline xhi(::BetaLogitQ) = 700.0

@inline function seed(D::BetaLogitQ, p::Float64)
    x0 = seed(BetaQ(D.a, D.b, D.lb), p)          # reuse the x-space regime seeds
    x0 = clamp(x0, 1e-300, 1.0 - 1e-16)
    return log(x0) - log1p(-x0)                   # logit
end

@inline function hh_terms(D::BetaLogitQ, y::Float64, u::Float64)
    a = D.a; b = D.b
    # stable x, 1-x, ln x, ln(1-x) from y
    x   = 1.0 / (1.0 + exp(-y))
    omx = 1.0 / (1.0 + exp(y))
    lnx   = -log1p(exp(-y))
    lnomx = -log1p(exp(y))
    Ix, _ = beta_inc(a, b, x)
    f = Ix - u
    fp = exp(a * lnx + b * lnomx - D.lb)          # F' in logit space
    A = a - (a + b) * x
    return f, fp, A, A * A - (a + b) * x * omx
end

# density-scaled convergence (as for GammaLogQ): the logit-Newton step
# |f/F'| = |Δ logit x| must be negligible, not just the CDF residual.
@inline converged(::BetaLogitQ, f, fp, tol) = abs(f) < tol && abs(f) <= 2e-14 * fp

"""
    beta_quantile_logit(a, b, p; tol=1e-14)

Beta(a,b) quantile via the logit-space solver (Hekimoglu's coordinate change on our
regime seeds). Tail-symmetric: p > ½ solves the mirrored problem.
"""
@inline function beta_quantile_logit(a::Float64, b::Float64, p::Float64; tol::Float64 = 1e-14)
    p <= 0.0 && return 0.0
    p >= 1.0 && return 1.0
    # exact closed forms (from Hekimoglu's engine): I_x is elementary here, and the
    # generic seeds are arbitrarily bad in these extreme-skew corners
    if b == 1.0
        return p^(1.0 / a)                       # I_x = x^a
    elseif a == 1.0
        return -expm1(log1p(-p) / b)             # I_x = 1-(1-x)^b
    elseif a == 0.5 && b == 0.5
        s = sinpi(0.5 * p)                       # arcsine law
        return s * s
    end
    if p <= 0.5
        y = solve(BetaLogitQ(a, b), p; tol = tol)
        return 1.0 / (1.0 + exp(-y))
    else
        y = solve(BetaLogitQ(b, a), 1.0 - p; tol = tol)
        # 1 - σ(y): σ computed as e^y/(1+e^y) so the final subtraction keeps
        # the last ulp below 1 (1/(1+e^y) would round through 1+tiny -> 1.0)
        ey = exp(y)
        return 1.0 - ey / (1.0 + ey)
    end
end
