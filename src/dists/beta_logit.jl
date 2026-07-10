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
    κ1::Float64        # mean (CF5 seed, precomputed per (a,b))
    sκ2::Float64       # √variance
    g1::Float64        # skewness κ3/κ2^{3/2}
    g2::Float64        # excess kurtosis κ4/κ2²
    g5::Float64        # κ5/κ2^{5/2}
end
function BetaLogitQ(a::Float64, b::Float64, lb::Float64)
    n = a + b
    # raw moments m_k = Π_{j<k} (a+j)/(n+j), cumulants, standardized ratios —
    # once per (a,b); the per-quantile CF5 seed is then pure Horner in z
    m1 = a/n
    m2 = m1*(a+1)/(n+1)
    m3 = m2*(a+2)/(n+2)
    m4 = m3*(a+3)/(n+3)
    m5 = m4*(a+4)/(n+4)
    κ1 = m1
    κ2 = m2 - m1^2
    κ3 = m3 - 3m1*m2 + 2m1^3
    κ4 = m4 - 4m1*m3 - 3m2^2 + 12m1^2*m2 - 6m1^4
    κ5 = m5 - 5m1*m4 - 10m2*m3 + 20m1^2*m3 + 30m1*m2^2 - 60m1^3*m2 + 24m1^5
    sκ2 = sqrt(max(κ2, 0.0))
    g1 = κ3 / max(κ2*sκ2, 1e-300)
    g2 = κ4 / max(κ2*κ2, 1e-300)
    g5 = κ5 / max(κ2*κ2*sκ2, 1e-300)
    BetaLogitQ(a, b, lb, κ1, sκ2, g1, g2, g5)
end
BetaLogitQ(a::Float64, b::Float64) = BetaLogitQ(a, b, logbeta(a, b))

@inline xlo(::BetaLogitQ) = -700.0     # bounds on y = logit(x)
@inline xhi(::BetaLogitQ) = 700.0

@inline function seed(D::BetaLogitQ, p::Float64)
    a = D.a; b = D.b
    x0 = -1.0
    if a >= 1.0 && b >= 1.0
        # CF5 central seed (Hekimoglu): 5th-cumulant Cornish–Fisher, precomputed
        # per (a,b) — accurate enough for the K4 certificate to fire centrally
        z = norminv(p)
        if abs(z) <= 2.5
            z2 = z*z
            g1 = D.g1; g2 = D.g2; g5 = D.g5
            w = z + (g1/6)*(z2-1) + (g2/24)*z*(z2-3) - (g1*g1/36)*z*(2z2-5) +
                (g5/120)*(z2*z2-6z2+3) - (g1*g2/24)*(z2*z2-5z2+2) +
                (g1^3/324)*(12z2*z2-53z2+17)
            x0 = D.κ1 + D.sκ2 * w
        end
    end
    if !(0.0 < x0 < 1.0)
        x0 = seed(BetaQ(a, b, D.lb), p)          # regime seeds (tails, sub-1 shapes)
    end
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

# f''''/f' in logit space: with n = a+b, A = a - nx, D = x(1-x),
# F⁗/F' = A³ - 3AnD - nD(1-2x)   (matches Hekimoglu's K4 = |5c2³-5c2c3+c4|)
@inline has_c4(::BetaLogitQ) = true
@inline function hh4_c4(D::BetaLogitQ, y::Float64)
    a = D.a; b = D.b; n = a + b
    x = 1.0 / (1.0 + exp(-y))
    omx = 1.0 / (1.0 + exp(y))
    A = a - n * x
    nD = n * x * omx
    return A * (A * A - 3.0 * nD) - nD * (1.0 - 2.0 * x)
end

"Certified variant of [`beta_quantile_logit`]: skips the confirmation CDF
evaluation when the K4 error model certifies the first HH-4 update below τ."
@inline function beta_quantile_logit_cert(a::Float64, b::Float64, p::Float64; tol::Float64 = 1e-14)
    p <= 0.0 && return 0.0
    p >= 1.0 && return 1.0
    if b == 1.0
        return p^(1.0 / a)
    elseif a == 1.0
        return -expm1(log1p(-p) / b)
    elseif a == 0.5 && b == 0.5
        s = sinpi(0.5 * p)
        return s * s
    end
    if p <= 0.5
        y = solve_certified(BetaLogitQ(a, b), p; tol = tol)
        return 1.0 / (1.0 + exp(-y))
    else
        y = solve_certified(BetaLogitQ(b, a), 1.0 - p; tol = tol)
        ey = exp(y)
        return 1.0 - ey / (1.0 + ey)
    end
end

"""
    beta_quantile_batch!(out, a, b, ps; tol=1e-14, certified=true)

Amortized batch of [`beta_quantile_logit`] / [`beta_quantile_logit_cert`]:
`BetaLogitQ(a,b)` AND its mirror `BetaLogitQ(b,a)` (needed for p > ½) are
constructed ONCE per shape pair — logbeta + the CF5 cumulants — and reused
across `ps`, like the reference C engines amortize per-(a,b) setup.
Bit-identical to the per-call scalar functions; allocation-free given a
preallocated `out`.
"""
function beta_quantile_batch!(out::Vector{Float64}, a::Float64, b::Float64, ps::Vector{Float64};
                              tol::Float64 = 1e-14, certified::Bool = true)
    length(out) == length(ps) || throw(DimensionMismatch("out and ps must have equal length"))
    # exact closed-form shapes: (a,b)-checks hoisted out of the loop
    if b == 1.0
        @inbounds for i in eachindex(ps)
            p = ps[i]
            out[i] = p <= 0.0 ? 0.0 : (p >= 1.0 ? 1.0 : p^(1.0 / a))
        end
        return out
    elseif a == 1.0
        @inbounds for i in eachindex(ps)
            p = ps[i]
            out[i] = p <= 0.0 ? 0.0 : (p >= 1.0 ? 1.0 : -expm1(log1p(-p) / b))
        end
        return out
    elseif a == 0.5 && b == 0.5
        @inbounds for i in eachindex(ps)
            p = ps[i]
            if p <= 0.0
                out[i] = 0.0
            elseif p >= 1.0
                out[i] = 1.0
            else
                s = sinpi(0.5 * p)
                out[i] = s * s
            end
        end
        return out
    end
    Dab = BetaLogitQ(a, b)             # amortized: constructed once per (a,b)
    Dba = BetaLogitQ(b, a)             # mirrored problem for p > ½
    @inbounds for i in eachindex(ps)
        p = ps[i]
        if p <= 0.0
            out[i] = 0.0
        elseif p >= 1.0
            out[i] = 1.0
        elseif p <= 0.5
            y = certified ? solve_certified(Dab, p; tol = tol) : solve(Dab, p; tol = tol)
            out[i] = 1.0 / (1.0 + exp(-y))
        else
            y = certified ? solve_certified(Dba, 1.0 - p; tol = tol) : solve(Dba, 1.0 - p; tol = tol)
            # 1 - σ(y), computed as in beta_quantile_logit for last-ulp stability
            ey = exp(y)
            out[i] = 1.0 - ey / (1.0 + ey)
        end
    end
    return out
end
