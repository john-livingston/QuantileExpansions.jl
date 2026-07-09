# Beta quantile: invert the regularized incomplete beta I_x(a,b) = p.
#
# Residual f = I_x(a,b) - p,  f' = ρ(x) = x^{a-1}(1-x)^{b-1}/B(a,b).
# L = log ρ ⇒ rational log-derivatives:
#   φ2 = L'  = (a-1)/x - (b-1)/(1-x)
#   ξ  = L'' + L'^2,   L'' = -(a-1)/x^2 - (b-1)/(1-x)^2
import SpecialFunctions: beta_inc, logbeta

struct BetaQ <: QuantileProblem
    a::Float64
    b::Float64
    lb::Float64       # log B(a,b)
end
BetaQ(a::Float64, b::Float64) = BetaQ(a, b, logbeta(a, b))

@inline xlo(::BetaQ) = 1e-300
@inline xhi(::BetaQ) = 1.0

# --- seed: NR-style (Cornish–Fisher bulk + power-law tails) -------------------
@inline function seed(D::BetaQ, p::Float64)
    a = D.a; b = D.b
    if a >= 1.0 && b >= 1.0
        pp = p < 0.5 ? p : 1.0 - p
        t = sqrt(-2.0 * log(pp))
        xnum = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t
        x = p < 0.5 ? -xnum : xnum
        al = (x * x - 3.0) / 6.0
        h = 2.0 / (1.0 / (2.0 * a - 1.0) + 1.0 / (2.0 * b - 1.0))
        w = (x * sqrt(al + h) / h) -
            (1.0 / (2.0 * b - 1.0) - 1.0 / (2.0 * a - 1.0)) * (al + 5.0 / 6.0 - 2.0 / (3.0 * h))
        return a / (a + b * exp(2.0 * w))
    else
        lna = log(a / (a + b)); lnb = log(b / (a + b))
        t = exp(a * lna) / a; u = exp(b * lnb) / b
        w = t + u
        if p < t / w
            return exp(log(a * w * p) / a)
        else
            return 1.0 - exp(log(b * w * (1.0 - p)) / b)
        end
    end
end

@inline function hh_terms(D::BetaQ, x::Float64, p::Float64)
    a = D.a; b = D.b
    a1 = a - 1.0; b1 = b - 1.0
    omx = 1.0 - x
    Ix, _ = beta_inc(a, b, x)
    f = Ix - p
    fp = exp(a1 * log(x) + b1 * log(omx) - D.lb)     # ρ(x)
    Lp = a1 / x - b1 / omx                            # L'
    Lpp = -a1 / (x * x) - b1 / (omx * omx)            # L''
    φ2 = Lp
    ξ = Lpp + Lp * Lp
    return f, fp, φ2, ξ
end

# Reduce to the small tail via I_x(a,b) = 1 - I_{1-x}(b,a): for p > 0.5 solve the
# mirrored problem for y = 1-x (kept away from the x→1 precision wall).
@inline function beta_quantile(a::Float64, b::Float64, p::Float64; tol::Float64 = 1e-13)
    if p <= 0.5
        return solve(BetaQ(a, b), p; tol = tol)
    else
        return 1.0 - solve(BetaQ(b, a), 1.0 - p; tol = tol)
    end
end
