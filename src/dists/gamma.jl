# Gamma quantile: invert the regularized lower incomplete gamma P(a,x) = p.
# Same architecture as BS-IV: regime-split closed-form seed + HH-4 polish.
#
# Residual f = P(a,x) - p,  f' = ρ(x) = x^{a-1} e^{-x}/Γ(a).
# L = log ρ = (a-1)log x - x - logΓ(a)  ⇒  rational log-derivatives:
#   φ2 = f''/f'  = L'  = (a-1)/x - 1
#   ξ  = f'''/f' = L'' + L'^2 = -(a-1)/x^2 + ((a-1)/x - 1)^2
import SpecialFunctions: gamma_inc, loggamma

struct GammaQ <: QuantileProblem
    a::Float64
    lga::Float64      # logΓ(a)
end
GammaQ(a::Float64) = GammaQ(a, loggamma(a))

@inline xlo(::GammaQ) = 1e-300
@inline xhi(::GammaQ) = Inf

# --- regime-split seed (Wilson–Hilferty + small-a tail; AS91/NR style) --------
@inline function seed(D::GammaQ, p::Float64)
    a = D.a
    if a > 1.0
        # Wilson–Hilferty normal seed
        zp = norminv(p)
        t = 1.0 - 1.0 / (9.0 * a) + zp / (3.0 * sqrt(a))
        x = a * t * t * t
        if x <= 0.0
            # WH cube went non-positive (moderate a, deep lower tail):
            # P ≈ x^a/Γ(a+1) ⇒ x ≈ (p Γ(a+1))^{1/a}
            x = exp((log(p) + D.lga + log(a)) / a)
        end
        return x
    else
        # small a: piecewise (lower power-law vs upper log)
        t = 1.0 - a * (0.253 + a * 0.12)
        if p < t
            return exp(log(p / t) / a)
        else
            return 1.0 - log(1.0 - (p - t) / (1.0 - t))
        end
    end
end

@inline function hh_terms(D::GammaQ, x::Float64, p::Float64)
    a = D.a; a1 = a - 1.0
    P, _ = gamma_inc(a, x, 0)
    f = P - p
    invx = 1.0 / x
    fp = exp(a1 * log(x) - x - D.lga)        # ρ(x)
    Lp = a1 * invx - 1.0                      # L'
    φ2 = Lp
    ξ = -a1 * invx * invx + Lp * Lp           # L'' + L'^2
    return f, fp, φ2, ξ
end

gamma_quantile(a::Float64, p::Float64; tol::Float64 = 1e-13) = solve(GammaQ(a), p; tol = tol)
