# Inverse Gaussian quantile — the structural "close cousin" of BS-IV.
# CDF: F(x;μ,λ) = Φ(α) + e^{2λ/μ} Φ(β),
#   α =  √(λ/x)(x/μ - 1),   β = -√(λ/x)(x/μ + 1).
# Same two-Φ shape as Black–Scholes; we port the same scaffold.
#
# Overflow-safe second term: e^{2λ/μ}Φ(β) = ½ erfcx(w)·G, where
#   w = √(λ/x)(x/μ + 1)/√2,   G = exp(-λ(x-μ)²/(2μ²x)) ≤ 1.
# G is exactly the density's exponential factor, so it is reused for f'.
#
# Density ρ(x)=√(λ/(2π x³))·G ⇒ rational log-derivatives:
#   L'  = -3/(2x) - λ/(2μ²) + λ/(2x²)
#   L'' = 3/(2x²) - λ/x³
#   φ2 = L',  ξ = L'' + L'^2

struct IGQ <: QuantileProblem
    μ::Float64
    λ::Float64
    invμ::Float64
    λ_2μ2::Float64        # λ/(2μ²)
    cfp::Float64          # √(λ/(2π)) = √λ·(1/√(2π))
    sqrtλ::Float64
    four_λ_μ::Float64     # 4λ/μ   (seed)
end
function IGQ(μ::Float64, λ::Float64)
    IGQ(μ, λ, 1.0/μ, λ/(2.0*μ*μ), sqrt(λ)*INV_SQRT2PI, sqrt(λ), 4.0*λ/μ)
end

@inline xlo(::IGQ) = 1e-300
@inline xhi(::IGQ) = Inf

# Seed: invert the dominant first Φ term  √(λ/x)(x/μ-1) = z_p, a quadratic in √x.
@inline function seed(D::IGQ, p::Float64)
    zp = norminv(p)
    s = sqrt(zp * zp + D.four_λ_μ)
    u = D.μ * (zp + s) / (2.0 * D.sqrtλ)
    x0 = u * u
    return x0 > 1e-300 ? x0 : 1e-300
end

@inline function hh_terms(D::IGQ, x::Float64, p::Float64)
    μ = D.μ; λ = D.λ
    invx = 1.0 / x
    sqrtλx = sqrt(λ * invx)
    xμ = x * D.invμ                       # x/μ
    α = sqrtλx * (xμ - 1.0)
    w = sqrtλx * (xμ + 1.0) * INV_SQRT2
    z = x - μ
    G = exp(-λ * z * z * 0.5 * D.invμ * D.invμ * invx)   # exp(-λ(x-μ)²/(2μ²x))
    F = normcdf(α) + 0.5 * erfcx_pos(w) * G
    f = F - p
    fp = D.cfp * invx * sqrt(invx) * G    # √(λ/(2π)) x^{-3/2} G
    Lp = -1.5 * invx - D.λ_2μ2 + 0.5 * λ * invx * invx
    Lpp = 1.5 * invx * invx - λ * invx * invx * invx
    φ2 = Lp
    ξ = Lpp + Lp * Lp
    return f, fp, φ2, ξ
end

ig_quantile(μ::Float64, λ::Float64, p::Float64; tol::Float64 = 1e-13) = solve(IGQ(μ, λ), p; tol = tol)

# f''''/f' in x space: with L = log ρ,
#   L'''= -3/x³ + 3λ/x⁴
# f''''/f' = L''' + 3L'L'' + L'³   (enables the K4 certificate)
@inline has_c4(::IGQ) = true
@inline function hh4_c4(D::IGQ, x::Float64)
    λ = D.λ
    invx = 1.0 / x
    invx2 = invx * invx
    Lp = -1.5 * invx - D.λ_2μ2 + 0.5 * λ * invx2
    Lpp = 1.5 * invx2 - λ * invx2 * invx
    Lppp = (-3.0 + 3.0 * λ * invx) * invx2 * invx
    return Lppp + Lp * (3.0 * Lpp + Lp * Lp)
end

"Certified variant of [`ig_quantile`]: skips the confirmation CDF evaluation
when the K4 error model certifies the first HH-4 update below τ."
@inline ig_quantile_cert(μ::Float64, λ::Float64, p::Float64; tol::Float64 = 1e-13) =
    solve_certified(IGQ(μ, λ), p; tol = tol)
