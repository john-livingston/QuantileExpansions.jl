# Batched, SIMD-vectorized fixed-step Inverse Gaussian quantile (fixed (μ,λ),
# batch over p — the natural sampling workload). Same design as the BS batch:
# branch-free seed (norminv_bf + a quadratic in √x) and exactly N HH-4 updates,
# W lanes wide. Per iteration the CDF needs ONE vexp: the identity
# α²/2 = λ(x−μ)²/(2μ²x) makes the density's exponential factor G equal to
# exp(-α²/2), and erfcx (blended) supplies the second Φ term without an exp.
using SIMD: Vec, vload, vstore

@inline function _ig_solve_lanes(D::IGQ, p, ::Val{N}) where {N}
    # seed: invert the dominant first Φ term — quadratic in √x
    zp = norminv_bf(p)
    s = sqrt(zp * zp + D.four_λ_μ)
    u = D.μ * (zp + s) / (2.0 * D.sqrtλ)
    x = max(u * u, 1e-300)
    λ = D.λ
    for _ in 1:N
        invx = 1.0 / x
        sqrtλx = sqrt(λ * invx)
        xμ = x * D.invμ
        α = sqrtλx * (xμ - 1.0)
        w = sqrtλx * (xμ + 1.0) * INV_SQRT2
        G = vexp(-0.5 * α * α)                    # = exp(-λ(x-μ)²/(2μ²x))
        F = phi_withg_bf(α, G) + 0.5 * erfcx_bf(w) * G
        f = F - p
        fp = D.cfp * invx * sqrt(invx) * G
        Lp = -1.5 * invx - D.λ_2μ2 + 0.5 * λ * invx * invx
        Lpp = 1.5 * invx * invx - λ * invx * invx * invx
        φ2 = Lp
        ξ = Lpp + Lp * Lp
        r = f / fp
        denom = -6.0 + r * (6.0 * φ2 - r * ξ)
        xn = x + 3.0 * r * (2.0 - r * φ2) / denom
        xn = sel(abs(denom) < 1e-20, x - r, xn)
        xn = sel(isfinite(xn), xn, x)
        x = max(xn, 1e-300)
    end
    return x
end

"""
    ig_quantile_batch!(out, μ, λ, ps, ::Val{N}, ::Val{W})

Batched Inverse Gaussian quantiles at fixed (μ, λ): branch-free seed + exactly
N HH-4 updates, vectorized W lanes wide. Allocation-free.
"""
function ig_quantile_batch!(out::Vector{Float64}, μ::Float64, λ::Float64,
                            ps::Vector{Float64}, ::Val{N}, ::Val{W}) where {N,W}
    n = length(ps)
    @assert length(out) == n
    D = IGQ(μ, λ)
    i = 1
    @inbounds while i + W - 1 <= n
        p = vload(Vec{W,Float64}, ps, i)
        vstore(_ig_solve_lanes(D, p, Val(N)), out, i)
        i += W
    end
    @inbounds while i <= n
        out[i] = _ig_solve_lanes(D, ps[i], Val(N))
        i += 1
    end
    return out
end

"Threads × SIMD variant of [`ig_quantile_batch!`] (contiguous per-thread chunks)."
function ig_quantile_batch_threaded!(out::Vector{Float64}, μ::Float64, λ::Float64,
                                     ps::Vector{Float64}, ::Val{N}, ::Val{W}) where {N,W}
    n = length(ps)
    @assert length(out) == n
    D = IGQ(μ, λ)
    nt = Threads.nthreads()
    chunk = cld(n, nt)
    Threads.@threads :static for t in 1:nt
        lo = (t - 1) * chunk + 1
        hi = min(t * chunk, n)
        lo > hi && continue
        i = lo
        @inbounds while i + W - 1 <= hi
            p = vload(Vec{W,Float64}, ps, i)
            vstore(_ig_solve_lanes(D, p, Val(N)), out, i)
            i += W
        end
        @inbounds while i <= hi
            out[i] = _ig_solve_lanes(D, ps[i], Val(N))
            i += 1
        end
    end
    return out
end
