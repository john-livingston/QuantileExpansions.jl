include("../src/QuantileExpansions.jl")
using .QuantileExpansions
using Test
import Distributions
import SpecialFunctions: erfinv, gamma_inc, beta_inc

# The paper's 328-point delta-constrained grid, shared by every BS testset:
# (k, c, v_true) triples over vols {0.01, 0.05..2.0} x deltas {0.05..0.95}.
function paper_grid()
    vols = vcat(0.01, collect(0.05:0.05:2.0))
    deltas = [0.05, 0.20, 0.30, 0.45, 0.55, 0.70, 0.80, 0.95]
    pts = Tuple{Float64,Float64,Float64}[]
    for v in vols, D in deltas
        k = v*(0.5v - sqrt(2)*erfinv(2D-1)); d1 = -k/v+0.5v; d2 = d1-v
        push!(pts, (k, normcdf(d1) - exp(k)*normcdf(d2), v))
    end
    return pts
end

@testset "special functions" begin
    import SpecialFunctions
    for x in range(-10, 10, length=2001)
        @test normcdf(x) ≈ 0.5*SpecialFunctions.erfc(-x/sqrt(2)) atol=1e-15 rtol=1e-13
        # internal exp-reuse helper must agree with normcdf given the true g
        @test QuantileExpansions.normcdf_withg(x, exp(-0.5x^2)) ≈ normcdf(x) atol=1e-15 rtol=1e-13
    end
    for y in range(0, 50, length=2001)
        @test erfcx_pos(y) ≈ SpecialFunctions.erfcx(y) rtol=1e-13
    end
    # out-of-domain guard (near-parity ITM cancellation): finite, no DomainError
    @test norminv(0.0) == -38.0
    @test norminv(-2.3e-15) == -38.0
    @test norminv(1.0) == 38.0
end

@testset "BS-IV round trip (paper grid)" begin
    maxerr = 0.0
    for (k, c, v) in paper_grid()
        maxerr = max(maxerr, abs(bs_implied_vol(k,c) - v))
        maxerr = max(maxerr, abs(bs_implied_vol_generic(k,c) - v))
    end
    @test maxerr < 1e-12
end

@testset "branch-free fixed-step BS kernel" begin
    e2 = 0.0; e3 = 0.0
    for (k, c, v) in paper_grid()
        e2 = max(e2, abs(bs_implied_vol_fixed(k, c, Val(2)) - v))
        e3 = max(e3, abs(bs_implied_vol_fixed(k, c, Val(3)) - v))
    end
    # Quartic convergence from the seed: worst grid-node seed δ≈0.211 ⇒
    # δ^16 ≈ 1.5e-11 after 2 steps, δ^64 ⇒ machine floor after 3.
    # Bounds guard the published grid figures (2.9e-11 / 1.3e-15) with ~3x
    # headroom for cross-platform FP differences.
    @test e2 < 1e-10
    @test e3 < 5e-15
    @test bs_implied_vol_fixed(0.1, 0.06) == bs_implied_vol_fixed(0.1, 0.06, Val(3))  # default
    @test (@allocated bs_implied_vol_fixed(0.1, 0.06, Val(2))) == 0
end

@testset "SIMD batch kernel" begin
    # vexp: branch-free exp within ~1 ulp of Base.exp
    me = 0.0
    for x in range(-700.0, 700.0, length=20001)
        me = max(me, abs(vexp(x) - exp(x)) / max(abs(exp(x)), 1e-300))
    end
    @test me < 5e-16
    # batch agrees with the scalar fixed kernel and holds its accuracy bounds
    pts = paper_grid()
    ks = [p[1] for p in pts]; cs = [p[2] for p in pts]; vt = [p[3] for p in pts]
    out = similar(ks)
    for NIT in (2, 3), W in (2, 4, 8), vseed in (true, false)
        bs_implied_vol_fixed_batch!(out, ks, cs, Val(NIT), Val(W); vector_seed=vseed)
        dvs = maximum(abs(out[i] - bs_implied_vol_fixed(ks[i], cs[i], Val(NIT))) for i in eachindex(ks))
        @test dvs < 1e-13          # only vexp/vlog-vs-libm rounding may differ
    end
    bs_implied_vol_fixed_batch!(out, ks, cs, Val(3), Val(4))
    @test maximum(abs(out[i] - vt[i]) for i in eachindex(ks)) < 5e-15
    # allocation-free: fused always; two-pass with preallocated workspace
    @test (@allocated bs_implied_vol_fixed_batch!(out, ks, cs, Val(2), Val(4); vector_seed=true)) == 0
    ws = BSFixedWorkspace(length(ks))
    @test (@allocated bs_implied_vol_fixed_batch!(out, ks, cs, Val(2), Val(4); vector_seed=false, ws=ws)) == 0
end

@testset "near-parity ITM prices do not throw (regression)" begin
    # bs price at k=-4, v=0.005: cstar rounds to -2.3e-15 via cancellation;
    # previously norminv(log of negative) threw DomainError.
    k = -4.0
    c = 0.9816843611112658
    for f in (bs_implied_vol, bs_implied_vol_generic,
              (k,c) -> bs_implied_vol_fixed(k, c, Val(2)),
              (k,c) -> bs_implied_vol_fixed(k, c, Val(3)))
        v = f(k, c)
        @test isfinite(v) && v >= 1e-10
    end
end

@testset "Gamma vs Distributions (forward residual)" begin
    mf = 0.0
    for a in [0.5,1.0,2.0,5.0,20.0,100.0], p in [1e-4,0.01,0.1,0.5,0.9,0.99,1-1e-4]
        x = gamma_quantile(a,p)
        mf = max(mf, abs(gamma_inc(a,x,0)[1] - p))
    end
    @test mf < 1e-12
end

@testset "Gamma log-space solver (Hekimoglu port)" begin
    # exact branches
    @test gamma_quantile_log(1.0, 0.3) ≈ -log1p(-0.3) rtol=1e-15
    @test gamma_quantile_log(0.5, 0.7) ≈ Distributions.quantile(Distributions.Gamma(0.5,1.0), 0.7) rtol=1e-8
    # log-x accuracy across shapes incl. deep tails (the x-space solver's weak spot)
    ml = 0.0
    for a in [0.75, 2.0, 5.0, 10.0, 50.0, 100.0],
        u in [1e-8, 1e-4, 0.01, 0.3, 0.5, 0.9, 0.999, 1-1e-6]
        xr = Distributions.quantile(Distributions.Gamma(a,1.0), u)
        ml = max(ml, abs(log(gamma_quantile_log(a, u)) - log(xr)))
    end
    @test ml < 1e-9
    @test (@allocated gamma_quantile_log(5.0, 0.3)) == 0
end

@testset "Beta logit-space solver" begin
    # exact closed-form branches
    @test beta_quantile_logit(100.0, 1.0, 1e-8) ≈ (1e-8)^(1/100) rtol=1e-15
    @test beta_quantile_logit(1.0, 3.0, 0.2) ≈ -expm1(log1p(-0.2)/3) rtol=1e-15
    @test beta_quantile_logit(0.5, 0.5, 0.3) ≈ sinpi(0.15)^2 rtol=1e-15
    # logit-metric accuracy incl. the skewed corners where x-space degrades
    ml = 0.0
    for (a,b) in ((0.75,2.0),(2.0,5.0),(5.0,0.2),(20.0,12.5),(100.0,100.0)),
        u in [1e-8,1e-4,0.01,0.3,0.5,0.9,0.999,1-1e-6]
        xr = Distributions.quantile(Distributions.Beta(a,b), u)
        (xr<=0 || xr>=1) && continue
        xl = beta_quantile_logit(a,b,u)
        (xl<=0 || xl>=1) && continue
        ml = max(ml, abs((log(xl)-log1p(-xl)) - (log(xr)-log1p(-xr))))
    end
    @test ml < 1e-9
    @test (@allocated beta_quantile_logit(2.0, 5.0, 0.4)) == 0
end

@testset "Beta vs Distributions (forward residual)" begin
    mf = 0.0
    for a in [0.5,1.0,2.0,5.0,20.0], b in [0.5,1.0,2.0,5.0,20.0], p in [1e-3,0.01,0.1,0.5,0.9,0.99,0.999]
        x = beta_quantile(a,b,p)
        (0 < x < 1) && (mf = max(mf, abs(beta_inc(a,b,x)[1] - p)))
    end
    @test mf < 1e-12
end

@testset "K4 certified solve" begin
    # certified variants must agree with the full solvers EXACTLY wherever the
    # certificate fires (it may only skip provably-converged confirmation evals)
    for a in [0.75, 2.0, 5.0, 50.0], u in [0.05, 0.2, 0.5, 0.8, 0.95, 1e-4, 0.999]
        @test gamma_quantile_log_cert(a, u) == gamma_quantile_log(a, u)
    end
    for (a,b) in ((2.0,5.0),(20.0,12.5),(5.0,0.2)), u in [0.05, 0.3, 0.5, 0.7, 0.95, 1e-4]
        @test beta_quantile_logit_cert(a, b, u) == beta_quantile_logit(a, b, u)
    end
    @test (@allocated gamma_quantile_log_cert(5.0, 0.3)) == 0
    @test (@allocated beta_quantile_logit_cert(2.0, 5.0, 0.4)) == 0
end

@testset "Inverse Gaussian vs Distributions" begin
    mf = 0.0
    for (μ,λ) in [(1.0,0.5),(1.0,3.0),(1.0,50.0),(2.0,1.0),(3.0,0.3)], p in [1e-4,0.01,0.1,0.5,0.9,0.99,1-1e-4]
        x = ig_quantile(μ,λ,p)
        mf = max(mf, abs(Distributions.cdf(Distributions.InverseGaussian(μ,λ),x) - p))
    end
    @test mf < 1e-12
end

@testset "SIMD IG batch" begin
    ps = collect(range(0.001, 0.999, length=512))
    out = similar(ps)
    for (μ,λ) in ((1.0,3.0),(1.0,0.5),(2.0,1.0)), W in (2,4,8)
        ig_quantile_batch!(out, μ, λ, ps, Val(3), Val(W))
        mf = maximum(abs(Distributions.cdf(Distributions.InverseGaussian(μ,λ),out[i]) - ps[i]) for i in eachindex(ps))
        @test mf < 1e-13
        mr = maximum(abs(out[i]-ig_quantile(μ,λ,ps[i]))/max(out[i],1e-300) for i in eachindex(ps))
        @test mr < 1e-10
    end
    @test (@allocated ig_quantile_batch!(out, 1.0, 3.0, ps, Val(3), Val(8))) == 0
end

@testset "IFT implied-vol sensitivities" begin
    for v in (0.1, 0.5, 1.5), D in (0.2, 0.5, 0.8)
        k = v*(0.5v - sqrt(2)*erfinv(2D-1))
        d1 = -k/v+0.5v; d2 = d1-v
        c = normcdf(d1) - exp(k)*normcdf(d2)
        vs, dvdc, dvdk = bs_implied_vol_grad(k, c)
        @test vs ≈ v atol=1e-12
        hc = 1e-6*c
        @test dvdc ≈ (bs_implied_vol(k,c+hc)-bs_implied_vol(k,c-hc))/(2hc) rtol=1e-6
        hk = 1e-6*max(abs(k),0.1)
        @test dvdk ≈ (bs_implied_vol(k+hk,c)-bs_implied_vol(k-hk,c))/(2hk) rtol=1e-6
    end
    @test (@allocated bs_implied_vol_grad(0.1, 0.06)) == 0
end

@testset "allocation-free scalar solvers" begin
    @test (@allocated bs_implied_vol(0.1, 0.06)) == 0
    @test (@allocated ig_quantile(1.0, 3.0, 0.7)) == 0
end
