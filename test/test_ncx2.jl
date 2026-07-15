# Correctness test for the experimental scaled-noncentral-chi-square quantile
# (E5). Verifies ncx2_quantile matches Distributions' quantile(NoncentralChisq)
# to < 1e-9 relative across the d/λ/u grid, for both polish methods (Newton and
# the analytic-Bessel HH-4). Combos where Distributions itself errors are noted
# and skipped. Run:  julia --project=. test/test_ncx2.jl
using Test, Distributions, Printf
include(joinpath(@__DIR__, "..", "src", "experimental", "ncx2_quantile.jl"))
using .NCX2Quantile
import SpecialFunctions: loggamma

const DS = (0.5, 1.0, 2.35, 5.0, 20.0, 100.0)
const LS = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0)
const US = (1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999, 1 - 1e-6)
# 1e-9 is deliberately loose: the reference is Distributions.quantile, whose
# Marcum-Q inversion loses precision in the deep tails, so a tighter bound (or a
# denser tail grid) could fail on the reference's error, not ours. For true tail
# ground truth use the COS oracle or a BigFloat Poisson-mixture (see RESULTS.md).
const RELTOL = 1e-9
const SCALE = 2.5   # arbitrary c > 0 to exercise the scaling X = c·χ²

pass = 0; fail = 0; skip = 0
failures = Tuple{Float64,Float64,Float64,Symbol,Float64,Float64}[]

@testset "ncx2_quantile vs Distributions" begin
    for d in DS, λ in LS, u in US
        local qref
        try
            qref = quantile(NoncentralChisq(d, λ), u)
        catch e
            @info "SKIP: Distributions errored" d λ u err = typeof(e)
            global skip += 1
            continue
        end
        if !(isfinite(qref) && qref > 0)
            @info "SKIP: reference non-finite/non-positive" d λ u qref
            global skip += 1
            continue
        end
        for method in (:newton, :hh4)
            x = ncx2_quantile(d, λ, u; c = SCALE, method = method)
            rel = abs(x - SCALE * qref) / (SCALE * qref)
            if rel < RELTOL
                global pass += 1
            else
                global fail += 1
                push!(failures, (d, λ, u, method, x / SCALE, rel))
            end
            @test rel < RELTOL
        end
    end
end


@testset "ncx2_quantile guards" begin
    D = NoncentralChisq(5.0, 10.0)
    q, _ = quantile_hh4(D, 5.0, 10.0, 0.5, 2000.0)
    @test abs(cdf(D, q) - 0.5) < 1e-12
    @test_throws ArgumentError ncx2_quantile(5.0, 10.0, 0.5; seed = :bad)
    @test_throws ArgumentError ncx2_quantile(5.0, 10.0, 0.5; method = :bad)
end

# Load-bearing test for the analytic Bessel log-density derivatives (phi2 = f'/f,
# xi = f''/f). The 504-grid solver test does NOT validate these: HH-4 safeguards
# to Newton, so it converges even with wrong derivatives. Here bessel_logderivs is
# compared against an INDEPENDENT reference: the noncentral chi-square(d,λ) density
# as a BigFloat Poisson mixture of central chi-square(d+2j) densities (no Bessel
# functions, not Distributions), differentiated by BigFloat central differences.
@testset "analytic Bessel log-density derivatives (load-bearing)" begin
    function _logf_ncx2_big(d, λ, yb::BigFloat)
        db = BigFloat(d); hλ = BigFloat(λ) / 2; ln2 = log(BigFloat(2))
        central_logpdf(k) = (k / 2 - 1) * log(yb) - yb / 2 - (k / 2) * ln2 - loggamma(k / 2)
        hλ == 0 && return central_logpdf(db)                     # λ=0: central chi-square(d)
        jmax = ceil(Int, Float64(hλ) + 14 * sqrt(Float64(hλ) + 1) + 40)
        acc = zero(BigFloat)
        for j in 0:jmax
            logw = -hλ + j * log(hλ) - loggamma(BigFloat(j) + 1) # log Poisson(j; λ/2)
            acc += exp(logw + central_logpdf(db + 2j))
        end
        return log(acc)
    end
    setprecision(BigFloat, 160) do
        for (d, λ, y) in ((2.35, 0.0, 3.0), (2.35, 1.0, 4.0), (5.0, 10.0, 15.0),
                          (4.0, 4.0, 8.0), (20.0, 100.0, 120.0), (1.0, 5.0, 6.0))
            yb = BigFloat(y); h = yb * BigFloat("1e-6")
            lm = _logf_ncx2_big(d, λ, yb - h)
            l0 = _logf_ncx2_big(d, λ, yb)
            lp = _logf_ncx2_big(d, λ, yb + h)
            Mp_ref = Float64((lp - lm) / (2h))                   # (log f)' = f'/f
            X2_ref = Float64((lp - 2l0 + lm) / (h * h)) + Mp_ref^2  # f''/f = M'' + M'^2
            Mp, X2 = bessel_logderivs(d, λ, y)
            @test isapprox(Mp, Mp_ref; rtol = 1e-8, atol = 1e-10)
            @test isapprox(X2, X2_ref; rtol = 1e-7, atol = 1e-9)
        end
    end
end

@printf("\nGrid: %d (d,λ,u) combos × 2 methods; pass=%d fail=%d skip=%d\n",
        length(DS) * length(LS) * length(US), pass, fail, skip)
if !isempty(failures)
    println("FAILURES (rel >= $RELTOL):")
    for (d, λ, u, m, x, rel) in failures
        @printf("  d=%.2f λ=%.1f u=%.6g method=%s x=%.8g rel=%.3e\n", d, λ, u, m, x, rel)
    end
end
