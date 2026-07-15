# Correctness test for the experimental scaled-noncentral-chi-square quantile
# (E5). Verifies ncx2_quantile matches Distributions' quantile(NoncentralChisq)
# to < 1e-9 relative across the d/λ/u grid, for both polish methods (Newton and
# the analytic-Bessel HH-4). Combos where Distributions itself errors are noted
# and skipped. Run:  julia --project=. test/test_ncx2.jl
using Test, Distributions, Printf
include(joinpath(@__DIR__, "..", "src", "experimental", "ncx2_quantile.jl"))
using .NCX2Quantile

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

@printf("\nGrid: %d (d,λ,u) combos × 2 methods; pass=%d fail=%d skip=%d\n",
        length(DS) * length(LS) * length(US), pass, fail, skip)
if !isempty(failures)
    println("FAILURES (rel >= $RELTOL):")
    for (d, λ, u, m, x, rel) in failures
        @printf("  d=%.2f λ=%.1f u=%.6g method=%s x=%.8g rel=%.3e\n", d, λ, u, m, x, rel)
    end
end
