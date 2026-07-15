# E5 benchmark + analysis for the scaled noncentral chi-square quantile.
# Produces: seed accuracy, iteration counts (Patnaik vs Sankaran, Newton vs
# HH-4), ns/quantile (seed-only, seed+Newton, Distributions reference), and the
# per-step Bessel cost. Every number printed here is measured, not estimated.
# Run:  julia --project=. bench/bench_ncx2.jl
using Distributions, SpecialFunctions, BenchmarkTools, Printf, Statistics
include(joinpath(@__DIR__, "..", "src", "experimental", "ncx2_quantile.jl"))
using .NCX2Quantile

const DS = (0.5, 1.0, 2.35, 5.0, 20.0, 100.0)
const LS = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0)
const US = (1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999, 1 - 1e-6)
# "bulk" = away from the extreme tails, where asymptotic seeds are designed to work
const UBULK = (0.1, 0.5, 0.9, 0.999)

# ----------------------------------------------------------------------------
println("="^78)
println("1. SEED ACCURACY  (relative error of the raw seed vs reference quantile)")
println("="^78)
function seed_relerr(seedfn)
    all = Float64[]; bulk = Float64[]
    for d in DS, λ in LS, u in US
        q = quantile(NoncentralChisq(d, λ), u)
        (isfinite(q) && q > 0) || continue
        s = seedfn(d, λ, u)
        re = abs(s - q) / q
        push!(all, re)
        u in UBULK && push!(bulk, re)
    end
    return all, bulk
end
for (name, fn) in (("Patnaik", patnaik_seed), ("Sankaran", sankaran_seed))
    all, bulk = seed_relerr(fn)
    @printf("%-9s  full grid: median=%.2e  90pct=%.2e  max=%.2e   |  bulk(u∈%s): median=%.2e max=%.2e\n",
            name, median(all), quantile(all, 0.9), maximum(all),
            string(UBULK), median(bulk), maximum(bulk))
end

# ----------------------------------------------------------------------------
println("\n" * "="^78)
println("2. ITERATION COUNTS to xtol=1e-13  (converged = rel<1e-9 vs reference)")
println("   in-basin = seed rel-err < 0.5 (quartic acceleration applies);")
println("   out-of-basin = deep-tail seed failures, safeguarded bisection.")
println("="^78)
function iter_stats(seedfn, method)
    inb = Int[]; outb = Int[]; nconv = 0; ntot = 0
    for d in DS, λ in LS, u in US
        D = NoncentralChisq(d, λ)
        q = quantile(D, u)
        (isfinite(q) && q > 0) || continue
        x0 = seedfn(d, λ, u)
        srel = abs(x0 - q) / q
        x, it = method === :newton ? quantile_newton(D, u, x0) :
                                     quantile_hh4(D, d, λ, u, x0)
        ntot += 1
        abs(x - q) / q < 1e-9 && (nconv += 1)
        (srel < 0.5 ? push!(inb, it) : push!(outb, it))
    end
    return inb, outb, nconv, ntot
end
@printf("%-9s %-7s | %-28s | %-24s | converged\n", "seed", "method",
        "in-basin iters (n, mean, max)", "out-basin (n, mean, max)")
for (sname, sfn) in (("Patnaik", patnaik_seed), ("Sankaran", sankaran_seed)),
    m in (:newton, :hh4)
    inb, outb, nconv, ntot = iter_stats(sfn, m)
    @printf("%-9s %-7s | %5d  mean=%.2f  max=%-3d | %4d  mean=%5.1f  max=%-3d | %d/%d\n",
            sname, string(m), length(inb), mean(inb), maximum(inb),
            length(outb), (isempty(outb) ? 0.0 : mean(outb)),
            (isempty(outb) ? 0 : maximum(outb)), nconv, ntot)
end

# ----------------------------------------------------------------------------
println("\n" * "="^78)
println("3. SPEED  (BenchmarkTools, allocation-free; ns/quantile)")
println("="^78)
reps = ((2.35, 1.0, 0.9), (5.0, 10.0, 0.5), (20.0, 100.0, 0.3), (100.0, 1000.0, 0.999))
@printf("%-24s | %10s %11s %10s %10s | %9s %8s\n", "(d, λ, u)", "seed-only",
        "seed+Newton", "seed+HH4", "reference", "N speedup", "N iters")
for (d, λ, u) in reps
    D = NoncentralChisq(d, λ)
    tseed = @belapsed sankaran_seed($d, $λ, $u)
    tnewt = @belapsed ncx2_quantile($d, $λ, $u; method = :newton)
    thh4  = @belapsed ncx2_quantile($d, $λ, $u; method = :hh4)
    tref  = @belapsed quantile($D, $u)
    _, nit = quantile_newton(D, u, sankaran_seed(d, λ, u))
    @printf("(%.2f, %.1f, %.4g)%s | %7.1f ns %8.0f ns %7.0f ns %7.0f ns | %8.1fx %7d\n",
            d, λ, u, " "^max(0, 8 - length(@sprintf("%.4g", u))),
            tseed*1e9, tnewt*1e9, thh4*1e9, tref*1e9, tref/tnewt, nit)
end

# ----------------------------------------------------------------------------
println("\n" * "="^78)
println("4. PER-STEP COST BREAKDOWN  (representative d=5, λ=10, y=15)")
println("="^78)
d = 5.0; λ = 10.0; y = 15.0; D = NoncentralChisq(d, λ)
tcdf = @belapsed cdf($D, $y)
tpdf = @belapsed pdf($D, $y)
tbes = @belapsed bessel_logderivs($d, $λ, $y)
tnorm = @belapsed sankaran_seed($d, $λ, $0.3)
@printf("cdf (Marcum Q, residual)     : %8.1f ns\n", tcdf*1e9)
@printf("pdf (density, Newton f')     : %8.1f ns\n", tpdf*1e9)
@printf("bessel_logderivs (φ2,ξ)      : %8.1f ns   <-- the extra HH-4 special branch\n", tbes*1e9)
@printf("Sankaran seed (incl norminv) : %8.1f ns\n", tnorm*1e9)
@printf("\nBessel branch as fraction of one cdf residual eval: %.0f%%\n", 100*tbes/tcdf)
@printf("Bessel branch in pdf-equivalents                  : %.1f x pdf\n", tbes/tpdf)
