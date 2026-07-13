# Fast semi-analytic gamma (ODE5 seed-only + one-update) vs the full log-HH4
# solver. Mirrors bench_gamma_log's protocol: fixed shape a, u-grid, per-(a)
# batch amortization, single thread, ns per quantile, min time / N.
#
# Grids are chosen so each row exercises ONE region: the central band
# (|z| <= 2.5, u in [0.007, 0.993]) is pure seed-only at a = 50 and pure
# one-update at a = 5; the full grid [1e-8, 1-1e-8] is the mixed batch.
include("../src/QuantileExpansions.jl"); using .QuantileExpansions
using .QuantileExpansions: _gamma_seed_cf5, norminv
using BenchmarkTools, Printf

function main()
    N = 16384
    us_central = collect(range(0.007, 0.993, length=N))   # |z| <= ~2.5
    us_full    = collect(range(1e-8, 1.0 - 1e-8, length=N))
    out = similar(us_central)
    t(b) = round(minimum(b).time / N, digits=1)

    # raw seed floor: norminv + CF5 polynomial, no incomplete gamma at all
    seedonly!(o, a) = (@inbounds for i in 1:N; o[i] = _gamma_seed_cf5(a, norminv(us_central[i])); end; o)
    norminv!(o)     = (@inbounds for i in 1:N; o[i] = norminv(us_central[i]); end; o)

    @printf("%-40s %10s\n", "kernel (a=50 unless noted)", "ns/q")
    @printf("%-40s %10.1f\n", "norminv only (floor)",              t(@benchmark $norminv!($out)))
    @printf("%-40s %10.1f\n", "CF5 seed = norminv + polynomial",   t(@benchmark $seedonly!($out, 50.0)))
    @printf("%-40s %10.1f\n", "fast seed-only region (a=50)",       t(@benchmark gamma_quantile_fast_batch!($out, 50.0, $us_central)))
    @printf("%-40s %10.1f\n", "fast one-update region (a=5)",       t(@benchmark gamma_quantile_fast_batch!($out, 5.0, $us_central)))
    @printf("%-40s %10.1f\n", "fast mixed full range (a=50)",       t(@benchmark gamma_quantile_fast_batch!($out, 50.0, $us_full)))
    @printf("%-40s %10.1f\n", "full certified solve, central (a=50)", t(@benchmark gamma_quantile_batch!($out, 50.0, $us_central)))
    @printf("%-40s %10.1f\n", "full certified solve, full range (a=50)", t(@benchmark gamma_quantile_batch!($out, 50.0, $us_full)))
    @printf("%-40s %10.1f\n", "full solve (uncertified), central (a=50)", t(@benchmark gamma_quantile_batch!($out, 50.0, $us_central; certified=false)))

    # allocation check
    a1 = @allocated gamma_quantile_fast_batch!(out, 50.0, us_central)
    @printf("\nallocations, fast batch: %d bytes\n", a1)
end
main()
