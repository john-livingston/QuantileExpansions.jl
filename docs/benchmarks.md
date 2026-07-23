# Benchmarks

All numbers are measured (BenchmarkTools minimum over preallocated,
allocation-free batches). Primary host: Apple M4, Julia 1.12, `-O3`; the C
reference is the paper authors' own kernel compiled on the same machine with
clang `-O3 -march=native -ffast-math`. x86 numbers come from GitHub Actions
runners via the repository's CI benchmark job. Accuracy figures are worst-case
over **dense parameter sweeps** (144k points for BS), not just grid nodes —
grid-node maxima understate off-node error.

## Black–Scholes implied volatility

![Single-thread progression](assets/bs_progression-light.png#only-light)
![Single-thread progression](assets/bs_progression-dark.png#only-dark)

| kernel | 1 thread | 10 threads × SIMD | accuracy (dense) |
|---|---:|---:|---:|
| adaptive (`bs_implied_vol_generic`) | 69.1 ns | 8.2 ns | 8.9e-14 |
| fixed-2 fast mode | 54.0 ns | **3.6 ns** | 4.7e-11 |
| fixed-3 full precision | 78.7 ns | **4.8 ns** | 1.6e-15 |
| SIMD batch fixed-3 (W=8, serial) | 44.6 ns | — | 1.6e-15 |

That is ~208M full-precision inversions per second on a laptop. On x86 CI
runners the SIMD batch runs 2.65× the same-host scalar speed (Zen3/AVX2, with
the fused vector-seed strategy winning at W=8; on 2-wide NEON the scalar-seed
two-pass strategy wins — the API picks per ISA).

## All four distributions

![Speed vs references](assets/speed_vs_ref-light.png#only-light)
![Speed vs references](assets/speed_vs_ref-dark.png#only-dark)

| Target | per-quantile | reference | speedup | notes |
|---|---:|---:|---:|---|
| BS-IV | 69.1 ns | C: 113.7 ns | 1.65× | + SIMD/threads above |
| Inverse Gaussian | 76.2 ns | 574.9 ns | 7.55× | SIMD batch: 37 ns; threaded: 4.4 ns |
| Gamma (log-space) | 129.8 ns | 269.9 ns | 2.1× | SIMD (a≥20): ~60 ns; fast mode: ~6 ns |
| Beta (logit-space) | 302.7 ns | 821.1 ns | 2.7× | ODE5 certified: 139 to 211 ns (n≤33); seed: ~15 ns |

Gamma/beta cost is dominated by the incomplete-gamma/-beta CDF inside the
residual; the win comes from needing only ~2.5–3 CDF evaluations per quantile.

## Accuracy: coordinate changes in the tails

Worst |Δ ln x| (gamma) / |Δ logit x| (beta) versus Distributions.jl over
u ∈ [1e-8, 1−1e-8]:

| shape | x-space solver | log/logit solver |
|---|---:|---:|
| gamma a=50 | 7.1e-07 | **8.7e-11** |
| gamma a=100 | 1.8e-07 | **2.1e-11** |
| beta (5, 0.2) | 1.3e-06 | **4.3e-14** |
| beta (20, 12.5) | 2.0e-07 | **2.0e-14** |

The x-space solvers are *faster* in those rows only because their absolute
residual test stops before the tail has actually converged — the log/logit
formulations with density-scaled stopping are the recommended paths.

## Reproducing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'  # correctness, 25 testsets
julia --project=. -O3 -t auto bench/run_all.jl
julia --project=. -O3 bench/bench_simd.jl     # prints host CPU + SIMD table
bash ref/fetch.sh                             # fetch the C reference (not vendored)
cc -O3 -march=native -ffast-math ref/bench_iv_c_all_hh4.c -o ref/bench_c && ./ref/bench_c
```

The full results log, including negative results (seed-dispatch experiments
that didn't pan out, certificates that don't fire), is in
[`RESULTS.md`](https://github.com/john-livingston/QuantileExpansions.jl/blob/main/RESULTS.md).
