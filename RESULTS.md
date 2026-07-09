# Regime-Split Quantile Solver — Results

A single, distribution-generic regime-split quantile solver in Julia. Every
target is an inverse-CDF problem `F(x)=p`, solved by a closed-form regime seed +
one universal **HH-4** (Householder order 3, quartic) polish. The generic
`solve(D, p)` is written **once**; each distribution supplies a tiny interface
(`seed`, `hh_terms`) and the compiler monomorphizes it into specialized,
allocation-free, C-speed code.

**Hardware:** Apple (arm64, darwin 24.6), 14 cores. Julia 1.12.6 (`-O3`).
C reference compiled with Apple clang 17 `-O3 -march=native -ffast-math`.

## Headline: BS-IV beats the C reference on identical hardware

Identical 328-point delta-constrained grid (the paper's `build_grid`), tiled
×5000 = 1.64M IVs, best-of-N timing, allocation-free.

| Implementation              | Serial ns/IV | Threaded ns/IV | max abs err |
|-----------------------------|-------------:|---------------:|------------:|
| **Julia (this repo)**       |     **77.3** | **9.7** (14t)  |     ~1e-14  |
| C reference (same Mac)      |        113.7 |              — |    8.5e-14  |
| Numba (spec, other HW)      |          134 |       24 (12t) |     ~1e-14  |

**1.47× faster than C single-threaded, ~6× better max error; 2.5× faster than
Numba's parallel number.** (C/Numba spec numbers were 87–125 ns on other HW;
on this Mac the same C source runs 113.7 ns.)

## All four distributions through ONE generic solver

Each converges to a forward residual `|F(x)−p| ≈ 1e-13`; speed vs Julia's
`Distributions.quantile` (which wraps the standard C/Rmath algorithms).

| Target            | per-quantile | reference   | speedup | mean iters |
|-------------------|-------------:|------------:|--------:|-----------:|
| Black–Scholes IV  |     77.3 ns  | C: 113.7 ns | 1.47×   | 2.84       |
| Inverse Gaussian  |     76.2 ns  | 574.9 ns    | 7.55×   | 3.36       |
| Gamma             |    174.3 ns  | 269.7 ns    | 1.55×   | 2.57       |
| Beta              |    461.5 ns  | 821.1 ns    | 1.78×   | 2.94       |

(Gamma/Beta per-quantile cost is dominated by the incomplete-gamma/-beta CDF in
the residual; the win is **fewer expensive CDF evals** — a great seed + quartic
polish needs ~2.5–3 evals where bracketing libraries need more.)

## Why it's fast

- **Generic monomorphization (the spec's main payoff).** The BS solver routed
  through the generic interface is *as fast as* (slightly faster than) the
  hand-written one — Julia inlines `seed`/`hh_terms` per type with zero
  abstraction overhead. One solver, four distributions, C-speed each.
- **HH-4 needs only rational log-derivatives.** For residual `f=F(x)−p`,
  `f'=ρ(x)`, and `φ2=f''/f'=L'(x)`, `ξ=f'''/f'=L''(x)+L'(x)²` where `L=log ρ`.
  `L'` is rational for normal/IG/gamma/beta ⇒ the quartic step is nearly free.
- **Single-exp Cody `erfc`** (the exp-split accuracy trick is unneeded at 1e-14
  price tolerance) and **`normcdf_pdf`** returning Φ(d1) and φ(d1) from one
  `exp(-d1²/2)` (erfc's internal gaussian factor *is* φ/a1).
- **`exp(κ)` hoisted** out of the BS HH-4 loop.
- **IG overflow-safe form:** `e^{2λ/μ}Φ(β) = ½ erfcx(w)·G` with `G≤1`, and
  `erfcx` (scaled erfc) avoids an exp entirely on its mid/large ranges; `G` is
  reused as the density's exponential factor.
- **Safeguarded polish:** HH-4 → Newton → damped-bracket fallback keeps the
  generic solver robust in extreme tails without slowing the (never-triggered)
  fast path.

## Accuracy notes

All targets hit `|F(x)−p| < 1e-13` across wide parameter grids. Relative error in
`x` vs the reference library matches to machine precision in the bulk; in extreme
tails (p≈1e-8) it is conditioning-limited (`dx/x ≈ (1/a)·dp/p`), i.e. an absolute-
vs-relative tolerance artifact, not a solver failure. Beta near `x→1` uses the
symmetry `I_x(a,b)=1−I_{1−x}(b,a)` to stay off the precision wall.

## Reproduce

```
julia --project=. -O3 -t 14 bench/run_all.jl      # full table above
julia --project=. test/grid_bs.jl                 # BS accuracy on paper grid
julia --project=. test/test_gamma.jl              # vs Distributions.jl

# C head-to-head (upstream ref is fetched, not vendored — see ref/README.md):
bash ref/fetch.sh
cc -O3 -march=native -ffast-math ref/bench_iv_c_all_hh4.c -o ref/bench_c && ./ref/bench_c
```
