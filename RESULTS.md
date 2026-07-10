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
| **Julia (this repo)**       |     **69.1** | **8.2** (10t)  |    4.6e-14  |
| C reference (same Mac)      |        113.7 |              — |    8.5e-14  |
| Numba (spec, other HW)      |          134 |       24 (12t) |     ~1e-14  |

**1.65× faster than C single-threaded; 2.9× faster than Numba's parallel
number.** (C/Numba spec numbers were 87–125 ns on other HW; on this Mac the same
C source runs 113.7 ns.)

Two BS entry points trade speed against accuracy. `bs_implied_vol_generic` (the
one benchmarked above) tests `|f| < tol` *before* updating, so it stops one
update early: **8.9e-14** worst-case on a dense delta×vol sweep (4.6e-14 on the
grid nodes). `bs_implied_vol` updates then tests, spending one extra step to
reach **1.6e-15** dense — ~50× better than C — at a few ns more. Both are far
inside any practical tolerance. (Accuracy figures here and below are worst-case
over a dense 144k-point sweep of delta ∈ [0.05,0.95] × vol ∈ [0.01,2.0], not
just the 328 benchmark nodes, which understate off-node error.)

## All four distributions through ONE generic solver

Each converges to a forward residual `|F(x)−p| ≈ 1e-13`; speed vs Julia's
`Distributions.quantile` (which wraps the standard C/Rmath algorithms).

| Target            | per-quantile | reference   | speedup | mean iters |
|-------------------|-------------:|------------:|--------:|-----------:|
| Black–Scholes IV  |     69.1 ns  | C: 113.7 ns | 1.65×   | 2.84       |
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
- **Second `exp` eliminated** via the exact identity `exp(-d₂²/2) = exp(-d₁²/2)·e^{-κ}`
  (since `d₂=d₁-v` and `d₁v - v²/2 = -κ`): the gaussian factor for `d₂` is free
  once `d₁`'s is known, so each HH-4 iteration costs one `exp`, not two.
- **IG overflow-safe form:** `e^{2λ/μ}Φ(β) = ½ erfcx(w)·G` with `G≤1`, and
  `erfcx` (scaled erfc) avoids an exp entirely on its mid/large ranges; `G` is
  reused as the density's exponential factor.
- **Safeguarded polish:** HH-4 → Newton → damped-bracket fallback keeps the
  generic solver robust in extreme tails without slowing the (never-triggered)
  fast path.

## Branch-free fixed-step kernel (`bs_implied_vol_fixed`)

The adaptive solver spends ~0.8 of its ~2.8 residual evaluations merely *proving*
convergence. Running a fixed number of HH-4 updates instead removes both that
cost and the iteration-count divergence between inputs — the property SIMD needs.

| kernel   | serial ns/IV | threaded (10t) | max abs err (dense) | (grid nodes) |
|----------|-------------:|---------------:|--------------------:|-------------:|
| adaptive |         69.1 |            8.2 |             8.9e-14 |      4.6e-14 |
| fixed-2  |     **~52**  |        **6.3** |          **2.3e-8** |      2.9e-11 |
| fixed-3  |         ~79  |              — |             1.6e-15 |      1.3e-15 |

Accuracy follows from quartic convergence alone: a seed with relative error `δ`
lands at `δ^(4^N)`. The 328 grid *nodes* understate this badly: the worst node
seed has `δ ≈ 0.211`, but a dense sweep between nodes finds `δ ≈ 0.33` (at
v≈0.76, delta≈0.08), and `0.33¹⁶ ≈ 2.5e-8` — matching the measured dense
fixed-2 error of 2.3e-8. Three steps give `δ⁶⁴ ≈ 0`, i.e. the machine floor
across the entire region, so **fixed-3 is a safe branch-free default; fixed-2
is a ~1e-8 fast mode**.

**Validity domain.** These figures hold for delta ∈ [0.05, 0.95],
vol ∈ [0.01, 2.0]. Outside the delta band the seed can leave the quartic
convergence basin and fixed-step errors grow to percent level with no error
signal (e.g. 5.4% at delta=0.001) — use the adaptive `bs_implied_vol`, which
holds ~1e-15 there, for deep-OTM inputs. Results clamp to v ∈ [1e-10, 5.0]
(the domain the C reference enforces); NaN inputs propagate to NaN.

Caveat: only the *iteration* is branch-free. `bs_seed` still branches on regime
and the Cody `erfc` on `|d|` range; both must be bucketed or blended before this
can actually be vectorized.

## Seed admissibility — a negative result

Two HH-4 steps reach `ε` from relative seed error `δ` iff `δ ≤ δ* = ε^(1/16)`
(13.3% for `ε=1e-14`). On the reference grid **19 of 328 seeds violate `δ*`** and
11 need a 3rd update, clustered in three places:

| cluster | regime          | where                          | δ_seed |
|---------|-----------------|--------------------------------|-------:|
| 1       | `mild-P7`       | κ≈0.17–0.44, delta=0.05, low v |  0.211 |
| 2       | `P3`            | κ≈1.31–1.34 (Κ4 edge), high v  |  0.189 |
| 3       | `tail-override` | κ≈0.51–0.58 (just past κ*=0.5) |  0.191 |

Five re-tunings were tried — a second Mills pass, a `d₂`-based polynomial-validity
switch, extending `avg(P3,P7)` through Κ4, and a `κ*` sweep — and **all were worse
or equal** to the published boundaries. The `d₂` switch in particular confirmed
the authors' `κ > 0.5` guard: the deep/Mills seed is *worse* than P7 in that
corner. The regime map of arXiv:2606.10245 is a genuine local optimum.

Consequence: making `fixed-2` exact to 1e-14 needs the worst-case seed error cut
from 33% (dense worst, at v≈0.76, delta≈0.08) to 13.3% — concentrated in the
**low-delta edge of `mild-P7`**. That requires a better expansion there (P9/P11,
or a small-`v` asymptotic seed), not a re-tuned boundary.

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
