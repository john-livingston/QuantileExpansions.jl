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
| fixed-2  |     **~54**  |        **6.3** |         **4.7e-11** |      2.9e-11 |
| fixed-3  |         ~79  |              — |             1.6e-15 |      1.3e-15 |

Accuracy follows from quartic convergence alone: a seed with relative error `δ`
lands at `δ^(4^N)`. The 328 grid *nodes* understate this badly: the worst node
seed has `δ ≈ 0.211`, but a dense sweep between nodes finds `δ ≈ 0.23` after the
d2-validity dispatch (below; `δ ≈ 0.34` before it, which gave the earlier
`2.3e-8` figure), and `0.23¹⁶ ≈ 7e-11` — near the measured dense fixed-2 error
of 4.7e-11. Three steps give `δ⁶⁴ ≈ 0`, i.e. the machine floor across the
entire region, so **fixed-3 is a safe branch-free default; fixed-2 is a
~5e-11 fast mode** (see the closing section for the dispatch that landed this).

**Validity domain.** These figures hold for delta ∈ [0.05, 0.95],
vol ∈ [0.01, 2.0]. Outside the delta band the seed can leave the quartic
convergence basin and fixed-step errors grow to percent level with no error
signal (e.g. 5.4% at delta=0.001) — use the adaptive `bs_implied_vol`, which
holds ~1e-15 there, for deep-OTM inputs. Results clamp to v ∈ [1e-10, 5.0]
(the domain the C reference enforces); NaN inputs propagate to NaN.

Caveat: only the *iteration* is branch-free. `bs_seed` still branches on regime
and the Cody `erfc` on `|d|` range; both must be bucketed or blended before this
can actually be vectorized.

## SIMD batch kernel (`bs_implied_vol_fixed_batch!`)

The fixed-step kernel's uniform instruction path enables explicit vectorization
(SIMD.jl `Vec{W,Float64}`), built on a branch-free `vexp` (Cody–Waite +
degree-13 polynomial + exponent bit-trick, 1 ulp), a branch-free `vlog`, a
blended 3-branch Cody Φ, and a blended 3-branch Acklam `norminv_bf` — the
spec's "transcendental-free seed chain" idea realized as select-blended
vector code.

Two seed strategies behind one API (the polish is vectorized in both):
- **fused** (`vector_seed=true`): reduction, regime seed, and polish all W lanes
  wide in one pass. The blended seed evaluates *all five* regime candidates
  (incl. the deep/Mills seed's two `norminv_bf`) on every lane — ~3× the work
  of the average scalar seed, paid for only when lanes are wide.
- **two-pass** (`vector_seed=false`): scalar branchy seed staged through a
  workspace, vectorized polish only.

Measured single-thread, 1.64M-IV grid, best width (W=8) on each host —
within-run comparisons only (CI VMs vary between runs):

| host                        | scalar f2/f3 | two-pass f2/f3 | fused f2/f3   | winner  |
|-----------------------------|-------------:|---------------:|--------------:|---------|
| Apple Silicon (NEON 2-wide) | 51.6 / 78.7  | **33.2 / 44.6**| 41.1 / 51.9   | two-pass|
| GitHub Zen3 (AVX2 4-wide)   | 138.5 / 191.4| 66.4 / 81.7    | **57.2 / 72.2**| fused  |
| GitHub Zen4 (AVX-512)*      | 120.9 / 167.8| **51.1 / 61.3**| —             | (earlier run) |

*The Zen4 row is an earlier run before the fused strategy existed; its two-pass
polish vectorized 4.5× (that core's double-pump ceiling).

The ISA asymmetry is exactly the blend-overhead story: on 2-wide NEON the
vector seed's ~3× work blowup cannot be covered by 2 lanes (fused *loses*
there); from ~8 effective lanes on x86 it wins (fixed-3: 81.7 → 72.2 ns on
Zen3). The default is therefore `vector_seed = (Sys.ARCH === :x86_64)`.

Consequences:
- **Full-precision fixed-3 batch is faster than scalar fast-mode fixed-2 on
  every host measured** — on batches there is no speed reason to give up
  accuracy: 44.6 ns on NEON (2.5× the reference C), 2.65× the same-host scalar
  on Zen3.
- Accuracy is identical to the scalar fixed kernel up to `vexp`/`vlog`-vs-libm
  rounding (≤1e-13 pointwise; same 2.9e-11 / 1.3e-15 grid figures), same
  validity domain (delta ∈ [0.05, 0.95]).
- Remaining levers: the mild-P7 seed improvement (would make 2-step batches
  exact at 1e-14, cutting another ~30%), further divide reduction in the
  blended Φ, and thread × SIMD composition.

## Gamma in log space (`gamma_quantile_log`) — port of A. Hekimoglu's engine

Port of the collaborator's C engine (`gamma_quant_full_c5_dynamic_boundary_
widesafe_engine.c`, "full" mode) into the generic solver: the unknown becomes
y = ln x, which keeps x > 0 with no clamping and collapses the HH-4 ratios to
polynomials (F''/F' = a−x, F'''/F' = (a−x)²−x). His method drops into the same
`solve()` as a new `QuantileProblem` — the port is the seeds, the regime map,
and one interface addition: an overridable `converged()` hook, because tail
accuracy requires his **density-scaled stopping** (|f| < tol AND |f/F'| ≲ 2e-14;
an absolute CDF residual alone fires before x converges where the density is
tiny).

Seeds per regime: exact a=1 / a=½; lower-tail series reversion (c2..c4) gated
by his **analytic c5 boundary** (first omitted coefficient certifies the seed
to 1e-13 post-polish — the same admissibility mathematics as our δ*, used as a
per-point certificate); gamma-Mills survival seed for the far upper tail (gated
on its asymptotics actually holding, x ≫ a); 5th-order Cornish-Fisher for
central large-a; Wilson-Hilferty fallback.

Head-to-head (per-a batches, 16384 u-points on [1e-8, 1−1e-8], amortized
constructors, single thread; accuracy = max |Δ ln x| vs Distributions):

| a    | ours ns | log-port ns | Distr. ns | ours max err | log-port max err |
|------|--------:|------------:|----------:|-------------:|-----------------:|
| 0.75 | 225.6   | 246.4       | 321.9     | 6.9e-11      | 1.3e-10          |
| 1    | 170.3   | **2.7**     | 155.1     | 6.1e-11      | 3.6e-15          |
| 2    | 147.6   | 132.1       | 241.6     | 8.8e-11      | 2.3e-14          |
| 5    | 143.6   | 129.8       | 269.9     | 1.6e-10      | 3.7e-11          |
| 10   | 178.4   | 165.3       | 336.0     | 4.3e-09      | 4.7e-11          |
| 50   | 99.7    | 151.0       | 280.1     | **7.1e-07**  | 8.7e-11          |
| 100  | 99.1    | 149.9       | 280.4     | 1.8e-07      | 2.1e-11          |

**Verdict: the log-space formulation wins.** Comparable-or-better speed in the
bulk, and uniformly ~1e-10 log-accuracy where our x-space solver's absolute
tolerance lets tail error degrade to 1e-7 (it is *faster* there precisely
because it stops too early). Both beat Distributions by ~1.5–2.5×.
`gamma_quantile_log` is the recommended gamma path; `gamma_quantile` is kept
as the x-space baseline. Port caveats: the Mills seed needed an explicit
validity gate (yt > 2·max(a−1,1)) his dispatch achieves differently, and the
a=½ exact branch inherits Acklam-norminv accuracy (~1e-8 deep-tail log error)
exactly as his engine does.

## Beta in logit space (`beta_quantile_logit`)

The same coordinate-change treatment applied to beta, from the collaborator's
beta engine: unknown y = logit(x), so x stays in (0,1) intrinsically, x and 1−x
are both computed stably from y at either endpoint, and the HH-4 ratios are
polynomials (F''/F' = a−(a+b)x, F'''/F' = (a−(a+b)x)² − (a+b)x(1−x)). Reuses
the existing x-space regime seeds mapped through logit, plus his exact
closed-form branches (b=1 ⇒ x=p^{1/a}, a=1, and the (½,½) arcsine law) — the
generic seeds are arbitrarily bad in exactly those corners. Density-scaled
convergence as for gamma.

| (a,b)      | x-space ns | logit ns | Distr. ns | x-space maxerr | logit maxerr |
|------------|-----------:|---------:|----------:|---------------:|-------------:|
| (0.5,0.5)  |      239.5 |  **3.3** |     439.0 |        4.6e-13 | 7.5e-10*     |
| (0.75,2)   |      617.8 |    675.5 |    1315.4 |        1.8e-10 | 1.9e-14      |
| (2,5)      |      302.6 |    381.6 |     537.7 |        2.0e-12 | 2.0e-14      |
| (5,0.2)    |      674.0 |    664.0 |    1276.6 |    **1.3e-06** | 4.3e-14      |
| (20,12.5)  |      421.1 |    515.6 |     924.7 |        2.0e-07 | 2.0e-14      |
| (100,100)  |      417.7 |    504.8 |     680.3 |        1.2e-12 | 1.9e-14      |

(accuracy = max |Δ logit x| vs Distributions on u ∈ [1e-8, 1−1e-8]; *the
(½,½) figure is the Float64 representation limit of x near 1, not solver error)

Same verdict as gamma: **the logit formulation holds ~2e-14 uniformly** where
the x-space solver degrades to ~1e-6 in skewed corners; its modest bulk-speed
cost is the density-scaled stop demanding true convergence (x-space's apparent
speed there comes from stopping early, wrong). Both beat Distributions ~1.7–2×.
`beta_quantile_logit` is the recommended beta path.

## SIMD Inverse Gaussian batch (`ig_quantile_batch!`)

The BS SIMD treatment applied to IG (fixed (μ,λ), batch over p — the natural
sampling workload): branch-free seed (`norminv_bf` + a quadratic in √x) and
exactly N HH-4 updates, W lanes wide. Per iteration the CDF costs **one vexp**:
α²/2 equals the density exponent (so the gaussian factor G = e^{-α²/2} is
shared, as in the scalar solver), and the second Φ term uses a blended
branch-free `erfcx`. N=3 already reaches machine forward residual (max
|F(x)−p| ≈ 5e-16 — the IG seed is the best in the repo), so fixed-3 is exact:

| (μ,λ)    | scalar ns/q | batch W=8 N=3 | speedup | vs Distributions (~600 ns) |
|----------|------------:|--------------:|--------:|---------------------------:|
| (1, 3)   |        63.8 |      **37.2** |    1.7× |                        ~16× |
| (1, 0.5) |        82.0 |      **37.2** |    2.2× |                        ~16× |
| (2, 1)   |        82.0 |      **37.3** |    2.2× |                        ~16× |

(2-wide NEON; x86 wide-vector gains follow the BS pattern.)

## K4 acceptance certificates (`solve_certified`)

Generalization of Hekimoglu's y6/K4 acceptance certificates to the whole
interface. His beta-specific K4 turns out to be exactly the classical
Householder-3 asymptotic error constant

    e_next ≈ K4·e⁴,   K4 = |5c₂³ − 5c₂c₃ + c₄|,
    c₂ = φ₂/2,  c₃ = ξ/6,  c₄ = (f⁗/f′)/24

(verified algebraically against his `(6A³+7nAD−nD(1−2x))/24`), and f⁗/f′ is
rational for every distribution here, like the other ratios. `solve_certified`
evaluates `hh_terms` once at the seed, applies the HH-4 update, and **exits
without a confirmation evaluation** when `16·K4·r⁴ ≤ τ`; uncertified points
fall through to the adaptive loop, so it is never less accurate — on every test
point the certified and full solvers agree *bit-for-bit*. Implemented for
`GammaLogQ` (f⁗/f′ = A³−3Ax−x) and `BetaLogitQ` (A³−3AnD−nD(1−2x)), plus his
**CF5 central seed** for beta (5th-cumulant Cornish-Fisher, cumulants
precomputed per (a,b) in the constructor).

Central-grid effect (u ∈ [0.15, 0.85], per-shape batches):
- gamma: 180 → **126 ns** (1.43×) at a = 5 and a = 50
- beta (20,12.5): 372 → **295 ns** (1.26×); the CF5 seed also cut the
  *uncertified* baseline from 524 to 372 ns by itself
- beta (2,5): no certificate gain — at small n the CF5 seed is not sharp
  enough for K4·r⁴ ≤ τ; this is exactly the pocket his ODE5 seed serves.

Still unported from his beta engine: the ODE5 central seed (per-(a,b)
z-polynomials) and the zero-evaluation y6 seed-intrinsic certificate — together
those are his 18–22 ns central tier — and the endpoint power-series I_x.

## Threads × SIMD, amortized batches, and certificate coverage

Three additions rounding out the performance surface (Apple M4, 10 threads):

**Threads × SIMD** (`bs_implied_vol_fixed_batch_threaded!`,
`ig_quantile_batch_threaded!`): the SIMD batch kernels over contiguous
per-thread chunks (disjoint index ranges, so even the two-pass workspace is
shared safely; threaded ≡ serial bit-for-bit).

| kernel                     | 1 thread | 10 threads × W=8 | throughput |
|----------------------------|---------:|-----------------:|-----------:|
| BS fixed-2 (fast mode)     |  33.2 ns |      **3.64 ns** |   ~275 M/s |
| BS fixed-3 (full 1.3e-15)  |  44.6 ns |      **4.81 ns** |   ~208 M/s |
| IG fixed-3 (machine-exact) |  37.2 ns |      **4.35 ns** |   ~230 M/s |

**Amortized batch APIs** (`gamma_quantile_batch!`, `beta_quantile_batch!`):
construct the problem struct once per shape (lnΓ / logbeta / CF5 cumulants /
c5 boundary) and reuse across the probability vector — the reference engines'
batch protocol. Bit-for-bit identical to the per-call solvers; gamma a=5:
88.6 → 82.7 ns, beta (20,12.5): 337.5 → 302.7 ns (7–10%).

**K4 certificates for IG and BS** (`ig_quantile_cert`, `bs_implied_vol_cert`):
the `hh4_c4` ratios (IG: L‴+3L′L″+L′³ with L‴ = (−3+3λ/x)/x³; BS:
(s³−3s(q+s)+3(q+2s))/v³, s = d₁d₂, q = d₁²+d₂²), FD-validated to ~7e-9. Honest
result: **near-neutral speed** (BS 67.3 vs 67.4 ns; IG 61.3 vs 63.7). The
certificate only fires where the seed is already superb — 48/328 BS grid
points, essentially never for IG's quadratic seed at τ=1e-14 (it activates in
the near-Gaussian λ/μ ≳ 100 regime). Certificates pay where seeds are
CF5-grade (gamma central: 1.43×); coverage here completes the interface, not
the speedup. One documented edge: on extreme IG stress points the certified
and full paths can differ at the last-3-ulps level (both correct; the full
path's residual-based stop takes one extra ulp-level step).

## Implied-vol sensitivities (`bs_implied_vol_grad`)

The spec's "AD through the solver" idea, resolved the right way: the implicit
function theorem differentiates the fixed point C_BS(k, v(k,c)) = c directly,
giving exact first-order sensitivities for one extra Φ/φ evaluation after the
solve — no dual numbers through the polish loop:

    dv/dc = 1/vega = 1/φ(d1)          dv/dk = e^k·Φ(d2)/φ(d1)

Validated against central finite differences of the solver itself to ~3e-9
(the FD truncation floor; the IFT values are exact to solver precision). The
fixed point is differentiable wherever vega > 0, i.e. the whole solver domain.

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
from 33% (dense worst, at v≈0.76, delta≈0.08) to 13.3%. Follow-up experiments
sharpened where that worst error comes from and what does/doesn't fix it:

- **Higher-order surrogates fail outright.** P9 (one or two Newton steps)
  explodes to δ ~ 10³: the odd Taylor polynomial of Φ diverges outside its
  radius, and degree 9 diverges harder than 7 — sharper inside, wilder outside.
  A second Newton step on P7 is also *worse* (δ 0.34 → 0.83): the avg(P3,P7)
  band works by truncation-error *cancellation* (P3 undershoots, P7 overshoots),
  which iterating toward either surrogate's own root destroys.
- **The real defect is dispatch, not the seeds.** At the dense worst point
  (κ≈1.29, cstar≈0.021), the P1 seed (δ=0.023) and Mills seed (δ=0.047) are both
  good — the polynomial *Newton step* is what ruins it, because |d₂(v₁)| ≈ 2.2
  is far outside the surrogate's truncation tolerance. Conversely at
  (κ≈0.502, cstar≈0.0208) the tail filter (`κ>0.5 ∧ c*<0.02128`) misfires onto
  the Mills seed (δ=0.24) when P7 would be near-exact (δ=0.006).
- **A d₂-validity dispatch improves the worst case ~30% but hits a floor.**
  Routing on |d₂(v₁)| > 1.8 (the truncation variable itself, rather than the
  (κ, c*) proxies) gives dense worst δ = **0.231** — i.e. fixed-2 dense error
  ~6.6e-11, a ~350× accuracy gain at equal speed — but three independent
  refinements all converge to the same ~0.23 floor: in the low-delta low-vol
  corners *no* closed-form candidate (P1/P3/P7/avg/Mills/ATM) is better than
  ~0.23. Passing δ* = 0.133 there needs new mathematics (e.g. Schadner's
  variance-space coordinates), not dispatch tuning.

The d₂-dispatch is now **landed** (threshold tuned to 1.75, giving dense worst
δ = 0.228 and fixed-2 dense error 4.7e-11); see the closing section for the
landing measurements and a variance-coordinate attempt at the ~0.23 corner floor.

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

## SIMD gamma via fixed-length Temme CDF (exp/gamma-simd)

The scalar gamma solver's residual is the regularized incomplete gamma `P(a,x)`,
which is intrinsically iterative (series below `x≈a`, continued fraction above,
each with a data-dependent term count). That kills lane-uniform SIMD: lanes in a
vector bundle would need different iteration counts. This experiment replaces the
CDF with Temme's uniform asymptotic expansion (DLMF 8.12) truncated at FIXED
order, so the whole residual is straight-line and vectorizes.

### Method

With `λ = x/a` and `½η² = λ − 1 − ln λ`, `sign(η) = sign(λ−1)`,

```
P(a,x) = Φ(η√a) − e^{−½aη²}/√(2πa) · Σ_{k=0}^{6} c_k(η)/aᵏ
```

The `erfc` term collapses to `Φ(η√a)` sharing the single `e^{−½aη²}` with the
series prefactor (one `vexp`), evaluated by the existing branch-free blended-Cody
`phi_withg_bf`. Each coefficient function `c_k(η)` is a fixed Taylor polynomial
whose removable singularity at `η=0` (`λ=1`) is baked in, so no branch is needed
near the median. The `c_k(η)` were generated to high order in BigFloat (300 bits)
by series reversion of `η(μ)` plus the DLMF 8.12.10 recurrence
`c_k = η⁻¹ c_{k−1}′ + (−1)ᵏ g_k/μ` (Stirling `g_k`); the pole cancels to ~1e-90.
Graded Taylor degrees `(24,22,20,16,14,12,10)` (higher `c_k` are `aᵏ`-suppressed
so need fewer terms). `½η²` uses an accurate branch-free `log1p` (atanh series near
`μ=0`, an 11-term reduced log in the tails; the package's seed-grade `vlog` is only
~5e-13 and is insufficient here). The log-space HH-4 polish is unchanged from
`gamma_log.jl` (`F''/F' = a−x`, `F'''/F' = (a−x)²−x`), and `F' = x·pdf` is recovered
from the same `g` via a per-shape constant (`F' = g·aᵃe⁻ᵃ/Γ(a)`), so a residual +
its derivative cost one extra `vexp` beyond `x = eʸ`. Seed: lane-uniform
Cornish-Fisher (ODE5) blended with Wilson-Hilferty. `a < a_min` delegates to the
scalar amortized `gamma_quantile_batch!`.

### CDF accuracy and a_min

`max |P_temme(a,x) − P_exact(a,x)|` over a dense sweep of `x ∈ [q(1e-8), q(1−1e-8)]`,
Temme evaluated in BigFloat, reference a BigFloat confluent series (NOT
`SpecialFunctions.gamma_inc`; see below). With `K=6` the error is set by the
`a`-series truncation near the center and drops ~1 order per unit `a`:

| a | 10 | 12 | 15 | 18 | 20 | 50 | 100 | 500 |
|---|----|----|----|----|----|----|-----|-----|
| max\|ΔP\| | 3.5e-12 | 9.3e-13 | 1.8e-13 | 4.8e-14 | 2.2e-14 | 2.4e-17 | 1.3e-19 | 7.8e-25 |

The CDF crosses `1e-13` at **a ≈ 18**; we set **`a_min = 20`** (error 2.2e-14,
~4.5× headroom, and the operating point where the graded tail degrees also carry
the end-to-end quantile metric below). `a<20` cannot reach 1e-13 at K=6 and falls
back to the scalar batch. (Pushing `a_min` toward the task's hoped-for ~10 needs
K=8, i.e. two more degree-16 polynomials per residual, not worth the cost.)

### End-to-end quantile accuracy

`max |Δ ln x|` of the Float64 SIMD kernel vs a BigFloat quantile reference, N HH-4
steps. Two ranges are reported because the metric behaves differently in the deep
tails: for `u→0/1`, `δ(ln x) = δP/(x·pdf)` and `x·pdf → 0`, so the accuracy is
capped by the Float64 representation of `u` itself (`≈1e-16/(x·pdf)`), a limit that
binds ANY solver taking a Float64 `u`, not a defect of the expansion.

| a | bulk `u∈[1e-5,1−1e-5]` | full `u∈[1e-7,1−1e-7]` |
|---|------------------------|-------------------------|
| 20  | 1.6e-14 | 1.2e-12 |
| 30  | 5.8e-15 | 3.4e-12 |
| 50  | 7.1e-15 | 3.7e-12 |
| 100 | 3.6e-15 | 6.6e-12 |
| 500 | 1.8e-15 | 5.4e-12 |

`N=2, 3, 4` are bit-identical: the CF/WH seed + 2 HH-4 steps already reach the
Temme fixed point (quartic convergence), so N=2 suffices for `a≥20`. In the bulk
the kernel is `~1e-14`, roughly the K=6 CDF floor divided by the density.

### Accuracy vs the scalar/library path (an unexpected win)

`SpecialFunctions.gamma_inc` (the DiffEq port used by the scalar `gamma_quantile_log`
and by `Distributions.quantile`) has a **localized ~2e-6 accuracy defect** in the
transition region `x≈a` for large `a` (verified: `gamma_inc(50, 49.9554)` is off by
`−2.09e-6` vs a BigFloat confluent series, while `gamma_inc(50, 50.0)` is exact to
3e-17). Consequently the scalar solver's true error vs ground truth is:

| a | 20 | 30 | 50 | 100 | 500 |
|---|----|----|----|-----|-----|
| scalar `gamma_quantile_log`, max fwd \|Δln x\| | 1.2e-12 | 2.0e-6 | 7.4e-7 | 2.0e-7 | 1.1e-8 |
| Temme SIMD kernel | 1.6e-14 | 5.8e-15 | 7.1e-15 | 3.6e-15 | 1.8e-15 |

The existing test suite does not catch the scalar defect because it compares the
solver against `Distributions.quantile`, which shares the same `gamma_inc` bias
(it cancels). Measured against a BigFloat reference, **the fixed-length Temme kernel
is 8 to 9 orders of magnitude more accurate than the scalar/library path for `a≥30`**.
This alone makes the kernel worth landing, independent of any SIMD speedup.

Because of that defect, the scalar `gamma_quantile_log` / `_cert` /
`gamma_quantile_batch!` route through this Temme CDF for `a >= 20`, so the
default path is unbiased near `x = a`. Amortized this costs ~262 ns/q flat vs
~207 ns for the old `gamma_inc` solve at `a >= 50` (about 25% slower, spent on
the spike fix plus branch-free uniformity) and is ~15% faster at `a = 20` where
`gamma_inc`'s series near `x = a` is expensive. The scalar speed is immaterial
for one-off calls; bulk workloads should use the SIMD batch (~60 ns/q, same
accuracy). This makes the accuracy correct-by-default while `gamma_quantile_fast`
(~6 ns/q, ~1e-5) and `exp(solve(GammaLogQ(a), u))` (the `gamma_inc` path) remain
explicit opt-ins.

### Timing

`bench/bench_gamma_simd.jl`, ns per quantile, 16384-point `u` grid, single thread,
Apple M4 (aarch64). NEON is 2-wide, so `W=4/8` are emulated, yet they are FASTER,
because the wider `Vec` lets the compiler unroll and hide the long dependency chain
of the residual (2 vexp + phi + ~118-term polynomial evaluation) across more
in-flight lanes. `N=2` and `N=3` are bit-identical in output, so `N=2` is the
operating point.

| a | scalar log | Distributions | W2 N2 / N3 | W4 N2 / N3 | W8 N2 / N3 |
|---|-----------|---------------|------------|------------|------------|
| 20  | 186.7 | 380.4 | 113.9 / 157.1 | 88.2 / 121.0 | 59.8 / 84.6 |
| 50  | 153.6 | 269.5 | 115.2 / 159.8 | 86.4 / 122.9 | 59.9 / 83.3 |
| 100 | 150.5 | 264.1 | 115.4 / 162.5 | 86.7 / 124.0 | 60.0 / 89.0 |
| 500 | 156.8 | 267.5 | 113.7 / 159.0 | 85.9 / 120.7 | 59.3 / 84.4 |

At the `N=2` operating point: native `W=2` (~114 ns/q) is **~1.3× faster** than the
scalar log solver and ~2.4× faster than `Distributions`; the compiler's wide-unroll
`W=8` (~60 ns/q) is **~2.5× faster** than scalar and **~4.4× faster** than
`Distributions`. Timing is flat across `a` (fully branch-free), unlike the scalar
solver whose regime map costs more at `a=20`.

### Verdict

**Land-worthy.** The fixed-length Temme CDF makes the gamma quantile branch-free and
it wins on both axes for `a ≥ a_min = 20`:

- **Speed:** 1.3× (native 2-wide) to 2.5× (wide-unroll) over the scalar log solver,
  4.4× over `Distributions`, at N=2.
- **Accuracy:** `~1e-14` in the bulk, and 8 to 9 orders more accurate than the scalar /
  `Distributions` path for `a ≥ 30`, which inherit `gamma_inc`'s ~2e-6 defect near
  `x≈a`.

Cost: validity is bounded below by `a_min = 20` (K=6 CDF floor); smaller `a`
delegates to the scalar batch. Deep-tail `|Δln x|` for `u` within `~1e-7` of 0/1 is
capped near `1e-12` by the Float64 representation of `u`, which limits every solver
equally and is not specific to Temme.

Surprises: (1) `SpecialFunctions.gamma_inc` is only ~2e-6 accurate in a localized band
near `x=a` for large `a`, which the scalar solver and `Distributions.quantile` both inherit
it, and the package's own tests miss it by comparing the two biased results to each
other; (2) emulated wide `Vec` beats native 2-wide NEON here (latency hiding, not
throughput); (3) N=2 already reaches the Temme fixed point (quartic HH-4 from a
Cornish-Fisher seed); (4) the quantile `|Δln x|` metric is tail-sensitive via the
`1/(x·pdf)` factor, forcing higher `c_0`/`c_1` Taylor degrees than the center-dominated
CDF-error criterion alone would suggest.

### Reproduce

```
julia --project=. test/runtests.jl                    # incl. the Temme testset
julia --project=. bench/bench_gamma_simd.jl            # timing table above
```

## d2-dispatch landing and variance-coordinate corner seeds (exp/bs-corner-seed)

Two stages: (1) land the d2-validity seed dispatch measured earlier but left
unlanded (a known win), (2) try to break the ~0.23 seed-error floor in the
low-vol corner with variance (`w = v²`) coordinates. Numbers are on the dense
`(delta, vol)` sweep of the earlier sections (delta ∈ [0.05, 0.95],
vol ∈ [0.01, 2.0]), measured here on a 324k-point grid (600 × 540) with the
worst points cross-checked on 144k / 233k / 454k grids; Apple M4, Julia 1.12.

### Stage 1 — d2-validity dispatch (landed)

`bs_seed` (and its branch-free twin `bs_seed_bf`) previously routed the deep tail
with the `(κ, c*)` proxy filter `c* < 0.02128 ∧ κ > 0.5`. That filter *just
misses* the dense worst point (κ ≈ 1.33, c* ≈ 0.0213, v ≈ 0.75): c* sits a hair
above 0.02128, so the point stays on the P3 polynomial-CDF Newton step, whose
argument |d2(v1)| ≈ 2.2 is far outside the odd-Taylor truncation radius — giving
seed δ ≈ 0.34 and fixed-2 dense error 2.7e-8. The fix dispatches on the
truncation variable itself: compute `d2` at the P1 seed, `d2(v1) = -κ/v1 - v1/2`,
and route to the Mills seed whenever `|d2(v1)| > D2_VALID` (replacing the proxy
filter entirely). This is a single lane-friendly compare, so `bs_seed_bf` mirrors
it branch-free (`(abs(d2) <= D2_VALID) & (κ <= Κ4)` select).

Threshold tuning on the 324k grid (worst seed δ over the whole sweep; corner δ
restricted to vol < 0.4; fixed-2 dense max |Δv|):

| D2_VALID | worst δ | corner δ | fixed-2 dense |
|---------:|--------:|---------:|--------------:|
| 1.70     | 0.243   | 0.236    | 9.1e-12       |
| 1.72     | 0.238   | 0.211    | 1.8e-11       |
| 1.74     | 0.232   | 0.213    | 3.5e-11       |
| **1.75** | **0.228** | **0.216** | **4.7e-11** |
| 1.76     | 0.226   | 0.219    | 6.1e-11       |
| 1.78     | 0.225   | 0.225    | 1.0e-10       |
| 1.80     | 0.231   | 0.231    | 1.7e-10       |

The valley is flat and stable across grid resolutions (144k/233k/454k agree to
±0.001 in δ and within ~10% in fixed-2). 1.80 reproduces the earlier report's
δ = 0.231 exactly, but its fixed-2 (1.7e-10) sits *above* the 1e-10 target — the
earlier "6.6e-11" was the nominal `δ¹⁶`; the measured error runs ~2.5× higher
from the K4 constant at the low-vol worst point. **D2_VALID = 1.75 is landed**:
it minimizes worst δ near its floor while keeping fixed-2 comfortably under 1e-10.

Landed vs baseline (whole sweep):

| metric                         | baseline | d2-dispatch (1.75) |
|--------------------------------|---------:|-------------------:|
| dense worst seed δ             | 0.344    | **0.228**          |
| corner (vol < 0.4) worst δ     | 0.241    | **0.216**          |
| fixed-2 dense max \|Δv\|       | 2.7e-8   | **4.7e-11** (~570×)|
| fixed-3 dense max \|Δv\|       | 1.4e-15  | 1.7e-15            |
| adaptive dense max \|Δv\|      | 1.7e-15  | 1.7e-15            |

Soundness checks: the adaptive solver still converges everywhere it did before
(dense max |Δv| 1.7e-15; deep-OTM delta ∈ [0.001, 0.999] max |Δv| 2.6e-14, so
removing the proxy filter did not shrink any convergence basin); the SIMD batch
stays bit-consistent with the scalar fixed kernel (max diff 1.8e-15 over all
N ∈ {2,3}, W ∈ {2,4,8}, both seed strategies); all existing testsets pass, and a
new `d2-validity seed dispatch` testset bounds the dense-subsample fixed-2 and
seed δ.

Timing (best-of-N, 1.64M-IV tiled grid, Apple M4). The dispatch routes more
points to the two-`norminv` Mills seed (paper grid 28.7% → 43.0%; dense
23.9% → 33.5%), which costs the scalar seed ~2 ns:

| kernel                | baseline | d2-dispatch | Δ      |
|-----------------------|---------:|------------:|--------|
| scalar fixed-2        | ~53.9 ns | ~55.9 ns    | +3.7%  |
| scalar fixed-3        | ~81.4 ns | ~83.4 ns    | +2.5%  |
| SIMD W=8 fused f3     | 53.9 ns  | 53.3 ns     | ~0%    |
| SIMD W=8 two-pass f3  | 45.6 ns  | 46.3 ns     | +1.5%  |

The scalar fixed kernels take a ~3% hit (the extra Mills routing); the SIMD
batch — the production fast path — is essentially neutral, because the fused
kernel already evaluates every regime candidate (including the two `norminv_bf`)
on every lane, so shifting the *selection* toward Mills changes nothing. This is
the honest cost of the 570× fast-mode accuracy gain: it is not free on the
scalar path, but it is free where it is used at scale.

### Stage 2 — variance coordinates and the corner floor (negative, with a map)

Goal: break the seed-δ floor so `fixed-2` becomes dense-exact (δ ≤ δ* = 0.133).
The result is negative, but the investigation relocated the floor and corrected
the earlier "no candidate beats ~0.23 in the corner" claim.

**Where the floor actually is.** The dense worst *seed* δ (0.228) sits at high
vol (κ ≈ 1.21, v = 2.0), but that point has a tiny K4 constant, so it barely
contributes to `fixed-2` error. The `fixed-2` *error* is instead pinned by a
low-vol point (D = 0.05, v = 0.268, κ = 0.478, c* = 0.0051), and there the
dispatch picks P7 (δ = 0.216) even though **P1 (δ = 0.112) and Mills (δ = 0.132)
both pass δ***. So the corner is dispatch-limited, not floor-limited: an oracle
that picks the best of {P1, P3, P7, avg, Mills, ATM} per point reaches corner
(vol < 0.4) worst δ = **0.116**, comfortably under δ*.

**The true residual floor is at high vol.** The same oracle, taken over the
whole sweep, still cannot beat δ = **0.189** at (κ ≈ 1.38, v = 2.0, c* ≈ 0.44) —
there *no* candidate is better. Feeding the oracle seed into `fixed-2` gives a
dense error of **4.8e-14**: even perfect per-point dispatch does not make
`fixed-2` dense-exact, because 0.189¹⁶·K4 lands at ~5e-14. That 4.8e-14 is the
hard ceiling for this candidate set; the current landed 4.7e-11 is ~1000× above
it, and the entire gap is dispatch entanglement, not seed quality.

**Variance-coordinate candidates (w = v²), replacing the Mills seed in the
routed band unless noted:**

| candidate                              | worst δ | corner δ | fixed-2 dense |
|----------------------------------------|--------:|---------:|--------------:|
| Mills (current landed)                 | 0.228   | 0.216    | 4.7e-11       |
| cf_w1: κ²/(2w) = −ln c*                 | 0.538   | 0.494    | 2.9e-3        |
| cf_w2: κ²/(2w) = −ln c* + κ/2           | 0.592   | 0.511    | 5.3e-3        |
| cf_w3: fixed-point incl. −w/8 + prefactor | 0.979 | 0.352    | 1.8e-2        |
| log-price Newton in w (one price eval) | 0.216   | 0.216    | 4.7e-11       |
| oracle (best per point, cheating)      | 0.189   | 0.116    | 4.8e-14       |

The closed-form variance seeds derive from the Mills asymptotic
`ln c* ≈ −κ²/(2w) + κ/2 − w/8 − ln(√(2π)(κ²/w − w/4)/v)` (using the exact
identity `e^κ φ(d2) = φ(d1)`), which is nearly linear in `1/w` deep in the tail.
But the `fixed-2`-binding corner is only *moderately* OTM (c* ≈ 5e-3, not tiny),
where the neglected prefactor and `1/d²` Mills terms dominate; the leading-order
inversions are off by 50–100%. The log-price Newton step — the recipe meant to
linearize the deep-tail exponential — does converge nicely where c is genuinely
tiny, but the binding points are P7-routed, not tail-routed, so it moves worst δ
only 0.228 → 0.216 and leaves `fixed-2` unchanged; and since a seed that spends a
price evaluation costs ~one HH-4 step, seed + `fixed-2` ≈ `fixed-3` cost, and
`fixed-3` is already machine-exact everywhere. So the w-Newton has no practical
lever.

**Why the dispatch gain is not cheaply reachable.** The oracle's 1000× headroom
needs a per-point selector between P1, P7 and Mills. But the candidate-optimal
regions are entangled in exactly the coordinates a cheap seed can test: routing
the poly band to the (free) P1 seed on `|d2(v1)| ∈ (D2_LO, 1.75]` blows `fixed-2`
up to 3e-5, because a high-vol high-κ point (κ ≈ 1.10, v ≈ 1.55) lands in the
same `|d2(v1)|` band with P1 δ = 0.47; adding a low-vol gate `v1 < V_LO` still
leaks P1 into a mid-vol strip (κ ≈ 0.5–0.7, v ≈ 0.5–0.9) and degrades `fixed-2`
to 1e-8…1e-6. No single- or two-parameter rule over (κ, c*, v1, |d2(v1)|)
separates "P1-good corner" from "P1-bad elsewhere" — consistent with the earlier
finding that the arXiv regime map is a genuine local optimum. Selecting on the
actual price residual would work but costs one price evaluation per candidate,
i.e. more than `fixed-3`.

**Verdict.** Stage 2 is negative: variance coordinates do not yield a cheaper
`fixed-2`-exact seed. The refined understanding is the deliverable — the corner
is dispatch-limited (oracle δ = 0.116, not a 0.23 floor), the genuine seed floor
is δ = 0.189 at high vol (a hard 4.8e-14 `fixed-2` ceiling for this candidate
set), and no cheap selector realizes the gap. Breaking it needs either a new
high-vol seed family (to lower the 0.189 floor) or a cost-free residual proxy for
per-point selection — neither delivered by the w-coordinate recipes tried here.
The landed Stage-1 dispatch (4.7e-11, ~570× over baseline) stands as the win.

## Beta ODE5 central seed and y6 zero-eval certificate (exp/beta-ode5-y6)

Port of A. Hekimoglu's ODE5 central seed and its seed-intrinsic y6 term for beta,
wired into the existing logit solver. In the high-concentration central region
the quantile is written in standardized coordinates, X = p + eps*sqrt(pq)*Y with
p = a/n, n = a+b, eps = n^(-1/2), and Y is expanded against z = Phi^(-1)(u):
Y = z + eps*y1(z) + ... + eps^5*y5(z). The coefficient polynomials y_k depend
only on p and are built once per shape by solving the quantile ODE order by order
from the beta log-density expansion (y_k has degree k+1; the omitted eps^6*y6
term is the leading seed-error estimate). The build is an isbits-tuple port of his
BPoly series algebra (max degree 8, so a 10-slot polynomial), which makes
`BetaODE5(a,b)` allocation-free. New API: `BetaODE5`, `beta_ode5_seed`,
`beta_ode5_seed_batch!`, `beta_quantile_ode5`, `beta_quantile_ode5_batch!`,
`beta_quantile_ode5_mode`.

### The zero-eval seed is asymptotic, not machine-precision

The premise to test was whether the raw seed plus the seed-intrinsic y6 term can
return a central quantile at 1e-14 with ZERO incomplete-beta evaluations. Measured
over the gate (n >= 4, p in [0.05, 0.95], |z| <= 2.5) the raw seed is an
asymptotic-in-n object, exact only as n -> Inf:

| shape | n | raw-seed max \|D logit x\| |
|-------|----:|--------------------------:|
| (3,1.5)   | 4.5 | 1.1e-01 |
| (2,5)     | 7   | 2.3e-02 |
| (4,4)     | 8   | 4.1e-03 |
| (8,3)     | 11  | 3.6e-03 |
| (20,12.5) | 32.5| 2.3e-05 |
| (50,50)   | 100 | 3.2e-07 |
| (100,100) | 200 | 2.7e-08 |

A y6 seed certificate e0 = |c*eps^7*y6/(x0(1-x0))| <= tau therefore fires
essentially nowhere (only at isolated z where y6(z)=0): reaching e0 <= 1e-14
across |z| <= 2.5 needs n > ~6000. And his shipped acceptance test K4*e0^4 <= tau
bounds the error AFTER one HH-4 update, so returning the raw seed under it
"certifies" points whose seed error is up to ~5e-3 (the reference
`beta_quant_ode5_else_hh4` direct batch is his Monte-Carlo tier, not a 1e-14
quantile). A zero-eval central quantile at 1e-14 is not attainable from any
truncation of this expansion at practical n. We keep the raw seed as an explicit
approximate tier (`beta_ode5_seed`, `beta_ode5_seed_batch!`) for Monte-Carlo and
calibration, and report its cost below.

### Where the sharp seed pays off: certificate coverage

The ODE5 seed is far sharper than the CF5 seed, so the existing real-residual K4
certificate (one CDF evaluation gives the true residual r = f/F', then
16*K4*r^4 <= tau accepts a single logit-HH-4 update) fires over a much larger
central band. `beta_quantile_ode5` uses that fast path where it fires and falls
back bit-for-bit to `beta_quantile_batch!(certified=true)` on every other point
(measured deviation on uncertified points: exactly 0.0). Coverage of the central
gate, certified accuracy vs Distributions.jl, and certified-vs-full deviation
(safety 16):

| shape | n | ODE5 cov | CF5 cov | cert err (logit) | cert vs full |
|-------|----:|-------:|-------:|-----------------:|-------------:|
| (3,1.5)   | 4.5 |  24% |   0.4% | 8.9e-15 | 2.0e-14 |
| (2,5)     | 7   |  59% |   0.8% | 1.4e-15 | 2.0e-14 |
| (4,4)     | 8   |  85% |   2.1% | 1.7e-15 | 1.9e-14 |
| (8,3)     | 11  |  78% |   2.1% | 3.1e-15 | 2.1e-14 |
| (20,12.5) | 32.5| 100% |  22.8% | 2.7e-15 | 2.1e-14 |
| (50,50)   | 100 | 100% |  95.7% | 1.4e-15 | 3.8e-15 |
| (100,100) | 200 | 100% | 100.0% | 1.1e-15 | 1.4e-15 |

Certified points match Distributions.jl to <= 2.9e-15 in logit and the full
solver to <= 2.1e-14 (within tol) for every shape, including n = 4.5 where the
analytic e0 model alone would be optimistic (it predicts ~1e-10 there); using the
true residual r for the acceptance test is what keeps it sound.

### Benchmarks (Apple M4, -O3, amortized per-shape batches, ns/q)

Central band u in [0.15, 0.85] (its |z| <= 1.04 is fully inside the seed gate):

| shape | ODE5 cov | rawseed (0-eval) | ode5 (1e-14) | cert (old) | full (old) | Distributions |
|-------|-------:|-----:|-----:|----------:|----------:|-----:|
| (2,5)     |  78% | 17.7 | 211 | 347 | 357 | 502 |
| (4,4)     | 100% | 15.3 | 139 | 342 | 372 | 511 |
| (8,3)     | 100% | 17.4 | 144 | 369 | 393 | 628 |
| (20,12.5) | 100% | 17.7 | 191 | 280 | 352 | 831 |
| (50,50)   | 100% | 15.1 | 159 | 149 | 284 | 653 |
| (100,100) | 100% | 15.0 | 178 | 180 | 340 | 733 |

Full band u in [0.001, 0.999]:

| shape | ODE5 cov | ode5 | cert (old) | full (old) | Distributions |
|-------|-------:|-----:|----------:|----------:|-----:|
| (2,5)     |  58% | 245 | 327 | 354 | 491 |
| (4,4)     |  85% | 179 | 327 | 373 | 492 |
| (8,3)     |  77% | 214 | 356 | 383 | 596 |
| (20,12.5) |  99% | 195 | 300 | 344 | 823 |
| (50,50)   |  99% | 173 | 149 | 295 | 565 |
| (100,100) |  99% | 187 | 180 | 330 | 700 |

Per-shape precompute is 14-28 us; break-even against the old certified path is
~150-250 quantiles, so this is a batch API (matching his batch protocol). Numbers
carry ~10% run-to-run variance on a loaded machine; the ODE5-vs-old ordering is
stable across runs.

- Zero-eval seed tier (`rawseed`): 15-18 ns/q, matching his ~18-22 ns central
  tier, at the asymptotic accuracy tabled above (not a 1e-14 quantile).
- Certified 1e-14 tier: 1.4-2.5x faster than the existing certified path at small
  and moderate n (139-211 vs 342-369 ns central at n <= 33), because ODE5 lifts
  coverage from CF5's few percent to ~100%. At large n it is neutral (50,50) to
  slightly slower (100,100): CF5 already certifies ~100% there and the ODE5 seed
  costs more to evaluate.

### Verdict

Land-worthy as the beta central accelerator, with the headline reframed honestly:

1. The certified `beta_quantile_ode5` path is a clear win where it matters (small
   and moderate n, the pocket the earlier RESULTS note flagged CF5 could not
   reach): 1.4-2.5x over the existing certified path at equal-or-better accuracy
   (certified err <= 2.9e-15), sound (bit-identical fallback), allocation-free.
   It is neutral at very large n, where CF5 already covers.
2. The "zero incomplete-beta evaluation at 1e-14" headline is NOT achievable: the
   seed is asymptotic in n, so any y6-only certificate that keeps 1e-14 has
   near-zero coverage. The zero-eval path is shipped as an explicit Monte-Carlo
   seed (15-18 ns/q, ~1e-3 to ~1e-8 logit accuracy), clearly separated from the
   machine-precision quantile.

Reproduce: `julia --project=. test/runtests.jl` (testset "Beta ODE5 central seed
+ y6 certificate"); `julia --project=. -O3 bench/bench_beta_ode5.jl`.

## Gamma large-a ODE5 seed and certified seed-only regions (exp/gamma-ode5)

Port of the "semianalytic" mode of A. Hekimoglu's combined engine
(`gamma_quant_final_combined_engine.c`, `gamma_quant_semianalytic_one`). The
full solver `gamma_quantile_log` spends almost all of its ~90-130 ns/q inside
the incomplete-gamma CDF that `hh_terms` evaluates. This mode asks a different
question: in which parts of the (a, u) plane is a closed-form regime seed
already accurate enough to return with ZERO incomplete-gamma evaluations? That
is the unlock, because the seed is just `norminv` plus a polynomial.

`gamma_quantile_fast(a, u)` and the amortized `gamma_quantile_fast_batch!`
(new, `src/dists/gamma_fast.jl`) dispatch a region map:

- **seed-only** (0 CDF evals): a >= 10 central band (|z| <= 2.5) returns the
  CF5/ODE5 seed `_gamma_seed_cf5`; a < 2 lower tail returns the order-3 series
  seed `_gamma_seed_lower`.
- **one-update** (1 CDF eval): 2 <= a < 10 central/upper and a >= 10 moderate
  tail get the seed plus one log-HH4 step (bit-identical to the exact solver's
  first iterate); 2 <= a < 10 lower tail (u < 0.08) gets series seed + one step.
- **two-update** (2 CDF evals): a < 2 mid-u transition, series seed + two
  log-Newton steps.
- **fallback**: deep upper (q <= 1e-5) hands back to `exp(solve(GammaLogQ))`,
  bit-for-bit the exact path (verified: max abs diff 0.0 over the fallback
  region across a in {2,5,10,25,50,100}).

Two deliberate deviations from the C, both accuracy-improving: the lower seed is
the repo's order-3 reversion (C uses order 2), and the deep-upper is the exact
solver rather than C's approximate Mills/log-survival Newton.

### Accuracy (measured, max |Δ ln x| vs `Distributions.quantile`)

Seed-only central band (|z| <= 2.5), the headline region:

| a    | 10     | 25     | 50     | 100    | 250    | 500    | 1000   | 5000    |
|------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|--------:|
| err  | 1.2e-5 | 2.8e-6 | 7.4e-7 | 2.0e-7 | 3.7e-8 | 1.1e-8 | 3.7e-9 | 4.4e-10 |

This is NOT machine precision, and it does not become machine precision at any
practical shape. The error is the truncation of the 5th-order Cornish-Fisher
seed and scales as a clean power law:

```
err_central(a) ~= 1.04e-3 * a^-1.85       (fit over a in [25, 500])
```

so the smallest shape for which the seed-only central region is genuinely
certified below a tolerance tau is:

| tau        | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-13   |
|------------|-----:|-----:|-----:|------:|--------:|
| min a      |  3.5 |   43 |  514 |  6190 | 259000  |

A certified-to-1e-13 seed-only gamma quantile therefore requires a ~ 2.6e5:
empty for any realistic shape. Seed-only is a pricing/QMC-tolerance mode, not a
machine-precision unlock, exactly as the reference engine's own header states.

One-update region: machine precision (~1e-14) for a >= 25; for small shapes a
single step from a poor extreme-tail seed leaves ~2.6e-6 (a=2), ~2.3e-7 (a=5),
~6.6e-7 (a=10) at the tail edges (u -> 1e-6 or the q = 1e-5 fallback boundary).

### Coverage

For a >= 10 the seed-only central band is 98.8% of u in [1e-8, 1-1e-8] (the
|z| <= 2.5 interval u in [0.0062, 0.9938]); the remaining 1.2% is one-update.
For a in [2, 10) there is no seed-only region (all one-update); for a < 2 only
the lower tail (u < 0.15 at 0.5 < a < 2) is seed-only. Over a 10-shape by dense-u
sweep the overall seed-only fraction is 0.79.

### Benchmark (a=50, 16384 u-points, min time / N, single thread, -O3)

| kernel                                   | ns/q  |
|------------------------------------------|------:|
| norminv only (floor)                     |   1.3 |
| CF5 seed = norminv + polynomial          |   5.7 |
| **fast seed-only region (a=50)**         |   **6.2** |
| fast one-update region (a=5)             |  54.5 |
| fast mixed full range (a=50)             |   7.2 |
| full certified solve, central (a=50)     |  86.8 |
| full certified solve, full range (a=50)  |  89.0 |
| full solve (uncertified), central (a=50) | 149.1 |

Seed-only lands at 6.2 ns/q, within 9% of the bare `norminv` + polynomial floor
and **14x faster** than the full certified path (86.8 ns/q). Because the central
band is 98.8% of u, the mixed full-range fast batch is 7.2 ns/q, **12x** the
full path, with the residual cost coming from the ~1.2% one-update tail points at
~54 ns each. The batch is allocation-free (0 bytes). The one-update region alone
(54.5 ns/q) is dominated by its single incomplete-gamma call and only ~1.6x
faster than the full path, so it is not the win.

### Verdict

**Land-worthy as an explicitly-labeled fast/approximate mode; not a replacement
for `gamma_quantile_log`.** The seed-only large-a central region is a real 14x
speedup that reaches the cost of `norminv` plus a polynomial, and it is the
right tool wherever ~1e-6 to 1e-8 relative accuracy suffices (Monte Carlo / QMC
inverse-transform sampling, option pricing), which is exactly the tolerance
band where a >= 43 already certifies below 1e-6. The honest negative result is
the machine-precision framing: no seed-only region is certifiable to 1e-13 for
any practical shape (a ~ 2.6e5 required), and the one-update regions overlap the
existing `solve_certified` cost while sacrificing tail accuracy at small a, so
they add little over the certified full solver. The value is entirely in the
zero-CDF-eval seed-only path, gated to the shapes and tolerances where it is
provably good.

Reproduce: `julia --project=. -O3 bench/bench_gamma_ode5.jl` and the
`Gamma fast semi-analytic (ODE5 seed-only)` testset in `test/runtests.jl`.
