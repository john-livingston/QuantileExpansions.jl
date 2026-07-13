# API reference

Hand-maintained (Julia package; no autodoc). All functions are allocation-free
scalar kernels unless noted; batch functions require a preallocated `out`.

## Black–Scholes implied volatility

Inputs: log-moneyness `k = log(K/F)`, normalized undiscounted forward call
price `c`. Output: total volatility `v = σ√T`.

### `bs_implied_vol(k, c; tol=1e-14, maxiter=8) -> v`
Adaptive solver, residual test after each update: ~1.6e-15 dense worst-case.
NaN inputs propagate to NaN.

### `bs_implied_vol_generic(k, c; tol=1e-14) -> v`
Adaptive solver through the generic interface, residual test before each
update — one step faster, ~8.9e-14 dense worst-case.

### `bs_implied_vol_cert(k, c; tol=1e-14) -> v`
Certified variant: skips the confirmation evaluation when the K4 error model
proves the first update sufficient. Identical results; near-neutral speed on
typical grids.

### `bs_implied_vol_fixed(k, c, Val(N)=Val(3)) -> v`
Branch-free, exactly `N` HH-4 updates, **no convergence test**. Valid for
delta ∈ [0.05, 0.95], vol ∈ [0.01, 2.0]: `Val(3)` → ~1.6e-15, `Val(2)` →
~4.7e-11 there. Outside that band errors grow with no signal — use the adaptive
solvers. Results clamp to v ∈ [1e-10, 5].

### `bs_implied_vol_fixed_batch!(out, ks, cs, Val(N), Val(W); vector_seed, ws)`
SIMD batch (`W` lanes; 8 recommended). `vector_seed=true` (x86 default) runs
the fully-fused branch-free path; `false` (ARM default) a scalar-seed two-pass
path — pass `ws=BSFixedWorkspace(n)` to make repeat two-pass calls
allocation-free.

### `bs_implied_vol_fixed_batch_threaded!(out, ks, cs, Val(N), Val(W); ...)`
The batch kernel across all Julia threads (contiguous chunks). Bit-for-bit
identical to the serial batch.

### `bs_implied_vol_grad(k, c; tol=1e-14) -> (v, ∂v/∂c, ∂v/∂k)`
Implied vol plus exact first-order sensitivities via the implicit function
theorem.

### `bs_price(κ, v) -> c`
Forward map: normalized OTM call price at `κ = |k|`.

## Inverse Gaussian

### `ig_quantile(μ, λ, p; tol=1e-13) -> x`
Scalar IG quantile; forward residual ~1e-13.

### `ig_quantile_cert(μ, λ, p; tol=1e-13) -> x`
Certified variant (fires only in the near-Gaussian λ/μ ≳ 100 regime;
otherwise equivalent).

### `ig_quantile_batch!(out, μ, λ, ps, Val(N), Val(W))`
Fixed-shape SIMD batch; `Val(3)` reaches machine forward residual.

### `ig_quantile_batch_threaded!(out, μ, λ, ps, Val(N), Val(W))`
Threads × SIMD variant; bit-for-bit identical to serial.

## Gamma

Scale-1 quantiles; multiply by θ for `Gamma(a, θ)`.

### `gamma_quantile_log(a, u; tol=1e-14) -> x`
**Recommended.** For `a < 20`, the log-space regime-split solver (port of
Hekimoglu's engine): series / Cornish-Fisher / Mills seeds, density-scaled
stopping, ~1e-10 log-accuracy into the deep tails. For `a >= 20` it routes
through the fixed-order Temme CDF (~1e-14, and unbiased near `x = a` where
`SpecialFunctions.gamma_inc` carries a localized ~2e-6 defect). Exact closed
forms at `a=1`, `a=½`.

### `gamma_quantile_log_cert(a, u; tol=1e-14) -> x`
Certified variant — 1.43× faster in the central region at identical results.

### `gamma_quantile_batch!(out, a, us; tol=1e-14, certified=true)`
Fixed-shape batch with the per-shape constructor work amortized. For `a >= 20`
it uses the same Temme path as the scalar functions (bit-identical).

### `gamma_quantile_batch_simd!(out, a, us, Val(N), Val(W))`
Fixed-shape SIMD batch for `a >= 20` (`GAMMA_SIMD_AMIN`): the iterative
incomplete-gamma CDF is replaced by a fixed-order Temme uniform asymptotic
expansion, so the whole solve is branch-free and vectorizes. `Val(2)` already
reaches the Temme fixed point (~1e-14 in the bulk); `W=8` recommended. About
60 ns/q, the same accuracy as the scalar path and faster. Smaller `a` delegates
to `gamma_quantile_batch!`.

### `gamma_quantile_batch_simd_threaded!(out, a, us, Val(N), Val(W))`
Threads × SIMD variant; bit-for-bit identical to the serial SIMD batch.

### `gamma_quantile_fast(a, u) -> x`
Fast semi-analytic mode (the reference engine's "semianalytic" tier): a certified
regime seed with zero incomplete-gamma evaluations in the large-a central and
lower-tail regions, one cheap update in moderate regions, and the exact
`gamma_quantile_log` result in the deep upper tail. About 6 ns/q. NOT machine
precision: max `|Δ ln x|` runs ~1e-5 (a=10) to ~1e-8 (a=500) in the seed-only
region. Use `gamma_quantile_log` when full accuracy is required.

### `gamma_quantile_fast_batch!(out, a, us)`
Amortized batch of `gamma_quantile_fast`; bit-identical to the scalar function.

### `gamma_quantile(a, p; tol=1e-13) -> x`
x-space baseline (Wilson–Hilferty seed). Kept for comparison; tails degrade
to ~1e-7 log-accuracy at large `a`.

## Beta

### `beta_quantile_logit(a, b, p; tol=1e-14) -> x`
**Recommended.** Logit-space solver (Hekimoglu's coordinate change + exact
endpoint branches over this package's regime seeds, CF5 central seed):
~2e-14 uniformly. Exact closed forms at `b=1`, `a=1`, `(½,½)`.

### `beta_quantile_logit_cert(a, b, p; tol=1e-14) -> x`
Certified variant.

### `beta_quantile_batch!(out, a, b, ps; tol=1e-14, certified=true)`
Fixed-shape batch; logbeta and CF5 cumulants computed once per (a,b).

### `beta_quantile_ode5(a, b, u; tol=1e-14, τ=1e-14, safety=16.0) -> x`
Beta(a,b) quantile using Hekimoglu's ODE5 central seed. In the
high-concentration central region the sharp seed lets the K4 certificate accept
a single logit-HH-4 update (one incomplete-beta evaluation); elsewhere, and on
any point the certificate declines, it is bit-for-bit `beta_quantile_logit`.
Certified points match the full solver to <= ~3e-15 in logit. Faster than
`beta_quantile_logit` at small/mid `n = a+b` (1.4 to 2.5x at `n <= 33`), the
pocket where the CF5 seed alone cannot certify.

### `beta_quantile_ode5_batch!(out, a, b, us; tol=1e-14, τ=1e-14, safety=16.0)`
Fixed-shape batch; the per-(a,b) ODE5 precompute is amortized. Bit-identical to
scalar `beta_quantile_ode5`.

### `beta_ode5_seed(a, b, u) -> x`
Raw ODE5 central seed, ZERO incomplete-beta evaluations (~15 to 18 ns/q).
Asymptotic in `n = a+b` (exact as `n -> Inf`): logit accuracy ~1e-3 at `n≈5`,
~1e-8 at `n≈200`. NOT machine precision; this is the fast Monte-Carlo /
calibration tier. Use `beta_quantile_ode5` for a certified 1e-14 quantile.
Outside the central gate it defers to `beta_quantile_logit`.

### `beta_ode5_seed_batch!(out, a, b, us)`
Amortized batch of `beta_ode5_seed`.

### `beta_quantile_ode5_mode(a, b, u) -> 0 or 1`
Diagnostic: 1 if `u` is on the certified ODE5 fast path for Beta(a,b), else 0.
Used to measure certified-region coverage.

### `beta_quantile(a, b, p; tol=1e-13) -> x`
x-space baseline; loses accuracy near endpoints for skewed shapes.

## Generic interface

### `QuantileProblem`
Abstract supertype. Implement `seed(D, p)` and
`hh_terms(D, x, p) -> (f, f′, f″/f′, f‴/f′)`; optionally `xlo`/`xhi` bounds,
`converged(D, f, fp, tol)` (density-scaled stopping), and
`hh4_c4(D, x) = f⁗/f′` + `has_c4` to enable certification.

### `solve(D, p; tol=1e-14, maxiter=8) -> x`
The universal safeguarded HH-4 loop (Newton and damped-bracket fallbacks).

### `solve_certified(D, p; tol=1e-14, τ=1e-14, safety=16.0, maxiter=8) -> x`
As `solve`, but exits without a confirmation evaluation when
`safety·K4·r⁴ ≤ τ` at the seed.

## Special functions (exported utilities)

`normcdf`, `normpdf`, `norminv` (Acklam, guarded), `normcdf_pdf` (Φ and φ from
one exp), `erfc_hi`, `erfcx_pos`, and the branch-free SIMD-generic variants
`vexp`, `vlog`.
