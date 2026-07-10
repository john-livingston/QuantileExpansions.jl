# Guide

## Choosing a solver

| You want | Use |
|---|---|
| One implied vol, max accuracy | `bs_implied_vol(k, c)` — ~1.6e-15 |
| One implied vol, fastest adaptive | `bs_implied_vol_generic(k, c)` — ~9e-14 |
| Many implied vols | `bs_implied_vol_fixed_batch!` (+ `_threaded!` across cores) |
| Implied vol + sensitivities | `bs_implied_vol_grad(k, c)` |
| IG / gamma / beta quantiles | `ig_quantile`, `gamma_quantile_log`, `beta_quantile_logit` |
| Many quantiles at fixed shape | `ig_quantile_batch!`, `gamma_quantile_batch!`, `beta_quantile_batch!` |

## Black–Scholes implied volatility

Inputs are log-moneyness $k = \log(K/F)$ and the normalized, undiscounted
forward call price $c$; the result is total volatility $v = \sigma\sqrt{T}$.

```julia
v = bs_implied_vol(0.1, 0.06)              # adaptive, ~1.6e-15
v = bs_implied_vol_generic(0.1, 0.06)      # adaptive, stops one step earlier
v = bs_implied_vol_fixed(0.1, 0.06, Val(3))  # branch-free, exactly 3 updates
```

!!! warning "Fixed-step validity domain"
    `bs_implied_vol_fixed` runs a fixed number of updates with **no
    convergence test**. Its accuracy guarantees hold for delta ∈ [0.05, 0.95],
    vol ∈ [0.01, 2.0]: `Val(3)` reaches ~1.6e-15 across that whole region,
    `Val(2)` is a ~1e-8 fast mode. **Outside the delta band errors grow to
    percent level with no error signal** — use the adaptive `bs_implied_vol`
    for deep-OTM inputs.

### Batches: SIMD and threads

```julia
out = similar(ks)
bs_implied_vol_fixed_batch!(out, ks, cs, Val(3), Val(8))            # SIMD, W=8 lanes
bs_implied_vol_fixed_batch_threaded!(out, ks, cs, Val(3), Val(8))   # + all threads
```

Two seed strategies sit behind `vector_seed`: the fully-fused branch-free path
(default on x86, best with wide vectors) and a scalar-seed two-pass path
(default on ARM/NEON). Pass `ws = BSFixedWorkspace(length(ks))` to make
repeated two-pass calls allocation-free. Threaded and serial results are
bit-for-bit identical.

### Sensitivities

```julia
v, dvdc, dvdk = bs_implied_vol_grad(k, c)
```

Exact first-order sensitivities via the implicit function theorem:
$\partial v/\partial c = 1/\varphi(d_1)$ and
$\partial v/\partial k = e^k \Phi(d_2)/\varphi(d_1)$ — one extra $\Phi/\varphi$
evaluation after the solve, no AD required.

## Inverse Gaussian

```julia
x = ig_quantile(1.0, 3.0, 0.7)                          # (μ, λ, p)
ig_quantile_batch!(out, 1.0, 3.0, ps, Val(3), Val(8))   # fixed shape, SIMD
ig_quantile_batch_threaded!(out, 1.0, 3.0, ps, Val(3), Val(8))
```

The batch kernel reaches machine forward residual at `Val(3)` — there is no
accuracy trade.

## Gamma

Two solvers; **the log-space one is recommended**:

```julia
x = gamma_quantile_log(5.0, 0.3)        # port of Hekimoglu's engine, ~1e-10 tails
x = gamma_quantile(5.0, 0.3)            # x-space baseline (tails degrade ~1e-7)
x = gamma_quantile_log_cert(5.0, 0.3)   # certified: skips proven-redundant CDF evals
gamma_quantile_batch!(out, 5.0, us)     # fixed shape, constructor amortized
```

`a = 1` and `a = ½` take exact closed forms. Results are for scale 1;
multiply by θ for `Gamma(a, θ)`.

## Beta

```julia
x = beta_quantile_logit(2.0, 5.0, 0.4)        # recommended: logit-space
x = beta_quantile_logit_cert(2.0, 5.0, 0.4)   # certified variant
beta_quantile_batch!(out, 2.0, 5.0, ps)       # fixed shape, CF5 cumulants amortized
```

Exact closed forms for `b = 1`, `a = 1`, and `(½, ½)` (arcsine). The logit
solver holds ~2e-14 uniformly, including skewed shapes where x-space solvers
lose 6+ digits near the endpoints.

## The generic interface

Every solver above is an instance of one loop. A distribution implements:

```julia
struct MyDist <: QuantileProblem ... end
seed(D::MyDist, p)         # closed-form regime seed
hh_terms(D::MyDist, x, p)  # (f, f′, f″/f′, f‴/f′) — residual + rational ratios
```

and inherits `solve`, `solve_certified`, the safeguards, and the stopping
logic. See [Method](method.md) for why the ratios are cheap and the
[API reference](api.md) for the full surface.
