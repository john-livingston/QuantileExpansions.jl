# QuantileExpansions.jl

A single, distribution-generic **regime-split quantile solver** in Julia. Every
target is an inverse-CDF problem $F(x) = p$, solved by a closed-form regime
seed plus one universal quartic polish step (Householder order 3, "HH-4"). The
generic `solve(D, p)` is written once; each distribution supplies a tiny
interface and the compiler monomorphizes it into specialized, allocation-free,
C-beating code.

Covers **Black–Scholes implied volatility**, **Inverse Gaussian**, **Gamma**,
and **Beta** — with SIMD batch kernels, thread parallelism, error-certified
fast paths, and exact implied-vol sensitivities.

![BS-IV performance progression](assets/bs_progression-light.png#only-light)
![BS-IV performance progression](assets/bs_progression-dark.png#only-dark)

## Headline results

Measured on Apple Silicon (details and cross-ISA numbers in
[Benchmarks](benchmarks.md)):

| Target | this package | reference | speedup | accuracy¹ |
|---|---:|---|---:|---:|
| Black–Scholes IV | ~69 ns/IV | authors' C kernel: ~110 ns | **1.6×** | 8.9e-14 |
| BS-IV, SIMD batch (full precision) | ~45 ns/IV | — | 2.5× vs C | 1.6e-15 |
| BS-IV, threads × SIMD | **4.8 ns/IV** | — | ~208 M/s | 1.6e-15 |
| Inverse Gaussian | 76 ns/q | Distributions.jl: 575 ns | **7.6×** | 1e-13 |
| IG, threads × SIMD | **4.4 ns/q** | — | ~230 M/s | machine |
| Gamma (log-space) | 130 ns/q | Distributions.jl: 270 ns | 2.1× | ~1e-10 tails |
| Beta (logit-space) | 303 ns/q | Distributions.jl: 821 ns | 2.7× | ~2e-14 |

¹ worst case over dense parameter sweeps, not just benchmark grid nodes.

## Quick start

The package is not yet registered. Clone and use with the project environment:

```julia
using QuantileExpansions

bs_implied_vol(0.1, 0.06)          # implied total vol from (log-moneyness, price)
ig_quantile(1.0, 3.0, 0.7)         # Inverse Gaussian quantile (μ, λ, p)
gamma_quantile_log(2.5, 0.4)       # Gamma quantile (shape a, p)
beta_quantile_logit(2.0, 5.0, 0.4) # Beta quantile (a, b, p)
```

See the [Guide](usage.md) for batch, threaded, certified, and sensitivity APIs.

## Provenance

The BS-IV method and all seed/boundary derivations are from **Hekimoglu &
Gökgöz**, *A Fast Implied Volatility Method with Expansions*
([arXiv:2606.10245](https://arxiv.org/abs/2606.10245)). The cross-distribution
generalization is A. Hekimoglu's idea; the log-space gamma solver is a port of
his reference engine, and the logit-space beta solver adopts his coordinate
change and exact endpoint branches. The generic-dispatch Julia architecture,
the BS performance work (exp-reuse identity, branch-free fixed-step kernels,
SIMD batching), and the IG solver are original to this repository. MIT
licensed.
