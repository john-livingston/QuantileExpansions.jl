# QuantileExpansions

A single, distribution-generic **regime-split quantile solver** in Julia. Every
target is an inverse-CDF problem `F(x) = p`, solved by a closed-form regime seed
plus one universal quartic polish step (**HH-4**, Householder order 3). The
generic `solve(D, p)` is written **once**; each distribution supplies a tiny
`(seed, hh_terms)` interface and the compiler monomorphizes it into specialized,
allocation-free, C-speed code.

Covers **Black–Scholes implied vol**, **Inverse Gaussian**, **Gamma**, and **Beta**.

## Results

Apple Silicon (10 performance cores), Julia 1.12, `-O3`. Identical grids and
methodology to the reference; speed vs `Distributions.quantile` (which wraps the
standard C/Rmath algorithms), and vs the paper authors' own C kernel for BS-IV
(clang `-O3 -march=native -ffast-math`, same machine).

| Target            | this repo    | reference            | speedup | accuracy¹         |
|-------------------|-------------:|----------------------|--------:|-------------------|
| Black–Scholes IV  |   ~69 ns/IV  | C kernel: ~110 ns/IV | **1.6×**| max \|Δv\| 8.9e-14 |
| BS-IV (threaded)  |  ~8.2 ns/IV  | Numba: 24 ns (12t)   | 2.9×    | —                 |
| BS-IV fast mode²  |   ~52 ns/IV  | C kernel: ~110 ns/IV | **2.1×**| max \|Δv\| 2.3e-8 |
| BS-IV fast (thr.) |  ~6.3 ns/IV  | Numba: 24 ns (12t)   | 3.8×    | —                 |

¹ Worst case over a **dense** sweep of delta ∈ [0.05, 0.95] × vol ∈ [0.01, 2.0]
(144k off-node points), not just the 328-node benchmark grid. The one-extra-step
`bs_implied_vol` variant reaches 1.6e-15 over the same sweep.
² Fast mode (`bs_implied_vol_fixed` with 2 fixed steps, no convergence test) is
**only valid inside that delta band** — outside it errors grow to percent level
with no error signal; use the adaptive solver for deep-OTM inputs. The 3-step
variant holds 1.6e-15 across the whole band at ~79 ns.
| Inverse Gaussian  |   ~76 ns/q   | Distributions: ~600  | **7.6×**| \|F(x)−p\| ~1e-13 |
| Gamma             |  ~185 ns/q   | Distributions: ~285  | 1.5×    | \|F(x)−p\| ~1e-13 |
| Beta              |  ~470 ns/q   | Distributions: ~820  | 1.8×    | \|F(x)−p\| ~1e-13 |

**The BS-IV solver beats the reference C on identical hardware**, with ~6× better
max error, single-threaded. See [`RESULTS.md`](RESULTS.md) for the full writeup.

## Quick start

```julia
include("src/QuantileExpansions.jl"); using .QuantileExpansions

bs_implied_vol(0.1, 0.06)              # Black–Scholes implied total vol from (log-moneyness, price)
bs_implied_vol_fixed(0.1, 0.06, Val(2))  # branch-free fast mode (~1e-8 in the 5–95 delta band)
ig_quantile(1.0, 3.0, 0.7)     # Inverse Gaussian quantile (μ, λ, p)
gamma_quantile(2.5, 0.4)       # Gamma quantile (shape a, p)
beta_quantile(2.0, 5.0, 0.4)   # Beta quantile (a, b, p)
```

Run the tests and the full benchmark table:

```
julia --project=. test/runtests.jl               # correctness (all four distributions)
julia --project=. -O3 -t auto bench/run_all.jl    # the results table above
```

## How it works

For residual `f = F(x) − p`, HH-4 needs only `f' = ρ(x)` and the log-density
ratios `f''/f' = L'(x)`, `f'''/f' = L''(x) + L'(x)²` where `L = log ρ`. `L'` is
**rational** for the normal / IG / gamma / beta densities, so the quartic step is
nearly free — and each seed lands within the polisher's admissibility threshold,
so convergence takes ~2–3 iterations. The regime boundaries come from CDF-
truncation error tolerances, not fitting. See [`METHOD.md`](METHOD.md) for the method
and derivations.

## Layout

```
src/
  QuantileExpansions.jl   module entry / exports
  core/solver.jl          generic HH-4 solve() + admissibility/stopping
  core/specialfuns.jl     Φ, Φ⁻¹, erfc, erfcx (own high-accuracy ports)
  dists/                  blackscholes, inverse_gaussian, gamma, beta
test/                     round-trip + vs-Distributions correctness
bench/                    per-distribution and consolidated benchmarks
ref/                      upstream C/Numba reference — fetched, not vendored
                          (see ref/README.md)
```

## Provenance

The BS-IV method and all seed/boundary derivations are from Hekimoglu & Gökgöz,
*A Fast Implied Volatility Method with Expansions* ([arXiv:2606.10245](https://arxiv.org/abs/2606.10245)).
The cross-distribution generalization — one parameterized solver spanning
BS / IG / gamma / beta — and this Julia implementation are original to this repo.

## License

[MIT](LICENSE).
