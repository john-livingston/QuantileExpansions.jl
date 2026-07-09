# Reference implementation (not vendored)

The C and Python benchmarks compared against in [`../RESULTS.md`](../RESULTS.md)
come from the original authors:

> **A Fast Implied Volatility Method with Expansions**
> S. Hekimoglu & F. Gökgöz — arXiv:2606.10245
> Code: https://github.com/variancegamma/A-Fast-IV-method-with-expansions

Their repository carries **no license**, so the source is *not* redistributed
here. Instead, [`fetch.sh`](fetch.sh) downloads it at a pinned commit
(`e741041809dd`) into this directory:

```
bash ref/fetch.sh
```

This fetches:
- `iv_regime_final_v2.c` — the regime-split IV kernel (the C we benchmark against)
- `bench_iv_c_all_hh4.c` — its benchmark harness (same 328-point grid we use)
- `taylor_iv_efficiency_v6.py` — the Numba reference

Then, to reproduce the C number in `RESULTS.md`:

```
cc -O3 -march=native -ffast-math ref/bench_iv_c_all_hh4.c -o ref/bench_c && ./ref/bench_c
```

The Julia solver in `../src/` has **no dependency** on any of this — it is only
for reproducing the head-to-head comparison.
