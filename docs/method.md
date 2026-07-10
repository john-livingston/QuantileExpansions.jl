# Method

Every target is an inverse-CDF problem: given a CDF $F$ and a probability (or
price) $p$, solve $F(x) = p$. The forward map is explicit; the inverse is not
elementary. The recipe (Hekimoglu & Gökgöz,
[arXiv:2606.10245](https://arxiv.org/abs/2606.10245)) has three reusable
ingredients.

## 1. Regime-split seeds

The domain is partitioned by the asymptotic behavior of $F$, and each regime
gets a **closed-form seed** derived from the local structure — series inversion
near the center, polynomial-CDF surrogates in the middle, Mills-ratio / tail
asymptotics in the tails. No iteration at the seed stage, and the regime
boundaries come from truncation-error tolerances, not fitting.

For Black–Scholes (log-moneyness $k$, total volatility $v = \sigma\sqrt{T}$):

$$c_{BS}(k, v) = \Phi(d_1) - e^k\,\Phi(d_2), \qquad
d_1 = -\tfrac{k}{v} + \tfrac{v}{2}, \quad d_2 = d_1 - v$$

the seeds are a 4th-order Taylor inversion at the money, one Newton step on
degree-3/7 polynomial surrogates of $\Phi$ in the mild-OTM band (averaged where
they bracket the truth), and a Mills-ratio-corrected quadratic in the tails.

## 2. One universal quartic polish

A single Householder step of order 3 ("HH-4") is applied by a solver written
once, generically:

$$x \leftarrow x + \frac{3r\,(2 - r\,\varphi_2)}{-6 + 6r\varphi_2 - r^2\xi},
\qquad r = \frac{f}{f'}$$

The step needs only the derivative *ratios* $\varphi_2 = f''/f'$ and
$\xi = f'''/f'$ — and with $L = \log f'$ these are $L'$ and $L'' + L'^2$, which
are **rational functions** for every distribution here:

| Problem | coordinate | $f''/f'$ |
|---|---|---|
| Black–Scholes | $v$ | $d_1 d_2 / v$ |
| Inverse Gaussian | $x$ | $-\tfrac{3}{2x} - \tfrac{\lambda}{2\mu^2} + \tfrac{\lambda}{2x^2}$ |
| Gamma | $y = \ln x$ | $a - x$ |
| Beta | $y = \operatorname{logit} x$ | $a - (a+b)x$ |

so the quartic step costs barely more than a Newton step.

## 3. Admissibility: why 2–3 steps suffice

Quartic convergence contracts a relative seed error $\delta$ to
$\delta^{4^N}$ after $N$ steps. A seed within
$\delta^* = \varepsilon^{1/16}$ (13.3% for $\varepsilon = 10^{-14}$) is
therefore guaranteed to converge in two steps — the regime map is designed to
keep every seed inside that basin.

![Quartic convergence](assets/convergence-light.png#only-light)
![Quartic convergence](assets/convergence-dark.png#only-dark)

This is also what makes **branch-free fixed-step kernels** honest: run exactly
$N$ updates with no convergence test, and the error is a pure function of seed
quality. Uniform instruction paths across inputs are what SIMD vectorization
requires — see [Benchmarks](benchmarks.md).

## Coordinate changes (Hekimoglu)

Solving gamma in $y = \ln x$ and beta in $y = \operatorname{logit} x$ keeps the
iterate in-domain with no clamping, makes both endpoint neighborhoods
numerically stable, and simplifies the ratios (table above). Combined with a
**density-scaled stopping rule** — require the Newton step $|f/f'|$, not just
the CDF residual, to be negligible — this holds ~1e-10 relative accuracy deep
into the tails where absolute-residual stopping silently loses 6+ digits.

## Acceptance certificates

The classical Householder-3 error model

$$e_{\text{next}} \approx K_4\, e^4, \qquad
K_4 = \left|5c_2^3 - 5c_2 c_3 + c_4\right|,\quad
c_2 = \tfrac{\varphi_2}{2},\; c_3 = \tfrac{\xi}{6},\; c_4 = \tfrac{f''''/f'}{24}$$

can be evaluated *at the seed* ($f''''/f'$ is rational too). When
$K_4 r^4$ is provably below tolerance, the solver applies the update and
returns **without a confirmation evaluation** — Hekimoglu's per-point
certificate idea, generalized across the interface. Certified and full results
agree bit-for-bit; the certificate only fires where the skip is provably safe,
which in practice requires a CF5-grade seed (it pays 1.43× for central gamma,
and is near-neutral where seeds are coarser).

## Sensitivities without AD

The implicit function theorem differentiates the fixed point directly:
$\partial v/\partial c = 1/\varphi(d_1)$,
$\partial v/\partial k = e^k\Phi(d_2)/\varphi(d_1)$ — exact wherever vega is
positive, i.e. the whole domain. No dual numbers pass through the iteration.
