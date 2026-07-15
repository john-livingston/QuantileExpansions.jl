# Regime-Split Quantile Solver — Working Context

**Purpose of this file.** Self-contained context for an independent coding session. Goal: take a proven fast Black–Scholes implied-volatility (BS-IV) inversion method and generalize it into a _single, modular, distribution-generic quantile solver_ — initially in Julia — covering BS-IV, **inverse Gaussian (IG)**, **gamma**, and **beta**. The IG case is structurally almost identical to BS; gamma/beta reuse the same architecture with different machinery.

Provenance: the BS-IV method is Hekimoglu & Gökgöz, _A Fast Implied Volatility Method with Expansions_ (arXiv:2606.10245). The cross-distribution generalization is the lead author's idea; the IG↔BS correspondence detail, gamma/beta machinery, and Julia plan below are synthesized working notes.

---

## 1. The abstraction

Every target is an **inverse-CDF / quantile** problem: given a CDF $F$ and a probability (or price) $p$, solve $F(x) = p$ for $x$. The forward map is explicit; the inverse is not elementary. The proven recipe has **three reusable ingredients**:

1. **Regime-split the domain** by asymptotic behavior of $F$.
2. **Closed-form analytical seed per regime**, derived from the local asymptotic structure (series inversion near the center; polynomial-CDF approximation in the middle; Mills-ratio / tail asymptotics in the tail). No iteration at the seed stage.
3. **One universal high-order polish step** (4th-order Householder, "HH-4") that is cheap because $\log f$ has a **rational derivative**, making $f''/f'$ and $f'''/f'$ closed-form.

Regime boundaries are derived from **CDF-truncation error tolerances**, not fitted. A seed whose relative error is below the polisher's admissibility threshold converges in ≤2 steps.

---

## 2. The proven template: BS-IV

Normalized, undiscounted call price (forward measure): $$c_{BS}(k,v) = \Phi(d_1) - e^{k},\Phi(d_2),\qquad d_1=-\tfrac{k}{v}+\tfrac{v}{2},\quad d_2=d_1-v=-\tfrac{k}{v}-\tfrac{v}{2}$$ where $k=\log(K/F)$ is log-moneyness, $v=\sigma\sqrt{T}$ is **total volatility** (the unknown), $\Phi$ is the standard normal CDF. $a_1 = 1/\sqrt{2\pi}=0.3989422804$. Let $\kappa=|k|$.

### 2.1 Seeds

**ATM (Taylor inversion), $|k|<\kappa_1$.** Time value $c_{tv}=c-\max(1-e^k,0)$. At $k=0$: $c_{tv}=2\Phi(v/2)-1=\mathrm{erf}(v/(2\sqrt2))$. With $s=\sqrt{2\pi},c_{tv}$, $$v = s\Big(1+\tfrac{s^2}{24}+\tfrac{7s^4}{1920}+\tfrac{127 s^6}{322560}\Big)+O(s^8).$$ First-order truncation $v_0=s$ is the Brenner–Subrahmanyam estimate. Use the 4th-order form.

**Mild-OTM (polynomial-CDF seed).** Approximate $\Phi$ by an odd polynomial $P_{2m+1}$. $P_1(x)=\tfrac12+a_1 x$ → quadratic in $v$ → closed form. With $\varphi=k+\tfrac{k^2}{2}+\tfrac{k^3}{6}+\tfrac{k^4}{24}$ (4th-order $e^k$) and observed $c$: $$v_{P1}=\frac{2c+\varphi+\sqrt{N}}{2a_1(2+\varphi)},\qquad N=(2c+\varphi)^2-8a_1^2,k,\varphi,(2+\varphi).$$ Fallback if $N\le0$: drop $\sqrt N$ (clip discriminant to zero). Higher orders are **one Newton step** on the $P_3$/$P_7$ price surrogates, in closed form: $$v_{P3}=v_{P1}\Big(1+\tfrac{G_3(w)}{D_3(w)}\Big),\quad v_{P7}=v_{P1}\Big(1+\tfrac{G_7(w)}{D_7(w)}\Big),\quad w=v_{P1}.$$ $G_3,D_3$ share 4 of 5 terms; $G_7,D_7$ share a polynomial $Q_7$ (compute once). Full $G,D$ expressions are in the paper's Appendix 8.3–8.4 (port verbatim when needed).

**Deep-OTM (Mills-ratio corrected).** Always invert an OTM-equivalent price: $c_{seed}=c$ for $k\ge0$; $c_{seed}=e^{-k}(c-1+e^k)$ for $k<0$. Plain quadratic seed: $v_q=z+\sqrt{z^2+2\kappa}$, $z=\Phi^{-1}(c_{seed})$. Mills identity (Lemma 8.5/8.6): $c_{BS}\approx\Phi(d_1)\cdot \dfrac{v^2}{\kappa+v^2/2}$. Correction $\rho_q=v_q^2/(\kappa+v_q^2/2)$, then $$z_q=\Phi^{-1}(c_{seed}/\rho_q),\qquad v_{q,1}=z_q+\sqrt{z_q^2+2\kappa}.$$ $\Phi^{-1}$ via Acklam rational approx, **or** the fully algebraic Soranzo–Epure invertible $\Phi$ (solve a quadratic; transcendental-free — important for SIMD/GPU).

### 2.2 Regime map (Definition 2.9)

Boundaries $\kappa_1=0.001,\ \kappa_2=0.81,\ \kappa_3=1.155,\ \kappa_4=1.347$; tail filter $c_{seed}<c^\star=0.02128$ and $|k|>\kappa^\star=0.5$.

|Regime|Condition|Seed|
|---|---|---|
|ATM|$\|k\|<\kappa_1$|$v_{ATM}$ (4th-order Taylor)|
|Mild-OTM|$\kappa_1\le\|k\|\le\kappa_2$|$\max(v_{P7}, v_{ATM})$|
|Transition (bracket)|$\kappa_2<\|k\|\le\kappa_3$|$\max(\tfrac12(v_{P3}+v_{P7}), v_{ATM})$|
|Transition (revert)|$\kappa_3<\|k\|\le\kappa_4$|$\max(v_{P3}, v_{ATM})$|
|Deep-OTM|$\|k\|>\kappa_4$|$\max(v_{q,1}, v_{ATM})$|
|Tail filter (override)|$c_{seed}<0.02128 \wedge \|k\|>0.5$|$v_{q,1}$|

Boundaries come from $|a_9 x^9|\le\varepsilon_\Phi$ on $x=d_2$, peak $k^{peak}=\tfrac12(\varepsilon_\Phi/|a_9|)^{2/9}$, $|a_9|=1/(3456\sqrt{2\pi})$: $\varepsilon_\Phi=10^{-3}!\to!0.81$, $5!\times!10^{-3}!\to!1.155$, $10^{-2}!\to!1.347$.

### 2.3 Polisher: HH-4 (Householder, $d=3$, quartic)

$f=c_{BS}-c$, $f'=\phi(d_1)$ (normal density). Use the exact identities $\partial d_1/\partial v=-d_2/v$, $\partial d_2/\partial v=-d_1/v$: $$r=\frac{f}{f'},\quad \varphi_2=\frac{f''}{f'}=\frac{d_1 d_2}{v},\quad \xi=\frac{f'''}{f'}=\frac{(d_1d_2)^2-(d_1^2+d_2^2)-d_1d_2}{v^2},$$ $$v\leftarrow v+\frac{3r(2-r\varphi_2)}{-6+6r\varphi_2-r^2\xi}\quad(\text{Newton fallback }v\leftarrow v-r\text{ if denom}<10^{-20}).$$ Stop at $|f|<10^{-14}$. **Admissibility:** two net HH-4 steps reach tolerance $\varepsilon_v$ from relative seed error $\delta_v^\star=\varepsilon_v^{1/16}$; for $\varepsilon_v=10^{-14}$, $\delta_v^\star\approx13.3\%$. The seed map keeps mean updates **below 2**.

### 2.4 Reference performance (C, GCC 16.1)

~87–125 ns/IV; **1.7–1.8× faster** than Jäckel's "Let's Be Rational" on identical hardware; max abs error $\sim10^{-14}$. Python/Numba: 134 ns/IV serial, 24 ns/IV at 12 threads.

---

## 3. Per-distribution machinery

### 3.1 Inverse Gaussian — the close cousin (port the BS scaffold almost verbatim)

CDF ($\mu$ mean, $\lambda$ shape): $$F(x;\mu,\lambda)=\Phi!\Big(\sqrt{\tfrac{\lambda}{x}}\big(\tfrac{x}{\mu}-1\big)\Big)+e^{2\lambda/\mu},\Phi!\Big(-\sqrt{\tfrac{\lambda}{x}}\big(\tfrac{x}{\mu}+1\big)\Big).$$ This is **exactly the BS two-$\Phi$ shape**: $\Phi(\alpha)+e^{c}\Phi(\beta)$ with arguments differing by a structured shift, and the unknown $x$ entering nonlinearly through both $\sqrt{\lambda/x}$ and $x/\mu$ (mirroring $v$ in $d_1,d_2$).

Correspondence:

|BS-IV|IG quantile|
|---|---|
|unknown $v=\sigma\sqrt T$|unknown $x$|
|$d_1=-k/v+v/2$, $d_2=-k/v-v/2$|$\alpha=\sqrt{\lambda/x}(x/\mu-1)$, $\beta=-\sqrt{\lambda/x}(x/\mu+1)$|
|prefactor $e^k$|prefactor $e^{2\lambda/\mu}$|
|ATM seed = Taylor inversion at $k=0$|seed = Taylor inversion near mode/median|
|Mills tail cancellation (Lemma 8.5)|identical mechanism for the IG tail quantile|
|polynomial-CDF mild seed|carries over directly|

Plan: parameterize the regime-split solver on $(\alpha(x),\beta(x),\text{prefactor})$. The log-derivatives of the IG density are rational in $x$, so HH-4 stays cheap. Expect the full scaffold to transfer with only the argument definitions and prefactor changed.

### 3.2 Gamma (looser analogy, same architecture)

Target: invert the regularized lower incomplete gamma $P(a,x)=p$. No two-$\Phi$ structure, but regime-split is standard:

- **Large $a$:** Wilson–Hilferty normal seed $x\approx a\big(1-\tfrac{1}{9a}+\tfrac{z_p}{\sqrt{9a}}\big)^3$, $z_p=\Phi^{-1}(p)$.
- **Small-$x$ lower tail:** $P(a,x)\approx \tfrac{x^a}{\Gamma(a+1)}\Rightarrow x\approx(p,a,\Gamma(a))^{1/a}$.
- **Upper tail:** asymptotic series of $Q(a,x)=1-P$.
- **Polish:** density log-derivative $\tfrac{d}{dx}\log f=\tfrac{a-1}{x}-1$ is rational ⇒ $f''/f'$, $f'''/f'$ closed-form ⇒ HH-4 nearly free per step.

### 3.3 Beta (looser analogy, same architecture)

Target: invert the regularized incomplete beta $I_x(a,b)=p$.

- **Seeds:** normal/Cornish–Fisher for moderate $a,b$; small-$x$ / $(1-x)$ tail power-law seeds by symmetry $I_x(a,b)=1-I_{1-x}(b,a)$; Newton seeds otherwise.
- **Polish:** $\tfrac{d}{dx}\log f=\tfrac{a-1}{x}-\tfrac{b-1}{1-x}$ is rational ⇒ HH-4 cheap.

---

## 4. Julia implementation plan

### 4.1 Generic architecture (the main payoff)

Write the regime-split solver **once**, generic over distribution; dispatch the per-distribution pieces. The compiler **monomorphizes** each instantiation into specialized, inlined, C-speed code — exactly the reuse that's painful in C (macros / function pointers, where pointers cost the inlining that's load-bearing here).

Dispatch points (a small interface each distribution implements):

- `cdf_args(dist, x)` → the structured arguments (e.g. $(d_1,d_2)$ / $(\alpha,\beta)$).
- `forward(dist, x)` and `residual(dist, x, p)`.
- `logderiv_ratios(dist, x)` → $(f'/f,\ f''/f',\ f'''/f')$ for HH-4.
- `seed(dist, regime, p, params)` → closed-form seed.
- `regime(dist, p, params)` → regime selector + boundaries.

The HH-4 loop and the admissibility/stopping logic are distribution-agnostic and written once over the interface.

### 4.2 Suggested module layout

```
QuantileExpansions/
  src/
    QuantileExpansions.jl     # exports, the generic solve(dist, p; tol)
    core/
      householder.jl          # HH-4 step + Newton fallback, generic
      regime.jl               # regime dispatch + admissibility (δ_v* = tol^(1/16))
      specialfuns.jl          # Φ, Φ⁻¹ (Acklam + Soranzo–Epure algebraic), erfc
    dists/
      blackscholes.jl         # the reference template (validate against C numbers)
      inverse_gaussian.jl     # closest port
      gamma.jl
      beta.jl
  test/                       # ground-truth round-trip: x → p=F(x) → invert → compare
  bench/                      # @btime, threaded (Polyester @batch), vs Numba/C numbers
```

### 4.3 Performance targets & pitfalls

Target: **parity with C** (shared LLVM/Clang backend), ~90–115 ns/IV for a type-stable, allocation-free scalar solver with `@inbounds @fastmath` and a good `erfc`/$\Phi^{-1}$ (Soranzo–Epure ports in ~10 lines, stays fast). Beats Numba serial (134); thread comparison vs 24 ns/IV is fair. Pitfalls that wrongly make Julia "look slow":

- any `Union`/`Any` type instability in the hot path,
- any heap allocation inside the loop,
- benchmarking with `@time` (times compilation) instead of `@btime`.

Recommended split: **keep the C kernel as the deployable artifact**; use Julia as the unifying research harness where all four inversions are one parameterized solver; `ccall` out only where a specific special function genuinely beats Julia's.

---

## 5. Ideas worth exploring

- **Transcendental-free path:** swap all $\Phi/\Phi^{-1}$ for Soranzo–Epure → the entire seed chain is arithmetic-only → unlocks clean SIMD and a GPU port.
- **Batched/SIMD inversion:** the regime _branch_ blocks vectorization; sort/bucket inputs by regime, then vectorize within each bucket.
- **AD through the solver** (Julia ForwardDiff/Enzyme): get sensitivities/greeks "for free"; verify the fixed point is differentiable.
- **Alternative coordinates:** Schadner (arXiv:2604.24480) recasts BS-IV as an inverse-Gaussian quantile in variance space — directly relevant to the IG work; compare whether his transform simplifies any regime.
- **Codegen the C from the Julia spec:** if the generic Julia solver is the single source of truth, emit the per-distribution C kernel from it (metaprogramming).
- **Benchmark against existing libraries:** Boost, Cephes, R `qgamma`/`qbeta`, Distributions.jl — establish where the expansion approach actually wins.
- **Extend the family:** Student-t and other distributions whose CDF shares the "$\Phi$ + weighted $\Phi$" or "rational log-density" structure. Noncentral chi-square was explored (see the noncentral chi-square marginal section of RESULTS.md) and does NOT fit: its log-density carries a modified-Bessel ratio, not a rational function, so it needs a dedicated Bessel branch rather than the near-free HH-4 path (still a 4x to 16x win over Distributions, just not through this mechanism).

---

## 6. Source pointers

- Main method & all seed/boundary derivations: Hekimoglu & Gökgöz, arXiv:2606.10245 (Appendices 8.1–8.8 hold the full $G/D$ polynomials, Soranzo–Epure $\Phi^{-1}$, Mills derivation, admissibility proofs). Code: github.com/variancegamma/A-Fast-IV-method-with-expansions.
- Benchmark reference: P. Jäckel, _Let's Be Rational_, Wilmott 2015.
- $\Phi^{-1}$ approximations: Acklam (2003); Soranzo & Epure (arXiv:1201.1320).
