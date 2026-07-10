#!/usr/bin/env python3
"""Generate docs figures (light + dark variants) into docs/assets/.

All data are measured benchmark numbers from RESULTS.md — no randomness, so the
output is deterministic. Colors are the validated reference palette (dataviz
skill: CVD >= 12 in both modes; light-mode aqua < 3:1 carries the relief rule,
satisfied with direct value labels).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

PAL = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", sub="#52514e",
                  grid="#e4e3df", s1="#2a78d6", s2="#1baf7a"),
    "dark":  dict(surface="#1a1a19", text="#ffffff", sub="#c3c2b7",
                  grid="#33322f", s1="#3987e5", s2="#199e70"),
}


def style(ax, p):
    ax.set_facecolor(p["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(p["grid"])
    ax.tick_params(colors=p["sub"], labelsize=9)
    ax.xaxis.label.set_color(p["sub"])
    ax.yaxis.label.set_color(p["sub"])
    ax.title.set_color(p["text"])


def save(fig, name, mode):
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=200, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    plt.close(fig)


# 1. BS-IV single-thread progression -----------------------------------------
def fig_progression(mode):
    p = PAL[mode]
    labels = ["C reference (same host)", "first Julia port", "+ exp-reuse identity",
              "branch-free fixed-2", "SIMD batch fixed-3", "SIMD batch fixed-2"]
    ns = [113.7, 100.5, 69.1, 51.9, 44.6, 33.2]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    fig.patch.set_facecolor(p["surface"])
    y = range(len(ns))
    ax.barh(y, ns, height=0.55, color=p["s1"], zorder=3)
    ax.set_yticks(list(y), labels)
    ax.invert_yaxis()
    ax.set_xlabel("nanoseconds per implied vol (single thread)")
    ax.xaxis.grid(True, color=p["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for yi, v in zip(y, ns):
        ax.text(v + 1.5, yi, f"{v:g} ns", va="center", ha="left",
                color=p["text"], fontsize=9)
    ax.set_xlim(0, 130)
    ax.set_title("Black–Scholes implied vol: single-thread progression", fontsize=11, loc="left", color=p["text"])
    style(ax, p)
    save(fig, "bs_progression", mode)


# 2. per-quantile speed vs reference libraries --------------------------------
def fig_speed(mode):
    p = PAL[mode]
    dists = ["BS-IV", "Inverse\nGaussian", "Gamma", "Beta"]
    ours = [69.1, 76.2, 129.8, 302.7]
    ref = [113.7, 574.9, 269.9, 821.1]
    ref_name = ["C ref", "Distributions.jl", "Distributions.jl", "Distributions.jl"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    fig.patch.set_facecolor(p["surface"])
    x = range(len(dists))
    w = 0.36
    ax.bar([i - w/2 - 0.01 for i in x], ours, width=w, color=p["s1"],
           label="QuantileExpansions", zorder=3)
    ax.bar([i + w/2 + 0.01 for i in x], ref, width=w, color=p["s2"],
           label="reference", zorder=3)
    for i, (o, r, rn) in enumerate(zip(ours, ref, ref_name)):
        ax.text(i - w/2 - 0.01, o + 12, f"{o:g}", ha="center", color=p["text"], fontsize=8.5)
        ax.text(i + w/2 + 0.01, r + 12, f"{r:g}\n({rn})", ha="center",
                color=p["sub"], fontsize=8)
    ax.set_xticks(list(x), dists)
    ax.set_ylabel("nanoseconds per quantile (single thread)")
    ax.set_ylim(0, 980)
    ax.yaxis.grid(True, color=p["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=9)
    for t in leg.get_texts():
        t.set_color(p["text"])
    ax.set_title("Per-quantile speed vs reference implementations", fontsize=11, loc="left", color=p["text"])
    style(ax, p)
    save(fig, "speed_vs_ref", mode)


# 3. quartic convergence from the seed ----------------------------------------
def fig_convergence(mode):
    p = PAL[mode]
    floor = 1.1e-16
    steps = [0, 1, 2, 3]
    series = [("worst dense seed (δ = 0.33)", 0.33, p["s1"]),
              ("typical seed (δ = 0.05)", 0.05, p["s2"])]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    fig.patch.set_facecolor(p["surface"])
    for name, d, col in series:
        errs = [max(d ** (4 ** n), floor) for n in steps]
        ax.plot(steps, errs, "-o", color=col, lw=2, ms=7, label=name, zorder=3,
                markeredgecolor=p["surface"], markeredgewidth=1.5)
    ax.axhline(floor, color=p["sub"], lw=1, ls="--", zorder=1, alpha=0.6)
    ax.text(0.02, floor * 3.0, "double-precision floor", color=p["sub"], fontsize=8, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(3e-17, 3)
    ax.set_xticks(steps)
    ax.set_xlabel("HH-4 updates applied")
    ax.set_ylabel("relative error in v")
    ax.yaxis.grid(True, color=p["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(p["text"])
    ax.set_title("Quartic convergence: seed error δ contracts as δ^(4ᴺ)", fontsize=11, loc="left", color=p["text"])
    style(ax, p)
    save(fig, "convergence", mode)


for mode in ("light", "dark"):
    fig_progression(mode)
    fig_speed(mode)
    fig_convergence(mode)
print("wrote", sorted(f.name for f in OUT.glob("*.png")))
