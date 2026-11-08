"""Slide figures for the SCF contribution (16:9, large fonts), from exps/scf/results/summary.json.

  python paper_acl/make_slide_figures.py

  figures/slide_scf_accuracy.png  A-bar and A_B of RSIAT, SCF-ridge, SCF-PGM per dataset (mean +- SD, 3 seeds)
  figures/slide_scf_gain.png      paired gain of SCF over RSIAT with 95% t-interval
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
RUNS = json.loads((ROOT / "exps/scf/results/summary.json").read_text(encoding="utf-8"))["runs"]
DATASETS = (("cifar224", "CIFAR-100"), ("imagenetr", "ImageNet-R"), ("imageneta", "ImageNet-A"), ("cub", "CUB-200"))
HEADS = (("linear", "RSIAT", "#8a8985"), ("fuse_ridge_stats_b0p25", "SCF, ridge (classical)", "#eb6834"),
         ("fuse_pgm_stats_b0p5", "SCF, PGM (quantum)", "#2a78d6"))
METRICS = (("aia", "Average accuracy $\\bar A$ (%)"), ("final", "Final accuracy $A_B$ (%)"))
T975 = {2: 12.706, 3: 4.303, 4: 3.182}  # two-sided 95% t quantiles, n-1 degrees of freedom
INK, MUTED = "#0b0b0b", "#52514e"


def values(ds, head, metric):
    """Per-seed values, ordered by seed so that heads are paired."""
    runs = sorted((r for r in RUNS if r["dataset"] == ds), key=lambda r: r["seed"])
    curves = [r["heads"][head]["curve"] for r in runs]
    return np.array([np.mean(c) if metric == "aia" else c[-1] for c in curves])


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=0.3, lw=0.6)
    ax.set_axisbelow(True)


def accuracy():
    """Two rows (A-bar, A_B) x four datasets; each panel has its own y-range so the differences are visible."""
    fig, axes = plt.subplots(2, 4, figsize=(13.3, 6.6))
    for row, (metric, label) in enumerate(METRICS):
        for col, (ds, title) in enumerate(DATASETS):
            ax = axes[row, col]
            for i, (head, name, color) in enumerate(HEADS):
                v = values(ds, head, metric)
                ax.bar(i, v.mean(), width=0.62, color=color, yerr=v.std(ddof=1), capsize=4,
                       error_kw=dict(lw=1, ecolor=INK), label=name)
                ax.text(i, v.mean() + v.std(ddof=1) + 0.05, "{:.2f}".format(v.mean()), ha="center", va="bottom",
                        fontsize=11, color=INK)
            lows = [values(ds, h, metric).mean() - values(ds, h, metric).std(ddof=1) for h, _, _ in HEADS]
            highs = [values(ds, h, metric).mean() + values(ds, h, metric).std(ddof=1) for h, _, _ in HEADS]
            ax.set_ylim(min(lows) - 0.8, max(highs) + 0.6)
            ax.set_xticks([])
            clean(ax)
            if row == 0:
                ax.set_title(title, fontsize=14, color=INK)
            if col == 0:
                ax.set_ylabel(label, fontsize=13)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in HEADS]
    fig.legend(handles, [n for _, n, _ in HEADS], loc="lower center", ncol=3, fontsize=13, frameon=False)
    fig.text(0.5, 0.075, "Mean $\\pm$ SD over seeds 1993, 2015, 2026. The y-axis does not start at 0.",
             ha="center", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(OUT / "slide_scf_accuracy.png", dpi=200)
    plt.close(fig)


def gain():
    """Paired gain over RSIAT: same seed, same network, so the run-to-run noise cancels."""
    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.4), sharey=True)
    x = np.arange(len(DATASETS))
    rows = []
    for ax, (metric, label) in zip(axes, METRICS):
        for k, (head, name, color) in enumerate(HEADS[1:]):
            means, half = [], []
            for ds, _ in DATASETS:
                d = values(ds, head, metric) - values(ds, "linear", metric)
                h = T975[len(d)] * d.std(ddof=1) / np.sqrt(len(d))
                means.append(d.mean())
                half.append(h)
                rows.append((metric, name, ds, d.mean(), d.mean() - h, d.mean() + h))
            pos = x + (k - 0.5) * 0.36
            ax.bar(pos, means, width=0.34, color=color, yerr=half, capsize=4, error_kw=dict(lw=1, ecolor=INK),
                   label=name)
            for p, m, h in zip(pos, means, half):
                top = m + h if m >= 0 else m - h
                ax.text(p, top + (0.03 if m >= 0 else -0.03), "{:+.2f}".format(m), ha="center",
                        va="bottom" if m >= 0 else "top", fontsize=11, color=color, fontweight="bold")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x, [t for _, t in DATASETS], fontsize=13)
        ax.set_title("Gain in " + label.split(" (")[0].replace("accuracy", "accuracy"), fontsize=14)
        clean(ax)
    axes[0].set_ylabel("SCF $-$ RSIAT (points)", fontsize=13)
    axes[0].legend(fontsize=12, loc="upper left", frameon=False)
    fig.text(0.5, 0.01, "Paired difference over 3 seeds (same trained network); error bars: 95% confidence interval.",
             ha="center", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "slide_scf_gain.png", dpi=200)
    plt.close(fig)
    return rows


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    accuracy()
    for metric, name, ds, m, lo, hi in gain():
        print("{:5s} {:24s} {:10s} {:+.2f} [{:+.2f}, {:+.2f}]".format(metric, name, ds, m, lo, hi))
    for ds, _ in DATASETS:
        print(ds, *("{}={:.2f}/{:.2f}".format(h, statistics.mean(values(ds, h, "aia")),
                                             statistics.mean(values(ds, h, "final"))) for h, _, _ in HEADS))
