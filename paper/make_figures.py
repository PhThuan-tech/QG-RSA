"""Figures for the paper, from the measured logs.

  python paper/make_figures.py

Inputs: exps/scf/results/summary.json (4 datasets x 3 seeds, scripts/summarize_scf.py).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
runs = json.loads((ROOT / "exps/scf/results/summary.json").read_text(encoding="utf-8"))["runs"]
DATASETS = (("cifar224", "CIFAR-100 (B0Inc10)", 10), ("imagenetr", "ImageNet-R (B0Inc20)", 20),
            ("imageneta", "ImageNet-A (B0Inc20)", 20), ("cub", "CUB-200 (B0Inc10)", 10))
COLORS = {"linear": "#444444", "fuse_ridge_stats_b0p25": "#1f77b4", "fuse_pgm_stats_b0p5": "#d62728"}
LABELS = {"linear": "RSIAT", "fuse_ridge_stats_b0p25": "SCF (ridge)", "fuse_pgm_stats_b0p5": "SCF (PGM)"}


def curves(ds, head):
    return np.array([r["heads"][head]["curve"] for r in runs if r["dataset"] == ds])


def style(ax, title):
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)


# Figure: accuracy versus number of classes (mean over 3 seeds), as in the RSIAT paper.
fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
for ax, (ds, title, inc) in zip(axes, DATASETS):
    for head in COLORS:
        c = curves(ds, head)
        x = inc * np.arange(1, c.shape[1] + 1)
        ax.plot(x, c.mean(0), color=COLORS[head], marker="." if head != "linear" else None, markersize=3,
                linewidth=1.2, label=LABELS[head], linestyle="--" if head == "linear" else "-")
    gain = curves(ds, "fuse_ridge_stats_b0p25")[:, -1].mean() - curves(ds, "linear")[:, -1].mean()
    ax.text(0.97, 0.94, "final $\\Delta$ {:+.2f}".format(gain), transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=COLORS["fuse_ridge_stats_b0p25"])
    style(ax, title)
    ax.set_xlabel("number of classes", fontsize=7)
axes[0].set_ylabel("top-1 accuracy (%)", fontsize=8)
axes[0].legend(fontsize=6.5, loc="lower left")
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "accuracy_curves.pdf")
fig.savefig(OUT / "accuracy_curves.png", dpi=160)

# Figure: paired difference to RSIAT at every stage (mean +- SD over seeds).
fig, axes = plt.subplots(1, 4, figsize=(11, 2.5))
for ax, (ds, title, inc) in zip(axes, DATASETS):
    base = curves(ds, "linear")
    for head in ("fuse_ridge_stats_b0p25", "fuse_pgm_stats_b0p5"):
        d = curves(ds, head) - base
        x = np.arange(d.shape[1])
        ax.plot(x, d.mean(0), color=COLORS[head], marker=".", markersize=3, linewidth=1.2, label=LABELS[head])
        ax.fill_between(x, d.mean(0) - d.std(0, ddof=1), d.mean(0) + d.std(0, ddof=1), color=COLORS[head],
                        alpha=0.12, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.6)
    style(ax, title)
    ax.set_xlabel("incremental stage $t$", fontsize=7)
axes[0].set_ylabel(r"$\Delta$ top-1 vs. RSIAT (pp)", fontsize=8)
axes[0].legend(fontsize=6.5, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "delta_per_stage.pdf")
fig.savefig(OUT / "delta_per_stage.png", dpi=160)
print("saved", sorted(p.name for p in OUT.glob("*.pdf")))
