"""Figures for the paper, from the measured A2 logs (exps/qsd/results/a2/summary.json).

  python paper/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
runs = json.loads((ROOT / "exps/qsd/results/a2/summary.json").read_text(encoding="utf-8"))["runs"]
HEADS = (("fuse_ridge_stats_b0p25", "fusion: ridge", "#1f77b4", "-"),
         ("fuse_pgm_stats_b0p5", "fusion: PGM", "#d62728", "-"),
         ("pgm_stats", "PGM alone", "#d62728", ":"),
         ("ridge_stats", "ridge alone", "#1f77b4", ":"),
         ("pgm_stats_oracle", "PGM, oracle states", "#7f7f7f", "--"))
TITLES = {"imagenetr": "ImageNet-R (6 runs)", "imageneta": "ImageNet-A (5 runs)"}

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0), sharey=False)
for ax, ds in zip(axes, ("imagenetr", "imageneta")):
    group = [r for r in runs if r["dataset"] == ds]
    for head, label, color, style in HEADS:
        delta = np.array([np.array(r["heads"][head]["curve"]) - np.array(r["heads"]["linear"]["curve"])
                          for r in group])
        ax.plot(range(delta.shape[1]), delta.mean(0), style, color=color, marker="." if style == "-" else None,
                label=label, linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(TITLES[ds], fontsize=9)
    ax.set_xlabel("incremental stage $t$", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
axes[0].set_ylabel(r"$\Delta$ top-1 vs. RSIAT (pp)", fontsize=8)
axes[1].legend(fontsize=7, loc="lower left")
fig.tight_layout()
out = ROOT / "paper/figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "delta_per_stage.pdf")
fig.savefig(out / "delta_per_stage.png", dpi=160)
print("saved", out / "delta_per_stage.pdf")
