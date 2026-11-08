"""Figures of the ACL version, from the files in the repository.

  python paper_acl/make_figures.py

  figures/pipeline.pdf   system overview (drawn, no data)
  figures/frontier.pdf   old/new trade-off of every stored run, ImageNet-R seed 1993 (scripts/frontier.py)
  figures/comparison.pdf published exemplar-free methods next to our runs (RSIAT Table 1, exps/scf)
  figures/probe_auc.pdf  class separability per task, QKSR probe (exps/qksr2)
  figures/accuracy_curves.pdf, figures/delta_per_stage.pdf  per-stage curves (exps/scf/results/summary.json)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(ROOT / "scripts"))
import frontier  # noqa: E402

INK, MUTED = "#0b0b0b", "#52514e"
QUANTUM, CLASSIC, BASE = "#2a78d6", "#eb6834", "#d4d3cf"  # validated categorical slots 1-2 + neutral


def box(ax, x, y, w, h, text, face, edge=None, size=6.6, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                facecolor=face, edgecolor=edge or face, linewidth=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, weight=weight)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8, shrinkA=0, shrinkB=0))


def pipeline():
    fig, ax = plt.subplots(figsize=(6.3, 2.75))
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(-0.01, 1.0)
    ax.axis("off")
    light_q, light_c = "#d6e6f8", "#fbdccd"
    # backbone and memory
    box(ax, 0.00, 0.40, 0.13, 0.22, "image\ntask $t$", "#f0efec")
    box(ax, 0.17, 0.36, 0.17, 0.30, "frozen ViT-B/16\n+ shared adapter\n(RSIAT)", BASE, weight="bold")
    box(ax, 0.40, 0.40, 0.11, 0.22, "feature\n$\\mathbf{z}\\in\\mathbb{R}^{768}$", "#f0efec")
    box(ax, 0.40, 0.05, 0.11, 0.22, "memory\n$(\\mu_c,\\Sigma_c)$", "#f0efec")
    arrow(ax, 0.13, 0.51, 0.17, 0.51)
    arrow(ax, 0.34, 0.51, 0.40, 0.51)
    arrow(ax, 0.455, 0.40, 0.455, 0.27)
    # training-time placements (top)
    ax.text(0.17, 0.95, "training-time placements (change the adapter)", fontsize=6.6, color=MUTED, style="italic")
    box(ax, 0.17, 0.72, 0.17, 0.19, "PQK steering\n(task 0)\ntwin: cosine / RBF", light_q, QUANTUM)
    box(ax, 0.36, 0.72, 0.19, 0.19, "PQK separation\n(QKSR, QKSR-v2)\ntwin: RBF, no CNOT", light_q, QUANTUM)
    box(ax, 0.57, 0.72, 0.19, 0.19, "Holevo $\\chi$ / residual\nPOVM loss\ntwin: MCR$^2$ / Gaussian", light_q, QUANTUM)
    arrow(ax, 0.255, 0.72, 0.255, 0.66)
    arrow(ax, 0.455, 0.72, 0.30, 0.66)
    arrow(ax, 0.665, 0.72, 0.32, 0.64)
    # inference-time placements (right)
    ax.text(0.58, 0.60, "inference-time placements (same network)", fontsize=6.6, color=MUTED, style="italic")
    box(ax, 0.58, 0.40, 0.13, 0.17, "linear head\n(baseline)", BASE)
    box(ax, 0.58, 0.05, 0.13, 0.27, "PGM on class\nstates $\\rho_c$\ntwin: ridge\non $(\\mu_c,\\Sigma_c)$", light_q, QUANTUM)
    box(ax, 0.77, 0.22, 0.12, 0.22, "calibrated\nfusion (SCF)", light_c, CLASSIC, weight="bold")
    box(ax, 0.92, 0.25, 0.08, 0.16, "class\n$\\hat y$", "#f0efec")
    arrow(ax, 0.51, 0.51, 0.58, 0.49)
    arrow(ax, 0.51, 0.16, 0.58, 0.18)
    arrow(ax, 0.71, 0.48, 0.77, 0.38)
    arrow(ax, 0.71, 0.20, 0.77, 0.28)
    arrow(ax, 0.89, 0.33, 0.92, 0.33)
    # legend
    ax.add_patch(Rectangle((0.78, 0.84), 0.025, 0.05, facecolor=light_q, edgecolor=QUANTUM, lw=0.8))
    ax.text(0.81, 0.865, "quantum component", fontsize=6.2, va="center", color=INK)
    ax.add_patch(Rectangle((0.78, 0.76), 0.025, 0.05, facecolor=light_c, edgecolor=CLASSIC, lw=0.8))
    ax.text(0.81, 0.785, "positive result", fontsize=6.2, va="center", color=INK)
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "pipeline.pdf")
    fig.savefig(OUT / "pipeline_preview.png", dpi=170)
    plt.close(fig)


SHORT = {"qksr": "QKSR", "holevo_vn": "Holevo $\\chi$", "holevo_mcr2": "MCR$^2$ twin", "holevo_spread": "$S(\\bar\\rho)$ only",
         "hp_full": "$\\chi$+POVM", "hp_povm": "POVM", "upper_oracle": "oracle stats"}


def short(p):
    if p["group"] == "inference":
        return "SVF" if p["label"] == "SVF" else "fusion (SCF)"
    fam = p["family"]
    return SHORT["qksr"] if fam.startswith("qksr") else SHORT.get(fam, fam)


def frontier_figure():
    rows = frontier.collect()
    reference, points = frontier.analyse(rows)
    verdict = frontier.verdicts(reference, points)
    scope = [p for p in points if (p["dataset"], p["seed"]) == ("imagenetr", 1993)]
    sd_old, sd_new = verdict["sd"]["old"], verdict["sd"]["new"]
    fig, ax = plt.subplots(figsize=(3.15, 2.6))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axhline(0, color=BASE, lw=0.7, zorder=1)
    ax.axvline(0, color=BASE, lw=0.7, zorder=1)
    ax.add_patch(Rectangle((-2 * sd_old, -2 * sd_new), 4 * sd_old, 4 * sd_new, color="#bbbbbb", alpha=0.35, lw=0,
                           label="baseline $\\pm$2 SD"))
    xs = np.linspace(-0.6, 1.9, 2)
    ax.plot(xs, -xs, color="#777777", ls="--", lw=0.8, label="old gain = new loss")
    style = {"training": (QUANTUM, "o", "training-time"), "stats_ca": ("#1baf7a", "s", "oracle statistics"),
             "inference": (CLASSIC, "^", "inference-time")}
    for group, (color, marker, label) in style.items():
        sel = [p for p in scope if p["group"] == group]
        ax.scatter([p["d_old"] for p in sel], [p["d_new"] for p in sel], c=color, marker=marker, s=18,
                   edgecolors="white", linewidths=0.5, zorder=3, label=label)
        names = {}
        for p in sel:
            names.setdefault(short(p), []).append((p["d_old"], p["d_new"]))
        for name, xy in names.items():
            if name in ("QKSR", "fusion (SCF)", "SVF", "oracle stats"):
                x, y = np.mean(xy, axis=0)
                offset = {"QKSR": (10, -14), "fusion (SCF)": (-12, 12), "SVF": (4, 5), "oracle stats": (-38, 6)}[name]
                ax.annotate(name, (x, y), fontsize=5.6, xytext=offset, textcoords="offset points", color=MUTED,
                            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.4) if name != "SVF" else None)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-1.75, 0.7)
    ax.set_xlabel("$\\Delta$ accuracy, tasks 0–4 (points)", fontsize=6.8)
    ax.set_ylabel("$\\Delta$ accuracy, tasks 5–9 (points)", fontsize=6.8)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.4, loc="lower left", frameon=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "frontier.pdf")
    fig.savefig(OUT / "frontier_preview.png", dpi=200)
    plt.close(fig)
    return verdict


# Final accuracy A_B reported in Table 1 of RSIAT (Zhao et al., CVPR 2026): exemplar-free methods,
# ViT-B/16-IN21K, CIFAR B0Inc10, IN-R B0Inc20, IN-A B0Inc20, CUB B0Inc10 (authors' partitions, one run).
PUBLISHED = {  # method: (CIFAR, IN-R, IN-A, CUB) final accuracy
    "L2P": (84.29, 69.77, 41.71, 56.25), "DualPrompt": (82.30, 67.18, 41.67, 66.54),
    "CODA-Prompt": (86.85, 72.27, 42.73, 73.37), "APER+Adapter": (85.75, 67.95, 49.37, 86.73),
    "EASE": (87.76, 76.17, 55.04, 86.81), "SLCA": (91.53, 77.00, 58.74, 82.19),
    "RanPAC": (90.86, 77.94, 61.82, 89.40), "MOS": (91.70, 77.93, 59.12, 90.12),
    "SSIAT": (91.35, 79.38, 62.43, 89.53), "RSIAT (published)": (92.20, 82.75, 66.23, 90.59),
}
DATASETS = (("cifar224", "CIFAR-100"), ("imagenetr", "ImageNet-R"), ("imageneta", "ImageNet-A"), ("cub", "CUB-200"))


def ours():
    import json
    import statistics
    runs = json.loads((ROOT / "exps/scf/results/summary.json").read_text(encoding="utf-8"))["runs"]
    out = {}
    for label, head in (("RSIAT (ours)", "linear"), ("SCF, ridge (ours)", "fuse_ridge_stats_b0p25"),
                        ("SCF, PGM (ours)", "fuse_pgm_stats_b0p5")):
        out[label] = tuple(statistics.mean(r["heads"][head]["curve"][-1] for r in runs if r["dataset"] == ds)
                           for ds, _ in DATASETS)
    return out


def comparison():
    mine = ours()
    names = list(PUBLISHED) + list(mine)
    values = {**PUBLISHED, **mine}
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 2.55), sharey=True)
    y = np.arange(len(names))[::-1]
    for k, (ax, (_, title)) in enumerate(zip(axes, DATASETS)):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for i, name in enumerate(names):
            v = values[name][k]
            if name in mine:
                color, marker = (CLASSIC, "D") if "ridge" in name else (QUANTUM, "s") if "PGM" in name else (MUTED, "o")
            else:
                color, marker = ("#8a8984", "o") if name != "RSIAT (published)" else (INK, "o")
            ax.plot([0, v], [y[i], y[i]], color="#e6e5e1", lw=0.8, zorder=1)
            ax.scatter([v], [y[i]], color=color, marker=marker, s=14, zorder=3, edgecolors="white", linewidths=0.4)
        ax.axhline(len(mine) - 0.5, color=BASE, lw=0.6, ls=":")
        lo = min(values[n][k] for n in names)
        ax.set_xlim(lo - 3, max(values[n][k] for n in names) + 2)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=5.8)
        ax.grid(axis="x", color="#eeeeec", lw=0.5)
        ax.set_xlabel("final accuracy (%)", fontsize=6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names, fontsize=6)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    fig.savefig(OUT / "comparison.pdf")
    fig.savefig(OUT / "comparison_preview.png", dpi=200)
    plt.close(fig)


def probe_auc():
    import json
    import re
    log = next((ROOT / "exps/qksr2/results/p0/logs/adapter/imagenetr/0/20").glob("p0_qksr_cur_*.log"))
    ends = {int(t): json.loads(p) for t, p in
            re.findall(r"QKSRPROBE end task=(\d+) (\{.*\})", log.read_text(encoding="utf-8", errors="replace"))}
    tasks = sorted(ends)
    series = (("cosine on $\\mathbf{z}$ (768-d)", lambda e: e["twin_cosine_z"]["auc_class"], QUANTUM, "o"),
              ("Euclidean on $\\mathbf{z}$", lambda e: e["twin_euclid_z"]["auc_class"], CLASSIC, "s"),
              ("RBF on 8 angles", lambda e: e["twin_euclid_theta"]["auc_class"], "#1baf7a", "^"),
              ("PQK", lambda e: e["pqk"]["auc_class"], "#eda100", "D"))
    fig, ax = plt.subplots(figsize=(3.15, 2.2))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for label, get, color, marker in series:
        vals = [get(ends[t]) for t in tasks]
        ax.plot(tasks, vals, color=color, lw=1.4, marker=marker, markersize=3.2, label=label)
        ax.annotate(label, (tasks[-1], vals[-1]), xytext=(4, 0), textcoords="offset points", fontsize=5.6,
                    va="center", color=MUTED)
    ax.axhline(0.5, color=BASE, lw=0.7, ls=":")
    ax.text(0.1, 0.505, "chance", fontsize=5.6, color=MUTED, va="bottom")
    ax.set_xlim(-0.3, 13.2)
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(tasks)
    ax.set_xlabel("task", fontsize=6.8)
    ax.set_ylabel("class AUC", fontsize=6.8)
    ax.tick_params(labelsize=6)
    ax.grid(axis="y", color="#eeeeec", lw=0.5)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "probe_auc.pdf")
    fig.savefig(OUT / "probe_auc_preview.png", dpi=200)
    plt.close(fig)


HEADS = (("linear", "RSIAT", "#52514e"), ("fuse_ridge_stats_b0p25", "SCF (ridge)", CLASSIC),
         ("fuse_pgm_stats_b0p5", "SCF (PGM)", QUANTUM))
STAGES = (("cifar224", "CIFAR-100 (B0Inc10)", 10), ("imagenetr", "ImageNet-R (B0Inc20)", 20),
          ("imageneta", "ImageNet-A (B0Inc20)", 20), ("cub", "CUB-200 (B0Inc10)", 10))


def stage_curves():
    import json
    runs = json.loads((ROOT / "exps/scf/results/summary.json").read_text(encoding="utf-8"))["runs"]
    return lambda ds, head: np.array([r["heads"][head]["curve"] for r in runs if r["dataset"] == ds])


def light(ax, title):
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6.5)
    ax.grid(alpha=0.25, lw=0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def accuracy_curves():
    curves = stage_curves()
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6))
    for ax, (ds, title, inc) in zip(axes, STAGES):
        for head, label, color in HEADS:
            c = curves(ds, head)
            x = inc * np.arange(1, c.shape[1] + 1)
            ax.plot(x, c.mean(0), color=color, marker=None if head == "linear" else ".", markersize=3, lw=1.2,
                    ls="--" if head == "linear" else "-", label=label)
        gain = curves(ds, HEADS[1][0])[:, -1].mean() - curves(ds, "linear")[:, -1].mean()
        ax.text(0.97, 0.94, "final $\\Delta$ (ridge) {:+.2f}".format(gain), transform=ax.transAxes, ha="right",
                va="top", fontsize=7, color=CLASSIC)
        light(ax, title)
        ax.set_xlabel("number of classes", fontsize=7)
    axes[0].set_ylabel("top-1 accuracy (%)", fontsize=8)
    axes[0].legend(fontsize=6.5, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_curves.pdf")
    plt.close(fig)


def delta_per_stage():
    curves = stage_curves()
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.5))
    for ax, (ds, title, _) in zip(axes, STAGES):
        base = curves(ds, "linear")
        for head, label, color in HEADS[1:]:
            d = curves(ds, head) - base
            x = np.arange(d.shape[1])
            ax.plot(x, d.mean(0), color=color, marker=".", markersize=3, lw=1.2, label=label)
            ax.fill_between(x, d.mean(0) - d.std(0, ddof=1), d.mean(0) + d.std(0, ddof=1), color=color,
                            alpha=0.12, lw=0)
        ax.axhline(0, color=INK, lw=0.6)
        light(ax, title)
        ax.set_xlabel("incremental stage $t$", fontsize=7)
    axes[0].set_ylabel("$\\Delta$ top-1 vs. RSIAT (points)", fontsize=8)
    axes[0].legend(fontsize=6.5, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "delta_per_stage.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pipeline()
    print(frontier_figure())
    comparison()
    probe_auc()
    accuracy_curves()
    delta_per_stage()
