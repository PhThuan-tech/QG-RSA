"""Exploratory (NOT pre-registered) fusion analyses on the saved A2 scores.

Weights are chosen on the tuning group (ImageNet-R seed 1993) and reported on held-out,
like the pre-registered rule, but the grids below were defined after seeing A2.

  python -I exps/qsd/results/a2_explore.py exps/qsd/results/a2
"""
import glob
import itertools
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = sys.argv[1]
T = 10


def runs():
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*", "logs", "adapter", "*", "0", "20", "qsd2*_stats"))):
        name = os.path.basename(d)
        seed = int(name.split("_")[1] if name.split("_")[1].isdigit() else name.split("_")[2])
        ds = d.split(os.sep)[-4]
        tasks = []
        for t in range(T):
            with np.load(os.path.join(d, "qsd_scores_task_%d.npz" % t)) as z:
                s = {k: z[k].astype(np.float64) if k != "target" else z[k] for k in z.files}
            with open(os.path.join(d, "qsd_task_%d.json" % t), encoding="utf-8") as f:
                s["k"] = json.load(f)["diagnostics"]["fuse_scale"]
            tasks.append(s)
        out.append({"ds": ds, "seed": seed, "name": name, "tasks": tasks})
    return out


def aia(run, weights):
    acc = []
    for s in run["tasks"]:
        f = s["linear"].copy()
        for src, w in weights.items():
            if w:
                f += w * s["k"][src] * s[src]
        acc.append((f.argmax(1) == s["target"]).mean() * 100)
    return np.mean(acc)


def report(title, grid, rs):
    tune = [r for r in rs if (r["ds"], r["seed"]) == ("imagenetr", 1993)]
    held = [r for r in rs if r not in tune]
    base = {id(r): aia(r, {}) for r in rs}
    best = max(grid, key=lambda w: np.mean([aia(r, w) - base[id(r)] for r in tune]))
    print("\n" + title, "-> chọn", best)
    for label, group in (("tuning", tune), ("held-out R", [r for r in held if r["ds"] == "imagenetr"]),
                         ("held-out A", [r for r in held if r["ds"] == "imageneta"]), ("held-out gộp", held)):
        d = np.array([aia(r, best) - base[id(r)] for r in group])
        half = 0 if len(d) < 2 else {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}[len(d)] \
            * d.std(ddof=1) / np.sqrt(len(d))
        print("  %-13s n=%d  Δ AIA %+.3f [%+.3f, %+.3f]" % (label, len(d), d.mean(), d.mean() - half, d.mean() + half))
    return best


rs = runs()
print(len(rs), "runs")
report("1. ridge, lưới mịn hơn", [{"ridge_stats": w} for w in (0.03, 0.06, 0.1, 0.15, 0.25, 0.35)], rs)
report("2. pgm_stats, lưới mịn hơn", [{"pgm_stats": w} for w in (0.15, 0.25, 0.35, 0.5, 0.7, 1.0)], rs)
ws = (0, 0.05, 0.1, 0.2, 0.3)
wp = (0, 0.15, 0.25, 0.35, 0.5)
report("3. linear + ridge + pgm_stats", [{"ridge_stats": a, "pgm_stats": b} for a, b in itertools.product(ws, wp)], rs)
report("4. linear + ridge + pgm_stats_hier", [{"ridge_stats": a, "pgm_stats_hier": b}
                                               for a, b in itertools.product(ws, (0, 0.25, 0.5, 0.75, 1.0))], rs)
