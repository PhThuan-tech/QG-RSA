"""Ablation of statistics-calibrated fusion (SCF) for the ICISN paper: is kappa needed, is w sensitive?

Offline, no GPU. Inputs: the 12 runs of exps/scf/results/summary.json (4 datasets x 3 seeds) and the
test scores each run saved per task (``<log>_stats/qsd_scores_task_<k>.npz``: linear logits, ridge
scores, PGM log-probabilities, float16).

Rows (mean over 3 seeds):
  linear head (RSIAT); ridge alone; PGM alone                          <- logged curves
  linear + ridge, no kappa (kappa = 1, w = 1); same for PGM            <- recomputed from saved scores
  SCF ridge w in {0.25 (frozen), 1, 4}; SCF PGM w in {0.5 (frozen), 1, 4} <- logged curves

Check: SCF is also recomputed from the saved scores with the logged kappa of every task; the script
prints the largest gap to the logged curve (float16 scores may flip a few near-ties).

  python scripts/scf_ablation.py [--output paper_icisn/results/scf_ablation.md] [--json ...]
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "exps/scf/results/summary.json"
DATASETS = (("cifar224", "CIFAR-100"), ("imagenetr", "ImageNet-R"), ("imageneta", "ImageNet-A"), ("cub", "CUB-200"))
SCALE = re.compile(r"QSD task=(\d+) diag=(\{.*\}) seconds")
ROWS = (
    ("linear", "Linear head (RSIAT)", "log"),
    ("ridge_stats", "Ridge alone", "log"),
    ("pgm_stats", "PGM alone", "log"),
    ("nokappa_ridge", "Linear + ridge, no κ (κ=1, w=1)", "scores"),
    ("nokappa_pgm", "Linear + PGM, no κ (κ=1, w=1)", "scores"),
    ("fuse_ridge_stats_b0p25", "SCF ridge, w=0.25 (frozen)", "log"),
    ("fuse_ridge_stats_b1", "SCF ridge, w=1", "log"),
    ("fuse_ridge_stats_b4", "SCF ridge, w=4", "log"),
    ("fuse_pgm_stats_b0p5", "SCF PGM, w=0.5 (frozen)", "log"),
    ("fuse_pgm_stats_b1", "SCF PGM, w=1", "log"),
    ("fuse_pgm_stats_b4", "SCF PGM, w=4", "log"),
)


def stats_dir(log):
    log = ROOT / Path(log.replace("\\", "/"))
    return log, log.with_name(log.stem + "_stats")


def kappas(log):
    """Logged fusion scale per task: {task: {source: kappa}}."""
    out = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SCALE.search(line)
        if m:
            diag = json.loads(m.group(2))
            if "fuse_scale" in diag:
                out[int(m.group(1))] = diag["fuse_scale"]
    return out


def recompute(run):
    """Curves from saved scores: no-kappa fusion, and SCF with the logged kappa (as a check)."""
    log, folder = stats_dir(run["log"])
    scale = kappas(log)
    curves = {"nokappa_ridge": [], "nokappa_pgm": [], "check_ridge": [], "check_pgm": []}
    for task in range(run["tasks"]):
        z = np.load(folder / "qsd_scores_task_{}.npz".format(task))
        lin, target = z["linear"].astype(np.float64), z["target"]
        acc = lambda s: float((s.argmax(1) == target).mean() * 100)
        for source, key, w in (("ridge_stats", "ridge", 0.25), ("pgm_stats", "pgm", 0.5)):
            g = z[source].astype(np.float64)
            curves["nokappa_" + key].append(acc(lin + g))
            if task in scale:
                curves["check_" + key].append(acc(lin + w * scale[task][source] * g))
    return curves


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output")
    parser.add_argument("--json")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    runs = json.loads(SUMMARY.read_text(encoding="utf-8"))["runs"]

    table, gaps = {}, []
    for run in runs:
        curves = {head: h["curve"] for head, h in run["heads"].items()}
        extra = recompute(run)
        curves.update({k: v for k, v in extra.items() if k.startswith("nokappa")})
        for key, logged in (("check_ridge", "fuse_ridge_stats_b0p25"), ("check_pgm", "fuse_pgm_stats_b0p5")):
            if len(extra[key]) == len(curves[logged]):
                gaps.append(max(abs(a - b) for a, b in zip(extra[key], curves[logged])))
        for row, _, _ in ROWS:
            c = curves[row]
            table.setdefault((run["dataset"], row), []).append((statistics.mean(c), c[-1]))

    lines = ["# SCF ablation (mean over 3 seeds)", "",
             "Kiểm tra: SCF tính lại từ score đã lưu lệch tối đa {:.2f} điểm so với curve trong log "
             "(score lưu float16).".format(max(gaps) if gaps else float("nan")), ""]
    result = {}
    for metric, idx, label in (("final", 1, "A_B (final accuracy)"), ("aia", 0, "Ā (average incremental accuracy)")):
        lines += ["## " + label, "", "| Variant | " + " | ".join(n for _, n in DATASETS) + " | Mean |",
                  "|---|" + "---:|" * (len(DATASETS) + 1)]
        for row, name, _ in ROWS:
            vals = [statistics.mean(v[idx] for v in table[(ds, row)]) for ds, _ in DATASETS]
            result.setdefault(metric, {})[row] = dict(zip([d for d, _ in DATASETS], vals))
            lines.append("| {} | {} | {:.2f} |".format(name, " | ".join("{:.2f}".format(v) for v in vals),
                                                       statistics.mean(vals)))
        lines.append("")
    text = "\n".join(lines)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
