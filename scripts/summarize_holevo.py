"""Summarise Holevo-CIL (exps/holevo/README.md) and Holevo-POVM (exps/holevo_povm/README.md).

Runs of families holevo_* and hp_* (replicate 0) are parsed from their logs; pair ratios come from
``<log>_stats/predictions_task_<last>.npz``. Screening rules use the tuning run (ImageNet-R 1993)
against the pre-registered baseline of the four ``noise`` runs; held-out rules need RSIAT runs of
the same seeds (family ``qsd2``, round A2), e.g. ``--root exps/qsd/results/a2``.

  python scripts/summarize_holevo.py --root <logs> [--root <logs> ...] [--output summary.md] [--json summary.json]
  python scripts/summarize_holevo.py --check-baseline exps/measurement/results/logs/adapter/imagenetr/0/20
"""
import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_level_stats import RUN_NAME, split_family  # noqa: E402
from scripts.summarize_ca_heads import fmt, fmt_ci, interval  # noqa: E402
from scripts.summarize_scf import CNN, CURVE, NAMES, ORDER  # noqa: E402
from utils.qsd import head_metrics  # noqa: E402

# Pre-registered (exps/holevo/README.md 3.5, 4; exps/holevo_povm/README.md 4). Do not edit after seeing results.
# Baseline: the four runs noise, noise_rep1..3 (ImageNet-R 1993, exps/measurement/results), measured
# 2026-10-30 before any Holevo run (--check-baseline recomputes them).
BASE = {"final": 82.588, "pair_ratio": 1.3806, "pair_ratio_sd": 0.0305, "last3": 84.732, "task0": 96.05}
DELTA_M = max(0.10, 2 * BASE["pair_ratio_sd"])
MIN_O, MAX_P, MIN_D, MIN_H = 0.5, 1.0, 0.5, 0.5
TUNING = ("imagenetr", 1993)
HELD_OUT = {"imagenetr": (2015, 2026), "imageneta": (1993, 2015, 2026)}
BASELINE_FAMILY = "qsd2"  # RSIAT of round A2, same seeds, replicate 0
OBJECTIVE = {"holevo_vn": "vn", "holevo_mcr2": "mcr2", "holevo_spread": "spread", "hp_full": "vn",
             "hp_povm": "vn", "hp_twin": "mcr2", "hp_twin_povm": "mcr2", "hp_diag": "vn"}
SPECS = {
    "A (Holevo-CIL)": {"primary": "holevo_vn", "twin": "holevo_mcr2", "families": ["holevo_vn", "holevo_mcr2",
                                                                                   "holevo_spread"]},
    "A+B (Holevo-POVM)": {"primary": "hp_full", "twin": "hp_twin", "diag": "hp_diag",
                          "families": ["hp_full", "hp_povm", "hp_twin", "hp_twin_povm", "hp_diag"]},
}
CONT = re.compile(r"Continual metrics after task (\d+): BWT=([-\d.]+), forgetting=([-\d.]+), task_acc=(\[.*\])")
LAMBDA = re.compile(r"HOLEVO lambda (\{.*\})")
ORACLE = re.compile(r"HOLEVO oracle task=(\d+) phase=(start|end) (\{.*\})")
EPOCH = re.compile(r"HOLEVO epoch task=(\d+) epoch=(\d+) (\{.*\})")
MECH = re.compile(r"QSD mech head=linear: (\{.*\})")


def pair_ratio_from(stats_dir, task):
    path = Path(stats_dir) / "predictions_task_{}.npz".format(task)
    if not path.is_file():
        return None, None
    with np.load(path) as z:
        pred, target = z["pred"], z["target"]
    classes = int(target.max()) + 1
    sizes = [classes // (task + 1)] * (task + 1)
    m = head_metrics(pred, target, sizes, classes - sizes[-1])
    return m["pair_ratio"], m


def baseline_from_stats(folder):
    """Recompute BASE from the noise runs' predictions (exps/holevo/README.md 3.5)."""
    ratios, finals, last3 = [], [], []
    for stats in sorted(glob.glob(str(Path(folder) / "noise*_stats"))):
        last = max(int(re.search(r"(\d+)\.npz$", p).group(1)) for p in glob.glob(stats + "/predictions_task_*.npz"))
        ratio, m = pair_ratio_from(stats, last)
        ratios.append(ratio)
        finals.append(m["top1"])
        last3.append(statistics.mean(m["task_accuracies"][-3:]))
    return {"n": len(ratios), "final": statistics.mean(finals), "pair_ratio": statistics.mean(ratios),
            "pair_ratio_sd": statistics.stdev(ratios) if len(ratios) > 1 else None, "last3": statistics.mean(last3)}


def parse(log):
    match = RUN_NAME.match(log.stem)
    if not match:
        return None
    family, replicate = split_family(match.group("prefix"))
    if replicate != 0 or (family not in OBJECTIVE and family != BASELINE_FAMILY):
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    curves = [json.loads(v) for v in CNN.findall(text)]
    if not curves:
        return None
    cnn = curves[-1]
    run = {"family": family, "dataset": log.parent.parent.parent.name, "seed": int(match.group("seed")),
           "log": str(log), "tasks": len(cnn), "cnn": cnn, "aia": statistics.mean(cnn), "final": cnn[-1]}
    rows = CONT.findall(text)
    if rows:
        _, _, forgetting, row = rows[-1]
        row = json.loads(row)
        run.update(forgetting=float(forgetting), row=row, task0=row[0], task1=row[1] if len(row) > 1 else None,
                   last3=statistics.mean(row[-3:]))
    ratio, _ = pair_ratio_from(str(log)[:-len(".log")] + "_stats", len(cnn) - 1)
    if ratio is None:
        mech = MECH.findall(text)
        ratio = json.loads(mech[-1]).get("pair_ratio") if mech else None
    run["pair_ratio"] = ratio
    run["qsd"] = {head: json.loads(v)[-1] for head, v in CURVE.findall(text)}
    lam = LAMBDA.findall(text)
    run["lambda"] = json.loads(lam[-1]) if lam else None
    oracle = {}
    for task, phase, report in ORACLE.findall(text):
        oracle.setdefault(int(task), {})[phase] = json.loads(report)
    run["oracle"] = oracle
    epochs = {}
    for task, _, stats in EPOCH.findall(text):
        epochs[int(task)] = json.loads(stats)  # keeps the last epoch of every task
    run["last_epoch"] = epochs
    return run


def collect(roots):
    runs = {}
    for root in roots:
        for log in sorted(Path(root).rglob("*.log")):
            run = parse(log)
            if run:
                key = (run["family"], run["dataset"], run["seed"])
                if key not in runs or run["tasks"] > runs[key]["tasks"]:
                    runs[key] = run
    return runs


# ----------------------------------------------------------------------------- rules
def rule_g(run):
    """Median over tasks of the change of chi_oracle and of the stale objective within each task."""
    kind = OBJECTIVE[run["family"]]
    d_or, d_st = [], []
    for task, phases in sorted(run["oracle"].items()):
        if "start" in phases and "end" in phases:
            d_or.append(phases["end"]["oracle"][kind] - phases["start"]["oracle"][kind])
            d_st.append(phases["end"]["stale"][kind] - phases["start"]["stale"][kind])
    if not d_or:
        return None
    med_or, med_st = statistics.median(d_or), statistics.median(d_st)
    last = run["oracle"][max(run["oracle"])].get("end", {})
    return {"objective": kind, "median_oracle": med_or, "median_stale": med_st, "passes": med_or > 0,
            "gamed": med_st > 0 and med_or <= 0, "I_orc": last.get("I_orc"),
            "trace_distance": last.get("trace_distance"),
            "spearman_hint": [round(a, 5) for a in d_or]}


def screen(run, diag=None):
    out = {"R": run["cnn"][0] == BASE["task0"]}
    g = rule_g(run)
    out["G_detail"] = g
    out["G"] = None if g is None else g["passes"]
    if g is not None and diag is not None and g.get("I_orc") is not None:
        diag_g = rule_g(diag)
        if diag_g and diag_g.get("I_orc") is not None:
            out["I_orc_vs_diag"] = g["I_orc"] - diag_g["I_orc"]
            out["G"] = g["passes"] and g["I_orc"] < diag_g["I_orc"]
    pr = run.get("pair_ratio")
    out["M"] = None if pr is None else pr <= BASE["pair_ratio"] - DELTA_M
    out["delta_final"] = run["final"] - BASE["final"]
    out["O"] = out["delta_final"] >= MIN_O
    out["P"] = None if "last3" not in run else run["last3"] >= BASE["last3"] - MAX_P
    if diag is not None and "pgm_stats" in run["qsd"] and "pgm_stats" in diag["qsd"]:
        out["delta_pgm"] = run["qsd"]["pgm_stats"] - diag["qsd"]["pgm_stats"]
        out["D"] = out["delta_pgm"] >= MIN_D
    keys = ("R", "G", "M", "O", "P")
    out["go"] = all(out[k] is True for k in keys)
    return out


def paired(runs, family, rival, value=lambda r: r["final"], rival_value=None):
    rival_value = rival_value or value
    diffs = []
    for (fam, ds, seed), run in runs.items():
        if fam != family or seed not in HELD_OUT.get(ds, ()):
            continue
        other = runs.get((rival, ds, seed)) if rival else run
        if other is None:
            continue
        a, b = value(run), rival_value(other)
        if a is not None and b is not None:
            diffs.append(a - b)
    return interval(diffs)


def held_out(runs, spec):
    expected = sum(len(s) for s in HELD_OUT.values())
    primary = spec["primary"]
    out = {"H": paired(runs, primary, BASELINE_FAMILY), "Q": paired(runs, primary, spec["twin"])}
    if primary == "hp_full":
        out["Q_dec"] = paired(runs, primary, None, lambda r: r["qsd"].get("pgm_stats"),
                              lambda r: r["qsd"].get("ridge_stats"))
    for key, entry in list(out.items()):
        complete = entry["n"] == expected
        if key == "H":
            out["H_holds"] = (entry["mean"] >= MIN_H and entry["ci95"][0] > 0) if complete and "ci95" in entry else None
        else:
            out[key + "_holds"] = (entry["ci95"][0] > 0) if complete and "ci95" in entry else None
    out["expected"] = expected
    return out


# ----------------------------------------------------------------------------- report
def verdict(value):
    return "chưa có" if value is None else ("ĐẠT" if value else "TRƯỢT")


def render(runs):
    lines = ["# Holevo-CIL (A) và Holevo-POVM (A+B)", "",
             "Baseline sàng lọc (4 run `noise`, ImageNet-R 1993): final {:.3f}, pair ratio {:.4f} ± {:.4f} "
             "(δ_M = {:.3f}), trung bình 3 task cuối {:.3f}, task 0 {:.2f}.".format(
                 BASE["final"], BASE["pair_ratio"], BASE["pair_ratio_sd"], DELTA_M, BASE["last3"], BASE["task0"]),
             "", "## Từng run", "",
             "| Dataset | Seed | Run | AIA | Final | Forgetting ↓ | Task 0 | Task 1 | 3 task cuối | Pair ratio | PGM | Ridge | λ |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    order = lambda key: (ORDER.index(key[1]) if key[1] in ORDER else 9, key[2], key[0])
    for key in sorted(runs, key=order):
        r = runs[key]
        lam = ", ".join("{} {:.3g}".format(n, v["lambda"]) for n, v in (r["lambda"] or {}).items())
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            NAMES.get(r["dataset"], r["dataset"]), r["seed"], r["family"], fmt(r["aia"], 2), fmt(r["final"], 2),
            fmt(r.get("forgetting"), 2), fmt(r.get("task0"), 1), fmt(r.get("task1"), 1), fmt(r.get("last3"), 2),
            fmt(r.get("pair_ratio"), 3), fmt(r["qsd"].get("pgm_stats"), 2), fmt(r["qsd"].get("ridge_stats"), 2), lam))
    decisions = {}
    for title, spec in SPECS.items():
        lines += ["", "## " + title, "", "### Sàng lọc (ImageNet-R 1993, so với baseline `noise`)", ""]
        diag = runs.get((spec.get("diag"), *TUNING)) if spec.get("diag") else None
        found = False
        decisions[title] = {"screen": {}}
        for family in spec["families"]:
            run = runs.get((family, *TUNING))
            if run is None:
                continue
            found = True
            s = screen(run, diag if family != spec.get("diag") else None)
            decisions[title]["screen"][family] = s
            g = s["G_detail"] or {}
            parts = ["R {}".format(verdict(s["R"])),
                     "G {} (median Δχ_oracle {}, Δχ_stale {}{}{})".format(
                         verdict(s["G"]), fmt(g.get("median_oracle"), 4), fmt(g.get("median_stale"), 4),
                         ", BỊ LÁCH" if g.get("gamed") else "",
                         ", I_orc − diag {}".format(fmt(s["I_orc_vs_diag"], 4)) if "I_orc_vs_diag" in s else ""),
                     "M {} (pair ratio {} ≤ {:.3f})".format(verdict(s["M"]), fmt(run.get("pair_ratio"), 3),
                                                          BASE["pair_ratio"] - DELTA_M),
                     "O {} (Δ final {})".format(verdict(s["O"]), fmt(s["delta_final"], 2)),
                     "P {} (3 task cuối {} ≥ {:.2f})".format(verdict(s["P"]), fmt(run.get("last3"), 2),
                                                           BASE["last3"] - MAX_P)]
            if "D" in s:
                parts.append("D {} (Δ PGM {})".format(verdict(s["D"]), fmt(s["delta_pgm"], 2)))
            tag = " **(chính)**" if family == spec["primary"] else ""
            lines.append("- `{}`{}: ".format(family, tag) + "; ".join(parts))
            if g.get("trace_distance"):
                lines.append("  - khoảng cách vết (trạng thái ‖v‖² so với thật), task cuối: trung bình {}, max {}".format(
                    fmt(g["trace_distance"]["mean"], 3), fmt(g["trace_distance"]["max"], 3)))
        if not found:
            lines.append("- Chưa có run sàng lọc.")
        else:
            prim = decisions[title]["screen"].get(spec["primary"])
            if prim is not None:
                go = prim["go"]
                lines.append("- **Kết luận sàng lọc của `{}`: {}**".format(
                    spec["primary"], "đi tiếp sang held-out" if go else
                    ("theo luật: M đạt mà O trượt → chạy lưới siêu tham số" if prim["M"] and not prim["O"] and
                     prim["G"] is not False and prim["P"] is not False else "DỪNG hoặc xem luật trượt ở trên")))
            twin = runs.get((spec["twin"], *TUNING))
            if prim is not None and twin is not None:
                lines.append("- Tham khảo (không phải luật): {} − {} final = {}".format(
                    spec["primary"], spec["twin"], fmt(runs[(spec["primary"], *TUNING)]["final"] - twin["final"], 2)))
        h = held_out(runs, spec)
        decisions[title]["held_out"] = h
        lines += ["", "### Held-out (n = {}: ImageNet-R 2015, 2026; ImageNet-A 1993, 2015, 2026)".format(h["expected"]),
                  ""]
        labels = {"H": "H: Δ final {} − RSIAT (≥ {:g}, CI > 0)".format(spec["primary"], MIN_H),
                  "Q": "Q: Δ final {} − {} (CI > 0)".format(spec["primary"], spec["twin"]),
                  "Q_dec": "Q_dec: PGM − ridge trên biểu diễn {} (CI > 0)".format(spec["primary"])}
        for key, label in labels.items():
            if key in h:
                e = h[key]
                lines.append("- {}: {} {} (n={}) → **{}**".format(
                    label, fmt(e["mean"]), fmt_ci(e), e["n"],
                    verdict(h[key + "_holds"]) if h[key + "_holds"] is not None else
                    "chưa đủ run ({}/{})".format(e["n"], h["expected"])))
    checks = ["{} {} {}: task 0 {}".format(k[0], NAMES.get(k[1], k[1]), k[2], fmt(r["cnn"][0], 2))
              for k, r in sorted(runs.items(), key=lambda kv: order(kv[0])) if k[0] != BASELINE_FAMILY]
    lines += ["", "**Kiểm tra hồi quy (task 0 phải = {:.2f}):** {}".format(BASE["task0"], "; ".join(checks) or
                                                                           "không có run"), ""]
    return "\n".join(lines) + "\n", decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--json")
    parser.add_argument("--check-baseline")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    if args.check_baseline:
        print(json.dumps({"recomputed": baseline_from_stats(args.check_baseline), "registered": BASE}, indent=2))
        return
    if not args.root:
        parser.error("--root is required")
    runs = collect(args.root)
    text, decisions = render(runs)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        slim = [{k: v for k, v in r.items() if k not in ("cnn", "row")} for r in runs.values()]
        Path(args.json).write_text(json.dumps({"runs": slim, "decisions": decisions}, indent=2, default=str),
                                   encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
