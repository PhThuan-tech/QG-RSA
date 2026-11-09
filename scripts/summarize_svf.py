"""Summarise snapshot-view fusion (SVF, exps/svf/README.md).

Every run of family ``svf`` (replicate 0) is parsed from its log. The fusion weight of
``fuse_snap_ridge`` and of ``fuse3_snap`` is chosen once on the tuning run (ImageNet-R
seed 1993) by final accuracy and then frozen; the decision rules use held-out runs only.
Rules are pre-registered in exps/svf/README.md and fixed below.

  python scripts/summarize_svf.py --root <logs> [--root <logs> ...] [--output summary.md] [--json summary.json]
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_level_stats import RUN_NAME, split_family  # noqa: E402
from scripts.summarize_ca_heads import fmt, fmt_ci, interval  # noqa: E402
from scripts.summarize_scf import CNN, CURVE, NAMES, ORDER, TASKS, continual  # noqa: E402
from utils.qsd import FUSE_WEIGHTS, fuse3_name, fuse_name  # noqa: E402

# Pre-registered (exps/svf/README.md). Do not edit after seeing results.
FAMILY = "svf"
TUNING = ("imagenetr", 1993)
HELD_OUT = {"imagenetr": (2015, 2026), "imageneta": (1993, 2015, 2026)}
MIN_GAIN = 1.0  # rule S: held-out final gain of SVF over RSIAT, with CI > 0
SCREEN_GAIN = 0.3  # stopping rule on the tuning run: fuse3 - SCF ridge at the chosen weight
SCF_HEAD = "fuse_ridge_stats_b0p25"
WEIGHT_FAMILIES = {"snap": lambda w: fuse_name("snap_ridge", w), "fuse3": fuse3_name}
MECH = re.compile(r"QSD mech head=(\w+): (\{.*\})")


def parse(log):
    match = RUN_NAME.match(log.stem)
    if not match:
        return None
    family, replicate = split_family(match.group("prefix"))
    if family != FAMILY or replicate != 0:
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    curves = {h: json.loads(v) for h, v in CURVE.findall(text)}
    if "linear" not in curves:
        return None
    rows, mech = {}, {}
    for head, values in TASKS.findall(text):
        rows.setdefault(head, []).append(json.loads(values))
    for head, values in MECH.findall(text):
        mech[head] = json.loads(values)  # keeps the last task
    cnn = [json.loads(v) for v in CNN.findall(text)]
    heads = {}
    for head, curve in curves.items():
        forgetting, _ = continual(rows[head]) if head in rows else (None, None)
        last = rows[head][-1] if head in rows else []
        heads[head] = {"aia": statistics.mean(curve), "final": curve[-1], "forgetting": forgetting, "curve": curve,
                       "task0": last[0] if last else None, "task1": last[1] if len(last) > 1 else None,
                       "pair_ratio": mech.get(head, {}).get("pair_ratio"),
                       "task_prediction": mech.get(head, {}).get("task_prediction")}
    return {"dataset": log.parent.parent.parent.name, "seed": int(match.group("seed")), "log": str(log),
            "tasks": len(curves["linear"]), "cnn_curve": cnn[-1] if cnn else None, "heads": heads}


def collect(roots):
    runs = {}
    for root in roots:
        for log in sorted(Path(root).rglob("*.log")):
            run = parse(log)
            if run:  # one run per (dataset, seed); keep the longest (completed) one
                key = (run["dataset"], run["seed"])
                if key not in runs or run["tasks"] > runs[key]["tasks"]:
                    runs[key] = run
    return list(runs.values())


def choose_weights(runs):
    """One weight per fusion family, by final accuracy on the tuning run (ties: smaller weight)."""
    tuning = [r for r in runs if (r["dataset"], r["seed"]) == TUNING]
    if not tuning:
        return None
    heads, chosen = tuning[0]["heads"], {}
    for family, name in WEIGHT_FAMILIES.items():
        options = [w for w in FUSE_WEIGHTS if name(w) in heads]
        if options:
            chosen[family] = name(max(options, key=lambda w: (heads[name(w)]["final"], -w)))
    return chosen


def delta(group, head, rival, metric="final"):
    return interval([r["heads"][head][metric] - r["heads"][rival][metric] for r in group
                     if head in r["heads"] and rival in r["heads"] and r["heads"][head][metric] is not None
                     and r["heads"][rival][metric] is not None])


def screen(runs, chosen):
    """Stopping rule on the tuning run (exps/svf/README.md): optimistic, used only to decide more seeds."""
    tuning = [r for r in runs if (r["dataset"], r["seed"]) == TUNING]
    if not tuning or not chosen or "fuse3" not in chosen or SCF_HEAD not in tuning[0]["heads"]:
        return None
    heads = tuning[0]["heads"]
    gap = lambda head: heads[head]["final"] - heads[SCF_HEAD]["final"]
    gain = gap(chosen["fuse3"])
    return {"gain": gain, "go": gain >= SCREEN_GAIN,
            "by_weight": {fuse3_name(w): gap(fuse3_name(w)) for w in FUSE_WEIGHTS if fuse3_name(w) in heads}}


def decide(runs, chosen):
    held = [r for r in runs if r["seed"] in HELD_OUT.get(r["dataset"], ())]
    expected = sum(len(seeds) for seeds in HELD_OUT.values())
    out = {"chosen": chosen, "held_out": len(held), "expected": expected, "holds": None}
    if not chosen or "snap" not in chosen:
        return out
    gain = delta(held, chosen["snap"], "linear")
    extra = delta(held, chosen["fuse3"], SCF_HEAD) if "fuse3" in chosen else interval([])
    out.update(gain=gain, over_scf=extra,
               per_dataset={ds: delta([r for r in held if r["dataset"] == ds], chosen["snap"], "linear")
                            for ds in HELD_OUT})
    if len(held) == expected:
        out["holds"] = ("ci95" in gain and gain["mean"] >= MIN_GAIN and gain["ci95"][0] > 0
                        and "ci95" in extra and extra["ci95"][0] > 0)
    return out


def render(runs):
    chosen = choose_weights(runs)
    methods = [("linear", "RSIAT"), (SCF_HEAD, "SCF ridge"), ("snap_ridge", "snapshot alone")]
    if chosen:
        methods += [(chosen[f], label) for f, label in (("snap", "SVF"), ("fuse3", "SVF + SCF")) if f in chosen]
    lines = ["# SVF: fusion view plastic + snapshot task 0", "",
             "Trọng số chọn trên run tuning ImageNet-R 1993: " +
             (", ".join("`{}`".format(h) for h in chosen.values()) if chosen else "**chưa có run tuning**"), "",
             "## Từng run", "",
             "| Dataset | Seed | Phương pháp | AIA | Final | Forgetting ↓ | Task 0 | Task 1 | Pair ratio |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    order = lambda r: (ORDER.index(r["dataset"]) if r["dataset"] in ORDER else 9, r["seed"])
    for run in sorted(runs, key=order):
        tag = " (tuning)" if (run["dataset"], run["seed"]) == TUNING else ""
        for head, label in methods:
            if head in run["heads"]:
                h = run["heads"][head]
                lines.append("| {} | {}{} | {} | {} | {} | {} | {} | {} | {} |".format(
                    NAMES.get(run["dataset"], run["dataset"]), run["seed"], tag, label, fmt(h["aia"], 2),
                    fmt(h["final"], 2), fmt(h["forgetting"], 2), fmt(h["task0"], 1), fmt(h["task1"], 1),
                    fmt(h["pair_ratio"], 2)))
    s = screen(runs, chosen)
    lines += ["", "## Luật dừng sàng lọc (run tuning, lạc quan)", ""]
    if s is None:
        lines.append("- Chưa có run tuning ImageNet-R 1993.")
    else:
        lines.append("- Δ final SVF+SCF − SCF ridge tại `{}`: {} (ngưỡng +{:g}) → **{}**".format(
            chosen["fuse3"], fmt(s["gain"], 2), SCREEN_GAIN,
            "đi tiếp (chạy held-out)" if s["go"] else "DỪNG, không chạy thêm seed"))
        lines.append("- Theo trọng số: " + ", ".join("{} {}".format(h, fmt(v, 2)) for h, v in s["by_weight"].items()))
    d = decide(runs, chosen)
    d["screen"] = s
    lines += ["", "## Quyết định (quy tắc đặt trước, chỉ run held-out)", ""]
    if "gain" not in d:
        lines.append("- Chưa có run tuning: chưa quyết định.")
    else:
        g, o = d["gain"], d["over_scf"]
        lines.append("- Δ final SVF − RSIAT: {} {} (n={}); {}".format(
            fmt(g["mean"]), fmt_ci(g), g["n"], ", ".join("{} {} (n={})".format(NAMES[ds], fmt(e["mean"]), e["n"])
                                                         for ds, e in d["per_dataset"].items())))
        lines.append("- Δ final SVF+SCF − SCF ridge: {} {} (n={})".format(fmt(o["mean"]), fmt_ci(o), o["n"]))
        verdict = ("chưa đủ run ({}/{})".format(d["held_out"], d["expected"]) if d["holds"] is None
                   else ("ĐẠT" if d["holds"] else "KHÔNG ĐẠT"))
        lines.append("- **Rule S** (gain ≥ {:g} với CI > 0, và phần thêm so với SCF có CI > 0): **{}**".format(
            MIN_GAIN, verdict))
    checks = ["{} {}: {}".format(NAMES.get(r["dataset"], r["dataset"]), r["seed"],
                                 "=" if r["heads"]["linear"]["curve"] == r["cnn_curve"] else "≠ (LỖI)")
              for r in sorted(runs, key=order)]
    lines += ["", "**Kiểm tra hồi quy (head linear = CNN curve):** " + ("; ".join(checks) or "không có run"), ""]
    return "\n".join(lines) + "\n", d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--output")
    parser.add_argument("--json")
    args = parser.parse_args()
    runs = collect(args.root)
    text, decision = render(runs)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        slim = [{k: v for k, v in r.items() if k != "cnn_curve"} for r in runs]
        Path(args.json).write_text(json.dumps({"runs": slim, "decision": decision}, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
