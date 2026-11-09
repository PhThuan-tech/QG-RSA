"""Summarise QRF-CA runs: every CA head is evaluated inside the same training run.

All heads of one run share the adapter, the class statistics and the test set,
so a head-minus-linear difference contains no training noise of its own; the
replicates (rng_replicate) supply the run-level spread for the t interval.

  python scripts/summarize_ca_heads.py --root <logs> [--reference-root <noise logs>] \
      [--output summary.md] [--json summary.json]
"""
import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_level_stats import RUN_NAME, split_family, t975  # noqa: E402

HEAD_CURVE = re.compile(r"CAHEAD curve head=(\w+): (\[.*\])")
HEAD_TASKS = re.compile(r"CAHEAD tasks head=(\w+): (\[.*\])")
CNN_CURVE = re.compile(r"CNN top1 curve: (\[.*\])")

# Pre-registered decision rules (exps/qrfca/README.md). Do not edit after seeing results.
MIN_RUNS = 3
MIN_GAIN = 0.3  # pp of AIA over the linear RSIAT head
NONLINEAR = ("qda", "rff_ridge", "rp_ridge", "pqk_ridge", "pqk_noent_ridge")
QUANTUM_RIVALS = ("rff_ridge", "rp_ridge", "qda", "pqk_noent_ridge")


def parse_log(log):
    match = RUN_NAME.match(log.stem)
    if not match:
        return None
    curves, tasks, cnn = {}, {}, None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = HEAD_CURVE.search(line)
        if m:
            curves[m.group(1)] = json.loads(m.group(2))
        m = HEAD_TASKS.search(line)
        if m:
            tasks[m.group(1)] = json.loads(m.group(2))
        m = CNN_CURVE.search(line)
        if m:
            cnn = json.loads(m.group(1))
    if not curves:
        return None
    family, replicate = split_family(match.group("prefix"))
    heads = {}
    for head, curve in curves.items():
        per_task = tasks.get(head, [])
        heads[head] = {"curve": curve, "aia": sum(curve) / len(curve), "final": curve[-1],
                       "task0": per_task[0] if per_task else None,
                       "task1": per_task[1] if len(per_task) > 1 else None}
    return {"dataset": log.parent.parent.parent.name, "family": family, "replicate": replicate,
            "seed": int(match.group("seed")), "tasks": max(len(c) for c in curves.values()),
            "cnn_curve": cnn, "heads": heads, "log": str(log)}


def collect(roots):
    runs = [r for root in roots for r in (parse_log(p) for p in sorted(Path(root).rglob("*.log"))) if r]
    longest = {}
    for run in runs:
        longest[run["dataset"]] = max(longest.get(run["dataset"], 0), run["tasks"])
    return [r for r in runs if r["tasks"] == longest[r["dataset"]]
            and all(len(h["curve"]) == r["tasks"] for h in r["heads"].values())]


def reference_curves(roots):
    """CNN curves of plain RSIAT runs, keyed by (dataset, seed, replicate)."""
    from scripts.run_level_stats import collect as collect_runs
    out = {}
    for run in collect_runs(roots):
        text = Path(run["log"]).read_text(encoding="utf-8", errors="replace")
        curve = [json.loads(m) for m in CNN_CURVE.findall(text)][-1]
        out.setdefault((run["dataset"], run["seed"], run["replicate"]), []).append((run["family"], curve))
    return out


def interval(values):
    entry = {"n": len(values), "mean": statistics.mean(values) if values else None}
    if len(values) >= 2:
        sd = statistics.stdev(values)
        half = t975(len(values) - 1) * sd / math.sqrt(len(values))
        entry.update(sd=sd, ci95=(entry["mean"] - half, entry["mean"] + half))
    return entry


def contrast(group, head, rival, metric="aia"):
    values = [r["heads"][head][metric] - r["heads"][rival][metric] for r in group
              if head in r["heads"] and rival in r["heads"] and r["heads"][head][metric] is not None
              and r["heads"][rival][metric] is not None]
    return interval(values)


def verdicts(group):
    primary, quantum = {}, {}
    for head in NONLINEAR:
        if not all(head in r["heads"] for r in group):
            continue
        c = {m: contrast(group, head, "linear", m) for m in ("aia", "final", "task0", "task1")}
        aia = c["aia"]
        if aia["n"] < MIN_RUNS:
            verdict = "chưa đủ run (cần ≥ {})".format(MIN_RUNS)
        elif aia["ci95"][0] > 0 and aia["mean"] >= MIN_GAIN:
            verdict = "ĐẠT: CI 95% > 0 và Δ AIA ≥ {}".format(MIN_GAIN)
        elif aia["ci95"][1] < 0:
            verdict = "KÉM HƠN head tuyến tính"
        elif aia["ci95"][0] > 0:
            verdict = "tốt hơn nhưng Δ AIA < {}".format(MIN_GAIN)
        else:
            verdict = "không kết luận: CI 95% chứa 0"
        c["verdict"] = verdict
        primary[head] = c
    if all("pqk_ridge" in r["heads"] for r in group):
        for rival in QUANTUM_RIVALS:
            if all(rival in r["heads"] for r in group):
                quantum[rival] = contrast(group, "pqk_ridge", rival)
        complete = len(quantum) == len(QUANTUM_RIVALS) and all(
            q["n"] >= MIN_RUNS and q["ci95"][0] > 0 for q in quantum.values())
        quantum["verdict"] = ("ĐẠT: pqk_ridge thắng mọi đối chứng (CI 95% > 0)" if complete
                              else "chưa chứng minh được đóng góp quantum")
    return primary, quantum


def fmt(value, digits=3):
    return "" if value is None else "{:.{}f}".format(value, digits)


def fmt_ci(entry):
    return "[{}, {}]".format(fmt(entry["ci95"][0]), fmt(entry["ci95"][1])) if "ci95" in entry else ""


def render(runs, references):
    lines = ["# QRF-CA: so sánh head CA trong cùng run", ""]
    groups = {}
    for run in runs:
        groups.setdefault((run["dataset"], run["family"], run["seed"]), []).append(run)
    summary = {}
    for (dataset, family, seed), group in sorted(groups.items()):
        group.sort(key=lambda r: r["replicate"])
        heads = [h for h in ("linear",) + NONLINEAR if all(h in r["heads"] for r in group)]
        lines += ["## {} | {} | seed {} | {} replicate".format(dataset, family, seed, len(group)), "",
                  "| Head | AIA (mean) | Final (mean) | Task 0 | Task 1 |", "|---|---:|---:|---:|---:|"]
        for head in heads:
            vals = {m: [r["heads"][head][m] for r in group if r["heads"][head][m] is not None]
                    for m in ("aia", "final", "task0", "task1")}
            lines.append("| {} | {} | {} | {} | {} |".format(
                head, *(fmt(statistics.mean(v)) if v else "" for v in vals.values())))
        primary, quantum = verdicts(group)
        lines += ["", "### Head phi tuyến so với head tuyến tính RSIAT (ghép trong cùng run)", "",
                  "| Head | n | Δ AIA | CI 95% | Δ final | Δ task 0 | Kết luận |",
                  "|---|---:|---:|---|---:|---:|---|"]
        for head, c in primary.items():
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                head, c["aia"]["n"], fmt(c["aia"]["mean"]), fmt_ci(c["aia"]), fmt(c["final"]["mean"]),
                fmt(c["task0"]["mean"]), c["verdict"]))
        if quantum:
            lines += ["", "### pqk_ridge so với các đối chứng (Δ AIA)", "",
                      "| Đối chứng | n | Δ AIA | CI 95% |", "|---|---:|---:|---|"]
            for rival, q in quantum.items():
                if rival != "verdict":
                    lines.append("| {} | {} | {} | {} |".format(rival, q["n"], fmt(q["mean"]), fmt_ci(q)))
            lines += ["", "**Kết luận quantum:** " + quantum["verdict"]]
        checks = []
        for r in group:
            linear = r["heads"].get("linear", {}).get("curve")
            ok_self = linear is not None and r["cnn_curve"] is not None and linear == r["cnn_curve"]
            ref = [fam for fam, curve in references.get((dataset, r["seed"], r["replicate"]), [])
                   if curve == r["cnn_curve"]]
            checks.append("rep {}: head linear {} CNN curve; trùng từng chữ số với baseline: {}".format(
                r["replicate"], "=" if ok_self else "≠", ", ".join(ref) if ref else "không có"))
        lines += ["", "**Kiểm tra hồi quy:** " + "; ".join(checks), ""]
        summary["{}|{}|{}".format(dataset, family, seed)] = {"primary": primary, "quantum": quantum}
    return "\n".join(lines) + "\n", summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--reference-root", action="append", default=[],
                        help="Plain RSIAT logs for the bit-identity check (e.g. noise replicates)")
    parser.add_argument("--output")
    parser.add_argument("--json")
    args = parser.parse_args()
    runs = collect(args.root)
    text, summary = render(runs, reference_curves(args.reference_root) if args.reference_root else {})
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({"runs": runs, "summary": summary}, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
