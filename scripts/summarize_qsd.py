"""Summarise QSD-CIL stage A: all heads are evaluated inside the same training run.

Head differences within a run carry no training noise; replicates and seeds give
the run-level spread for the t interval. Decision rules are pre-registered in
exps/qsd/README.md and fixed below.

  python scripts/summarize_qsd.py --root <logs> [--root <logs> ...] \
      [--reference-root <plain RSIAT logs>] [--output summary.md] [--json summary.json]
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_level_stats import RUN_NAME, split_family  # noqa: E402
from scripts.summarize_ca_heads import fmt, fmt_ci, interval, reference_curves  # noqa: E402

HEAD_CURVE = re.compile(r"QSD curve head=(\w+): (\[.*\])")
HEAD_TASKS = re.compile(r"QSD tasks head=(\w+): (\[.*\])")
HEAD_MECH = re.compile(r"QSD mech head=(\w+): (\{.*\})")
CNN_CURVE = re.compile(r"CNN top1 curve: (\[.*\])")

# Pre-registered (exps/qsd/README.md). Do not edit after seeing results.
MIN_RUNS = 3
MIN_GAIN = 0.3
ORDER = ("linear", "linear_rerun", "linear_noscale", "ridge_stats", "pgm_stale", "pgm_channel",
         "pgm_channel_hier", "pgm_fuse", "pgm_nonlinear")
PRIMARY = ("pgm_channel", "pgm_channel_hier", "pgm_fuse")
CLAIMS = (
    ("pgm_channel", "pgm_stale", "Kênh vận chuyển có tác dụng"),
    ("pgm_nonlinear", "pgm_channel", "Mã hoá phi tuyến (hai bản sao) có lợi"),
    ("linear_noscale", "linear_rerun", "Bỏ heuristic co mean của CA có lợi"),
    ("ridge_stats", "linear", "Ridge giải tích trên thống kê có lợi"),
)
INFO = (("linear_rerun", "linear", "Nhiễu lấy mẫu của chính CA (thông tin)"),)
METRICS = ("aia", "final", "task0", "task1", "task_prediction", "pair_ratio")


def parse_log(log):
    match = RUN_NAME.match(log.stem)
    if not match:
        return None
    curves, tasks, mech, cnn = {}, {}, {}, None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        for pattern, store in ((HEAD_CURVE, curves), (HEAD_TASKS, tasks), (HEAD_MECH, mech)):
            m = pattern.search(line)
            if m:
                store[m.group(1)] = json.loads(m.group(2))
        m = CNN_CURVE.search(line)
        if m:
            cnn = json.loads(m.group(1))
    if not curves:
        return None
    family, replicate = split_family(match.group("prefix"))
    heads = {}
    for head, curve in curves.items():
        per_task, m = tasks.get(head, []), mech.get(head, {})
        heads[head] = {"curve": curve, "aia": sum(curve) / len(curve), "final": curve[-1],
                       "task0": per_task[0] if per_task else None,
                       "task1": per_task[1] if len(per_task) > 1 else None,
                       "task_prediction": m.get("task_prediction"), "pair_ratio": m.get("pair_ratio")}
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


def contrast(group, head, rival, metric="aia"):
    return interval([r["heads"][head][metric] - r["heads"][rival][metric] for r in group
                     if head in r["heads"] and rival in r["heads"]
                     and r["heads"][head][metric] is not None and r["heads"][rival][metric] is not None])


def passes(entry, min_gain=0.0):
    return entry["n"] >= MIN_RUNS and "ci95" in entry and entry["ci95"][0] > 0 and entry["mean"] >= min_gain


def verdicts(group):
    primary = {}
    for head in PRIMARY:
        if not all(head in r["heads"] for r in group):
            continue
        vs_linear, vs_ridge = contrast(group, head, "linear"), contrast(group, head, "ridge_stats")
        if vs_linear["n"] < MIN_RUNS:
            verdict = "chưa đủ run (cần ≥ {})".format(MIN_RUNS)
        elif passes(vs_linear, MIN_GAIN) and passes(vs_ridge):
            verdict = "ĐẠT: thắng linear (CI > 0, Δ ≥ {}) và thắng ridge_stats".format(MIN_GAIN)
        elif passes(vs_linear, MIN_GAIN):
            verdict = "thắng linear nhưng không thắng ridge_stats"
        elif "ci95" in vs_linear and vs_linear["ci95"][1] < 0:
            verdict = "KÉM HƠN linear"
        else:
            verdict = "không kết luận"
        primary[head] = {"vs_linear": vs_linear, "vs_ridge": vs_ridge,
                         "final": contrast(group, head, "linear", "final"),
                         "task0": contrast(group, head, "linear", "task0"), "verdict": verdict}
    claims = {}
    for head, rival, label in CLAIMS + INFO:
        if all(head in r["heads"] and rival in r["heads"] for r in group):
            c = contrast(group, head, rival)
            c["label"] = label
            c["holds"] = passes(c) if (head, rival, label) in CLAIMS else None
            claims["{} − {}".format(head, rival)] = c
    return primary, claims


def render_group(title, group, references):
    lines = ["## " + title, "",
             "| Head | AIA | Final | Task 0 | Task 1 | Dự đoán task (%) | Tỉ số nhầm khác/trong task |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for head in ORDER:
        if not all(head in r["heads"] for r in group):
            continue
        row = []
        for metric in METRICS:
            values = [r["heads"][head][metric] for r in group if r["heads"][head][metric] is not None]
            row.append(fmt(statistics.mean(values), 3 if metric == "pair_ratio" else 2) if values else "")
        lines.append("| {} | {} |".format(head, " | ".join(row)))
    primary, claims = verdicts(group)
    lines += ["", "### Phương pháp (so trong cùng run)", "",
              "| Head | n | Δ AIA vs linear | CI 95% | Δ AIA vs ridge_stats | CI 95% | Δ final | Δ task 0 | Kết luận |",
              "|---|---:|---:|---|---:|---|---:|---:|---|"]
    for head, p in primary.items():
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            head, p["vs_linear"]["n"], fmt(p["vs_linear"]["mean"]), fmt_ci(p["vs_linear"]),
            fmt(p["vs_ridge"]["mean"]), fmt_ci(p["vs_ridge"]), fmt(p["final"]["mean"]),
            fmt(p["task0"]["mean"]), p["verdict"]))
    lines += ["", "### Các khẳng định thành phần (Δ AIA)", "", "| So sánh | Nội dung | n | Δ | CI 95% | Giữ được? |",
              "|---|---|---:|---:|---|---|"]
    for name, c in claims.items():
        holds = "" if c["holds"] is None else ("có" if c["holds"] else "không")
        lines.append("| {} | {} | {} | {} | {} | {} |".format(name, c["label"], c["n"], fmt(c["mean"]),
                                                             fmt_ci(c), holds))
    checks = []
    for r in group:
        linear = r["heads"].get("linear", {}).get("curve")
        same = linear is not None and r["cnn_curve"] is not None and linear == r["cnn_curve"]
        ref = [fam for fam, curve in references.get((r["dataset"], r["seed"], r["replicate"]), [])
               if curve == r["cnn_curve"]]
        checks.append("seed {} rep {}: head linear {} CNN curve; trùng baseline: {}".format(
            r["seed"], r["replicate"], "=" if same else "≠", ", ".join(ref) if ref else "không có"))
    lines += ["", "**Kiểm tra hồi quy:** " + "; ".join(checks), ""]
    return lines, {"primary": primary, "claims": claims}


def render(runs, references):
    lines = ["# QSD-CIL giai đoạn A: so sánh trong cùng run", ""]
    summary = {}
    groups, pooled = {}, {}
    for run in runs:
        groups.setdefault((run["dataset"], run["family"], run["seed"]), []).append(run)
        pooled.setdefault((run["dataset"], run["family"]), []).append(run)
    for (dataset, family, seed), group in sorted(groups.items()):
        group.sort(key=lambda r: r["replicate"])
        block, data = render_group("{} | {} | seed {} | {} run".format(dataset, family, seed, len(group)),
                                   group, references)
        lines += block
        summary["{}|{}|{}".format(dataset, family, seed)] = data
    for (dataset, family), group in sorted(pooled.items()):
        if len({r["seed"] for r in group}) > 1:
            group.sort(key=lambda r: (r["seed"], r["replicate"]))
            block, data = render_group("{} | {} | gộp mọi seed | {} run (xác nhận)".format(dataset, family, len(group)),
                                       group, references)
            lines += block
            summary["{}|{}|pooled".format(dataset, family)] = data
    return "\n".join(lines) + "\n", summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--reference-root", action="append", default=[])
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
