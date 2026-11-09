"""Summarize SSCA/oracle diagnostic runs from their logs and stats reports.

Reports only what completed runs wrote; missing values stay empty. Oracle rows
read old-class images and are upper bounds, not exemplar-free results. The
scenario verdict applies the thresholds fixed in exps/diagnostics/README.md.
"""
import argparse
import ast
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.stats_diagnostics import binomial_two_sided, mcnemar

CURVE = re.compile(r"CNN top1 curve: (\[.*\])")
GROUPED = re.compile(r"CNN: (\{.*\})")
RUN_NAME = re.compile(r"^(?P<prefix>.+?)_(?P<seed>\d+)_(?P<convnet>.+)$")
ESTIMATORS = ("stale", "legacy", "paired", "applied")
# Pre-registered thresholds (percentage points of AIA); see exps/diagnostics/README.md.
HEADROOM_LARGE, HEADROOM_SMALL, MIN_EFFECT, ALPHA = 2.0, 0.5, 0.3, 0.05
ROLES = {"paired": "diag_paired", "oracle_mean": "diag_oracle_mean",
         "oracle_mean_cov": "diag_oracle_mean_cov"}
REPLICATE_PREFIX = "diag_legacy_rep"


def parse_log(path):
    curve, grouped = None, []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = CURVE.search(line)
        if match:
            curve = ast.literal_eval(match.group(1))
        match = GROUPED.search(line)
        if match:
            grouped.append(ast.literal_eval(match.group(1)))
    return curve, grouped


def get(value, *keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def task_index(path):
    return int(path.stem.rsplit("_", 1)[1])


def error_decomposition(prediction_files):
    """Final-task error split into within-task / to-older-task / to-newer-task (% of test images)."""
    if not prediction_files:
        return None
    starts = sorted({int(np.load(p)["known_classes"]) for p in prediction_files})
    final = np.load(prediction_files[-1])
    pred, target = final["pred"].astype(int), final["target"].astype(int)
    task_of = lambda labels: np.searchsorted(starts, labels, side="right") - 1
    wrong = pred != target
    tp, tt = task_of(pred), task_of(target)
    pct = lambda mask: float(mask.mean() * 100)
    return {"error": pct(wrong), "within_task": pct(wrong & (tp == tt)),
            "to_older_task": pct(wrong & (tp < tt)), "to_newer_task": pct(wrong & (tp > tt))}


def collect(root):
    runs = []
    for log in sorted(Path(root).rglob("*.log")):
        match = RUN_NAME.match(log.stem)
        if not match:
            continue
        curve, grouped = parse_log(log)
        stats_dir = log.with_name(log.stem + "_stats")
        if not stats_dir.is_dir():
            stats_dir = log.with_name(log.stem + "_qbridge")
        tasks = []
        for path in sorted(stats_dir.glob("stats_task_*.json"), key=task_index):
            report = json.loads(path.read_text(encoding="utf-8"))
            errors = report["estimator_mean_errors"]
            task = report["task"]
            post = grouped[task] if task < len(grouped) else {}
            pre = report.get("pre_ca_accuracy") or {}
            tasks.append({
                "task": task,
                **{"rel_" + name: get(errors, name, "all", "relative", "median") for name in ESTIMATORS},
                "rel_paired_origin0": get(errors, "paired", "by_origin_task", "0", "relative", "median"),
                "rel_legacy_origin0": get(errors, "legacy", "by_origin_task", "0", "relative", "median"),
                "cov_rel": get(report, "stored_covariance_errors", "all", "relative_frobenius", "median"),
                "drift_rel_norm": get(report, "oracle_drift", "all", "relative_to_oracle_norm", "median"),
                "ess": get(report, "ssca_kernel", "effective_sample_size_mean"),
                "floor_rows": get(report, "ssca_kernel", "fraction_rows_floor_dominated"),
                "pre_total": pre.get("total"), "pre_old": pre.get("old"), "pre_new": pre.get("new"),
                "pre_old_to_new": pre.get("old_to_new_error_rate"),
                "post_total": post.get("total"), "post_old": post.get("old"), "post_new": post.get("new"),
            })
        predictions = sorted(stats_dir.glob("predictions_task_*.npz"), key=task_index)
        runs.append({
            "log": str(log), "prefix": match.group("prefix"), "seed": int(match.group("seed")),
            "completed_tasks": len(curve) if curve else 0,
            "aia": sum(curve) / len(curve) if curve else None,
            "final": curve[-1] if curve else None,
            "final_old": grouped[-1].get("old") if grouped else None,
            "final_new": grouped[-1].get("new") if grouped else None,
            "final_predictions": str(predictions[-1]) if predictions else None,
            "errors": error_decomposition([str(p) for p in predictions]),
            "curve": curve, "tasks": tasks,
        })
    return runs


def fmt(value, digits=3):
    return "" if value is None else "{:.{}f}".format(value, digits)


def matched(runs, prefix, reference):
    """(run, reference run) pairs with the same seed and number of completed tasks."""
    by_key = {(r["prefix"], r["seed"]): r for r in runs if r["aia"] is not None}
    pairs = []
    for (p, seed), run in sorted(by_key.items()):
        ref = by_key.get((reference, seed))
        if p == prefix and ref is not None and run["completed_tasks"] == ref["completed_tasks"]:
            pairs.append((run, ref))
    return pairs


def paired_test(pairs):
    """Per-seed McNemar on final-task predictions plus the pooled stratified test."""
    per_seed, b_total, c_total = {}, 0, 0
    for run, ref in pairs:
        if not run["final_predictions"] or not ref["final_predictions"]:
            continue
        a, r = np.load(run["final_predictions"]), np.load(ref["final_predictions"])
        if not np.array_equal(a["target"], r["target"]):
            raise ValueError("Test images differ between {} and {}".format(run["log"], ref["log"]))
        # b: only the run is correct; c: only the reference is correct.
        result = mcnemar(a["pred"] == a["target"], r["pred"] == r["target"])
        per_seed[run["seed"]] = result
        b_total += result["only_a_correct"]
        c_total += result["only_b_correct"]
    if not per_seed:
        return None
    return {"per_seed": per_seed, "only_run_correct": b_total, "only_reference_correct": c_total,
            "p_value": binomial_two_sided(b_total, c_total)}


def contrast(runs, prefix, reference):
    pairs = matched(runs, prefix, reference)
    if not pairs:
        return None
    deltas = {key: [run[key] - ref[key] for run, ref in pairs] for key in ("aia", "final")}
    return {"prefix": prefix, "seeds": [run["seed"] for run, _ in pairs],
            **{key: statistics.mean(v) for key, v in deltas.items()},
            **{key + "_sd": statistics.stdev(v) if len(v) > 1 else None for key, v in deltas.items()},
            "per_seed_aia": deltas["aia"], "mcnemar": paired_test(pairs)}


def noise_floor(replicates):
    deltas = [abs(d) for c in replicates for d in c["per_seed_aia"]]
    return statistics.mean(deltas) if deltas else None


def judge(c, threshold):
    big = abs(c["aia"]) > threshold
    test = c["mcnemar"]
    significant = test is not None and test["p_value"] < ALPHA
    if big and significant:
        return "**vượt sàn nhiễu + McNemar p<0.05**"
    if big:
        return "vượt sàn nhiễu (McNemar chưa có ý nghĩa)"
    return "trong sàn nhiễu" + (" (McNemar p<0.05)" if significant else "")


def verdict(contrasts):
    """Map measured contrasts to the pre-registered scenarios A-D."""
    lines = []
    noise = noise_floor(contrasts.get("replicates", []))
    threshold = max(MIN_EFFECT, 2 * noise) if noise is not None else MIN_EFFECT
    if noise is None:
        lines.append("- Sàn nhiễu: **chưa đo** (thiếu `diag_legacy_rep*`); dùng ngưỡng tối thiểu "
                     "{:.1f} pp, kết luận yếu hơn.".format(MIN_EFFECT))
    else:
        lines.append("- Sàn nhiễu SGD (|replicate − legacy| AIA trung bình): **{:.3f} pp** → "
                     "hiệu ứng chỉ được tính khi |Δ AIA| > {:.3f} pp.".format(noise, threshold))

    def real(name):
        c = contrasts.get(name)
        if c is None:
            return None
        big = abs(c["aia"]) > threshold
        test = c["mcnemar"]
        significant = test is not None and test["p_value"] < ALPHA
        return big, significant

    full, mean_only = contrasts.get("oracle_mean_cov"), contrasts.get("oracle_mean")
    paired = contrasts.get("paired")
    if full is None:
        lines.append("- **Chưa đủ run oracle** để xếp kịch bản.")
        return lines, None
    headroom = full["aia"]
    lines.append("- Headroom thống kê lớp cũ (oracle_mean_cov − legacy): **{:+.3f} pp AIA**".format(headroom))
    if mean_only is not None:
        lines.append("  - phần do mean: {:+.3f} pp; phần do covariance: {:+.3f} pp".format(
            mean_only["aia"], headroom - mean_only["aia"]))
    paired_state = real("paired")
    if paired is not None:
        test = paired["mcnemar"]
        lines.append("- Paired − legacy: {:+.3f} pp AIA; McNemar final-task p = {} → {}".format(
            paired["aia"], fmt(test["p_value"], 4) if test else "n/a",
            "vượt sàn nhiễu" if paired_state and paired_state[0] else "trong sàn nhiễu"))
    paired_works = bool(paired_state and paired_state[0])
    if headroom >= HEADROOM_LARGE:
        scenario = "A" if paired_works else "B"
    elif headroom >= HEADROOM_SMALL:
        scenario = "C"
    else:
        scenario = "D"
    return lines, scenario


SCENARIOS = {
    "A": "Thống kê lớp cũ là chỗ nghẽn và SSCA ghép cặp đã lấy lại một phần → đi tiếp: covariance transport, Q-SSCA.",
    "B": "Thống kê lớp cũ là chỗ nghẽn nhưng ghép cặp chưa đủ → xem ESS/σ, phần covariance, transport bằng T; Q-SSCA là ứng viên chính.",
    "C": "Headroom vừa phải → chọn phần lớn hơn giữa mean và covariance; cân nhắc kết hợp với hướng representation.",
    "D": "Thống kê lớp cũ KHÔNG phải chỗ nghẽn → đổi hướng sang representation/plasticity (xem bảng trước/sau CA).",
}


def render(runs, reference):
    lines = ["# SSCA / oracle diagnostics", "",
             "Oracle prefixes read old-class training images: upper bounds only.", "",
             "| Run | Seed | Tasks | AIA | Final | Final old | Final new |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for run in sorted(runs, key=lambda r: (r["prefix"], r["seed"])):
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            run["prefix"], run["seed"], run["completed_tasks"], fmt(run["aia"]),
            fmt(run["final"], 2), fmt(run["final_old"], 2), fmt(run["final_new"], 2)))

    contrasts = {}
    rows = []
    for prefix in sorted({r["prefix"] for r in runs} - {reference}):
        c = contrast(runs, prefix, reference)
        if c is None:
            continue
        rows.append(c)
        for role, name in ROLES.items():
            if name == prefix:
                contrasts[role] = c
        if prefix.startswith(REPLICATE_PREFIX):
            contrasts.setdefault("replicates", []).append(c)
    if rows:
        lines += ["", "## Paired deltas vs `{}`".format(reference), "",
                  "Mean +/- sample SD over matched seeds. McNemar: exact test on final-task per-image "
                  "correctness, pooled over seeds (b = only run correct, c = only reference correct).", "",
                  "| Run | Seeds | Delta AIA | Delta final | McNemar b / c | p |",
                  "|---|---|---:|---:|---:|---:|"]
        for c in rows:
            cell = lambda m, s: fmt(m) + ("" if s is None else " +/- " + fmt(s))
            test = c["mcnemar"]
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                c["prefix"], ",".join(map(str, c["seeds"])), cell(c["aia"], c["aia_sd"]),
                cell(c["final"], c["final_sd"]),
                "{} / {}".format(test["only_run_correct"], test["only_reference_correct"]) if test else "",
                fmt(test["p_value"], 4) if test else ""))

    verdict_lines, scenario = verdict(contrasts)
    lines += ["", "## Kịch bản theo quy tắc đặt trước", ""] + verdict_lines
    if scenario:
        lines += ["", "**Kịch bản {}:** {}".format(scenario, SCENARIOS[scenario]),
                  "", "Hành động chi tiết: mục 'Cây quyết định' trong `exps/diagnostics/README.md`."]

    noise = noise_floor(contrasts.get("replicates", []))
    threshold = max(MIN_EFFECT, 2 * noise) if noise is not None else MIN_EFFECT
    candidates = [c for c in rows if not c["prefix"].startswith(REPLICATE_PREFIX)
                  and c["prefix"] not in ROLES.values()]
    if candidates:
        lines += ["", "## Ứng viên phương pháp (ngưỡng |Δ AIA| > {:.3f} pp)".format(threshold), "",
                  "| Run | Δ AIA | Δ final | Đánh giá |", "|---|---:|---:|---|"]
        for c in candidates:
            lines.append("| {} | {:+.3f} | {:+.3f} | {} |".format(c["prefix"], c["aia"], c["final"], judge(c, threshold)))
    lines += ["", "## Phân rã lỗi task cuối (% ảnh test)", "",
              "Inter-task = to-older + to-newer. Ứng viên CA nhắm vào inter-task.", "",
              "| Run | Seed | Error | Within-task | To older task | To newer task |",
              "|---|---:|---:|---:|---:|---:|"]
    for run in sorted(runs, key=lambda r: (r["prefix"], r["seed"])):
        e = run["errors"]
        if e:
            lines.append("| {} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
                run["prefix"], run["seed"], e["error"], e["within_task"], e["to_older_task"], e["to_newer_task"]))

    lines += ["", "## Before vs after CA (test accuracy)", "",
              "| Run | Seed | Task | Pre-CA total | Pre-CA old | Pre-CA new | Pre-CA old→new err | "
              "Post-CA total | Post-CA old | Post-CA new |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in sorted(runs, key=lambda r: (r["prefix"], r["seed"])):
        for t in run["tasks"]:
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                run["prefix"], run["seed"], t["task"], fmt(t["pre_total"], 2), fmt(t["pre_old"], 2),
                fmt(t["pre_new"], 2), fmt(t["pre_old_to_new"]), fmt(t["post_total"], 2),
                fmt(t["post_old"], 2), fmt(t["post_new"], 2)))

    lines += ["", "## Old-mean estimation error per task",
              "", "Relative error = ||estimate - oracle|| / ||stale - oracle|| (median over old classes); "
              "1.0 means no better than skipping drift compensation.", "",
              "| Run | Seed | Task | Stale | Legacy | Paired | Applied | Legacy (task-0 classes) | "
              "Paired (task-0 classes) | Cov rel. error | Drift / norm | Kernel ESS | Floor rows |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in sorted(runs, key=lambda r: (r["prefix"], r["seed"])):
        for t in run["tasks"]:
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                run["prefix"], run["seed"], t["task"], fmt(t["rel_stale"]), fmt(t["rel_legacy"]),
                fmt(t["rel_paired"]), fmt(t["rel_applied"]), fmt(t["rel_legacy_origin0"]),
                fmt(t["rel_paired_origin0"]), fmt(t["cov_rel"]), fmt(t["drift_rel_norm"]),
                fmt(t["ess"], 1), fmt(t["floor_rows"], 2)))
    return "\n".join(lines) + "\n", scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True,
                        help="Directory containing run logs (searched recursively); repeatable")
    parser.add_argument("--reference", default="diag_legacy", help="Prefix used as the paired baseline")
    parser.add_argument("--output", help="Write the Markdown summary here")
    parser.add_argument("--json", help="Write the parsed records here")
    args = parser.parse_args()
    runs = [run for root in args.root for run in collect(root)]
    text, scenario = render(runs, args.reference)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps({"scenario": scenario, "runs": runs}, indent=2), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
