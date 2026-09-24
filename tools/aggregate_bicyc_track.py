#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path


ARMS = ("B0_official", "B1_forward", "B2_bidirectional", "B3_cycle")
MODES = ("official", "forward", "bidirectional", "cycle")


def write_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def record_for(metrics, task):
    matches = [r for r in metrics["task_records"] if r["task"] == task]
    if len(matches) != 1:
        raise RuntimeError("Expected one record for task {}".format(task))
    return matches[0]


def post(record):
    return record["post_ca"]["metrics"]


def gate(candidate, reference, label):
    c, r = post(candidate), post(reference)
    overall = c["top1"] > r["top1"]
    old = c["grouped"]["old"] >= r["grouped"]["old"]
    return {
        "comparison": label,
        "rule": "task2 post-CA Top1 strictly improves and old Top1 does not decrease",
        "overall_condition": {"candidate": c["top1"], "reference": r["top1"], "pass": overall},
        "old_condition": {"candidate": c["grouped"]["old"], "reference": r["grouped"]["old"], "pass": old},
        "pass": overall and old,
        "decision": "SURVIVES" if overall and old else "DEPRIORITIZE",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--configs", nargs=4, required=True)
    args = parser.parse_args()
    root = Path(args.root)
    common = json.loads((root / "common_task0/compatibility.json").read_text())
    if common["status"] != "PASS":
        raise RuntimeError("Common task0 did not pass")

    configs = [json.loads(Path(p).read_text()) for p in args.configs]
    if tuple(c["arm"] for c in configs) != ARMS or tuple(c["bicyc_mode"] for c in configs) != MODES:
        raise RuntimeError("Config order/modes do not match locked arms")
    allowed_differences = {
        "arm", "bicyc_mode", "prefix", "output_root", "metrics_output",
        "task_metrics_output", "per_class_output", "transport_metrics_output",
        "progress_path",
    }
    base = configs[0]
    for cfg in configs[1:]:
        for key in set(base) | set(cfg):
            if key not in allowed_differences and base.get(key) != cfg.get(key):
                raise RuntimeError("Uncontrolled config mismatch for {}".format(key))

    metrics = {}
    for arm, mode in zip(ARMS, MODES):
        data = json.loads((root / arm / "metrics.json").read_text())
        if data["status"] != "PASS" or data["mode"] != mode:
            raise RuntimeError("Arm {} did not pass or mode mismatched".format(arm))
        tasks = [r["task"] for r in data["task_records"]]
        if tasks != [0, 1, 2]:
            raise RuntimeError("Arm {} task coverage is {}".format(arm, tasks))
        task0 = record_for(data, 0)
        if (post(task0)["top1"], post(task0)["top5"]) != (99.1, 100.0):
            raise RuntimeError("Task0 mismatch for {}".format(arm))
        metrics[arm] = data

    # Mechanism invariants and matched initialization.
    for task in (1, 2):
        a_hashes = []
        for arm in ARMS[1:]:
            t = record_for(metrics[arm], task)["transport"]
            a_hashes.append(t["initial_hashes"]["A"]["sha256"])
        if len(set(a_hashes)) != 1:
            raise RuntimeError("A initialization mismatch at task {}".format(task))
        d_hashes = []
        for arm in ("B2_bidirectional", "B3_cycle"):
            t = record_for(metrics[arm], task)["transport"]
            d_hashes.append(t["initial_hashes"]["D"]["sha256"])
        if len(set(d_hashes)) != 1:
            raise RuntimeError("D initialization mismatch at task {}".format(task))

        b0 = record_for(metrics["B0_official"], task)["transport"]
        if "official_ssca" not in b0 or b0.get("initial_hashes"):
            raise RuntimeError("B0 mechanism invariant failed")
        if b0["ca_covariance_stabilization"]["fallback_count"] != 0:
            raise RuntimeError("B0 unexpectedly required CA covariance fallback")
        b1 = record_for(metrics["B1_forward"], task)["transport"]
        b2 = record_for(metrics["B2_bidirectional"], task)["transport"]
        b3 = record_for(metrics["B3_cycle"], task)["transport"]
        for name, transport in (("B0", b0), ("B1", b1), ("B2", b2), ("B3", b3)):
            if len(transport["stage1_epochs"]) != 30:
                raise RuntimeError("{} task{} did not run 30 Stage-I epochs".format(name, task))
        for name, transport in (("B1", b1), ("B2", b2), ("B3", b3)):
            if len(transport["stage2_epochs"]) != 30:
                raise RuntimeError("{} task{} did not run 30 A-only Stage-II epochs".format(name, task))
            if not transport["paired_after_stage2"]["same_input_tensor_same_iteration"]:
                raise RuntimeError("{} task{} pairing invariant failed".format(name, task))
            stabilization = transport["ca_covariance_stabilization"]
            if any(item["relative_jitter"] > 1e-3
                   for item in stabilization["classes"]):
                raise RuntimeError("{} task{} CA jitter exceeded bound".format(name, task))
            hp = transport["hyperparameters"]
            if (hp["lambda_bi"], hp["lambda_cycle"], hp["pair_preserve"],
                    hp["bicyc_anti_collapse"]) != (5.0, 1.0, False, False):
                raise RuntimeError("{} task{} hyperparameter invariant failed".format(name, task))
        if b1.get("D_final") is not None:
            raise RuntimeError("B1 unexpectedly contains D")
        if b2["paired_after_stage1"].get("cycle_new_mse") is not None:
            raise RuntimeError("B2 unexpectedly contains cycle")
        if b3["paired_after_stage1"].get("cycle_new_mse") is None:
            raise RuntimeError("B3 cycle diagnostics missing")
        for t in (b1, b2, b3):
            if not t["statistics_transport"]["finite"]:
                raise RuntimeError("Non-finite analytic statistics transport")
            if t["statistics_transport"]["max_covariance_asymmetry"] != 0.0:
                raise RuntimeError("Analytic covariance is not exactly symmetric")

    task2 = {arm: record_for(metrics[arm], 2) for arm in ARMS}
    decisions = {
        "B1": gate(task2["B1_forward"], task2["B0_official"], "B1 vs B0: dedicated forward transport"),
        "B2": gate(task2["B2_bidirectional"], task2["B1_forward"], "B2 vs B1: backward alignment"),
        "B3": gate(task2["B3_cycle"], task2["B2_bidirectional"], "B3 vs B2: cycle consistency"),
    }

    summary_rows = []
    for arm in ARMS:
        for task in (0, 1, 2):
            r = record_for(metrics[arm], task)
            for stage in ("pre_ca", "post_ca"):
                m = r[stage]["metrics"]
                summary_rows.append({
                    "arm": arm, "mode": metrics[arm]["mode"], "task": task,
                    "stage": stage, "top1": m["top1"], "top5": m["top5"],
                    "old_top1": m["grouped"]["old"],
                    "new_top1": m["grouped"]["new"],
                    "ca_gain_top1": r["ca_gain_top1"],
                })
    report = {
        "status": "PASS",
        "scope": "CIFAR-100 task0->2, seed 1993, RSIAT setting only",
        "implementation_label": "[IMPLEMENTATION ADAPTATION] RSIAT + BiCyc-style bidirectional/cycle transport",
        "batch_size": base["batch_size"],
        "common_task0": common,
        "arms": metrics,
        "initialization_invariants": {
            str(task): {
                "A_sha256": record_for(metrics["B1_forward"], task)["transport"]["initial_hashes"]["A"]["sha256"],
                "D_sha256": record_for(metrics["B2_bidirectional"], task)["transport"]["initial_hashes"]["D"]["sha256"],
            } for task in (1, 2)
        },
        "ca_covariance_pd_fallback": {
            arm: {
                str(task): record_for(metrics[arm], task)["transport"].get(
                    "ca_covariance_stabilization")
                for task in (1, 2)
            } for arm in ARMS
        },
        "decisions": decisions,
    }
    write_json(root / "reports/final_metrics.json", report)
    write_json(root / "reports/decisions.json", decisions)
    with (root / "reports/task_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader(); writer.writerows(summary_rows)

    def fmt(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return "{:.6g}".format(value)
        return str(value)

    commit = (root / "reports/git_commit.txt").read_text().strip()
    parent = (root / "reports/git_parent.txt").read_text().strip()
    lines = [
        "# Track B — RSIAT + BiCyc-style transport",
        "",
        "**PASS**",
        "",
        "- Commit: `{}`".format(commit),
        "- Parent: `{}`".format(parent),
        "- Batch size: `{}`".format(base["batch_size"]),
        "- Common task0 SHA-256: `{}`".format(common["destination"]["sha256"]),
        "",
        "[IMPLEMENTATION ADAPTATION] This is not an exact BiCyc reproduction.",
        "",
        "## Scope",
        "CIFAR-100, task0→2, seed 1993, one common task0 checkpoint.",
        "",
        "## Runtime per arm",
        "",
        "| Arm | Runtime (s) |",
        "|---|---:|",
    ]
    for arm in ARMS:
        lines.append("| {} | {} |".format(arm, fmt(metrics[arm]["runtime_seconds"])))
    lines.extend([
        "", "## Task metrics", "",
        "| Arm | Task | Stage | Top1 | Top5 | Old Top1 | New Top1 | CA gain |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary_rows:
        lines.append("| {arm} | {task} | {stage} | {top1} | {top5} | {old_top1} | {new_top1} | {ca_gain_top1} |".format(
            **{key: fmt(value) for key, value in row.items()}))
    lines.extend([
        "", "## Matched initial map hashes", "",
        "| Transition | A SHA-256 | D SHA-256 |",
        "|---|---|---|",
    ])
    for task in (1, 2):
        init = report["initialization_invariants"][str(task)]
        lines.append("| task{} | `{}` | `{}` |".format(
            task, init["A_sha256"], init["D_sha256"]))
    lines.extend([
        "", "## Transport diagnostics after Stage II", "",
        "| Arm | Task | A MSE | D MSE | Cycle-new | Cycle-old | ||W_A||F | ||b_A||2 | sv min/med/max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for arm in ARMS[1:]:
        for task in (1, 2):
            t = record_for(metrics[arm], task)["transport"]
            paired = t["paired_after_stage2"] or {}
            A = t["A_final"]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {}/{}/{} |".format(
                arm, task, fmt(paired.get("a_mse")), fmt(paired.get("d_mse")),
                fmt(paired.get("cycle_new_mse")), fmt(paired.get("cycle_old_mse")),
                fmt(A.get("weight_frobenius_norm")), fmt(A.get("bias_l2_norm")),
                fmt(A.get("singular_min")), fmt(A.get("singular_median")),
                fmt(A.get("singular_max"))))
    lines.extend([
        "", "## CA covariance numerical fallback", "",
        "The stored statistics remain the exact analytic affine transport. A minimal diagonal jitter is used only when the unchanged float32 CA MultivariateNormal rejects a covariance; relative jitter is capped at 1e-3.",
        "", "| Arm | Task | Classes requiring fallback | Max relative jitter |",
        "|---|---:|---:|---:|",
    ])
    for arm in ARMS:
        for task in (1, 2):
            stabilization = record_for(metrics[arm], task)["transport"]["ca_covariance_stabilization"]
            classes = stabilization["classes"]
            maximum = max((item["relative_jitter"] for item in classes), default=0.0)
            lines.append("| {} | {} | {} | {} |".format(
                arm, task, stabilization["fallback_count"], fmt(maximum)))
    lines.extend([
        "", "## Decision rule",
        "Task2 post-CA Top1 must strictly improve over the immediate comparator and old-class Top1 must not decrease.",
        "", "## Decisions",
    ])
    for key in ("B1", "B2", "B3"):
        d = decisions[key]
        lines.append("- **{} {}** — overall {} vs {}; old {} vs {}.".format(
            key, d["decision"], fmt(d["overall_condition"]["candidate"]),
            fmt(d["overall_condition"]["reference"]),
            fmt(d["old_condition"]["candidate"]),
            fmt(d["old_condition"]["reference"])))
    lines.extend([
        "", "## Evidence and interpretation",
        "[EVIDENCE] Every arm completed exactly tasks 0, 1, and 2; task0 was byte-identical and reproduced Top1 99.1 / Top5 100.0.",
        "[EVIDENCE] A initialization hashes match across B1/B2/B3 and D hashes match across B2/B3 for each transition.",
        "[INFERENCE] The decisions above apply only to this deterministic CIFAR-100 task0→2, seed-1993 RSIAT screen.",
        "[INFERENCE] One seed is not evidence of statistical significance or generalization.",
        "", "## Deviations",
        "[IMPLEMENTATION ADAPTATION] Dedicated affine A/D replace BiCyc MLP maps and enable analytic Gaussian transport.",
        "[IMPLEMENTATION ADAPTATION] Exact transported covariances remain stored; CA applies bounded diagonal jitter only when its unchanged float32 MultivariateNormal rejects positive-definiteness.",
        "[ENVIRONMENT DEVIATION] The final B3 batch64 smoke OOMed under shared WDDM VRAM pressure; the preregistered fallback set batch32 identically for B0/B1/B2/B3. No other scientific hyperparameter changed.",
    ])
    (root / "reports/final_report.md").write_text("\n".join(lines) + "\n")
    print("TRACK_B_DECISIONS " + json.dumps(decisions, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
