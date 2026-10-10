"""A0/A1/A2 configuration contracts and per-task experiment records."""

import copy
import csv
import json
import math
from pathlib import Path
import statistics


SEEDS = (1993, 2015, 2026)
VARIANTS = {"A0": (False, False), "A1": (True, False), "A2": (True, True)}
VARYING_FIELDS = {"seed", "prefix", "use_quantum_kernel_base", "use_quantum_kernel_inc"}
CSV_FIELDS = (
    "variant", "seed", "task", "train_epochs", "ca_epochs", "ca_applied",
    "evaluation_split", "pre_ca_total", "pre_ca_old", "pre_ca_new",
    "post_ca_total", "post_ca_old", "post_ca_new", "pre_ca_forgetting",
    "post_ca_forgetting", "running_aia", "adapter_stage_seconds", "ca_stage_seconds",
    "training_seconds", "task_seconds",
)


def build_configs(source, quantum_defaults=None):
    """Only change experiment switches and operational output/logging policy."""
    if source.get("resume") or source.get("max_tasks_per_run") is not None:
        raise ValueError("Failure localization requires fresh, complete runs without task resume.")
    base = copy.deepcopy(source)
    for key, value in (quantum_defaults or {}).items():
        if key.startswith("q_") or key.startswith("rs_margin_q") or key in {
            "inc_loss_mode", "rs_margin_inc",
        }:
            base.setdefault(key, copy.deepcopy(value))
    if base.get("q_kernel_type", "pqk") != "pqk":
        raise ValueError("A1/A2 must use the current PQK; choose its existing source config.")
    if float(base.get("val_ratio", 0.0) or 0.0) != 0:
        raise ValueError("This locked benchmark study uses val_ratio=0; do not tune on test metrics.")
    base.update({
        "offline": True, "resume": False, "checkpoint_policy": "final_only",
        "keep_last_checkpoint": False, "save_checkpoints": True,
        "compact_diagonal_checkpoint": False, "quiet_task_logging": True,
    })
    configs = []
    for seed in SEEDS:
        for variant, (use_base, use_inc) in VARIANTS.items():
            config = copy.deepcopy(base)
            config.update({"seed": [seed], "prefix": variant,
                           "use_quantum_kernel_base": use_base,
                           "use_quantum_kernel_inc": use_inc})
            configs.append(config)
    validate_configs(configs)
    return configs


def validate_configs(configs):
    if len(configs) != 9:
        raise ValueError("Expected exactly nine configurations.")
    common = None
    pairs = set()
    for config in configs:
        variant = config["prefix"]
        seed = config["seed"]
        if variant not in VARIANTS or not isinstance(seed, list) or len(seed) != 1 or seed[0] not in SEEDS:
            raise ValueError("Unknown A0/A1/A2 variant or seed.")
        pair = (variant, seed[0])
        if pair in pairs:
            raise ValueError("Duplicate run configuration.")
        pairs.add(pair)
        if (config["use_quantum_kernel_base"], config["use_quantum_kernel_inc"]) != VARIANTS[variant]:
            raise ValueError("Incorrect QKSR switches for {}.".format(variant))
        shared = {k: v for k, v in config.items() if k not in VARYING_FIELDS}
        if common is not None and shared != common:
            raise ValueError("Configurations differ beyond variant switches, prefix and seed.")
        common = shared
        if not config.get("offline") or config.get("resume") or config.get("checkpoint_policy") != "final_only":
            raise ValueError("Study requires offline mode, no resume and final-only checkpointing.")
        if config.get("q_kernel_type", "pqk") != "pqk" or float(config.get("val_ratio", 0) or 0) != 0:
            raise ValueError("Study requires existing PQK and locked test evaluation.")
        if config.get("max_tasks_per_run") is not None or config.get("compact_diagonal_checkpoint"):
            raise ValueError("Study requires complete tasks and full-covariance final checkpoints.")
    return common


def forgetting(matrix):
    """Mean prior-best accuracy minus current accuracy over OLD tasks (pp)."""
    if len(matrix) <= 1:
        return None
    current = matrix[-1]
    return statistics.mean(
        max(row[i] for row in matrix[:-1] if len(row) > i) - current[i]
        for i in range(len(current) - 1)
    )


def _finite(value):
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    """Create a new artifact, never silently replace historical output."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_finite(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


class TaskRecorder:
    def __init__(self, directory, variant, seed):
        self.directory = Path(directory)
        self.variant, self.seed = variant, seed
        self.rows, self.pre_matrix, self.post_matrix = [], [], []
        with (self.directory / "tasks.csv").open("x", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=CSV_FIELDS).writeheader()
        (self.directory / "tasks.jsonl").open("x", encoding="utf-8").close()

    def add(self, task, config, post_result, diagnostic, timings, task_seconds):
        if task != len(self.rows):
            raise ValueError("Task records must be complete and in order.")
        pre = diagnostic["pre_ca"] if diagnostic else {
            "grouped": post_result["grouped"], "task_accuracies": post_result["task_accuracies"],
        }
        post = diagnostic["post_ca"] if diagnostic else pre
        if post["grouped"] != post_result["grouped"] or post["task_accuracies"] != post_result["task_accuracies"]:
            raise ValueError("Post-CA diagnostic and final evaluation differ.")
        self.pre_matrix.append(pre["task_accuracies"])
        self.post_matrix.append(post["task_accuracies"])
        self.rows.append({
            "variant": self.variant, "seed": self.seed, "task": task,
            "train_epochs": config["init_epochs"] if task == 0 else config["inc_epochs"],
            "ca_epochs": config["ca_epochs"] if task > 0 and config["ca"] else 0,
            "ca_applied": bool(diagnostic and diagnostic["ca_applied"]),
            "evaluation_split": "test",
            **{stage + "_" + group: (None if task == 0 and group == "old" else result["grouped"][group])
               for stage, result in (("pre_ca", pre), ("post_ca", post)) for group in ("total", "old", "new")},
            "pre_ca_forgetting": forgetting(self.pre_matrix),
            "post_ca_forgetting": forgetting(self.post_matrix),
            "running_aia": None,
            "adapter_stage_seconds": timings.get("adapter", 0.0),
            "ca_stage_seconds": timings.get("ca", 0.0),
            "training_seconds": sum(timings.values()), "task_seconds": task_seconds,
        })
        row = self.rows[-1]
        row["running_aia"] = statistics.mean(r["post_ca_total"] for r in self.rows)
        record = {**row, "pre_ca_task_accuracies": pre["task_accuracies"],
                  "post_ca_task_accuracies": post["task_accuracies"], "diagnostics": diagnostic}
        with (self.directory / "tasks.csv").open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=CSV_FIELDS).writerow(row)
        with (self.directory / "tasks.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_finite(record), allow_nan=False) + "\n")
        return row

    def summary(self, expected_tasks, run_seconds):
        if len(self.rows) != expected_tasks:
            raise ValueError("Incomplete run cannot produce final metrics.")
        values = [row["post_ca_forgetting"] for row in self.rows[1:]]
        return {
            "variant": self.variant, "seed": self.seed, "completed_tasks": len(self.rows),
            "average_incremental_accuracy": self.rows[-1]["running_aia"],
            "final_accuracy": self.rows[-1]["post_ca_total"],
            "average_forgetting": statistics.mean(values) if values else None,
            "final_forgetting": self.rows[-1]["post_ca_forgetting"],
            "training_seconds": sum(row["training_seconds"] for row in self.rows),
            "total_task_seconds": sum(row["task_seconds"] for row in self.rows),
            "run_wall_seconds": run_seconds,
            "pre_ca_task_accuracy_matrix": self.pre_matrix,
            "post_ca_task_accuracy_matrix": self.post_matrix,
        }


def task_line(row):
    def number(value):
        return "n/a" if value is None else "{:.2f}".format(value)
    def triple(stage):
        return "/".join(number(row[stage + "_" + key]) for key in ("total", "old", "new"))
    return ("{variant} seed={seed} task={task} epochs={train_epochs}+CA:{ca_epochs} "
            "pre(total/old/new)={pre} post={post} forgetting={forget}pp time={seconds:.1f}s").format(
                **row, pre=triple("pre_ca"), post=triple("post_ca"),
                forget=number(row["post_ca_forgetting"]), seconds=row["task_seconds"],
            )
