"""Summarize actual completed runs and paired deltas; no missing results imputed."""
import argparse
import json
import statistics
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.qbridge import stable_hash, write_json


def protocol_hash(config):
    # Exclude only variant choices, paths and logging/checkpoint policy. Preserve
    # all other settings, including unknown source-config budget/data switches.
    excluded = {"qbridge", "qbridge_experiment", "prefix", "output_root", "config", "overlay",
                "data_root", "device", "use_quantum_kernel_inc", "q_inc_train_mode",
                "save_checkpoints", "keep_last_checkpoint", "resume", "resume_path",
                "compact_diagonal_checkpoint", "isolate_runs",
                # Observation only: diagnostics never change the trained model.
                "stats_diagnostics", "save_predictions"}
    return stable_hash({k: v for k, v in config.items() if k not in excluded})


def summarize(root):
    runs = {}
    skipped = []
    for path in Path(root).rglob("manifest.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema") != "qbridge-v1.1-run":
            continue
        if record["status"] != "completed" or (path.parent / "failure.json").exists():
            skipped.append(str(path))
            continue
        cfg = record["config"]
        key = (cfg["dataset"], cfg["seed"], cfg["qbridge"]["variant"])
        if key in runs:
            raise ValueError("Duplicate completed run for {}".format(key))
        task = json.loads((path.parent / ("task_%d.json" % record["last_completed_task"])).read_text())
        runs[key] = {"dataset": key[0], "seed": key[1], "variant": key[2],
                     "base_hash": record["base_checkpoint_sha256"], "path": str(path),
                     "protocol_hash": protocol_hash(cfg),
                     "aia": task["aia"], "final_top1": task["accuracy"]["top1"],
                     "forgetting": task["forgetting"], "old_new": task["accuracy"]["grouped"]}
    contrasts = []
    for left, right in (("R3", "R2"), ("R5", "R3"), ("R6", "R5"), ("R5", "R4"),
                        ("R5", "R7"), ("R5", "R8"), ("R5", "R9"), ("R5", "R10")):
        for dataset in sorted({k[0] for k in runs}):
            paired = []
            for ds, seed, variant in sorted(runs):
                if ds != dataset or variant != left or (ds, seed, right) not in runs:
                    continue
                a, b = runs[(ds, seed, left)], runs[(ds, seed, right)]
                if a["base_hash"] != b["base_hash"]:
                    raise ValueError("Unpaired base checkpoints: {} {}".format(a["path"], b["path"]))
                if a["protocol_hash"] != b["protocol_hash"]:
                    raise ValueError("Unmatched training/data budgets: {} {}".format(a["path"], b["path"]))
                paired.append({"seed": seed, **{key: a[key]-b[key] for key in ("aia", "final_top1", "forgetting")}})
            if paired:
                contrasts.append({"dataset": dataset, "contrast": left+"-"+right, "pairs": paired,
                                  "summary": {key: {"mean": statistics.mean(row[key] for row in paired),
                                                    "std": statistics.stdev(row[key] for row in paired) if len(paired)>1 else None}
                                              for key in ("aia", "final_top1", "forgetting")}})
    return {"status": "measured_runs_only" if runs else "no_completed_benchmark_runs",
            "runs": list(runs.values()), "paired_contrasts": contrasts, "skipped_incomplete": skipped,
            "claim": "Descriptive paired differences; no automatic significance or quantum-advantage claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(args.output, summarize(args.root))
