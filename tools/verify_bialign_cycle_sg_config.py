#!/usr/bin/env python3
"""Fail closed unless Cycle-SG is a controlled Cycle config ablation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CYCLE = ROOT / "exps" / "RSIAT_BiAlign_Cycle.json"
CYCLE_SG = ROOT / "exps" / "RSIAT_BiAlign_Cycle_SG.json"
NON_SCIENTIFIC_KEYS = {
    "arm", "bicyc_mode", "metrics_output", "output_root",
    "per_class_output", "prefix", "progress_path", "scientific_label",
    "task_metrics_output", "transport_metrics_output",
}
EXPECTED_CHECKPOINT_SHA256 = (
    "d13bd496019dc5739eefd709c6bf60e28506577dbbb41b50ee81a5e109e55ea0")
EXPECTED_REPORT_SHA256 = (
    "309132ff37a15254bd967262f19c6e4f62c487850aab75d4e7be4143a531f73c")
EXPECTED_LABEL = (
    "[IMPLEMENTATION ABLATION] BiAlign + Cycle with stop-gradient cycle "
    "inputs; cycle regularizes P_t/D_t only and does not directly update "
    "current representation.")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_fields(config):
    return {key: value for key, value in config.items()
            if key not in NON_SCIENTIFIC_KEYS}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    cycle = json.loads(CYCLE.read_text())
    sg = json.loads(CYCLE_SG.read_text())
    require(cycle["bicyc_mode"] == "bialign_cycle", "Cycle mode changed")
    require(sg["bicyc_mode"] == "bialign_cycle_sg", "Wrong SG mode")
    require(scientific_fields(cycle) == scientific_fields(sg),
            "Cycle-SG config is not scientifically matched")
    require(sg["lambda_cycle"] == 1.0, "lambda_cycle must equal 1.0")
    require(sg["scientific_label"] == EXPECTED_LABEL,
            "Scientific label does not state the SG ablation")
    require(sg["resume_path"] == cycle["resume_path"],
            "Configs do not share the task0 checkpoint")
    require(sg["resume_sha256"] == EXPECTED_CHECKPOINT_SHA256,
            "Wrong configured task0 checkpoint hash")
    require(file_sha256(sg["resume_path"]) == EXPECTED_CHECKPOINT_SHA256,
            "Task0 checkpoint file hash mismatch")
    require(sg["common_task0_report_sha256"] == EXPECTED_REPORT_SHA256,
            "Wrong configured compatibility-report hash")
    require(file_sha256(sg["common_task0_report"]) == EXPECTED_REPORT_SHA256,
            "Compatibility-report file hash mismatch")
    require(sg["seed"] == [1993], "Seed changed")
    require(sg["batch_size"] == 32, "Batch size changed")
    require(sg["init_epochs"] == 10, "Base epoch count changed")
    require(sg["inc_epochs"] == 30, "Incremental epoch count changed")
    require(sg["ca_epochs"] == 10, "CA epoch count changed")
    require(sg["ssca"] is True and sg["ca"] is True,
            "Official SSCA/CA must remain enabled")
    require(sg["stats_batch_size"] == 32, "Statistics batch size changed")
    require(sg["ca_covariance_pd_fallback"] is False,
            "Covariance fallback must remain disabled")
    print("BIALIGN_CYCLE_SG_CONFIG_PASS")
    print("only_mechanism_switch=bicyc_mode: bialign_cycle -> bialign_cycle_sg")
    print("task0_checkpoint_sha256=" + EXPECTED_CHECKPOINT_SHA256)


if __name__ == "__main__":
    main()
