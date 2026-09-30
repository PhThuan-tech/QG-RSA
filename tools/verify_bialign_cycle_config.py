#!/usr/bin/env python3
"""Fail closed unless BiAlign-cycle is a controlled extension of BiAlign."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIALIGN = ROOT / "exps" / "RSIAT_BiAlign.json"
CYCLE = ROOT / "exps" / "RSIAT_BiAlign_Cycle.json"
BIALIGN_SMOKE = ROOT / "exps" / "RSIAT_BiAlign_smoke.json"
CYCLE_SMOKE = ROOT / "exps" / "RSIAT_BiAlign_Cycle_smoke.json"
NON_SCIENTIFIC_KEYS = {
    "arm", "bicyc_mode", "lambda_cycle", "metrics_output", "output_root",
    "per_class_output", "prefix", "progress_path", "scientific_label",
    "task_metrics_output", "transport_metrics_output",
}
EXPECTED_CHECKPOINT_SHA256 = (
    "d13bd496019dc5739eefd709c6bf60e28506577dbbb41b50ee81a5e109e55ea0")
EXPECTED_REPORT_SHA256 = (
    "309132ff37a15254bd967262f19c6e4f62c487850aab75d4e7be4143a531f73c")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_fields(config):
    return {
        key: value for key, value in config.items()
        if key not in NON_SCIENTIFIC_KEYS
    }


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    bialign = json.loads(BIALIGN.read_text())
    cycle = json.loads(CYCLE.read_text())
    bialign_smoke = json.loads(BIALIGN_SMOKE.read_text())
    cycle_smoke = json.loads(CYCLE_SMOKE.read_text())
    require(bialign["bicyc_mode"] == "bialign", "BiAlign mode changed")
    require(cycle["bicyc_mode"] == "bialign_cycle", "Wrong cycle mode")
    require(cycle["lambda_cycle"] == 1.0, "lambda_cycle must equal 1.0")
    require("lambda_cycle" not in bialign, "Plain BiAlign config changed")
    require(
        scientific_fields(bialign) == scientific_fields(cycle),
        "Full BiAlign-cycle config is not scientifically matched")
    require(
        bialign_smoke["bicyc_mode"] == "bialign",
        "BiAlign smoke mode changed")
    require(
        cycle_smoke["bicyc_mode"] == "bialign_cycle",
        "Wrong cycle smoke mode")
    require(
        cycle_smoke["lambda_cycle"] == 1.0,
        "Smoke lambda_cycle must equal 1.0")
    require(
        "lambda_cycle" not in bialign_smoke,
        "Plain BiAlign smoke config changed")
    require(
        scientific_fields(bialign_smoke) == scientific_fields(cycle_smoke),
        "BiAlign-cycle smoke config is not scientifically matched")
    require(
        bialign["resume_path"] == cycle["resume_path"],
        "Full configs do not share a task0 checkpoint")
    require(
        cycle["resume_sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "Wrong configured task0 checkpoint hash")
    require(
        file_sha256(cycle["resume_path"]) == EXPECTED_CHECKPOINT_SHA256,
        "Task0 checkpoint file hash mismatch")
    require(
        cycle["common_task0_report_sha256"] == EXPECTED_REPORT_SHA256,
        "Wrong configured compatibility-report hash")
    require(
        file_sha256(cycle["common_task0_report"]) == EXPECTED_REPORT_SHA256,
        "Compatibility-report file hash mismatch")
    require(cycle["seed"] == [1993], "Seed changed")
    require(cycle["init_epochs"] == 10, "Base epoch count changed")
    require(cycle["inc_epochs"] == 30, "Incremental epoch count changed")
    require(cycle["ca_epochs"] == 10, "CA epoch count changed")
    require(
        cycle["ssca"] is True and cycle["ca"] is True,
        "Official SSCA/CA must remain enabled")
    require(
        cycle["ca_sampling_mode"] == "analytic_transport",
        "CA sampling mode changed")
    require(
        cycle_smoke["resume_path"] == cycle["resume_path"],
        "Smoke does not share the full config task0 checkpoint")
    require(
        cycle_smoke["resume_sha256"] == EXPECTED_CHECKPOINT_SHA256,
        "Wrong smoke task0 checkpoint hash")
    require(cycle_smoke["inc_epochs"] == 1, "Smoke must use one Stage-I epoch")
    require(cycle_smoke["ca_epochs"] == 1, "Smoke must use one CA epoch")
    require(
        cycle_smoke["max_tasks_per_run"] == 1,
        "Smoke must process exactly one task")
    require(cycle_smoke["stop_after_task"] == 1, "Smoke must stop at task1")
    require(
        cycle["ca_covariance_pd_fallback"] is False,
        "Covariance fallback must remain disabled")
    for forbidden in (
            "lambda_bi", "transport_stage2_epochs", "pair_preserve",
            "bicyc_anti_collapse"):
        require(forbidden not in cycle, "Forbidden config key: " + forbidden)
    print("BIALIGN_CYCLE_CONFIG_PASS")
    print("method_delta=bicyc_mode: bialign -> bialign_cycle; lambda_cycle=1.0")
    print("task0_checkpoint_sha256=" + EXPECTED_CHECKPOINT_SHA256)


if __name__ == "__main__":
    main()
