#!/usr/bin/env python3
"""Fail closed unless full official/BiAlign configs are scientifically matched."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "exps" / "RSIAT_BiAlign_official.json"
BIALIGN = ROOT / "exps" / "RSIAT_BiAlign.json"
NON_SCIENTIFIC_KEYS = {
    "arm", "bicyc_mode", "metrics_output", "output_root", "per_class_output",
    "prefix", "progress_path", "scientific_label", "task_metrics_output",
    "transport_metrics_output",
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


def main():
    official = json.loads(OFFICIAL.read_text())
    bialign = json.loads(BIALIGN.read_text())
    assert official["bicyc_mode"] == "official"
    assert bialign["bicyc_mode"] == "bialign"
    official_science = {
        key: value for key, value in official.items()
        if key not in NON_SCIENTIFIC_KEYS}
    bialign_science = {
        key: value for key, value in bialign.items()
        if key not in NON_SCIENTIFIC_KEYS}
    assert official_science == bialign_science
    assert official["resume_path"] == bialign["resume_path"]
    assert official["resume_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert bialign["resume_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert file_sha256(official["resume_path"]) == EXPECTED_CHECKPOINT_SHA256
    assert official["common_task0_report"] == bialign["common_task0_report"]
    assert official["common_task0_report_sha256"] == EXPECTED_REPORT_SHA256
    assert file_sha256(official["common_task0_report"]) == EXPECTED_REPORT_SHA256
    assert official["seed"] == bialign["seed"] == [1993]
    assert official["batch_size"] == bialign["batch_size"] == 32
    assert official["stats_batch_size"] == bialign["stats_batch_size"] == 32
    assert official["init_epochs"] == bialign["init_epochs"] == 10
    assert official["inc_epochs"] == bialign["inc_epochs"] == 30
    assert official["ca_epochs"] == bialign["ca_epochs"] == 10
    assert official["ssca"] is bialign["ssca"] is True
    assert official["ca"] is bialign["ca"] is True
    assert official["ca_covariance_pd_fallback"] is False
    assert bialign["ca_covariance_pd_fallback"] is False
    print("MATCHED_CONFIG_PASS")
    print("only_method_switch=bicyc_mode: official -> bialign")
    print("task0_checkpoint_sha256=" + EXPECTED_CHECKPOINT_SHA256)


if __name__ == "__main__":
    main()
