#!/usr/bin/env python3
"""Strictly verify and stage one common task-0 checkpoint for B0/B1/B2/B3."""

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.data_manager import DataManager
from trainer import _set_device
from utils import model_factory
from utils.bicyc_transport import tensor_mapping_sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-a", required=True)
    parser.add_argument("--source-b", required=True)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_rng(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json_atomic(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    args = parse_args()
    source_a, source_b = Path(args.source_a).resolve(), Path(args.source_b).resolve()
    destination = Path(args.destination).resolve()
    hash_a, hash_b = sha256_file(source_a), sha256_file(source_b)
    expected_hash = "d13bd496019dc5739eefd709c6bf60e28506577dbbb41b50ee81a5e109e55ea0"
    if hash_a != hash_b or hash_a != expected_hash:
        raise RuntimeError("Common task0 sources are not the approved byte-identical checkpoint")

    checkpoint = torch.load(source_a, map_location="cpu", weights_only=False)
    required = {"cur_task", "known_classes", "total_classes", "model_state_dict",
                "class_means", "class_covs", "class_order", "task_sizes"}
    if required - set(checkpoint):
        raise RuntimeError("Common task0 checkpoint is incomplete")
    if (checkpoint["cur_task"], checkpoint["known_classes"],
            checkpoint["total_classes"], checkpoint["task_sizes"]) != (0, 10, 10, [10]):
        raise RuntimeError("Checkpoint is not completed task0")
    if "old_ae_state_dict" in checkpoint:
        raise RuntimeError("Task0 unexpectedly contains RSIAT P")

    configs = [json.loads(Path(path).read_text()) for path in args.configs]
    if {cfg["bicyc_mode"] for cfg in configs} != {
            "official", "forward", "bidirectional", "cycle"}:
        raise RuntimeError("Expected exactly all four Track-B modes")
    controlled = (
        "dataset", "shuffle", "init_cls", "increment", "model_name",
        "convnet_type", "init_lr", "batch_size", "weight_decay", "scale",
        "ca_epochs", "ffn_num", "optimizer", "inc_epochs", "beta", "gamma",
        "ae_code_dims", "ae_residual_mode", "ae_init_lr", "ae_weight_decay",
    )
    for cfg in configs[1:]:
        for key in controlled:
            if cfg[key] != configs[0][key]:
                raise RuntimeError("Arm config mismatch for {}".format(key))
    raw = configs[0]
    if raw["seed"] != [1993] or raw["batch_size"] not in (64, 32):
        raise RuntimeError("Seed/batch preflight mismatch")
    metadata = checkpoint["run_metadata"]
    for key in ("dataset", "init_cls", "increment", "model_name", "convnet_type"):
        if metadata[key] != raw[key]:
            raise RuntimeError("Checkpoint/config mismatch for {}".format(key))
    if metadata["seed"] != 1993:
        raise RuntimeError("Checkpoint seed mismatch")

    runtime = copy.deepcopy(raw); runtime["seed"] = 1993
    _set_device(runtime); reset_rng(1993)
    data_manager = DataManager(
        raw["dataset"], raw["shuffle"], 1993, raw["init_cls"], raw["increment"])
    if list(data_manager._class_order) != list(checkpoint["class_order"]):
        raise RuntimeError("Task0 class order mismatch")
    model = model_factory.get_model(runtime["model_name"], runtime)
    model.class_order = list(data_manager._class_order)
    model._rebuild_classifier(model._network, checkpoint["task_sizes"])
    missing, unexpected = model._network.load_state_dict(
        checkpoint["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError("Strict model-state load failed")
    if model.load_checkpoint(str(source_a)) != 0:
        raise RuntimeError("Checkpoint load returned wrong task")
    test_dataset = data_manager.get_dataset(np.arange(10), source="test", mode="test")
    model.test_loader = DataLoader(
        test_dataset, batch_size=raw["batch_size"], shuffle=False,
        num_workers=raw["num_workers"], pin_memory=raw["pin_memory"],
        persistent_workers=raw["persistent_workers"])
    evaluated = model.eval_task_detailed()
    if evaluated["metrics"]["top1"] != 99.1 or evaluated["metrics"]["top5"] != 100.0:
        raise RuntimeError("Task0 metrics did not reproduce exactly")
    # At task0 there are no old classes. Checkpoint restoration necessarily sets
    # known_classes=10, so BaseLearner groups all samples as old and emits NaN for
    # new. Normalize only the reporting semantics; total/top-k evidence is unchanged.
    evaluated["metrics"]["grouped"]["old"] = None
    evaluated["metrics"]["grouped"]["new"] = evaluated["metrics"]["top1"]

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source_a, temporary); os.replace(temporary, destination)
    if sha256_file(destination) != hash_a:
        raise RuntimeError("Staged common checkpoint hash mismatch")

    state = checkpoint["model_state_dict"]
    classifier_state = {k: v for k, v in state.items() if k.startswith("fc.")}
    representation_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    report = {
        "status": "PASS",
        "byte_identical_sources": True,
        "source_a": {"path": str(source_a), "sha256": hash_a},
        "source_b": {"path": str(source_b), "sha256": hash_b},
        "destination": {"path": str(destination), "sha256": hash_a},
        "checkpoint_metadata": metadata,
        "strict_model_state_load": True,
        "class_order_match": True,
        "task0_metrics_reproduced": evaluated,
        "task0_group_semantics": "no old classes; new equals all task0 classes",
        "classifier_sha256": tensor_mapping_sha256(classifier_state),
        "representation_sha256": tensor_mapping_sha256(representation_state),
        "class_means_sha256": tensor_mapping_sha256({"means": checkpoint["class_means"]}),
        "class_covariances_sha256": tensor_mapping_sha256({"covs": checkpoint["class_covs"]}),
        "batch_size": raw["batch_size"],
        "compatible_arms": [cfg["arm"] for cfg in configs],
    }
    write_json_atomic(args.report, report)
    print("COMMON_TASK0 " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
