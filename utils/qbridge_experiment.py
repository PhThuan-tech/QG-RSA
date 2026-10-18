"""Explicit base branching and auditable experiment artifacts (opt-in runner)."""
import copy
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from utils.qbridge import config_from_args, sha256_file, trusted_load, write_json


def merge_dict(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def preflight_base(args):
    """Fail before model creation/downloads; never silently start a new base run."""
    config_from_args(args)
    qb = args.get("qbridge", {})
    path = str(qb.get("base_checkpoint_path", "")).format(seed=args["seed"], dataset=args["dataset"])
    if not Path(path).is_file():
        raise ValueError("An existing paired base checkpoint is required: " + path)
    checkpoint = trusted_load(path)
    if checkpoint.get("cur_task") != 0:
        raise ValueError("Q-BRIDGE branching requires a completed task-0 checkpoint")
    required = {"model_state_dict", "quantum_kernel_state_dict", "class_means", "class_covs",
                "class_order", "run_metadata", "known_classes", "total_classes"}
    if required - checkpoint.keys():
        raise ValueError("Base checkpoint lacks full state/metadata: " + str(required - checkpoint.keys()))
    if checkpoint["known_classes"] != args["init_cls"] or checkpoint["total_classes"] != args["init_cls"]:
        raise ValueError("Base checkpoint must be saved after after_task() with matching base class count")
    metadata = checkpoint["run_metadata"]
    for key in ("dataset", "seed", "init_cls", "increment", "model_name", "convnet_type"):
        if key not in metadata or metadata[key] != args[key]:
            raise ValueError("Base provenance mismatch/missing field: " + key)
    for key in ("q_num_qubits", "q_num_layers", "q_kernel_type", "q_kernel_order", "q_gamma_mode", "q_reupload"):
        if key in metadata and args.get(key) != metadata[key]:
            raise ValueError("Use the exact original base config; mismatched field: " + key)
    if metadata.get("use_quantum_kernel_base") is not True:
        raise ValueError("Base checkpoint must have trained QKSR at task 0")
    if not metadata.get("q_kernel_type", "pqk").startswith("pqk"):
        raise ValueError("A PQK base checkpoint is required")
    for key in ("class_means", "class_covs"):
        if not torch.isfinite(torch.as_tensor(checkpoint[key])).all():
            raise ValueError("Nonfinite base statistics: " + key)
    if checkpoint["class_covs"].shape != (args["init_cls"], 768, 768):
        raise ValueError("Base requires full 768D covariance; diagonal-only checkpoint is not equivalent")
    if torch.as_tensor(checkpoint["class_means"]).shape != (args["init_cls"], 768):
        raise ValueError("Base mean shape does not match class count / feature dimension")
    if len(set(checkpoint["class_order"])) != len(checkpoint["class_order"]):
        raise ValueError("Invalid duplicate base class order")
    args["qbridge"]["base_checkpoint_path"] = path
    for key in ("calibrator_artifact_path", "selector_artifact_path"):
        if qb.get(key):
            args["qbridge"][key] = str(qb[key]).format(seed=args["seed"], dataset=args["dataset"])
    if args["qbridge"].get("calibration", {}).get("split_seed") == "{seed}":
        args["qbridge"]["calibration"]["split_seed"] = int(args["seed"])
    args["_qbridge_base_hash"] = sha256_file(path)
    args["_qbridge_restore_only"] = True
    args["qbridge_experiment"] = True
    if args.get("resume", False):
        raise ValueError("v1.1 branches from base only; incremental resume is not supported")
    if args.get("compact_diagonal_checkpoint", False):
        raise ValueError("Q-BRIDGE preserves full covariance checkpoints")
    return path


@contextmanager
def preserve_rng():
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def evaluate_stage(learner, stage):
    """Read-only post-adaptation evaluation, isolated from training randomness."""
    with preserve_rng():
        modes = [(module, module.training) for module in learner._network.modules()]
        pred, target = learner._eval_cnn(learner.test_loader)
        metrics = learner._evaluate(pred, target)
        for module, training in modes:
            module.training = training
        grouped = metrics["grouped"]
        record = {"stage": stage, "task": learner._cur_task, "metrics": metrics}
        bridge = getattr(learner, "qbridge_runtime", None)
        if bridge is not None:
            record["guard"] = bridge.guard_metrics(
                learner._network.fc, learner.old_ae, learner._qb_seen_ids, learner._qb_new_ids)
        # Explicit direction of errors on real evaluation images. Evaluation only.
        pred = pred[:, 0]
        old = target < learner._known_classes
        record["old_to_new_error_rate"] = float((pred[old] >= learner._known_classes).mean()) if old.any() else None
        record["new_to_old_error_rate"] = float((pred[~old] < learner._known_classes).mean()) if (~old).any() else None
        learner.qbridge_stage_records.append(record)
        write_json(Path(learner.args["_qbridge_report_dir"]) / "stages.json", learner.qbridge_stage_records)
        return grouped


def run_manifest(args, status, **extra):
    public = {k: v for k, v in args.items() if not k.startswith("_")}
    public["device"] = [str(d) for d in args["device"]]
    value = {"schema": "qbridge-v1.1-run", "status": status,
             "base_checkpoint_sha256": args["_qbridge_base_hash"], "config": public,
             "checkpoint_policy": "final_dataset_task_only",
             "restart_policy": "paired base restart; no incremental exact-resume claim",
             "forgetting_convention": "mean over old tasks of best pre-current accuracy minus current accuracy",
             **extra}
    write_json(Path(args["_qbridge_report_dir"]) / "manifest.json", value)
