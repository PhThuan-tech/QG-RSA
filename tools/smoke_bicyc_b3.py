#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data.data_manager import DataManager
from trainer import _set_device, _set_random
from utils import model_factory
from utils.bicyc_transport import module_state_sha256, tensor_mapping_sha256
from utils.toolkit import AutoencoderSigmoid


def has_nonzero_grad(parameters):
    return any(p.grad is not None and p.grad.detach().abs().sum() > 0 for p in parameters)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--roundtrip-checkpoint", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    smoke_output = Path(args.output).resolve()
    cfg["progress_path"] = str(smoke_output.with_suffix(".progress.json"))
    cfg["output_root"] = str(smoke_output.parent / "smoke_runtime")
    if cfg["bicyc_mode"] != "cycle":
        raise RuntimeError("Smoke requires B3 cycle config")
    seed = cfg["seed"][0]
    cfg["seed"] = seed
    _set_random(seed); _set_device(cfg)
    dm = DataManager(cfg["dataset"], cfg["shuffle"], seed,
                     cfg["init_cls"], cfg["increment"])
    learner = model_factory.get_model(cfg["model_name"], cfg)
    learner.class_order = list(dm._class_order)
    learner.load_checkpoint(args.checkpoint)

    learner._cur_task += 1
    learner.old_ae = AutoencoderSigmoid(
        input_dims=768, code_dims=cfg["ae_code_dims"],
        residual_mode=cfg["ae_residual_mode"]).to(learner._device)
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    learner._initialize_transition_maps()
    rng_preserved = (
        torch.equal(cpu_rng_before, torch.get_rng_state())
        and all(torch.equal(a, b) for a, b in
                zip(cuda_rng_before, torch.cuda.get_rng_state_all()))
    )
    if not rng_preserved:
        raise RuntimeError("A/D initialization perturbed process RNG state")
    expected_shape = (768, 768)
    if tuple(learner.forward_transport.weight.shape) != expected_shape:
        raise RuntimeError("A is not nn.Linear(768, 768)")
    if tuple(learner.backward_transport.weight.shape) != expected_shape:
        raise RuntimeError("D is not nn.Linear(768, 768)")
    task_size = dm.get_task_size(learner._cur_task)
    learner.task_sizes.append(task_size)
    learner._total_classes = learner._known_classes + task_size
    learner._network.update_fc(task_size)
    learner._network.to(learner._device)
    learner._network_module_ptr = learner._network
    learner.old_network_module_ptr = learner._old_network

    dataset = dm.get_dataset(
        np.arange(learner._known_classes, learner._total_classes),
        source="train", mode="train")
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=0, pin_memory=True)
    _, inputs, targets = next(iter(loader))
    inputs = inputs.to(learner._device); targets = targets.to(learner._device)
    groups = [
        {"params": learner._network.convnet.parameters(), "lr": cfg["init_lr"],
         "weight_decay": cfg["weight_decay"]},
        {"params": learner._network.fc.parameters(), "lr": cfg["init_lr"],
         "weight_decay": cfg["weight_decay"]},
        {"params": learner.old_ae.parameters(), "lr": cfg["ae_init_lr"],
         "weight_decay": cfg["ae_weight_decay"]},
        {"params": learner.forward_transport.parameters(),
         "lr": cfg["transport_stage1_lr"],
         "weight_decay": cfg["transport_stage1_weight_decay"]},
        {"params": learner.backward_transport.parameters(),
         "lr": cfg["transport_stage1_lr"],
         "weight_decay": cfg["transport_stage1_weight_decay"]},
    ]
    optimizer = optim.SGD(groups, momentum=0.9)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    logits, loss_c, loss_rsiat, details = learner._compute_rt_loss(
        inputs, targets, epoch=0, warmup_epoch=cfg["warmup_epoch"])
    loss = loss_c + loss_rsiat + details["transport"]
    optimizer.zero_grad(); loss.backward()
    checks = {
        "A_grad": has_nonzero_grad(learner.forward_transport.parameters()),
        "D_grad": has_nonzero_grad(learner.backward_transport.parameters()),
        "P_grad": has_nonzero_grad(learner.old_ae.parameters()),
        "current_representation_grad": has_nonzero_grad(learner._network.convnet.parameters()),
        "old_model_grad": has_nonzero_grad(learner._old_network.parameters()),
    }
    if not all(checks[k] for k in ("A_grad", "D_grad", "P_grad", "current_representation_grad")):
        raise RuntimeError("Expected gradients missing: {}".format(checks))
    if checks["old_model_grad"]:
        raise RuntimeError("Frozen old model received gradients")
    optimizer.step()

    statistics_transport = learner._analytic_transport_old_statistics()

    # Supply deterministic, well-conditioned placeholders only for the ten new
    # classes so this smoke can exercise the real CA sampler without a full
    # feature-statistics pass. Old-class statistics are the real analytic A
    # transport from the approved common task0 checkpoint.
    missing_classes = learner._total_classes - learner._class_means.shape[0]
    if missing_classes > 0:
        learner._class_means = np.concatenate([
            learner._class_means,
            np.zeros((missing_classes, learner.feature_dim), dtype=np.float64),
        ], axis=0)
        learner._class_covs = torch.cat([
            learner._class_covs,
            torch.zeros(
                missing_classes, learner.feature_dim, learner.feature_dim,
                dtype=learner._class_covs.dtype),
        ], dim=0)
    identity = torch.eye(
        learner.feature_dim, dtype=learner._class_covs.dtype) * 0.05
    for class_id in range(learner._known_classes, learner._total_classes):
        learner._class_means[class_id] = 0.0
        learner._class_means[class_id, class_id - learner._known_classes] = 0.1
        learner._class_covs[class_id] = identity
    test_dataset = dm.get_dataset(
        np.arange(learner._total_classes), source="test", mode="test")
    learner.test_loader = DataLoader(
        Subset(test_dataset, range(min(64, len(test_dataset)))),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    learner._stage2_compact_classifier(task_size, ca_epochs=1)
    ca_stabilization = learner.ca_covariance_stabilization
    if any(item["relative_jitter"] > cfg["ca_pd_max_relative_jitter"]
           for item in ca_stabilization["classes"]):
        raise RuntimeError("CA fallback exceeded the preregistered numerical bound")

    learner._known_classes = learner._total_classes
    roundtrip_path = Path(args.roundtrip_checkpoint)
    roundtrip_path.parent.mkdir(parents=True, exist_ok=True)
    learner.save_checkpoint(str(roundtrip_path))
    raw_roundtrip = torch.load(
        roundtrip_path, map_location="cpu", weights_only=False)
    trained_a_hash = module_state_sha256(learner.forward_transport)
    trained_d_hash = module_state_sha256(learner.backward_transport)
    if tensor_mapping_sha256(raw_roundtrip["forward_transport_state_dict"]) != trained_a_hash:
        raise RuntimeError("A state changed during disk checkpoint roundtrip")
    if tensor_mapping_sha256(raw_roundtrip["backward_transport_state_dict"]) != trained_d_hash:
        raise RuntimeError("D state changed during disk checkpoint roundtrip")
    if "old_ae_state_dict" not in raw_roundtrip:
        raise RuntimeError("P state missing from disk checkpoint roundtrip")

    report = {
        "status": "PASS", "batch_size": cfg["batch_size"],
        "input_shape": list(inputs.shape),
        "A_architecture": "nn.Linear(768, 768, bias=True)",
        "D_architecture": "nn.Linear(768, 768, bias=True)",
        "loss": float(loss.detach()),
        "loss_classification": float(loss_c.detach()),
        "loss_rsiat": float(loss_rsiat.detach()),
        "loss_a": float(details["loss_a"].detach()),
        "loss_d": float(details["loss_d"].detach()),
        "cycle_new": float(details["cycle_new"].detach()),
        "cycle_old": float(details["cycle_old"].detach()),
        "gradients": checks,
        "rng_preserved_by_A_D_initialization": rng_preserved,
        "initial_hashes": learner.transport_initial_hashes,
        "statistics_transport": statistics_transport,
        "ca_covariance_pd_fallback": ca_stabilization,
        "checkpoint_roundtrip": {
            "path": str(roundtrip_path.resolve()),
            "sha256": sha256_file(roundtrip_path),
            "A_trained_state_sha256": trained_a_hash,
            "D_trained_state_sha256": trained_d_hash,
            "P_present": True,
        },
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "pid": os.getpid(),
    }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("B3_SMOKE " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
