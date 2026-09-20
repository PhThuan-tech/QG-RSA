#!/usr/bin/env python3
"""Read-only diagnostics for the zero-based RSIAT transition task 1 -> 2.

The script reconstructs the official repository networks, strictly loads the
completed-task checkpoints, and writes only metrics.json and per_class.csv in
--output-dir.  It never trains or changes checkpoint/data files.
"""

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.data import build_transform  # noqa: E402
from utils.inc_net import SimpleVitNet  # noqa: E402
from utils.toolkit import AutoencoderSigmoid  # noqa: E402

SIGMA = 4.0
RBF_FLOOR = 1e-5
COV_RIDGE = 1e-3
SUPPORTED_TRANSITION = 2
NONNEGATIVE_TOLERANCE = 1e-6
CONFIG_FIELDS = (
    "model_name", "convnet_type", "ffn_num", "ae_code_dims", "init_cls",
    "increment", "dataset", "seed",
)
CHECKPOINT_METADATA_ALIASES = {
    "model_name": ("model_name", "model"),
    "convnet_type": ("convnet_type", "backbone"),
    "init_cls": ("init_cls",),
    "increment": ("increment",),
    "dataset": ("dataset",),
    "seed": ("seed",),
}


def fail(message):
    raise RuntimeError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path):
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        fail("Malformed checkpoint {}: top level is not a dictionary".format(path))
    required = {
        "cur_task", "known_classes", "total_classes", "model_state_dict",
        "task_sizes", "class_order", "class_means", "old_ae_state_dict",
    }
    missing = sorted(required - set(value))
    if missing:
        fail("Malformed checkpoint {}: missing {}".format(path, missing))
    return value


def scalar_config_value(value):
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def canonicalize_run_metadata(metadata, checkpoint_name):
    if not isinstance(metadata, dict):
        fail("{} checkpoint run_metadata is not a dictionary".format(
            checkpoint_name))
    canonical, resolved_keys = {}, {}
    for field, aliases in CHECKPOINT_METADATA_ALIASES.items():
        present = [alias for alias in aliases if alias in metadata]
        if not present:
            fail("{} checkpoint run_metadata has no key for {!r}; accepted aliases are {}"
                 .format(checkpoint_name, field, list(aliases)))
        values = [scalar_config_value(metadata[key]) for key in present]
        if any(value != values[0] for value in values[1:]):
            fail("{} checkpoint run_metadata has conflicting aliases for {!r}: {}"
                 .format(checkpoint_name, field,
                         {key: metadata[key] for key in present}))
        canonical[field] = values[0]
        resolved_keys[field] = present

    uses_canonical = all(
        aliases[0] in metadata for aliases in CHECKPOINT_METADATA_ALIASES.values())
    uses_legacy_model_aliases = (
        "model" in metadata or "backbone" in metadata)
    if uses_canonical and not uses_legacy_model_aliases:
        schema = "canonical_model_name_convnet_type"
    elif ("model" in metadata and "backbone" in metadata and
          "model_name" not in metadata and "convnet_type" not in metadata):
        schema = "legacy_model_backbone"
    else:
        schema = "mixed_or_dual_aliases"
    return canonical, {
        "schema": schema,
        "available_keys": sorted(metadata),
        "canonical_to_present_keys": resolved_keys,
        "canonical_values": canonical,
    }


def validate_inputs(config, old_ckpt, new_ckpt, transition_task):
    if transition_task != SUPPORTED_TRANSITION:
        fail("This v1 diagnostic supports only zero-based transition task 1->2; "
             "pass --transition-task 2")
    for field in ("model_name", "convnet_type", "ffn_num", "ae_code_dims",
                  "init_cls", "increment", "dataset"):
        if field not in config:
            fail("Config is missing required reconstruction field {!r}".format(field))
    if str(config["dataset"]).lower() != "cifar224":
        fail("v1 requires dataset='cifar224', got {!r}".format(config["dataset"]))
    if "adapter" not in str(config["convnet_type"]).lower():
        fail("v1 requires the official adapter backbone, got {!r}".format(
            config["convnet_type"]))

    expected_tasks = (transition_task - 1, transition_task)
    validation = {
        "config_fields": {}, "checkpoint_structure": {},
        "run_metadata_schema": {},
    }
    resolved_metadata = {}
    for name, ckpt, expected_task in (
        ("old", old_ckpt, expected_tasks[0]), ("new", new_ckpt, expected_tasks[1])
    ):
        try:
            cur_task = int(ckpt["cur_task"])
            known = int(ckpt["known_classes"])
            total = int(ckpt["total_classes"])
            task_sizes = [int(x) for x in ckpt["task_sizes"]]
            order = [int(x) for x in ckpt["class_order"]]
        except (TypeError, ValueError) as exc:
            fail("Malformed {} checkpoint metadata: {}".format(name, exc))
        if cur_task != expected_task:
            fail("{} checkpoint cur_task={} but expected {}".format(
                name, cur_task, expected_task))
        if len(task_sizes) != cur_task + 1 or sum(task_sizes) != total:
            fail("{} checkpoint has inconsistent task_sizes/cur_task/total_classes"
                 .format(name))
        if known != total:
            fail("{} checkpoint must be saved at a completed task boundary: "
                 "known_classes={} total_classes={}".format(name, known, total))
        if len(order) != 100 or sorted(order) != list(range(100)):
            fail("{} checkpoint class_order is not a permutation of CIFAR-100"
                 .format(name))
        means = torch.as_tensor(ckpt["class_means"])
        if means.ndim != 2 or means.shape[0] < total:
            fail("{} checkpoint class_means has invalid shape {}".format(
                name, tuple(means.shape)))
        validation["checkpoint_structure"][name] = {
            "cur_task": cur_task, "known_classes": known,
            "total_classes": total, "task_sizes": task_sizes,
            "class_means_shape": list(means.shape),
        }

        metadata, metadata_schema = canonicalize_run_metadata(
            ckpt.get("run_metadata"), name)
        resolved_metadata[name] = metadata
        validation["run_metadata_schema"][name] = metadata_schema
        for field in CONFIG_FIELDS:
            cfg_value = scalar_config_value(config.get(field))
            if field in metadata and cfg_value is not None:
                saved = scalar_config_value(metadata[field])
                if saved != cfg_value:
                    fail("Config/checkpoint mismatch for {}.{}: {!r} != {!r}"
                         .format(name, field, saved, cfg_value))
                validation["config_fields"].setdefault(field, {})[name] = "matched"
            else:
                validation["config_fields"].setdefault(field, {})[name] = (
                    "not_present_in_checkpoint_metadata")

    for field in CHECKPOINT_METADATA_ALIASES:
        if resolved_metadata["old"][field] != resolved_metadata["new"][field]:
            fail("Checkpoint run_metadata differs for {!r}".format(field))

    old_sizes = [int(x) for x in old_ckpt["task_sizes"]]
    new_sizes = [int(x) for x in new_ckpt["task_sizes"]]
    if new_sizes[:-1] != old_sizes:
        fail("Task-size history differs between checkpoints")
    if old_sizes != [10, 10] or new_sizes != [10, 10, 10]:
        fail("v1 requires task sizes [10,10] -> [10,10,10], got {} -> {}"
             .format(old_sizes, new_sizes))
    if list(old_ckpt["class_order"]) != list(new_ckpt["class_order"]):
        fail("class_order differs between checkpoints")
    if torch.as_tensor(old_ckpt["class_means"]).shape[1] != 768:
        fail("Expected 768-dimensional checkpoint means")

    ffn_dims = set()
    for checkpoint in (old_ckpt, new_ckpt):
        for key, value in checkpoint["model_state_dict"].items():
            if key.endswith("adaptmlp.down_proj.weight"):
                ffn_dims.add(int(torch.as_tensor(value).shape[0]))
    if ffn_dims != {int(config["ffn_num"])}:
        fail("Config ffn_num={} disagrees with checkpoint adapter dimensions {}"
             .format(config["ffn_num"], sorted(ffn_dims)))
    projector_state = new_ckpt["old_ae_state_dict"]
    code_weight = projector_state.get("encoder.2.weight")
    if code_weight is None:
        fail("Projector state lacks encoder.2.weight needed to validate ae_code_dims")
    checkpoint_code_dims = int(torch.as_tensor(code_weight).shape[0])
    if checkpoint_code_dims != int(config["ae_code_dims"]):
        fail("Config ae_code_dims={} disagrees with checkpoint projector dimension {}"
             .format(config["ae_code_dims"], checkpoint_code_dims))
    validation["config_tensor_shape_checks"] = {
        "ffn_num": int(config["ffn_num"]),
        "ae_code_dims": checkpoint_code_dims,
    }
    return validation


def rebuild_network(config, checkpoint, device, label):
    # Deliberately use the repository constructor.  For this model it obtains
    # official pretrained weights before the full checkpoint is strictly loaded.
    network = SimpleVitNet(config, True)
    for task_size in checkpoint["task_sizes"]:
        network.update_fc(int(task_size))
    incompatible = network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        fail("{} strict load was not exact: missing={} unexpected={}".format(
            label, incompatible.missing_keys, incompatible.unexpected_keys))
    network.to(device).eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def rebuild_projector(config, checkpoint, device):
    projector = AutoencoderSigmoid(input_dims=768,
                                   code_dims=int(config["ae_code_dims"]))
    incompatible = projector.load_state_dict(checkpoint["old_ae_state_dict"],
                                              strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        fail("Final P2 strict load was not exact: missing={} unexpected={}".format(
            incompatible.missing_keys, incompatible.unexpected_keys))
    projector.to(device).eval()
    for parameter in projector.parameters():
        parameter.requires_grad_(False)
    return projector


@contextmanager
def sample_rng(seed):
    """Isolate all RNGs used by torchvision transforms for one sample."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def sample_seed(base, repeat, view, sample_id):
    payload = "{}:{}:{}:{}".format(base, repeat, view, sample_id).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF


class OracleDataset(Dataset):
    def __init__(self, images, mapped_labels, ids, transform):
        self.images = images
        self.labels = mapped_labels
        self.ids = np.asarray(ids, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, row):
        sample_id = int(self.ids[row])
        image = self.transform(Image.fromarray(self.images[sample_id]))
        return sample_id, image, int(self.labels[sample_id])


class PairedAugmentedDataset(Dataset):
    def __init__(self, images, mapped_labels, left_ids, right_ids, transform,
                 base_seed, repeat, protocol):
        self.images = images
        self.labels = mapped_labels
        self.left_ids = np.asarray(left_ids, dtype=np.int64)
        self.right_ids = np.asarray(right_ids, dtype=np.int64)
        self.transform = transform
        self.base_seed = base_seed
        self.repeat = repeat
        self.protocol = protocol
        if len(self.left_ids) != len(self.right_ids):
            fail("Internal pairing error: unequal protocol sides")

    def __len__(self):
        return len(self.left_ids)

    def render(self, sample_id, view):
        seed = sample_seed(self.base_seed, self.repeat, view, int(sample_id))
        with sample_rng(seed):
            return self.transform(Image.fromarray(self.images[int(sample_id)]))

    def __getitem__(self, row):
        left_id = int(self.left_ids[row])
        right_id = int(self.right_ids[row])
        left = self.render(left_id, 0)
        if self.protocol == "A":
            right = left.clone()  # exact materialized tensor, not a second transform
        else:
            right = self.render(right_id, 1)
        return (left_id, right_id, left, right,
                int(self.labels[left_id]), int(self.labels[right_id]))


def make_loader(dataset, args, shuffle=False):
    generator = torch.Generator().manual_seed(int(args.seed))
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.num_workers, pin_memory=args.device_obj.type == "cuda",
        persistent_workers=args.num_workers > 0, generator=generator,
    )


def extract_oracle(loader, networks, device):
    outputs = [[] for _ in networks]
    ids, labels = [], []
    with torch.inference_mode():
        for batch_ids, images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            for index, network in enumerate(networks):
                outputs[index].append(network.extract_vector(images).cpu())
            ids.append(batch_ids.cpu())
            labels.append(batch_labels.cpu())
    return ([torch.cat(parts) for parts in outputs], torch.cat(ids),
            torch.cat(labels))


def extract_augmented(loader, old_network, new_network, device):
    f1, f2, left_ids, right_ids, left_labels, right_labels = [], [], [], [], [], []
    with torch.inference_mode():
        for lid, rid, left, right, llab, rlab in loader:
            f1.append(old_network.extract_vector(
                left.to(device, non_blocking=True)).cpu())
            f2.append(new_network.extract_vector(
                right.to(device, non_blocking=True)).cpu())
            left_ids.append(lid); right_ids.append(rid)
            left_labels.append(llab); right_labels.append(rlab)
    return {
        "f1": torch.cat(f1), "f2": torch.cat(f2),
        "left_ids": torch.cat(left_ids), "right_ids": torch.cat(right_ids),
        "left_labels": torch.cat(left_labels),
        "right_labels": torch.cat(right_labels),
    }


def apply_projector(projector, tensor, device, batch_size):
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(tensor), batch_size):
            pieces.append(projector(tensor[start:start + batch_size].to(
                device, non_blocking=True)).cpu())
    return torch.cat(pieces) if pieces else torch.empty_like(tensor)


def cosine_rows(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-12)


def vector_error(a, b):
    difference = (a.double() - b.double())
    l2 = torch.linalg.vector_norm(difference)
    target_norm = torch.linalg.vector_norm(b.double())
    return {
        "l2": float(l2),
        "relative_l2": float(l2 / target_norm.clamp_min(1e-30)),
        "rmse": float(torch.sqrt(torch.mean(difference.square()))),
        "cosine": float(cosine_rows(a.reshape(1, -1), b.reshape(1, -1))[0]),
    }


def distribution(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if values.numel() == 0:
        fail("Cannot summarize an empty metric")
    quantiles = torch.quantile(values, torch.tensor(
        [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=torch.float64))
    return {
        "mean": float(values.mean()),
        "mean_abs": float(values.abs().mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()), "max": float(values.max()),
        "quantiles": {name: float(value) for name, value in zip(
            ("q00", "q05", "q25", "q50", "q75", "q95", "q100"), quantiles)},
        "negative_fraction": float((values < 0).double().mean()),
        "positive_fraction": float((values > 0).double().mean()),
        "zero_fraction": float((values == 0).double().mean()),
        "abs_lt_0_1_fraction": float((values.abs() < 0.1).double().mean()),
        "abs_lt_0_2_fraction": float((values.abs() < 0.2).double().mean()),
        "count": int(values.numel()),
    }


def mapping_metrics(mapped, target):
    delta = mapped.double() - target.double()
    per_sample_l2 = torch.linalg.vector_norm(delta, dim=1)
    return {
        "mse": float(delta.square().mean()),
        "rmse": float(torch.sqrt(delta.square().mean())),
        "mean_sample_l2": float(per_sample_l2.mean()),
        "std_sample_l2": float(per_sample_l2.std(unbiased=False)),
        "mean_cosine": float(cosine_rows(mapped, target).double().mean()),
        "count": int(len(mapped)),
    }


def transport_baseline_metrics(identity_features, projected_features, target):
    identity = mapping_metrics(identity_features, target)
    projector = mapping_metrics(projected_features, target)
    ratio = projector["rmse"] / max(identity["rmse"], 1e-30)
    return {
        "identity": identity,
        "projector": projector,
        "projector_over_identity_rmse_ratio": ratio,
        "relative_improvement": 1.0 - ratio,
    }


def projector_residual_statistics(delta):
    values = delta.double().flatten()
    if values.numel() == 0:
        fail("Cannot summarize an empty projector residual")
    probabilities = torch.tensor(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99],
        dtype=torch.float64,
    )
    quantiles = torch.quantile(values, probabilities)
    return {
        "elementwise_mean": float(values.mean()),
        "rms": float(torch.sqrt(values.square().mean())),
        "elementwise_std": float(values.std(unbiased=False)),
        "elementwise_min": float(values.min()),
        "elementwise_max": float(values.max()),
        "quantiles": {
            name: float(value) for name, value in zip(
                ("q01", "q05", "q25", "q50", "q75", "q95", "q99"),
                quantiles,
            )
        },
        "fraction_delta_lt_0": float((values < 0).double().mean()),
        "fraction_delta_lt_0_01": float((values < 0.01).double().mean()),
        "fraction_delta_in_0_4_0_6": float(
            ((values >= 0.4) & (values <= 0.6)).double().mean()),
        "fraction_delta_gt_0_9": float((values > 0.9).double().mean()),
        "nonnegative_tolerance": NONNEGATIVE_TOLERANCE,
        "fraction_below_negative_tolerance": float(
            (values < -NONNEGATIVE_TOLERANCE).double().mean()),
        "nonnegative_up_to_tolerance": bool(
            values.min() >= -NONNEGATIVE_TOLERANCE),
        "element_count": int(values.numel()),
    }


def true_drift_statistics(drift):
    values = drift.double().flatten()
    if values.numel() == 0:
        fail("Cannot summarize an empty representation drift")
    positive_part = values.clamp_min(0)
    negative_part_magnitude = (-values).clamp_min(0)
    return {
        "elementwise_mean": float(values.mean()),
        "rms": float(torch.sqrt(values.square().mean())),
        "elementwise_std": float(values.std(unbiased=False)),
        "fraction_d_lt_0": float((values < 0).double().mean()),
        "fraction_d_gt_0": float((values > 0).double().mean()),
        "rms_positive_part": float(torch.sqrt(
            positive_part.square().mean())),
        "rms_negative_part": float(torch.sqrt(
            negative_part_magnitude.square().mean())),
        "part_rms_definition": (
            "sqrt(mean(max(+/-d, 0)^2)) over all elements"),
        "element_count": int(values.numel()),
    }


def projector_error_decomposition(z1, z2, projected_z1):
    if z1.shape != z2.shape or z1.shape != projected_z1.shape:
        fail("RQ5 tensors have inconsistent shapes: z1={}, z2={}, projected_z1={}"
             .format(tuple(z1.shape), tuple(z2.shape),
                     tuple(projected_z1.shape)))
    z1 = z1.double()
    z2 = z2.double()
    projected_z1 = projected_z1.double()
    drift = (z2 - z1).flatten()
    delta = (projected_z1 - z1).flatten()
    identity_mse = drift.square().mean()
    residual_energy = delta.square().mean()
    cross_term = 2.0 * (drift * delta).mean()
    reconstructed = identity_mse + residual_energy - cross_term
    directly_measured = (projected_z1 - z2).square().mean()

    cosine_denominator = (
        torch.linalg.vector_norm(drift) *
        torch.linalg.vector_norm(delta)
    ).clamp_min(1e-30)
    centered_drift = drift - drift.mean()
    centered_delta = delta - delta.mean()
    correlation_denominator = (
        torch.linalg.vector_norm(centered_drift) *
        torch.linalg.vector_norm(centered_delta)
    ).clamp_min(1e-30)
    return {
        "identity_mse": float(identity_mse),
        "residual_energy": float(residual_energy),
        "cross_term_2E_d_dot_delta": float(cross_term),
        "reconstructed_projector_mse": float(reconstructed),
        "directly_measured_projector_mse": float(directly_measured),
        "absolute_numerical_decomposition_error": float(
            torch.abs(reconstructed - directly_measured)),
        "flattened_d_delta_cosine": float(
            torch.dot(drift, delta) / cosine_denominator),
        "flattened_d_delta_pearson_correlation": float(
            torch.dot(centered_drift, centered_delta) /
            correlation_denominator),
        "formula": "MSE(P2(z1),z2) = E[d^2] + E[delta^2] - 2E[d*delta]",
    }


def projector_support_diagnostics(z1, z2, projected_z1, class_range):
    delta = projected_z1.double() - z1.double()
    drift = z2.double() - z1.double()
    return {
        "classes": list(class_range),
        "sample_count": int(z1.shape[0]),
        "feature_dimension": int(z1.shape[1]),
        "residual_delta": projector_residual_statistics(delta),
        "true_drift_d": true_drift_statistics(drift),
        "error_decomposition": projector_error_decomposition(
            z1, z2, projected_z1),
    }


def projector_parameter_delta(task1_state, task2_state):
    task1_keys = set(task1_state)
    task2_keys = set(task2_state)
    if task1_keys != task2_keys:
        fail("Projector state keys differ between task1 and task2: only_task1={}, "
             "only_task2={}".format(sorted(task1_keys - task2_keys),
                                    sorted(task2_keys - task1_keys)))

    total_delta_squared = 0.0
    total_task1_squared = 0.0
    total_task2_squared = 0.0
    parameter_count = 0
    identical = True
    per_layer = {}
    for key in sorted(task1_keys):
        task1_tensor = torch.as_tensor(task1_state[key]).double()
        task2_tensor = torch.as_tensor(task2_state[key]).double()
        if task1_tensor.shape != task2_tensor.shape:
            fail("Projector tensor shape differs for {}: {} != {}".format(
                key, tuple(task1_tensor.shape), tuple(task2_tensor.shape)))
        difference = task2_tensor - task1_tensor
        task1_l2 = torch.linalg.vector_norm(task1_tensor)
        task2_l2 = torch.linalg.vector_norm(task2_tensor)
        delta_l2 = torch.linalg.vector_norm(difference)
        per_layer[key] = {
            "task1_l2_norm": float(task1_l2),
            "task2_l2_norm": float(task2_l2),
            "l2_parameter_delta": float(delta_l2),
            "relative_l2_delta_vs_task1": float(
                delta_l2 / task1_l2.clamp_min(1e-30)),
            "parameter_count": int(task1_tensor.numel()),
            "identical": bool(torch.equal(task1_tensor, task2_tensor)),
        }
        total_delta_squared += float(difference.square().sum())
        total_task1_squared += float(task1_tensor.square().sum())
        total_task2_squared += float(task2_tensor.square().sum())
        parameter_count += int(task1_tensor.numel())
        identical = identical and per_layer[key]["identical"]

    total_delta_l2 = math.sqrt(total_delta_squared)
    total_task1_l2 = math.sqrt(total_task1_squared)
    return {
        "task1_state": "task_1.pkl.old_ae_state_dict",
        "task2_state": "task_2.pkl.old_ae_state_dict",
        "total_l2_parameter_delta": total_delta_l2,
        "relative_parameter_delta_vs_task1": (
            total_delta_l2 / max(total_task1_l2, 1e-30)),
        "task1_total_l2_norm": total_task1_l2,
        "task2_total_l2_norm": math.sqrt(total_task2_squared),
        "parameter_count": parameter_count,
        "tensor_count": len(task1_keys),
        "identical": bool(identical),
        "per_layer": per_layer,
    }


def cross_class_oracle_error_summary(records):
    l2 = np.asarray(
        [record["predicted_to_oracle_f2_l2"] for record in records],
        dtype=np.float64)
    relative_l2 = np.asarray(
        [record["predicted_to_oracle_f2_relative_l2"] for record in records],
        dtype=np.float64)
    return {
        "oracle_l2_mean": float(l2.mean()),
        "oracle_l2_median": float(np.median(l2)),
        "oracle_l2_max": float(l2.max()),
        "oracle_relative_l2_mean": float(relative_l2.mean()),
        "oracle_relative_l2_median": float(np.median(relative_l2)),
        "oracle_relative_l2_max": float(relative_l2.max()),
    }


def baseline_mean_summary(records):
    l2 = np.asarray([record["l2"] for record in records], dtype=np.float64)
    relative_l2 = np.asarray(
        [record["relative_l2"] for record in records], dtype=np.float64)
    cosine = np.asarray(
        [record["cosine"] for record in records], dtype=np.float64)
    return {
        "l2_mean": float(l2.mean()),
        "l2_median": float(np.median(l2)),
        "l2_max": float(l2.max()),
        "relative_l2_mean": float(relative_l2.mean()),
        "relative_l2_median": float(np.median(relative_l2)),
        "relative_l2_max": float(relative_l2.max()),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
    }


def aggregate_records(records):
    keys = sorted(set.intersection(*(set(record) for record in records)))
    result = {}
    for key in keys:
        values = [record[key] for record in records]
        if all(isinstance(value, (int, float)) and not isinstance(value, bool)
               for value in values):
            array = np.asarray(values, dtype=np.float64)
            result[key] = {"mean": float(array.mean()),
                           "std": float(array.std(ddof=0))}
    return result


def covariance_from_features(features):
    if features.ndim != 2 or features.shape[0] < 2:
        fail("Need at least two feature rows for torch.cov")
    x = features.to(dtype=torch.float64)
    return torch.cov(x.T) + torch.eye(x.shape[1], dtype=torch.float64) * COV_RIDGE


def checkpoint_covariances(checkpoint):
    total = int(checkpoint["total_classes"])
    if "class_covs" in checkpoint:
        covs = torch.as_tensor(checkpoint["class_covs"], dtype=torch.float64)
        representation = "full"
    elif "class_variances" in checkpoint:
        variances = torch.as_tensor(checkpoint["class_variances"], dtype=torch.float64)
        covs = torch.diag_embed(variances)
        representation = "diagonal_reconstructed"
    else:
        fail("Checkpoint lacks class_covs/class_variances")
    if covs.ndim != 3 or covs.shape[0] < total or covs.shape[1:] != (768, 768):
        fail("Malformed checkpoint covariance shape {}".format(tuple(covs.shape)))
    return covs[:total], representation


def covariance_comparison(stored, oracle, full_eigh):
    diff = stored - oracle
    oracle_norm = torch.linalg.matrix_norm(oracle)
    result = {
        "exact_equal": bool(torch.equal(stored, oracle)),
        "max_absolute_difference": float(diff.abs().max()),
        "frobenius_difference": float(torch.linalg.matrix_norm(diff)),
        "relative_frobenius": float(torch.linalg.matrix_norm(diff) /
                                    oracle_norm.clamp_min(1e-30)),
        "trace_ratio": float(torch.trace(stored) /
                             torch.trace(oracle).clamp_min(1e-30)),
        "stored_trace": float(torch.trace(stored)),
        "oracle_trace": float(torch.trace(oracle)),
        "diagonal_relative_l2": float(torch.linalg.vector_norm(
            torch.diagonal(diff)) / torch.linalg.vector_norm(
                torch.diagonal(oracle)).clamp_min(1e-30)),
        "stored_symmetry_residual": float(torch.linalg.matrix_norm(
            stored - stored.T)),
        "oracle_symmetry_residual": float(torch.linalg.matrix_norm(
            oracle - oracle.T)),
    }
    if full_eigh:
        stored_eig = torch.linalg.eigvalsh(stored).clamp_min(0)
        oracle_eig = torch.linalg.eigvalsh(oracle).clamp_min(0)
        result.update({
            "eigenvalue_relative_l2": float(torch.linalg.vector_norm(
                stored_eig - oracle_eig) /
                torch.linalg.vector_norm(oracle_eig).clamp_min(1e-30)),
            "stored_effective_rank": effective_rank(stored_eig),
            "oracle_effective_rank": effective_rank(oracle_eig),
            "stored_min_eigenvalue": float(stored_eig.min()),
            "oracle_min_eigenvalue": float(oracle_eig.min()),
        })
    return result


def effective_rank(eigenvalues):
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return float(torch.exp(entropy))


def ssca_repeat(features, old_means, stored_new_means, oracle_new_means):
    f1 = features["f1"].double()
    f2 = features["f2"].double()
    means = old_means.double()
    displacement = f2 - f1
    squared_distance = torch.cdist(means, f1).square()
    raw = torch.exp(-squared_distance / (2.0 * SIGMA ** 2))
    floored = raw + RBF_FLOOR
    weights = floored / floored.sum(dim=1, keepdim=True)
    predicted = means + weights @ displacement
    entropy = -(weights * weights.clamp_min(1e-300).log()).sum(dim=1)
    normalized_entropy = entropy / math.log(weights.shape[1])
    ess = 1.0 / weights.square().sum(dim=1)
    floor_mask = raw < RBF_FLOOR
    floor_mass_fraction = (RBF_FLOOR * raw.shape[1]) / floored.sum(dim=1)

    per_class = []
    for class_id in range(len(means)):
        record = {
            "class_id": class_id,
            "ess": float(ess[class_id]),
            "max_weight": float(weights[class_id].max()),
            "entropy": float(entropy[class_id]),
            "normalized_entropy_uniformity": float(normalized_entropy[class_id]),
            "raw_rbf_min": float(raw[class_id].min()),
            "raw_rbf_max": float(raw[class_id].max()),
            "raw_rbf_mean": float(raw[class_id].mean()),
            "raw_below_floor_fraction": float(floor_mask[class_id].double().mean()),
            "floor_added_mass_fraction": float(floor_mass_fraction[class_id]),
        }
        for prefix, target in (("stored_task2", stored_new_means),
                               ("oracle_f2", oracle_new_means)):
            for metric, value in vector_error(predicted[class_id], target[class_id]).items():
                record["predicted_to_{}_{}".format(prefix, metric)] = value
        per_class.append(record)

    global_record = {
        "ess_mean": float(ess.mean()), "ess_min": float(ess.min()),
        "max_weight_mean": float(weights.max(dim=1).values.mean()),
        "entropy_mean": float(entropy.mean()),
        "normalized_entropy_uniformity_mean": float(normalized_entropy.mean()),
        "raw_rbf_min": float(raw.min()), "raw_rbf_max": float(raw.max()),
        "raw_rbf_mean": float(raw.mean()),
        "raw_below_floor_fraction": float(floor_mask.double().mean()),
        "floor_added_mass_fraction_mean": float(floor_mass_fraction.mean()),
        "same_id_fraction": float((features["left_ids"] ==
                                   features["right_ids"]).double().mean()),
        "same_class_fraction": float((features["left_labels"] ==
                                      features["right_labels"]).double().mean()),
        "cross_class_pair_count": int((features["left_labels"] !=
                                       features["right_labels"]).sum()),
    }
    for prefix, target in (("stored_task2", stored_new_means.double()),
                           ("oracle_f2", oracle_new_means.double())):
        errors = predicted - target
        global_record["predicted_to_{}_rmse".format(prefix)] = float(
            torch.sqrt(errors.square().mean()))
        global_record["predicted_to_{}_mean_class_l2".format(prefix)] = float(
            torch.linalg.vector_norm(errors, dim=1).mean())
    return global_record, per_class


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose RSIAT zero-based transition task 1->2 without training")
    parser.add_argument("--ckpt-old", required=True)
    parser.add_argument("--ckpt-new", required=True)
    parser.add_argument("--transition-task", required=True, type=int)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--estimator-repeats", type=int, default=5)
    parser.add_argument("--full-eigh", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0 or args.estimator_repeats <= 0:
        parser.error("batch-size/repeats must be positive and num-workers nonnegative")
    try:
        args.device_obj = torch.device(args.device)
    except (TypeError, RuntimeError) as exc:
        parser.error("invalid --device: {}".format(exc))
    if args.device_obj.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    return args


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    old_path = Path(args.ckpt_old).resolve()
    new_path = Path(args.ckpt_new).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    for path, label in ((config_path, "config"), (old_path, "old checkpoint"),
                        (new_path, "new checkpoint")):
        if not path.is_file():
            fail("{} does not exist: {}".format(label, path))
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        fail("Config top level must be a JSON object")
    config = dict(config)
    config["seed"] = scalar_config_value(config.get("seed"))
    args.seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device_obj.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    old_ckpt = load_checkpoint(old_path)
    new_ckpt = load_checkpoint(new_path)
    validation = validate_inputs(config, old_ckpt, new_ckpt, args.transition_task)

    print("[1/6] Reconstructing official networks (pretrained download/cache may be used)...")
    old_network = rebuild_network(config, old_ckpt, args.device_obj, "f1/task1")
    new_network = rebuild_network(config, new_ckpt, args.device_obj, "f2/task2")
    projector = rebuild_projector(config, new_ckpt, args.device_obj)
    validation["strict_load"] = {
        "old_network": "zero missing/unexpected", "new_network": "zero missing/unexpected",
        "final_P2": "zero missing/unexpected",
    }

    print("[2/6] Loading CIFAR-100 train split with download=False...")
    cifar = datasets.CIFAR100(str(data_root), train=True, download=False)
    images = cifar.data
    original_labels = np.asarray(cifar.targets, dtype=np.int64)
    if len(images) != 50000 or set(original_labels.tolist()) != set(range(100)):
        fail("Unexpected CIFAR-100 train data/labels under {}".format(data_root))
    class_order = np.asarray(new_ckpt["class_order"], dtype=np.int64)
    inverse_order = np.empty(100, dtype=np.int64)
    inverse_order[class_order] = np.arange(100)
    mapped_labels = inverse_order[original_labels]
    selected_ids = np.flatnonzero(mapped_labels < int(new_ckpt["total_classes"]))
    oracle_transform = transforms.Compose(build_transform(False, None))
    train_transform = transforms.Compose(build_transform(True, None))
    oracle_dataset = OracleDataset(images, mapped_labels, selected_ids, oracle_transform)
    oracle_loader = make_loader(oracle_dataset, args)

    print("[3/6] Extracting deterministic f1/f2 oracle features...")
    (oracle_features, oracle_ids, oracle_labels) = extract_oracle(
        oracle_loader, [old_network, new_network], args.device_obj)
    oracle_f1, oracle_f2 = oracle_features
    if not torch.equal(oracle_ids, torch.as_tensor(selected_ids)):
        fail("Oracle loader did not preserve original global sample IDs/order")
    by_class = {
        class_id: torch.nonzero(
            oracle_labels == class_id, as_tuple=False).flatten()
        for class_id in range(30)
    }
    if any(len(rows) != 500 for rows in by_class.values()):
        fail("Expected exactly 500 CIFAR-100 train samples per selected class")
    oracle_means_f2 = torch.stack([oracle_f2[by_class[c]].double().mean(0)
                                   for c in range(30)])
    old_means = torch.as_tensor(old_ckpt["class_means"][:20], dtype=torch.float64)
    stored_new_means = torch.as_tensor(new_ckpt["class_means"][:30], dtype=torch.float64)

    per_class = {class_id: {
        "class_id": class_id, "original_class_id": int(class_order[class_id]),
        "task_id": class_id // int(config["increment"]),
        "support": "old" if class_id < 20 else "current",
        "n_oracle": int(len(by_class[class_id])),
    } for class_id in range(30)}
    for class_id in range(30):
        comparison = vector_error(stored_new_means[class_id], oracle_means_f2[class_id])
        for key, value in comparison.items():
            per_class[class_id]["stored_task2_mean_to_oracle_f2_" + key] = value

    print("[4/6] Running RQ1 repeated estimators and RQ2/RQ4/RQ5 geometry...")
    # Keep original CIFAR train-row IDs even though rows are grouped by current class.
    current_global_ids = oracle_ids[
        torch.cat([by_class[c] for c in range(20, 30)])
    ].numpy()
    rq1 = {
        "question": "How do pairing protocols affect the exact SSCA mean transport estimator?",
        "kernel": {"sigma": SIGMA, "raw": "exp(-||x-mu||^2/(2*sigma^2))",
                   "floor_added_before_normalization": RBF_FLOOR},
        "repeat_seeds": [args.seed + repeat for repeat in range(args.estimator_repeats)],
        "protocols": {},
    }
    no_shift_records, historical_stored_records = [], []
    for class_id in range(20):
        no_shift = vector_error(old_means[class_id], oracle_means_f2[class_id])
        historical = vector_error(
            stored_new_means[class_id], oracle_means_f2[class_id])
        no_shift_records.append(no_shift)
        historical_stored_records.append(historical)
        per_class[class_id].update({
            "rq1_no_shift_" + key: value for key, value in no_shift.items()
        })
    rq1["baselines"] = {
        "no_shift_task1_old_mean_vs_oracle_f2": {
            "per_old_class": {
                str(class_id): record
                for class_id, record in enumerate(no_shift_records)
            },
            "cross_old_class_summary": baseline_mean_summary(no_shift_records),
        },
        "historical_stored_task2_ssca_vs_oracle_f2": {
            "per_old_class": {
                str(class_id): record
                for class_id, record in enumerate(historical_stored_records)
            },
            "cross_old_class_summary": baseline_mean_summary(
                historical_stored_records),
        },
    }
    rq1_class_records = defaultdict(lambda: defaultdict(list))
    for protocol in ("A", "B", "C"):
        repeats, repeat_cross_class_summaries = [], []
        for repeat in range(args.estimator_repeats):
            repeat_seed = args.seed + repeat
            if protocol == "C":
                left = np.random.default_rng(repeat_seed).permutation(current_global_ids)
                right = np.random.default_rng(repeat_seed + 10_000_019).permutation(current_global_ids)
            else:
                left = current_global_ids.copy()
                right = current_global_ids.copy()
            dataset = PairedAugmentedDataset(
                images, mapped_labels, left, right, train_transform,
                repeat_seed, repeat, protocol)
            features = extract_augmented(make_loader(dataset, args), old_network,
                                         new_network, args.device_obj)
            global_record, class_records = ssca_repeat(
                features, old_means, stored_new_means[:20], oracle_means_f2[:20])
            global_record["repeat"] = repeat
            global_record["repeat_seed"] = repeat_seed
            repeats.append(global_record)
            repeat_summary = cross_class_oracle_error_summary(class_records)
            repeat_summary["repeat"] = repeat
            repeat_summary["repeat_seed"] = repeat_seed
            repeat_cross_class_summaries.append(repeat_summary)
            for record in class_records:
                rq1_class_records[protocol][record["class_id"]].append(record)
        pooled_class_records = [
            record
            for records in rq1_class_records[protocol].values()
            for record in records
        ]
        per_class_repeat_mean_records = []
        for class_id in range(20):
            records = rq1_class_records[protocol][class_id]
            per_class_repeat_mean_records.append({
                "predicted_to_oracle_f2_l2": float(np.mean([
                    record["predicted_to_oracle_f2_l2"] for record in records
                ])),
                "predicted_to_oracle_f2_relative_l2": float(np.mean([
                    record["predicted_to_oracle_f2_relative_l2"]
                    for record in records
                ])),
            })
        rq1["protocols"][protocol] = {
            "definition": {
                "A": "same exact materialized augmented tensor per original ID for f1 and f2",
                "B": "same original ID with independently seeded augmented views",
                "C": "independent ID shuffles and views paired deliberately by emitted row position",
            }[protocol],
            "repeats": repeats,
            "aggregate_mean_std": aggregate_records(repeats),
            "cross_old_class_oracle_summary": {
                "per_repeat": repeat_cross_class_summaries,
                "across_repeat_summaries_mean_std": aggregate_records(
                    [
                        {key: value for key, value in summary.items()
                         if key.startswith("oracle_")}
                        for summary in repeat_cross_class_summaries
                    ]),
                "pooled_repeat_class_pairs": cross_class_oracle_error_summary(
                    pooled_class_records),
                "per_class_repeat_mean_then_cross_class": (
                    cross_class_oracle_error_summary(
                        per_class_repeat_mean_records)),
            },
            "per_old_class": {},
        }
        for class_id, records in rq1_class_records[protocol].items():
            aggregate = aggregate_records(records)
            rq1["protocols"][protocol]["per_old_class"][str(class_id)] = {
                "repeats": records,
                "aggregate_mean_std": aggregate,
            }
            for metric, summary in aggregate.items():
                per_class[class_id]["rq1_{}_{}_mean".format(protocol, metric)] = summary["mean"]
                per_class[class_id]["rq1_{}_{}_std".format(protocol, metric)] = summary["std"]

    rq1["comparison_summary"] = {
        "no_shift": rq1["baselines"]["no_shift_task1_old_mean_vs_oracle_f2"]
        ["cross_old_class_summary"],
        "historical_stored_task2_ssca": rq1["baselines"]
        ["historical_stored_task2_ssca_vs_oracle_f2"]
        ["cross_old_class_summary"],
    }
    for protocol in ("A", "B", "C"):
        protocol_summary = rq1["protocols"][protocol][
            "cross_old_class_oracle_summary"][
                "per_class_repeat_mean_then_cross_class"]
        rq1["comparison_summary"][protocol] = {
            "l2_mean": protocol_summary["oracle_l2_mean"],
            "l2_median": protocol_summary["oracle_l2_median"],
            "l2_max": protocol_summary["oracle_l2_max"],
            "relative_l2_mean": protocol_summary[
                "oracle_relative_l2_mean"],
            "relative_l2_median": protocol_summary[
                "oracle_relative_l2_median"],
            "relative_l2_max": protocol_summary[
                "oracle_relative_l2_max"],
            "aggregation": "mean each old class across repeats, then summarize across old classes",
        }

    projected_old_means = apply_projector(projector, old_means.float(),
                                           args.device_obj, args.batch_size)
    current_rows = torch.cat([by_class[c] for c in range(20, 30)])
    projected_current_f1 = apply_projector(projector, oracle_f1[current_rows],
                                            args.device_obj, args.batch_size)
    current_f2 = oracle_f2[current_rows]
    cosine_matrix = F.normalize(projected_old_means.float(), p=2, dim=1) @ (
        F.normalize(projected_current_f1.float(), p=2, dim=1).T)
    rq2 = {
        "wording": "evaluate exact signed-cosine objective geometry at final P2",
        "formula": "cos(P2(mu_task1[c]), P2(f1(x))) for old prototype c and current sample x",
        "historical_replay_claimed": False,
        "sample_transform": "deterministic repository oracle/test transform",
        "aggregate": distribution(cosine_matrix),
        "per_prototype": {}, "per_current_class": {}, "per_prototype_current_class": {},
    }
    for old_class in range(20):
        summary = distribution(cosine_matrix[old_class])
        rq2["per_prototype"][str(old_class)] = summary
        for key in ("mean", "mean_abs", "std", "min", "max", "negative_fraction",
                    "abs_lt_0_1_fraction", "abs_lt_0_2_fraction"):
            per_class[old_class]["rq2_as_prototype_" + key] = summary[key]
        rq2["per_prototype_current_class"][str(old_class)] = {}
        for current_class in range(20, 30):
            local = (oracle_labels[current_rows] == current_class)
            rq2["per_prototype_current_class"][str(old_class)][str(current_class)] = (
                distribution(cosine_matrix[old_class, local]))
    for current_class in range(20, 30):
        local = (oracle_labels[current_rows] == current_class)
        summary = distribution(cosine_matrix[:, local].reshape(-1))
        rq2["per_current_class"][str(current_class)] = summary
        for key in ("mean", "mean_abs", "std", "min", "max", "negative_fraction",
                    "abs_lt_0_1_fraction", "abs_lt_0_2_fraction"):
            per_class[current_class]["rq2_as_current_class_" + key] = summary[key]

    rq4 = {
        "question": "Does final P2 generalize from current support to old support?",
        "A_current_support": {}, "B_old_support_oracle_only_no_fitting": {},
        "C_old_class_mean_and_Jensen_gap": {},
    }
    old_rows = torch.cat([by_class[c] for c in range(20)])
    projected_old_f1 = apply_projector(projector, oracle_f1[old_rows],
                                        args.device_obj, args.batch_size)
    old_f2 = oracle_f2[old_rows]
    rq4["A_current_support"]["aggregate"] = transport_baseline_metrics(
        oracle_f1[current_rows], projected_current_f1, current_f2)
    rq4["B_old_support_oracle_only_no_fitting"]["aggregate"] = (
        transport_baseline_metrics(
            oracle_f1[old_rows], projected_old_f1, old_f2))
    for class_id in range(20, 30):
        local = oracle_labels[current_rows] == class_id
        metric = transport_baseline_metrics(
            oracle_f1[current_rows][local], projected_current_f1[local],
            current_f2[local])
        rq4["A_current_support"][str(class_id)] = metric
        for baseline in ("identity", "projector"):
            per_class[class_id].update({
                "rq4_current_{}_{}".format(baseline, key): value
                for key, value in metric[baseline].items()
            })
        per_class[class_id]["rq4_current_projector_over_identity_rmse_ratio"] = (
            metric["projector_over_identity_rmse_ratio"])
        per_class[class_id]["rq4_current_relative_improvement"] = metric[
            "relative_improvement"]
    for class_id in range(20):
        local = oracle_labels[old_rows] == class_id
        metric = transport_baseline_metrics(
            oracle_f1[old_rows][local], projected_old_f1[local], old_f2[local])
        rq4["B_old_support_oracle_only_no_fitting"][str(class_id)] = metric
        for baseline in ("identity", "projector"):
            per_class[class_id].update({
                "rq4_old_{}_{}".format(baseline, key): value
                for key, value in metric[baseline].items()
            })
        per_class[class_id]["rq4_old_projector_over_identity_rmse_ratio"] = (
            metric["projector_over_identity_rmse_ratio"])
        per_class[class_id]["rq4_old_relative_improvement"] = metric[
            "relative_improvement"]
        mean_projected_samples = projected_old_f1[local].double().mean(0)
        p2_mean = projected_old_means[class_id].double()
        oracle_mean = oracle_means_f2[class_id]
        record = {}
        for label, a, b in (
            ("P2_mu_task1_to_mean_P2_f1", p2_mean, mean_projected_samples),
            ("P2_mu_task1_to_oracle_f2_mean", p2_mean, oracle_mean),
            ("mean_P2_f1_to_oracle_f2_mean", mean_projected_samples, oracle_mean),
        ):
            for key, value in vector_error(a, b).items():
                record[label + "_" + key] = value
        record["jensen_noncommutativity_gap_l2"] = float(torch.linalg.vector_norm(
            p2_mean - mean_projected_samples))
        rq4["C_old_class_mean_and_Jensen_gap"][str(class_id)] = record
        per_class[class_id].update({"rq4_C_" + k: v for k, v in record.items()})
    current_transport = rq4["A_current_support"]["aggregate"]
    old_transport = rq4["B_old_support_oracle_only_no_fitting"]["aggregate"]
    rq4["support_mismatch_comparison"] = {
        "interpretation": (
            "Compare projector against its identity baseline within each support; "
            "do not infer support mismatch from raw old/current projector RMSE alone."),
        "current_support": {
            "identity_rmse": current_transport["identity"]["rmse"],
            "projector_rmse": current_transport["projector"]["rmse"],
            "projector_over_identity_rmse_ratio": current_transport[
                "projector_over_identity_rmse_ratio"],
            "relative_improvement": current_transport["relative_improvement"],
        },
        "old_support": {
            "identity_rmse": old_transport["identity"]["rmse"],
            "projector_rmse": old_transport["projector"]["rmse"],
            "projector_over_identity_rmse_ratio": old_transport[
                "projector_over_identity_rmse_ratio"],
            "relative_improvement": old_transport["relative_improvement"],
        },
        "old_minus_current_relative_improvement": (
            old_transport["relative_improvement"] -
            current_transport["relative_improvement"]),
        "old_over_current_projector_rmse_ratio_secondary_only": (
            old_transport["projector"]["rmse"] /
            max(current_transport["projector"]["rmse"], 1e-30)),
    }

    rq5_projector_residual = {
        "question": (
            "Does the final sigmoid-residual projector add a large positive "
            "residual that worsens identity transport?"),
        "definitions": {
            "z1": "f1(x)",
            "z2": "f2(x)",
            "delta": "P2(z1) - z1",
            "d": "z2 - z1",
            "architecture_expectation": (
                "AutoencoderSigmoid makes delta elementwise nonnegative; "
                "verify at tolerance 1e-6"),
            "primary_support": "current classes 20-29",
            "secondary_support": "old classes 0-19, oracle-only",
        },
        "current_support_primary": projector_support_diagnostics(
            oracle_f1[current_rows], current_f2, projected_current_f1,
            range(20, 30)),
        "old_support_secondary_oracle_only": projector_support_diagnostics(
            oracle_f1[old_rows], old_f2, projected_old_f1, range(0, 20)),
        "P1_to_P2_parameter_delta": projector_parameter_delta(
            old_ckpt["old_ae_state_dict"], new_ckpt["old_ae_state_dict"]),
    }

    print("[5/6] Computing exact float64 covariance diagnostics...")
    old_covs, old_cov_repr = checkpoint_covariances(old_ckpt)
    new_covs, new_cov_repr = checkpoint_covariances(new_ckpt)
    rq3 = {
        "question": "How do stored task2 covariances compare with deterministic f2 oracle covariances?",
        "oracle_formula": "torch.cov(float64_features.T) + 1e-3 I",
        "ridge": COV_RIDGE, "full_eigh": bool(args.full_eigh),
        "checkpoint_representations": {"task1": old_cov_repr, "task2": new_cov_repr},
        "old_classes_stored_task2_vs_oracle": {},
        "old_classes_carryover_task1_to_task2": {},
        "current_classes_sanity": {},
    }
    old_oracle_comparisons, carry_comparisons, current_comparisons = [], [], []
    for class_id in range(30):
        oracle_cov = covariance_from_features(oracle_f2[by_class[class_id]])
        comparison = covariance_comparison(new_covs[class_id], oracle_cov, args.full_eigh)
        bucket = ("old_classes_stored_task2_vs_oracle" if class_id < 20
                  else "current_classes_sanity")
        rq3[bucket][str(class_id)] = comparison
        (old_oracle_comparisons if class_id < 20 else current_comparisons).append(comparison)
        per_class[class_id].update({"rq3_task2_vs_oracle_" + k: v
                                    for k, v in comparison.items()})
        if class_id < 20:
            carry = covariance_comparison(new_covs[class_id], old_covs[class_id],
                                          args.full_eigh)
            # Here the denominator/reference is task1, as named explicitly.
            rq3["old_classes_carryover_task1_to_task2"][str(class_id)] = carry
            carry_comparisons.append(carry)
            per_class[class_id].update({"rq3_carryover_task2_vs_task1_" + k: v
                                        for k, v in carry.items()})
    rq3["aggregates"] = {
        "old_task2_vs_oracle": aggregate_records(old_oracle_comparisons),
        "old_carryover_task2_vs_task1": aggregate_records(carry_comparisons),
        "current_task2_vs_oracle_sanity": aggregate_records(current_comparisons),
    }
    validation["current_class_statistics_sanity"] = {
        "mean_l2": distribution(torch.tensor([
            per_class[class_id]["stored_task2_mean_to_oracle_f2_l2"]
            for class_id in range(20, 30)
        ], dtype=torch.float64)),
        "covariance_relative_frobenius": distribution(torch.tensor([
            rq3["current_classes_sanity"][str(class_id)]["relative_frobenius"]
            for class_id in range(20, 30)
        ], dtype=torch.float64)),
    }

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        fail("Could not determine git commit: {}".format(exc))
    provenance = {
        "script_version": 1,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "checkpoints": {
            "old": {"path": str(old_path), "sha256": sha256_file(old_path)},
            "new": {"path": str(new_path), "sha256": sha256_file(new_path)},
        },
        "git_commit": git_commit,
        "data": {"root": str(data_root), "dataset": "CIFAR100",
                 "split": "train", "download": False,
                 "original_global_sample_ids_preserved": True,
                 "class_order_source": "checkpoints_authoritative"},
        "device": str(args.device_obj), "batch_size": args.batch_size,
        "num_workers": args.num_workers, "seed": args.seed,
        "deterministic_algorithms": True,
        "network_reconstruction": "official SimpleVitNet constructor then strict full state load",
        "validated_config_fields": {
            field: scalar_config_value(config.get(field)) for field in CONFIG_FIELDS
        },
        "evaluation_only": True,
    }
    metrics = {
        "schema_version": 1,
        "provenance": provenance, "validation": validation,
        "transition": {"zero_based_from_task": 1, "zero_based_to_task": 2,
                       "old_classes": [0, 19], "current_classes": [20, 29]},
        "RQ1": rq1, "RQ2": rq2, "RQ3": rq3, "RQ4": rq4,
        "RQ5_projector_residual": rq5_projector_residual,
    }

    print("[6/6] Writing metrics.json and per_class.csv...")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    csv_path = output_dir / "per_class.csv"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    rows = [per_class[class_id] for class_id in range(30)]
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    preferred = ["class_id", "original_class_id", "task_id", "support", "n_oracle"]
    fieldnames = preferred + [name for name in fieldnames if name not in preferred]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done: {}".format(metrics_path))
    print("Done: {}".format(csv_path))
    print("RQ1 oracle RMSE (A/B/C): {}".format(
        ", ".join("{}={:.6g}".format(p, rq1["protocols"][p]
                  ["aggregate_mean_std"]["predicted_to_oracle_f2_rmse"]["mean"])
                  for p in ("A", "B", "C"))))
    print("RQ4 current identity/projector RMSE={:.6g}/{:.6g}, improvement={:.3%}".format(
        rq4["support_mismatch_comparison"]["current_support"]["identity_rmse"],
        rq4["support_mismatch_comparison"]["current_support"]["projector_rmse"],
        rq4["support_mismatch_comparison"]["current_support"]
        ["relative_improvement"]))
    print("RQ4 old identity/projector RMSE={:.6g}/{:.6g}, improvement={:.3%}".format(
        rq4["support_mismatch_comparison"]["old_support"]["identity_rmse"],
        rq4["support_mismatch_comparison"]["old_support"]["projector_rmse"],
        rq4["support_mismatch_comparison"]["old_support"]
        ["relative_improvement"]))
    rq5_current = rq5_projector_residual["current_support_primary"]
    rq5_delta = rq5_current["residual_delta"]
    rq5_drift = rq5_current["true_drift_d"]
    rq5_decomposition = rq5_current["error_decomposition"]
    rq5_parameter_delta = rq5_projector_residual["P1_to_P2_parameter_delta"]
    print("RQ5 current delta RMS/mean/median={:.6g}/{:.6g}/{:.6g}, "
          "negative={:.3%}".format(
              rq5_delta["rms"], rq5_delta["elementwise_mean"],
              rq5_delta["quantiles"]["q50"],
              rq5_delta["fraction_delta_lt_0"]))
    print("RQ5 current drift negative={:.3%}, identity/projector MSE={:.6g}/{:.6g}, "
          "residual energy={:.6g}, cross term={:.6g}".format(
              rq5_drift["fraction_d_lt_0"],
              rq5_decomposition["identity_mse"],
              rq5_decomposition["directly_measured_projector_mse"],
              rq5_decomposition["residual_energy"],
              rq5_decomposition["cross_term_2E_d_dot_delta"]))
    print("RQ5 P1->P2 relative parameter change={:.6g}, identical={}".format(
        rq5_parameter_delta["relative_parameter_delta_vs_task1"],
        rq5_parameter_delta["identical"]))
    print("RQ2 final-P2 signed cosine mean={:.6g}, mean_abs={:.6g}, negative={:.3%}".format(
        rq2["aggregate"]["mean"], rq2["aggregate"]["mean_abs"],
        rq2["aggregate"]["negative_fraction"]))
    print("RQ3 old covariance relative Frobenius mean={:.6g}, current sanity mean={:.6g}".format(
        rq3["aggregates"]["old_task2_vs_oracle"]["relative_frobenius"]["mean"],
        rq3["aggregates"]["current_task2_vs_oracle_sanity"]
        ["relative_frobenius"]["mean"]))
    no_shift_summary = rq1["comparison_summary"]["no_shift"]
    stored_summary = rq1["comparison_summary"]["historical_stored_task2_ssca"]
    print("RQ1 no-shift old-mean L2 mean/median/max={:.6g}/{:.6g}/{:.6g}".format(
        no_shift_summary["l2_mean"], no_shift_summary["l2_median"],
        no_shift_summary["l2_max"]))
    print("Historical task2 SSCA old-mean L2 mean/median/max={:.6g}/{:.6g}/{:.6g}".format(
        stored_summary["l2_mean"], stored_summary["l2_median"],
        stored_summary["l2_max"]))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(2)
