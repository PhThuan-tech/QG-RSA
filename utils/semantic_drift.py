"""Opt-in, non-mutating representation-drift diagnostics."""

import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


def _json_default(value):
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def _pair_metrics(pre, post, eps=1e-12):
    cosine_similarity = F.cosine_similarity(pre, post, dim=1, eps=eps)
    cosine_distance = 1.0 - cosine_similarity
    l2 = torch.linalg.vector_norm(post - pre, dim=1)
    pre_norm = torch.linalg.vector_norm(pre, dim=1)
    post_norm = torch.linalg.vector_norm(post, dim=1)
    norm_ratio = post_norm / (pre_norm + eps)
    return {
        "cosine": cosine_distance.cpu().numpy(),
        "l2": l2.cpu().numpy(),
        "relative_l2": (l2 / (pre_norm + eps)).cpu().numpy(),
        "norm_ratio": norm_ratio.cpu().numpy(),
        "abs_log_norm_ratio": torch.abs(torch.log(norm_ratio.clamp_min(eps))).cpu().numpy(),
        "angle_deg": torch.rad2deg(
            torch.acos(cosine_similarity.clamp(-1.0, 1.0))
        ).cpu().numpy(),
    }


def _mean_covariance(features):
    mean = features.mean(dim=0)
    if features.shape[0] > 1:
        covariance = torch.cov(features.T)
    else:
        covariance = torch.zeros(
            (features.shape[1], features.shape[1]), dtype=features.dtype
        )
    return mean, covariance


class SemanticDriftObserver:
    """Measures f_(t-1)(x) -> f_t(x) without gradients or model updates."""

    def __init__(self, learner):
        self.learner = learner
        self.config = dict(learner.args.get("semantic_drift", {}))
        self.enabled = bool(self.config.get("enabled", False))
        self.flags = {
            "compute_sample_metrics": bool(self.config.get("compute_sample_metrics", True)),
            "compute_prototype_metrics": bool(self.config.get("compute_prototype_metrics", True)),
            "compute_distribution_metrics": bool(self.config.get("compute_distribution_metrics", True)),
            "compute_separation_metrics": bool(self.config.get("compute_separation_metrics", True)),
            "compute_relational_metrics": bool(self.config.get("compute_relational_metrics", True)),
            "compute_ssca_diagnostic": bool(self.config.get("compute_ssca_diagnostic", True)),
            "save_per_class": bool(self.config.get("save_per_class", True)),
            "save_per_sample": bool(self.config.get("save_per_sample", False)),
            "compute_full_covariance": bool(self.config.get("compute_full_covariance", False)),
        }
        self._pre = None
        self._post = None
        self._task_result = None
        if self.enabled:
            root = Path(learner.args.get("output_root", "")) / "semantic_drift"
            self.output_dir = root / str(learner.args["prefix"])
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "config_snapshot.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(
                    dict(learner.args),
                    stream,
                    indent=2,
                    sort_keys=True,
                    default=_json_default,
                )

    def _rng_snapshot(self):
        return (
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            np.random.get_state(),
            random.getstate(),
        )

    def _restore_rng(self, state):
        torch.set_rng_state(state[0])
        if state[1] is not None:
            torch.cuda.set_rng_state_all(state[1])
        np.random.set_state(state[2])
        random.setstate(state[3])

    def _loader_features(self, loader, model):
        was_training = model.training
        model.eval()
        features, labels, ids = [], [], []
        try:
            with torch.no_grad():
                for sample_ids, inputs, targets in loader:
                    features.append(
                        model.extract_vector(
                            inputs.to(self.learner._device, non_blocking=True)
                        ).detach().cpu()
                    )
                    labels.append(targets.detach().cpu())
                    ids.append(sample_ids.detach().cpu())
        finally:
            model.train(was_training)
        if not features:
            raise RuntimeError("Semantic drift probe set is empty.")
        return torch.cat(features), torch.cat(labels), torch.cat(ids)

    def _probe_loader(self, dataset):
        max_per_class = self.config.get("max_samples_per_class")
        selected = dataset
        if max_per_class is not None:
            selected_indices, counts = [], {}
            for index in range(len(dataset)):
                label = int(dataset.labels[index])
                if counts.get(label, 0) < int(max_per_class):
                    selected_indices.append(index)
                    counts[label] = counts.get(label, 0) + 1
            selected = Subset(dataset, selected_indices)
        return DataLoader(
            selected,
            batch_size=self.learner.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def prepare(self, dataset, old_model):
        if not self.enabled or self.learner._cur_task < 1:
            return
        state = self._rng_snapshot()
        try:
            self._pre = self._loader_features(self._probe_loader(dataset), old_model)
        finally:
            self._restore_rng(state)

    def _validate_pair(self, pre_labels, post_labels, pre_ids, post_ids):
        if not torch.equal(pre_ids, post_ids) or not torch.equal(
            pre_labels, post_labels
        ):
            raise RuntimeError("Semantic drift probe pairing mismatch.")

    def measure_post(self, dataset, current_model):
        if not self.enabled or self._pre is None:
            return
        state = self._rng_snapshot()
        try:
            self._post = self._loader_features(
                self._probe_loader(dataset), current_model
            )
            pre_features, pre_labels, pre_ids = self._pre
            post_features, post_labels, post_ids = self._post
            self._validate_pair(pre_labels, post_labels, pre_ids, post_ids)
            old_limit = self.learner._known_classes
            old_mask = pre_labels < old_limit
            new_mask = ~old_mask
            result = {
                "task_id": int(self.learner._cur_task),
                "known_classes_before": int(old_limit),
                "new_classes": int(
                    self.learner._total_classes - old_limit
                ),
                "old_probe": self._probe_info(pre_labels, old_mask),
                "new_probe": self._probe_info(pre_labels, new_mask),
                "old_sample_drift": {},
                "new_task_shift": {},
                "old_new_drift_ratio": {},
                "old_prototype_drift": {},
                "old_distribution_drift": {},
                "old_relational_drift": {},
                "old_new_separation": {},
                "ssca_diagnostic": {},
                "prototype_memory_diagnostic": {},
                "evaluation": {},
            }
            if self.flags["compute_sample_metrics"]:
                metrics = _pair_metrics(pre_features, post_features)
                result["old_sample_drift"] = self._group_summary(metrics, old_mask)
                result["new_task_shift"] = self._group_summary(metrics, new_mask)
                result["old_new_drift_ratio"] = self._drift_ratios(
                    metrics, old_mask, new_mask
                )
                self._add_class_metrics(
                    result,
                    pre_features,
                    post_features,
                    pre_labels,
                    old_mask,
                    pre_ids,
                )
            self._add_geometry_metrics(
                result, pre_features, post_features, pre_labels, old_mask, new_mask
            )
            self._task_result = result
        finally:
            self._restore_rng(state)

    @staticmethod
    def _probe_info(labels, mask):
        selected = labels[mask]
        return {
            "num_classes": int(torch.unique(selected).numel()),
            "num_samples": int(selected.numel()),
            "class_ids": [int(value) for value in torch.unique(selected).tolist()],
        }

    @staticmethod
    def _group_summary(metrics, mask):
        indices = mask.numpy()
        return {name: _summary(values[indices]) for name, values in metrics.items()}

    @staticmethod
    def _drift_ratios(metrics, old_mask, new_mask):
        ratios = {}
        for name in ("cosine", "relative_l2"):
            old_values = metrics[name][old_mask.numpy()]
            new_values = metrics[name][new_mask.numpy()]
            if not old_values.size or not new_values.size:
                continue
            denominator = float(new_values.mean())
            if denominator <= 1e-12:
                logging.warning(
                    "Semantic drift %s ratio denominator is near zero.", name
                )
            ratios[name] = float(old_values.mean() / (denominator + 1e-12))
        return ratios

    def _add_class_metrics(self, result, pre, post, labels, mask, sample_ids):
        records = {}
        per_sample = []
        for class_id in torch.unique(labels[mask]).tolist():
            class_mask = labels == class_id
            values = _pair_metrics(pre[class_mask], post[class_mask])
            records[str(int(class_id))] = {
                "class_id": int(class_id),
                "sample_count": int(class_mask.sum()),
                "sample_cosine": _summary(values["cosine"]),
                "sample_relative_l2": _summary(values["relative_l2"]),
                "sample_l2": _summary(values["l2"]),
                "norm_ratio": _summary(values["norm_ratio"]),
                "abs_log_norm_ratio": _summary(values["abs_log_norm_ratio"]),
                "angle_deg": _summary(values["angle_deg"]),
            }
            if self.flags["save_per_sample"]:
                selected_indices = torch.where(class_mask)[0]
                for row, index in enumerate(selected_indices.tolist()):
                    per_sample.append(
                        {
                            "sample_id": int(sample_ids[index]),
                            "class_id": int(class_id),
                            "cosine": float(values["cosine"][row]),
                            "l2": float(values["l2"][row]),
                            "relative_l2": float(values["relative_l2"][row]),
                            "norm_ratio": float(values["norm_ratio"][row]),
                            "abs_log_norm_ratio": float(
                                values["abs_log_norm_ratio"][row]
                            ),
                            "angle_deg": float(values["angle_deg"][row]),
                        }
                    )
        values = {
            key: np.array([record[key]["mean"] for record in records.values()])
            for key in ("sample_cosine", "sample_relative_l2")
        }
        result["old_sample_drift"]["heterogeneity"] = {
            key: _summary(value) for key, value in values.items()
        }
        if self.flags["save_per_class"]:
            result["old_sample_drift"]["per_class"] = records
        if self.flags["save_per_sample"]:
            result["old_sample_drift"]["per_sample"] = per_sample

    def _add_geometry_metrics(self, result, pre, post, labels, old_mask, new_mask):
        old_ids = [int(value) for value in torch.unique(labels[old_mask]).tolist()]
        new_ids = [int(value) for value in torch.unique(labels[new_mask]).tolist()]
        old_pre, old_post = [], []
        old_cov_pre, old_cov_post = [], []
        for class_id in old_ids:
            selected = labels == class_id
            mean_pre = pre[selected].mean(dim=0)
            mean_post = post[selected].mean(dim=0)
            old_pre.append(mean_pre)
            old_post.append(mean_post)
            if self.flags["compute_distribution_metrics"] and self.flags["compute_full_covariance"]:
                _, cov_pre = _mean_covariance(pre[selected])
                _, cov_post = _mean_covariance(post[selected])
                old_cov_pre.append(cov_pre)
                old_cov_post.append(cov_post)
        if not old_ids:
            return
        old_pre = torch.stack(old_pre)
        old_post = torch.stack(old_post)
        if self.flags["compute_prototype_metrics"]:
            proto = _pair_metrics(old_pre, old_post)
            result["old_prototype_drift"] = {
                "cosine": _summary(proto["cosine"]),
                "l2": _summary(proto["l2"]),
                "relative_l2": _summary(proto["relative_l2"]),
                "norm_ratio": _summary(proto["norm_ratio"]),
            }
            if self.flags["save_per_class"]:
                result["old_prototype_drift"]["per_class"] = {
                    str(class_id): {
                        "class_id": class_id,
                        "prototype_cosine_drift": float(proto["cosine"][index]),
                        "prototype_l2": float(proto["l2"][index]),
                        "prototype_relative_l2": float(proto["relative_l2"][index]),
                        "prototype_norm_ratio": float(proto["norm_ratio"][index]),
                    }
                    for index, class_id in enumerate(old_ids)
                }
        if self.flags["compute_distribution_metrics"]:
            variances_pre = torch.stack([
                pre[labels == class_id].var(dim=0, unbiased=False)
                for class_id in old_ids
            ])
            variances_post = torch.stack([
                post[labels == class_id].var(dim=0, unbiased=False)
                for class_id in old_ids
            ])
            trace_pre = variances_pre.sum(dim=1)
            trace_post = variances_post.sum(dim=1)
            diag_drift = torch.linalg.vector_norm(
                variances_post - variances_pre, dim=1
            ) / (torch.linalg.vector_norm(variances_pre, dim=1) + 1e-12)
            trace_ratio = trace_post / (trace_pre + 1e-12)
        else:
            trace_pre = trace_post = diag_drift = trace_ratio = None
        dispersion_pre = torch.tensor(
            [
                ((pre[labels == class_id] - old_pre[index]) ** 2)
                .sum(dim=1)
                .mean()
                for index, class_id in enumerate(old_ids)
            ]
        )
        dispersion_post = torch.tensor(
            [
                ((post[labels == class_id] - old_post[index]) ** 2)
                .sum(dim=1)
                .mean()
                for index, class_id in enumerate(old_ids)
            ]
        )
        if self.flags["compute_distribution_metrics"]:
            dispersion_ratio = dispersion_post / (dispersion_pre + 1e-12)
            distribution = {
                "trace_ratio": _summary(trace_ratio.numpy()),
                "diagonal_covariance_drift": _summary(diag_drift.numpy()),
                "within_class_dispersion_ratio": _summary(dispersion_ratio.numpy()),
            }
            if self.flags["save_per_class"]:
                distribution["per_class"] = {
                    str(class_id): {
                        "class_id": class_id,
                        "trace_ratio": float(trace_ratio[index]),
                        "diagonal_covariance_drift": float(diag_drift[index]),
                        "within_class_dispersion_ratio": float(dispersion_ratio[index]),
                    }
                    for index, class_id in enumerate(old_ids)
                }
            result["old_distribution_drift"] = distribution
        if self.flags["compute_relational_metrics"]:
            self._add_relational_metrics(result, old_pre, old_post, old_ids)
        if self.flags["compute_separation_metrics"]:
            self._add_separation_metrics(
                result, old_pre, old_post, pre, post, labels, old_ids, new_ids
            )
        if self.flags["compute_full_covariance"]:
            result["_old_probe_covariances"] = {
                "pre": [value.numpy().tolist() for value in old_cov_pre],
                "post": [value.numpy().tolist() for value in old_cov_post],
                "class_ids": old_ids,
            }

    def _add_relational_metrics(self, result, pre_means, post_means, class_ids):
        if len(class_ids) < 2:
            result["old_relational_drift"] = {"pair_count": 0}
            return
        pre_gram = F.normalize(pre_means, dim=1) @ F.normalize(
            pre_means, dim=1
        ).T
        post_gram = F.normalize(post_means, dim=1) @ F.normalize(
            post_means, dim=1
        ).T
        upper = torch.triu(torch.ones_like(pre_gram, dtype=torch.bool), diagonal=1)
        difference = post_gram - pre_gram
        result["old_relational_drift"] = {
            "pair_count": int(upper.sum()),
            "normalized_frobenius": float(
                torch.linalg.vector_norm(difference)
                / (torch.linalg.vector_norm(pre_gram) + 1e-12)
            ),
            "absolute_cosine_change": _summary(
                torch.abs(difference[upper]).numpy()
            ),
            "class_ids": class_ids,
        }

    def _add_separation_metrics(
        self, result, old_pre, old_post, pre, post, labels, old_ids, new_ids
    ):
        if not new_ids:
            result["old_new_separation"] = {"pair_count": 0}
            return
        new_pre = torch.stack(
            [pre[labels == class_id].mean(0) for class_id in new_ids]
        )
        new_post = torch.stack(
            [post[labels == class_id].mean(0) for class_id in new_ids]
        )
        pre_similarity = F.normalize(old_pre, dim=1) @ F.normalize(
            new_pre, dim=1
        ).T
        post_similarity = F.normalize(old_post, dim=1) @ F.normalize(
            new_post, dim=1
        ).T
        pre_nearest = pre_similarity.max(dim=0).values
        post_nearest = post_similarity.max(dim=0).values
        result["old_new_separation"] = {
            "pair_count": int(len(old_ids) * len(new_ids)),
            "interpretation": "higher cosine similarity means less separation",
            "pre_pairwise_cosine": _summary(pre_similarity.flatten().numpy()),
            "post_pairwise_cosine": _summary(post_similarity.flatten().numpy()),
            "delta_pairwise_cosine": _summary(
                (post_similarity - pre_similarity).flatten().numpy()
            ),
            "pre_max_old_cosine_per_new": _summary(pre_nearest.numpy()),
            "post_max_old_cosine_per_new": _summary(post_nearest.numpy()),
            "delta_max_old_cosine_per_new": _summary(
                (post_nearest - pre_nearest).numpy()
            ),
            "new_class_ids": new_ids,
        }

    def record_ssca(self, means_before, means_after):
        if (
            not self.enabled
            or not self.flags["compute_ssca_diagnostic"]
            or self._task_result is None
            or self._pre is None
        ):
            return
        pre_features, pre_labels, _ = self._pre
        post_features, post_labels, _ = self._post
        class_ids = [
            int(value)
            for value in torch.unique(pre_labels[pre_labels < self.learner._known_classes]).tolist()
        ]
        true_pre = torch.stack(
            [pre_features[pre_labels == class_id].mean(0) for class_id in class_ids]
        )
        true_post = torch.stack(
            [post_features[post_labels == class_id].mean(0) for class_id in class_ids]
        )
        before = torch.as_tensor(means_before[class_ids], dtype=true_post.dtype)
        after = torch.as_tensor(means_after[class_ids], dtype=true_post.dtype)
        true_shift = true_post - true_pre
        correction = after - before
        direction = F.cosine_similarity(true_shift, correction, dim=1, eps=1e-12)
        true_norm = torch.linalg.vector_norm(true_shift, dim=1)
        correction_norm = torch.linalg.vector_norm(correction, dim=1)
        before_error = torch.linalg.vector_norm(before - true_post, dim=1)
        after_error = torch.linalg.vector_norm(after - true_post, dim=1)
        correction_error = torch.linalg.vector_norm(
            correction - true_shift, dim=1
        )
        magnitude_ratio = correction_norm / (true_norm + 1e-12)
        result = self._task_result["ssca_diagnostic"] = {
            "oracle": "empirical post-task probe mean",
            "direction_agreement": _summary(direction.numpy()),
            "magnitude_ratio": _summary(magnitude_ratio.numpy()),
            "correction_error": _summary(correction_error.numpy()),
            "before_error": _summary(before_error.numpy()),
            "after_error": _summary(after_error.numpy()),
            "improvement": _summary((before_error - after_error).numpy()),
            "per_class": {},
        }
        for index, class_id in enumerate(class_ids):
            result["per_class"][str(class_id)] = {
                "class_id": class_id,
                "true_mean_shift_norm": float(true_norm[index]),
                "ssca_shift_norm": float(correction_norm[index]),
                "ssca_direction_agreement": float(direction[index]),
                "ssca_magnitude_ratio": float(magnitude_ratio[index]),
                "ssca_correction_error": float(correction_error[index]),
                "before_error": float(before_error[index]),
                "after_error": float(after_error[index]),
                "improvement": float(before_error[index] - after_error[index]),
            }
        self._task_result["prototype_memory_diagnostic"] = {
            "stored_before_vs_empirical_pre": _summary(
                torch.linalg.vector_norm(before - true_pre, dim=1).numpy()
            ),
            "stored_before_vs_empirical_post": _summary(before_error.numpy()),
            "stored_after_vs_empirical_post": _summary(after_error.numpy()),
            "oracle": "empirical post-task probe mean",
        }
        covariance_info = self._task_result.get("_old_probe_covariances")
        if covariance_info is not None and hasattr(self.learner, "_class_covs"):
            empirical_cov = torch.as_tensor(
                covariance_info["post"], dtype=true_post.dtype
            )
            stored_cov = self.learner._class_covs[class_ids].detach().cpu().to(
                dtype=true_post.dtype
            )
            cov_delta = torch.linalg.vector_norm(
                stored_cov - empirical_cov, dim=(1, 2)
            )
            cov_ref = torch.linalg.vector_norm(empirical_cov, dim=(1, 2))
            diag_delta = torch.linalg.vector_norm(
                torch.diagonal(stored_cov, dim1=1, dim2=2)
                - torch.diagonal(empirical_cov, dim1=1, dim2=2),
                dim=1,
            )
            diag_ref = torch.linalg.vector_norm(
                torch.diagonal(empirical_cov, dim1=1, dim2=2), dim=1
            )
            task_size = int(self.learner.args.get("increment", 10))
            current_task = int(self.learner._cur_task)
            effective_means = []
            for class_id, mean in zip(class_ids, after):
                task_id = int(class_id) // task_size
                decay = (task_id + 1) / (current_task + 1) * 0.1
                effective_means.append(mean * (0.9 + decay))
            effective_means = torch.stack(effective_means)
            self._task_result["prototype_memory_diagnostic"].update(
                {
                    "stored_covariance_vs_empirical_post_frobenius": _summary(
                        cov_delta.numpy()
                    ),
                    "stored_covariance_vs_empirical_post_relative_frobenius": _summary(
                        (cov_delta / (cov_ref + 1e-12)).numpy()
                    ),
                    "stored_covariance_vs_empirical_post_diagonal": _summary(
                        (diag_delta / (diag_ref + 1e-12)).numpy()
                    ),
                    "effective_ca_mean_vs_empirical_post": _summary(
                        torch.linalg.vector_norm(
                            effective_means - true_post, dim=1
                        ).numpy()
                    ),
                    "ca_mean_scaling": [
                        float(
                            0.9
                            + (
                                (int(class_id) // task_size + 1)
                                / (current_task + 1)
                                * 0.1
                            )
                        )
                        for class_id in class_ids
                    ],
                }
            )

    def record_parameter_audit(self, model):
        if not self.enabled:
            return
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        self._parameter_audit = {
            "total_parameters": int(total),
            "trainable_parameters": int(trainable),
            "trainable_parameter_names": [
                name for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ],
        }

    def record_evaluation(self, evaluation):
        if not self.enabled or self._task_result is None:
            return
        grouped = evaluation.get("grouped", {})
        self._task_result["evaluation"] = {
            "overall_accuracy": evaluation.get("top1"),
            "top5_accuracy": evaluation.get("top5"),
            "old_class_accuracy": grouped.get("old"),
            "new_class_accuracy": grouped.get("new"),
            "grouped": dict(grouped),
        }
        self.write_task()

    def write_task(self):
        if not self.enabled or self._task_result is None:
            return
        result = dict(self._task_result)
        if hasattr(self, "_parameter_audit"):
            result["parameter_audit"] = self._parameter_audit
        result.pop("_old_probe_means", None)
        result.pop("_old_probe_covariances", None)
        path = self.output_dir / "task_{:02d}.json".format(
            self.learner._cur_task
        )
        with path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
        summary_path = self.output_dir / "summary.csv"
        row = {
            "task_id": result["task_id"],
            "old_cosine_mean": result["old_sample_drift"]["cosine"].get("mean"),
            "old_relative_l2_mean": result["old_sample_drift"]["relative_l2"].get("mean"),
            "new_cosine_mean": result["new_task_shift"]["cosine"].get("mean"),
            "new_relative_l2_mean": result["new_task_shift"]["relative_l2"].get("mean"),
            "old_new_cosine_ratio": result["old_new_drift_ratio"].get("cosine"),
            "old_new_relative_l2_ratio": result["old_new_drift_ratio"].get("relative_l2"),
            "ssca_correction_error": result["ssca_diagnostic"].get("correction_error", {}).get("mean"),
            "ssca_direction_agreement": result["ssca_diagnostic"].get("direction_agreement", {}).get("mean"),
        }
        write_header = not summary_path.exists()
        with summary_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        logging.info("Semantic drift diagnostics written to %s", path)
