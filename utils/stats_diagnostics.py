"""Old-class statistic estimators and oracle diagnostics for SSCA.

Legacy SSCA extracts the pre- and post-adaptation features from the shuffled,
augmented training loader and subtracts them row by row, so row ``i`` of the
two passes is generally a different image under a different crop. The
``paired`` mode extracts both passes from one fixed-order, deterministic-view
loader so that ``Y2[i] - Y1[i]`` is the drift of the same image.

The oracle reads old-class training images through the current encoder and
therefore violates the exemplar-free protocol. It is a diagnostic upper bound,
never a method result.
"""
import json
import logging
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

SSCA_MODES = ("legacy", "paired")
CA_STATS_SOURCES = ("estimated", "oracle_mean", "oracle_mean_cov")
# Same floors as models.base: kernel weight floor and covariance ridge.
WEIGHT_FLOOR = 1e-5
COV_RIDGE = 1e-3
# Per-task bandwidth: 2*sigma^2 = median over old classes of the squared
# distance to the nearest new-task sample, so the typical class gives its
# nearest sample weight e^-1 instead of underflowing below WEIGHT_FLOOR.
AUTO_SIGMA = "auto_nn"


def stats_settings(args):
    """Validate and resolve the SSCA/statistics configuration keys."""
    mode = args.get("ssca_mode", "legacy")
    if mode not in SSCA_MODES:
        raise ValueError("ssca_mode must be one of {}.".format(list(SSCA_MODES)))
    source = args.get("ca_stats_source", "estimated")
    if source not in CA_STATS_SOURCES:
        raise ValueError("ca_stats_source must be one of {}.".format(list(CA_STATS_SOURCES)))
    sigma = args.get("ssca_sigma", 4.0)
    if sigma != AUTO_SIGMA:
        sigma = float(sigma)
        if not (math.isfinite(sigma) and sigma > 0.0):
            raise ValueError("ssca_sigma must be a positive finite number or '{}'.".format(AUTO_SIGMA))
    if mode == "paired" and args.get("ssca") is not True:
        raise ValueError("ssca_mode='paired' requires ssca=true; otherwise no drift is applied.")
    diagnostics = bool(args.get("stats_diagnostics", False))
    replicate = args.get("rng_replicate", 0)
    if isinstance(replicate, bool) or not isinstance(replicate, int) or replicate < 0:
        raise ValueError("rng_replicate must be a nonnegative integer.")
    return {
        "rng_replicate": replicate,
        "mode": mode,
        "sigma": sigma,
        "diagnostics": diagnostics,
        "ca_stats_source": source,
        "needs_paired": mode == "paired" or diagnostics,
        "needs_oracle": diagnostics or source != "estimated",
    }


def ca_settings(args):
    """Validated CA options: (logit_norm or None, cross-task cosine margin)."""
    logit_norm = args.get("ca_logit_norm")
    if logit_norm is not None:
        logit_norm = float(logit_norm)
        if not (math.isfinite(logit_norm) and logit_norm > 0):
            raise ValueError("ca_logit_norm must be null or positive.")
    margin = float(args.get("ca_task_margin", 0.0))
    if not (math.isfinite(margin) and margin >= 0):
        raise ValueError("ca_task_margin must be nonnegative.")
    return logit_norm, margin


def deterministic_loader(dataset, batch_size, num_workers, pin_memory, seed):
    """Fixed-order loader whose private generator leaves the global RNG untouched."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=False, generator=generator)


def kernel_weights(Y1, means, sigma):
    """Row-normalized SSCA kernel weights ``[C_old, N]`` in float64."""
    Y1 = np.asarray(Y1, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    sq = (np.sum(means ** 2, axis=1)[:, None] + np.sum(Y1 ** 2, axis=1)[None, :]
          - 2.0 * means @ Y1.T)
    sq = np.maximum(sq, 0.0)
    weights = np.exp(-sq / (2.0 * sigma ** 2)) + WEIGHT_FLOOR
    return weights / weights.sum(axis=1, keepdims=True), sq


def resolve_sigma(sigma, Y1, means):
    """Numeric bandwidth; ``auto_nn`` is computed from the pre-adaptation features only."""
    if sigma != AUTO_SIGMA:
        return float(sigma)
    _, sq = kernel_weights(Y1, means, 1.0)
    value = math.sqrt(float(np.median(sq.min(axis=1))) / 2.0)
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError("auto_nn bandwidth is degenerate (zero nearest-sample distance).")
    return value


def kernel_displacement(Y1, Y2, means, sigma):
    """Matrix form of ``BaseLearner.displacement`` (equal up to float rounding)."""
    Y1 = np.asarray(Y1, dtype=np.float64)
    Y2 = np.asarray(Y2, dtype=np.float64)
    if Y1.shape != Y2.shape:
        raise ValueError("Pre/post feature arrays must have the same shape.")
    weights, _ = kernel_weights(Y1, means, sigma)
    return weights @ (Y2 - Y1)


def kernel_summary(Y1, means, sigma):
    """How local the SSCA kernel is: ESS near N means uniform, near 1 means nearest-sample."""
    weights, sq = kernel_weights(Y1, means, sigma)
    ess = 1.0 / np.sum(weights ** 2, axis=1)
    raw = np.exp(-sq / (2.0 * sigma ** 2))
    return {
        "sigma": float(sigma),
        "samples": int(weights.shape[1]),
        "effective_sample_size_mean": float(ess.mean()),
        "effective_sample_size_min": float(ess.min()),
        "nearest_sq_distance_median": float(np.median(sq.min(axis=1))),
        "sq_distance_median": float(np.median(sq)),
        "fraction_rows_floor_dominated": float(np.mean(raw.max(axis=1) < WEIGHT_FLOOR)),
        "feature_norm_median": float(np.median(np.linalg.norm(np.asarray(Y1, np.float64), axis=1))),
    }


def class_statistics(vectors):
    """Mean and covariance with exactly the conventions of ``_compute_class_mean``."""
    vectors = np.asarray(vectors)
    mean = np.mean(vectors, axis=0)
    cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(vectors.shape[1]) * COV_RIDGE
    return mean, cov


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _describe(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": None, "median": None, "max": None}
    return {"mean": float(values.mean()), "median": float(np.median(values)), "max": float(values.max())}


def mean_errors(estimate, oracle, stale):
    """Per-class error of an old-mean estimate against the oracle mean.

    ``relative`` divides by the oracle drift ``||stale - oracle||``: 1.0 means no
    better than skipping compensation, 0.0 means exact.
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    stale = np.asarray(stale, dtype=np.float64)
    l2 = np.linalg.norm(estimate - oracle, axis=1)
    drift = np.linalg.norm(stale - oracle, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(drift > 0, l2 / drift, np.nan)
        cosine = np.sum(estimate * oracle, axis=1) / (
            np.linalg.norm(estimate, axis=1) * np.linalg.norm(oracle, axis=1))
    return {"l2": l2, "relative": relative, "cosine": cosine}


def summarize_by_origin(per_class, origin_task):
    """Aggregate per-class metric arrays overall and by the task each class came from."""
    origin_task = np.asarray(origin_task)
    summary = {"all": {name: _describe(values) for name, values in per_class.items()}, "by_origin_task": {}}
    for task in sorted(set(origin_task.tolist())):
        mask = origin_task == task
        summary["by_origin_task"][str(task)] = {
            name: _describe(np.asarray(values)[mask]) for name, values in per_class.items()}
    return summary


def covariance_errors(stored, oracle):
    """Relative Frobenius error and trace ratio of one stored covariance."""
    stored = stored.to(torch.float64)
    oracle = oracle.to(torch.float64)
    return (float(torch.linalg.norm(stored - oracle) / torch.linalg.norm(oracle)),
            float(torch.trace(stored) / torch.trace(oracle)))


def origin_tasks(data_manager, task_count, known_classes):
    """Task index of every class position ``0..known_classes-1``."""
    origin = []
    for task in range(task_count):
        origin.extend([task] * int(data_manager.get_task_size(task)))
    if len(origin) < known_classes:
        raise ValueError("Task schedule does not cover all old classes.")
    return np.asarray(origin[:known_classes])


def reseed_replicate(args):
    """Start a noise-floor replicate: same base and method, different SGD randomness.

    Called once at the start of task 1, so task 0 (and a shared Q-BRIDGE base)
    stays identical while every later shuffle, augmentation and CA sample differs.
    """
    import random
    replicate = int(args.get("rng_replicate", 0))
    seed = int(args.get("train_seed", args.get("seed", 0))) + 7919 * replicate
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info("STATS rng_replicate=%d: reseeded task-1 onward with %d", replicate, seed)


def error_directions(pred, target, known_classes):
    """Old->new and new->old confusion rates of top-1 predictions."""
    pred, target = np.asarray(pred), np.asarray(target)
    old = target < known_classes
    return {
        "old_to_new_error_rate": float((pred[old] >= known_classes).mean()) if old.any() else None,
        "new_to_old_error_rate": float((pred[~old] < known_classes).mean()) if (~old).any() else None,
    }


def save_predictions(folder, task, pred, target, known_classes):
    """Top-1 predictions in the fixed test-loader order, for paired McNemar tests."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "predictions_task_%d.npz" % task)
    np.savez_compressed(path, pred=np.asarray(pred, dtype=np.int32),
                        target=np.asarray(target, dtype=np.int32), known_classes=int(known_classes))
    return path


def mcnemar(correct_a, correct_b):
    """Exact two-sided McNemar test on paired per-image correctness.

    Returns discordant counts b (only A right), c (only B right) and the
    binomial p-value. Summing b and c over seeds gives a stratified exact test
    because each seed's discordant pairs are independent given their totals.
    """
    correct_a, correct_b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError("McNemar needs predictions on the same test images.")
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    return {"only_a_correct": b, "only_b_correct": c, "p_value": binomial_two_sided(b, c)}


def binomial_two_sided(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def write_report(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (np.floating, float)):
            return _finite(value)
        if isinstance(value, np.integer):
            return int(value)
        return value

    with open(path, "w", encoding="utf-8") as output:
        json.dump(clean(payload), output, indent=2, allow_nan=False)
    logging.info("STATS report written: %s", path)
