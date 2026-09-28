"""Statistical helpers for the paired QKSR confirmatory protocol."""

from typing import Dict, Mapping, Sequence

import numpy as np
from scipy import stats


def paired_comparison(
    treatment: Sequence[float],
    control: Sequence[float],
    confidence: float = 0.95,
    bootstrap_resamples: int = 10000,
    seed: int = 0,
    minimum_effect: float = 0.3,
) -> Dict[str, object]:
    """Summarize paired seed results with a percentile bootstrap CI and t-test."""
    treatment = np.asarray(treatment, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if treatment.shape != control.shape or treatment.ndim != 1:
        raise ValueError("treatment and control must be one-dimensional paired arrays.")
    if treatment.size < 2:
        raise ValueError("At least two paired seeds are required.")
    if not np.isfinite(treatment).all() or not np.isfinite(control).all():
        raise ValueError("Paired results must be finite.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")
    if bootstrap_resamples < 1000:
        raise ValueError("Use at least 1000 bootstrap resamples.")

    differences = treatment - control
    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0, differences.size, size=(bootstrap_resamples, differences.size)
    )
    bootstrap_means = differences[sampled_indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    t_result = stats.ttest_rel(treatment, control)
    mean_difference = float(differences.mean())
    return {
        "n": int(differences.size),
        "paired_differences": differences.tolist(),
        "mean_difference": mean_difference,
        "ci": [float(lower), float(upper)],
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_pvalue": float(t_result.pvalue),
        "minimum_effect": float(minimum_effect),
        "supports_improvement": bool(lower > 0.0 and mean_difference >= minimum_effect),
    }


def holm_adjust(pvalues: Mapping[str, float]) -> Dict[str, float]:
    """Return Holm step-down adjusted p-values keyed like the input mapping."""
    if not pvalues:
        return {}
    items = sorted((name, float(value)) for name, value in pvalues.items())
    if any(not 0.0 <= value <= 1.0 for _, value in items):
        raise ValueError("p-values must be in [0, 1].")
    items.sort(key=lambda item: item[1])
    count = len(items)
    adjusted = {}
    running_max = 0.0
    for rank, (name, value) in enumerate(items):
        corrected = min(1.0, (count - rank) * value)
        running_max = max(running_max, corrected)
        adjusted[name] = running_max
    return adjusted
