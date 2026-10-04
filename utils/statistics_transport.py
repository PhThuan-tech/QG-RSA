"""Guarded, exemplar-free transport using paired CURRENT-training features.

This is a local residual-ridge ablation inspired by drift-compensation work,
not an implementation of LDC or MACIL. No old images/test labels are needed.
"""

import math

import torch


def estimate_gaussian_statistics(vectors, shrinkage=0.0, jitter=1e-3):
    values = torch.as_tensor(vectors, dtype=torch.float64)
    if values.ndim != 2 or not len(values) or not torch.isfinite(values).all():
        raise ValueError("Class features must be a nonempty finite [N,D] matrix.")
    if not 0 <= shrinkage <= 1 or not math.isfinite(jitter) or jitter <= 0:
        raise ValueError("Invalid covariance shrinkage or jitter.")
    mean = values.mean(0)
    # torch.cov divides by N-1. A singleton must not create NaN memory.
    covariance = torch.cov(values.T) if len(values) > 1 else values.new_zeros(values.shape[1], values.shape[1])
    covariance = covariance.reshape(values.shape[1], values.shape[1])
    covariance = (1 - shrinkage) * covariance + shrinkage * torch.diag(covariance.diagonal())
    covariance = (covariance + covariance.T) / 2
    covariance += jitter * torch.eye(len(mean), dtype=values.dtype, device=values.device)
    return mean, covariance


def _fit_residual_map(old, new, rank, ridge, max_change):
    center = old.mean(0)
    delta = new - old
    shift = delta.mean(0)
    centered = old - center
    _, singular, right = torch.linalg.svd(centered, full_matrices=False)
    available = int((singular > singular[0].clamp_min(1e-12) * 1e-6).sum())
    rank = min(rank, available, len(old) - 1, old.shape[1])
    basis = right[:rank].T
    coordinates = centered @ basis
    if rank:
        gram = coordinates.T @ coordinates
        regularizer = ridge * (gram.trace() / rank).clamp_min(1e-12)
        coefficient = torch.linalg.solve(
            gram + regularizer * torch.eye(rank, dtype=old.dtype, device=old.device),
            coordinates.T @ (delta - shift),
        )
        # ||basis @ coefficient||_2 = ||coefficient||_2 for orthonormal basis.
        operator_norm = torch.linalg.matrix_norm(coefficient, ord=2)
        coefficient *= torch.clamp(max_change / operator_norm.clamp_min(1e-12), max=1.0)
    else:
        coefficient = old.new_empty((0, old.shape[1]))
    return center, shift, basis, coefficient


def _map_features(features, fit):
    center, shift, basis, coefficient = fit
    return features + shift + ((features - center) @ basis) @ coefficient


@torch.no_grad()
def transport_gaussian_statistics(
    means, covariances, old_features, new_features, *, rank=32, ridge=0.01,
    max_change=0.25, support_scale=1.0, support_floor=0.05,
    validation_fraction=0.2, seed=1993, jitter=1e-4, chunk_size=8,
):
    """Transport mean AND covariance, rejecting unsupported/unvalidated maps.

    Row-vector map: T(x)=x+shift+(x-center)@V@R. For class trust a,
    M=I+a*V@R and Sigma'=M.T@Sigma@M. Use the low-rank expansion to avoid
    cubic covariance transforms. Internal holdout is from current TRAIN only.
    """
    if int(rank) != rank or rank < 1 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("Transport requires rank>=1 and positive finite ridge.")
    if not 0 < max_change < 1 or not math.isfinite(support_scale) or support_scale <= 0:
        raise ValueError("max_change must be in (0,1); support_scale must be positive.")
    if (not 0 < validation_fraction < 1 or not 0 <= support_floor <= 1
            or int(chunk_size) != chunk_size or chunk_size < 1 or not math.isfinite(jitter) or jitter <= 0):
        raise ValueError("Invalid transport holdout/support/chunk/jitter.")
    old = torch.as_tensor(old_features).detach().to(dtype=torch.float64)
    new = torch.as_tensor(new_features, device=old.device).detach().to(dtype=torch.float64)
    original_cov = torch.as_tensor(covariances)
    mean = torch.as_tensor(means, device=old.device, dtype=torch.float64)
    if old.ndim != 2 or old.shape != new.shape or mean.ndim != 2 or mean.shape[1] != old.shape[1]:
        raise ValueError("Transport feature/mean shapes do not match.")
    if original_cov.shape != (len(mean), old.shape[1], old.shape[1]):
        raise ValueError("Transport covariance shape does not match means.")
    if not all(torch.isfinite(value).all() for value in (old, new, mean, original_cov)):
        raise ValueError("Transport inputs must be finite.")
    info = {"accepted": False, "reason": "insufficient_pairs", "pairs": len(old)}
    if len(old) < 4 or not len(mean):
        return mean.cpu().numpy(), original_cov.clone(), info
    identity_mse = (new - old).square().mean()
    if identity_mse <= 1e-12:
        info["reason"] = "no_drift"
        return mean.cpu().numpy(), original_cov.clone(), info

    # Private CPU RNG: fitting/probes cannot shift augmentation/replay RNG.
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(len(old), generator=generator).to(old.device)
    holdout_count = min(max(1, int(round(len(old) * validation_fraction))), len(old) - 2)
    held_out, training = order[:holdout_count], order[holdout_count:]
    trial = _fit_residual_map(old[training], new[training], rank, ridge, max_change)
    trial_error = (_map_features(old[held_out], trial) - new[held_out]).square().mean()
    reference_error = (old[held_out] - new[held_out]).square().mean()
    gain = (1 - trial_error / reference_error.clamp_min(1e-12)).clamp(0, 1)
    info.update({"validation_map_mse": float(trial_error), "validation_identity_mse": float(reference_error)})
    if not torch.isfinite(gain) or gain <= 0:
        info["reason"] = "holdout_not_better_than_identity"
        return mean.cpu().numpy(), original_cov.clone(), info

    fit = _fit_residual_map(old, new, rank, ridge, max_change)
    center, shift, basis, coefficient = fit
    spread = (old - center).square().sum(1).median().clamp_min(1e-8)
    updated_means = mean.clone()
    updated_cov = original_cov.clone()
    trusts = []
    for start in range(0, len(mean), chunk_size):
        current_mean = mean[start:start + chunk_size]
        # Nearest current-training anchor, not nearest test/old image.
        distances = torch.cdist(current_mean, old).square().min(1).values
        trust = gain * torch.exp(-distances / (2 * support_scale ** 2 * spread))
        trust = torch.where(trust >= support_floor, trust, torch.zeros_like(trust))
        trusts.append(trust)
        updated_means[start:start + chunk_size] = current_mean + trust[:, None] * (
            shift + ((current_mean - center) @ basis) @ coefficient
        )
        covariance = original_cov[start:start + chunk_size].to(device=old.device, dtype=old.dtype)
        # Low-rank expansion of M.T Sigma M: O(C*D^2*rank), not O(C*D^3).
        cov_basis = covariance @ basis
        cross = cov_basis @ coefficient
        correction = coefficient.T @ (basis.T @ cov_basis) @ coefficient
        candidate = covariance + trust[:, None, None] * (cross + cross.transpose(-1, -2))
        candidate += trust[:, None, None].square() * correction
        candidate = (candidate + candidate.transpose(-1, -2)) / 2
        candidate += (trust > 0)[:, None, None] * jitter * torch.eye(
            old.shape[1], dtype=old.dtype, device=old.device
        )
        updated_cov[start:start + chunk_size] = candidate.to(updated_cov)
    trusts = torch.cat(trusts)
    info.update({
        "accepted": bool((trusts > 0).any()), "reason": "guarded_ridge",
        "rank": basis.shape[1], "validation_gain": float(gain),
        "trust_min": float(trusts.min()), "trust_mean": float(trusts.mean()),
        "trust_max": float(trusts.max()), "supported_classes": int((trusts > 0).sum()),
    })
    return updated_means.cpu().numpy(), updated_cov, info
