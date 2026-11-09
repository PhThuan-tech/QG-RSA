import math
import warnings

import torch
import torch.nn as nn

from KeepLora.peft_modules import gradient_svd


class MOnlyKeepLoRA(nn.Module):
    """Gradient-initialized low-rank update constrained to a feature residual."""

    def __init__(self, in_features, out_features, rank, scaling):
        super().__init__()
        if rank < 1:
            raise ValueError("M-only KeepLoRA rank must be positive.")
        if not math.isfinite(float(scaling)) or scaling <= 0:
            raise ValueError("M-only KeepLoRA scaling must be finite and positive.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.scaling = float(scaling)
        self.register_buffer(
            "lora_A", torch.empty(self.in_features, self.rank), persistent=True
        )
        self.lora_B = nn.Parameter(torch.empty(self.rank, self.out_features))

    def forward(self, inputs):
        return self.scaling * ((inputs @ self.lora_A) @ self.lora_B)

    def get_delta_weight(self):
        return self.scaling * (self.lora_A @ self.lora_B).transpose(0, 1)

    def merge_into(self, weight):
        weight.data.add_(self.get_delta_weight().to(weight.device, weight.dtype))

    @classmethod
    def from_gradient(
        cls,
        gradient,
        previous_basis,
        rank_cap,
        scaling_mode,
        target_scaling,
        lora_alpha,
        empty_residual_policy,
        rank_tolerance=None,
    ):
        if gradient.ndim != 2:
            raise ValueError("Expected a 2-D Linear weight gradient.")
        if rank_cap < 1:
            raise ValueError("rank_cap must be positive.")
        if scaling_mode not in ("fixed_effective_scaling", "alpha_over_rank"):
            raise ValueError(
                "scaling_mode must be 'fixed_effective_scaling' or 'alpha_over_rank'."
            )
        if empty_residual_policy not in ("no_update", "error"):
            raise ValueError(
                "empty_residual_policy must be 'no_update' or 'error'."
            )

        full_gradient = gradient.detach().transpose(0, 1).float()
        if not torch.isfinite(full_gradient).all():
            raise ValueError("KeepLoRA initialization gradient contains NaN or Inf.")
        original_norm = torch.linalg.vector_norm(full_gradient)
        basis = previous_basis.to(full_gradient)
        if (
            basis.ndim != 2
            or basis.shape[0] != full_gradient.shape[0]
            or basis.shape[1] > full_gradient.shape[0]
        ):
            raise ValueError(
                "Feature basis dimension {} does not match Linear input dimension {}."
                .format(
                    tuple(basis.shape),
                    full_gradient.shape[0],
                )
            )
        residual_dimension = full_gradient.shape[0] - basis.shape[1]
        safe_coordinates = None
        projected = full_gradient
        if basis.numel():
            # SVD in residual coordinates cannot return a protected direction,
            # even when roundoff dominates the projected gradient.
            coordinates = torch.linalg.qr(basis, mode="complete").Q
            safe_coordinates = coordinates[:, basis.shape[1]:]
            projected = safe_coordinates.transpose(0, 1) @ full_gradient
        projected_norm = torch.linalg.vector_norm(projected)
        projection_tolerance = (
            torch.finfo(projected.dtype).eps * max(full_gradient.shape) * original_norm
            if basis.numel()
            else 0.0
        )

        stats = {
            "gradient_norm": float(original_norm.item()),
            "projected_gradient_norm": float(projected_norm.item()),
            "projected_gradient_ratio": float(
                (projected_norm / original_norm.clamp_min(1e-12)).item()
            ),
            "residual_dimension": int(max(residual_dimension, 0)),
            "configured_rank": int(rank_cap),
        }
        if residual_dimension == 0 or projected_norm <= projection_tolerance:
            if empty_residual_policy == "error":
                raise RuntimeError(
                    "M-only initialization has no useful residual gradient "
                    "(residual_dimension={}, projected_gradient_norm={:.8g})."
                    .format(residual_dimension, projected_norm.item())
                )
            warnings.warn(
                "M-only initialization produced no update; residual dimension "
                "or projected gradient is zero.",
                RuntimeWarning,
            )
            stats.update(
                actual_rank=0,
                scaling=0.0,
                alpha=0.0,
                top_rank_energy_ratio=0.0,
                a_orthogonality_error=0.0,
                a_orthonormality_error=0.0,
                effective_update_norm=0.0,
            )
            return None, stats

        feasible_rank = min(rank_cap, residual_dimension, full_gradient.shape[1])
        devices = [gradient.device] if gradient.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            u, singular_values, v = gradient_svd(projected, feasible_rank)
        if singular_values.numel() == 0 or singular_values[0] <= 0:
            numerical_rank = 0
        else:
            if rank_tolerance is None:
                tolerance = (
                    torch.finfo(singular_values.dtype).eps
                    * max(projected.shape)
                    * singular_values[0]
                )
            else:
                tolerance = float(rank_tolerance) * singular_values[0]
            numerical_rank = int((singular_values > tolerance).sum().item())
        actual_rank = min(feasible_rank, numerical_rank)
        if actual_rank < 1:
            if empty_residual_policy == "error":
                raise RuntimeError("M-only initialization gradient has numerical rank 0.")
            warnings.warn(
                "M-only initialization gradient has numerical rank 0; no update "
                "will be applied.",
                RuntimeWarning,
            )
            stats.update(
                actual_rank=0,
                scaling=0.0,
                alpha=0.0,
                top_rank_energy_ratio=0.0,
                a_orthogonality_error=0.0,
                a_orthonormality_error=0.0,
                effective_update_norm=0.0,
            )
            return None, stats

        if scaling_mode == "fixed_effective_scaling":
            scaling = float(target_scaling)
            alpha = scaling * actual_rank
        else:
            scaling = float(lora_alpha) / actual_rank
            alpha = float(lora_alpha)
        if not math.isfinite(scaling) or scaling <= 0:
            raise ValueError("Resolved M-only KeepLoRA scaling must be positive.")

        factor = cls(
            full_gradient.shape[0], full_gradient.shape[1], actual_rank, scaling
        ).to(device=gradient.device, dtype=gradient.dtype)
        directions = u[:, :actual_rank]
        if safe_coordinates is not None:
            directions = safe_coordinates @ directions
        factor.lora_A.copy_(directions.to(factor.lora_A))
        factor.lora_B.data.copy_(
            (singular_values[:actual_rank, None] * v[:, :actual_rank].T).to(
                factor.lora_B
            )
        )
        a_directions = factor.lora_A.float()
        if basis.numel():
            a_orthogonality_error = torch.linalg.matrix_norm(
                basis.transpose(0, 1) @ a_directions
            ).item()
        else:
            a_orthogonality_error = 0.0
        a_identity = torch.eye(
            actual_rank,
            device=a_directions.device,
            dtype=a_directions.dtype,
        )
        a_orthonormality_error = torch.linalg.matrix_norm(
            a_directions.transpose(0, 1) @ a_directions - a_identity
        ).item()
        retained_energy = singular_values[:actual_rank].square().sum()
        total_energy = projected_norm.square().clamp_min(1e-12)
        stats.update(
            actual_rank=actual_rank,
            scaling=scaling,
            alpha=alpha,
            top_rank_energy_ratio=float((retained_energy / total_energy).item()),
            a_orthogonality_error=float(a_orthogonality_error),
            a_orthonormality_error=float(a_orthonormality_error),
            effective_update_norm=float(
                torch.linalg.vector_norm(factor.get_delta_weight()).item()
            ),
        )
        return factor, stats


def update_feature_basis(moment, previous_basis, threshold):
    """Extend an orthonormal basis with residual task-feature directions."""
    if not 0 < float(threshold) <= 1:
        raise ValueError("feature threshold must be in (0, 1].")
    if moment.ndim != 2 or moment.shape[0] != moment.shape[1]:
        raise ValueError("Feature second moment must be a square matrix.")
    covariance = moment.detach().float()
    if not torch.isfinite(covariance).all():
        raise ValueError("Feature second moment contains NaN or Inf.")
    covariance = (covariance + covariance.transpose(0, 1)) * 0.5
    dimension = covariance.shape[0]
    previous = previous_basis.to(device=covariance.device, dtype=covariance.dtype)
    if (
        previous.ndim != 2
        or previous.shape[0] != dimension
        or previous.shape[1] > dimension
    ):
        raise ValueError("Previous feature basis has an incompatible dimension.")
    safe_coordinates = None
    total_energy = covariance.trace().clamp_min(0)
    previous_energy = covariance.new_zeros(())
    if previous.shape[1]:
        coordinates = torch.linalg.qr(previous, mode="complete").Q
        previous, safe_coordinates = coordinates.split(
            [previous.shape[1], dimension - previous.shape[1]], dim=1
        )
        previous_energy = (previous.T @ covariance @ previous).trace().clamp_min(0)
        residual_covariance = safe_coordinates.T @ covariance @ safe_coordinates
    else:
        residual_covariance = covariance

    residual_energy = residual_covariance.trace().clamp_min(0)
    required_energy = float(threshold) * total_energy - previous_energy
    new_basis = covariance.new_empty((dimension, 0))
    captured_energy = previous_energy
    if residual_covariance.numel() and required_energy > 0 and residual_energy > 0:
        eigenvalues, eigenvectors = torch.linalg.eigh(residual_covariance)
        eigenvalues = eigenvalues.flip(0).clamp_min(0)
        tolerance = torch.finfo(eigenvalues.dtype).eps * dimension * eigenvalues[0]
        numerical_rank = int((eigenvalues > tolerance).sum().item())
        if numerical_rank:
            cumulative = eigenvalues[:numerical_rank].cumsum(0)
            rank = min(
                int(torch.searchsorted(cumulative, required_energy).item()) + 1,
                numerical_rank,
            )
            new_basis = eigenvectors[:, -rank:].flip(1)
            if safe_coordinates is not None:
                new_basis = safe_coordinates @ new_basis
            captured_energy = captured_energy + cumulative[rank - 1]

    combined = torch.cat((previous, new_basis), dim=1)
    identity = torch.eye(
        combined.shape[1], device=combined.device, dtype=combined.dtype
    )
    orthogonality_error = (
        float(torch.linalg.matrix_norm(combined.transpose(0, 1) @ combined - identity).item())
        if combined.shape[1]
        else 0.0
    )
    cross_orthogonality_error = (
        float(
            torch.linalg.matrix_norm(
                previous.transpose(0, 1) @ new_basis
            ).item()
        )
        if previous.shape[1] and new_basis.shape[1]
        else 0.0
    )
    energy_denominator = torch.where(total_energy > 0, total_energy, 1.0)
    return combined, {
        "previous_rank": int(previous.shape[1]),
        "new_rank": int(new_basis.shape[1]),
        "rank": int(combined.shape[1]),
        "residual_dimension": int(dimension - combined.shape[1]),
        "captured_energy": (captured_energy / energy_denominator).clamp(0, 1).item(),
        "residual_energy_ratio": (residual_energy / energy_denominator).clamp(0, 1).item(),
        "orthogonality_error": orthogonality_error,
        "cross_orthogonality_error": cross_orthogonality_error,
    }
