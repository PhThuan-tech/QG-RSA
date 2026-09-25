"""Fail-closed utilities for the Track-B CA numerical rescue screen.

[NUMERICAL IMPLEMENTATION FIX] Old-class Gaussian residuals are factorized in
the source representation and pushed through the affine weight. No covariance
repair, clipping, jitter tuning, gradient clipping, or trainable parameter is
introduced.
"""
from __future__ import annotations

import math
import platform
import random
import sys
from typing import Any, Dict, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {
        "python", "numpy", "torch_cpu", "torch_cuda_all",
        "cudnn_deterministic", "cudnn_benchmark",
        "deterministic_algorithms", "cuda_matmul_allow_tf32", "cudnn_allow_tf32",
    }
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError("RNG snapshot missing fields: {}".format(missing))
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda_all"]
    if torch.cuda.is_available():
        if cuda_states is None or len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Captured CUDA RNG state does not match visible CUDA devices")
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states is not None:
        raise RuntimeError("Captured CUDA RNG exists but CUDA is unavailable")
    torch.backends.cudnn.deterministic = bool(state["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(state["cudnn_benchmark"])
    torch.use_deterministic_algorithms(bool(state["deterministic_algorithms"]), warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = bool(state["cuda_matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(state["cudnn_allow_tf32"])


def environment_record() -> Dict[str, Any]:
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception:
        torchvision_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def tensor_finite_fraction(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 1.0
    return float(torch.isfinite(tensor).sum().item() / tensor.numel())


def parameter_finite_fraction(module: nn.Module, gradients: bool = False) -> float:
    finite = 0
    total = 0
    for parameter in module.parameters():
        value = parameter.grad if gradients else parameter.detach()
        if value is None:
            continue
        finite += int(torch.isfinite(value).sum().item())
        total += int(value.numel())
    return 1.0 if total == 0 else float(finite / total)


def feature_summary(samples: torch.Tensor) -> Dict[str, Any]:
    samples = samples.detach()
    vector_finite = torch.isfinite(samples).all(dim=1)
    finite_vectors = samples[vector_finite].float()
    result: Dict[str, Any] = {
        "sample_count": int(samples.shape[0]),
        "sample_finite_fraction": float(vector_finite.float().mean().item()),
        "coordinate_finite_fraction": tensor_finite_fraction(samples),
        "first_nonfinite": None,
    }
    if not bool(vector_finite.all()):
        location = (~torch.isfinite(samples)).nonzero(as_tuple=False)[0].detach().cpu().tolist()
        value = float(samples[tuple(location)].detach().cpu().item())
        result["first_nonfinite"] = {
            "sample": int(location[0]),
            "coordinate": int(location[1]),
            "kind": "nan" if math.isnan(value) else "inf",
        }
    if finite_vectors.numel():
        norms = torch.linalg.vector_norm(finite_vectors, dim=1).detach().cpu().double()
        result.update({
            "norm_min": float(norms.min()),
            "norm_p50": float(torch.quantile(norms, 0.50)),
            "norm_mean": float(norms.mean()),
            "norm_p95": float(torch.quantile(norms, 0.95)),
            "norm_p99": float(torch.quantile(norms, 0.99)),
            "norm_max": float(norms.max()),
            "coordinate_abs_max": float(finite_vectors.abs().max()),
            "coordinate_std": float(finite_vectors.std(unbiased=False)),
        })
    else:
        result.update({key: None for key in (
            "norm_min", "norm_p50", "norm_mean", "norm_p95", "norm_p99",
            "norm_max", "coordinate_abs_max", "coordinate_std",
        )})
    return result


@torch.no_grad()
def sample_source_then_affine(
    source_mean: torch.Tensor,
    source_covariance: torch.Tensor,
    transported_target_mean: torch.Tensor,
    affine: nn.Linear,
    sample_count: int,
) -> Dict[str, Any]:
    """Sample source Gaussian and transform residual through affine W.

    ``raw_transformed = A(u)`` verifies the requested affine pushforward.
    ``ca_samples = transported_target_mean + W @ (u-source_mean)`` preserves the
    existing CA target mean, including its task-age multiplier, while avoiding
    factorization of W Sigma W^T. Both have covariance W Sigma W^T.
    """
    device = affine.weight.device
    source_mean = torch.as_tensor(source_mean, dtype=torch.float64, device=device)
    source_covariance = torch.as_tensor(source_covariance, device=device).float()
    target_mean = torch.as_tensor(transported_target_mean, device=device).float()
    distribution = MultivariateNormal(source_mean.float(), source_covariance)
    scale_tril = distribution.scale_tril
    if not bool(torch.isfinite(scale_tril).all()):
        bad = (~torch.isfinite(scale_tril)).nonzero(as_tuple=False)[0].cpu().tolist()
        raise RuntimeError("Source Gaussian scale_tril is non-finite at {}".format(bad))
    source_samples = distribution.sample((int(sample_count),))
    if not bool(torch.isfinite(source_samples).all()):
        raise RuntimeError("Source Gaussian samples contain NaN/Inf")
    raw_transformed = affine(source_samples)
    residual = source_samples - source_mean.float().unsqueeze(0)
    ca_samples = target_mean.unsqueeze(0) + F.linear(
        residual, affine.weight.detach(), bias=None)
    if not bool(torch.isfinite(raw_transformed).all()):
        raise RuntimeError("Raw affine-transformed samples contain NaN/Inf")
    if not bool(torch.isfinite(ca_samples).all()):
        raise RuntimeError("CA rescue samples contain NaN/Inf")
    return {
        "source_samples": source_samples,
        "raw_transformed": raw_transformed,
        "ca_samples": ca_samples,
        "source_scale_tril_finite": True,
        "source_scale_tril_diagonal_min": float(torch.diagonal(scale_tril).min().item()),
        "source_scale_tril_diagonal_max": float(torch.diagonal(scale_tril).max().item()),
    }


@torch.no_grad()
def affine_distribution_diagnostic(
    source_samples: torch.Tensor,
    raw_transformed: torch.Tensor,
    source_mean: torch.Tensor,
    source_covariance: torch.Tensor,
    affine: nn.Linear,
) -> Dict[str, Any]:
    """Compare empirical A(u) moments with exact affine Gaussian moments."""
    source_mean64 = torch.as_tensor(source_mean, dtype=torch.float64, device="cpu")
    source_cov64 = torch.as_tensor(source_covariance, dtype=torch.float64, device="cpu")
    weight64 = affine.weight.detach().cpu().double()
    bias64 = affine.bias.detach().cpu().double()
    analytic_mean = source_mean64 @ weight64.T + bias64
    analytic_covariance = weight64 @ source_cov64 @ weight64.T
    empirical = raw_transformed.detach().cpu().double()
    empirical_mean = empirical.mean(dim=0)
    empirical_covariance = torch.cov(empirical.T)
    mean_denominator = max(float(torch.linalg.vector_norm(analytic_mean)), 1e-12)
    covariance_denominator = max(float(torch.linalg.matrix_norm(analytic_covariance)), 1e-12)
    algebraic = source_samples.detach().cpu().double() @ weight64.T + bias64
    return {
        "sample_count": int(empirical.shape[0]),
        "analytic_mean_norm": float(torch.linalg.vector_norm(analytic_mean)),
        "empirical_mean_norm": float(torch.linalg.vector_norm(empirical_mean)),
        "empirical_mean_relative_l2_error": float(
            torch.linalg.vector_norm(empirical_mean - analytic_mean) / mean_denominator),
        "analytic_covariance_frobenius_norm": float(torch.linalg.matrix_norm(analytic_covariance)),
        "empirical_covariance_frobenius_norm": float(torch.linalg.matrix_norm(empirical_covariance)),
        "empirical_covariance_relative_frobenius_error": float(
            torch.linalg.matrix_norm(empirical_covariance - analytic_covariance)
            / covariance_denominator),
        "sample_transform_max_abs_error_float64_recompute": float(
            (empirical - algebraic).abs().max()),
    }
