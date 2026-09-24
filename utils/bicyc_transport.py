"""Utilities for the RSIAT + BiCyc-style transport experiment.

This is an implementation adaptation, not an exact BiCyc reproduction.
"""

import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


VALID_MODES = ("official", "forward", "bidirectional", "cycle")


@contextmanager
def preserve_global_rng_state():
    """Restore all process RNG states after an isolated operation."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def make_deterministic_affine(dimension, seed, device=None):
    """Construct PyTorch-default nn.Linear without perturbing global RNG state."""
    with preserve_global_rng_state():
        torch.manual_seed(int(seed))
        layer = nn.Linear(int(dimension), int(dimension), bias=True)
        if device is not None:
            layer = layer.to(device)
    return layer


def tensor_mapping_sha256(mapping):
    digest = hashlib.sha256()
    for key in sorted(mapping):
        tensor = torch.as_tensor(mapping[key]).detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def module_state_sha256(module):
    if module is None:
        return None
    return tensor_mapping_sha256(module.state_dict())


def same_input_feature_pair(current_extractor, old_extractor, inputs):
    """Extract z_new/z_old from the exact same already-augmented tensor."""
    z_new = current_extractor(inputs)
    with torch.no_grad():
        z_old = old_extractor(inputs)
    return z_new, z_old


def bicyc_loss_terms(z_new, z_old, forward_map=None, backward_map=None, mode="official"):
    """Return unweighted losses with the approved gradient routing."""
    if mode not in VALID_MODES:
        raise ValueError("Unknown BiCyc mode: {}".format(mode))
    zero = z_new.new_zeros(())
    terms = {
        "loss_a": zero,
        "loss_d": zero,
        "cycle_new": zero,
        "cycle_old": zero,
    }
    if mode == "official":
        return terms
    if forward_map is None:
        raise ValueError("Mode {} requires forward map A".format(mode))

    terms["loss_a"] = F.mse_loss(
        forward_map(z_old.detach()), z_new.detach())
    if mode in ("bidirectional", "cycle"):
        if backward_map is None:
            raise ValueError("Mode {} requires backward map D".format(mode))
        terms["loss_d"] = F.mse_loss(
            backward_map(z_new), z_old.detach())
    if mode == "cycle":
        terms["cycle_new"] = F.mse_loss(
            forward_map(backward_map(z_new).detach()), z_new.detach())
        terms["cycle_old"] = F.mse_loss(
            backward_map(forward_map(z_old.detach()).detach()), z_old.detach())
    return terms


def weighted_bicyc_loss(terms, mode, lambda_bi=5.0, lambda_cycle=1.0):
    if mode == "official":
        return terms["loss_a"]
    alignment = terms["loss_a"]
    if mode in ("bidirectional", "cycle"):
        alignment = alignment + terms["loss_d"]
    total = float(lambda_bi) * alignment
    if mode == "cycle":
        total = total + float(lambda_cycle) * (
            terms["cycle_new"] + terms["cycle_old"])
    return total


def analytic_affine_gaussian_transport(means, covariances, affine):
    """Exactly push Gaussian statistics through y = W x + b on CPU float64."""
    means_tensor = torch.as_tensor(means, dtype=torch.float64, device="cpu")
    covs_tensor = torch.as_tensor(covariances, dtype=torch.float64, device="cpu")
    weight = affine.weight.detach().cpu().to(torch.float64)
    bias = affine.bias.detach().cpu().to(torch.float64)
    transported_means = means_tensor @ weight.T + bias
    transported_covs = torch.matmul(
        torch.matmul(weight.unsqueeze(0), covs_tensor),
        weight.T.unsqueeze(0),
    )
    transported_covs = 0.5 * (
        transported_covs + transported_covs.transpose(-1, -2))
    if not torch.isfinite(transported_means).all():
        raise RuntimeError("Non-finite analytically transported means")
    if not torch.isfinite(transported_covs).all():
        raise RuntimeError("Non-finite analytically transported covariances")
    return transported_means, transported_covs


def affine_geometry(module):
    weight = module.weight.detach().cpu().to(torch.float64)
    bias = module.bias.detach().cpu().to(torch.float64)
    singular_values = torch.linalg.svdvals(weight)
    minimum = float(singular_values.min())
    maximum = float(singular_values.max())
    median = float(singular_values.median())
    return {
        "weight_frobenius_norm": float(torch.linalg.norm(weight)),
        "bias_l2_norm": float(torch.linalg.norm(bias)),
        "singular_min": minimum,
        "singular_median": median,
        "singular_max": maximum,
        "condition_number": None if minimum == 0.0 else maximum / minimum,
        "state_sha256": module_state_sha256(module),
    }
