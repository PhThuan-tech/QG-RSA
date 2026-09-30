"""BiAlign-only modules and loss terms.

This is an [IMPLEMENTATION ADAPTATION] of the bidirectional-alignment idea,
not a full BiCyc implementation. The optional cycle objective reuses the same
P_t and D_t; this module contains no statistics transport.
"""

import torch
from torch import nn
from torch.nn import functional as F

from utils.bicyc_transport import preserve_global_rng_state


class SignedResidualProjector(nn.Module):
    """Lightweight signed residual map initialized to the identity.

    The final projection is unconstrained and zero-initialized. Consequently
    ``forward(x)`` is exactly ``x`` at construction time while the Jacobian
    with respect to ``x`` still contains the identity path.
    """

    def __init__(self, input_dim=768, hidden_dim=64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.down = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(self.hidden_dim, self.input_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = self.up(self.activation(self.down(x)))
        return x + residual


def make_identity_reverse_projector(input_dim, hidden_dim, seed, device=None):
    """Build a deterministic identity-initialized D_t without advancing RNG."""
    with preserve_global_rng_state():
        torch.manual_seed(int(seed))
        module = SignedResidualProjector(input_dim, hidden_dim)
        if device is not None:
            module = module.to(device)
    return module


def bialign_loss_terms(z_new, z_old, forward_projector, reverse_projector):
    """Compute BiAlign losses with the pre-registered gradient routing.

    L_fwd updates P_t only. L_back updates D_t and the current representation.
    The previous representation is always detached.
    """
    if forward_projector is None:
        raise ValueError("BiAlign requires RSIAT forward projector P_t")
    if reverse_projector is None:
        raise ValueError("BiAlign requires reverse projector D_t")

    detached_old = z_old.detach()
    mapped_old = forward_projector(detached_old)
    loss_fwd = F.mse_loss(mapped_old, z_new.detach())
    loss_back = F.mse_loss(reverse_projector(z_new), detached_old)
    return {
        "mapped_old": mapped_old,
        "loss_fwd": loss_fwd,
        "loss_back": loss_back,
        "loss_bialign": loss_fwd + loss_back,
    }


def bialign_cycle_loss_terms(
        z_new, z_old, forward_projector, reverse_projector,
        stop_gradient_cycle_new=False):
    """Add cycle consistency to the unchanged BiAlign terms.

    The default path preserves validated BiAlign+Cycle behavior: ``cycle_new``
    updates P_t, D_t, and the current representation. With
    ``stop_gradient_cycle_new=True``, z_new is detached *before* D_t so the
    cycle updates P_t/D_t only. Targets and ``cycle_old`` are unchanged.
    """
    terms = bialign_loss_terms(
        z_new, z_old, forward_projector, reverse_projector)
    detached_old = z_old.detach()
    cycle_new_input = z_new.detach() if stop_gradient_cycle_new else z_new
    reversed_new = reverse_projector(cycle_new_input)
    cycle_new = F.mse_loss(
        forward_projector(reversed_new), z_new.detach())
    cycle_old = F.mse_loss(
        reverse_projector(terms["mapped_old"]), detached_old)
    terms.update({
        "cycle_new": cycle_new,
        "cycle_old": cycle_old,
        "loss_cycle": cycle_new + cycle_old,
    })
    return terms


def loss_gradient_records(loss, modules):
    """Measure per-module gradients for one loss without accumulating .grad."""
    trainable = {}
    flat_parameters = []
    for name, module in modules.items():
        parameters = [] if module is None else [
            parameter for parameter in module.parameters()
            if parameter.requires_grad
        ]
        trainable[name] = parameters
        flat_parameters.extend(parameters)

    gradients = torch.autograd.grad(
        loss, flat_parameters, retain_graph=True, allow_unused=True)
    records = {}
    offset = 0
    for name, parameters in trainable.items():
        module_gradients = gradients[offset:offset + len(parameters)]
        offset += len(parameters)
        present = [gradient for gradient in module_gradients
                   if gradient is not None]
        all_finite = all(bool(torch.isfinite(gradient).all())
                         for gradient in present)
        squared_norm = sum(
            (gradient.detach().double().pow(2).sum()
             for gradient in present),
            start=loss.detach().new_zeros((), dtype=torch.double),
        )
        records[name] = {
            "has_gradient": bool(present),
            "l2_norm": float(torch.sqrt(squared_norm).cpu()),
            "all_finite": bool(all_finite),
        }
    return records


def module_gradient_record(module):
    """Return a finite-aware L2 gradient summary without mutating gradients."""
    if module is None:
        return {"has_gradient": False, "l2_norm": 0.0, "all_finite": True}
    squared_norm = None
    has_gradient = False
    all_finite = True
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        has_gradient = True
        gradient = parameter.grad.detach()
        all_finite = all_finite and bool(torch.isfinite(gradient).all())
        contribution = gradient.double().pow(2).sum()
        squared_norm = (
            contribution if squared_norm is None else squared_norm + contribution
        )
    norm = 0.0 if squared_norm is None else float(torch.sqrt(squared_norm).cpu())
    return {
        "has_gradient": has_gradient,
        "l2_norm": norm,
        "all_finite": bool(all_finite),
    }
