"""BiAlign-only modules and loss terms.

This is an [IMPLEMENTATION ADAPTATION] of the bidirectional-alignment idea,
not a full BiCyc implementation. In particular, this module contains no cycle
objective and no statistics transport.
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
