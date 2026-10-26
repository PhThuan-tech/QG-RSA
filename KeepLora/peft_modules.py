import math

import torch
import torch.nn as nn


class KeepLoRA(nn.Module):
    """Task-local KeepLoRA update for a frozen linear weight."""

    def __init__(self, in_dim, out_dim, r=8, lora_alpha=8, use_rslora=False, dtype=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = r
        self.scaling = lora_alpha / math.sqrt(r) if use_rslora else lora_alpha / r
        self.dtype = dtype
        self.register_buffer("lora_A", torch.zeros(in_dim, r, dtype=dtype), persistent=False)
        self.lora_B = nn.Parameter(torch.zeros(r, out_dim, dtype=dtype))
        self.register_buffer("principal_basis", torch.empty(in_dim, 0, dtype=dtype))
        self.register_buffer("feature_basis", torch.empty(in_dim, 0, dtype=dtype))
        self._feature_second_moment = None
        self._feature_count = 0
        self._feature_limit = 0
        self._collecting_features = False

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for name in ("principal_basis", "feature_basis"):
            key = prefix + name
            if key in state_dict:
                setattr(self, name, torch.empty_like(state_dict[key], device=getattr(self, name).device))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def reset_parameters(self):
        self.lora_A.zero_()
        self.lora_B.data.zero_()

    def initialize_principal_subspace(self, weight, energy):
        if self.principal_basis.numel() > 0:
            return
        # KeepLoRA builds this basis from the PTM weight itself before task 0.
        # RSIAT's Q/K/V/O matrices are square, so this has the same dimension
        # expected by the original gradient-projection implementation.
        u, singular_values, _ = torch.linalg.svd(weight.detach().float(), full_matrices=False)
        cumulative = torch.cumsum(singular_values.square(), dim=0)
        width = max(int(torch.sum(cumulative < energy * cumulative[-1]).item()), 1)
        self.principal_basis = u[:, :width].to(weight.device, dtype=weight.dtype)

    def _knowledge_basis(self):
        bases = [basis for basis in (self.principal_basis, self.feature_basis) if basis.numel()]
        if not bases:
            return None
        return torch.linalg.qr(torch.cat(bases, dim=1), mode="reduced").Q

    def initialize_from_gradient(self, gradient):
        if gradient is None:
            self.reset_parameters()
            return 0.0, 0.0
        projected = gradient.detach().transpose(0, 1).float()
        gradient_norm = torch.linalg.vector_norm(projected).item()
        basis = self._knowledge_basis()
        if basis is not None:
            basis = basis.to(projected.device, dtype=projected.dtype)
            projected = projected - basis @ (basis.transpose(0, 1) @ projected)
        projected_gradient_norm = torch.linalg.vector_norm(projected).item()
        if torch.count_nonzero(projected) == 0:
            self.reset_parameters()
            return gradient_norm, projected_gradient_norm
        # Match KeepLoRA's gradient-initialization path: a low-rank SVD is
        # sufficient because only rank columns are retained for A/B.
        sketch_rank = min(4 * self.rank, min(projected.shape))
        u, singular_values, v = torch.svd_lowrank(
            projected, q=sketch_rank, niter=4
        )
        width = min(self.rank, u.shape[1], v.shape[1])
        self.reset_parameters()
        self.lora_A[:, :width].copy_(u[:, :width].to(self.lora_A.dtype))
        self.lora_B.data[:width].copy_(
            (singular_values[:width].unsqueeze(1) * v.transpose(0, 1)[:width]).to(
                self.lora_B.dtype
            )
        )
        return gradient_norm, projected_gradient_norm

    def get_delta_weight(self):
        return self.scaling * (self.lora_A @ self.lora_B).transpose(0, 1)

    def subtract_from(self, weight):
        weight.data.sub_(self.get_delta_weight().to(weight.device, dtype=weight.dtype))

    def merge_into(self, weight):
        weight.data.add_(self.get_delta_weight().to(weight.device, dtype=weight.dtype))

    def start_feature_collection(self, max_samples):
        self._feature_second_moment = None
        self._feature_count = 0
        self._feature_limit = max_samples
        self._collecting_features = True

    def take_feature_matrix(self):
        if self._feature_second_moment is None or self._feature_count == 0:
            return None
        # If X is the token-feature matrix, C = X.T @ X has eigenvectors U
        # and eigenvalues S^2.  U @ diag(sqrt(S^2)) has the same left singular
        # vectors *and singular-value energy* as X.T, which is the matrix
        # supplied to KeepLoRA's original principal-subspace update.
        covariance = self._feature_second_moment / self._feature_count
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
        matrix = eigenvectors * eigenvalues.clamp_min_(0).sqrt().unsqueeze(0)
        self.release_buffer()
        return matrix

    def set_feature_basis(self, basis):
        self.feature_basis = basis.to(self.feature_basis.device, dtype=self.feature_basis.dtype)

    def project_features_to_residual(self, features):
        basis = self._knowledge_basis()
        if basis is None:
            return features
        basis = basis.to(features.device, dtype=features.dtype)
        return features - basis @ (basis.transpose(0, 1) @ features)

    def finish_feature_collection(self, energy):
        features = self.take_feature_matrix()
        if features is None:
            return
        features = features.float()
        features = self.project_features_to_residual(features)
        if torch.count_nonzero(features) == 0:
            return
        u, singular_values, _ = torch.linalg.svd(features, full_matrices=False)
        cumulative = torch.cumsum(singular_values.square(), dim=0)
        width = int(torch.searchsorted(cumulative, energy * cumulative[-1]).item()) + 1
        new_basis = u[:, :width].to(self.feature_basis.device, dtype=self.feature_basis.dtype)
        self.feature_basis = torch.cat((self.feature_basis, new_basis), dim=1)
        combined = self._knowledge_basis()
        self.feature_basis = combined[:, self.principal_basis.shape[1] :].to(self.feature_basis.dtype)

    def accumulate_features(self, x):
        if not self._collecting_features:
            return
        x = x.detach().reshape(-1, x.shape[-1])
        remaining = self._feature_limit - self._feature_count
        if self._feature_limit > 0:
            x = x[:remaining]
        if x.numel() == 0:
            return
        x = x.float().cpu()
        second_moment = x.transpose(0, 1) @ x
        if self._feature_second_moment is None:
            self._feature_second_moment = second_moment
        else:
            self._feature_second_moment.add_(second_moment)
        self._feature_count += x.shape[0]

    def release_buffer(self):
        self._feature_second_moment = None
        self._feature_count = 0
        self._feature_limit = 0
        self._collecting_features = False

    def forward(self, x):
        self.accumulate_features(x)
        return self.scaling * ((x @ self.lora_A) @ self.lora_B)
