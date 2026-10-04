"""Projected local quantum metrics used by QKSR.

The implementation intentionally uses only PyTorch operations.  The simulated
Ry/CNOT circuit is real-valued, batched, differentiable, and never loops over
samples or sample pairs.  Small Python loops are limited to circuit structure
(qubits and layers).
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Representation = Union[Tensor, Dict[str, Tensor]]


def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }
    try:
        return mapping[str(dtype).lower()]
    except KeyError as exc:
        raise ValueError("q_dtype must be 'float32' or 'float64'.") from exc


class ClassicalProjector(nn.Module):
    """Map ViT features to bounded rotation angles without affine LayerNorm."""

    def __init__(self, input_dim: int, num_qubits: int, dtype: torch.dtype):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False, dtype=dtype)
        self.linear = nn.Linear(input_dim, num_qubits, dtype=dtype)

    def forward(self, features: Tensor) -> Tensor:
        features = features.to(dtype=self.linear.weight.dtype)
        return math.pi * torch.tanh(self.linear(self.norm(features)))


class QuantumFeatureEncoder(nn.Module):
    """Exact batched statevector simulator for a shallow Ry/CNOT circuit."""

    def __init__(
        self,
        num_qubits: int,
        num_layers: int,
        dtype: torch.dtype,
        entangle: bool = True,
        reupload: bool = False,
        frozen_random: bool = False,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.entangle = entangle
        self.reupload = reupload
        theta = torch.randn(num_layers, num_qubits, dtype=dtype) * 0.01
        if frozen_random:
            self.register_buffer("theta", theta)
        else:
            self.theta = nn.Parameter(theta)

        basis_size = 1 << num_qubits
        permutations = []
        indices = torch.arange(basis_size, dtype=torch.long)
        for control in range(num_qubits):
            target = (control + 1) % num_qubits
            control_mask = 1 << (num_qubits - 1 - control)
            target_mask = 1 << (num_qubits - 1 - target)
            control_on = (indices & control_mask) != 0
            permutations.append(torch.where(control_on, indices ^ target_mask, indices))
        self.register_buffer(
            "_cnot_permutations", torch.stack(permutations), persistent=False
        )

    def _zero_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        state = torch.zeros(batch_size, 1 << self.num_qubits, device=device, dtype=dtype)
        state[:, 0] = 1.0
        return state

    def _apply_ry(self, state: Tensor, angles: Tensor, qubit: int) -> Tensor:
        batch_size = state.shape[0]
        state_nd = state.reshape(batch_size, *([2] * self.num_qubits))
        state_nd = state_nd.movedim(qubit + 1, -1)
        amplitudes = state_nd.reshape(batch_size, -1, 2)

        half = angles / 2.0
        cosine = torch.cos(half)
        sine = torch.sin(half)
        if cosine.ndim == 0:
            cosine = cosine.reshape(1, 1)
            sine = sine.reshape(1, 1)
        else:
            cosine = cosine.reshape(batch_size, 1)
            sine = sine.reshape(batch_size, 1)

        amp0, amp1 = amplitudes[..., 0], amplitudes[..., 1]
        rotated = torch.stack(
            (cosine * amp0 - sine * amp1, sine * amp0 + cosine * amp1), dim=-1
        )
        rotated = rotated.reshape(state_nd.shape).movedim(-1, qubit + 1)
        return rotated.reshape(batch_size, -1)

    def _apply_encoding(self, state: Tensor, angles: Tensor) -> Tensor:
        for qubit in range(self.num_qubits):
            state = self._apply_ry(state, angles[:, qubit], qubit)
        return state

    def _apply_variational(self, state: Tensor, layer: int) -> Tensor:
        for qubit in range(self.num_qubits):
            state = self._apply_ry(state, self.theta[layer, qubit], qubit)
        return state

    def _apply_cnot(self, state: Tensor, control: int) -> Tensor:
        permutation = self._cnot_permutations[control]
        return state.index_select(1, permutation)

    def _apply_entangling(self, state: Tensor) -> Tensor:
        if not self.entangle:
            return state
        for control in range(self.num_qubits):
            state = self._apply_cnot(state, control)
        return state

    def forward(
        self, angles: Tensor, return_intermediate: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        state = self._zero_state(angles.shape[0], angles.device, angles.dtype)
        history: List[Tensor] = []

        if not self.reupload:
            state = self._apply_encoding(state, angles)
            history.append(state)

        for layer in range(self.num_layers):
            if self.reupload:
                state = self._apply_encoding(state, angles)
            state = self._apply_variational(state, layer)
            state = self._apply_entangling(state)
            history.append(state)

        if return_intermediate:
            return state, history
        return state


class LocalRDMExtractor(nn.Module):
    """Extract one- and nearest-neighbour two-qubit reduced density matrices."""

    def __init__(self, num_qubits: int, kernel_order: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.kernel_order = kernel_order

    def _reduced_density(self, state: Tensor, selected: Tuple[int, ...]) -> Tensor:
        batch_size = state.shape[0]
        state_nd = state.reshape(batch_size, *([2] * self.num_qubits))
        remaining = [qubit for qubit in range(self.num_qubits) if qubit not in selected]
        permutation = [0] + [qubit + 1 for qubit in remaining + list(selected)]
        subsystem_dim = 1 << len(selected)
        amplitudes = state_nd.permute(permutation).contiguous().reshape(
            batch_size, -1, subsystem_dim
        )
        return torch.einsum("bra,brc->bac", amplitudes, amplitudes.conj())

    def forward(self, state: Tensor) -> Dict[str, Tensor]:
        one_body = [
            self._reduced_density(state, (qubit,))
            for qubit in range(self.num_qubits)
        ]
        result = {"one": torch.stack(one_body, dim=1)}
        if self.kernel_order == 2:
            two_body = [
                self._reduced_density(state, (qubit, (qubit + 1) % self.num_qubits))
                for qubit in range(self.num_qubits)
            ]
            result["two"] = torch.stack(two_body, dim=1)
        return result


class QuantumKernelModule(nn.Module):
    """Common metric interface for PQK and capacity controls."""

    KERNEL_TYPES = {
        "pqk",
        "pqk_no_cnot",
        "pqk_random_frozen",
        "rbf_proj",
        "mlp_small",
        "mlp_cap",
    }
    GAMMA_MODES = {"bounded_learned", "median_fixed", "free_learned"}

    def __init__(
        self,
        input_dim: int = 768,
        num_qubits: int = 8,
        num_layers: int = 2,
        kernel_type: str = "pqk",
        kernel_order: int = 1,
        order2_weight: float = 1.0,
        reupload: bool = False,
        gamma_mode: str = "bounded_learned",
        dtype: Union[str, torch.dtype] = torch.float32,
        init_seed: int = 0,
    ):
        super().__init__()
        if not 2 <= int(num_qubits) <= 12:
            raise ValueError("num_qubits must be in [2, 12].")
        if not 1 <= int(num_layers) <= 3:
            raise ValueError("num_layers must be in [1, 3].")
        if kernel_type not in self.KERNEL_TYPES:
            raise ValueError("Unknown q_kernel_type: {}".format(kernel_type))
        if kernel_order not in (1, 2):
            raise ValueError("q_kernel_order must be 1 or 2.")
        if order2_weight < 0:
            raise ValueError("q_order2_weight must be non-negative.")
        if gamma_mode not in self.GAMMA_MODES:
            raise ValueError("Unknown q_gamma_mode: {}".format(gamma_mode))

        self.input_dim = int(input_dim)
        self.num_qubits = int(num_qubits)
        self.num_layers = int(num_layers)
        self.kernel_type = kernel_type
        self.kernel_order = int(kernel_order)
        self.order2_weight = float(order2_weight)
        self.reupload = bool(reupload)
        self.gamma_mode = gamma_mode
        self.dtype = _resolve_dtype(dtype)
        self.init_seed = int(init_seed)
        self._inc_mode = "trainable"
        self._last_kernel: Optional[Tensor] = None
        self._epoch_diagnostics = None

        # Only the CPU generator is touched, and its state is restored.  Module
        # construction therefore cannot shift either training or CUDA RNG.
        cpu_rng_state = torch.get_rng_state()
        try:
            torch.random.default_generator.manual_seed(self.init_seed)
            self.projector = ClassicalProjector(input_dim, num_qubits, self.dtype)
            self.encoder: Optional[QuantumFeatureEncoder] = None
            self.extractor: Optional[LocalRDMExtractor] = None
            self.metric: Optional[nn.Module] = None

            if kernel_type.startswith("pqk"):
                self.encoder = QuantumFeatureEncoder(
                    num_qubits=num_qubits,
                    num_layers=num_layers,
                    dtype=self.dtype,
                    entangle=kernel_type != "pqk_no_cnot",
                    reupload=reupload,
                    frozen_random=kernel_type == "pqk_random_frozen",
                )
                self.extractor = LocalRDMExtractor(num_qubits, kernel_order)
            elif kernel_type.startswith("mlp_"):
                hidden_dim = num_qubits if kernel_type == "mlp_small" else 64
                output_dim = (2 if kernel_order == 1 else 11) * num_qubits
                self.metric = nn.Sequential(
                    nn.Linear(num_qubits, hidden_dim, dtype=self.dtype),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, output_dim, dtype=self.dtype),
                )

            self.register_buffer("gamma0", torch.tensor(1.0, dtype=self.dtype))
            self.register_buffer("gamma_initialized", torch.tensor(False))
            if gamma_mode != "median_fixed":
                self.gamma_param = nn.Parameter(torch.tensor(0.0, dtype=self.dtype))
            else:
                self.register_parameter("gamma_param", None)
        finally:
            torch.set_rng_state(cpu_rng_state)

    @property
    def gamma(self) -> Tensor:
        if self.gamma_mode == "median_fixed":
            return self.gamma0
        if self.gamma_mode == "bounded_learned":
            return self.gamma0 * torch.pow(
                torch.tensor(10.0, dtype=self.gamma0.dtype, device=self.gamma0.device),
                torch.tanh(self.gamma_param),
            )
        return F.softplus(self.gamma_param).clamp_min(torch.finfo(self.dtype).eps)

    def encode(self, features: Tensor) -> Representation:
        projected = self.projector(features)
        if self.kernel_type == "rbf_proj":
            return projected
        if self.metric is not None:
            return self.metric(projected)
        if self.encoder is None or self.extractor is None:
            raise RuntimeError("PQK encoder was not initialized.")
        return self.extractor(self.encoder(projected))

    @staticmethod
    def _tensor_distance(rep_a: Tensor, rep_b: Tensor) -> Tensor:
        difference = rep_a[:, None, ...] - rep_b[None, :, ...]
        reduce_dims = tuple(range(2, difference.ndim))
        return difference.abs().square().sum(dim=reduce_dims)

    def distance(self, rep_a: Representation, rep_b: Representation) -> Tensor:
        if isinstance(rep_a, Tensor):
            if not isinstance(rep_b, Tensor):
                raise TypeError("Metric representations must have matching types.")
            return self._tensor_distance(rep_a, rep_b)
        if not isinstance(rep_b, dict):
            raise TypeError("Metric representations must have matching types.")
        distance = self._tensor_distance(rep_a["one"], rep_b["one"])
        if self.kernel_order == 2:
            distance = distance + self.order2_weight * self._tensor_distance(
                rep_a["two"], rep_b["two"]
            )
        return distance

    def kernel(self, rep_a: Representation, rep_b: Representation) -> Tensor:
        distance = self.distance(rep_a, rep_b).clamp_min(0.0)
        exponent = -self.gamma.to(distance.dtype) * distance
        # A convex mixture with the constant kernel keeps values strictly
        # positive without the PSD-breaking hard floor max(RBF, epsilon).
        rbf = torch.exp(exponent.clamp(max=0.0))
        floor = torch.finfo(distance.dtype).tiny
        return (1.0 - floor) * rbf + floor

    def forward(self, a: Tensor, b: Optional[Tensor] = None) -> Tensor:
        rep_a = self.encode(a)
        rep_b = rep_a if b is None else self.encode(b)
        result = self.kernel(rep_a, rep_b)
        self._last_kernel = result.detach()
        self._record_kernel_stats(result.detach(), self_kernel=b is None)
        return result

    def reset_epoch_diagnostics(self) -> None:
        """Enable bounded-memory diagnostics without changing the training RNG."""
        self._epoch_diagnostics = {
            "batches": 0, "empty_batches": 0, "pairs": 0,
            "sum": None, "squared_sum": None, "min": None, "max": None,
            "histogram": None, "gradients": {},
        }

    @torch.no_grad()
    def _record_kernel_stats(self, kernel: Tensor, self_kernel: bool) -> None:
        stats = self._epoch_diagnostics
        if stats is None:
            return
        stats["batches"] += 1
        # The trivial K(x_i,x_i)=1 can hide collapsed off-diagonal values.
        # Even square CROSS kernels still include every pair.
        if self_kernel:
            mask = ~torch.eye(kernel.shape[0], dtype=torch.bool, device=kernel.device)
            values = kernel[mask]
        else:
            values = kernel.reshape(-1)
        if not values.numel():
            stats["empty_batches"] += 1
            return
        stats["pairs"] += values.numel()
        values = values.to(torch.float64)
        batch_sum = values.sum()
        squared_sum = values.square().sum()
        histogram = torch.histc(values.float(), bins=10, min=0.0, max=1.0).to(torch.int64)
        if stats["sum"] is None:
            stats.update({
                "sum": batch_sum, "squared_sum": squared_sum,
                "min": values.min(), "max": values.max(), "histogram": histogram,
            })
        else:
            stats["sum"] += batch_sum
            stats["squared_sum"] += squared_sum
            stats["min"] = torch.minimum(stats["min"], values.min())
            stats["max"] = torch.maximum(stats["max"], values.max())
            stats["histogram"] += histogram

    @torch.no_grad()
    def record_gradient_health(self) -> None:
        """Record after backward; defer device synchronization until epoch end."""
        if self._epoch_diagnostics is None:
            return
        gradients = self._epoch_diagnostics["gradients"]
        for name, parameter in self.named_parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().abs().mean()
            if name not in gradients:
                gradients[name] = [value, 1]
            else:
                gradients[name][0] += value
                gradients[name][1] += 1

    def epoch_gradient_health(self) -> Dict[str, float]:
        if self._epoch_diagnostics is None:
            return {}
        return {
            name: float((total / count).item())
            for name, (total, count) in self._epoch_diagnostics["gradients"].items()
        }

    def epoch_kernel_stats(self) -> Dict[str, object]:
        stats = self._epoch_diagnostics
        if stats is None:
            return {}
        summary = {key: stats[key] for key in ("batches", "empty_batches", "pairs")}
        if not stats["pairs"]:
            return summary
        mean = stats["sum"] / stats["pairs"]
        variance = (stats["squared_sum"] / stats["pairs"] - mean.square()).clamp_min(0.0)
        summary.update({
            "min": float(stats["min"].item()), "mean": float(mean.item()),
            "std": float(variance.sqrt().item()), "max": float(stats["max"].item()),
            "histogram_10_bins_0_1": stats["histogram"].tolist(),
            "self_diagonal_excluded": True,
        })
        return summary

    @torch.no_grad()
    def calibrate_gamma(
        self,
        features_a: Tensor,
        features_b: Optional[Tensor] = None,
        exclude_diagonal: Optional[bool] = None,
        force: bool = False,
    ) -> float:
        """Set gamma0 from median self- or cross-distances exactly once."""
        if bool(self.gamma_initialized.item()) and not force:
            return float(self.gamma0.item())
        if exclude_diagonal is None:
            exclude_diagonal = features_b is None

        rep_a = self.encode(features_a)
        rep_b = rep_a if features_b is None else self.encode(features_b)
        distances = self.distance(rep_a, rep_b).detach()
        if exclude_diagonal:
            if distances.shape[0] != distances.shape[1]:
                raise ValueError("Diagonal exclusion requires a square self-distance matrix.")
            mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)
            values = distances[mask]
        else:
            values = distances.reshape(-1)

        values = values[torch.isfinite(values)]
        median = values.median() if values.numel() else distances.new_tensor(0.0)
        if not torch.isfinite(median) or median <= 1e-8:
            logging.warning(
                "QKSR gamma calibration distances are degenerate; using gamma0=1.0."
            )
            gamma0 = distances.new_tensor(1.0)
        else:
            gamma0 = median.reciprocal()
        self.gamma0.copy_(gamma0.to(self.gamma0))

        if self.gamma_mode == "free_learned":
            # Stable inverse softplus: x + log(1 - exp(-x)).
            x = self.gamma0.clamp_min(torch.finfo(self.gamma0.dtype).eps)
            inverse = x + torch.log(-torch.expm1(-x))
            self.gamma_param.copy_(inverse)
        elif self.gamma_param is not None:
            self.gamma_param.zero_()
        self.gamma_initialized.fill_(True)
        return float(self.gamma0.item())

    def set_inc_mode(self, mode: str) -> None:
        if mode not in {"frozen", "trainable"}:
            raise ValueError("q_inc_train_mode must be 'frozen' or 'trainable'.")
        self._inc_mode = mode
        trainable = mode == "trainable"
        for parameter in self.parameters():
            parameter.requires_grad_(trainable)
            if not trainable:
                parameter.grad = None

    def param_groups(
        self,
        adapter_lr: float,
        weight_decay: float,
        metric_lr_mult: Optional[float],
    ) -> List[dict]:
        multiplier = 0.1 if metric_lr_mult is None else float(metric_lr_mult)
        if multiplier <= 0:
            raise ValueError("q_metric_lr_mult must be positive.")
        projector_params = []
        metric_params = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("projector."):
                projector_params.append(parameter)
            else:
                metric_params.append(parameter)
        groups = []
        if projector_params:
            groups.append(
                {
                    "params": projector_params,
                    "lr": float(adapter_lr),
                    "weight_decay": float(weight_decay),
                    "q_group": "projector",
                }
            )
        if metric_params:
            groups.append(
                {
                    "params": metric_params,
                    "lr": float(adapter_lr) * multiplier,
                    "weight_decay": 0.0,
                    "q_group": "metric",
                }
            )
        return groups

    def gradient_health(self) -> Dict[str, float]:
        health = {}
        for name, parameter in self.named_parameters():
            if parameter.grad is not None:
                health[name] = float(parameter.grad.detach().abs().mean().item())
        return health

    def parameter_counts(self) -> Dict[str, int]:
        projector = sum(parameter.numel() for parameter in self.projector.parameters())
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "projector": projector,
            "metric": total - projector,
            "total": total,
            "trainable": trainable,
        }

    def last_kernel_stats(self) -> Dict[str, object]:
        if self._last_kernel is None or self._last_kernel.numel() == 0:
            return {}
        values = self._last_kernel.float().reshape(-1)
        histogram = torch.histc(values, bins=10, min=0.0, max=1.0).to(torch.int64)
        quantiles = torch.quantile(
            values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=values.device)
        )
        return {
            "min": float(values.min().item()),
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "max": float(values.max().item()),
            "quantiles_05_25_50_75_95": [float(value) for value in quantiles.tolist()],
            "histogram_10_bins_0_1": histogram.tolist(),
        }
