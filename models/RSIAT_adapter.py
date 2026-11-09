import copy
import hashlib
import logging
import random
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleVitNet
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseLearner
from utils.toolkit import count_parameters, log_count_parameter, target2onehot, tensor2numpy
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import AutoencoderSigmoid
from utils.semantic_drift import SemanticDriftObserver
from KeepLora.keeplora_controller import KeepLoRAController
from KeepLora.m_only_adapter import MOnlyKeepLoRA, update_feature_basis
import math
num_workers = 8
KEEPLORA_FEATURE_RTOL = 1e-4
KEEPLORA_FEATURE_ATOL = 1e-3


def _validate_ssca_scale_config(args):
    if "ssca_scale_mode" not in args:
        return "none", None

    mode = args["ssca_scale_mode"]
    if mode in ("legacy", "none"):
        return mode, None
    if mode not in ("global_decay", "age_decay"):
        raise ValueError(
            "ssca_scale_mode must be 'none', 'global_decay', or 'age_decay', "
            "got {!r}."
            .format(mode)
        )

    gamma_key = (
        "ssca_global_gamma" if mode == "global_decay" else "ssca_age_gamma"
    )
    gamma = args.get(gamma_key)
    if isinstance(gamma, bool):
        raise ValueError(
            "{} must be a finite number in [0, 1].".format(gamma_key)
        )
    try:
        gamma = float(gamma)
    except (OverflowError, TypeError, ValueError):
        raise ValueError(
            "{} must be a finite number in [0, 1].".format(gamma_key)
        ) from None
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError(
            "{} must be a finite number in [0, 1].".format(gamma_key)
        )
    return mode, gamma


def _validate_ssca_drift_probe_mode(args):
    mode = args.get("ssca_drift_probe_mode")
    if mode is not None and mode not in (
        "stochastic_unpaired",
        "paired_deterministic",
    ):
        raise ValueError(
            "ssca_drift_probe_mode must be 'stochastic_unpaired' or "
            "'paired_deterministic', got {!r}.".format(mode)
        )
    return mode


def _ssca_scale_for_task(args, task_index):
    if (
        isinstance(task_index, (bool, np.bool_))
        or not isinstance(task_index, (int, np.integer))
        or task_index < 0
    ):
        raise ValueError("task_index must be a non-negative integer.")
    mode, gamma = _validate_ssca_scale_config(args)
    if mode in ("none", "legacy"):
        return 1.0
    if mode != "global_decay":
        raise ValueError(
            "_ssca_scale_for_task only supports none and global_decay modes."
        )
    if task_index <= 1:
        return 1.0
    return gamma ** (task_index - 1)


def _ssca_scales_for_classes(args, task_index, task_sizes, known_classes, dtype):
    mode, gamma = _validate_ssca_scale_config(args)
    if (
        isinstance(task_index, (bool, np.bool_))
        or not isinstance(task_index, (int, np.integer))
        or task_index < 0
    ):
        raise ValueError("task_index must be a non-negative integer.")
    if (
        isinstance(known_classes, (bool, np.bool_))
        or not isinstance(known_classes, (int, np.integer))
        or known_classes < 0
    ):
        raise ValueError("known_classes must be a non-negative integer.")

    if mode in ("none", "legacy"):
        scales = np.ones(known_classes, dtype=dtype)
        groups = [("all_old", 1.0)] if known_classes else []
    elif mode == "global_decay":
        scale = _ssca_scale_for_task(args, task_index)
        scales = np.full(known_classes, scale, dtype=dtype)
        groups = [("all_old", scale)] if known_classes else []
    else:
        try:
            raw_sizes = list(task_sizes)
        except TypeError:
            raise ValueError(
                "age_decay task_sizes must contain positive integers."
            ) from None
        if any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            for size in raw_sizes
        ):
            raise ValueError(
                "age_decay task_sizes must contain positive integers."
            )
        sizes = [int(size) for size in raw_sizes]
        if len(sizes) != task_index + 1 or any(size <= 0 for size in sizes):
            raise ValueError(
                "age_decay requires positive task_sizes through the current "
                "task; task_index={}, task_sizes={!r}."
                .format(task_index, task_sizes)
            )
        if sum(sizes[:-1]) != known_classes:
            raise ValueError(
                "age_decay task sizes do not match old-class prototypes: "
                "sum(previous_task_sizes)={}, known_classes={}."
                .format(sum(sizes[:-1]), known_classes)
            )

        class_origins = np.repeat(
            np.arange(len(sizes) - 1, dtype=np.int64),
            np.asarray(sizes[:-1], dtype=np.int64),
        )
        exponents = np.maximum(task_index - class_origins - 1, 0)
        scales = np.power(gamma, exponents).astype(dtype, copy=False)
        groups = []
        start = 0
        for origin_task, size in enumerate(sizes[:-1]):
            end = start + size
            if end > start:
                groups.append(
                    (origin_task, float(scales[start]))
                )
            start = end

    return mode, gamma, scales.reshape(known_classes, 1), groups


def _apply_ssca_gap(class_means, known_classes, gap, enabled, scale):
    if enabled:
        old_class_mean = class_means[:known_classes]
        old_class_mean += scale * gap
        class_means[:known_classes] = old_class_mean


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if not any(token in args["convnet_type"] for token in ("adapter", "keeplora")):
            raise NotImplementedError("RSIAT requires an adapter or KeepLoRA backbone")
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.num_workers = int(args.get("num_workers", num_workers))
        self.pin_memory = bool(args.get("pin_memory", self._device.type == "cuda"))
        self.persistent_workers = self.num_workers > 0 and bool(
            args.get("persistent_workers", True)
        )
        self.init_lr = args["init_lr"]

        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        _validate_ssca_scale_config(self.args)
        _validate_ssca_drift_probe_mode(self.args)
        self.semantic_drift = SemanticDriftObserver(self)

        self._old_most_sentive = []
        self._update_grads = {}

        self.logit_norm = None
        self.tuned_epochs = None
        self.task_sizes = []
        self.rs_loss_func = RS_Loss(self.args["alpha"], self.args["rs_margin"])
        self.old_ae = None
        self.rae_zero_init = bool(args.get("rae_zero_init", False))
        self.rae_lifecycle = str(args.get("rae_lifecycle", "shared")).lower()
        if self.rae_lifecycle not in ("shared", "per_task"):
            raise ValueError(
                "rae_lifecycle must be 'shared' or 'per_task', got {!r}."
                .format(self.rae_lifecycle)
            )
        self.rae_generation = 0
        self.rae_task_id = None
        self._rae_last_state_snapshot = None
        self._uses_keeplora = "keeplora" in args["convnet_type"].lower()
        self.shared_adapter_update_mode = str(
            args.get("shared_adapter_update_mode", "baseline")
        ).lower()
        if self.shared_adapter_update_mode not in ("baseline", "keeplora_m_only", "dense_control"):
            raise ValueError(
                "shared_adapter_update_mode must be 'baseline', "
                "'keeplora_m_only' or 'dense_control'."
            )
        self._uses_m_only = self.shared_adapter_update_mode == "keeplora_m_only"
        self._uses_dense_control = self.shared_adapter_update_mode == "dense_control"
        if (self._uses_m_only or self._uses_dense_control) and (
            args.get("model_name") != "adapter"
            or "_adapter" not in str(args["convnet_type"]).lower()
            or self._uses_keeplora
        ):
            raise ValueError(
                "M-only and dense control require the RSIAT Shared Adapter backbone."
            )
        if args.get("rebuild_task0_feature_basis", False) and not self._uses_m_only:
            raise ValueError("rebuild_task0_feature_basis requires keeplora_m_only.")
        self._m_only_bases = {}
        self.m_only_init_loader = None
        self.keeplora_init_mode = str(args.get("keeplora_init_mode", "cosine")).lower()
        if self._uses_m_only:
            self._validate_m_only_config()
        convnet_trainable = sum(
            parameter.numel()
            for parameter in self._network.convnet.parameters()
            if parameter.requires_grad
        )
        if self._uses_keeplora:
            adaptation_name = "KeepLoRA-{}".format(
                str(args.get("keeplora_target", "attention")).lower()
            )
            logging.info(
                "%s trainable adaptation parameters=%d; "
                "convnet trainable parameters=%d",
                adaptation_name,
                convnet_trainable,
                convnet_trainable,
            )
        else:
            adapter_trainable = sum(
                parameter.numel()
                for name, parameter in self._network.convnet.named_parameters()
                if parameter.requires_grad and "adaptmlp" in name
            )
            logging.info(
                "RSIAT-Adapter trainable adaptation parameters=%d; "
                "convnet trainable parameters=%d",
                adapter_trainable,
                convnet_trainable,
            )
        if self.keeplora_init_mode not in ("cosine", "full_rsiat"):
            raise ValueError(
                "keeplora_init_mode must be 'cosine' or 'full_rsiat', got {!r}."
                .format(self.keeplora_init_mode)
            )
        self._log_trainable_parameter_budget("initialization")

    def _log_trainable_parameter_budget(self, stage):
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        named_parameters = list(network.named_parameters())
        total_network_trainable = sum(
            parameter.numel()
            for _, parameter in named_parameters
            if parameter.requires_grad
        )
        if self._uses_keeplora:
            adaptation_marker = "keeplora"
            method = "KeepLoRA-{}".format(
                str(self.args.get("keeplora_target", "attention")).lower()
            )
        else:
            adaptation_marker = "adaptmlp"
            method = "RSIAT-Adapter"
        adaptation_trainable = sum(
            parameter.numel()
            for name, parameter in named_parameters
            if parameter.requires_grad and adaptation_marker in name
        )
        classifier_trainable = sum(
            parameter.numel()
            for parameter in network.fc.parameters()
            if parameter.requires_grad
        ) if network.fc is not None else 0
        rae_trainable = sum(
            parameter.numel()
            for parameter in self.old_ae.parameters()
            if parameter.requires_grad
        ) if self.old_ae is not None else 0
        logging.info(
            "Parameter budget stage=%s method=%s adaptation_trainable=%d "
            "classifier_trainable=%d rae_trainable=%d model_trainable=%d",
            stage,
            method,
            adaptation_trainable,
            classifier_trainable,
            rae_trainable,
            total_network_trainable + rae_trainable,
        )
        if self._uses_keeplora:
            targets = list(network.convnet.named_keeplora_targets())
            layer_ids = {
                name.split(".")[1]
                for name, _, _ in targets
                if name.startswith("blocks.")
            }
            if self.args.get("keeplora_target", "attention") == "mlp":
                fc1_count = sum(
                    adapter.lora_B.numel()
                    for name, _, adapter in targets
                    if name.endswith(".fc1") and adapter.lora_B.requires_grad
                )
                fc2_count = sum(
                    adapter.lora_B.numel()
                    for name, _, adapter in targets
                    if name.endswith(".fc2") and adapter.lora_B.requires_grad
                )
                logging.info(
                    "KeepLoRA-MLP layers=%d fc1_rank=%d fc2_rank=%d "
                    "fc1_trainable=%d fc2_trainable=%d",
                    len(layer_ids),
                    int(self.args.get("keeplora_fc1_rank", self.args.get("keeplora_rank", 32))),
                    int(self.args.get("keeplora_fc2_rank", self.args.get("keeplora_rank", 32))),
                    fc1_count,
                    fc2_count,
                )

    def _after_load_checkpoint(self, checkpoint):
        """Restore learner-specific state after BaseLearner restores the network."""
        saved_adapter_hash = checkpoint.get("shared_adapter_weight_sha256")
        if (
            int(checkpoint.get("cur_task", -1)) == 0
            and saved_adapter_hash is not None
            and saved_adapter_hash != self._shared_adapter_weight_checksum()
        ):
            raise ValueError(
                "Task-0 Shared Adapter checksum does not match the checkpoint."
            )
        saved_adapter_state_hash = checkpoint.get(
            "shared_adapter_state_sha256"
        )
        if (
            int(checkpoint.get("cur_task", -1)) == 0
            and saved_adapter_state_hash is not None
            and saved_adapter_state_hash
            != self._shared_adapter_state_checksum()
        ):
            raise ValueError(
                "Task-0 full Shared Adapter state checksum does not match "
                "the checkpoint."
            )
        m_only_state = checkpoint.get("shared_adapter_m_only_state")
        self._task_boundary_rng_state = checkpoint.get("dense_control_task_boundary_rng_state")
        if getattr(self, "_uses_m_only", False):
            self._m_only_bases = {}
        if m_only_state is not None:
            task_zero = (
                int(checkpoint.get("cur_task", -1)) == 0
                and checkpoint.get("mid_task_training_state") is None
            )
            dense_control_start = (
                getattr(self, "_uses_dense_control", False)
                and task_zero
                and not checkpoint.get("shared_adapter_m_only_active_factors")
            )
            if not (self._uses_m_only or dense_control_start):
                raise ValueError(
                    "Checkpoint contains Shared Adapter M-only state, but the "
                    "current mode is not keeplora_m_only."
                )
            if int(m_only_state.get("version", 0)) != 1:
                raise ValueError("Unsupported Shared Adapter M-only checkpoint state.")
            rebuild_basis = task_zero and self.args.get("rebuild_task0_feature_basis", False)
            if self._uses_m_only and not rebuild_basis:
                self._m_only_bases = {
                    name: basis.detach().cpu().float()
                    for name, basis in m_only_state.get("feature_bases", {}).items()
                }
                saved_basis_hash = m_only_state.get("feature_basis_sha256")
                if (
                    saved_basis_hash is not None
                    and saved_basis_hash != self._m_only_feature_basis_checksum()
                ):
                    raise ValueError(
                        "Shared Adapter feature-basis checksum does not match checkpoint."
                    )
            elif rebuild_basis:
                logging.info("Task-0 adapter weights reused. M0 will be rebuilt before Task 1.")
            if checkpoint.get("mid_task_training_state") is None:
                self._task_boundary_rng_state = m_only_state.get("task_boundary_rng_state")
        if self._cur_task >= 1:
            self.old_ae = self._build_rae()
            if "old_ae_state_dict" not in checkpoint:
                raise ValueError(
                    "Checkpoint is missing old_ae_state_dict required to resume task {}."
                    .format(self._cur_task + 1)
                )
            self.old_ae.load_state_dict(checkpoint["old_ae_state_dict"])
            self.old_ae.to(self._device)
            self.rae_generation = int(
                checkpoint.get("rae_generation", self.rae_generation + 1)
            )
            self.rae_task_id = checkpoint.get("rae_task_id", self._cur_task)
            self._rae_last_state_snapshot = self._rae_state_snapshot()

        mid_task_state = checkpoint.get("mid_task_training_state")
        if mid_task_state is not None:
            if not self._uses_m_only:
                raise ValueError(
                    "Mid-task resume is supported only for keeplora_m_only."
                )
            if int(mid_task_state.get("task_id", -1)) != self._cur_task:
                raise ValueError(
                    "Mid-task checkpoint task_id does not match cur_task."
                )
            old_state = checkpoint.get("old_network_state_dict")
            if old_state is None:
                raise ValueError(
                    "Mid-task checkpoint is missing old_network_state_dict."
                )
            old_network = self._old_network
            for block in old_network.convnet.blocks:
                adapter = getattr(block, "adaptmlp", None)
                if adapter is not None:
                    adapter.remove_m_only_factors()
            old_state = {
                key: value
                for key, value in old_state.items()
                if not key.startswith("fc.")
            }
            missing, unexpected = old_network.load_state_dict(
                old_state, strict=False
            )
            unexpected = [key for key in unexpected if not key.startswith("fc.")]
            missing = [key for key in missing if not key.startswith("fc.")]
            if missing or unexpected:
                raise ValueError(
                    "Could not restore the frozen previous-task network: "
                    "missing={}, unexpected={}".format(missing, unexpected)
                )
            self._old_network = old_network.freeze().to(self._device)

        self._network_module_ptr = self._network
        self.old_network_module_ptr = self._old_network

    def _prepare_checkpoint_model(self, checkpoint):
        mid_task_state = checkpoint.get("mid_task_training_state")
        if mid_task_state is None:
            return
        if not self._uses_m_only:
            raise ValueError(
                "Mid-task resume is supported only for keeplora_m_only."
            )
        network = (
            self._network.module
            if isinstance(self._network, nn.DataParallel)
            else self._network
        )
        module_metadata = checkpoint.get(
            "shared_adapter_m_only_active_factors", []
        )
        metadata_by_target = {
            (item["prefix"], item["kind"]): item
            for item in module_metadata
        }
        for prefix, adapter, kind, linear in self._m_only_adapter_targets():
            linear.weight.requires_grad_(False)
            bias = linear.bias
            if bias is not None:
                bias.requires_grad_(
                    self.args.get(
                        "shared_adapter_m_only_bias_policy", "freeze_bias"
                    ).lower() == "train_bias_unconstrained"
                )
            item = metadata_by_target.get((prefix, kind))
            if item is None:
                continue
            factor = MOnlyKeepLoRA(
                item["in_features"],
                item["out_features"],
                item["rank"],
                item["scaling"],
            ).to(device=linear.weight.device, dtype=linear.weight.dtype)
            if (
                factor.in_features != linear.in_features
                or factor.out_features != linear.out_features
            ):
                raise ValueError(
                    "Checkpoint M-only factor shape does not match {}."
                    .format(self._m_only_weight_key(prefix, kind))
                )
            if kind == "down":
                adapter.attach_m_only_factors(factor, adapter.keeplora_m_only_up)
            else:
                adapter.attach_m_only_factors(adapter.keeplora_m_only_down, factor)
        expected_targets = {
            (prefix, kind)
            for prefix, _, kind, _ in self._m_only_adapter_targets()
        }
        unexpected_targets = set(metadata_by_target) - expected_targets
        if unexpected_targets:
            raise ValueError(
                "Checkpoint contains unknown M-only factor targets: {}"
                .format(sorted(unexpected_targets))
            )

    def _checkpoint_extra_state(self, *, mid_task=False):
        if self.args.get("model_name") != "adapter":
            return {}
        state = {
            "shared_adapter_weight_sha256": self._shared_adapter_weight_checksum(),
            "shared_adapter_state_sha256": self._shared_adapter_state_checksum(),
        }
        if self._uses_m_only:
            feature_bases = {
                name: basis.detach().cpu()
                for name, basis in self._m_only_bases.items()
            }
            state["shared_adapter_m_only_state"] = {
                "version": 1,
                "feature_bases": feature_bases,
                "feature_basis_sha256": self._m_only_feature_basis_checksum(
                    feature_bases
                ),
            }
            if not mid_task:
                state["shared_adapter_m_only_state"]["task_boundary_rng_state"] = (
                    self._capture_training_rng_state()
                )
            active_factors = []
            for prefix, adapter, kind, _ in self._m_only_adapter_targets():
                factor = getattr(
                    adapter, "keeplora_m_only_{}".format(kind), None
                )
                if factor is None:
                    continue
                active_factors.append(
                    {
                        "prefix": prefix,
                        "kind": kind,
                        "in_features": factor.in_features,
                        "out_features": factor.out_features,
                        "rank": factor.rank,
                        "scaling": factor.scaling,
                    }
                )
            state["shared_adapter_m_only_active_factors"] = active_factors
        elif getattr(self, "_uses_dense_control", False) and not mid_task:
            state["dense_control_task_boundary_rng_state"] = self._capture_training_rng_state()
        return state

    def _shared_adapter_state_checksum(self):
        network = (
            self._network.module
            if isinstance(self._network, nn.DataParallel)
            else self._network
        )
        digest = hashlib.sha256()
        for block_index, block in enumerate(network.convnet.blocks):
            adapter = getattr(block, "adaptmlp", None)
            if adapter is None:
                continue
            for name, tensor in sorted(adapter.state_dict().items()):
                value = tensor.detach().cpu().contiguous()
                digest.update(
                    "{}.adaptmlp.{}:{}:{}:".format(
                        block_index, name, tuple(value.shape), value.dtype
                    ).encode("utf-8")
                )
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def _m_only_feature_basis_checksum(self, feature_bases=None):
        if feature_bases is None:
            feature_bases = self._m_only_bases
        digest = hashlib.sha256()
        for name in sorted(feature_bases):
            tensor = feature_bases[name].detach().cpu().float().contiguous()
            digest.update(
                "{}:{}:".format(name, tuple(tensor.shape)).encode("utf-8")
            )
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def _m_only_initial_factors_checksum(self):
        digest = hashlib.sha256()
        for prefix, adapter, kind, _ in self._m_only_adapter_targets():
            factor = getattr(adapter, "keeplora_m_only_{}".format(kind), None)
            if factor is None:
                continue
            for name, tensor in (
                ("A", factor.lora_A),
                ("B", factor.lora_B),
            ):
                value = tensor.detach().cpu().contiguous()
                digest.update(
                    "{}.{}.{}:{}:{}:".format(
                        prefix, kind, name, tuple(value.shape), factor.scaling
                    ).encode("utf-8")
                )
                digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def _shared_adapter_weight_checksum(self):
        network = (
            self._network.module
            if isinstance(self._network, nn.DataParallel)
            else self._network
        )
        digest = hashlib.sha256()
        for block_index, block in enumerate(network.convnet.blocks):
            adapter = getattr(block, "adaptmlp", None)
            if adapter is None:
                continue
            for name, linear in (
                ("down_proj", adapter.down_proj),
                ("up_proj", adapter.up_proj),
            ):
                for parameter_name, parameter in (
                    ("weight", linear.weight),
                    ("bias", linear.bias),
                ):
                    if parameter is None:
                        continue
                    tensor = parameter.detach().cpu().contiguous()
                    digest.update(
                        "{}.adaptmlp.{}.{}:{}:".format(
                            block_index,
                            name,
                            parameter_name,
                            tuple(tensor.shape),
                        ).encode("utf-8")
                    )
                    digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def after_task(self):
        if self._uses_keeplora:
            self._finalize_keeplora_task()
        elif self._uses_m_only:
            self._finalize_shared_adapter_m_only_task()
        if self.old_ae is not None:
            self._rae_last_state_snapshot = self._rae_state_snapshot()
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network

    def _validate_m_only_config(self):
        rank_down = int(self.args.get("rank_down_max", 32))
        rank_up = int(self.args.get("rank_up_max", 32))
        threshold = float(self.args.get("feature_threshold", 0.99))
        scaling_mode = str(
            self.args.get("scaling_mode", "fixed_effective_scaling")
        ).lower()
        target_scaling = float(self.args.get("target_scaling", 0.0625))
        lora_alpha = float(self.args.get("lora_alpha", 2.0))
        bias_policy = str(
            self.args.get("shared_adapter_m_only_bias_policy", "freeze_bias")
        ).lower()
        optimizer = str(
            self.args.get(
                "incremental_optimizer",
                self.args.get("shared_adapter_m_only_optimizer", "sgd"),
            )
        ).lower()
        empty_policy = str(
            self.args.get("shared_adapter_m_only_empty_residual_policy", "no_update")
        ).lower()
        if rank_down < 1 or rank_up < 1:
            raise ValueError("rank_down_max and rank_up_max must be positive.")
        if not 0 < threshold <= 1:
            raise ValueError("feature_threshold must be in (0, 1].")
        if scaling_mode not in ("fixed_effective_scaling", "alpha_over_rank"):
            raise ValueError(
                "scaling_mode must be 'fixed_effective_scaling' or 'alpha_over_rank'."
            )
        if target_scaling <= 0 or lora_alpha <= 0:
            raise ValueError("target_scaling and lora_alpha must be positive.")
        if bias_policy not in ("freeze_bias", "train_bias_unconstrained"):
            raise ValueError(
                "shared_adapter_m_only_bias_policy must be 'freeze_bias' or "
                "'train_bias_unconstrained'."
            )
        if optimizer not in ("adamw", "sgd"):
            raise ValueError(
                "incremental_optimizer must be 'adamw' or 'sgd'."
            )
        momentum = float(
            self.args.get("shared_adapter_m_only_sgd_momentum", 0.9)
        )
        if not math.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError(
                "shared_adapter_m_only_sgd_momentum must be finite and in [0, 1)."
            )
        if empty_policy not in ("no_update", "error"):
            raise ValueError(
                "shared_adapter_m_only_empty_residual_policy must be "
                "'no_update' or 'error'."
            )
        for key in ("shared_adapter_m_only_grad_batches",):
            if int(self.args.get(key, 0)) < 0:
                raise ValueError("{} must be non-negative.".format(key))
        if int(self.args.get("shared_adapter_m_only_feature_rows_per_batch", 0)) < 0:
            raise ValueError(
                "shared_adapter_m_only_feature_rows_per_batch must be non-negative."
            )
        for module in ("keeplora", "classifier", "rae"):
            lr = float(
                self.args.get(
                    "shared_adapter_m_only_adamw_{}_lr".format(module),
                    self.args.get(
                        "adamw_rae_lr" if module == "rae" else "adamw_lr",
                        self.args.get(
                            "ae_init_lr" if module == "rae" else "init_lr", 1e-3
                        ),
                    ),
                )
            )
            weight_decay = float(
                self.args.get(
                    "shared_adapter_m_only_adamw_{}_weight_decay".format(module),
                    self.args.get(
                        "adamw_rae_weight_decay"
                        if module == "rae"
                        else "adamw_weight_decay",
                        self.args.get(
                            "ae_weight_decay"
                            if module == "rae"
                            else "weight_decay",
                            0.0,
                        ),
                    ),
                )
            )
            if not math.isfinite(lr) or lr <= 0:
                raise ValueError(
                    "M-only AdamW {} learning rate must be finite and > 0."
                    .format(module)
                )
            if not math.isfinite(weight_decay) or weight_decay < 0:
                raise ValueError(
                    "M-only AdamW {} weight decay must be finite and >= 0."
                    .format(module)
                )

    def _m_only_adapter_targets(self):
        network, convnet = self._keeplora_convnet()
        for block_index, block in enumerate(convnet.blocks):
            adapter = getattr(block, "adaptmlp", None)
            if adapter is None:
                continue
            yield (
                "blocks.{}.adaptmlp".format(block_index),
                adapter,
                "down",
                adapter.down_proj,
            )
            yield (
                "blocks.{}.adaptmlp".format(block_index),
                adapter,
                "up",
                adapter.up_proj,
            )

    def _m_only_data_loader(self, dataset, task_id, purpose, shuffle=False):
        purpose_offsets = {"gradient": 104729, "features": 130363}
        generator = torch.Generator()
        generator.manual_seed(
            int(self.args["seed"])
            + 1000003 * int(task_id)
            + purpose_offsets[purpose]
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=(purpose == "gradient"),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            generator=generator,
        )

    @staticmethod
    def _snapshot_random_state():
        return (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )

    @staticmethod
    def _restore_random_state(state):
        python_state, numpy_state, torch_state, cuda_state = state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    def _collect_shared_adapter_m_only_basis(self, class_indices, task_id):
        if self.data_manager is None:
            raise RuntimeError("DataManager is unavailable for M-only basis collection.")
        dataset = self.data_manager.get_dataset(
            np.asarray(class_indices), source="train", mode="test"
        )
        loader = self._m_only_data_loader(
            dataset, task_id, purpose="features", shuffle=False
        )
        network, _ = self._keeplora_convnet()
        adapters = {}
        for prefix, adapter, _, _ in self._m_only_adapter_targets():
            adapters[prefix] = adapter
        rows_per_batch = int(
            self.args.get("shared_adapter_m_only_feature_rows_per_batch", 0)
        )
        for adapter in adapters.values():
            adapter.start_m_only_feature_collection(rows_per_batch)

        was_training = network.training
        network.eval()
        try:
            with torch.no_grad():
                for _, inputs, _ in loader:
                    network.extract_vector(inputs.to(self._device, non_blocking=True))
        finally:
            if was_training:
                network.train()

        threshold = float(self.args.get("feature_threshold", 0.99))
        for prefix, adapter in adapters.items():
            moments = adapter.finish_m_only_feature_collection()
            for name in ("down", "up"):
                key = "{}.{}_proj".format(prefix, name)
                previous = self._m_only_bases.get(
                    key, torch.empty(moments[name].shape[0], 0)
                )
                basis, stats = update_feature_basis(
                    moments[name], previous, threshold
                )
                self._m_only_bases[key] = basis.detach().cpu()
                logging.info(
                    "M-only feature basis task=%d module=%s input_dim=%d "
                    "rank_M=%d previous_rank=%d new_directions=%d "
                    "residual_dimension=%d captured_energy=%.8g "
                    "residual_energy_ratio=%.8g orthogonality_error=%.8g "
                    "cross_orthogonality_error=%.8g adapter_output_norm=%.8g "
                    "rows_per_batch=%d",
                    task_id,
                    key,
                    moments[name].shape[0],
                    stats["rank"],
                    stats["previous_rank"],
                    stats["new_rank"],
                    stats["residual_dimension"],
                    stats["captured_energy"],
                    stats["residual_energy_ratio"],
                    stats["orthogonality_error"],
                    stats["cross_orthogonality_error"],
                    moments["output_norm"],
                    rows_per_batch,
                )
                logging.info(
                    "M-only feature sample count task=%d module=%s "
                    "captured_rows=%d",
                    task_id,
                    key,
                    moments["down_rows" if name == "down" else "up_rows"],
                )

    def _ensure_m_only_task_zero_basis(self):
        expected_keys = {
            self._m_only_weight_key(prefix, kind)
            for prefix, _, kind, _ in self._m_only_adapter_targets()
        }
        missing_keys = expected_keys - set(self._m_only_bases)
        if not missing_keys:
            return
        if self._m_only_bases or self._cur_task != 1:
            raise RuntimeError(
                "Previous-task M-only basis is incomplete; missing {}."
                .format(sorted(missing_keys))
            )
        if self._known_classes <= 0:
            raise RuntimeError("Cannot initialize M-only factors without Task-0 classes.")
        self._collect_shared_adapter_m_only_basis(
            np.arange(self._known_classes), task_id=0
        )

    def _initialize_shared_adapter_m_only_task(self):
        if self._cur_task < 1:
            raise RuntimeError("M-only KeepLoRA must not be initialized in Task 0.")
        self._ensure_m_only_task_zero_basis()
        targets = list(self._m_only_adapter_targets())
        weights = [linear.weight for _, _, _, linear in targets]
        original_requires_grad = [weight.requires_grad for weight in weights]
        for weight in weights:
            weight.requires_grad_(True)

        gradients = {
            self._m_only_weight_key(prefix, kind): torch.zeros_like(linear.weight)
            for prefix, _, kind, linear in targets
        }
        max_batches = int(self.args.get("shared_adapter_m_only_grad_batches", 0))
        seen_batches = 0
        probe_inputs = None
        invariance_reference = None
        network, _ = self._keeplora_convnet()
        was_training = network.training
        random_state = self._snapshot_random_state()
        try:
            network.train()
            for batch_index, (_, inputs, labels) in enumerate(self.m_only_init_loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                seen_batches += 1
                inputs = inputs.to(self._device, non_blocking=True)
                labels = labels.to(self._device, non_blocking=True)
                if probe_inputs is None:
                    probe_inputs = inputs.detach().clone()
                    network.eval()
                    with torch.no_grad():
                        invariance_reference = network.extract_vector(probe_inputs)
                    network.train()
                _, loss_c, loss_incremental = self._compute_rt_loss(
                    inputs,
                    labels,
                    epoch=0,
                    warmup_epoch=self.args["warmup_epoch"],
                )
                gradients_for_batch = torch.autograd.grad(
                    loss_c + loss_incremental,
                    weights,
                    allow_unused=True,
                )
                for (prefix, _, kind, _), gradient in zip(
                    targets, gradients_for_batch
                ):
                    if gradient is not None:
                        gradients[self._m_only_weight_key(prefix, kind)].add_(
                            gradient.detach()
                        )
        finally:
            self._restore_random_state(random_state)
            network.train(was_training)
            for weight, requires_grad in zip(weights, original_requires_grad):
                weight.requires_grad_(requires_grad)
        if seen_batches == 0:
            raise RuntimeError("M-only KeepLoRA requires at least one gradient batch.")
        for gradient in gradients.values():
            gradient.div_(seen_batches)

        rank_caps = {
            "down": int(self.args.get("rank_down_max", 32)),
            "up": int(self.args.get("rank_up_max", 32)),
        }
        factors = {}
        for prefix, adapter, kind, linear in targets:
            weight = linear.weight
            key = self._m_only_weight_key(prefix, kind)
            basis = self._m_only_bases.get(
                key, torch.empty(weight.shape[1], 0)
            ).to(device=weight.device, dtype=torch.float32)
            factor, stats = MOnlyKeepLoRA.from_gradient(
                gradients[key],
                basis,
                rank_caps[kind],
                str(self.args.get("scaling_mode", "fixed_effective_scaling")).lower(),
                float(self.args.get("target_scaling", 0.0625)),
                float(self.args.get("lora_alpha", 2.0)),
                str(
                    self.args.get(
                        "shared_adapter_m_only_empty_residual_policy", "no_update"
                    )
                ).lower(),
            )
            if factor is not None:
                original_weight = weight.detach().clone()
                factor = factor.to(device=weight.device, dtype=weight.dtype)
                weight.data.sub_(
                    factor.get_delta_weight().to(weight.device, weight.dtype)
                )
                reconstructed = weight.detach() + factor.get_delta_weight().to(
                    weight.device, weight.dtype
                )
                forward_error = float((reconstructed - original_weight).abs().max().item())
                if not torch.allclose(
                    reconstructed, original_weight, rtol=1e-4, atol=1e-5
                ):
                    raise RuntimeError(
                        "M-only factorization changed the effective weight for {}."
                        .format(key)
                    )
            else:
                forward_error = 0.0
            weight.requires_grad_(False)
            if linear.bias is not None:
                linear.bias.requires_grad_(
                    self.args.get(
                        "shared_adapter_m_only_bias_policy", "freeze_bias"
                    ).lower() == "train_bias_unconstrained"
                )
            factors.setdefault(prefix, {})[kind] = factor
            logging.info(
                "M-only init task=%d module=%s configured_rank=%d actual_rank=%d "
                "B_parameters=%d residual_dimension=%d gradient_norm=%.8g "
                "projected_gradient_norm=%.8g projected_gradient_ratio=%.8g "
                "top_rank_energy_ratio=%.8g A_orthogonality_error=%.8g "
                "A_orthonormality_error=%.8g effective_update_norm=%.8g "
                "effective_scaling=%.8g effective_alpha=%.8g "
                "weight_compensation_error=%.8g bias_policy=%s",
                self._cur_task,
                key,
                rank_caps[kind],
                stats["actual_rank"],
                0 if factor is None else factor.lora_B.numel(),
                stats["residual_dimension"],
                stats["gradient_norm"],
                stats["projected_gradient_norm"],
                stats["projected_gradient_ratio"],
                stats["top_rank_energy_ratio"],
                stats["a_orthogonality_error"],
                stats["a_orthonormality_error"],
                stats["effective_update_norm"],
                stats["scaling"],
                stats["alpha"],
                forward_error,
                self.args.get("shared_adapter_m_only_bias_policy", "freeze_bias"),
            )
        for prefix, adapter in {
            (prefix, adapter) for prefix, adapter, _, _ in targets
        }:
            adapter.attach_m_only_factors(
                factors[prefix].get("down"), factors[prefix].get("up")
            )

        if invariance_reference is not None:
            network.eval()
            try:
                with torch.no_grad():
                    actual_features = network.extract_vector(probe_inputs)
            finally:
                network.train(was_training)
            maximum_error = float(
                (actual_features - invariance_reference).abs().max().item()
            )
            if not torch.allclose(
                actual_features,
                invariance_reference,
                rtol=KEEPLORA_FEATURE_RTOL,
                atol=KEEPLORA_FEATURE_ATOL,
            ):
                raise RuntimeError(
                    "Attaching M-only factors changed the Shared Adapter forward "
                    "(max error {:.8g}, rtol={:g}, atol={:g}).".format(
                        maximum_error, KEEPLORA_FEATURE_RTOL, KEEPLORA_FEATURE_ATOL
                    )
                )
            logging.info(
                "M-only factorization forward invariance task=%d max_abs_error=%.8g",
                self._cur_task,
                maximum_error,
            )
        logging.info(
            "M-only initialization task=%d gradient_batches=%d feature_basis_sha256=%s "
            "factor_AB_sha256=%s",
            self._cur_task,
            seen_batches,
            self._m_only_feature_basis_checksum(),
            self._m_only_initial_factors_checksum(),
        )
        self._log_m_only_parameter_budget("task_{}".format(self._cur_task))

    @staticmethod
    def _m_only_weight_key(prefix, kind):
        return "{}.{}_proj".format(prefix, kind)

    def _merge_shared_adapter_m_only_task(self):
        merged_count = 0
        for prefix, adapter, kind, linear in self._m_only_adapter_targets():
            factor = getattr(adapter, "keeplora_m_only_{}".format(kind), None)
            if factor is None:
                continue
            weight = linear.weight
            logging.info(
                "M-only effective update task=%d module=%s norm=%.8g scaling=%.8g",
                self._cur_task,
                self._m_only_weight_key(prefix, kind),
                torch.linalg.vector_norm(factor.get_delta_weight()).item(),
                factor.scaling,
            )
            before = weight.detach().clone() + factor.get_delta_weight().to(
                weight.device, weight.dtype
            )
            factor.merge_into(weight)
            error = float((weight.detach() - before).abs().max().item())
            if not torch.allclose(weight.detach(), before, rtol=1e-4, atol=1e-5):
                raise RuntimeError(
                    "M-only merge changed the effective weight for {}."
                    .format(self._m_only_weight_key(prefix, kind))
                )
            merged_count += 1
            logging.info(
                "M-only merge task=%d module=%s effective_weight_error=%.8g",
                self._cur_task,
                self._m_only_weight_key(prefix, kind),
                error,
            )
        for _, adapter in {
            (prefix, adapter)
            for prefix, adapter, _, _ in self._m_only_adapter_targets()
        }:
            adapter.remove_m_only_factors()
            adapter.down_proj.weight.requires_grad_(False)
            adapter.up_proj.weight.requires_grad_(False)
            if self.args.get(
                "shared_adapter_m_only_bias_policy", "freeze_bias"
            ).lower() == "freeze_bias":
                if adapter.down_proj.bias is not None:
                    adapter.down_proj.bias.requires_grad_(False)
                if adapter.up_proj.bias is not None:
                    adapter.up_proj.bias.requires_grad_(False)
        logging.info(
            "M-only task merge complete task=%d merged_linear_updates=%d",
            self._cur_task,
            merged_count,
        )

    def _finalize_shared_adapter_m_only_task(self):
        current_classes = np.arange(self._known_classes, self._total_classes)
        self._collect_shared_adapter_m_only_basis(
            current_classes, task_id=self._cur_task
        )

    def _log_m_only_parameter_budget(self, stage):
        network, convnet = self._keeplora_convnet()
        dense_adapter_parameters = sum(
            parameter.numel()
            for block in convnet.blocks
            for adapter in [getattr(block, "adaptmlp", None)]
            if adapter is not None
            for linear in (adapter.down_proj, adapter.up_proj)
            for parameter in linear.parameters()
        )
        b_parameters = sum(
            factor.lora_B.numel()
            for _, adapter, kind, _ in self._m_only_adapter_targets()
            for factor in [
                getattr(adapter, "keeplora_m_only_{}".format(kind), None)
            ]
            if factor is not None
        )
        logging.info(
            "M-only parameter budget stage=%s total_model_parameters=%d "
            "dense_adapter_parameters=%d incremental_B_parameters=%d "
            "inference_model_parameters=%d",
            stage,
            sum(parameter.numel() for parameter in network.parameters()),
            dense_adapter_parameters,
            b_parameters,
            sum(parameter.numel() for parameter in network.parameters())
            - b_parameters,
        )

    def _keeplora_convnet(self):
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        return network, network.convnet

    def _build_rae(self):
        return AutoencoderSigmoid(
            input_dims=768,
            code_dims=self.args["ae_code_dims"],
            zero_residual=self.rae_zero_init,
            baseline_init_order=(
                getattr(self, "_uses_m_only", False)
                or getattr(self, "_uses_dense_control", False)
            ),
        )

    def _create_rae(self, task_id):
        self.old_ae = self._build_rae().to(self._device)
        self.rae_generation += 1
        self.rae_task_id = task_id
        self._rae_last_state_snapshot = None

    def _prepare_rae_for_task(self):
        if self._cur_task >= 1 and (
            self.old_ae is None or self.rae_lifecycle == "per_task"
        ):
            self._create_rae(self._cur_task)
        elif self._cur_task >= 1 and self.old_ae is not None:
            same_state = (
                self._rae_last_state_snapshot is None
                or self._rae_state_matches_snapshot()
            )
            logging.info(
                "RAE mode=%s task_id=%d projector_generation_id=%d "
                "shared_projector_reused=true same_state_as_previous_task=%s "
                "object_id=%d",
                self._rae_mode(),
                self._cur_task,
                self.rae_generation,
                same_state,
                id(self.old_ae),
            )

    def _rae_mode(self):
        if self.rae_zero_init and self.rae_lifecycle == "shared":
            if not getattr(self, "_uses_keeplora", False):
                return "zero_shared_baseline"
            return "zero_shared"
        return "{}_{}".format(
            "zero" if self.rae_zero_init else "random",
            self.rae_lifecycle,
        )

    def _rae_state_snapshot(self):
        if self.old_ae is None:
            return None
        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.old_ae.state_dict().items()
        }

    def _rae_state_matches_snapshot(self):
        if self.old_ae is None or self._rae_last_state_snapshot is None:
            return False
        current = self.old_ae.state_dict()
        return (
            current.keys() == self._rae_last_state_snapshot.keys()
            and all(
                torch.equal(
                    tensor.detach().cpu(),
                    self._rae_last_state_snapshot[name],
                )
                for name, tensor in current.items()
            )
        )

    def _diagnose_rae_initialization(self, train_loader):
        if self.old_ae is None or self._cur_task < 1:
            return
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            _, inputs, _ = next(iter(train_loader))
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        inputs = inputs[:8].to(self._device, non_blocking=True)
        was_training = self.old_ae.training
        self.old_ae.eval()
        reference = self._old_network.extract_vector(inputs)
        with torch.no_grad():
            residual = self.old_ae.decoder(self.old_ae.encoder(reference))
            denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
            residual_ratio = torch.linalg.vector_norm(residual) / denominator
            identity_error = torch.linalg.vector_norm(residual) / denominator

        align_loss = None
        align_gradient_norm = None
        orth_gradient_norm = None
        representative_name = None
        representative_weight = None
        current_features = self._network_module_ptr.extract_vector(inputs)
        align_loss, orth_loss = self._inc_loss_components(
            current_features, reference
        )
        if self._uses_keeplora:
            _, convnet = self._keeplora_convnet()
            targets = list(convnet.named_keeplora_targets())
            if targets:
                representative_name, representative_weight, _ = targets[0]
                was_requires_grad = representative_weight.requires_grad
                representative_weight.requires_grad_(True)
                try:
                    align_gradient = torch.autograd.grad(
                        align_loss,
                        representative_weight,
                        allow_unused=True,
                        retain_graph=True,
                    )[0]
                    orth_gradient = torch.autograd.grad(
                        orth_loss,
                        representative_weight,
                        allow_unused=True,
                    )[0]
                    align_gradient_norm = (
                        0.0
                        if align_gradient is None
                        else align_gradient.detach().norm().item()
                    )
                    orth_gradient_norm = (
                        0.0
                        if orth_gradient is None
                        else orth_gradient.detach().norm().item()
                    )
                finally:
                    representative_weight.requires_grad_(was_requires_grad)
        if was_training:
            self.old_ae.train()
        logging.info(
            "RAE mode=%s task_id=%d projector_generation_id=%d "
            "initial_residual_ratio=%.8g initial_identity_error=%.8g "
            "initial_L_align=%s initial_grad_norm_L_align=%s "
            "initial_grad_norm_L_orth=%s representative_weight=%s "
            "object_id=%d",
            self._rae_mode(),
            self._cur_task,
            self.rae_generation,
            residual_ratio.item(),
            identity_error.item(),
            "n/a" if align_loss is None else "%.8g" % align_loss.detach().item(),
            "n/a" if align_gradient_norm is None else "%.8g" % align_gradient_norm,
            "n/a" if orth_gradient_norm is None else "%.8g" % orth_gradient_norm,
            representative_name or "n/a",
            id(self.old_ae),
        )

    def _initialize_keeplora_task(self, train_loader):
        """Initialize KeepLoRA from the configured RSIAT loss gradient."""
        network, convnet = self._keeplora_convnet()
        targets = list(convnet.named_keeplora_targets())
        if not targets:
            return

        weights = [weight for _, weight, _ in targets]
        for weight in weights:
            weight.requires_grad_(True)
        gradients = {name: torch.zeros_like(weight) for name, weight, _ in targets}
        # KeepLoRA estimates its initialization gradient over the task loader.
        # A positive limit remains available for quick smoke experiments.
        batches = int(self.args.get("keeplora_grad_batches", 0))
        seen_batches = 0
        loss_cos = AngularPenaltySMLoss(
            loss_type="cosface", eps=1e-7, s=self.args["scale"], m=self.args["margin"]
        )
        verify_invariance = bool(
            self.args.get("keeplora_verify_init_invariance", False)
        )
        probe_inputs = None
        reference_features = None
        reference_logits = None
        loss_sums = {}
        total_loss_sum = 0.0

        network.train()
        try:
            for batch_index, (_, inputs, targets_label) in enumerate(train_loader):
                if batches > 0 and batch_index >= batches:
                    break
                seen_batches += 1
                inputs = inputs.to(self._device, non_blocking=True)
                targets_label = targets_label.to(self._device, non_blocking=True)

                if verify_invariance and probe_inputs is None:
                    probe_inputs = inputs.detach().clone()
                    network.eval()
                    with torch.no_grad():
                        reference_features = network.extract_vector(probe_inputs)
                        reference_logits = network.fc(reference_features)["logits"]
                    network.train()

                _, loss, loss_components = self._keeplora_initialization_loss(
                    network, inputs, targets_label, loss_cos
                )
                current_grads = torch.autograd.grad(loss, weights, allow_unused=True)
                for (name, _, _), gradient in zip(targets, current_grads):
                    if gradient is not None:
                        gradients[name].add_(gradient)
                for name, value in loss_components.items():
                    loss_sums[name] = loss_sums.get(name, 0.0) + value.detach().item()
                total_loss_sum += loss.detach().item()
        finally:
            for weight in weights:
                weight.requires_grad_(False)

        if seen_batches == 0:
            raise RuntimeError("KeepLoRA requires at least one batch to initialize a task.")
        for name in gradients:
            gradients[name].div_(seen_batches)

        controller = KeepLoRAController(convnet, logging.info)
        gradient_stats = controller.initialize_from_gradients(
            gradients, verify_invariance=verify_invariance
        )
        network.zero_grad(set_to_none=True)
        logging.info(
            "KeepLoRA initialization mode=%s, task=%d, gradient_batches=%d, "
            "full_initialization_loss=%.6f",
            self.keeplora_init_mode,
            self._cur_task,
            seen_batches,
            total_loss_sum / seen_batches,
        )
        for name, total in loss_sums.items():
            logging.info(
                "KeepLoRA initialization task=%d mean_%s=%.6f",
                self._cur_task,
                name,
                total / seen_batches,
            )
        for name, stats in gradient_stats.items():
            logging.info(
                "KeepLoRA initialization task=%d target=%s "
                "gradient_norm_before_projection=%.6f "
                "gradient_norm_after_projection=%.6f",
                self._cur_task,
                name,
                stats["gradient_norm_before_projection"],
                stats["gradient_norm_after_projection"],
            )
            if verify_invariance:
                logging.info(
                    "KeepLoRA initialization task=%d target=%s "
                    "weight_merge_max_abs_error=%.8g",
                    self._cur_task,
                    name,
                    stats["weight_merge_max_abs_error"],
                )

        if self.keeplora_init_mode == "full_rsiat" and self._cur_task > 0:
            logging.info(
                "L_orth is included in the scalar full RSIAT initialization loss; "
                "its direct gradient with respect to current KeepLoRA weights "
                "may be zero under the existing RSIAT computational graph."
            )

        if verify_invariance:
            network.eval()
            with torch.no_grad():
                actual_features = network.extract_vector(probe_inputs)
                actual_logits = network.fc(actual_features)["logits"]
            network.train()
            feature_error = (actual_features - reference_features).abs().max().item()
            logit_error = (actual_logits - reference_logits).abs().max().item()
            logging.info(
                "KeepLoRA initialization forward invariance task=%d "
                "feature_max_abs_error=%.8g logit_max_abs_error=%.8g",
                self._cur_task,
                feature_error,
                logit_error,
            )
            if not torch.allclose(
                actual_features, reference_features,
                rtol=KEEPLORA_FEATURE_RTOL, atol=KEEPLORA_FEATURE_ATOL,
            ) or not torch.allclose(
                actual_logits, reference_logits, rtol=1e-4, atol=1e-5
            ):
                raise RuntimeError(
                    "KeepLoRA initialization changed the model output beyond "
                    "the configured numerical tolerance."
                )  #tolerance values are set to match the original KeepLoRA implementation - mong muốn là sau khi mới khởi tạo keepLora thì các đầu ra phải giống ban đầu (nhưng do python float32 nên có sai số nhỏ, nên phải dùng allclose để kiểm tra)

    def _keeplora_initialization_loss(self, network, inputs, targets, loss_cos):
        features = network.extract_vector(inputs)
        logits = network.fc(features)["logits"]
        loss_c = loss_cos(
            logits[:, self._known_classes:], targets - self._known_classes
        )
        components = {"L_cos": loss_c}

        if self.keeplora_init_mode == "cosine":
            components["L_init"] = loss_c
            return logits, loss_c, components

        if self._cur_task == 0:
            loss_rs = self.rs_loss_func(features, targets)
            full_loss = loss_c + self.args["lambda_rs"] * loss_rs
            components["L_RS"] = loss_rs
        else:
            features_old = self.old_network_module_ptr.extract_vector(inputs)
            loss_align, loss_orth = self._inc_loss_components(features, features_old)
            full_loss = (
                loss_c
                + self.args["beta"] * loss_align
                + self.args["gamma"] * loss_orth
            )
            components["L_align"] = loss_align
            components["L_orth"] = loss_orth

        components["L_init"] = full_loss
        return logits, full_loss, components

    def _finalize_keeplora_task(self):
        """Persist task directions, then merge the task-local update into the ViT."""
        network, convnet = self._keeplora_convnet()
        controller = KeepLoRAController(convnet, logging.info)
        controller.begin_feature_collection()
        network.eval()
        max_batches = int(self.args.get("keeplora_feature_batches", 0))
        with torch.no_grad():
            for batch_index, (_, inputs, _) in enumerate(self.keeplora_accum_loader):
                if max_batches > 0 and batch_index >= max_batches:
                    break
                network.extract_vector(inputs.to(self._device, non_blocking=True))
        controller.finish_task()
        logging.info("Merged KeepLoRA update and saved feature subspaces for task %d.", self._cur_task)


    def extract_features(self, trainloader, model, args, return_indices=False):
        model = model.eval()
        embedding_list = []
        label_list = []
        index_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                indices, data, label = batch
                data = data.to(self._device, non_blocking=True)
                label = label.to(self._device, non_blocking=True)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
                if return_indices:
                    index_list.append(torch.as_tensor(indices).cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        if return_indices:
            return (
                embedding_list,
                label_list,
                torch.cat(index_list, dim=0),
            )
        return embedding_list, label_list

    def _build_ssca_drift_probe_loader(self, data_manager):
        mode = _validate_ssca_drift_probe_mode(self.args)
        if mode is None:
            return self.train_loader

        dataset_mode = "test" if mode == "paired_deterministic" else "train"
        dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode=dataset_mode,
        )
        generator = torch.Generator()
        generator.manual_seed(int(self.args["seed"]) + self._cur_task)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(mode == "stochastic_unpaired"),
            num_workers=int(self.args.get("stats_num_workers", 2)),
            pin_memory=self.pin_memory,
            persistent_workers=False,
            generator=generator,
        )

    def _extract_ssca_drift_probe_features(self, loader):
        rng_state = self.semantic_drift._rng_snapshot()
        try:
            probe_seed = int(
                torch.empty((), dtype=torch.int64)
                .random_(generator=loader.generator)
                .item()
            )
            random.seed(probe_seed)
            np.random.seed(probe_seed % (2**32))
            torch.manual_seed(probe_seed)
            return self.extract_features(
                loader, self._network, None, return_indices=True
            )
        finally:
            self.semantic_drift._restore_rng(rng_state)

    def incremental_train(self, data_manager):
        mid_task_state = getattr(self, "_mid_task_training_state", None)
        resuming_mid_task = mid_task_state is not None
        if resuming_mid_task:
            if not self._uses_m_only:
                raise ValueError(
                    "Mid-task resume is supported only for keeplora_m_only."
                )
            if int(mid_task_state.get("task_id", -1)) != self._cur_task:
                raise ValueError(
                    "Mid-task checkpoint task_id does not match the loaded model."
                )
        else:
            boundary_rng_state = getattr(self, "_task_boundary_rng_state", None)
            if boundary_rng_state is not None:
                self._restore_training_rng_state(
                    boundary_rng_state, restore_cuda=self._device.type == "cuda"
                )
                self._task_boundary_rng_state = None
            self._cur_task += 1
            self._prepare_rae_for_task()

        task_size = data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + task_size
        if resuming_mid_task:
            if len(self.task_sizes) != self._cur_task + 1:
                raise ValueError(
                    "Mid-task checkpoint has an invalid task_sizes state."
                )
        else:
            self.task_sizes.append(task_size)
            self._network.update_fc(task_size)
        self._log_trainable_parameter_budget("task_{}".format(self._cur_task))
        self._network_module_ptr = self._network
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
    
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        persistent_workers = self.persistent_workers and not (
            self._cur_task > 0 and (self._uses_m_only or self._uses_dense_control)
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
        )
        probe_mode = _validate_ssca_drift_probe_mode(self.args)
        ssca_drift_probe_loader = self.train_loader
        if self._cur_task > 0 and probe_mode is not None:
            ssca_drift_probe_loader = self._build_ssca_drift_probe_loader(
                data_manager
            )
        # KeepLoRA's framework uses a separate shuffled, drop-last loader for
        # its gradient-estimation and feature-subspace passes.  The primary
        # RSIAT training loader remains unchanged.
        if self._uses_keeplora:
            self.keeplora_accum_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=persistent_workers,
            )
        if self._uses_m_only and self._cur_task > 0:
            self.m_only_init_loader = self._m_only_data_loader(
                train_dataset,
                self._cur_task,
                purpose="gradient",
                shuffle=True,
            )
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=persistent_workers,
        )

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self.semantic_drift.enabled and not (self._uses_m_only and self._cur_task > 0):
            self.semantic_drift.record_parameter_audit(self._network)

        if self._cur_task > 0 and self.semantic_drift.enabled:
            self.semantic_drift.prepare(
                self.test_loader.dataset, self.old_network_module_ptr
            )

        if self._cur_task >0:
            self._network.to(self._device)
            if resuming_mid_task:
                pre_task_state = mid_task_state.get("pre_task_state")
                if pre_task_state is None or "features" not in pre_task_state:
                    raise ValueError(
                        "Mid-task checkpoint is missing pre-training features "
                        "required for SSCA."
                    )
                train_embeddings_old = pre_task_state["features"]
                if probe_mode is not None:
                    if "labels" not in pre_task_state or "indices" not in pre_task_state:
                        raise ValueError(
                            "Mid-task checkpoint is missing SSCA probe labels or indices."
                        )
                    old_probe_labels = pre_task_state["labels"]
                    old_probe_indices = pre_task_state["indices"]
                    probe_rng_state = pre_task_state.get("probe_rng_state")
                    if probe_rng_state is not None:
                        ssca_drift_probe_loader.generator.set_state(probe_rng_state)
                self._mid_task_pre_task_state = pre_task_state
            elif probe_mode is None:
                train_embeddings_old, _ = self.extract_features(
                    self.train_loader, self._network, None
                )
                self._mid_task_pre_task_state = {
                    "features": train_embeddings_old.detach().cpu()
                }
            else:
                (
                    train_embeddings_old,
                    old_probe_labels,
                    old_probe_indices,
                ) = self._extract_ssca_drift_probe_features(
                    ssca_drift_probe_loader
                )
                self._mid_task_pre_task_state = {
                    "features": train_embeddings_old.detach().cpu(),
                    "labels": old_probe_labels.detach().cpu(),
                    "indices": old_probe_indices.detach().cpu(),
                    "probe_rng_state": ssca_drift_probe_loader.generator.get_state(),
                }

        self._diagnose_rae_initialization(self.train_loader)
        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        if self._cur_task > 0 and self.semantic_drift.enabled:
            self.semantic_drift.measure_post(
                self.test_loader.dataset, self._network
            )

        if self._cur_task > 0:
            if probe_mode is None:
                train_embeddings_new, _ = self.extract_features(
                    self.train_loader, self._network, None
                )
            else:
                (
                    train_embeddings_new,
                    new_probe_labels,
                    new_probe_indices,
                ) = self._extract_ssca_drift_probe_features(
                    ssca_drift_probe_loader
                )
                id_match = old_probe_indices.eq(new_probe_indices)
                label_match = old_probe_labels.eq(new_probe_labels)
                if (
                    probe_mode == "paired_deterministic"
                    and (
                        not torch.equal(old_probe_indices, new_probe_indices)
                        or not torch.equal(old_probe_labels, new_probe_labels)
                    )
                ):
                    raise RuntimeError(
                        "Paired deterministic SSCA probe changed sample "
                        "identity or label between pre- and post-training."
                    )
                logging.info(
                    "SSCA drift probe mode=%s pre_post_sample_id_match=%.6f "
                    "pre_post_label_match=%.6f",
                    probe_mode,
                    float(id_match.float().mean()),
                    float(label_match.float().mean()),
                )
            means_before_ssca = self._class_means.copy()
            old_class_mean = self._class_means[:self._known_classes]
            gap = self.displacement(
                train_embeddings_old, train_embeddings_new, old_class_mean, 4.0
            )
            if self.args["ssca"] is True:
                mode, gamma, alpha_per_class, task_groups = (
                    _ssca_scales_for_classes(
                        self.args,
                        self._cur_task,
                        self.task_sizes,
                        self._known_classes,
                        gap.dtype,
                    )
                )
                _apply_ssca_gap(
                    self._class_means,
                    self._known_classes,
                    gap,
                    enabled=True,
                    scale=alpha_per_class,
                )
                group_summary = ", ".join(
                    "task{}={:.6f}".format(task, scale)
                    if isinstance(task, int)
                    else "{}={:.6f}".format(task, scale)
                    for task, scale in task_groups
                )
                logging.info(
                    "SSCA scale mode=%s task=%d gamma=%s "
                    "alpha_min=%.6f alpha_max=%.6f alpha_mean=%.6f "
                    "origin_task_scales=[%s]",
                    mode,
                    self._cur_task,
                    "n/a" if gamma is None else "{:.6f}".format(gamma),
                    float(alpha_per_class.min()) if self._known_classes else 1.0,
                    float(alpha_per_class.max()) if self._known_classes else 1.0,
                    float(alpha_per_class.mean()) if self._known_classes else 1.0,
                    group_summary,
                )
            else:
                _apply_ssca_gap(
                    self._class_means,
                    self._known_classes,
                    gap,
                    enabled=False,
                    scale=1.0,
                )
            if self.semantic_drift.enabled:
                self.semantic_drift.record_ssca(
                    means_before_ssca, self._class_means
                )
                self.semantic_drift.write_task()

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if self._cur_task>0 and self.args['ca_epochs']>0 and self.args['ca'] is True:
            self.semantic_drift.measure_pre_ca()
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
        if self._uses_m_only and self._cur_task > 0:
            self._merge_shared_adapter_m_only_task()
        if resuming_mid_task:
            self._mid_task_training_state = None
        self._mid_task_pre_task_state = None

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self._rae_optimizer_mode()
        if self._uses_keeplora:
            self._initialize_keeplora_task(self.keeplora_accum_loader)
        elif (
            self._uses_m_only
            and self._cur_task > 0
            and getattr(self, "_mid_task_training_state", None) is None
        ):
            self._initialize_shared_adapter_m_only_task()
        observer = getattr(self, "semantic_drift", None)
        if getattr(self, "_uses_m_only", False) and self._cur_task > 0 and observer is not None:
            observer.record_parameter_audit(self._network)
        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {'name': 'backbone_last_block',
                 'params': self._network.convnet.blocks[-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'name': 'backbone_other_blocks',
                 'params': self._network.convnet.blocks[:-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'name': 'classifier', 'params': self._network.fc.parameters(),
                 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
            ]
            optimizer, classifier_optimizer = self._build_initial_optimizers(
                param_groups
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            classifier_scheduler = (
                optim.lr_scheduler.CosineAnnealingLR(
                    classifier_optimizer,
                    T_max=self.tuned_epochs,
                    eta_min=self.min_lr,
                )
                if classifier_optimizer is not None
                else None
            )
            log_count_parameter(param_groups)
            self._init_train(
                train_loader,
                test_loader,
                optimizer,
                scheduler,
                self.args['warmup_epoch'],
                classifier_optimizer=classifier_optimizer,
                classifier_scheduler=classifier_scheduler,
            )
        else:
            self.tuned_epochs = self.args['inc_epochs']
            param_groups = []
            param_groups.append(
                {'name': 'backbone', 'params': self._network.convnet.parameters(),
                 'lr': self.init_lr, 'weight_decay': self.weight_decay})
            param_groups.append(
                {'name': 'classifier', 'params': self._network.fc.parameters(),
                 'lr': self.init_lr, 'weight_decay': self.weight_decay})
            param_groups.append(
                {'name': 'rae', 'params': self.old_ae.parameters(),
                 'lr': self.args['ae_init_lr'], 'weight_decay': self.args['ae_weight_decay']})
            optimizers = self._build_incremental_optimizers(
                param_groups
            )
            if len(optimizers) == 3:
                optimizer, rae_optimizer, classifier_optimizer = optimizers
            else:
                optimizer, rae_optimizer = optimizers
                classifier_optimizer = None
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            rae_scheduler = (
                optim.lr_scheduler.CosineAnnealingLR(
                    rae_optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr
                )
                if rae_optimizer is not None
                else None
            )
            classifier_scheduler = (
                optim.lr_scheduler.CosineAnnealingLR(
                    classifier_optimizer,
                    T_max=self.tuned_epochs,
                    eta_min=self.min_lr,
                )
                if classifier_optimizer is not None
                else None
            )
            log_count_parameter(param_groups)
            self._init_train(
                train_loader,
                test_loader,
                optimizer,
                scheduler,
                self.args['warmup_epoch'],
                rae_optimizer=rae_optimizer,
                rae_scheduler=rae_scheduler,
                classifier_optimizer=classifier_optimizer,
                classifier_scheduler=classifier_scheduler,
            )

    def _classifier_optimizer_mode(self):
        if getattr(self, "_uses_m_only", False) and self._cur_task > 0:
            return "main"
        mode = str(self.args.get("classifier_optimizer", "main")).lower()
        if mode not in ("main", "sgd", "adamw"):
            raise ValueError(
                "classifier_optimizer must be 'main', 'sgd', or 'adamw', got {!r}."
                .format(mode)
            )
        return mode

    @staticmethod
    def _trainable_parameter_ids(param_groups):
        return {
            id(parameter)
            for group in param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        }

    @staticmethod
    def _optimizer_parameter_ids(optimizer):
        return {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        }

    def _assert_optimizer_ownership(self, named_optimizers, expected_ids):
        owners = {}
        for name, optimizer in named_optimizers:
            if optimizer is None:
                continue
            parameter_ids = self._optimizer_parameter_ids(optimizer)
            for parameter_id in parameter_ids:
                if parameter_id in owners:
                    raise RuntimeError(
                        "Parameter is assigned to both {} and {} optimizers."
                        .format(owners[parameter_id], name)
                    )
                owners[parameter_id] = name
        actual_ids = set(owners)
        if actual_ids != expected_ids:
            raise RuntimeError(
                "Optimizer parameter coverage mismatch: missing={}, unexpected={}"
                .format(
                    len(expected_ids - actual_ids),
                    len(actual_ids - expected_ids),
                )
            )

    def _build_initial_optimizers(self, param_groups):
        classifier_mode = self._classifier_optimizer_mode()
        if classifier_mode == "main":
            return self._build_main_optimizer(
                param_groups, incremental=False
            ), None

        groups = [dict(group, params=list(group["params"])) for group in param_groups]
        classifier_groups = [
            group for group in groups if group["name"] == "classifier"
        ]
        if len(classifier_groups) != 1:
            raise RuntimeError(
                "A separate classifier optimizer requires exactly one classifier group."
            )
        classifier_group = classifier_groups[0]
        main_groups = [group for group in groups if group is not classifier_group]
        main_optimizer = self._build_main_optimizer(
            main_groups,
            incremental=False,
            expected_parameter_ids=self._trainable_parameter_ids(main_groups),
        )
        classifier_optimizer = self._build_separate_classifier_optimizer(
            classifier_group, classifier_mode
        )
        self._assert_optimizer_ownership(
            (
                ("main/backbone", main_optimizer),
                ("{}/classifier".format(classifier_mode.upper()), classifier_optimizer),
            ),
            self._trainable_parameter_ids(groups),
        )
        return main_optimizer, classifier_optimizer

    def _rae_optimizer_mode(self):
        if getattr(self, "_uses_m_only", False) and self._cur_task > 0:
            return "main"
        mode = str(self.args.get("rae_optimizer", "main")).lower()
        if mode not in ("main", "sgd"):
            raise ValueError(
                "rae_optimizer must be 'main' or 'sgd', got {!r}."
                .format(mode)
            )
        main_optimizer = str(self.args.get("optimizer", "sgd")).lower()
        if mode == "sgd" and main_optimizer != "adamw":
            raise ValueError(
                "rae_optimizer='sgd' requires optimizer='adamw'."
            )
        return mode

    def _build_incremental_optimizers(self, param_groups):
        classifier_mode = self._classifier_optimizer_mode()
        if classifier_mode != "main":
            return self._build_incremental_optimizers_with_separate_classifier(
                param_groups
            )
        if self._rae_optimizer_mode() != "sgd":
            return self._build_main_optimizer(
                param_groups, incremental=True
            ), None

        for group in param_groups:
            group["params"] = list(group["params"])
        rae_groups = [group for group in param_groups if group["name"] == "rae"]
        if len(rae_groups) != 1:
            raise RuntimeError(
                "Separate RAE SGD requires exactly one RAE parameter group."
            )
        rae_group = rae_groups[0]
        main_groups = [
            group for group in param_groups if group["name"] != "rae"
        ]
        main_optimizer = self._build_main_optimizer(
            main_groups, incremental=True, include_rae=False
        )
        rae_optimizer = self._build_rae_sgd_optimizer(rae_group)
        main_ids = {
            id(parameter)
            for group in main_optimizer.param_groups
            for parameter in group["params"]
        }
        rae_ids = {
            id(parameter)
            for group in rae_optimizer.param_groups
            for parameter in group["params"]
        }
        expected_rae_ids = {
            id(parameter)
            for parameter in self.old_ae.parameters()
            if parameter.requires_grad
        }
        if main_ids & rae_ids or rae_ids != expected_rae_ids:
            raise RuntimeError(
                "Separate RAE optimizer parameter coverage mismatch."
            )
        return main_optimizer, rae_optimizer

    def _build_incremental_optimizers_with_separate_classifier(self, param_groups):
        groups = [dict(group, params=list(group["params"])) for group in param_groups]
        classifier_mode = self._classifier_optimizer_mode()
        classifier_groups = [
            group for group in groups if group["name"] == "classifier"
        ]
        if len(classifier_groups) != 1:
            raise RuntimeError(
                "A separate classifier optimizer requires exactly one classifier group."
            )
        classifier_group = classifier_groups[0]
        classifier_optimizer = self._build_separate_classifier_optimizer(
            classifier_group, classifier_mode
        )
        main_groups = [group for group in groups if group is not classifier_group]

        if self._rae_optimizer_mode() == "sgd":
            rae_groups = [group for group in main_groups if group["name"] == "rae"]
            if len(rae_groups) != 1:
                raise RuntimeError(
                    "Separate RAE SGD requires exactly one RAE parameter group."
                )
            rae_group = rae_groups[0]
            backbone_groups = [
                group for group in main_groups if group is not rae_group
            ]
            main_optimizer = self._build_main_optimizer(
                backbone_groups,
                incremental=True,
                include_rae=False,
                expected_parameter_ids=self._trainable_parameter_ids(
                    backbone_groups
                ),
            )
            rae_optimizer = self._build_rae_sgd_optimizer(rae_group)
        else:
            main_optimizer = self._build_main_optimizer(
                main_groups,
                incremental=True,
                expected_parameter_ids=self._trainable_parameter_ids(main_groups),
            )
            rae_optimizer = None

        expected_ids = self._trainable_parameter_ids(groups)
        self._assert_optimizer_ownership(
            (
                ("main/backbone", main_optimizer),
                ("SGD/RAE", rae_optimizer),
                (
                    "{}/classifier".format(classifier_mode.upper()),
                    classifier_optimizer,
                ),
            ),
            expected_ids,
        )
        return main_optimizer, rae_optimizer, classifier_optimizer

    def _build_separate_classifier_optimizer(self, group, mode):
        if mode == "sgd":
            return self._build_classifier_sgd_optimizer(group)
        if mode == "adamw":
            return self._build_classifier_adamw_optimizer(group)
        raise ValueError(
            "A separate classifier optimizer requires mode 'sgd' or 'adamw'."
        )

    def _build_classifier_sgd_optimizer(self, group):
        parameters = [
            parameter for parameter in group["params"] if parameter.requires_grad
        ]
        expected_parameters = {
            id(parameter)
            for parameter in self._network.fc.parameters()
            if parameter.requires_grad
        }
        if not parameters:
            raise RuntimeError("Classifier SGD parameter group is empty.")
        if {id(parameter) for parameter in parameters} != expected_parameters:
            raise RuntimeError(
                "Classifier SGD parameter group does not match the current "
                "trainable classifier parameters."
            )

        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("Classifier SGD learning rate must be finite and > 0.")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError(
                "Classifier SGD weight decay must be finite and >= 0."
            )
        momentum = 0.9
        optimizer = optim.SGD(
            [
                {
                    "name": "classifier",
                    "params": parameters,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            ],
            momentum=momentum,
        )
        logging.info(
            "Classifier optimizer initialized task=%d optimizer=SGD "
            "tensors=%d trainable_parameters=%d lr=%g weight_decay=%g "
            "momentum=%g nesterov=%s scheduler=CosineAnnealingLR "
            "scheduler_step=per_epoch",
            self._cur_task,
            len(parameters),
            sum(parameter.numel() for parameter in parameters),
            lr,
            weight_decay,
            momentum,
            optimizer.defaults["nesterov"],
        )
        return optimizer

    def _build_classifier_adamw_optimizer(self, group):
        parameters = [
            parameter for parameter in group["params"] if parameter.requires_grad
        ]
        expected_parameters = {
            id(parameter)
            for parameter in self._network.fc.parameters()
            if parameter.requires_grad
        }
        if not parameters:
            raise RuntimeError("Classifier AdamW parameter group is empty.")
        if {id(parameter) for parameter in parameters} != expected_parameters:
            raise RuntimeError(
                "Classifier AdamW parameter group does not match the current "
                "trainable classifier parameters."
            )

        lr = float(self.args.get("adamw_lr", 1e-3))
        weight_decay = float(self.args.get("adamw_weight_decay", 0.01))
        betas = tuple(
            float(value)
            for value in self.args.get("adamw_betas", (0.9, 0.999))
        )
        eps = float(self.args.get("adamw_eps", 1e-8))
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("Classifier AdamW learning rate must be finite and > 0.")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError(
                "Classifier AdamW weight decay must be finite and >= 0."
            )
        if not math.isfinite(eps) or eps <= 0 or len(betas) != 2:
            raise ValueError(
                "Classifier AdamW requires finite eps > 0 and exactly two betas."
            )
        if any(not math.isfinite(beta) or beta < 0 or beta >= 1 for beta in betas):
            raise ValueError("Classifier AdamW beta values must be finite and in [0, 1).")

        optimizer = optim.AdamW(
            [
                {
                    "name": "classifier",
                    "params": parameters,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            ],
            betas=betas,
            eps=eps,
        )
        logging.info(
            "Classifier optimizer initialized task=%d optimizer=AdamW "
            "tensors=%d trainable_parameters=%d lr=%g weight_decay=%g "
            "betas=%s eps=%g scheduler=CosineAnnealingLR "
            "scheduler_step=per_epoch",
            self._cur_task,
            len(parameters),
            sum(parameter.numel() for parameter in parameters),
            lr,
            weight_decay,
            betas,
            eps,
        )
        return optimizer

    def _build_rae_sgd_optimizer(self, group):
        parameters = [
            parameter
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("RAE SGD parameter group is empty.")

        lr = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        momentum = float(self.args.get("rae_sgd_momentum", 0.9))
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("RAE SGD learning rate must be finite and > 0.")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError(
                "RAE SGD weight decay must be finite and >= 0."
            )
        if not math.isfinite(momentum) or momentum < 0:
            raise ValueError("rae_sgd_momentum must be finite and >= 0.")

        optimizer = optim.SGD(
            [{"name": "rae", "params": parameters}],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        logging.info(
            "RAE optimizer initialized task=%d optimizer=SGD tensors=%d "
            "trainable_parameters=%d lr=%g momentum=%g weight_decay=%g "
            "scheduler=CosineAnnealingLR scheduler_step=per_epoch",
            self._cur_task,
            len(parameters),
            sum(parameter.numel() for parameter in parameters),
            lr,
            momentum,
            weight_decay,
        )
        return optimizer

    def _build_main_optimizer(
        self,
        param_groups,
        incremental,
        include_rae=True,
        expected_parameter_ids=None,
    ):
        for group in param_groups:
            group["params"] = list(group["params"])
        grouped_ids = [
            id(parameter)
            for group in param_groups
            for parameter in group["params"]
        ]
        if len(grouped_ids) != len(set(grouped_ids)):
            raise RuntimeError("Optimizer parameter groups contain duplicates.")

        optimizer_name = str(self.args.get("optimizer", "sgd")).lower()
        if incremental and getattr(self, "_uses_m_only", False):
            optimizer_name = str(
                self.args.get(
                    "incremental_optimizer",
                    self.args.get("shared_adapter_m_only_optimizer", "sgd"),
                )
            ).lower()
            param_groups = [
                group
                for group in param_groups
                if group["params"]
            ]
            for group in param_groups:
                group["params"] = [
                    parameter
                    for parameter in group["params"]
                    if parameter.requires_grad
                ]
            param_groups = [group for group in param_groups if group["params"]]
            if not param_groups:
                raise RuntimeError(
                    "M-only incremental training has no active trainable parameters."
                )
        if optimizer_name == "sgd":
            momentum = (
                float(self.args.get("shared_adapter_m_only_sgd_momentum", 0.9))
                if incremental and getattr(self, "_uses_m_only", False)
                else 0.9
            )
            if incremental:
                optimizer = optim.SGD(param_groups, momentum=momentum)
            else:
                optimizer = optim.SGD(
                    param_groups,
                    momentum=0.9,
                    lr=self.init_lr,
                    weight_decay=self.weight_decay,
                )
            logging.info("SGD settings momentum=%g", optimizer.defaults["momentum"])
        elif optimizer_name in ("adam", "adamw"):
            lr = float(self.args.get("adamw_lr", 1e-3))
            weight_decay = float(self.args.get("adamw_weight_decay", 0.01))
            betas = tuple(
                float(value)
                for value in self.args.get("adamw_betas", (0.9, 0.999))
            )
            eps = float(self.args.get("adamw_eps", 1e-8))
            if lr <= 0 or weight_decay < 0 or eps <= 0 or len(betas) != 2:
                raise ValueError(
                    "AdamW requires lr > 0, weight_decay >= 0, eps > 0, "
                    "and exactly two beta values."
                )
            if any(beta < 0 or beta >= 1 for beta in betas):
                raise ValueError("AdamW beta values must be in [0, 1).")

            for group in param_groups:
                group["params"] = [
                    parameter
                    for parameter in group["params"]
                    if parameter.requires_grad
                ]
                group["lr"] = lr
                group["weight_decay"] = weight_decay
                if incremental and getattr(self, "_uses_m_only", False):
                    suffix = {
                        "backbone": "keeplora",
                        "classifier": "classifier",
                        "rae": "rae",
                    }.get(group["name"])
                    if suffix is not None:
                        group["lr"] = float(
                            self.args.get(
                                "shared_adapter_m_only_adamw_{}_lr".format(suffix),
                                self.args.get(
                                    "adamw_rae_lr" if suffix == "rae" else "adamw_lr",
                                    self.args.get(
                                        "ae_init_lr" if suffix == "rae" else "init_lr",
                                        lr,
                                    ),
                                ),
                            )
                        )
                        group["weight_decay"] = float(
                            self.args.get(
                                "shared_adapter_m_only_adamw_{}_weight_decay".format(
                                    suffix
                                ),
                                self.args.get(
                                    "adamw_rae_weight_decay"
                                    if suffix == "rae"
                                    else "adamw_weight_decay",
                                    self.args.get(
                                        "ae_weight_decay"
                                        if suffix == "rae"
                                        else "weight_decay",
                                        weight_decay,
                                    ),
                                ),
                            )
                        )
                elif incremental and group["name"] == "rae":
                    group["lr"] = float(self.args.get("adamw_rae_lr", lr))
                    group["weight_decay"] = float(
                        self.args.get("adamw_rae_weight_decay", weight_decay)
                    )
                if not group["params"] and not getattr(
                    self, "_uses_m_only", False
                ):
                    raise RuntimeError(
                        "Optimizer parameter group {!r} is empty."
                        .format(group["name"])
                    )
            optimizer = optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
            )
            logging.info(
                "AdamW settings lr=%g weight_decay=%g betas=%s eps=%g",
                lr,
                weight_decay,
                betas,
                eps,
            )
        else:
            raise ValueError(
                "optimizer must be 'sgd' or 'adamw', got {!r}."
                .format(optimizer_name)
            )

        network = (
            self._network.module
            if isinstance(self._network, nn.DataParallel)
            else self._network
        )
        if expected_parameter_ids is None:
            expected_parameters = {
                id(parameter)
                for parameter in network.parameters()
                if parameter.requires_grad
            }
            if incremental and include_rae and self.old_ae is not None:
                expected_parameters.update(
                    id(parameter)
                    for parameter in self.old_ae.parameters()
                    if parameter.requires_grad
                )
        else:
            expected_parameters = set(expected_parameter_ids)

        grouped_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if len(grouped_ids) != len(set(grouped_ids)):
            raise RuntimeError("Optimizer parameter groups contain duplicates.")
        grouped_trainable = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        }
        missing_parameters = expected_parameters - grouped_trainable
        unexpected_parameters = grouped_trainable - expected_parameters
        if missing_parameters or unexpected_parameters:
            raise RuntimeError(
                "Optimizer parameter coverage mismatch: missing={}, unexpected={}"
                .format(len(missing_parameters), len(unexpected_parameters))
            )

        logging.info(
            "Main optimizer=%s stage=%s scheduler=CosineAnnealingLR "
            "scheduler_step=per_epoch trainable_parameters=%d",
            optimizer.__class__.__name__,
            "incremental" if incremental else "initial",
            sum(
                parameter.numel()
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.requires_grad
            ),
        )
        for index, group in enumerate(optimizer.param_groups):
            logging.info(
                "Optimizer group=%s index=%d tensors=%d trainable_parameters=%d "
                "lr=%g weight_decay=%g",
                group.get("name", "group_{}".format(index)),
                index,
                len(group["params"]),
                sum(
                    parameter.numel()
                    for parameter in group["params"]
                    if parameter.requires_grad
                ),
                group["lr"],
                group["weight_decay"],
            )
        return optimizer

    def _init_train(
        self,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        warmup_epoch,
        rae_optimizer=None,
        rae_scheduler=None,
        classifier_optimizer=None,
        classifier_scheduler=None,
    ):
        mid_task_state = getattr(self, "_mid_task_training_state", None)
        start_epoch = 0
        if mid_task_state is not None:
            if int(mid_task_state.get("task_id", -1)) != self._cur_task:
                raise ValueError("Mid-task optimizer state belongs to another task.")
            if int(mid_task_state.get("epoch", -1)) < 0:
                raise ValueError("Mid-task checkpoint has an invalid epoch.")
            optimizer.load_state_dict(mid_task_state["optimizer_state_dict"])
            scheduler.load_state_dict(mid_task_state["scheduler_state_dict"])
            for name, component in (
                ("rae_optimizer", rae_optimizer),
                ("rae_scheduler", rae_scheduler),
                ("classifier_optimizer", classifier_optimizer),
                ("classifier_scheduler", classifier_scheduler),
            ):
                saved_state = mid_task_state.get(name + "_state_dict")
                if (component is None) != (saved_state is None):
                    raise ValueError(
                        "Mid-task checkpoint {} presence does not match this run."
                        .format(name)
                    )
                if component is not None:
                    component.load_state_dict(saved_state)
            start_epoch = int(mid_task_state["epoch"]) + 1
            if start_epoch > self.tuned_epochs:
                raise ValueError(
                    "Mid-task checkpoint epoch {} exceeds configured inc_epochs {}."
                    .format(start_epoch - 1, self.tuned_epochs)
                )
            self._restore_training_rng_state(mid_task_state["rng_state"])
            logging.info(
                "Resuming task %d from epoch %d/%d",
                self._cur_task,
                start_epoch + 1,
                self.tuned_epochs,
            )
        if start_epoch == self.tuned_epochs:
            logging.info(
                "Task %d already completed all configured training epochs.",
                self._cur_task,
            )
            return
        prog_bar = tqdm(range(start_epoch, self.tuned_epochs))
        eval_interval = self.args.get("eval_interval", 0)
        
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            losses_c, losses_rt = 0.0, 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                logits, loss_c, loss_rt = self._compute_rt_loss(inputs, targets, epoch, warmup_epoch)
                loss = loss_c + loss_rt
                optimizer.zero_grad()
                if rae_optimizer is not None:
                    rae_optimizer.zero_grad()
                if classifier_optimizer is not None:
                    classifier_optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if rae_optimizer is not None:
                    rae_optimizer.step()
                if classifier_optimizer is not None:
                    classifier_optimizer.step()
                losses += loss.item()
                losses_c += loss_c.item()
                losses_rt += loss_rt.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            if rae_scheduler is not None:
                rae_scheduler.step()
            if classifier_scheduler is not None:
                classifier_scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if self._should_eval_epoch(epoch, self.tuned_epochs, eval_interval):
                test_acc_msg = "{:.2f}".format(
                    self._compute_accuracy(self._network, test_loader)
                )
            else:
                test_acc_msg = "skipped"
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, Losses_rt {:.3f}, Train_accy {:.2f}, Test_accy {}".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / len(train_loader),
                losses_c/len(train_loader),
                losses_rt/len(train_loader),
                train_acc,
                test_acc_msg,
            )
            prog_bar.set_description(info)
            if (
                getattr(self, "_uses_m_only", False)
                and self._cur_task > 0
                and self.args.get("save_mid_task_checkpoints", False)
                and getattr(self, "_mid_task_checkpoint_path", None)
            ):
                training_state = {
                    "task_id": self._cur_task,
                    "epoch": epoch,
                    "pre_task_state": getattr(
                        self, "_mid_task_pre_task_state", None
                    ),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rae_optimizer_state_dict": (
                        None if rae_optimizer is None else rae_optimizer.state_dict()
                    ),
                    "rae_scheduler_state_dict": (
                        None if rae_scheduler is None else rae_scheduler.state_dict()
                    ),
                    "classifier_optimizer_state_dict": (
                        None
                        if classifier_optimizer is None
                        else classifier_optimizer.state_dict()
                    ),
                    "classifier_scheduler_state_dict": (
                        None
                        if classifier_scheduler is None
                        else classifier_scheduler.state_dict()
                    ),
                    "rng_state": self._capture_training_rng_state(),
                }
                self.save_checkpoint(
                    self._mid_task_checkpoint_path,
                    mid_task_training_state=training_state,
                )
        logging.info(info)

    @staticmethod
    def _capture_training_rng_state():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

    @staticmethod
    def _restore_training_rng_state(state, *, restore_cuda=True):
        required = {"python", "numpy", "torch", "cuda"}
        missing = required - set(state)
        if missing:
            raise ValueError(
                "Checkpoint is missing RNG state: {}."
                .format(sorted(missing))
            )
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if restore_cuda and state["cuda"] is not None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Checkpoint contains CUDA RNG state but CUDA is unavailable."
                )
            torch.cuda.set_rng_state_all(state["cuda"])

    def _inc_loss(self, features, features_old):
        loss_align, loss_orth = self._inc_loss_components(features, features_old)
        return self.args["beta"] * loss_align + self.args["gamma"] * loss_orth

    def _inc_loss_components(self, features, features_old):
        features_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        features_old_norm = F.normalize(features_old, p=2, dim=1)
        protos = torch.from_numpy(self._class_means).float().to(self._device,non_blocking=True)
        prototype_mode = self.args.get("lorth_prototype_mode", "projected")
        if prototype_mode == "projected":
            protos = self.old_ae(protos)
        elif prototype_mode != "raw":
            raise ValueError(
                "lorth_prototype_mode must be 'projected' or 'raw', got {!r}."
                .format(prototype_mode)
            )
        protos = F.normalize(protos, p=2, dim=1)
        similarity = torch.matmul(protos, features_old_norm.t())
        loss_orth = similarity.sum() / (similarity.shape[0]*similarity.shape[1])
        return loss_align, loss_orth
        
    def _compute_rt_loss(self, inputs, targets, epoch=None, warmup_epoch=10):     
        loss_cos=AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        features = self._network_module_ptr.extract_vector(inputs)
        logits = self._network_module_ptr.fc(features)["logits"]
        loss_c=loss_cos(logits[:, self._known_classes:], targets - self._known_classes)

        if self._cur_task == 0:
            lambda_rs = self.args["lambda_rs"] * min(1.0, epoch / warmup_epoch)
            loss_base = lambda_rs * self.rs_loss_func(features, targets)
            return logits, loss_c, loss_base
        
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        loss_inc = self._inc_loss(features, features_old)
        return logits, loss_c, loss_inc
    
class RS_Loss(nn.Module):
    def __init__(self, lamda=0.5, margin=0.5):
        super(RS_Loss, self).__init__()
        self.lamda = lamda
        self.margin = margin

    def forward(self, features, labels):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).float().to(device)
        eye = torch.eye(mask.size(0), device=device)
        mask_pos = mask - eye
        mask_neg = 1.0 - mask
        dot_prod = torch.matmul(features, features.t())

        pos_loss = F.relu(1.0 - dot_prod) * mask_pos
        neg_loss = F.relu(dot_prod - self.margin) * mask_neg
        loss = pos_loss.sum() / (mask_pos.sum() + 1e-6) + \
               self.lamda * neg_loss.sum() / (mask_neg.sum() + 1e-6)

        return loss
