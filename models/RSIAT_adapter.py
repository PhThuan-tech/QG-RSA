import copy
import hashlib
import json
import logging
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.bialign import (
    bialign_loss_terms,
    make_identity_reverse_projector,
    module_gradient_record,
)
from utils.bicyc_transport import (
    VALID_MODES,
    affine_geometry,
    analytic_affine_gaussian_transport,
    bicyc_loss_terms,
    make_deterministic_affine,
    module_state_sha256,
    preserve_global_rng_state,
    same_input_feature_pair,
    tensor_mapping_sha256,
    weighted_bicyc_loss,
)
from utils.ca_rescue import (
    affine_distribution_diagnostic,
    capture_rng_state,
    environment_record,
    feature_summary,
    parameter_finite_fraction,
    restore_rng_state,
    sample_source_then_affine,
    tensor_finite_fraction,
)
from utils.inc_net import SimpleVitNet
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import AutoencoderSigmoid, log_count_parameter, tensor2numpy

num_workers = 8
BIALIGN_MODE = "bialign"
TRACK_B_TRANSPORT_MODES = ("forward", "bidirectional", "cycle")
VALID_EXPERIMENT_MODES = VALID_MODES + (BIALIGN_MODE,)


class Learner(BaseLearner):
    """RSIAT with isolated optional transport or BiAlign experiment modes.

    [IMPLEMENTATION ADAPTATION] ``bialign`` replaces only RSIAT's incremental
    alignment term. It reuses old_ae as P_t and adds a lightweight D_t; it does
    not activate the dedicated Track-B affine maps, Stage II, or statistics
    transport.
    """

    def __init__(self, args):
        super().__init__(args)
        if "adapter" not in args["convnet_type"]:
            raise NotImplementedError("Adapter requires Adapter backbone")
        self._network = SimpleVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.num_workers = int(args.get("num_workers", num_workers))
        self.pin_memory = bool(args.get("pin_memory", self._device.type == "cuda"))
        self.persistent_workers = self.num_workers > 0 and bool(
            args.get("persistent_workers", True)
        )
        self.init_lr = args["init_lr"]
        self.weight_decay = (
            args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        )
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

        self._old_most_sentive = []
        self._update_grads = {}
        self.logit_norm = None
        self.tuned_epochs = None
        self.task_sizes = []
        self.rs_loss_func = RS_Loss(self.args["alpha"], self.args["rs_margin"])

        # Original RSIAT P: old -> new. It is not A, D, or part of the cycle.
        self.old_ae = None
        self.ae_residual_mode = self.args.get("ae_residual_mode", "sigmoid")
        if self.ae_residual_mode not in AutoencoderSigmoid.RESIDUAL_MODES:
            raise ValueError(
                "Unknown ae_residual_mode {!r}; expected one of {}".format(
                    self.ae_residual_mode, AutoencoderSigmoid.RESIDUAL_MODES
                )
            )

        self.bicyc_mode = self.args.get("bicyc_mode", "official")
        if self.bicyc_mode not in VALID_EXPERIMENT_MODES:
            raise ValueError(
                "Unknown bicyc_mode {!r}; expected one of {}".format(
                    self.bicyc_mode, VALID_EXPERIMENT_MODES
                )
            )
        self.lambda_bi = float(self.args.get("lambda_bi", 5.0))
        self.lambda_cycle = float(self.args.get("lambda_cycle", 1.0))
        self.transport_init_seed = int(self.args.get("transport_init_seed", 1993000))
        self.transport_stage1_lr = float(self.args.get("transport_stage1_lr", 5e-2))
        self.transport_stage1_weight_decay = float(
            self.args.get("transport_stage1_weight_decay", 1e-4)
        )
        self.transport_stage2_epochs = int(
            self.args.get("transport_stage2_epochs", 30)
        )
        self.transport_stage2_lr = float(
            self.args.get("transport_stage2_lr", 1e-2)
        )
        self.transport_stage2_weight_decay = float(
            self.args.get("transport_stage2_weight_decay", 5e-4)
        )
        self.forward_transport = None
        self.backward_transport = None
        self.reverse_projector = None
        self.bialign_reverse_hidden_dim = int(
            self.args.get("bialign_reverse_hidden_dim", 64))
        self.bialign_init_seed = int(
            self.args.get("bialign_init_seed", 1993000))
        self.transport_initial_hashes = {}
        self.transport_history = []
        self.pre_ca_metrics = {}
        self.experiment_records = []
        self._latest_stage1_history = []
        self._current_transport_record = None
        self.ca_sampling_mode = self.args.get(
            "ca_sampling_mode", "analytic_transport")
        if self.ca_sampling_mode not in (
            "analytic_transport", "sample_source_then_affine"
        ):
            raise ValueError(
                "Unknown ca_sampling_mode: {}".format(self.ca_sampling_mode))
        if (self.ca_sampling_mode == "sample_source_then_affine"
                and bool(self.args.get("ca_covariance_pd_fallback", False))):
            raise ValueError(
                "CA rescue forbids covariance fallback/repair; set "
                "ca_covariance_pd_fallback=false")
        self.ca_rescue_diagnostics = None
        self._ca_source_old_means = None
        self._ca_source_old_covs = None
        self._ca_source_statistics_task = None

    # ---------------- checkpoint state ----------------

    def _checkpoint_extra_state(self):
        payload = {
            "bicyc_mode": self.bicyc_mode,
            "transport_initial_hashes": copy.deepcopy(self.transport_initial_hashes),
            "transport_history": copy.deepcopy(self.transport_history),
            "pre_ca_metrics": copy.deepcopy(self.pre_ca_metrics),
            "experiment_records": copy.deepcopy(self.experiment_records),
        }
        if self.forward_transport is not None:
            payload["forward_transport_state_dict"] = {
                key: value.detach().cpu()
                for key, value in self.forward_transport.state_dict().items()
            }
        if self.backward_transport is not None:
            payload["backward_transport_state_dict"] = {
                key: value.detach().cpu()
                for key, value in self.backward_transport.state_dict().items()
            }
        if getattr(self, "reverse_projector", None) is not None:
            payload["bialign_reverse_projector_state_dict"] = {
                key: value.detach().cpu()
                for key, value in self.reverse_projector.state_dict().items()
            }
            payload["bialign_reverse_projector_lifecycle"] = (
                "identity_reset_each_incremental_transition")
        if getattr(self, "ca_rescue_diagnostics", None) is not None:
            payload["ca_rescue_diagnostics"] = copy.deepcopy(
                self.ca_rescue_diagnostics)
        return payload

    def _transport_seed(self, task, which):
        return self.transport_init_seed + int(task) * 100 + (1 if which == "A" else 2)

    def _bialign_seed(self, task):
        return self.bialign_init_seed + int(task)

    def _restore_transport_state(self, checkpoint):
        self.transport_history = copy.deepcopy(checkpoint.get("transport_history", []))
        self.pre_ca_metrics = copy.deepcopy(checkpoint.get("pre_ca_metrics", {}))
        self.experiment_records = copy.deepcopy(checkpoint.get("experiment_records", []))
        self.ca_rescue_diagnostics = copy.deepcopy(
            checkpoint.get("ca_rescue_diagnostics"))
        self.transport_initial_hashes = copy.deepcopy(
            checkpoint.get("transport_initial_hashes", {})
        )
        self.forward_transport = None
        self.backward_transport = None
        self.reverse_projector = None
        a_state = checkpoint.get("forward_transport_state_dict")
        d_state = checkpoint.get("backward_transport_state_dict")
        reverse_state = checkpoint.get("bialign_reverse_projector_state_dict")
        if a_state is not None:
            self.forward_transport = make_deterministic_affine(
                self.feature_dim, self._transport_seed(self._cur_task, "A"), self._device
            )
            self.forward_transport.load_state_dict(a_state, strict=True)
        if d_state is not None:
            self.backward_transport = make_deterministic_affine(
                self.feature_dim, self._transport_seed(self._cur_task, "D"), self._device
            )
            self.backward_transport.load_state_dict(d_state, strict=True)
        if reverse_state is not None:
            self.reverse_projector = make_identity_reverse_projector(
                self.feature_dim,
                self.bialign_reverse_hidden_dim,
                self._bialign_seed(self._cur_task),
                self._device,
            )
            self.reverse_projector.load_state_dict(reverse_state, strict=True)
        if self._cur_task >= 1:
            if (self.bicyc_mode in TRACK_B_TRANSPORT_MODES
                    and self.forward_transport is None):
                raise ValueError("Incremental transport checkpoint is missing A")
            if (self.bicyc_mode in ("bidirectional", "cycle")
                    and self.backward_transport is None):
                raise ValueError("Bidirectional transport checkpoint is missing D")
            if self.bicyc_mode == BIALIGN_MODE and self.reverse_projector is None:
                raise ValueError("BiAlign checkpoint is missing reverse projector D_t")

    def _after_load_checkpoint(self, checkpoint):
        """Restore learner-specific state after BaseLearner restores the network."""
        if self._cur_task >= 1:
            self.old_ae = AutoencoderSigmoid(
                input_dims=768,
                code_dims=self.args["ae_code_dims"],
                residual_mode=self.ae_residual_mode,
            )
            if "old_ae_state_dict" not in checkpoint:
                raise ValueError(
                    "Checkpoint is missing old_ae_state_dict required to resume task {}."
                    .format(self._cur_task + 1)
                )
            self.old_ae.load_state_dict(checkpoint["old_ae_state_dict"])
            self.old_ae.to(self._device)

        self._restore_transport_state(checkpoint)
        self._network_module_ptr = self._network
        self.old_network_module_ptr = self._old_network

    # ---------------- lifecycle ----------------

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network, "module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network

    def _write_progress(self, phase, epoch=None, total_epochs=None, **extra):
        path = self.args.get("progress_path")
        if not path:
            return
        payload = {
            "arm": self.args.get("arm", self.bicyc_mode),
            "mode": self.bicyc_mode,
            "task": int(self._cur_task),
            "phase": phase,
            "epoch": epoch,
            "total_epochs": total_epochs,
            "pid": os.getpid(),
            "time_unix": time.time(),
        }
        payload.update(extra)
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)

    def _initialize_transition_maps(self):
        self.forward_transport = None
        self.backward_transport = None
        self.reverse_projector = None
        self.transport_initial_hashes = {}
        if self._cur_task <= 0 or self.bicyc_mode == "official":
            return

        if self.bicyc_mode == BIALIGN_MODE:
            # [IMPLEMENTATION ADAPTATION] D_t is transition-specific. Reset it
            # to the exact identity for each old->new representation transition
            # while preserving global RNG state for matched-arm comparability.
            d_seed = self._bialign_seed(self._cur_task)
            self.reverse_projector = make_identity_reverse_projector(
                self.feature_dim,
                self.bialign_reverse_hidden_dim,
                d_seed,
                self._device,
            )
            self.transport_initial_hashes["D_t"] = {
                "seed": d_seed,
                "sha256": module_state_sha256(self.reverse_projector),
                "lifecycle": "identity_reset_each_incremental_transition",
            }
            return

        a_seed = self._transport_seed(self._cur_task, "A")
        self.forward_transport = make_deterministic_affine(
            self.feature_dim, a_seed, self._device
        )
        self.transport_initial_hashes["A"] = {
            "seed": a_seed,
            "sha256": module_state_sha256(self.forward_transport),
        }
        if self.bicyc_mode in ("bidirectional", "cycle"):
            d_seed = self._transport_seed(self._cur_task, "D")
            self.backward_transport = make_deterministic_affine(
                self.feature_dim, d_seed, self._device
            )
            self.transport_initial_hashes["D"] = {
                "seed": d_seed,
                "sha256": module_state_sha256(self.backward_transport),
            }

    def _uses_official_statistics_path(self):
        return self.bicyc_mode in ("official", BIALIGN_MODE)

    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for _, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device, non_blocking=True)
                label = label.to(self._device, non_blocking=True)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        return torch.cat(embedding_list, dim=0), torch.cat(label_list, dim=0)

    def _transport_eval_loader(self, data_manager):
        dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
        )

    @torch.no_grad()
    def _paired_transport_metrics(self, loader):
        if self.bicyc_mode == "official":
            return None
        modules = [self._network_module_ptr, self.old_network_module_ptr,
                   self.forward_transport, self.backward_transport]
        states = [m.training if m is not None else None for m in modules]
        for module in modules:
            if module is not None:
                module.eval()
        sums = {"a_mse": 0.0, "d_mse": 0.0,
                "cycle_new_mse": 0.0, "cycle_old_mse": 0.0}
        elements = 0
        samples = 0
        try:
            for _, inputs, _ in loader:
                inputs = inputs.to(self._device, non_blocking=True)
                z_new, z_old = same_input_feature_pair(
                    self._network_module_ptr.extract_vector,
                    self.old_network_module_ptr.extract_vector,
                    inputs,
                )
                terms = bicyc_loss_terms(
                    z_new, z_old, self.forward_transport,
                    self.backward_transport, self.bicyc_mode,
                )
                count = int(z_new.numel())
                elements += count
                samples += int(z_new.shape[0])
                sums["a_mse"] += float(terms["loss_a"]) * count
                if self.bicyc_mode in ("bidirectional", "cycle"):
                    sums["d_mse"] += float(terms["loss_d"]) * count
                if self.bicyc_mode == "cycle":
                    sums["cycle_new_mse"] += float(terms["cycle_new"]) * count
                    sums["cycle_old_mse"] += float(terms["cycle_old"]) * count
        finally:
            for module, state in zip(modules, states):
                if module is not None:
                    module.train(state)
        result = {
            key: (None if key.startswith("d_") and self.bicyc_mode == "forward"
                  else None if key.startswith("cycle_") and self.bicyc_mode != "cycle"
                  else value / elements)
            for key, value in sums.items()
        }
        result.update({
            "sample_count": samples,
            "same_input_tensor_same_iteration": True,
            "deterministic_test_transform": True,
        })
        return result

    def _fine_tune_forward_transport(self):
        if self.forward_transport is None:
            return []
        history = []
        started = time.time()
        # Stage II must not perturb the RNG subsequently used by CA/next task.
        with preserve_global_rng_state():
            optimizer = optim.SGD(
                self.forward_transport.parameters(),
                lr=self.transport_stage2_lr,
                momentum=0.9,
                weight_decay=self.transport_stage2_weight_decay,
            )
            self._network_module_ptr.eval()
            self.old_network_module_ptr.eval()
            if self.backward_transport is not None:
                self.backward_transport.eval()
            for epoch in range(self.transport_stage2_epochs):
                self.forward_transport.train()
                loss_sum = 0.0
                sample_count = 0
                for _, inputs, _ in self.train_loader:
                    inputs = inputs.to(self._device, non_blocking=True)
                    with torch.no_grad():
                        z_new, z_old = same_input_feature_pair(
                            self._network_module_ptr.extract_vector,
                            self.old_network_module_ptr.extract_vector,
                            inputs,
                        )
                    optimizer.zero_grad()
                    loss = F.mse_loss(
                        self.forward_transport(z_old.detach()), z_new.detach())
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite Stage-II A loss")
                    loss.backward()
                    optimizer.step()
                    loss_sum += float(loss.detach()) * int(inputs.shape[0])
                    sample_count += int(inputs.shape[0])
                epoch_record = {
                    "epoch": epoch + 1,
                    "loss_a": loss_sum / sample_count,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                history.append(epoch_record)
                logging.info("BICYC_STAGE2 %s", json.dumps(epoch_record, sort_keys=True))
                self._write_progress(
                    "A_FINE_TUNE", epoch + 1, self.transport_stage2_epochs,
                    loss_a=epoch_record["loss_a"],
                )
        self.forward_transport.eval()
        if history:
            history[-1]["total_runtime_seconds"] = time.time() - started
        return history

    def _analytic_transport_old_statistics(self):
        old_means = np.array(self._class_means[:self._known_classes], copy=True)
        old_covs = self._class_covs[:self._known_classes].detach().cpu().clone()
        if self.ca_sampling_mode == "sample_source_then_affine":
            # These are the exact stored task-(t-1) moments.  Keep them before
            # the analytic transport overwrites the learner's old statistics.
            self._ca_source_old_means = np.array(old_means, copy=True)
            self._ca_source_old_covs = old_covs.clone()
            self._ca_source_statistics_task = int(self._cur_task)
        means_new, covs_new = analytic_affine_gaussian_transport(
            old_means, old_covs, self.forward_transport
        )
        self._class_means[:self._known_classes] = means_new.numpy()
        self._class_covs[:self._known_classes] = covs_new.to(
            dtype=self._class_covs.dtype
        )
        symmetry_error = float(
            (covs_new - covs_new.transpose(-1, -2)).abs().max()
        )
        return {
            "rule": "mu_new=W mu_old+b; Sigma_new=W Sigma_old W^T",
            "old_class_count": int(self._known_classes),
            "pre_means_sha256": tensor_mapping_sha256({"means": old_means}),
            "pre_covariances_sha256": tensor_mapping_sha256({"covs": old_covs}),
            "post_means_sha256": tensor_mapping_sha256({"means": means_new}),
            "post_covariances_sha256": tensor_mapping_sha256({"covs": covs_new}),
            "mean_shift_l2_mean": float(torch.linalg.norm(
                means_new - torch.as_tensor(old_means), dim=1).mean()),
            "max_covariance_asymmetry": symmetry_error,
            "finite": bool(torch.isfinite(means_new).all() and torch.isfinite(covs_new).all()),
            "extra_shrinkage": False,
        }

    # ---------------- CA numerical rescue ----------------

    @staticmethod
    def _cpu_state_dict(module):
        return {
            key: value.detach().cpu()
            for key, value in module.state_dict().items()
        }

    @staticmethod
    def _sha256_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _loader_seed_metadata(loader):
        generator = getattr(loader, "generator", None)
        generator_state = None
        if generator is not None:
            generator_state = generator.get_state().cpu()
        return {
            "dataset_type": type(loader.dataset).__name__,
            "dataset_length": len(loader.dataset),
            "batch_size": loader.batch_size,
            "num_workers": loader.num_workers,
            "pin_memory": bool(loader.pin_memory),
            "persistent_workers": bool(getattr(loader, "persistent_workers", False)),
            "sampler_type": type(loader.sampler).__name__,
            "batch_sampler_type": type(loader.batch_sampler).__name__,
            "worker_init_fn": repr(loader.worker_init_fn),
            "generator_state": generator_state,
            "note": (
                "The historical task1 checkpoint did not store worker/process RNG. "
                "This metadata describes the controlled task2 rerun only."
            ),
        }

    def _write_ca_rescue_report(self):
        path = self.args.get("ca_rescue_report_path")
        if not path or self.ca_rescue_diagnostics is None:
            return
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(self.ca_rescue_diagnostics, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)

    def _save_exact_pre_ca_snapshot(self, pre_ca):
        path = self.args.get("ca_rescue_pre_ca_checkpoint")
        if not path:
            raise RuntimeError("ca_rescue_pre_ca_checkpoint is required")
        if self._cur_task != 2 or self._known_classes != 20 or self._total_classes != 30:
            raise RuntimeError("B1 rescue snapshot requires task2 cur=2 known=20 total=30")
        if self.bicyc_mode != "forward" or self.backward_transport is not None:
            raise RuntimeError("B1 rescue requires forward-only A with no D/cycle")
        if self._ca_source_statistics_task != self._cur_task:
            raise RuntimeError("Source task-(t-1) statistics were not captured")
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        if not hasattr(network.fc, "old_state_dict"):
            raise RuntimeError("Classifier backup is absent at CA entry")
        rng_state = capture_rng_state()
        network_state = self._cpu_state_dict(network)
        classifier_state = self._cpu_state_dict(network.fc)
        classifier_backup = {
            key: value.detach().cpu()
            for key, value in network.fc.old_state_dict.items()
        }
        a_state = self._cpu_state_dict(self.forward_transport)
        old_p_state = self._cpu_state_dict(self.old_ae)
        class_means = torch.as_tensor(self._class_means).detach().cpu()
        class_covariances = self._class_covs.detach().cpu()
        source_old_means = torch.as_tensor(
            self._ca_source_old_means, dtype=torch.float64).cpu()
        source_old_covariances = self._ca_source_old_covs.detach().cpu()
        snapshot = {
            "format_version": 1,
            "capture_stage": "task2_pre_ca_after_eval_before_ca",
            "labels": ["[CONTROLLED REPRODUCTION]", "[NUMERICAL IMPLEMENTATION FIX]"],
            "ca_sampling_mode": self.ca_sampling_mode,
            "cur_task": int(self._cur_task),
            "known_classes": int(self._known_classes),
            "total_classes": int(self._total_classes),
            "task_sizes": list(self.task_sizes),
            "class_order": list(self.class_order),
            "network_state_dict": network_state,
            "classifier_state_dict": classifier_state,
            "classifier_backup_state_dict": classifier_backup,
            "forward_transport_state_dict": a_state,
            "old_p_state_dict": old_p_state,
            "class_means": class_means,
            "class_covariances": class_covariances,
            "source_old_means": source_old_means,
            "source_old_covariances": source_old_covariances,
            "rng_state": rng_state,
            "loader_seed_metadata": {
                "train": self._loader_seed_metadata(self.train_loader),
                "test": self._loader_seed_metadata(self.test_loader),
            },
            "seed": int(self.args["seed"]),
            "batch_size": int(self.batch_size),
            "pre_ca": copy.deepcopy(pre_ca),
            "environment": environment_record(),
            "state_hashes": {
                "network": tensor_mapping_sha256(network_state),
                "classifier": tensor_mapping_sha256(classifier_state),
                "classifier_backup": tensor_mapping_sha256(classifier_backup),
                "A": tensor_mapping_sha256(a_state),
                "old_P": tensor_mapping_sha256(old_p_state),
                "class_means": tensor_mapping_sha256({"means": class_means}),
                "class_covariances": tensor_mapping_sha256({
                    "covariances": class_covariances}),
                "source_means": tensor_mapping_sha256({
                    "means": source_old_means}),
                "source_covariances": tensor_mapping_sha256({
                    "covariances": source_old_covariances}),
            },
        }
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = path + ".tmp"
        torch.save(snapshot, temporary)
        os.replace(temporary, path)
        return {
            "path": path,
            "sha256": self._sha256_file(path),
            "state_hashes": copy.deepcopy(snapshot["state_hashes"]),
            "capture_stage": snapshot["capture_stage"],
        }

    def _restore_exact_pre_ca_snapshot(
            self, path, expected_file_sha256, expected_state_hashes):
        observed_file_sha256 = self._sha256_file(path)
        if observed_file_sha256 != expected_file_sha256:
            raise RuntimeError(
                "Pre-CA snapshot file SHA256 changed before restore")
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if snapshot.get("capture_stage") != "task2_pre_ca_after_eval_before_ca":
            raise RuntimeError("Not an exact task2 pre-CA rescue snapshot")
        expected_metadata = {
            "cur_task": int(self._cur_task),
            "known_classes": int(self._known_classes),
            "total_classes": int(self._total_classes),
            "task_sizes": list(self.task_sizes),
            "class_order": list(self.class_order),
            "seed": int(self.args["seed"]),
            "batch_size": int(self.batch_size),
            "ca_sampling_mode": self.ca_sampling_mode,
        }
        for key, expected in expected_metadata.items():
            if snapshot.get(key) != expected:
                raise RuntimeError(
                    "Pre-CA snapshot metadata mismatch for {}".format(key))
        tensors = {
            "class_means": snapshot["class_means"],
            "class_covariances": snapshot["class_covariances"],
            "source_old_means": snapshot["source_old_means"],
            "source_old_covariances": snapshot["source_old_covariances"],
        }
        expected_shapes = {
            "class_means": (30, self.feature_dim),
            "class_covariances": (30, self.feature_dim, self.feature_dim),
            "source_old_means": (20, self.feature_dim),
            "source_old_covariances": (20, self.feature_dim, self.feature_dim),
        }
        for key, tensor in tensors.items():
            if tuple(tensor.shape) != expected_shapes[key]:
                raise RuntimeError(
                    "Pre-CA snapshot shape mismatch for {}".format(key))
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(
                    "Pre-CA snapshot contains NaN/Inf in {}".format(key))
        serialized_hashes = {
            "network": tensor_mapping_sha256(snapshot["network_state_dict"]),
            "classifier": tensor_mapping_sha256(snapshot["classifier_state_dict"]),
            "classifier_backup": tensor_mapping_sha256(
                snapshot["classifier_backup_state_dict"]),
            "A": tensor_mapping_sha256(snapshot["forward_transport_state_dict"]),
            "old_P": tensor_mapping_sha256(snapshot["old_p_state_dict"]),
            "class_means": tensor_mapping_sha256({
                "means": snapshot["class_means"]}),
            "class_covariances": tensor_mapping_sha256({
                "covariances": snapshot["class_covariances"]}),
            "source_means": tensor_mapping_sha256({
                "means": snapshot["source_old_means"]}),
            "source_covariances": tensor_mapping_sha256({
                "covariances": snapshot["source_old_covariances"]}),
        }
        if serialized_hashes != expected_state_hashes:
            raise RuntimeError(
                "Pre-CA snapshot content does not match caller-held hashes")
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        result = network.load_state_dict(snapshot["network_state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("Strict pre-CA network restore failed: {}".format(result))
        network.fc.load_state_dict(snapshot["classifier_state_dict"], strict=True)
        classifier_devices = {
            key: value.device for key, value in network.fc.state_dict().items()}
        network.fc.old_state_dict = {
            key: value.detach().to(classifier_devices[key]).clone()
            for key, value in snapshot["classifier_backup_state_dict"].items()}
        self.forward_transport.load_state_dict(
            snapshot["forward_transport_state_dict"], strict=True)
        self.old_ae.load_state_dict(snapshot["old_p_state_dict"], strict=True)
        self._class_means = snapshot["class_means"].cpu().numpy()
        self._class_covs = snapshot["class_covariances"].cpu()
        self._ca_source_old_means = snapshot["source_old_means"].cpu().numpy()
        self._ca_source_old_covs = snapshot["source_old_covariances"].cpu()
        self._ca_source_statistics_task = int(snapshot["cur_task"])
        network.to(self._device)
        self.forward_transport.to(self._device)
        self.old_ae.to(self._device)
        restored_hashes = {
            "network": tensor_mapping_sha256(self._cpu_state_dict(network)),
            "classifier": tensor_mapping_sha256(self._cpu_state_dict(network.fc)),
            "classifier_backup": tensor_mapping_sha256(network.fc.old_state_dict),
            "A": module_state_sha256(self.forward_transport),
            "old_P": module_state_sha256(self.old_ae),
            "class_means": tensor_mapping_sha256({
                "means": torch.as_tensor(self._class_means)}),
            "class_covariances": tensor_mapping_sha256({
                "covariances": self._class_covs}),
            "source_means": tensor_mapping_sha256({
                "means": torch.as_tensor(self._ca_source_old_means)}),
            "source_covariances": tensor_mapping_sha256({
                "covariances": self._ca_source_old_covs}),
        }
        for key, value in restored_hashes.items():
            if value != expected_state_hashes[key]:
                raise RuntimeError("Strict pre-CA restore hash mismatch: {}".format(key))
        restore_rng_state(snapshot["rng_state"])
        return {
            "status": "PASS",
            "strict": True,
            "file_sha256": observed_file_sha256,
            "restored_hashes": restored_hashes,
        }

    @staticmethod
    def _numeric_summary(values):
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(array.min()),
            "median": float(np.median(array)),
            "mean": float(array.mean()),
            "max": float(array.max()),
        }

    @torch.no_grad()
    def _build_rescue_ca_epoch(self, task_size, num_sampled_pcls=256,
                               collect_distribution_diagnostics=False):
        if self.forward_transport is None:
            raise RuntimeError("sample-source-then-affine requires A")
        if self._ca_source_old_means is None or self._ca_source_old_covs is None:
            raise RuntimeError("Source statistics are unavailable")
        previous_mode = self.forward_transport.training
        self.forward_transport.eval()
        sampled_data = []
        sampled_label = []
        per_class = []
        equivalence = []
        try:
            for class_id in range(self._total_classes):
                task_id = class_id // task_size
                alpha = (task_id + 1) / (self._cur_task + 1) * 0.1 + 0.9
                target_mean = torch.tensor(
                    self._class_means[class_id], dtype=torch.float64,
                    device=self._device) * alpha
                if class_id < self._known_classes:
                    result = sample_source_then_affine(
                        self._ca_source_old_means[class_id],
                        self._ca_source_old_covs[class_id],
                        target_mean,
                        self.forward_transport,
                        num_sampled_pcls,
                    )
                    samples = result["ca_samples"]
                    source_health = {
                        "scale_tril_finite": result["source_scale_tril_finite"],
                        "scale_tril_diagonal_min": result[
                            "source_scale_tril_diagonal_min"],
                        "scale_tril_diagonal_max": result[
                            "source_scale_tril_diagonal_max"],
                    }
                    if collect_distribution_diagnostics:
                        diagnostic = affine_distribution_diagnostic(
                            result["source_samples"], result["raw_transformed"],
                            self._ca_source_old_means[class_id],
                            self._ca_source_old_covs[class_id],
                            self.forward_transport,
                        )
                        weight = self.forward_transport.weight.detach().cpu().double()
                        bias = self.forward_transport.bias.detach().cpu().double()
                        source_mean = torch.as_tensor(
                            self._ca_source_old_means[class_id], dtype=torch.float64)
                        source_cov = self._ca_source_old_covs[class_id].double()
                        analytic_mean = source_mean @ weight.T + bias
                        analytic_cov = weight @ source_cov @ weight.T
                        stored_mean = torch.as_tensor(
                            self._class_means[class_id], dtype=torch.float64)
                        stored_cov = self._class_covs[class_id].double()
                        diagnostic.update({
                            "class_id": int(class_id),
                            "stored_analytic_mean_relative_l2_error": float(
                                torch.linalg.vector_norm(stored_mean - analytic_mean)
                                / max(float(torch.linalg.vector_norm(analytic_mean)), 1e-12)),
                            "stored_analytic_covariance_relative_frobenius_error": float(
                                torch.linalg.matrix_norm(stored_cov - analytic_cov)
                                / max(float(torch.linalg.matrix_norm(analytic_cov)), 1e-12)),
                        })
                        equivalence.append(diagnostic)
                else:
                    covariance = self._class_covs[class_id].to(
                        self._device).float()
                    distribution = MultivariateNormal(
                        target_mean.float(), covariance)
                    if not bool(torch.isfinite(distribution.scale_tril).all()):
                        raise RuntimeError(
                            "Current-class Gaussian factor is non-finite: {}".format(
                                class_id))
                    samples = distribution.sample((num_sampled_pcls,))
                    source_health = None
                summary = feature_summary(samples)
                if summary["sample_finite_fraction"] != 1.0:
                    raise RuntimeError(
                        "Rescue samples are non-finite for class {}".format(
                            class_id))
                per_class.append({
                    "class_id": int(class_id),
                    "split": "old" if class_id < self._known_classes else "new",
                    "mean_multiplier": float(alpha),
                    "features": summary,
                    "source_factorization": source_health,
                })
                sampled_data.append(samples)
                sampled_label.extend([class_id] * num_sampled_pcls)
            features = torch.cat(sampled_data, dim=0).float().to(self._device)
            targets = torch.tensor(sampled_label).long().to(self._device)
            permutation = torch.randperm(features.size(0))
            features = features[permutation]
            targets = targets[permutation]
        finally:
            self.forward_transport.train(previous_mode)
        aggregate = feature_summary(features)
        equivalence_summary = None
        if equivalence:
            equivalence_summary = {
                key: self._numeric_summary([item[key] for item in equivalence])
                for key in (
                    "empirical_mean_relative_l2_error",
                    "empirical_covariance_relative_frobenius_error",
                    "stored_analytic_mean_relative_l2_error",
                    "stored_analytic_covariance_relative_frobenius_error",
                    "sample_transform_max_abs_error_float64_recompute",
                )
            }
        return features, targets, {
            "aggregate": aggregate,
            "per_class": per_class,
            "affine_distribution_equivalence": equivalence,
            "affine_distribution_equivalence_summary": equivalence_summary,
            "mean_policy": (
                "raw A(u) is verified diagnostically; CA uses "
                "alpha*(W mu+b)+W*(u-mu) to preserve the historical "
                "task-age mean scaling exactly"
            ),
        }

    def _prepare_ca_rescue(self, pre_ca):
        if self.ca_sampling_mode != "sample_source_then_affine":
            return
        expected = float(self.args.get("ca_rescue_expected_pre_ca_top1", 96.27))
        tolerance = float(self.args.get("ca_rescue_pre_ca_tolerance_pp", 0.20))
        observed = float(pre_ca["metrics"]["top1"])
        difference = abs(observed - expected)
        self.ca_rescue_diagnostics = {
            "status": "PRE_CA_CAPTURED",
            "labels": ["[CONTROLLED REPRODUCTION]", "[NUMERICAL IMPLEMENTATION FIX]"],
            "scientific_claim_scope": (
                "Task2-only controlled rerun from exact task1 boundary; not an "
                "exact replay because historical task2 RNG/optimizer state was not saved."
            ),
            "sampling_fix": (
                "factorize stored source covariance, sample source Gaussian, "
                "apply affine residual; never factorize W Sigma W^T for old classes"
            ),
            "pre_ca": copy.deepcopy(pre_ca),
            "pre_ca_gate": {
                "expected_top1": expected,
                "observed_top1": observed,
                "absolute_difference_pp": difference,
                "tolerance_pp": tolerance,
                "status": "PASS" if difference <= tolerance else "FAIL",
            },
            "environment": environment_record(),
            "environment_label": "[ENVIRONMENT DEVIATION]",
            "controlled_resume": {
                "source_checkpoint": os.path.abspath(self.args["resume_path"]),
                "source_checkpoint_sha256": self.args.get(
                    "ca_rescue_source_task1_sha256"),
                "historical_task2_rng_available": False,
                "historical_task2_optimizer_scheduler_available": False,
            },
            "training_scope": {
                "resumed_completed_task": 1,
                "rerun_tasks": [2],
                "task0_rerun": False,
                "task1_rerun": False,
                "B0_B2_B3_run": False,
            },
            "method_changes": {
                "covariance_clipping": False,
                "jitter_tuning": False,
                "eigenvalue_repair": False,
                "gradient_clipping": False,
                "new_hyperparameters": False,
                "historical_mean_scaling_preserved": True,
            },
            "epochs": [],
        }
        snapshot = self._save_exact_pre_ca_snapshot(pre_ca)
        self.ca_rescue_diagnostics["pre_ca_checkpoint"] = snapshot
        self._write_ca_rescue_report()
        if difference > tolerance:
            self.ca_rescue_diagnostics["status"] = "FAIL_PRE_CA_REPRODUCTION_MISMATCH"
            self._write_ca_rescue_report()
            raise RuntimeError(
                "B1 pre-CA reproduction mismatch: observed {:.2f}, expected {:.2f}, "
                "difference {:.2f} pp > {:.2f} pp".format(
                    observed, expected, difference, tolerance))
        restore = self._restore_exact_pre_ca_snapshot(
            snapshot["path"], snapshot["sha256"], snapshot["state_hashes"])
        self.ca_rescue_diagnostics["strict_pre_ca_reload"] = restore
        with preserve_global_rng_state():
            _, _, preflight = self._build_rescue_ca_epoch(
                self.task_sizes[-1], collect_distribution_diagnostics=True)
        self.ca_rescue_diagnostics["pre_optimization_synthetic_check"] = preflight
        finite = preflight["aggregate"]["sample_finite_fraction"] == 1.0
        source_healthy = all(
            row["source_factorization"] is None
            or row["source_factorization"]["scale_tril_finite"]
            for row in preflight["per_class"])
        equivalence = preflight["affine_distribution_equivalence_summary"]
        equivalence_tolerances = {
            "stored_analytic_mean_relative_l2_error_max": 1e-10,
            "stored_analytic_covariance_relative_frobenius_error_max": 1e-6,
            "sample_transform_max_abs_error_float64_recompute_max": 1e-4,
        }
        equivalence_exact = (
            equivalence is not None
            and equivalence["stored_analytic_mean_relative_l2_error"]["max"]
            <= equivalence_tolerances[
                "stored_analytic_mean_relative_l2_error_max"]
            and equivalence[
                "stored_analytic_covariance_relative_frobenius_error"]["max"]
            <= equivalence_tolerances[
                "stored_analytic_covariance_relative_frobenius_error_max"]
            and equivalence[
                "sample_transform_max_abs_error_float64_recompute"]["max"]
            <= equivalence_tolerances[
                "sample_transform_max_abs_error_float64_recompute_max"]
        )
        self.ca_rescue_diagnostics["pre_optimization_gate"] = {
            "all_samples_finite": finite,
            "all_source_factors_finite": source_healthy,
            "stored_transport_and_affine_equivalence": equivalence_exact,
            "equivalence_tolerances": equivalence_tolerances,
            "status": (
                "PASS" if finite and source_healthy and equivalence_exact
                else "FAIL"),
        }
        if not finite or not source_healthy or not equivalence_exact:
            self.ca_rescue_diagnostics["status"] = "FAIL_SYNTHETIC_PREFLIGHT"
            self._write_ca_rescue_report()
            raise RuntimeError("B1 rescue synthetic preflight failed")
        self.ca_rescue_diagnostics["status"] = "PRE_CA_GATE_PASS"
        self._write_ca_rescue_report()

    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        if self.ca_sampling_mode != "sample_source_then_affine":
            return super()._stage2_compact_classifier(task_size, ca_epochs)
        if int(ca_epochs) != int(self.args["ca_epochs"]):
            raise RuntimeError("Rescue CA epoch count differs from locked config")
        if int(task_size) != int(self.task_sizes[-1]):
            raise RuntimeError("Rescue CA task size differs from task metadata")
        if bool(self.args.get("ca_covariance_pd_fallback", False)):
            raise RuntimeError("Rescue CA forbids covariance repair/fallback")
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        for parameter in network.fc.parameters():
            parameter.requires_grad = True
        parameters = [parameter for parameter in network.fc.parameters()
                      if parameter.requires_grad]
        optimizer = optim.SGD(
            [{"params": parameters, "lr": self.init_lr,
              "weight_decay": self.weight_decay}],
            lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=ca_epochs)
        self._network.to(self._device)
        self._network.eval()
        self.ca_covariance_stabilization = {
            "enabled": False,
            "mode": "sample_source_then_affine",
            "fallback_count": 0,
            "classes": [],
        }
        num_sampled_pcls = 256
        self.ca_rescue_diagnostics["ca_protocol"] = {
            "epochs": int(ca_epochs),
            "classes": int(self._total_classes),
            "samples_per_class_per_epoch": num_sampled_pcls,
            "batches_per_epoch": int(self._total_classes),
            "batch_size": num_sampled_pcls,
            "optimizer": "SGD",
            "learning_rate": float(self.init_lr),
            "momentum": 0.9,
            "weight_decay": float(self.weight_decay),
            "scheduler": "CosineAnnealingLR",
            "scheduler_T_max": int(ca_epochs),
            "covariance_fallback_enabled": False,
            "gradient_clipping": False,
        }
        for epoch in range(ca_epochs):
            features, targets, sampling = self._build_rescue_ca_epoch(
                task_size, num_sampled_pcls,
                collect_distribution_diagnostics=False)
            losses = []
            gradient_fraction_min = 1.0
            classifier_fraction_min = parameter_finite_fraction(network.fc)
            for batch in range(self._total_classes):
                inputs = features[
                    batch * num_sampled_pcls:(batch + 1) * num_sampled_pcls]
                batch_targets = targets[
                    batch * num_sampled_pcls:(batch + 1) * num_sampled_pcls]
                outputs = self._network.ca_forward(inputs)
                logits = self.args["scale"] * outputs["logits"]
                if self.logit_norm is not None:
                    per_task_norm = []
                    previous = 0
                    current = 0
                    for task_index in range(self._cur_task + 1):
                        current += self.task_sizes[task_index]
                        norm = torch.norm(
                            logits[:, previous:current], p=2, dim=-1,
                            keepdim=True) + 1e-7
                        per_task_norm.append(norm)
                        previous = current
                    norms = torch.cat(per_task_norm, dim=-1).mean(
                        dim=-1, keepdim=True)
                    decoupled = torch.div(
                        logits[:, :self._total_classes], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled, batch_targets)
                else:
                    loss = F.cross_entropy(
                        logits[:, :self._total_classes], batch_targets)
                if not bool(torch.isfinite(loss)):
                    self.ca_rescue_diagnostics["status"] = "FAIL_NONFINITE_CA_LOSS"
                    self.ca_rescue_diagnostics["first_ca_failure"] = {
                        "epoch": epoch + 1, "batch": batch,
                        "stage": "loss_before_backward"}
                    self._write_ca_rescue_report()
                    raise RuntimeError("Non-finite rescue CA loss")
                optimizer.zero_grad()
                loss.backward()
                gradient_fraction = parameter_finite_fraction(
                    network.fc, gradients=True)
                gradient_fraction_min = min(
                    gradient_fraction_min, gradient_fraction)
                if gradient_fraction != 1.0:
                    self.ca_rescue_diagnostics["status"] = "FAIL_NONFINITE_CA_GRADIENT"
                    self.ca_rescue_diagnostics["first_ca_failure"] = {
                        "epoch": epoch + 1, "batch": batch,
                        "stage": "gradient_after_backward"}
                    self._write_ca_rescue_report()
                    raise RuntimeError("Non-finite rescue CA gradient")
                optimizer.step()
                classifier_fraction = parameter_finite_fraction(network.fc)
                classifier_fraction_min = min(
                    classifier_fraction_min, classifier_fraction)
                if classifier_fraction != 1.0:
                    self.ca_rescue_diagnostics["status"] = "FAIL_NONFINITE_CLASSIFIER"
                    self.ca_rescue_diagnostics["first_ca_failure"] = {
                        "epoch": epoch + 1, "batch": batch,
                        "stage": "classifier_after_optimizer"}
                    self._write_ca_rescue_report()
                    raise RuntimeError("Non-finite rescue CA classifier")
                losses.append(float(loss.detach()))
            scheduler.step()
            epoch_record = {
                "epoch": epoch + 1,
                "batch_count": int(self._total_classes),
                "loss_mean": float(np.mean(losses)),
                "loss_min": float(np.min(losses)),
                "loss_max": float(np.max(losses)),
                "loss_finite": True,
                "synthetic_sample_finite_fraction": sampling[
                    "aggregate"]["sample_finite_fraction"],
                "gradient_finite_fraction_min": gradient_fraction_min,
                "classifier_finite_fraction_min": classifier_fraction_min,
                "classifier_finite_fraction_after_epoch": (
                    parameter_finite_fraction(network.fc)),
                "learning_rate_after_epoch": float(
                    optimizer.param_groups[0]["lr"]),
            }
            self.ca_rescue_diagnostics["epochs"].append(epoch_record)
            self.ca_rescue_diagnostics["status"] = "CA_RUNNING"
            self._write_ca_rescue_report()
            self._write_progress(
                "CA_RESCUE", epoch + 1, ca_epochs,
                ca_loss=epoch_record["loss_mean"])
            logging.info("CA_RESCUE %s", json.dumps(epoch_record, sort_keys=True))
        self.ca_rescue_diagnostics["status"] = "CA_COMPLETE_PENDING_EVALUATION"
        self._write_ca_rescue_report()

    def incremental_train(self, data_manager):
        task_started = time.time()
        self._cur_task += 1
        if self._cur_task == 1:
            self.old_ae = AutoencoderSigmoid(
                input_dims=768,
                code_dims=self.args["ae_code_dims"],
                residual_mode=self.ae_residual_mode,
            ).to(self._device)
        self._initialize_transition_maps()

        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + task_size
        self._network.update_fc(task_size)
        self._network_module_ptr = self._network
        logging.info("Learning on %s-%s", self._known_classes, self._total_classes)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        print("The number of training dataset:", len(self.train_dataset))
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers)

        if len(self._multiple_gpus) > 1:
            print("Multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        if self.bicyc_mode == BIALIGN_MODE:
            implementation_label = (
                "[IMPLEMENTATION ADAPTATION] RSIAT BiAlign: L_align replaced "
                "by L_fwd + L_back; official SSCA/CA retained")
        elif self.bicyc_mode in TRACK_B_TRANSPORT_MODES:
            implementation_label = (
                "[IMPLEMENTATION ADAPTATION] RSIAT + BiCyc-style "
                "bidirectional/cycle transport")
        else:
            implementation_label = "[CONTROL] official RSIAT"
        record = {
            "task": int(self._cur_task),
            "mode": self.bicyc_mode,
            "implementation_label": implementation_label,
            "initial_hashes": copy.deepcopy(self.transport_initial_hashes),
            "hyperparameters": {
                "lambda_bi": self.lambda_bi,
                "lambda_cycle": self.lambda_cycle,
                "pair_preserve": False,
                "bicyc_anti_collapse": False,
                "stage1_lr": self.transport_stage1_lr,
                "stage1_weight_decay": self.transport_stage1_weight_decay,
                "stage2_epochs": self.transport_stage2_epochs,
                "stage2_lr": self.transport_stage2_lr,
                "stage2_weight_decay": self.transport_stage2_weight_decay,
                "stage2_momentum": 0.9,
                "beta": self.args["beta"],
                "gamma": self.args["gamma"],
                "bialign_reverse_hidden_dim": self.bialign_reverse_hidden_dim,
                "bialign_reverse_lr": self.args["ae_init_lr"],
                "bialign_reverse_weight_decay": self.args["ae_weight_decay"],
                "bialign_reverse_lifecycle": (
                    "identity_reset_each_incremental_transition"),
                "cycle_loss_enabled": False if self.bicyc_mode == BIALIGN_MODE else None,
            },
            "paired_before_stage1": None,
            "paired_after_stage1": None,
            "paired_after_stage2": None,
            "stage1_epochs": [],
            "stage2_epochs": [],
            "statistics_transport": None,
        }

        train_embeddings_old = None
        if self._cur_task > 0:
            self._network.to(self._device)
            # Retain official extraction pass in every arm for matched RNG behavior.
            train_embeddings_old, _ = self.extract_features(
                self.train_loader, self._network, None)
            if self.bicyc_mode in TRACK_B_TRANSPORT_MODES:
                eval_loader = self._transport_eval_loader(data_manager)
                with preserve_global_rng_state():
                    record["paired_before_stage1"] = self._paired_transport_metrics(
                        eval_loader)

        self._write_progress("STAGE1", 0, self.args["inc_epochs"])
        self._train(self.train_loader, self.test_loader)
        record["stage1_epochs"] = copy.deepcopy(self._latest_stage1_history)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            self._network_module_ptr = self._network

        if self._cur_task > 0:
            # Retain official second extraction pass in every arm. For transport
            # arms these arrays are deliberately not used as offline pairs.
            train_embeddings_new, _ = self.extract_features(
                self.train_loader, self._network, None)
            if self._uses_official_statistics_path():
                # BiAlign deliberately retains the official RSIAT SSCA path.
                old_class_mean = self._class_means[:self._known_classes]
                gap = self.displacement(
                    train_embeddings_old, train_embeddings_new,
                    old_class_mean, 4.0)
                record["official_ssca"] = {
                    "enabled": bool(self.args["ssca"]),
                    "sigma": 4.0,
                    "gap_l2_mean": float(np.linalg.norm(gap, axis=1).mean()),
                }
                if self.args["ssca"] is True:
                    old_class_mean += gap
                    self._class_means[:self._known_classes] = old_class_mean
                if self.bicyc_mode == BIALIGN_MODE:
                    record["D_t_final"] = {
                        "state_sha256": module_state_sha256(self.reverse_projector),
                        "parameter_count": int(sum(
                            parameter.numel()
                            for parameter in self.reverse_projector.parameters())),
                        "signed_unconstrained_residual": True,
                        "identity_initialized_at_transition_start": True,
                    }
            else:
                eval_loader = self._transport_eval_loader(data_manager)
                with preserve_global_rng_state():
                    record["paired_after_stage1"] = self._paired_transport_metrics(
                        eval_loader)
                record["stage2_epochs"] = self._fine_tune_forward_transport()
                with preserve_global_rng_state():
                    record["paired_after_stage2"] = self._paired_transport_metrics(
                        eval_loader)
                record["statistics_transport"] = (
                    self._analytic_transport_old_statistics())
                record["A_final"] = affine_geometry(self.forward_transport)
                if self.backward_transport is not None:
                    record["D_final"] = affine_geometry(self.backward_transport)

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        # Pre-CA evaluation is an added diagnostic. Use an isolated zero-worker
        # loader so the official persistent test-loader lifecycle remains untouched;
        # restore parent RNG so CA and the next task see the exact baseline state.
        pre_ca_loader = DataLoader(
            self.test_loader.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
        )
        official_test_loader = self.test_loader
        with preserve_global_rng_state():
            self.test_loader = pre_ca_loader
            try:
                pre_ca = self.eval_task_detailed()
            finally:
                self.test_loader = official_test_loader
        self.pre_ca_metrics[str(self._cur_task)] = copy.deepcopy(pre_ca)
        record["pre_ca"] = copy.deepcopy(pre_ca)

        if self.ca_sampling_mode == "sample_source_then_affine":
            self._prepare_ca_rescue(pre_ca)
            record["ca_rescue"] = copy.deepcopy(self.ca_rescue_diagnostics)

        if self._cur_task > 0 and self.args["ca_epochs"] > 0 and self.args["ca"] is True:
            self._write_progress("CA", 0, self.args["ca_epochs"])
            self._stage2_compact_classifier(task_size, self.args["ca_epochs"])
            record["ca_covariance_stabilization"] = copy.deepcopy(
                self.ca_covariance_stabilization
            )
            if self.ca_sampling_mode == "sample_source_then_affine":
                record["ca_rescue"] = copy.deepcopy(
                    self.ca_rescue_diagnostics)
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
                self._network_module_ptr = self._network

        record["runtime_seconds"] = time.time() - task_started
        self.transport_history.append(copy.deepcopy(record))
        self._current_transport_record = record
        self._write_progress("TASK_MODEL_COMPLETE", self._cur_task, self._cur_task)

    # ---------------- Stage I ----------------

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {"params": self._network.convnet.blocks[-1].parameters(),
                 "lr": 0.01, "weight_decay": self.args["weight_decay"]},
                {"params": self._network.convnet.blocks[:-1].parameters(),
                 "lr": 0.01, "weight_decay": self.args["weight_decay"]},
                {"params": self._network.fc.parameters(),
                 "lr": 0.01, "weight_decay": self.args["weight_decay"]},
            ]
            if self.args["optimizer"] == "sgd":
                optimizer = optim.SGD(
                    param_groups, momentum=0.9, lr=self.init_lr,
                    weight_decay=self.weight_decay)
            elif self.args["optimizer"] == "adam":
                optimizer = optim.AdamW(
                    self._network.parameters(), lr=self.init_lr,
                    weight_decay=self.weight_decay)
            else:
                raise ValueError("Unknown optimizer")
        else:
            self.tuned_epochs = self.args["inc_epochs"]
            param_groups = [
                {"params": self._network.convnet.parameters(),
                 "lr": self.init_lr, "weight_decay": self.weight_decay},
                {"params": self._network.fc.parameters(),
                 "lr": self.init_lr, "weight_decay": self.weight_decay},
                {"params": self.old_ae.parameters(),
                 "lr": self.args["ae_init_lr"],
                 "weight_decay": self.args["ae_weight_decay"]},
            ]
            if self.forward_transport is not None:
                param_groups.append({
                    "params": self.forward_transport.parameters(),
                    "lr": self.transport_stage1_lr,
                    "weight_decay": self.transport_stage1_weight_decay,
                    "name": "A",
                })
            if self.backward_transport is not None:
                param_groups.append({
                    "params": self.backward_transport.parameters(),
                    "lr": self.transport_stage1_lr,
                    "weight_decay": self.transport_stage1_weight_decay,
                    "name": "D",
                })
            if self.reverse_projector is not None:
                # [IMPLEMENTATION ADAPTATION] D_t shares P_t's optimizer scale;
                # this introduces no additional tuned learning rate.
                param_groups.append({
                    "params": self.reverse_projector.parameters(),
                    "lr": self.args["ae_init_lr"],
                    "weight_decay": self.args["ae_weight_decay"],
                    "name": "D_t",
                })
            if self.args["optimizer"] == "sgd":
                optimizer = optim.SGD(param_groups, momentum=0.9)
            elif self.args["optimizer"] == "adam":
                raise ValueError(
                    "BiCyc Track B requires SGD so separate P/A/D groups step correctly")
            else:
                raise ValueError("Unknown optimizer")

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
        log_count_parameter(param_groups)
        self._init_train(
            train_loader, test_loader, optimizer, scheduler,
            self.args["warmup_epoch"])

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, warmup_epoch):
        prog_bar = tqdm(range(self.tuned_epochs))
        eval_interval = self.args.get("eval_interval", 0)
        history = []
        for epoch in prog_bar:
            self._network.train()
            if self.old_ae is not None:
                self.old_ae.train()
            if self.forward_transport is not None:
                self.forward_transport.train()
            if self.backward_transport is not None:
                self.backward_transport.train()
            if self.reverse_projector is not None:
                self.reverse_projector.train()

            sums = {key: 0.0 for key in (
                "total", "classification", "rsiat", "align", "orth",
                "loss_fwd", "loss_back", "bialign",
                "loss_a", "loss_d", "cycle_new", "cycle_old", "transport")}
            correct, total = 0, 0
            gradient_norms = None
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                logits, loss_c, loss_rsiat, details = self._compute_rt_loss(
                    inputs, targets, epoch, warmup_epoch)
                loss = loss_c + loss_rsiat
                if self.bicyc_mode in TRACK_B_TRANSPORT_MODES:
                    loss = loss + details["transport"]
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite Stage-I loss at task {} epoch {}".format(
                            self._cur_task, epoch + 1))
                optimizer.zero_grad()
                loss.backward()
                if (self.args.get("log_gradient_norms", False)
                        and gradient_norms is None):
                    network = (
                        self._network.module
                        if isinstance(self._network, nn.DataParallel)
                        else self._network)
                    gradient_norms = {
                        "current_adapter": module_gradient_record(network.convnet),
                        "P_t": module_gradient_record(self.old_ae),
                        "D_t": module_gradient_record(self.reverse_projector),
                        "old_model": module_gradient_record(
                            self.old_network_module_ptr),
                    }
                    if not all(item["all_finite"]
                               for item in gradient_norms.values()):
                        raise RuntimeError("Non-finite Stage-I gradient norm record")
                    if gradient_norms["old_model"]["has_gradient"]:
                        raise RuntimeError("Frozen old model received a gradient")
                    if self.bicyc_mode == BIALIGN_MODE:
                        missing = [
                            name for name in ("current_adapter", "P_t", "D_t")
                            if not gradient_norms[name]["has_gradient"]
                        ]
                        if missing:
                            raise RuntimeError(
                                "BiAlign expected nonzero gradients for: {}".format(
                                    ", ".join(missing)))
                optimizer.step()

                sums["total"] += float(loss.detach())
                sums["classification"] += float(loss_c.detach())
                sums["rsiat"] += float(loss_rsiat.detach())
                for key in ("align", "orth", "loss_fwd", "loss_back",
                            "bialign", "loss_a", "loss_d", "cycle_new",
                            "cycle_old", "transport"):
                    sums[key] += float(details[key].detach())
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()

            count = len(train_loader)
            epoch_record = {"epoch": epoch + 1}
            epoch_record.update({key: value / count for key, value in sums.items()})
            epoch_record["train_accuracy"] = float(
                np.around(tensor2numpy(correct) * 100 / total, decimals=2))
            epoch_record["learning_rates"] = [
                float(group["lr"]) for group in optimizer.param_groups]
            epoch_record["gradient_norms"] = gradient_norms
            if self._should_eval_epoch(epoch, self.tuned_epochs, eval_interval):
                epoch_record["test_accuracy"] = float(
                    self._compute_accuracy(self._network, test_loader))
            else:
                epoch_record["test_accuracy"] = None
            history.append(epoch_record)
            logging.info("RSIAT_STAGE1 %s", json.dumps(epoch_record, sort_keys=True))
            self._write_progress(
                "STAGE1", epoch + 1, self.tuned_epochs,
                total_loss=epoch_record["total"],
                loss_cos=epoch_record["classification"],
                loss_fwd=epoch_record["loss_fwd"],
                loss_back=epoch_record["loss_back"],
                loss_bialign=epoch_record["bialign"],
                loss_orth=epoch_record["orth"],
                loss_a=epoch_record["loss_a"],
                loss_d=epoch_record["loss_d"],
                cycle_new=epoch_record["cycle_new"],
                cycle_old=epoch_record["cycle_old"],
            )
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, L_cos {:.3f}, "
                "RSIAT {:.3f}, L_fwd {:.3f}, L_back {:.3f}, "
                "L_bialign {:.3f}, L_orth {:.3f}, A {:.3f}, D {:.3f}, "
                "CycN {:.3f}, CycO {:.3f}, Train_accy {:.2f}, Test_accy {}"
            ).format(
                self._cur_task, epoch + 1, self.tuned_epochs,
                epoch_record["total"], epoch_record["classification"],
                epoch_record["rsiat"], epoch_record["loss_fwd"],
                epoch_record["loss_back"], epoch_record["bialign"],
                epoch_record["orth"], epoch_record["loss_a"],
                epoch_record["loss_d"], epoch_record["cycle_new"],
                epoch_record["cycle_old"], epoch_record["train_accuracy"],
                "skipped" if epoch_record["test_accuracy"] is None
                else "{:.2f}".format(epoch_record["test_accuracy"]),
            )
            prog_bar.set_description(info)
        self._latest_stage1_history = history
        logging.info(info)

    def _inc_loss_components(self, features, features_old):
        mapped_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, mapped_old)
        features_old_norm = F.normalize(mapped_old, p=2, dim=1)
        protos = torch.from_numpy(self._class_means).float().to(
            self._device, non_blocking=True)
        protos = self.old_ae(protos)
        protos = F.normalize(protos, p=2, dim=1)
        similarity = torch.matmul(protos, features_old_norm.t())
        loss_orth = similarity.sum() / (similarity.shape[0] * similarity.shape[1])
        loss = self.args["beta"] * loss_align + self.args["gamma"] * loss_orth
        return loss, loss_align, loss_orth

    def _inc_loss(self, features, features_old):
        return self._inc_loss_components(features, features_old)[0]

    def _bialign_loss_components(self, features, features_old):
        terms = bialign_loss_terms(
            features, features_old, self.old_ae, self.reverse_projector)
        # Preserve RSIAT L_orth exactly: P_t maps both old samples and stored
        # prototypes, followed by the original normalization/similarity mean.
        features_old_norm = F.normalize(terms["mapped_old"], p=2, dim=1)
        protos = torch.from_numpy(self._class_means).float().to(
            self._device, non_blocking=True)
        protos = self.old_ae(protos)
        protos = F.normalize(protos, p=2, dim=1)
        similarity = torch.matmul(protos, features_old_norm.t())
        loss_orth = similarity.sum() / (similarity.shape[0] * similarity.shape[1])
        loss = (
            self.args["beta"] * terms["loss_bialign"]
            + self.args["gamma"] * loss_orth)
        return loss, terms, loss_orth

    def _compute_rt_loss(self, inputs, targets, epoch=None, warmup_epoch=10):
        loss_cos = AngularPenaltySMLoss(
            loss_type="cosface", eps=1e-7,
            s=self.args["scale"], m=self.args["margin"])
        if self._cur_task == 0:
            features = self._network_module_ptr.extract_vector(inputs)
            logits = self._network_module_ptr.fc(features)["logits"]
            loss_c = loss_cos(
                logits[:, self._known_classes:], targets - self._known_classes)
            zero = features.new_zeros(())
            lambda_rs = self.args["lambda_rs"] * min(1.0, epoch / warmup_epoch)
            loss_base = lambda_rs * self.rs_loss_func(features, targets)
            return logits, loss_c, loss_base, {
                "align": zero, "orth": zero,
                "loss_fwd": zero, "loss_back": zero, "bialign": zero,
                "loss_a": zero, "loss_d": zero, "cycle_new": zero,
                "cycle_old": zero, "transport": zero,
            }

        # This helper guarantees z_new/z_old consume the exact same augmented x.
        features, features_old = same_input_feature_pair(
            self._network_module_ptr.extract_vector,
            self.old_network_module_ptr.extract_vector,
            inputs,
        )
        logits = self._network_module_ptr.fc(features)["logits"]
        loss_c = loss_cos(
            logits[:, self._known_classes:], targets - self._known_classes)
        zero = features.new_zeros(())
        if self.bicyc_mode == BIALIGN_MODE:
            loss_rsiat, bialign_terms, loss_orth = (
                self._bialign_loss_components(features, features_old))
            # L_fwd is numerically the original symmetric MSE alignment value;
            # ``align`` remains as a compatibility metric, not an extra loss.
            return logits, loss_c, loss_rsiat, {
                "align": bialign_terms["loss_fwd"],
                "orth": loss_orth,
                "loss_fwd": bialign_terms["loss_fwd"],
                "loss_back": bialign_terms["loss_back"],
                "bialign": bialign_terms["loss_bialign"],
                "loss_a": zero,
                "loss_d": zero,
                "cycle_new": zero,
                "cycle_old": zero,
                "transport": zero,
            }

        loss_rsiat, loss_align, loss_orth = self._inc_loss_components(
            features, features_old)
        terms = bicyc_loss_terms(
            features, features_old, self.forward_transport,
            self.backward_transport, self.bicyc_mode)
        transport = weighted_bicyc_loss(
            terms, self.bicyc_mode, self.lambda_bi, self.lambda_cycle)
        return logits, loss_c, loss_rsiat, {
            "align": loss_align,
            "orth": loss_orth,
            "loss_fwd": zero,
            "loss_back": zero,
            "bialign": zero,
            "loss_a": terms["loss_a"],
            "loss_d": terms["loss_d"],
            "cycle_new": terms["cycle_new"],
            "cycle_old": terms["cycle_old"],
            "transport": transport,
        }


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
        return (
            pos_loss.sum() / (mask_pos.sum() + 1e-6)
            + self.lamda * neg_loss.sum() / (mask_neg.sum() + 1e-6)
        )
