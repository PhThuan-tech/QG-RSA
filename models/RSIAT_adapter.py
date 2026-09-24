import copy
import json
import logging
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
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
from utils.inc_net import SimpleVitNet
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import AutoencoderSigmoid, log_count_parameter, tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    """RSIAT plus optional dedicated BiCyc-style affine A/D maps.

    [IMPLEMENTATION ADAPTATION] The dedicated affine maps are intentionally
    separate from RSIAT old_ae (P), whose original loss and optimizer path stay
    unchanged.
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
        if self.bicyc_mode not in VALID_MODES:
            raise ValueError(
                "Unknown bicyc_mode {!r}; expected one of {}".format(
                    self.bicyc_mode, VALID_MODES
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
        self.transport_initial_hashes = {}
        self.transport_history = []
        self.pre_ca_metrics = {}
        self.experiment_records = []
        self._latest_stage1_history = []
        self._current_transport_record = None

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
        return payload

    def _transport_seed(self, task, which):
        return self.transport_init_seed + int(task) * 100 + (1 if which == "A" else 2)

    def _restore_transport_state(self, checkpoint):
        self.transport_history = copy.deepcopy(checkpoint.get("transport_history", []))
        self.pre_ca_metrics = copy.deepcopy(checkpoint.get("pre_ca_metrics", {}))
        self.experiment_records = copy.deepcopy(checkpoint.get("experiment_records", []))
        self.transport_initial_hashes = copy.deepcopy(
            checkpoint.get("transport_initial_hashes", {})
        )
        self.forward_transport = None
        self.backward_transport = None
        a_state = checkpoint.get("forward_transport_state_dict")
        d_state = checkpoint.get("backward_transport_state_dict")
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
        if self._cur_task >= 1:
            if self.bicyc_mode != "official" and self.forward_transport is None:
                raise ValueError("Incremental transport checkpoint is missing A")
            if self.bicyc_mode in ("bidirectional", "cycle") and self.backward_transport is None:
                raise ValueError("Bidirectional transport checkpoint is missing D")

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
        self.transport_initial_hashes = {}
        if self._cur_task <= 0 or self.bicyc_mode == "official":
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

        record = {
            "task": int(self._cur_task),
            "mode": self.bicyc_mode,
            "implementation_label": (
                "[IMPLEMENTATION ADAPTATION] RSIAT + BiCyc-style "
                "bidirectional/cycle transport"
            ),
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
            if self.bicyc_mode != "official":
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
            if self.bicyc_mode == "official":
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

        if self._cur_task > 0 and self.args["ca_epochs"] > 0 and self.args["ca"] is True:
            self._write_progress("CA", 0, self.args["ca_epochs"])
            self._stage2_compact_classifier(task_size, self.args["ca_epochs"])
            record["ca_covariance_stabilization"] = copy.deepcopy(
                self.ca_covariance_stabilization
            )
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

            sums = {key: 0.0 for key in (
                "total", "classification", "rsiat", "align", "orth",
                "loss_a", "loss_d", "cycle_new", "cycle_old", "transport")}
            correct, total = 0, 0
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                logits, loss_c, loss_rsiat, details = self._compute_rt_loss(
                    inputs, targets, epoch, warmup_epoch)
                loss = loss_c + loss_rsiat
                if self.bicyc_mode != "official":
                    loss = loss + details["transport"]
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite Stage-I loss at task {} epoch {}".format(
                            self._cur_task, epoch + 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                sums["total"] += float(loss.detach())
                sums["classification"] += float(loss_c.detach())
                sums["rsiat"] += float(loss_rsiat.detach())
                for key in ("align", "orth", "loss_a", "loss_d",
                            "cycle_new", "cycle_old", "transport"):
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
            if self._should_eval_epoch(epoch, self.tuned_epochs, eval_interval):
                epoch_record["test_accuracy"] = float(
                    self._compute_accuracy(self._network, test_loader))
            else:
                epoch_record["test_accuracy"] = None
            history.append(epoch_record)
            logging.info("BICYC_STAGE1 %s", json.dumps(epoch_record, sort_keys=True))
            self._write_progress(
                "STAGE1", epoch + 1, self.tuned_epochs,
                total_loss=epoch_record["total"],
                loss_a=epoch_record["loss_a"],
                loss_d=epoch_record["loss_d"],
                cycle_new=epoch_record["cycle_new"],
                cycle_old=epoch_record["cycle_old"],
            )
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, "
                "RSIAT {:.3f}, A {:.3f}, D {:.3f}, CycN {:.3f}, "
                "CycO {:.3f}, Train_accy {:.2f}, Test_accy {}"
            ).format(
                self._cur_task, epoch + 1, self.tuned_epochs,
                epoch_record["total"], epoch_record["classification"],
                epoch_record["rsiat"], epoch_record["loss_a"],
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
                "align": zero, "orth": zero, "loss_a": zero,
                "loss_d": zero, "cycle_new": zero,
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
