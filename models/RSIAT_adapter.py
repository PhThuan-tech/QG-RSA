import copy
import logging
import json
import os
import time
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from utils.inc_net import SimpleVitNet
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseLearner
from utils.toolkit import count_parameters, log_count_parameter, target2onehot, tensor2numpy
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import AutoencoderSigmoid
from utils.quantum_kernel import QuantumKernelModule
from utils import stats_diagnostics
from utils.stats_diagnostics import stats_settings
import math
num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        if 'adapter' not in args["convnet_type"]:
            raise NotImplementedError('Adapter requires Adapter backbone')
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
        from utils.qbridge import config_from_args
        self.qbridge_config = config_from_args(args)
        self.qbridge_runtime = None
        self.qbridge_stage_records = []
        stats_settings(args)

        self._old_most_sentive = []
        self._update_grads = {}

        # Existing CA logit normalization (task-wise norm, then temperature); off by default.
        self.logit_norm, _ = stats_diagnostics.ca_settings(args)
        # Fail at start-up, not after task 0, on a bad QRF-CA head configuration.
        from utils.ca_heads import head_settings
        from utils.qsd import qsd_settings
        head_settings(args)
        qsd_settings(args)
        # Holevo-CIL / Holevo-POVM (exps/holevo, exps/holevo_povm); None (off) leaves RSIAT unchanged.
        from utils.holevo import HolevoRuntime, holevo_settings
        holevo_cfg = holevo_settings(args)
        if holevo_cfg is not None and self.qbridge_config is not None:
            raise ValueError("holevo_mode / hp_mode cannot be combined with Q-BRIDGE.")
        self.holevo = HolevoRuntime(holevo_cfg, args["warmup_epoch"], self._device) if holevo_cfg else None
        # QKSR P0 probe (reports/Khắc phục điểm yếu QKSR.md): read-only measurements; None leaves training unchanged.
        from utils.qksr_probe import QKSRProbe, probe_settings
        probe_cfg = probe_settings(args)
        self.qksr_probe = QKSRProbe(probe_cfg, self._device) if probe_cfg else None
        self.tuned_epochs = None
        self.task_sizes = []
        self.rs_loss_func = RS_Loss(self.args["alpha"], self.args["rs_margin"])
        self.old_ae = None

        self.use_quantum_kernel_base = bool(
            args.get("use_quantum_kernel_base", False)
        )
        self.use_quantum_kernel_inc = bool(
            args.get("use_quantum_kernel_inc", False)
        )
        if self.use_quantum_kernel_inc:
            inc_train_mode = args.get("q_inc_train_mode", "frozen")
            if inc_train_mode not in {"frozen", "trainable"}:
                raise ValueError("q_inc_train_mode must be 'frozen' or 'trainable'.")
            if (
                inc_train_mode == "trainable"
                and args.get("q_gamma_mode", "bounded_learned") != "bounded_learned"
            ):
                raise ValueError(
                    "Trainable incremental QKSR requires q_gamma_mode='bounded_learned'."
                )
        if int(args.get("q_calib_samples", 512)) < 2:
            raise ValueError("q_calib_samples must be at least 2.")
        if not 0.0 < float(args.get("rs_margin_quantile", 0.9)) < 1.0:
            raise ValueError("rs_margin_quantile must be in (0, 1).")
        for margin_key, default in (("rs_margin_q", 0.5), ("rs_margin_inc", 0.3)):
            margin_value = float(args.get(margin_key, default))
            if not 0.0 <= margin_value <= 1.0:
                raise ValueError("{} must be in [0, 1].".format(margin_key))
        self.quantum_kernel = None
        self.q_calibration_loader = None
        self.q_calibration_indices = []
        if self.use_quantum_kernel_base or self.use_quantum_kernel_inc:
            init_seed = int(args.get("q_init_seed", 1234))
            # QuantumKernelModule only initializes CPU tensors.  Forking the CPU
            # generator keeps the paired baseline training RNG unchanged.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(init_seed)
                self.quantum_kernel = QuantumKernelModule(
                    input_dim=768,
                    num_qubits=int(args.get("q_num_qubits", 8)),
                    num_layers=int(args.get("q_num_layers", 2)),
                    kernel_type=args.get("q_kernel_type", "pqk"),
                    kernel_order=int(args.get("q_kernel_order", 1)),
                    order2_weight=float(args.get("q_order2_weight", 1.0)),
                    reupload=bool(args.get("q_reupload", False)),
                    gamma_mode=args.get("q_gamma_mode", "bounded_learned"),
                    dtype=args.get("q_dtype", "float32"),
                    init_seed=init_seed,
                ).to(self._device)
            logging.info(
                "QKSR parameter counts: %s", self.quantum_kernel.parameter_counts()
            )

    def _after_load_checkpoint(self, checkpoint):
        """Restore learner-specific state after BaseLearner restores the network."""
        if self._cur_task >= 1:
            self.old_ae = AutoencoderSigmoid(
                input_dims=768,
                code_dims=self.args["ae_code_dims"],
            )
            if "old_ae_state_dict" not in checkpoint:
                raise ValueError(
                    "Checkpoint is missing old_ae_state_dict required to resume task {}."
                    .format(self._cur_task + 1)
                )
            self.old_ae.load_state_dict(checkpoint["old_ae_state_dict"])
            self.old_ae.to(self._device)

        if self.quantum_kernel is not None:
            state_dict = checkpoint.get("quantum_kernel_state_dict")
            if state_dict is None:
                raise ValueError(
                    "Checkpoint is missing quantum_kernel_state_dict for a QKSR run."
                )
            self.quantum_kernel.load_state_dict(state_dict)
            if self._cur_task >= 1 and self.use_quantum_kernel_inc:
                self.quantum_kernel.set_inc_mode(
                    self.args.get("q_inc_train_mode", "frozen")
                )
            elif self._cur_task >= 1:
                self.quantum_kernel.set_inc_mode("frozen")

        self._network_module_ptr = self._network
        self.old_network_module_ptr = self._old_network

    def _validate_checkpoint(self, checkpoint):
        if not self.args.get("_qbridge_restore_only", False):
            return super()._validate_checkpoint(checkpoint)
        # Only incremental settings may differ when branching from the same base.
        if checkpoint.get("cur_task") != 0:
            raise ValueError("Q-BRIDGE base branching requires task 0")
        base_view = dict(checkpoint)
        metadata = dict(checkpoint.get("run_metadata", {}))
        for key in ("use_quantum_kernel_inc", "q_inc_train_mode", "inc_loss_mode", "q_inc_pair",
                    "ssca_mode", "ssca_sigma", "ca_stats_source", "rng_replicate",
                    "ca_logit_norm", "ca_task_margin"):
            metadata.pop(key, None)
        base_view["run_metadata"] = metadata
        super()._validate_checkpoint(base_view)
        required = {"class_order", "class_means", "class_covs", "quantum_kernel_state_dict"}
        if required - checkpoint.keys():
            raise ValueError("Incomplete paired base checkpoint")

    def initialize_qbridge_branch(self):
        from utils.qbridge import BridgeRuntime
        base_hash = self.args["_qbridge_base_hash"]
        self._qb_snapshot_id = base_hash + ":task0"
        self._qb_means_id = self._qb_snapshot_id
        if self.qbridge_config is not None:
            self.qbridge_runtime = BridgeRuntime(self.qbridge_config, self.quantum_kernel,
                                                  base_hash, self.args["seed"], self._device)
        self.quantum_kernel.set_inc_mode("frozen")

    def _start_qbridge_task(self):
        self._qb_seen_ids = list(self.class_order[:self._total_classes])
        self._qb_old_ids = list(self.class_order[:self._known_classes])
        self._qb_new_ids = list(self.class_order[self._known_classes:self._total_classes])
        if self.qbridge_runtime is None:
            raise ValueError("Q-BRIDGE requires the paired-base runner and artifacts")
        self.qbridge_runtime.start_task(
            torch.as_tensor(self._class_means, dtype=torch.float32, device=self._device),
            self._qb_old_ids, self._qb_snapshot_id, self._qb_means_id)
        self.qbridge_runtime.adapter_parameters = tuple(
            p for p in self._network_module_ptr.convnet.parameters() if p.requires_grad)
        from pathlib import Path
        from utils.qbridge import write_json
        write_json(Path(self.args["_qbridge_report_dir"]) / ("task_%d_provenance.json" % self._cur_task), {
            "task_index": self._cur_task, "class_order": list(self.class_order),
            "old_global_ids": self._qb_old_ids, "new_global_ids": self._qb_new_ids,
            "snapshot_encoder_id": self._qb_snapshot_id, "means_encoder_id": self._qb_means_id,
            "means_are_estimates": True, "metric_base_checkpoint_hash": self.args["_qbridge_base_hash"],
            "selector_artifact_hash": self.qbridge_runtime.artifact_hash,
            "calibrator_fit_id": self.qbridge_runtime.fit_id,
            "version_origin": "base binding reconstructed from validated checkpoint; later SSCA updates tagged by task"})

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network
        if self.args.get("qbridge_experiment", False):
            self._qb_snapshot_id = self.args["_qbridge_base_hash"] + ":task%d" % self._cur_task


    def extract_features(self, trainloader, model, args):
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.to(self._device, non_blocking=True)
                label = label.to(self._device, non_blocking=True)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        return embedding_list, label_list

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task == 1 and stats_settings(self.args)["rng_replicate"]:
            stats_diagnostics.reseed_replicate(self.args)

        if self._cur_task == 1:
            self.old_ae = AutoencoderSigmoid(input_dims=768, code_dims=self.args["ae_code_dims"])
            self.old_ae.to(self._device)
            
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + task_size
        # self._network.update_fc(data_manager.get_task_size(self._cur_task)*4)
        self._network.update_fc(task_size)
        self._network_module_ptr = self._network
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
    
        current_classes = np.arange(self._known_classes, self._total_classes)
        val_ratio = float(self.args.get("val_ratio", 0.0) or 0.0)
        self.val_loader = None
        if val_ratio > 0.0:
            split_seed = int(self.args["seed"]) + self._cur_task
            train_dataset, val_dataset = data_manager.get_dataset_with_validation(
                current_classes, val_ratio=val_ratio, seed=split_seed
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
            )
            logging.info(
                "Task %d validation split: train=%d, validation=%d, ratio=%.4f",
                self._cur_task, len(train_dataset), len(val_dataset), val_ratio,
            )
        else:
            train_dataset = data_manager.get_dataset(
                current_classes, source="train", mode="train"
            )

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

        # Gamma calibration uses the same training samples but deterministic
        # evaluation transforms.  A private generator prevents RNG drift.
        if self.quantum_kernel is not None and not (self.qbridge_config is not None and self._cur_task > 0):
            calibration_dataset = data_manager.get_eval_view(train_dataset)
            calibration_count = min(
                int(self.args.get("q_calib_samples", 512)), len(calibration_dataset)
            )
            if calibration_count < 2:
                raise ValueError("QKSR gamma calibration requires at least two samples.")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                int(self.args.get("q_init_seed", 1234)) + self._cur_task
            )
            indices = torch.randperm(len(calibration_dataset), generator=generator)[
                :calibration_count
            ].tolist()
            self.q_calibration_indices = indices
            self.q_calibration_loader = DataLoader(
                Subset(calibration_dataset, indices),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=self.pin_memory,
                generator=generator,
            )
            logging.info(
                "QKSR calibration task=%d samples=%d indices=%s",
                self._cur_task, calibration_count, indices,
            )

            if self._cur_task == 0:
                mode = "trainable" if self.use_quantum_kernel_base else "frozen"
            elif self.use_quantum_kernel_inc:
                mode = self.args.get("q_inc_train_mode", "frozen")
            else:
                mode = "frozen"
            self.quantum_kernel.set_inc_mode(mode)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

      
        stats_cfg = stats_settings(self.args)
        if self._cur_task >0:
            self._network.to(self._device)
            # The legacy pass always runs: it fixes the global RNG stream so
            # ssca_mode changes only the drift estimate, not training.
            train_embeddings_old, train_labels_old = self.extract_features(self.train_loader, self._network, None)
            if stats_cfg["needs_paired"]:
                paired_loader = stats_diagnostics.deterministic_loader(
                    data_manager.get_eval_view(train_dataset), self.batch_size, self.num_workers,
                    self.pin_memory, int(self.args["seed"]) + self._cur_task)
                paired_old, paired_labels = self.extract_features(paired_loader, self._network, None)

        if self._cur_task > 0:
            self._qksr_v2_refit(train_embeddings_old, train_labels_old)
        if getattr(self, "holevo", None) is not None and self._cur_task > 0:
            # Queue initialised from the mandatory pass above: no extra data pass, RNG unchanged.
            self.holevo.start_task(self, train_embeddings_old, train_labels_old)
        if self.qbridge_config is not None and self._cur_task > 0:
            self._start_qbridge_task()
        evaluation_loader = self.val_loader if self.val_loader is not None else self.test_loader
        self._train(self.train_loader, evaluation_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        if getattr(self, "holevo", None) is not None and self._cur_task > 0:
            self.holevo.end_task(self)

        if self.args.get("qbridge_experiment", False) and self._cur_task > 0:
            from utils.qbridge_experiment import evaluate_stage
            evaluate_stage(self, "pre_ca")

      
        estimates, kernel_report = {}, None
        if self._cur_task >0:
            train_embeddings_new, _ = self.extract_features(self.train_loader, self._network, None)
            stale_means = self._class_means[:self._known_classes].copy()
            estimates["stale"] = stale_means
            # auto_nn reads the features of the applied estimator, so turning
            # diagnostics on never changes the bandwidth.
            sigma = stats_diagnostics.resolve_sigma(
                stats_cfg["sigma"], paired_old if stats_cfg["mode"] == "paired" else train_embeddings_old,
                stale_means)
            stats_cfg["sigma_used"] = sigma
            if stats_cfg["sigma"] == stats_diagnostics.AUTO_SIGMA:
                logging.info("STATS task=%d ssca_sigma auto_nn -> %.4f", self._cur_task, sigma)
            gap = None
            if stats_cfg["mode"] == "legacy":
                gap = self.displacement(train_embeddings_old, train_embeddings_new, stale_means, sigma)
                estimates["legacy"] = stale_means + gap
            elif stats_cfg["diagnostics"]:
                estimates["legacy"] = stale_means + stats_diagnostics.kernel_displacement(
                    train_embeddings_old, train_embeddings_new, stale_means, sigma)
            if stats_cfg["needs_paired"]:
                paired_new, paired_labels_new = self.extract_features(paired_loader, self._network, None)
                if not torch.equal(paired_labels, paired_labels_new):
                    raise RuntimeError("Paired SSCA passes visited samples in different orders.")
                paired_gap = stats_diagnostics.kernel_displacement(paired_old, paired_new, stale_means, sigma)
                estimates["paired"] = stale_means + paired_gap
                if stats_cfg["mode"] == "paired":
                    gap = paired_gap
                kernel_report = stats_diagnostics.kernel_summary(paired_old, stale_means, sigma)
            if self.args['ssca'] is True:
                self._class_means[:self._known_classes] = stale_means + gap

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if getattr(self, "holevo", None) is not None and self._cur_task == 0:
            self.holevo.fix_encoder(self)
        if self._cur_task > 0 and stats_cfg["needs_oracle"]:
            self._old_statistics_oracle(data_manager, stats_cfg, estimates, kernel_report)
        if self.args.get("qbridge_experiment", False):
            self._qb_means_id = self.args["_qbridge_base_hash"] + ":task%d" % self._cur_task
        if self._cur_task>0 and self.args['ca_epochs']>0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
        if self.args.get("qbridge_experiment", False) and self._cur_task > 0:
            from utils.qbridge_experiment import evaluate_stage
            evaluate_stage(self, "post_ca")
            if self.qbridge_runtime is not None:
                self.qbridge_runtime.cache = None
        if getattr(self, "qksr_probe", None) is not None:
            self.qksr_probe.end_task(self)

    def _old_statistics_oracle(self, data_manager, stats_cfg, estimates, kernel_report):
        """Compare old-class statistics with the oracle computed from old images.

        Reading old-class training images violates exemplar-free CIL. With
        ca_stats_source set to an oracle option, the run is an upper bound for
        diagnosis and must not be reported as a method result.
        """
        known = self._known_classes
        # Test accuracy after adaptation and SSCA, before CA refits the head.
        from utils.qbridge_experiment import preserve_rng
        with preserve_rng():
            modes = [(module, module.training) for module in self._network.modules()]
            pred, target = self._eval_cnn(self.test_loader)
            for module, training in modes:
                module.training = training
        pre_ca = self._evaluate(pred, target)["grouped"]
        pre_ca.update(stats_diagnostics.error_directions(pred[:, 0], target, known))

        dataset = data_manager.get_dataset(np.arange(0, known), source="train", mode="test")
        loader = stats_diagnostics.deterministic_loader(
            dataset, self.batch_size, int(self.args.get("stats_num_workers", 4)),
            self.pin_memory, int(self.args["seed"]) + 1000 + self._cur_task)
        vectors, labels = self._extract_vectors(loader)
        origin = stats_diagnostics.origin_tasks(data_manager, self._cur_task, known)

        replace_cov = stats_cfg["ca_stats_source"] == "oracle_mean_cov"
        oracle_means = np.zeros((known, self._class_means.shape[1]))
        cov_relative = np.full(known, np.nan)
        cov_trace_ratio = np.full(known, np.nan)
        for class_idx in range(known):
            class_vectors = vectors[labels == class_idx]
            if len(class_vectors) < 2:
                raise ValueError("Oracle statistics need at least two samples of class {}.".format(class_idx))
            mean, cov = stats_diagnostics.class_statistics(class_vectors)
            oracle_means[class_idx] = mean
            cov_relative[class_idx], cov_trace_ratio[class_idx] = stats_diagnostics.covariance_errors(
                self._class_covs[class_idx], cov)
            if replace_cov:
                self._class_covs[class_idx] = cov

        stale = estimates["stale"]
        report_estimates = dict(estimates)
        report_estimates["applied"] = self._class_means[:known].copy()
        mean_report = {
            name: stats_diagnostics.summarize_by_origin(
                stats_diagnostics.mean_errors(estimate, oracle_means, stale), origin)
            for name, estimate in report_estimates.items()
        }
        drift = stats_diagnostics.summarize_by_origin({
            "l2": np.linalg.norm(stale - oracle_means, axis=1),
            "relative_to_oracle_norm": np.linalg.norm(stale - oracle_means, axis=1)
            / np.linalg.norm(oracle_means, axis=1),
        }, origin)
        if stats_cfg["ca_stats_source"] != "estimated":
            self._class_means[:known] = oracle_means
            logging.warning("STATS task=%d: CA uses ORACLE old-class %s (upper bound, not exemplar-free).",
                            self._cur_task, "means and covariances" if replace_cov else "means")

        report_dir = self.args.get("_qbridge_report_dir") or (
            str(self.args.get("_run_log_prefix", "stats")) + "_stats")
        stats_diagnostics.write_report(os.path.join(report_dir, "stats_task_%d.json" % self._cur_task), {
            "task": self._cur_task,
            "known_classes": known,
            "ssca": self.args.get("ssca"),
            "ssca_mode": stats_cfg["mode"],
            "ssca_sigma": stats_cfg["sigma"],
            "ssca_sigma_used": stats_cfg.get("sigma_used"),
            "ca_logit_norm": self.logit_norm,
            "ca_task_margin": float(self.args.get("ca_task_margin", 0.0)),
            "ca_stats_source": stats_cfg["ca_stats_source"],
            "oracle_is_exemplar_free": False,
            "rng_replicate": stats_cfg["rng_replicate"],
            "pre_ca_accuracy": pre_ca,
            "estimator_mean_errors": mean_report,
            "oracle_drift": drift,
            "stored_covariance_errors": stats_diagnostics.summarize_by_origin(
                {"relative_frobenius": cov_relative, "trace_ratio": cov_trace_ratio}, origin),
            "ssca_kernel": kernel_report,
        })
        logging.info(
            "STATS task=%d mode=%s source=%s median relative mean error: %s; median cov rel error=%.4f",
            self._cur_task, stats_cfg["mode"], stats_cfg["ca_stats_source"],
            ", ".join("{}={:.4f}".format(name, float("nan") if report["all"]["relative"]["median"] is None
                                          else report["all"]["relative"]["median"])
                      for name, report in mean_report.items()),
            float(np.nanmedian(cov_relative)),
        )

    def _quantum_used_this_task(self):
        return (
            self._cur_task == 0 and self.use_quantum_kernel_base
        ) or (
            self._cur_task > 0 and self.use_quantum_kernel_inc
        )

    @torch.no_grad()
    def _qksr_v2_refit(self, feats, labels):
        """QKSR-v2 P2 (exps/qksr2/README.md): refit the PQK projector / bandwidth at the start of a task.

        Uses only stored statistics of the old classes (after SSCA) and the mandatory pass over the new
        task's images with the network of the previous task; deterministic, no RNG, off by default.
        """
        proj_mode = self.args.get("q_proj_refit", "none")
        gamma_mode = self.args.get("q_gamma_refit", "none")
        if proj_mode not in ("none", "lda") or gamma_mode not in ("none", "target"):
            raise ValueError("q_proj_refit must be none|lda and q_gamma_refit none|target.")
        if proj_mode == "none" and gamma_mode == "none":
            return
        if self.quantum_kernel is None or not self.use_quantum_kernel_inc:
            raise ValueError("q_proj_refit / q_gamma_refit need the incremental QKSR kernel.")
        known, total = self._known_classes, self._total_classes
        dev = self._device
        feats = torch.as_tensor(feats).to(dev, dtype=torch.float64)
        labels = torch.as_tensor(labels).to(dev)
        old_means = torch.as_tensor(np.asarray(self._class_means[:known]), dtype=torch.float64, device=dev)
        info = {}
        if proj_mode == "lda":
            within = torch.zeros(feats.shape[1], feats.shape[1], dtype=torch.float64, device=dev)
            for c in range(known):  # stored covariances, accumulated one class at a time
                within += torch.as_tensor(self._class_covs[c]).to(dev, dtype=torch.float64)
            new_means = []
            for c in range(known, total):
                rows = feats[labels == c]
                mu = rows.mean(0)
                new_means.append(mu)
                within += (rows - mu).T @ (rows - mu) / max(rows.shape[0] - 1, 1)
            within /= total
            info["lda"] = self.quantum_kernel.refit_projector_lda(
                torch.cat([old_means, torch.stack(new_means)]), within,
                angle_std=float(self.args.get("q_angle_std", 0.8)), ridge=float(self.args.get("q_lda_ridge", 1e-3)))
        if gamma_mode == "target":
            dtype = self.quantum_kernel.gamma0.dtype
            info["gamma"] = self.quantum_kernel.set_gamma_target(
                old_means.to(dtype), feats[labels >= known][:2048].to(dtype),
                float(self.args.get("q_gamma_target", 0.2)))
        logging.info("QKSR refit task=%d %s", self._cur_task, json.dumps(info))

    def _quantum_trainable_this_task(self):
        if self.quantum_kernel is None or not self._quantum_used_this_task():
            return False
        if self._cur_task == 0:
            return True
        return self.args.get("q_inc_train_mode", "frozen") == "trainable"

    def _calibrate_quantum_kernel(self):
        if (
            self.quantum_kernel is None
            or not self._quantum_used_this_task()
            or bool(self.quantum_kernel.gamma_initialized.item())
        ):
            return
        if self.q_calibration_loader is None:
            raise RuntimeError("QKSR calibration loader has not been initialized.")

        network_was_training = self._network.training
        old_ae_was_training = self.old_ae.training if self.old_ae is not None else False
        self._network.eval()
        if self.old_ae is not None:
            self.old_ae.eval()

        features = []
        with torch.no_grad():
            for _, inputs, _ in self.q_calibration_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                if self._cur_task == 0:
                    encoded = self._network_module_ptr.extract_vector(inputs)
                else:
                    encoded = self.old_network_module_ptr.extract_vector(inputs)
                    encoded = self.old_ae(encoded)
                features.append(encoded)
            features = torch.cat(features, dim=0)

            if self._cur_task == 0:
                gamma0 = self.quantum_kernel.calibrate_gamma(
                    features, exclude_diagonal=True
                )
            else:
                prototypes = torch.from_numpy(
                    self._class_means[:self._known_classes]
                ).float().to(self._device, non_blocking=True)
                prototypes = self.old_ae(prototypes)
                gamma0 = self.quantum_kernel.calibrate_gamma(
                    prototypes, features, exclude_diagonal=False
                )

        if network_was_training:
            self._network.train()
        if self.old_ae is not None and old_ae_was_training:
            self.old_ae.train()
        logging.info(
            "QKSR gamma calibrated at task %d: gamma0=%.8g, mode=%s",
            self._cur_task, gamma0, self.args.get("q_gamma_mode", "bounded_learned"),
        )

    def _quantum_param_groups(self, adapter_lr):
        if not self._quantum_trainable_this_task():
            return []
        return self.quantum_kernel.param_groups(
            adapter_lr=adapter_lr,
            weight_decay=self.weight_decay,
            metric_lr_mult=self.args.get("q_metric_lr_mult", 0.1),
        )

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self._calibrate_quantum_kernel()

        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            adapter_lr = 0.01
            param_groups = [
                {'params': self._network.convnet.blocks[-1].parameters(), 'lr': adapter_lr,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.convnet.blocks[:-1].parameters(), 'lr': adapter_lr,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.fc.parameters(), 'lr': adapter_lr,
                 'weight_decay': self.args['weight_decay']}
            ]
            quantum_groups = self._quantum_param_groups(adapter_lr)
            param_groups.extend(quantum_groups)

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(
                    param_groups, momentum=0.9, lr=self.init_lr,
                    weight_decay=self.weight_decay,
                )
            elif self.args['optimizer'] == 'adam':
                if quantum_groups:
                    optimizer = optim.AdamW(
                        [{'params': self._network.parameters(), 'lr': self.init_lr,
                          'weight_decay': self.weight_decay}] + quantum_groups
                    )
                else:
                    optimizer = optim.AdamW(
                        self._network.parameters(), lr=self.init_lr,
                        weight_decay=self.weight_decay,
                    )
            else:
                raise ValueError("Unknown optimizer {}".format(self.args['optimizer']))
        else:
            self.tuned_epochs = self.args['inc_epochs']
            adapter_lr = self.init_lr
            param_groups = [
                {'params': self._network.convnet.parameters(), 'lr': adapter_lr,
                 'weight_decay': self.weight_decay},
                {'params': self._network.fc.parameters(), 'lr': adapter_lr,
                 'weight_decay': self.weight_decay},
                {'params': self.old_ae.parameters(), 'lr': self.args['ae_init_lr'],
                 'weight_decay': self.args['ae_weight_decay']},
            ]
            quantum_groups = self._quantum_param_groups(adapter_lr)
            param_groups.extend(quantum_groups)

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9)
            elif self.args['optimizer'] == 'adam':
                if self.use_quantum_kernel_inc or self.qbridge_config is not None:
                    # QKSR's incremental objective requires old_ae to update.
                    optimizer = optim.AdamW(param_groups)
                else:
                    # Preserve the original baseline behavior bit-for-bit.
                    optimizer = optim.AdamW(
                        self._network.parameters(), lr=self.init_lr,
                        weight_decay=self.weight_decay,
                    )
            else:
                raise ValueError("Unknown optimizer {}".format(self.args['optimizer']))

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr
        )
        log_count_parameter(optimizer.param_groups)
        self._init_train(
            train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch']
        )

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, warmup_epoch):
        prog_bar = tqdm(range(self.tuned_epochs), disable=self.args.get("qbridge_experiment", False))
        eval_interval = self.args.get("eval_interval", 0)
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        
        for _, epoch in enumerate(prog_bar):
            epoch_start = time.perf_counter()
            self._network.train()
            losses = 0.0
            losses_c, losses_rt = 0.0, 0.0
            correct, total = 0, 0
            qb_records = []
            if self.qbridge_runtime is not None:
                self.qbridge_runtime.observe_gradient = True

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                logits, loss_c, loss_rt = self._compute_rt_loss(inputs, targets, epoch, warmup_epoch)
                loss = loss_c + loss_rt
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if self.qbridge_runtime is not None:
                    qb_records.append(dict(self.qbridge_runtime.last))
                losses += loss.item()
                losses_c += loss_c.item()
                losses_rt += loss_rt.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            if getattr(self, "holevo", None) is not None and self._cur_task > 0:
                self.holevo.end_epoch(self._cur_task, epoch)
            if getattr(self, "qksr_probe", None) is not None:
                self.qksr_probe.end_epoch(self._cur_task, epoch)
            epoch_seconds = time.perf_counter() - epoch_start

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if self._should_eval_epoch(epoch, self.tuned_epochs, eval_interval):
                test_acc_msg = "{:.2f}".format(
                    self._compute_accuracy(self._network, test_loader)
                )
            else:
                test_acc_msg = "skipped"
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, Losses_rt {:.3f}, Train_accy {:.2f}, Test_accy {}, Time {:.2f}s".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / len(train_loader),
                losses_c/len(train_loader),
                losses_rt/len(train_loader),
                train_acc,
                test_acc_msg,
                epoch_seconds,
            )
            prog_bar.set_description(info)
            if self.args.get("qbridge_experiment", False):
                import json
                from pathlib import Path
                record = {"task": self._cur_task, "epoch": epoch + 1,
                          "seconds": epoch_seconds, "batches": len(train_loader),
                          "milliseconds_per_batch": epoch_seconds * 1000 / len(train_loader),
                          "loss": losses / len(train_loader),
                          "gpu_peak_bytes": torch.cuda.max_memory_allocated(self._device)
                          if self._device.type == "cuda" else None,
                          "device": str(self._device)}
                if qb_records:
                    record["mean"] = {key: float(np.mean([r[key] for r in qb_records]))
                                      for key in ("cross", "guard", "active_fraction", "active_weight_mean")}
                    record["first_batch"] = qb_records[0]
                    record["last_batch"] = qb_records[-1]
                with open(Path(self.args["_qbridge_report_dir"]) / "epochs.jsonl", "a", encoding="utf-8") as output:
                    output.write(json.dumps(record, allow_nan=False) + "\n")
                logging.info("QBRIDGE task=%d epoch=%d elapsed=%.2fs status=trained",
                             self._cur_task, epoch + 1, epoch_seconds)
            if self.quantum_kernel is not None and self._quantum_used_this_task():
                logging.info(
                    "QKSR diagnostics task=%d epoch=%d gamma=%.8g gradients=%s kernel=%s",
                    self._cur_task,
                    epoch + 1,
                    float(self.quantum_kernel.gamma.detach().item()),
                    self.quantum_kernel.gradient_health(),
                    self.quantum_kernel.last_kernel_stats(),
                )
        logging.info(info)
        if self._device.type == "cuda":
            logging.info(
                "Task %d peak GPU memory: %.2f MiB",
                self._cur_task,
                torch.cuda.max_memory_allocated(self._device) / (1024 ** 2),
            )

    def _inc_loss(self, features, features_old):
        kd, sep = self._inc_loss_terms(features, features_old)
        return kd + sep

    def _inc_loss_terms(self, features, features_old):
        """(beta * alignment, gamma * separation); their sum is the RSIAT / QKSR incremental loss."""
        features_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        protos = torch.from_numpy(self._class_means).float().to(self._device,non_blocking=True)
        protos = self.old_ae(protos)

        if self.use_quantum_kernel_inc:
            pair_mode = self.args.get("q_inc_pair", "old_proj")
            if pair_mode == "old_proj":
                comparison_features = features_old
            elif pair_mode == "current":
                comparison_features = features
            else:
                raise ValueError("q_inc_pair must be 'old_proj' or 'current'.")
            similarity = self.quantum_kernel(protos, comparison_features)
            inc_loss_mode = self.args.get("inc_loss_mode", "mean")
            if inc_loss_mode == "mean":
                loss_orth = similarity.mean()
            elif inc_loss_mode == "margin":
                margin = float(self.args.get("rs_margin_inc", 0.3))
                loss_orth = F.relu(similarity - margin).mean()
            else:
                raise ValueError("inc_loss_mode must be 'mean' or 'margin'.")
        else:
            features_old_norm = F.normalize(features_old, p=2, dim=1)
            protos = F.normalize(protos, p=2, dim=1)
            similarity = torch.matmul(protos, features_old_norm.t())
            loss_orth = similarity.sum() / (similarity.shape[0]*similarity.shape[1])
        return self.args["beta"] * loss_align, self.args["gamma"] * loss_orth
        
    def _compute_rt_loss(self, inputs, targets, epoch=None, warmup_epoch=10):     
        loss_cos=AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        features = self._network_module_ptr.extract_vector(inputs)
        logits = self._network_module_ptr.fc(features)["logits"]
        loss_c=loss_cos(logits[:, self._known_classes:], targets - self._known_classes)

        if self._cur_task == 0:
            lambda_rs = self.args["lambda_rs"] * min(1.0, epoch / warmup_epoch)
            quantum_module = (
                self.quantum_kernel if self.use_quantum_kernel_base else None
            )
            loss_base = lambda_rs * self.rs_loss_func(
                features,
                targets,
                quantum_kernel_module=quantum_module,
                margin_override=float(self.args.get("rs_margin_q", 0.5)),
                margin_mode=self.args.get("rs_margin_q_mode", "fixed"),
                margin_quantile=float(self.args.get("rs_margin_quantile", 0.9)),
            )
            if getattr(self, "qksr_probe", None) is not None:
                self.qksr_probe.on_step(self, {"c": loss_c, "base": loss_base})
            return logits, loss_c, loss_base
        
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        if getattr(self, "qbridge_config", None) is not None:
            bridge = self.qbridge_runtime
            if bridge is None:
                raise ValueError("Missing Q-BRIDGE paired-base runtime")
            features_old = features_old.detach()
            global_labels = torch.as_tensor(self.class_order, device=targets.device)[targets]
            if self.qbridge_config.get("classification_mode") == "all_seen_cosface":
                from utils.qbridge import id_positions
                scores = self._network_module_ptr.fc.qbridge_scores(
                    features, self._qb_seen_ids, self._qb_old_ids)
                loss_c = loss_cos(scores, id_positions(global_labels, self._qb_seen_ids, targets.device))
            loss_align = F.mse_loss(features, self.old_ae(features_old))
            observe_modules = getattr(bridge, "observe_gradient", False)
            added = bridge.losses(features, features_old, global_labels,
                                  self._network_module_ptr.fc, self.old_ae,
                                  self._qb_seen_ids, self._qb_new_ids)
            bridge.last.update(classification=float(loss_c.detach()), alignment=float(loss_align.detach()),
                               weighted_alignment=float((self.args["beta"] * loss_align).detach()),
                               weighted_bridge=float(added.detach()))
            if observe_modules:
                module_norms = {}
                total_loss = loss_c + self.args["beta"] * loss_align + added
                for name, module in (("adapter", self._network_module_ptr.convnet),
                                     ("head", self._network_module_ptr.fc), ("alignment_projector", self.old_ae)):
                    params = tuple(p for p in module.parameters() if p.requires_grad)
                    grads = torch.autograd.grad(total_loss, params, retain_graph=True, allow_unused=True) if params else ()
                    module_norms[name] = sum(float(g.detach().square().sum()) for g in grads if g is not None) ** .5
                bridge.last["total_loss_module_gradient_norms"] = module_norms
            return logits, loss_c, self.args["beta"] * loss_align + added
        loss_kd, loss_sep = self._inc_loss_terms(features, features_old)
        loss_inc = loss_kd + loss_sep
        if getattr(self, "qksr_probe", None) is not None:
            self.qksr_probe.on_step(self, {"c": loss_c, "kd": loss_kd, "sep": loss_sep})
        if getattr(self, "holevo", None) is not None:
            extra = self.holevo.step_loss(self, features, targets, loss_c, epoch)
            if extra is not None:  # None at ramp weight 0 and in diag mode: update identical to RSIAT
                loss_inc = loss_inc + extra.float()
        return logits, loss_c, loss_inc
    
class RS_Loss(nn.Module):
    def __init__(self, lamda=0.5, margin=0.5):
        super(RS_Loss, self).__init__()
        self.lamda = lamda
        self.margin = margin

    def forward(
        self,
        features,
        labels,
        quantum_kernel_module=None,
        margin_override=None,
        margin_mode="fixed",
        margin_quantile=0.9,
    ):
        device = features.device
        labels = labels[:, None]
        mask = torch.eq(labels, labels.t()).float().to(device)
        eye = torch.eye(mask.size(0), device=device)
        mask_pos = mask - eye
        mask_neg = 1.0 - mask
        if quantum_kernel_module is None:
            features = F.normalize(features, p=2, dim=1)
            dot_prod = torch.matmul(features, features.t())
            margin = self.margin
        else:
            dot_prod = quantum_kernel_module(features)
            margin = self.margin if margin_override is None else margin_override
            if margin_mode == "quantile":
                if not 0.0 < margin_quantile < 1.0:
                    raise ValueError("rs_margin_quantile must be in (0, 1).")
                negative_values = dot_prod[mask_neg.bool()]
                if negative_values.numel() > 0:
                    margin = torch.quantile(
                        negative_values.detach(), margin_quantile
                    )
            elif margin_mode != "fixed":
                raise ValueError("rs_margin_q_mode must be 'fixed' or 'quantile'.")

        pos_loss = F.relu(1.0 - dot_prod) * mask_pos
        neg_loss = F.relu(dot_prod - margin) * mask_neg
        loss = pos_loss.sum() / (mask_pos.sum() + 1e-6) + \
               self.lamda * neg_loss.sum() / (mask_neg.sum() + 1e-6)

        return loss
