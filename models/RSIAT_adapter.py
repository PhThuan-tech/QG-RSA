import copy
import logging
import time
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from utils.inc_net import SimpleVitNet
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseLearner
from utils.toolkit import count_parameters, log_count_parameter, target2onehot, tensor2numpy
from utils.loss import AngularPenaltySMLoss, prototype_relation_kl
from utils.toolkit import make_drift_projector
from utils.quantum_kernel import QuantumKernelModule
from utils.statistics_transport import transport_gaussian_statistics
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

        self._old_most_sentive = []
        self._update_grads = {}

        self.logit_norm = None
        self.tuned_epochs = None
        self.task_sizes = []
        self.rs_loss_func = RS_Loss(self.args["alpha"], self.args["rs_margin"])
        self.old_ae = None
        self._validate_retention_options()

        self.use_quantum_kernel_base = bool(
            args.get("use_quantum_kernel_base", False)
        )
        self.use_quantum_kernel_inc = bool(
            args.get("use_quantum_kernel_inc", False)
        )
        if self.use_quantum_kernel_inc:
            if args.get("q_inc_pair", "old_proj") not in {"old_proj", "current"}:
                raise ValueError("q_inc_pair must be 'old_proj' or 'current'.")
            if args.get("inc_loss_mode", "mean") not in {"mean", "margin"}:
                raise ValueError("inc_loss_mode must be 'mean' or 'margin'.")
            weight = float(args.get("q_inc_weight", 1.0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("q_inc_weight must be finite and nonnegative.")
            warmup = args.get("q_inc_warmup_epochs", 0)
            if int(warmup) != warmup or warmup < 0:
                raise ValueError("q_inc_warmup_epochs must be a nonnegative integer.")
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
        if args.get("ssca_feature_mode", "legacy") not in {"legacy", "paired_eval"}:
            raise ValueError("ssca_feature_mode must be 'legacy' or 'paired_eval'.")
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
                    input_dim=self._network.feature_dim,
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
            self.old_ae = self._new_drift_projector()
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

    def _validate_retention_options(self):
        if self.args.get("ae_type", "legacy_sigmoid") not in {"legacy_sigmoid", "signed_residual"}:
            raise ValueError("Invalid ae_type.")
        for key, default in (("relation_distill_weight", 0.0), ("stats_cov_shrinkage", 0.0)):
            value = float(self.args.get(key, default))
            if not math.isfinite(value) or value < 0 or (key == "stats_cov_shrinkage" and value > 1):
                raise ValueError("Invalid {}".format(key))
        temperature = float(self.args.get("relation_temperature", 0.2))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("relation_temperature must be positive and finite.")
        transport = self.args.get("statistics_transport", "legacy")
        if transport not in {"legacy", "guarded_ridge"}:
            raise ValueError("statistics_transport must be 'legacy' or 'guarded_ridge'.")
        if transport != "legacy" and (
            self.args.get("ssca_feature_mode", "legacy") != "paired_eval" or not self.args.get("ssca", False)
        ):
            raise ValueError("Guarded statistics transport requires SSCA and paired_eval features.")
        if transport != "legacy" and self.args.get("compact_diagonal_checkpoint", False):
            raise ValueError("Covariance transport requires full-covariance checkpoints, not diagonal compaction.")
        if transport != "legacy":
            rank = self.args.get("transport_rank", 32)
            ridge = float(self.args.get("transport_ridge", 0.01))
            change = float(self.args.get("transport_max_change", 0.25))
            support = float(self.args.get("transport_support_scale", 1.0))
            floor = float(self.args.get("transport_support_floor", 0.05))
            if int(rank) != rank or rank < 1 or not math.isfinite(ridge) or ridge <= 0:
                raise ValueError("Invalid transport rank/ridge.")
            if not 0 < change < 1 or not math.isfinite(support) or support <= 0 or not 0 <= floor <= 1:
                raise ValueError("Invalid transport change/support limit.")
        if self.args.get("q_detach_prototypes", False) and self.args.get("q_inc_pair", "old_proj") != "current":
            raise ValueError("q_detach_prototypes requires q_inc_pair='current'.")
        base_lr = self.args.get("base_adapter_lr")
        if base_lr is not None and (not math.isfinite(float(base_lr)) or float(base_lr) <= 0):
            raise ValueError("base_adapter_lr must be positive and finite.")

    def _new_drift_projector(self):
        projector_type = self.args.get("ae_type", "legacy_sigmoid")
        def construct():
            return make_drift_projector(self._network.feature_dim, self.args["ae_code_dims"], projector_type)
        if projector_type == "legacy_sigmoid" and not self.args.get("ae_reset_each_task", False):
            return construct()
        # New ablations must not change backbone/head initialization RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(self.args.get("ae_init_seed", 1234)) + self._cur_task)
            return construct()

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network


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

    def _build_ssca_loader(self, data_manager, train_dataset):
        """Opt-in matched images/views; legacy remains reproducible as a control.

        Displacement requires row i before/after training to describe the same
        image. A shuffled, augmented training loader does not satisfy this.
        Use only the actual training subset (never validation/test images).
        """
        mode = self.args.get("ssca_feature_mode", "legacy")
        if mode == "legacy":
            return self.train_loader
        if mode != "paired_eval":
            raise ValueError("ssca_feature_mode must be 'legacy' or 'paired_eval'.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.args["seed"]) + self._cur_task)
        return DataLoader(
            data_manager.get_eval_view(train_dataset),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            generator=generator,
        )

    def incremental_train(self, data_manager):
        self._cur_task += 1
        
        if self._cur_task == 1 or (self._cur_task > 1 and self.args.get("ae_reset_each_task", False)):
            self.old_ae = self._new_drift_projector().to(self._device)
            
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
        self.seen_val_loader = None
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
            # Recreate the SAME held-out partitions for earlier seen tasks.
            # Used only for evaluation; old images never enter model updates.
            validation_parts = []
            lower = 0
            for task_index, size in enumerate(self.task_sizes):
                _, held_out = data_manager.get_dataset_with_validation(
                    np.arange(lower, lower + size), val_ratio=val_ratio,
                    seed=int(self.args["seed"]) + task_index,
                )
                validation_parts.append(held_out)
                lower += size
            self.seen_val_loader = DataLoader(
                ConcatDataset(validation_parts), batch_size=self.batch_size, shuffle=False,
                num_workers=0, pin_memory=self.pin_memory,
                generator=torch.Generator().manual_seed(int(self.args["seed"]) + self._cur_task),
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
        if self.quantum_kernel is not None:
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

      
        if self._cur_task >0:
            self._network.to(self._device)
            drift_loader = self._build_ssca_loader(data_manager, train_dataset)
            logging.info("SSCA feature mode: %s", self.args.get("ssca_feature_mode", "legacy"))
            train_embeddings_old, drift_labels_old = self.extract_features(
                drift_loader, self._network_module_ptr, None
            )

        evaluation_loader = self._evaluation_loader()
        self._train(self.train_loader, evaluation_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

      
        if self._cur_task >0:
            train_embeddings_new, drift_labels_new = self.extract_features(
                drift_loader, self._network_module_ptr, None
            )
            if self.args.get("ssca_feature_mode", "legacy") == "paired_eval":
                if not torch.equal(drift_labels_old, drift_labels_new):
                    raise RuntimeError("SSCA before/after rows are not aligned.")
            if self.args.get("statistics_transport", "legacy") == "guarded_ridge":
                self._transport_old_statistics(train_embeddings_old, train_embeddings_new)
            else:
                old_class_mean = self._class_means[:self._known_classes]
                gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 4.0)
                if self.args['ssca'] is True:
                    old_class_mean +=gap
                    self._class_means[:self._known_classes] = old_class_mean

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        pre_ca = self._record_stage_accuracy("pre_ca")
        if self._cur_task>0 and self.args['ca_epochs']>0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
        post_ca = self._record_stage_accuracy("post_ca")
        if pre_ca is not None:
            logging.info("CA accuracy delta task=%d total=%.4f old=%.4f new=%.4f", self._cur_task,
                         post_ca["grouped"]["total"] - pre_ca["grouped"]["total"],
                         post_ca["grouped"]["old"] - pre_ca["grouped"]["old"],
                         post_ca["grouped"]["new"] - pre_ca["grouped"]["new"])

    def _transport_old_statistics(self, before, after):
        means, covariance, info = transport_gaussian_statistics(
            self._class_means[:self._known_classes], self._class_covs[:self._known_classes],
            before.to(self._device), after.to(self._device),
            rank=int(self.args.get("transport_rank", 32)), ridge=float(self.args.get("transport_ridge", 0.01)),
            max_change=float(self.args.get("transport_max_change", 0.25)),
            support_scale=float(self.args.get("transport_support_scale", 1.0)),
            support_floor=float(self.args.get("transport_support_floor", 0.05)),
            seed=int(self.args["seed"]) + self._cur_task,
        )
        self._class_means[:self._known_classes] = means
        self._class_covs[:self._known_classes] = covariance
        logging.info("Statistics transport task=%d: %s", self._cur_task, info)

    def _quantum_used_this_task(self):
        return (
            self._cur_task == 0 and self.use_quantum_kernel_base
        ) or (
            self._cur_task > 0 and self.use_quantum_kernel_inc
        )

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
                if self._cur_task == 0 or self.args.get("q_inc_pair", "old_proj") == "current":
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
            adapter_lr = float(self.args.get("base_adapter_lr", 0.01))
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
                if (self.use_quantum_kernel_inc
                        or self.args.get("ae_type", "legacy_sigmoid") != "legacy_sigmoid"
                        or self.args.get("ae_reset_each_task", False)
                        or self.args.get("relation_distill_weight", 0.0) > 0):
                    # Opt-in retention controls need the same projector updates,
                    # including the matched non-quantum AdamW control.
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
        prog_bar = tqdm(range(self.tuned_epochs))
        eval_interval = self.args.get("eval_interval", 0)
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        
        for _, epoch in enumerate(prog_bar):
            epoch_start = time.perf_counter()
            self._network.train()
            losses = 0.0
            losses_c, losses_rt = 0.0, 0.0
            correct, total = 0, 0
            if self.quantum_kernel is not None and self._quantum_used_this_task():
                self.quantum_kernel.reset_epoch_diagnostics()

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                logits, loss_c, loss_rt = self._compute_rt_loss(inputs, targets, epoch, warmup_epoch)
                loss = loss_c + loss_rt
                optimizer.zero_grad()
                loss.backward()
                if self.quantum_kernel is not None and self._quantum_used_this_task():
                    self.quantum_kernel.record_gradient_health()
                optimizer.step()
                losses += loss.item()
                losses_c += loss_c.item()
                losses_rt += loss_rt.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()
            epoch_seconds = time.perf_counter() - epoch_start

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if self._should_eval_epoch(epoch, self.tuned_epochs, eval_interval):
                test_acc_msg = "{:.2f}".format(
                    self._compute_accuracy(self._network, test_loader)
                )
            else:
                test_acc_msg = "skipped"
            metric_label = "Validation_accy" if getattr(self, "seen_val_loader", None) is not None else "Test_accy"
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_c {:.3f}, Losses_rt {:.3f}, Train_accy {:.2f}, {} {}, Time {:.2f}s".format(
                self._cur_task,
                epoch + 1,
                self.tuned_epochs,
                losses / len(train_loader),
                losses_c/len(train_loader),
                losses_rt/len(train_loader),
                train_acc,
                metric_label,
                test_acc_msg,
                epoch_seconds,
            )
            prog_bar.set_description(info)
            if self.quantum_kernel is not None and self._quantum_used_this_task():
                logging.info(
                    "QKSR diagnostics task=%d epoch=%d gamma=%.8g gradients_epoch=%s kernel_epoch=%s inc_weight=%.8g",
                    self._cur_task,
                    epoch + 1,
                    float(self.quantum_kernel.gamma.detach().item()),
                    self.quantum_kernel.epoch_gradient_health(),
                    self.quantum_kernel.epoch_kernel_stats(),
                    self._quantum_inc_weight(epoch) if self._cur_task > 0 else 0.0,
                )
        logging.info(info)
        if self._device.type == "cuda":
            logging.info(
                "Task %d peak GPU memory: %.2f MiB",
                self._cur_task,
                torch.cuda.max_memory_allocated(self._device) / (1024 ** 2),
            )

    def _quantum_inc_weight(self, epoch=0):
        weight = float(self.args.get("q_inc_weight", 1.0))
        warmup = int(self.args.get("q_inc_warmup_epochs", 0))
        if warmup:
            weight *= min(1.0, (max(0, epoch or 0) + 1) / warmup)
        return weight

    def _inc_loss(self, features, features_old, epoch=0):
        previous_features = features_old
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
            quantum_prototypes = protos.detach() if self.args.get("q_detach_prototypes", False) else protos
            similarity = self.quantum_kernel(quantum_prototypes, comparison_features)
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
        # Do not rescale alignment or the original cosine-based RSIAT loss.
        inc_weight = self._quantum_inc_weight(epoch) if self.use_quantum_kernel_inc else 1.0
        loss = self.args["beta"] * loss_align + self.args["gamma"] * inc_weight * loss_orth
        relation_weight = float(self.args.get("relation_distill_weight", 0.0))
        if relation_weight:
            relation = prototype_relation_kl(
                features, previous_features, protos,
                torch.from_numpy(self._class_means).float().to(self._device),
                temperature=float(self.args.get("relation_temperature", 0.2)),
            )
            loss = loss + relation_weight * relation
        return loss
        
    def _compute_rt_loss(self, inputs, targets, epoch=None, warmup_epoch=10):     
        loss_cos=AngularPenaltySMLoss(loss_type='cosface', eps=1e-7, s=self.args["scale"], m=self.args["margin"])
        features = self._network_module_ptr.extract_vector(inputs)
        logits = self._network_module_ptr.fc(features)["logits"]
        loss_c=loss_cos(logits[:, self._known_classes:], targets - self._known_classes)

        if self._cur_task == 0:
            progress = min(1.0, (epoch or 0) / warmup_epoch) if warmup_epoch > 0 else 1.0
            lambda_rs = self.args["lambda_rs"] * progress
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
            return logits, loss_c, loss_base
        
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        loss_inc = self._inc_loss(features, features_old, epoch=epoch)
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
