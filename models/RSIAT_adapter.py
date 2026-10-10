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
from torch.utils.data import DataLoader, Subset, RandomSampler
from utils.inc_net import SimpleVitNet
from torch.distributions.multivariate_normal import MultivariateNormal
from models.base import BaseLearner
from utils.toolkit import count_parameters, log_count_parameter, target2onehot, tensor2numpy
from utils.loss import AngularPenaltySMLoss
from utils.toolkit import AutoencoderSigmoid
from utils.quantum_kernel import QuantumKernelModule
from utils.experimental_integrity import (
    preserve_rng_state, seed_worker, dataset_manifest, pair_features,
    snapshot_moments, memory_drift_diagnostics,
)
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

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network


    def extract_features(self, trainloader, model, args=None, return_ids=False):
        was_training = model.training
        model.eval()
        embedding_list = []
        label_list = []
        id_list = []
        feature_model = model.module if isinstance(model, nn.DataParallel) else model
        with preserve_rng_state(), torch.no_grad():
            try:
                for sample_ids, data, label in trainloader:
                    data = data.to(self._device, non_blocking=True)
                    embedding = feature_model.extract_vector(data)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
                    id_list.append(sample_ids.cpu())
            finally:
                model.train(was_training)

        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        if return_ids:
            return torch.cat(id_list), embedding_list, label_list
        return embedding_list, label_list

    def _task_partition(self, data_manager, classes, task):
        ratio = float(self.args.get("val_ratio", 0.0) or 0.0)
        if ratio > 0:
            return data_manager.get_dataset_with_validation(
                classes, ratio, int(self.args["seed"]) + task,
            )
        return data_manager.get_dataset(classes, source="train", mode="train"), None

    @staticmethod
    def _partition_manifest(train, validation):
        return {
            "train": dataset_manifest(train),
            "validation": dataset_manifest(validation) if validation is not None else None,
        }

    def _validate_saved_partitions(self, data_manager):
        lower = 0
        for task, size in enumerate(self.task_sizes):
            if str(task) not in self.split_manifests and not self.resume_reproducible:
                # A legacy checkpoint cannot retroactively certify its old splits.
                lower += size
                continue
            train, validation = self._task_partition(
                data_manager, np.arange(lower, lower + size), task,
            )
            actual = self._partition_manifest(train, validation)
            if actual != self.split_manifests.get(str(task)):
                raise ValueError("Checkpoint dataset/split manifest differs at task {}.".format(task))
            lower += size
        self._pending_split_validation = False

    def _make_drift_loader(self, data_manager, training_dataset):
        # A single deterministic view of the TRAIN subset, reused on both sides.
        return DataLoader(
            data_manager.get_eval_view(training_dataset), batch_size=self.batch_size,
            shuffle=False, num_workers=0, pin_memory=self.pin_memory,
            generator=self._loader_generator("drift_workers"),
        )

    def incremental_train(self, data_manager):
        if getattr(self, "_pending_split_validation", False):
            self._validate_saved_partitions(data_manager)
        self._cur_task += 1
        
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
        train_dataset, val_dataset = self._task_partition(
            data_manager, current_classes, self._cur_task,
        )
        self.split_manifests[str(self._cur_task)] = self._partition_manifest(
            train_dataset, val_dataset,
        )
        if val_ratio > 0.0:
            # Evaluate old and new classes on validation during tuning, including
            # after CA; never route tuning metrics through the public test set.
            seen_validation = data_manager.get_seen_validation_dataset(
                self.task_sizes, val_ratio, self.args["seed"],
            )
            self.val_loader = DataLoader(
                seen_validation,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                generator=self._loader_generator("validation_workers"),
                worker_init_fn=seed_worker,
            )
            logging.info(
                "Task %d validation split: train=%d, validation=%d, ratio=%.4f",
                self._cur_task, len(train_dataset), len(val_dataset), val_ratio,
            )

        self.train_dataset = train_dataset
        print("The number of training dataset:", len(self.train_dataset))

        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=RandomSampler(
                train_dataset, generator=self._loader_generator("train_sampler"),
            ),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            generator=self._loader_generator("train_workers"),
            worker_init_fn=seed_worker,
        )
        self.test_loader = None
        if val_ratio == 0.0:
            test_dataset = data_manager.get_dataset(
                np.arange(0, self._total_classes), source="test", mode="test",
            )
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                generator=self._loader_generator("test_workers"),
                worker_init_fn=seed_worker,
            )
        self.evaluation_loader = self.val_loader if self.val_loader is not None else self.test_loader

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
            drift_loader = self._make_drift_loader(data_manager, train_dataset)
            before_features = self.extract_features(
                drift_loader, self._network, return_ids=True,
            )
            old_moments = snapshot_moments(
                self._class_means, self._class_covs, self._known_classes,
            )

        self._train(self.train_loader, self.evaluation_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

      
        if self._cur_task >0:
            after_features = self.extract_features(
                drift_loader, self._network, return_ids=True,
            )
            train_embeddings_old, train_embeddings_new, paired_ids = pair_features(
                before_features, after_features,
            )
            old_class_mean = self._class_means[:self._known_classes]
            gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 4.0)
            if self.args['ssca'] is True:
                old_class_mean +=gap
                self._class_means[:self._known_classes] = old_class_mean

        self._network.fc.backup()
        self._compute_class_mean(
            data_manager, check_diff=False, oracle=False, training_dataset=train_dataset,
        )
        diagnostic = None
        if self._cur_task > 0:
            diagnostic = {
                "task": self._cur_task,
                "drift_paired_sample_ids": paired_ids.tolist(),
                "statistics_train_count": len(train_dataset),
                "memory": memory_drift_diagnostics(
                    old_moments, self._class_means, self._class_covs,
                ),
                "pre_ca": self._record_classifier_alignment("pre_ca"),
                "ca_applied": bool(self.args['ca'] and self.args['ca_epochs'] > 0),
            }
            logging.info("Old memory drift diagnostic: %s", diagnostic["memory"])
        if self._cur_task>0 and self.args['ca_epochs']>0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
        if diagnostic is not None:
            diagnostic["post_ca"] = self._record_classifier_alignment("post_ca")
            self.task_diagnostics.append(diagnostic)

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

    def _build_optimizer(self):
        """Same non-metric groups for RSIAT and QKSR, at existing rates."""
        optimizer_name = self.args['optimizer']
        if self._cur_task == 0:
            adapter_lr = 0.01
            if optimizer_name == 'sgd':
                param_groups = [
                    {'params': self._network.convnet.blocks[-1].parameters(),
                     'lr': adapter_lr, 'weight_decay': self.weight_decay, 'group_name': 'last_block'},
                    {'params': self._network.convnet.blocks[:-1].parameters(),
                     'lr': adapter_lr, 'weight_decay': self.weight_decay, 'group_name': 'other_blocks'},
                    {'params': self._network.fc.parameters(),
                     'lr': adapter_lr, 'weight_decay': self.weight_decay, 'group_name': 'classifier'},
                ]
            else:
                # Keep the existing base AdamW network LR (init_lr); metric
                # groups retain their existing 0.01-based projector/metric LR.
                param_groups = [{'params': self._network.parameters(), 'lr': self.init_lr,
                                 'weight_decay': self.weight_decay, 'group_name': 'network'}]
        else:
            adapter_lr = self.init_lr
            param_groups = [
                {'params': self._network.convnet.parameters(), 'lr': adapter_lr,
                  'weight_decay': self.weight_decay, 'group_name': 'convnet'},
                {'params': self._network.fc.parameters(), 'lr': adapter_lr,
                  'weight_decay': self.weight_decay, 'group_name': 'classifier'},
                {'params': self.old_ae.parameters(), 'lr': self.args['ae_init_lr'],
                  'weight_decay': self.args['ae_weight_decay'], 'group_name': 'old_ae'},
            ]
        param_groups.extend(self._quantum_param_groups(adapter_lr))
        active_groups = []
        grouped_ids = []
        for group in param_groups:
            group['params'] = [p for p in group['params'] if p.requires_grad]
            if not group['params']:
                continue
            group.setdefault('group_name', 'metric_' + group.get('q_group', 'parameters'))
            active_groups.append(group)
            grouped_ids.extend(id(p) for p in group['params'])
        expected = [p for p in self._network.parameters() if p.requires_grad]
        if self._cur_task > 0:
            expected.extend(p for p in self.old_ae.parameters() if p.requires_grad)
        if self._quantum_trainable_this_task():
            expected.extend(p for p in self.quantum_kernel.parameters() if p.requires_grad)
        if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != {id(p) for p in expected}:
            raise ValueError("Optimizer groups omit or duplicate active parameters.")
        logging.info("Optimizer groups: %s", [
            {'name': g['group_name'], 'lr': g['lr'], 'weight_decay': g['weight_decay'],
             'parameters': sum(p.numel() for p in g['params'])} for g in active_groups
        ])
        if optimizer_name == 'sgd':
            return optim.SGD(active_groups, momentum=0.9, lr=self.init_lr)
        if optimizer_name == 'adam':
            return optim.AdamW(active_groups, lr=self.init_lr, weight_decay=self.weight_decay)
        raise ValueError("Unknown optimizer {}".format(optimizer_name))

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self._calibrate_quantum_kernel()
        self.tuned_epochs = self.args['init_epochs'] if self._cur_task == 0 else self.args['inc_epochs']
        optimizer = self._build_optimizer()

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr
        )
        log_count_parameter(optimizer.param_groups)
        self._init_train(
            train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch']
        )

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, warmup_epoch):
        quiet = self.args.get("quiet_task_logging", False)
        prog_bar = range(self.tuned_epochs) if quiet else tqdm(range(self.tuned_epochs))
        eval_interval = self.args.get("eval_interval", 0)
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        
        for _, epoch in enumerate(prog_bar):
            epoch_start = time.perf_counter()
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
                loss.backward()
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
            if not quiet:
                prog_bar.set_description(info)
            if not quiet and self.quantum_kernel is not None and self._quantum_used_this_task():
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
        return self.args["beta"] * loss_align + self.args["gamma"] * loss_orth
        
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
            return logits, loss_c, loss_base
        
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        loss_inc = self._inc_loss(features, features_old)
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
