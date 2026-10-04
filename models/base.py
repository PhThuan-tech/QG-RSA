import copy
import logging
import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import time
EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    @staticmethod
    def _should_eval_epoch(epoch, total_epochs, interval):
        """Evaluate on the final epoch and optionally at a fixed interval."""
        if epoch == total_epochs - 1:
            return True
        if interval is None or interval <= 0:
            return False
        return (epoch + 1) % interval == 0


    def _stage2_compact_classifier(self, task_size, ca_epochs=5):
        for p in self._network.fc.parameters():
            p.requires_grad = True

        run_epochs = ca_epochs
        crct_num = self._total_classes
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': self.init_lr,
                           'weight_decay': self.weight_decay}]

        optimizer = optim.SGD(network_params, lr=self.init_lr, momentum=0.9, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval()

        eval_interval = self.args.get("ca_eval_interval", 0)
        for epoch in range(run_epochs):
            losses = 0.
            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256

            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device) * (
                            0.9 + decay)

                cls_cov = self._class_covs[c_id].to(self._device)
                m = MultivariateNormal(cls_mean.float(), cls_cov.float())
                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([c_id] * num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
            sampled_label = torch.tensor(sampled_label).long().to(self._device)
            inputs = sampled_data
            targets = sampled_label
            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]

                # -stage two only use classifiers
                outputs = self._network.ca_forward(inp)
                logits = self.args['scale'] * outputs['logits']

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task + 1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]

                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)
                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if self._should_eval_epoch(epoch, run_epochs, eval_interval):
                test_acc_msg = "{:.3f}".format(
                    self._compute_accuracy(self._network, self.test_loader)
                )
            else:
                test_acc_msg = "skipped"
            info = 'CA Task {} => Loss {:.3f}, Test_accy {}'.format(
                self._cur_task, losses / self._total_classes, test_acc_msg)
            logging.info(info)


    def _checkpoint_run_metadata(self):
        """Return the fields that must not change when resuming a run."""
        keys = (
            "dataset", "seed", "init_cls", "increment", "model_name", "convnet_type",
            "ffn_num", "keeplora_rank", "keeplora_alpha", "keeplora_targets",
            "keeplora_weight_threshold", "keeplora_feature_threshold",
            "keeplora_feature_samples", "keeplora_grad_batches", "keeplora_feature_batches",
        )
        return {
            "format_version": 2,
            "keeplora_init_mode": str(
                self.args.get("keeplora_init_mode", "cosine")
            ).lower(),
            "rae_zero_init": bool(self.args.get("rae_zero_init", False)),
            "rae_lifecycle": str(self.args.get("rae_lifecycle", "shared")).lower(),
            **{key: self.args.get(key) for key in keys if key in self.args},
        }

    def save_checkpoint(self, filepath):
        """Persist the completed task state needed to start the next task.

        Checkpoints intentionally live at task boundaries. Optimizers and learning
        rate schedulers are recreated for each task, whereas class statistics,
        dynamic classifier heads, the frozen teacher, and the autoencoder must be
        restored exactly.
        """
        network = (
            self._network.module
            if isinstance(self._network, nn.DataParallel)
            else self._network
        )
        checkpoint = {
            "run_metadata": self._checkpoint_run_metadata(),
            "cur_task": self._cur_task,
            "known_classes": self._known_classes,
            "total_classes": self._total_classes,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in network.state_dict().items()
            },
        }

        if hasattr(self, "_class_means") and self._class_means is not None:
            checkpoint["class_means"] = torch.as_tensor(self._class_means).cpu()

        if hasattr(self, "_class_covs") and self._class_covs is not None:
            if self.args.get("compact_diagonal_checkpoint", False):
                checkpoint["class_variances"] = torch.diagonal(
                    self._class_covs.detach().cpu(), dim1=-2, dim2=-1
                )
            else:
                checkpoint["class_covs"] = self._class_covs.detach().cpu()

        if hasattr(self, "task_sizes"):
            checkpoint["task_sizes"] = list(self.task_sizes)
        if hasattr(self, "class_order"):
            checkpoint["class_order"] = list(self.class_order)
        if hasattr(self, "radius"):
            radius = self.radius
            checkpoint["radius"] = radius.detach().cpu() if torch.is_tensor(radius) else radius
        if hasattr(self, "old_ae") and self.old_ae is not None:
            checkpoint["old_ae_state_dict"] = {
                key: value.detach().cpu()
                for key, value in self.old_ae.state_dict().items()
            }
            checkpoint["rae_generation"] = int(getattr(self, "rae_generation", 0))
            checkpoint["rae_task_id"] = getattr(self, "rae_task_id", None)
        if hasattr(self, "cnn_curve"):
            checkpoint["cnn_curve"] = self.cnn_curve
        if hasattr(self, "nme_curve"):
            checkpoint["nme_curve"] = self.nme_curve

        checkpoint_dir = os.path.dirname(filepath)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        temporary_path = filepath + ".tmp"
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, filepath)
        logging.info("Saved checkpoint: %s", filepath)

    def _infer_task_sizes(self, checkpoint):
        if "task_sizes" in checkpoint:
            return list(checkpoint["task_sizes"])

        remaining = int(checkpoint["total_classes"])
        init_cls = self.args.get("init_cls", remaining)
        increment = self.args.get("increment", init_cls)
        task_sizes = []
        while remaining > 0:
            task_size = min(init_cls if not task_sizes else increment, remaining)
            task_sizes.append(task_size)
            remaining -= task_size
        return task_sizes

    @staticmethod
    def _rebuild_classifier(network, task_sizes):
        network.fc = None
        for task_size in task_sizes:
            network.update_fc(task_size)

    def _validate_checkpoint(self, checkpoint):
        saved_metadata = checkpoint.get("run_metadata", {})
        current_metadata = self._checkpoint_run_metadata()
        if saved_metadata.get("format_version", 1) >= 2:
            optional_metadata_keys = {
                "keeplora_init_mode", "rae_zero_init", "rae_lifecycle"
            }
            missing_metadata = [
                key for key in current_metadata
                if key != "format_version"
                and key not in optional_metadata_keys
                and key not in saved_metadata
            ]
            if missing_metadata:
                raise ValueError(
                    "Checkpoint is missing run metadata: {}".format(missing_metadata)
                )
        mismatches = [
            "{} (checkpoint={!r}, current={!r})".format(
                key, saved_metadata[key], current_metadata[key]
            )
            for key in current_metadata
            if key != "format_version"
            and key in saved_metadata
            and saved_metadata[key] != current_metadata[key]
        ]
        if mismatches:
            raise ValueError("Checkpoint does not match this experiment: " + ", ".join(mismatches))

        expected_order = getattr(self, "class_order", None)
        saved_order = checkpoint.get("class_order")
        if expected_order is not None and saved_order is not None:
            if list(expected_order) != list(saved_order):
                raise ValueError("Checkpoint class order does not match the current DataManager.")

    def load_checkpoint(self, filepath):
        try:
            checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before weights_only existed
            checkpoint = torch.load(filepath, map_location="cpu")

        required_keys = {"cur_task", "known_classes", "total_classes", "model_state_dict"}
        missing_keys = required_keys - set(checkpoint)
        if missing_keys:
            raise ValueError(
                "Unsupported checkpoint {}: missing {}".format(filepath, sorted(missing_keys))
            )
        self._validate_checkpoint(checkpoint)

        self._cur_task = int(checkpoint["cur_task"])
        self._known_classes = int(checkpoint["known_classes"])
        self._total_classes = int(checkpoint["total_classes"])
        self.task_sizes = self._infer_task_sizes(checkpoint)

        network = (
            self._network.module
            if isinstance(self._network, nn.DataParallel)
            else self._network
        )
        self._rebuild_classifier(network, self.task_sizes)
        missing_keys, unexpected_keys = network.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if missing_keys:
            logging.warning("Missing keys while loading checkpoint: %s", missing_keys)
        if unexpected_keys:
            logging.warning("Unexpected keys while loading checkpoint: %s", unexpected_keys)

        if "class_means" in checkpoint:
            self._class_means = checkpoint["class_means"].cpu().numpy()
        if "class_covs" in checkpoint:
            self._class_covs = checkpoint["class_covs"].cpu()
        elif "class_variances" in checkpoint:
            self._class_covs = torch.diag_embed(checkpoint["class_variances"].cpu())
        if "radius" in checkpoint:
            radius = checkpoint["radius"]
            self.radius = radius.item() if torch.is_tensor(radius) else radius
        if "cnn_curve" in checkpoint:
            self.cnn_curve = checkpoint["cnn_curve"]
        if "nme_curve" in checkpoint:
            self.nme_curve = checkpoint["nme_curve"]
        if "class_order" in checkpoint:
            self.class_order = list(checkpoint["class_order"])

        self._network = network.to(self._device)
        self._old_network = copy.deepcopy(self._network).freeze()
        self._old_network.to(self._device)

        if hasattr(self, "_after_load_checkpoint"):
            self._after_load_checkpoint(checkpoint)

        logging.info(
            "Loaded checkpoint %s; completed task=%d, known_classes=%d, total_classes=%d",
            filepath, self._cur_task, self._known_classes, self._total_classes,
        )
        return self._cur_task

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        grouped = {k: float(v) for k, v in grouped.items()}
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = float(np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        ))

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)
        return cnn_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device, non_blocking=True)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]


    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        for _, _inputs, _targets in loader:
            _targets = _targets.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device, non_blocking=True))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device, non_blocking=True))
                )

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _compute_class_mean(self, data_manager, check_diff=False, oracle=False):
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self._known_classes
            new_class_means = np.zeros((self._total_classes, self.feature_dim))
            new_class_means[:self._known_classes] = self._class_means
            self._class_means = new_class_means
            new_class_cov = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self._known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = np.zeros((self._total_classes, self.feature_dim))
            self._class_covs = torch.zeros((self._total_classes, self.feature_dim, self.feature_dim))

        radius = []
        for class_idx in range(self._known_classes, self._total_classes):

            data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1), source='train',
                                                                  mode='test', ret_data=True)
            stats_workers = int(self.args.get("stats_num_workers", 4))
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=stats_workers,
                pin_memory=self.args.get("pin_memory", self._device.type == "cuda"),
                persistent_workers=stats_workers > 0 and self.args.get("persistent_workers", True),
            )
            vectors, _ = self._extract_vectors(idx_loader)
            class_mean = np.mean(vectors, axis=0)
            if self._cur_task == 0:
                cov = np.cov(vectors.T)+ np.eye(class_mean.shape[-1]) * 1e-4
                radius.append(np.trace(cov) /768)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-3

            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov

        if self._cur_task == 0:
                self.radius = np.sqrt(np.mean(radius))
                print(self.radius)

    def displacement_cov(self, Y, class_mean, embedding_old, sigma):
        cov = None
        start_time = time.time()
        for _class in range(self._known_classes):
            loop_start_time = time.time()
            DY = self.cov_computation(Y, class_mean[_class])
            distance = np.sum((np.tile(Y[None, :, :], [1, 1, 1]) - np.tile(
                embedding_old[_class, None, :], [1, Y.shape[0], 1])) ** 2, axis=2)
            W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
            W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
            if cov is None:
                cov = np.sum(np.tile(W_norm[:, :, None, None], [
                    1, 1, DY.shape[1], DY.shape[2]]) * np.tile(DY[None, :, :, :], [W.shape[0], 1, 1, 1]), axis=1)
            else:
                displacement = np.sum(np.tile(W_norm[:, :, None, None], [
                    1, 1, DY.shape[1], DY.shape[2]]) * np.tile(DY[None, :, :, :], [W.shape[0], 1, 1, 1]), axis=1)
                cov = np.concatenate((cov, displacement))
            loop_end_time = time.time()
            print("single loop time: ", loop_end_time - loop_start_time)
        end_time = time.time()
        print("total loop time: ", end_time - start_time)

        cov = torch.tensor(cov)
        return cov

    def displacement(self, Y1, Y2, embedding_old, sigma):
        DY = Y2 - Y1
        distance = np.sum((np.tile(Y1[None, :, :], [embedding_old.shape[0], 1, 1]) - np.tile(
            embedding_old[:, None, :], [1, Y1.shape[0], 1])) ** 2, axis=2)
        W = np.exp(-distance / (2 * sigma ** 2)) + 1e-5
        W_norm = W / np.tile(np.sum(W, axis=1)[:, None], [1, W.shape[1]])
        displacement = np.sum(np.tile(W_norm[:, :, None], [
            1, 1, DY.shape[1]]) * np.tile(DY[None, :, :], [W.shape[0], 1, 1]), axis=1)
        return displacement
