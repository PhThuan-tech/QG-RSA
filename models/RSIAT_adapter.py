import copy
import logging
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
from KeepLora.keeplora_controller import KeepLoRAController
import math
num_workers = 8

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
        self.keeplora_init_mode = str(args.get("keeplora_init_mode", "cosine")).lower()
        if self.keeplora_init_mode not in ("cosine", "full_rsiat"):
            raise ValueError(
                "keeplora_init_mode must be 'cosine' or 'full_rsiat', got {!r}."
                .format(self.keeplora_init_mode)
            )

    def _after_load_checkpoint(self, checkpoint):
        """Restore learner-specific state after BaseLearner restores the network."""
        if self._cur_task >= 1:
            self.old_ae = AutoencoderSigmoid(
                input_dims=768,
                code_dims=self.args["ae_code_dims"],
                zero_residual=self.rae_zero_init,
            )
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

        self._network_module_ptr = self._network
        self.old_network_module_ptr = self._old_network

    def after_task(self):
        if self._uses_keeplora:
            self._finalize_keeplora_task()
        if self.old_ae is not None:
            self._rae_last_state_snapshot = self._rae_state_snapshot()
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network

    def _keeplora_convnet(self):
        network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        return network, network.convnet

    def _create_rae(self, task_id):
        self.old_ae = AutoencoderSigmoid(
            input_dims=768,
            code_dims=self.args["ae_code_dims"],
            zero_residual=self.rae_zero_init,
        ).to(self._device)
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
            if not self._uses_keeplora:
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
                actual_features, reference_features, rtol=1e-4, atol=1e-3
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
        
        self._prepare_rae_for_task()
            
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + task_size
        # self._network.update_fc(data_manager.get_task_size(self._cur_task)*4)
        self._network.update_fc(task_size)
        self._network_module_ptr = self._network
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
    
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")

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

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

      
        if self._cur_task >0:
            self._network.to(self._device)
            train_embeddings_old, _ = self.extract_features(self.train_loader, self._network, None)

        self._diagnose_rae_initialization(self.train_loader)
        self._train(self.train_loader, self.test_loader)
        
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

      
        if self._cur_task >0:
            train_embeddings_new, _ = self.extract_features(self.train_loader, self._network, None)
            old_class_mean = self._class_means[:self._known_classes]
            gap = self.displacement(train_embeddings_old, train_embeddings_new, old_class_mean, 4.0)
            if self.args['ssca'] is True:
                old_class_mean +=gap
                self._class_means[:self._known_classes] = old_class_mean

        self._network.fc.backup()
        self._compute_class_mean(data_manager, check_diff=False, oracle=False)
        if self._cur_task>0 and self.args['ca_epochs']>0 and self.args['ca'] is True:
            self._stage2_compact_classifier(task_size, self.args['ca_epochs'])
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._uses_keeplora:
            self._initialize_keeplora_task(self.keeplora_accum_loader)
        if self._cur_task == 0:
            self.tuned_epochs = self.args["init_epochs"]
            param_groups = [
                {'params': self._network.convnet.blocks[-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.convnet.blocks[:-1].parameters(), 'lr': 0.01,
                 'weight_decay': self.args['weight_decay']},
                {'params': self._network.fc.parameters(), 'lr': 0.01, 'weight_decay': self.args['weight_decay']}
            ]

            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
                
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])
        else:
            self.tuned_epochs = self.args['inc_epochs']
            param_groups = []
            param_groups.append(
                {'params': self._network.convnet.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay})
            param_groups.append(
                {'params': self._network.fc.parameters(), 'lr': self.init_lr, 'weight_decay': self.weight_decay})
            param_groups.append(
                {'params': self.old_ae.parameters(), 'lr': self.args['ae_init_lr'], 'weight_decay': self.args['ae_weight_decay']})
            
            if self.args['optimizer'] == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9)
            elif self.args['optimizer'] == 'adam':
                optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.tuned_epochs, eta_min=self.min_lr)
            log_count_parameter(param_groups)
            self._init_train(train_loader, test_loader, optimizer, scheduler, self.args['warmup_epoch'])

    def _init_train(self, train_loader, test_loader, optimizer, scheduler, warmup_epoch):
        prog_bar = tqdm(range(self.tuned_epochs))
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
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_c += loss_c.item()
                losses_rt += loss_rt.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            scheduler.step()

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
        logging.info(info)

    def _inc_loss(self, features, features_old):
        loss_align, loss_orth = self._inc_loss_components(features, features_old)
        return self.args["beta"] * loss_align + self.args["gamma"] * loss_orth

    def _inc_loss_components(self, features, features_old):
        features_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        features_old_norm = F.normalize(features_old, p=2, dim=1)
        protos = torch.from_numpy(self._class_means).float().to(self._device,non_blocking=True)
        protos = self.old_ae(protos)
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
