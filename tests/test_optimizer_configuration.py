import json
import pathlib
import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from models.RSIAT_adapter import Learner
from models.base import BaseLearner
from network.classifier import SimpleContinualLinear


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Module()
        self.convnet.blocks = nn.ModuleList(
            [nn.Linear(4, 4), nn.Linear(4, 4)]
        )
        self.fc = nn.Linear(4, 2)


class OptimizerConfigurationTests(unittest.TestCase):
    def make_learner(self, optimizer_name, incremental=False):
        network = TinyNetwork()
        network.convnet.blocks[0].weight.requires_grad_(False)
        old_ae = nn.Linear(4, 4) if incremental else None
        learner = SimpleNamespace(
            args={
                "optimizer": optimizer_name,
                "adamw_lr": 1e-3,
                "adamw_weight_decay": 0.01,
                "adamw_betas": [0.9, 0.999],
                "adamw_eps": 1e-8,
                "adamw_rae_lr": 5e-4,
                "adamw_rae_weight_decay": 0.002,
            },
            init_lr=0.018,
            weight_decay=0.0002,
            _network=network,
            old_ae=old_ae,
            _cur_task=1 if incremental else 0,
        )
        learner._build_main_optimizer = MethodType(
            Learner._build_main_optimizer, learner
        )
        learner._rae_optimizer_mode = MethodType(
            Learner._rae_optimizer_mode, learner
        )
        learner._build_incremental_optimizers = MethodType(
            Learner._build_incremental_optimizers, learner
        )
        learner._build_rae_sgd_optimizer = MethodType(
            Learner._build_rae_sgd_optimizer, learner
        )
        learner._classifier_optimizer_mode = MethodType(
            Learner._classifier_optimizer_mode, learner
        )
        learner._build_initial_optimizers = MethodType(
            Learner._build_initial_optimizers, learner
        )
        learner._build_incremental_optimizers_with_separate_classifier = MethodType(
            Learner._build_incremental_optimizers_with_separate_classifier,
            learner,
        )
        learner._build_separate_classifier_optimizer = MethodType(
            Learner._build_separate_classifier_optimizer, learner
        )
        learner._build_classifier_sgd_optimizer = MethodType(
            Learner._build_classifier_sgd_optimizer, learner
        )
        learner._build_classifier_adamw_optimizer = MethodType(
            Learner._build_classifier_adamw_optimizer, learner
        )
        learner._assert_optimizer_ownership = MethodType(
            Learner._assert_optimizer_ownership, learner
        )
        learner._trainable_parameter_ids = Learner._trainable_parameter_ids
        learner._optimizer_parameter_ids = Learner._optimizer_parameter_ids
        if incremental:
            groups = [
                {
                    "name": "backbone",
                    "params": network.convnet.parameters(),
                    "lr": learner.init_lr,
                    "weight_decay": learner.weight_decay,
                },
                {
                    "name": "classifier",
                    "params": network.fc.parameters(),
                    "lr": learner.init_lr,
                    "weight_decay": learner.weight_decay,
                },
                {
                    "name": "rae",
                    "params": old_ae.parameters(),
                    "lr": 5e-4,
                    "weight_decay": 0.002,
                },
            ]
        else:
            groups = [
                {
                    "name": "backbone_last_block",
                    "params": network.convnet.blocks[-1].parameters(),
                    "lr": 0.01,
                    "weight_decay": learner.weight_decay,
                },
                {
                    "name": "backbone_other_blocks",
                    "params": network.convnet.blocks[:-1].parameters(),
                    "lr": 0.01,
                    "weight_decay": learner.weight_decay,
                },
                {
                    "name": "classifier",
                    "params": network.fc.parameters(),
                    "lr": 0.01,
                    "weight_decay": learner.weight_decay,
                },
            ]
        return learner, groups

    def test_sgd_preserves_initial_and_incremental_group_recipes(self):
        learner, groups = self.make_learner("sgd")
        optimizer = learner._build_main_optimizer(groups, incremental=False)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [0.01] * 3)
        self.assertTrue(all(g["momentum"] == 0.9 for g in optimizer.param_groups))
        self.assertTrue(
            all(g["weight_decay"] == learner.weight_decay for g in optimizer.param_groups)
        )

        learner, groups = self.make_learner("sgd", incremental=True)
        optimizer = learner._build_main_optimizer(groups, incremental=True)
        self.assertEqual(
            [g["lr"] for g in optimizer.param_groups],
            [learner.init_lr, learner.init_lr, 5e-4],
        )
        self.assertTrue(all(g["momentum"] == 0.9 for g in optimizer.param_groups))
        self.assertIn(
            id(learner.old_ae.weight),
            {id(p) for group in optimizer.param_groups for p in group["params"]},
        )

    def test_m_only_adamw_uses_one_main_optimizer_with_module_specific_groups(self):
        learner, groups = self.make_learner("sgd", incremental=True)
        learner._uses_m_only = True
        learner.args.update(
            incremental_optimizer="adamw",
            shared_adapter_m_only_adamw_keeplora_lr=2e-3,
            shared_adapter_m_only_adamw_keeplora_weight_decay=0.02,
            shared_adapter_m_only_adamw_classifier_lr=3e-3,
            shared_adapter_m_only_adamw_classifier_weight_decay=0.03,
            shared_adapter_m_only_adamw_rae_lr=4e-4,
            shared_adapter_m_only_adamw_rae_weight_decay=0.004,
        )

        optimizer, rae_optimizer = learner._build_incremental_optimizers(groups)

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsNone(rae_optimizer)
        by_name = {group["name"]: group for group in optimizer.param_groups}
        self.assertEqual(
            [by_name[name]["lr"] for name in ("backbone", "classifier", "rae")],
            [2e-3, 3e-3, 4e-4],
        )
        self.assertEqual(
            [
                by_name[name]["weight_decay"]
                for name in ("backbone", "classifier", "rae")
            ],
            [0.02, 0.03, 0.004],
        )

    def test_m_only_sgd_uses_one_main_optimizer_and_configured_momentum(self):
        learner, groups = self.make_learner("sgd", incremental=True)
        learner._uses_m_only = True
        learner.args.update(
            incremental_optimizer="sgd",
            shared_adapter_m_only_sgd_momentum=0.8,
        )

        optimizer, rae_optimizer = learner._build_incremental_optimizers(groups)

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertIsNone(rae_optimizer)
        self.assertTrue(
            all(group["momentum"] == 0.8 for group in optimizer.param_groups)
        )

    def test_m_only_optimizer_choice_does_not_change_task_zero_baseline(self):
        learner, groups = self.make_learner("sgd", incremental=False)
        learner._uses_m_only = True
        learner.args["incremental_optimizer"] = "adamw"

        optimizer = learner._build_main_optimizer(groups, incremental=False)

        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertTrue(
            all(group["momentum"] == 0.9 for group in optimizer.param_groups)
        )

    def test_m_only_ca_stage_keeps_baseline_sgd_for_both_incremental_options(self):
        class ClassifierNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 2)

            def ca_forward(self, inputs):
                return {"logits": self.fc(inputs)}

        original_sgd = torch.optim.SGD
        for incremental_optimizer in ("adamw", "sgd"):
            learner = object.__new__(BaseLearner)
            learner.args = {
                "shared_adapter_update_mode": "keeplora_m_only",
                "incremental_optimizer": incremental_optimizer,
                "scale": 1.0,
                "ca_eval_interval": 0,
            }
            learner._network = ClassifierNetwork()
            learner._multiple_gpus = []
            learner._device = torch.device("cpu")
            learner._total_classes = 2
            learner._cur_task = 1
            learner.init_lr = 0.01
            learner.weight_decay = 0.001

            observed = []

            def make_sgd(*args, **kwargs):
                optimizer = original_sgd(*args, **kwargs)
                observed.append(optimizer)
                return optimizer

            with patch("models.base.optim.SGD", side_effect=make_sgd):
                BaseLearner._stage2_compact_classifier(
                    learner, task_size=2, ca_epochs=0
                )

            self.assertEqual(len(observed), 1)
            self.assertIsInstance(observed[0], original_sgd)
            self.assertEqual(observed[0].param_groups[0]["momentum"], 0.9)
            self.assertEqual(observed[0].param_groups[0]["lr"], 0.01)

    def test_sgd_initial_updates_match_the_original_optimizer_construction(self):
        torch.manual_seed(53)
        learner, groups = self.make_learner("sgd")
        optimizer = learner._build_main_optimizer(groups, incremental=False)

        torch.manual_seed(53)
        reference, _ = self.make_learner("sgd")
        reference_groups = [
            {
                "params": reference._network.convnet.blocks[-1].parameters(),
                "lr": 0.01,
                "weight_decay": reference.args.get(
                    "weight_decay", reference.weight_decay
                ),
            },
            {
                "params": reference._network.convnet.blocks[:-1].parameters(),
                "lr": 0.01,
                "weight_decay": reference.args.get(
                    "weight_decay", reference.weight_decay
                ),
            },
            {
                "params": reference._network.fc.parameters(),
                "lr": 0.01,
                "weight_decay": reference.args.get(
                    "weight_decay", reference.weight_decay
                ),
            },
        ]
        reference_optimizer = torch.optim.SGD(
            reference_groups,
            momentum=0.9,
            lr=learner.init_lr,
            weight_decay=learner.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=2, eta_min=0
        )
        reference_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            reference_optimizer, T_max=2, eta_min=0
        )
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            loss = sum(
                parameter.square().sum()
                for parameter in learner._network.parameters()
                if parameter.requires_grad
            )
            reference_loss = sum(
                parameter.square().sum()
                for parameter in reference._network.parameters()
                if parameter.requires_grad
            )
            loss.backward()
            reference_loss.backward()
            optimizer.step()
            reference_optimizer.step()
            scheduler.step()
            reference_scheduler.step()

        for (name, parameter), (reference_name, reference_parameter) in zip(
            learner._network.named_parameters(),
            reference._network.named_parameters(),
        ):
            self.assertEqual(name, reference_name)
            self.assertTrue(torch.equal(parameter, reference_parameter), name)

    def test_adamw_initial_groups_use_config_and_skip_frozen_parameters(self):
        learner, groups = self.make_learner("adamw")
        optimizer = learner._build_main_optimizer(groups, incremental=False)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [1e-3] * 3)
        self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [0.01] * 3)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(optimizer.defaults["eps"], 1e-8)
        self.assertNotIn(
            id(learner._network.convnet.blocks[0].weight),
            {id(p) for group in optimizer.param_groups for p in group["params"]},
        )

    def test_adamw_incremental_groups_include_and_update_rae(self):
        learner, groups = self.make_learner("adamw", incremental=True)
        optimizer = learner._build_main_optimizer(groups, incremental=True)
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [1e-3, 1e-3, 5e-4])
        self.assertEqual(
            [g["weight_decay"] for g in optimizer.param_groups],
            [0.01, 0.01, 0.002],
        )
        trainable = [
            p for group in optimizer.param_groups for p in group["params"]
            if p.requires_grad
        ]
        frozen = learner._network.convnet.blocks[0].weight.detach().clone()
        for parameter in trainable:
            parameter.data.fill_(1.0)
        before = [p.detach().clone() for p in trainable]
        optimizer.zero_grad(set_to_none=True)
        sum(p.square().sum() for p in trainable).backward()
        optimizer.step()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                and not torch.equal(p, old)
                for p, old in zip(trainable, before)
            )
        )
        self.assertTrue(
            torch.equal(learner._network.convnet.blocks[0].weight, frozen)
        )
        self.assertIn(id(learner.old_ae.weight), {id(p) for p in trainable})

    def test_adamw_backbone_and_sgd_momentum_rae_use_separate_optimizers(self):
        learner, groups = self.make_learner("adamw", incremental=True)
        learner.args["rae_optimizer"] = "sgd"
        learner.args["rae_sgd_momentum"] = 0.9

        main_optimizer, rae_optimizer = (
            learner._build_incremental_optimizers(groups)
        )

        self.assertIsInstance(main_optimizer, torch.optim.AdamW)
        self.assertIsInstance(rae_optimizer, torch.optim.SGD)
        self.assertEqual(
            [group["name"] for group in main_optimizer.param_groups],
            ["backbone", "classifier"],
        )
        self.assertEqual(len(rae_optimizer.param_groups), 1)
        rae_group = rae_optimizer.param_groups[0]
        self.assertEqual(rae_group["name"], "rae")
        self.assertEqual(rae_group["lr"], 5e-4)
        self.assertEqual(rae_group["weight_decay"], 0.002)
        self.assertEqual(rae_group["momentum"], 0.9)

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
        self.assertFalse(main_ids & rae_ids)
        self.assertEqual(
            main_ids | rae_ids,
            {
                id(parameter)
                for parameter in learner._network.parameters()
                if parameter.requires_grad
            }
            | {
                id(parameter)
                for parameter in learner.old_ae.parameters()
                if parameter.requires_grad
            },
        )

        all_parameters = [
            parameter
            for group in main_optimizer.param_groups + rae_optimizer.param_groups
            for parameter in group["params"]
        ]
        before = [parameter.detach().clone() for parameter in all_parameters]
        main_optimizer.zero_grad(set_to_none=True)
        rae_optimizer.zero_grad(set_to_none=True)
        sum(parameter.square().sum() for parameter in all_parameters).backward()
        main_optimizer.step()
        rae_optimizer.step()
        self.assertTrue(
            all(
                parameter.grad is not None and not torch.equal(parameter, old)
                for parameter, old in zip(all_parameters, before)
            )
        )

    def test_default_rae_optimizer_preserves_single_adamw(self):
        learner, groups = self.make_learner("adamw", incremental=True)
        optimizer, rae_optimizer = learner._build_incremental_optimizers(groups)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsNone(rae_optimizer)
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            ["backbone", "classifier", "rae"],
        )

    def test_classifier_optimizer_defaults_to_main_and_rejects_invalid_values(self):
        learner, _ = self.make_learner("adamw")
        self.assertEqual(learner._classifier_optimizer_mode(), "main")
        learner.args["classifier_optimizer"] = "adamw"
        self.assertEqual(learner._classifier_optimizer_mode(), "adamw")
        learner.args["classifier_optimizer"] = "invalid"
        with self.assertRaisesRegex(ValueError, "classifier_optimizer"):
            learner._classifier_optimizer_mode()

    def test_initial_classifier_adamw_is_separate_from_sgd_backbone(self):
        learner, groups = self.make_learner("sgd")
        learner.args["classifier_optimizer"] = "adamw"
        main_optimizer, classifier_optimizer = (
            learner._build_initial_optimizers(groups)
        )

        self.assertIsInstance(main_optimizer, torch.optim.SGD)
        self.assertIsInstance(classifier_optimizer, torch.optim.AdamW)
        self.assertEqual(classifier_optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(
            classifier_optimizer.param_groups[0]["weight_decay"], 0.01
        )
        self.assertEqual(
            classifier_optimizer.defaults["betas"], (0.9, 0.999)
        )
        main_ids = learner._optimizer_parameter_ids(main_optimizer)
        classifier_ids = learner._optimizer_parameter_ids(classifier_optimizer)
        self.assertFalse(main_ids & classifier_ids)
        self.assertEqual(
            classifier_ids,
            {id(parameter) for parameter in learner._network.fc.parameters()},
        )

    def test_incremental_classifier_adamw_is_separate_and_rae_stays_with_sgd(self):
        learner, groups = self.make_learner("sgd", incremental=True)
        learner.args["classifier_optimizer"] = "adamw"
        main_optimizer, rae_optimizer, classifier_optimizer = (
            learner._build_incremental_optimizers(groups)
        )

        self.assertIsInstance(main_optimizer, torch.optim.SGD)
        self.assertIsNone(rae_optimizer)
        self.assertIsInstance(classifier_optimizer, torch.optim.AdamW)
        main_ids = learner._optimizer_parameter_ids(main_optimizer)
        classifier_ids = learner._optimizer_parameter_ids(classifier_optimizer)
        self.assertFalse(main_ids & classifier_ids)
        self.assertIn(
            id(learner.old_ae.weight),
            main_ids,
            "The RSIAT baseline keeps RAE in its main SGD optimizer.",
        )
        self.assertEqual(
            main_ids | classifier_ids,
            {
                id(parameter)
                for parameter in learner._network.parameters()
                if parameter.requires_grad
            }
            | {
                id(parameter)
                for parameter in learner.old_ae.parameters()
                if parameter.requires_grad
            },
        )

    def test_initial_classifier_sgd_is_separate_and_matches_rsiat_recipe(self):
        learner, groups = self.make_learner("adamw")
        learner.args["classifier_optimizer"] = "sgd"
        main_optimizer, classifier_optimizer = (
            learner._build_initial_optimizers(groups)
        )

        self.assertIsInstance(main_optimizer, torch.optim.AdamW)
        self.assertIsInstance(classifier_optimizer, torch.optim.SGD)
        self.assertEqual(classifier_optimizer.defaults["momentum"], 0.9)
        self.assertFalse(classifier_optimizer.defaults["nesterov"])
        self.assertEqual(classifier_optimizer.param_groups[0]["lr"], 0.01)
        self.assertEqual(
            classifier_optimizer.param_groups[0]["weight_decay"],
            learner.weight_decay,
        )
        classifier_ids = learner._optimizer_parameter_ids(classifier_optimizer)
        main_ids = learner._optimizer_parameter_ids(main_optimizer)
        self.assertEqual(
            classifier_ids,
            {id(parameter) for parameter in learner._network.fc.parameters()},
        )
        self.assertFalse(main_ids & classifier_ids)
        self.assertEqual(
            main_ids | classifier_ids,
            {
                id(parameter)
                for parameter in learner._network.parameters()
                if parameter.requires_grad
            },
        )

    def test_incremental_raesgd_and_classifier_sgd_have_disjoint_full_coverage(self):
        learner, groups = self.make_learner("adamw", incremental=True)
        learner.args.update(
            {"rae_optimizer": "sgd", "classifier_optimizer": "sgd"}
        )
        main_optimizer, rae_optimizer, classifier_optimizer = (
            learner._build_incremental_optimizers(groups)
        )

        self.assertIsInstance(main_optimizer, torch.optim.AdamW)
        self.assertIsInstance(rae_optimizer, torch.optim.SGD)
        self.assertIsInstance(classifier_optimizer, torch.optim.SGD)
        self.assertEqual(
            classifier_optimizer.param_groups[0]["lr"], learner.init_lr
        )
        self.assertEqual(
            classifier_optimizer.param_groups[0]["weight_decay"],
            learner.weight_decay,
        )
        self.assertEqual(classifier_optimizer.defaults["momentum"], 0.9)
        groups_by_name = {
            "main": learner._optimizer_parameter_ids(main_optimizer),
            "rae": learner._optimizer_parameter_ids(rae_optimizer),
            "classifier": learner._optimizer_parameter_ids(classifier_optimizer),
        }
        self.assertFalse(groups_by_name["main"] & groups_by_name["rae"])
        self.assertFalse(groups_by_name["main"] & groups_by_name["classifier"])
        self.assertFalse(groups_by_name["rae"] & groups_by_name["classifier"])
        self.assertEqual(
            set().union(*groups_by_name.values()),
            {
                id(parameter)
                for parameter in learner._network.parameters()
                if parameter.requires_grad
            }
            | {
                id(parameter)
                for parameter in learner.old_ae.parameters()
                if parameter.requires_grad
            },
        )

    def test_classifier_sgd_is_independent_of_rae_optimizer_selection(self):
        learner, groups = self.make_learner("adamw", incremental=True)
        learner.args["classifier_optimizer"] = "sgd"
        optimizer_groups = [
            dict(group, params=list(group["params"])) for group in groups
        ]
        expected_ids = learner._trainable_parameter_ids(
            optimizer_groups
        )
        main_optimizer, rae_optimizer, classifier_optimizer = (
            learner._build_incremental_optimizers(optimizer_groups)
        )

        self.assertIsInstance(main_optimizer, torch.optim.AdamW)
        self.assertIsNone(rae_optimizer)
        self.assertIsInstance(classifier_optimizer, torch.optim.SGD)
        self.assertEqual(
            [group["name"] for group in main_optimizer.param_groups],
            ["backbone", "rae"],
        )
        main_ids = learner._optimizer_parameter_ids(main_optimizer)
        classifier_ids = learner._optimizer_parameter_ids(classifier_optimizer)
        self.assertFalse(main_ids & classifier_ids)
        self.assertEqual(main_ids | classifier_ids, expected_ids)

    def test_classifier_optimizer_tracks_parameters_after_task_expansion(self):
        learner, groups = self.make_learner("adamw")
        learner._network.fc = SimpleContinualLinear(4, 2)
        learner._network.fc.update(1, freeze_old=False)
        old_parameter = learner._network.fc.heads[0][0].weight
        new_parameter = learner._network.fc.heads[1][0].weight
        self.assertTrue(old_parameter.requires_grad)
        self.assertTrue(new_parameter.requires_grad)
        groups[-1]["params"] = learner._network.fc.parameters()
        learner.args["classifier_optimizer"] = "sgd"

        _, classifier_optimizer = learner._build_initial_optimizers(groups)
        classifier_ids = learner._optimizer_parameter_ids(classifier_optimizer)
        self.assertIn(id(old_parameter), classifier_ids)
        self.assertIn(id(new_parameter), classifier_ids)
        self.assertEqual(
            classifier_ids,
            {
                id(parameter)
                for parameter in learner._network.fc.parameters()
                if parameter.requires_grad
            },
        )

        learner, incremental_groups = self.make_learner(
            "adamw", incremental=True
        )
        learner._network.fc = SimpleContinualLinear(4, 2)
        learner._network.fc.update(1, freeze_old=False)
        old_parameter = learner._network.fc.heads[0][0].weight
        new_parameter = learner._network.fc.heads[1][0].weight
        incremental_groups[1]["params"] = learner._network.fc.parameters()
        learner.args.update(
            {"rae_optimizer": "sgd", "classifier_optimizer": "sgd"}
        )
        _, _, classifier_optimizer = learner._build_incremental_optimizers(
            incremental_groups
        )
        classifier_ids = learner._optimizer_parameter_ids(classifier_optimizer)
        self.assertIn(id(old_parameter), classifier_ids)
        self.assertIn(id(new_parameter), classifier_ids)
        self.assertEqual(
            classifier_ids,
            {
                id(parameter)
                for parameter in learner._network.fc.parameters()
                if parameter.requires_grad
            },
        )

    def test_separate_rae_sgd_requires_adamw_backbone_optimizer(self):
        learner, _ = self.make_learner("sgd", incremental=True)
        learner.args["rae_optimizer"] = "sgd"
        with self.assertRaisesRegex(ValueError, "requires optimizer='adamw'"):
            learner._rae_optimizer_mode()

    def test_incremental_training_steps_both_optimizers_and_schedulers(self):
        learner, groups = self.make_learner("adamw", incremental=True)
        learner.args.update(
            {"rae_optimizer": "sgd", "rae_sgd_momentum": 0.9, "eval_interval": 0}
        )
        main_optimizer, rae_optimizer = (
            learner._build_incremental_optimizers(groups)
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            main_optimizer, T_max=1, eta_min=0
        )
        rae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            rae_optimizer, T_max=1, eta_min=0
        )
        learner._device = torch.device("cpu")
        learner._cur_task = 1
        learner.tuned_epochs = 1
        learner.min_lr = 0
        learner._init_train = MethodType(Learner._init_train, learner)

        def compute_losses(self, inputs, targets, epoch, warmup_epoch):
            logits = self._network.fc(inputs)
            classification_loss = logits.square().mean()
            rae_loss = sum(
                parameter.square().mean()
                for parameter in self.old_ae.parameters()
            )
            return logits, classification_loss, rae_loss

        learner._compute_rt_loss = MethodType(compute_losses, learner)
        learner._should_eval_epoch = MethodType(
            lambda self, epoch, total_epochs, interval: False, learner
        )
        train_loader = [
            (None, torch.ones(2, 4), torch.tensor([0, 1]))
        ]

        learner._init_train(
            train_loader,
            [],
            main_optimizer,
            main_scheduler,
            warmup_epoch=0,
            rae_optimizer=rae_optimizer,
            rae_scheduler=rae_scheduler,
        )

        self.assertIn("step", main_optimizer.state[learner._network.fc.weight])
        self.assertIn("momentum_buffer", rae_optimizer.state[learner.old_ae.weight])
        self.assertEqual(main_scheduler.last_epoch, 1)
        self.assertEqual(rae_scheduler.last_epoch, 1)

    def test_classifier_sgd_training_updates_and_schedules_initial_and_incremental_tasks(self):
        learner, initial_groups = self.make_learner("adamw")
        learner.args.update({"classifier_optimizer": "sgd", "eval_interval": 0})
        main_optimizer, classifier_optimizer = (
            learner._build_initial_optimizers(initial_groups)
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            main_optimizer, T_max=1, eta_min=0
        )
        classifier_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            classifier_optimizer, T_max=1, eta_min=0
        )
        learner._device = torch.device("cpu")
        learner.tuned_epochs = 1
        learner.min_lr = 0
        learner._init_train = MethodType(Learner._init_train, learner)

        def initial_losses(self, inputs, targets, epoch, warmup):
            features = self._network.convnet.blocks[-1](inputs)
            logits = self._network.fc(features)
            return logits, logits.square().mean(), inputs.sum() * 0

        learner._compute_rt_loss = MethodType(initial_losses, learner)
        learner._should_eval_epoch = MethodType(
            lambda self, epoch, total_epochs, interval: False, learner
        )
        train_loader = [(None, torch.ones(2, 4), torch.tensor([0, 1]))]
        learner._init_train(
            train_loader,
            [],
            main_optimizer,
            main_scheduler,
            warmup_epoch=0,
            classifier_optimizer=classifier_optimizer,
            classifier_scheduler=classifier_scheduler,
        )

        classifier_parameter = learner._network.fc.weight
        self.assertIn(
            "momentum_buffer", classifier_optimizer.state[classifier_parameter]
        )
        self.assertIn(
            "step",
            main_optimizer.state[learner._network.convnet.blocks[-1].weight],
        )
        self.assertEqual(main_scheduler.last_epoch, 1)
        self.assertEqual(classifier_scheduler.last_epoch, 1)

        learner, incremental_groups = self.make_learner("adamw", incremental=True)
        learner.args.update(
            {
                "rae_optimizer": "sgd",
                "classifier_optimizer": "sgd",
                "eval_interval": 0,
            }
        )
        main_optimizer, rae_optimizer, classifier_optimizer = (
            learner._build_incremental_optimizers(incremental_groups)
        )
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=1, eta_min=0
            )
            for optimizer in (main_optimizer, rae_optimizer, classifier_optimizer)
        ]
        learner._device = torch.device("cpu")
        learner.tuned_epochs = 1
        learner.min_lr = 0
        learner._init_train = MethodType(Learner._init_train, learner)

        def incremental_losses(self, inputs, targets, epoch, warmup):
            features = self._network.convnet.blocks[-1](
                self._network.convnet.blocks[0](inputs)
            )
            logits = self._network.fc(features)
            classification = logits.square().mean()
            rae = sum(
                parameter.square().mean() for parameter in self.old_ae.parameters()
            )
            return logits, classification, rae

        learner._compute_rt_loss = MethodType(incremental_losses, learner)
        learner._should_eval_epoch = MethodType(
            lambda self, epoch, total_epochs, interval: False, learner
        )
        learner._init_train(
            train_loader,
            [],
            main_optimizer,
            schedulers[0],
            warmup_epoch=0,
            rae_optimizer=rae_optimizer,
            rae_scheduler=schedulers[1],
            classifier_optimizer=classifier_optimizer,
            classifier_scheduler=schedulers[2],
        )
        self.assertIn(
            "step",
            main_optimizer.state[learner._network.convnet.blocks[-1].weight],
        )
        self.assertIn(
            "momentum_buffer", rae_optimizer.state[learner.old_ae.weight]
        )
        self.assertIn(
            "momentum_buffer",
            classifier_optimizer.state[learner._network.fc.weight],
        )
        self.assertEqual(
            [scheduler.last_epoch for scheduler in schedulers], [1, 1, 1]
        )

    def test_duplicate_parameters_are_rejected(self):
        learner, groups = self.make_learner("adamw")
        groups.append(
            {
                "name": "duplicate",
                "params": [learner._network.fc.weight],
                "lr": 1e-3,
                "weight_decay": 0.01,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "duplicates"):
            learner._build_main_optimizer(groups, incremental=False)

    def test_optimizer_metadata_prevents_cross_optimizer_resume(self):
        sgd = SimpleNamespace(args={"optimizer": "sgd"})
        adamw_args = {
            "optimizer": "adamw",
            "adamw_lr": 1e-3,
            "adamw_weight_decay": 0.01,
            "adamw_betas": [0.9, 0.999],
            "adamw_eps": 1e-8,
        }
        saved = BaseLearner._checkpoint_run_metadata(sgd)
        current = SimpleNamespace(args=adamw_args, class_order=None)
        current._checkpoint_run_metadata = MethodType(
            BaseLearner._checkpoint_run_metadata, current
        )
        with self.assertRaisesRegex(ValueError, "optimizer"):
            BaseLearner._validate_checkpoint(
                current,
                {"run_metadata": saved},
            )

    def test_rae_optimizer_metadata_prevents_cross_optimizer_resume(self):
        base_args = {
            "optimizer": "adamw",
            "adamw_lr": 1e-3,
            "adamw_weight_decay": 0.01,
            "adamw_betas": [0.9, 0.999],
            "adamw_eps": 1e-8,
        }
        saved = BaseLearner._checkpoint_run_metadata(
            SimpleNamespace(args=base_args)
        )
        mixed_args = {
            **base_args,
            "rae_optimizer": "sgd",
            "rae_sgd_momentum": 0.9,
            "ae_init_lr": 5e-4,
            "ae_weight_decay": 0.002,
        }
        current = SimpleNamespace(args=mixed_args, class_order=None)
        current._checkpoint_run_metadata = MethodType(
            BaseLearner._checkpoint_run_metadata, current
        )
        with self.assertRaisesRegex(ValueError, "rae_optimizer"):
            BaseLearner._validate_checkpoint(
                current, {"run_metadata": saved}
            )

    def test_classifier_optimizer_metadata_distinguishes_runs_and_reads_legacy(self):
        legacy = BaseLearner._checkpoint_run_metadata(
            SimpleNamespace(args={"optimizer": "adamw"})
        )
        self.assertEqual(legacy["classifier_optimizer"], "main")
        legacy.pop("classifier_optimizer")

        default_run = SimpleNamespace(
            args={"optimizer": "adamw"}, class_order=None
        )
        default_run._checkpoint_run_metadata = MethodType(
            BaseLearner._checkpoint_run_metadata, default_run
        )
        BaseLearner._validate_checkpoint(
            default_run,
            {"run_metadata": legacy},
        )

        classifier_sgd_args = {
            "optimizer": "adamw",
            "classifier_optimizer": "sgd",
        }
        classifier_sgd_metadata = BaseLearner._checkpoint_run_metadata(
            SimpleNamespace(args=classifier_sgd_args)
        )
        self.assertEqual(classifier_sgd_metadata["classifier_sgd_momentum"], 0.9)
        classifier_sgd_run = SimpleNamespace(
            args=classifier_sgd_args, class_order=None
        )
        classifier_sgd_run._checkpoint_run_metadata = MethodType(
            BaseLearner._checkpoint_run_metadata, classifier_sgd_run
        )
        with self.assertRaisesRegex(ValueError, "classifier_optimizer"):
            BaseLearner._validate_checkpoint(
                classifier_sgd_run,
                {"run_metadata": legacy},
            )
        with self.assertRaisesRegex(ValueError, "classifier_optimizer"):
            BaseLearner._validate_checkpoint(
                default_run,
                {"run_metadata": classifier_sgd_metadata},
            )

        classifier_adamw_args = {
            "optimizer": "sgd",
            "classifier_optimizer": "adamw",
            "adamw_lr": 1e-3,
            "adamw_weight_decay": 0.01,
            "adamw_betas": [0.9, 0.999],
            "adamw_eps": 1e-8,
        }
        classifier_adamw_metadata = BaseLearner._checkpoint_run_metadata(
            SimpleNamespace(args=classifier_adamw_args)
        )
        self.assertEqual(classifier_adamw_metadata["classifier_adamw_lr"], 1e-3)
        classifier_adamw_run = SimpleNamespace(
            args=classifier_adamw_args, class_order=None
        )
        classifier_adamw_run._checkpoint_run_metadata = MethodType(
            BaseLearner._checkpoint_run_metadata, classifier_adamw_run
        )
        with self.assertRaisesRegex(ValueError, "classifier_optimizer"):
            BaseLearner._validate_checkpoint(
                classifier_adamw_run,
                {"run_metadata": classifier_sgd_metadata},
            )

    def test_new_classifier_configs_only_change_run_id_and_optimizer(self):
        cases = (
            (
                "keeplora_adamw_sgd_rae.json",
                "keeplora_adamw_sgd_rae_sgd_classifier.json",
            ),
            (
                "keeplora_adamw_mlp_sgd_rae.json",
                "keeplora_adamw_mlp_sgd_rae_sgd_classifier.json",
            ),
        )
        for old_name, new_name in cases:
            with self.subTest(config=new_name):
                old_config = json.loads(
                    (ROOT / "exps" / old_name).read_text(encoding="utf-8")
                )
                new_config = json.loads(
                    (ROOT / "exps" / new_name).read_text(encoding="utf-8")
                )
                self.assertEqual(new_config["classifier_optimizer"], "sgd")
                self.assertFalse(new_config["resume"])
                self.assertTrue(new_config["isolate_runs"])
                differences = {
                    key
                    for key in set(old_config) | set(new_config)
                    if old_config.get(key) != new_config.get(key)
                }
                self.assertEqual(
                    differences, {"prefix", "classifier_optimizer"}
                )
                self.assertNotEqual(new_config["prefix"], old_config["prefix"])

    def test_shared_adapter_adamw_classifier_config_preserves_baseline_recipe(self):
        baseline = json.loads(
            (ROOT / "exps" / "adapter_cifar224.json").read_text(
                encoding="utf-8"
            )
        )
        baseline_diagnostics = json.loads(
            (ROOT / "exps" / "adapter_cifar224_semantic_drift.json").read_text(
                encoding="utf-8"
            )
        )
        experiment = json.loads(
            (ROOT / "exps" / "adapter_sgd_adamw_classifier.json").read_text(
                encoding="utf-8"
            )
        )
        differences = {
            key
            for key in set(baseline) | set(experiment)
            if baseline.get(key) != experiment.get(key)
        }
        self.assertEqual(
            differences,
            {
                "prefix",
                "classifier_optimizer",
                "adamw_lr",
                "adamw_weight_decay",
                "adamw_betas",
                "adamw_eps",
                "semantic_drift",
            },
        )
        self.assertEqual(
            experiment["semantic_drift"],
            baseline_diagnostics["semantic_drift"],
        )
        self.assertTrue(experiment["semantic_drift"]["enabled"])
        self.assertEqual(experiment["optimizer"], "sgd")
        self.assertEqual(experiment["classifier_optimizer"], "adamw")
        self.assertFalse(experiment["resume"])
        self.assertTrue(experiment["isolate_runs"])

    def test_four_configs_are_isolated_and_paired(self):
        names = (
            "adapter_sgd.json",
            "keeplora_sgd.json",
            "adapter_adamw.json",
            "keeplora_adamw.json",
        )
        configs = {
            name: json.loads((ROOT / "exps" / name).read_text(encoding="utf-8"))
            for name in names
        }
        self.assertEqual(
            {config["prefix"] for config in configs.values()},
            {"adapter_sgd", "keeplora_sgd", "adapter_adamw", "keeplora_adamw"},
        )
        for config in configs.values():
            self.assertFalse(config["resume"])
            self.assertTrue(config["isolate_runs"])
        for adapter, keeplora in (
            (configs[names[0]], configs[names[1]]),
            (configs[names[2]], configs[names[3]]),
        ):
            for key in (
                "dataset", "seed", "init_cls", "increment", "init_epochs",
                "inc_epochs", "batch_size", "init_lr", "weight_decay",
                "ssca", "ca", "beta", "gamma",
            ):
                self.assertEqual(adapter[key], keeplora[key])
        self.assertEqual(
            [configs[name]["optimizer"] for name in names],
            ["sgd", "sgd", "adamw", "adamw"],
        )
        adam_adapter = configs["adapter_adamw.json"]
        adam_keeplora = configs["keeplora_adamw.json"]
        for key in (
            "adamw_lr", "adamw_weight_decay", "adamw_betas", "adamw_eps",
            "adamw_rae_lr", "adamw_rae_weight_decay",
        ):
            self.assertEqual(adam_adapter[key], adam_keeplora[key])

    def test_mixed_rae_optimizer_configs_preserve_the_backbone_recipes(self):
        configs = {
            name: json.loads((ROOT / "exps" / name).read_text(encoding="utf-8"))
            for name in (
                "keeplora_adamw.json",
                "keeplora_adamw_mlp.json",
                "keeplora_adamw_sgd_rae.json",
                "keeplora_adamw_mlp_sgd_rae.json",
            )
        }
        semantic_drift = configs["keeplora_adamw_mlp.json"]["semantic_drift"]
        for source_name, experiment_name in (
            ("keeplora_adamw.json", "keeplora_adamw_sgd_rae.json"),
            ("keeplora_adamw_mlp.json", "keeplora_adamw_mlp_sgd_rae.json"),
        ):
            source = configs[source_name].copy()
            experiment = configs[experiment_name]
            for key in (
                "adamw_rae_lr",
                "adamw_rae_weight_decay",
                "semantic_drift",
            ):
                source.pop(key, None)
            expected = source.copy()
            expected.update(
                {
                    "prefix": experiment["prefix"],
                    "rae_optimizer": "sgd",
                    "rae_sgd_momentum": 0.9,
                    "semantic_drift": semantic_drift,
                }
            )
            self.assertEqual(experiment, expected)
            self.assertFalse(experiment["resume"])
            self.assertTrue(experiment["isolate_runs"])
            self.assertTrue(experiment["ssca"])
            self.assertTrue(experiment["ca"])
            self.assertEqual(experiment["optimizer"], "adamw")
            self.assertEqual(
                experiment["ae_init_lr"], source["ae_init_lr"]
            )
            self.assertEqual(
                experiment["ae_weight_decay"], source["ae_weight_decay"]
            )

    def test_keeplora_adamw_ssca_off_configs_only_disable_compensation(self):
        pairs = (
            ("keeplora_adamw.json", "keeplora_adamw_nossca.json"),
            ("keeplora_adamw_mlp.json", "keeplora_adamw_mlp_nossca.json"),
        )
        for enabled_name, disabled_name in pairs:
            with self.subTest(pair=(enabled_name, disabled_name)):
                enabled = json.loads(
                    (ROOT / "exps" / enabled_name).read_text(encoding="utf-8")
                )
                disabled = json.loads(
                    (ROOT / "exps" / disabled_name).read_text(encoding="utf-8")
                )
                enabled_prefix = enabled.pop("prefix")
                disabled_prefix = disabled.pop("prefix")
                self.assertNotEqual(enabled_prefix, disabled_prefix)
                self.assertTrue(enabled.pop("ssca"))
                self.assertFalse(disabled.pop("ssca"))
                self.assertEqual(enabled, disabled)
                self.assertTrue(disabled["ca"])
                self.assertEqual(disabled["optimizer"], "adamw")
                self.assertFalse(disabled["resume"])
                self.assertTrue(disabled["isolate_runs"])

        self.assertEqual(
            json.loads(
                (ROOT / "exps" / "keeplora_adamw_nossca.json")
                .read_text(encoding="utf-8")
            )["keeplora_targets"],
            ["q", "k", "v", "o"],
        )
        self.assertEqual(
            json.loads(
                (ROOT / "exps" / "keeplora_adamw_mlp_nossca.json")
                .read_text(encoding="utf-8")
            )["keeplora_target"],
            "mlp",
        )


if __name__ == "__main__":
    unittest.main()
