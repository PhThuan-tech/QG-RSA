import json
import pathlib
import unittest
from types import MethodType, SimpleNamespace

import torch
from torch import nn

from models.RSIAT_adapter import Learner
from models.base import BaseLearner


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


if __name__ == "__main__":
    unittest.main()
