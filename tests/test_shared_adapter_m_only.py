import json
import os
import random
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from KeepLora.m_only_adapter import MOnlyKeepLoRA, update_feature_basis
from models.base import BaseLearner
from models.RSIAT_adapter import Learner
from network.vision_transformer_adapter import Adapter, VisionTransformer
from trainer import find_latest_checkpoint
from utils.toolkit import AutoencoderSigmoid


def adapter_config(update_mode="baseline"):
    return SimpleNamespace(
        d_model=8,
        attn_bn=4,
        ffn_adapt=True,
        ffn_num=4,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        shared_adapter_update_mode=update_mode,
        vpt_on=False,
        vpt_num=0,
    )


def tiny_vit(update_mode):
    return VisionTransformer(
        img_size=32,
        patch_size=16,
        num_classes=0,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        tuning_config=adapter_config(update_mode),
    ).eval()


class SharedAdapterMOnlyTests(unittest.TestCase):
    def test_learner_trains_merges_and_resumes_with_both_optimizers(self):
        class ThreeTasks:
            def __init__(self):
                self.labels = torch.arange(6).repeat_interleave(6)
                generator = torch.Generator().manual_seed(23)
                self.images = torch.randn(36, 3, 32, 32, generator=generator)

            def get_task_size(self, task):
                return 2

            def get_dataset(self, classes, source, mode, ret_data=False):
                indices = torch.where(torch.isin(self.labels, torch.as_tensor(classes)))[0]
                images, labels = self.images[indices], self.labels[indices]
                dataset = TensorDataset(indices, images, labels)
                return (images, labels, dataset) if ret_data else dataset

        def make_backbone(args, pretrained):
            backbone = tiny_vit(args["shared_adapter_update_mode"])
            backbone.out_dim = 8
            for name, parameter in backbone.named_parameters():
                parameter.requires_grad_("adaptmlp" in name)
            return backbone

        def make_rae(input_dims, code_dims, zero_residual, *, baseline_init_order):
            return AutoencoderSigmoid(
                8, code_dims, zero_residual, baseline_init_order=baseline_init_order
            )

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "exps",
            "exp1_h64_shared_adapter_task0.json",
        )
        with open(config_path, encoding="utf-8") as config_file:
            args = json.load(config_file)
        args.update(
            seed=31, device=[torch.device("cpu")], init_cls=2, increment=2,
            ffn_num=4, rank_down_max=2, rank_up_max=2, feature_threshold=0.7,
            batch_size=6, init_epochs=2, inc_epochs=2, warmup_epoch=1,
            ca_epochs=1, num_workers=0, stats_num_workers=0, pin_memory=False,
            ae_code_dims=4, save_mid_task_checkpoints=True,
        )
        data = ThreeTasks()
        probe_inputs = data.images[:2]
        with (
            tempfile.TemporaryDirectory() as checkpoint_dir,
            patch("utils.inc_net.get_convnet", side_effect=make_backbone),
            patch("models.RSIAT_adapter.AutoencoderSigmoid", side_effect=make_rae),
        ):
            args["output_root"] = checkpoint_dir
            torch.manual_seed(args["seed"])
            baseline = Learner(dict(args, shared_adapter_update_mode="baseline"))
            baseline.incremental_train(data)
            torch.manual_seed(args["seed"])
            common = Learner(args)
            common.incremental_train(data)
            common.after_task()
            self.assertEqual(
                common._shared_adapter_weight_checksum(),
                baseline._shared_adapter_weight_checksum(),
            )
            self.assertEqual(
                common._shared_adapter_state_checksum(),
                baseline._shared_adapter_state_checksum(),
            )
            self.assertEqual(len(common._m_only_bases), 4)
            task_zero = os.path.join(checkpoint_dir, "task_0.pkl")
            common.save_checkpoint(task_zero)

            task_one_initial_factors = []
            for optimizer in ("adamw", "sgd"):
                with self.subTest(optimizer=optimizer):
                    run_args = dict(args, incremental_optimizer=optimizer)
                    torch.manual_seed(args["seed"])
                    learner = Learner(run_args)
                    learner.topk = 2
                    self.assertEqual(learner.load_checkpoint(task_zero), 0)
                    mid_task = os.path.join(checkpoint_dir, "task_1_mid.pkl")
                    learner._mid_task_checkpoint_path = mid_task
                    merge = learner._merge_shared_adapter_m_only_task
                    factor_checksum = learner._m_only_initial_factors_checksum
                    save_checkpoint = learner.save_checkpoint

                    def record_init_checksum():
                        checksum = factor_checksum()
                        task_one_initial_factors.append(checksum)
                        return checksum

                    def save_first_epoch(path, mid_task_training_state):
                        if mid_task_training_state["epoch"] == 0:
                            save_checkpoint(path, mid_task_training_state)

                    @torch.no_grad()
                    def check_merge():
                        network = learner._network.eval()
                        before = network.extract_vector(probe_inputs)
                        merge()
                        after = network.extract_vector(probe_inputs)
                        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-4)

                    with (
                        patch.object(learner, "_merge_shared_adapter_m_only_task", check_merge),
                        patch.object(
                            learner, "_m_only_initial_factors_checksum",
                            side_effect=record_init_checksum,
                        ) as init_checksum,
                        patch.object(learner, "save_checkpoint", save_first_epoch),
                    ):
                        learner.incremental_train(data)
                        self.assertEqual(init_checksum.call_count, 1)
                    mid_state = torch.load(mid_task, weights_only=False)
                    self.assertTrue(mid_state["shared_adapter_m_only_active_factors"])
                    self.assertFalse(
                        learner._checkpoint_extra_state()["shared_adapter_m_only_active_factors"]
                    )
                    self.assertNotEqual(
                        learner._shared_adapter_weight_checksum(),
                        common._shared_adapter_weight_checksum(),
                    )
                    for _, _, _, linear in learner._m_only_adapter_targets():
                        self.assertFalse(linear.bias.requires_grad)
                    self.assertIn("old", learner.eval_task()["grouped"])
                    learner.after_task()
                    task_one = os.path.join(checkpoint_dir, "task_1.pkl")
                    learner.save_checkpoint(task_one)

                    uninterrupted_rng = torch.get_rng_state()
                    resumed = Learner(run_args)
                    self.assertEqual(resumed.load_checkpoint(mid_task), 1)
                    with patch.object(
                        resumed, "_initialize_shared_adapter_m_only_task"
                    ) as initialize:
                        resumed.incremental_train(data)
                    initialize.assert_not_called()
                    resumed.after_task()
                    for key, value in learner._network.state_dict().items():
                        torch.testing.assert_close(resumed._network.state_dict()[key], value)
                    self.assertEqual(
                        resumed._m_only_feature_basis_checksum(),
                        learner._m_only_feature_basis_checksum(),
                    )

                    # The separate mid-task replay consumed RNG in this test process.
                    torch.set_rng_state(uninterrupted_rng)
                    learner._mid_task_checkpoint_path = None
                    learner.incremental_train(data)
                    learner.after_task()

                    torch.manual_seed(args["seed"] + 1)
                    self.assertEqual(resumed.load_checkpoint(task_one), 1)
                    resumed._mid_task_checkpoint_path = None
                    resumed.incremental_train(data)
                    resumed.after_task()
                    self.assertEqual(resumed._cur_task, 2)
                    self.assertEqual(resumed.rae_generation, 1)
                    for key, value in learner._network.state_dict().items():
                        torch.testing.assert_close(resumed._network.state_dict()[key], value)
                    for key, value in learner.old_ae.state_dict().items():
                        torch.testing.assert_close(resumed.old_ae.state_dict()[key], value)
                    np.testing.assert_allclose(resumed._class_means, learner._class_means)
                    self.assertEqual(
                        resumed._m_only_feature_basis_checksum(),
                        learner._m_only_feature_basis_checksum(),
                    )
            self.assertEqual(len(task_one_initial_factors), 2)
            self.assertEqual(task_one_initial_factors[0], task_one_initial_factors[1])

    def test_m_only_rae_initialization_matches_original_rsiat_order(self):
        torch.manual_seed(17)
        reference = nn.Sequential(
            nn.Linear(768, 64), nn.GELU(), nn.Linear(64, 4), nn.GELU(),
            nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 768), nn.Sigmoid(),
        )
        torch.manual_seed(17)
        learner = object.__new__(Learner)
        learner.args = {"ae_code_dims": 4}
        learner._uses_m_only = True
        learner.rae_zero_init = False
        rae = learner._build_rae()
        for actual, expected in zip(rae.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_common_task_zero_config_collects_m0_without_incremental_factors(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "exps",
            "exp1_h64_shared_adapter_task0.json",
        )
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)

        self.assertEqual(config["shared_adapter_update_mode"], "keeplora_m_only")
        self.assertEqual(config["max_tasks_per_run"], 1)
        self.assertEqual(config["optimizer"], "sgd")
        self.assertEqual(config["incremental_optimizer"], "sgd")

    def test_task_zero_basis_checksum_is_stable_and_sensitive_to_basis(self):
        learner = object.__new__(Learner)
        basis = {
            "blocks.0.adaptmlp.down_proj": torch.eye(4, dtype=torch.float32)[:, :2],
            "blocks.0.adaptmlp.up_proj": torch.eye(3, dtype=torch.float32)[:, :1],
        }
        first = Learner._m_only_feature_basis_checksum(learner, basis)
        second = Learner._m_only_feature_basis_checksum(learner, basis)
        self.assertEqual(first, second)

        changed = dict(basis)
        changed["blocks.0.adaptmlp.down_proj"] = basis[
            "blocks.0.adaptmlp.down_proj"
        ].clone()
        changed["blocks.0.adaptmlp.down_proj"][0, 0] = 0.5
        self.assertNotEqual(
            first, Learner._m_only_feature_basis_checksum(learner, changed)
        )

    def test_mid_task_checkpoint_recreates_dynamic_factors_before_loading(self):
        class TinyNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.convnet = tiny_vit("keeplora_m_only")
                self.fc = nn.Linear(8, 3)

        def make_learner(network):
            learner = object.__new__(Learner)
            learner._network = network
            learner._uses_m_only = True
            learner.args = {
                "shared_adapter_m_only_bias_policy": "freeze_bias"
            }
            return learner

        source = TinyNetwork()
        source_learner = make_learner(source)
        active_factors = []
        factors_by_adapter = {}
        for prefix, adapter, kind, linear in source_learner._m_only_adapter_targets():
            if prefix != "blocks.0.adaptmlp":
                continue
            factor = MOnlyKeepLoRA(
                linear.in_features, linear.out_features, rank=2, scaling=0.125
            )
            factor.lora_A.copy_(
                torch.linalg.qr(
                    torch.randn(linear.in_features, 2), mode="reduced"
                ).Q
            )
            factor.lora_B.data.normal_()
            factors_by_adapter.setdefault(prefix, [adapter, {}])[1][kind] = factor
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
        for adapter, factors in factors_by_adapter.values():
            adapter.attach_m_only_factors(
                factors.get("down"), factors.get("up")
            )
        source_state = {
            key: value.detach().clone()
            for key, value in source.state_dict().items()
        }

        restored = TinyNetwork()
        restored_learner = make_learner(restored)
        Learner._prepare_checkpoint_model(
            restored_learner,
            {
                "mid_task_training_state": {"task_id": 1},
                "shared_adapter_m_only_active_factors": active_factors,
            },
        )
        restored.load_state_dict(source_state, strict=True)

        with torch.no_grad():
            inputs = torch.randn(2, 3, 32, 32)
            expected = source.convnet.forward_features(inputs)
            actual = restored.convnet.forward_features(inputs)
        torch.testing.assert_close(actual, expected)
        for prefix, adapter, kind, linear in restored_learner._m_only_adapter_targets():
            if not any(
                item["prefix"] == prefix
                and item["kind"] == kind
                for item in active_factors
            ):
                continue
            factor = getattr(adapter, "keeplora_m_only_{}".format(kind))
            self.assertFalse(linear.weight.requires_grad)
            self.assertFalse(factor.lora_A.requires_grad)
            self.assertTrue(factor.lora_B.requires_grad)

        ablation = make_learner(TinyNetwork())
        ablation.args["shared_adapter_m_only_bias_policy"] = "train_bias_unconstrained"
        ablation._prepare_checkpoint_model({"mid_task_training_state": {"task_id": 1}})
        for _, _, _, linear in ablation._m_only_adapter_targets():
            self.assertTrue(linear.bias.requires_grad)

    def test_mid_task_rng_state_round_trip_restores_all_generators(self):
        random_state = Learner._capture_training_rng_state()
        expected_python = random.random()
        expected_numpy = np.random.random()
        expected_torch = torch.rand(3)

        Learner._restore_training_rng_state(random_state)

        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.random(), expected_numpy)
        torch.testing.assert_close(torch.rand(3), expected_torch)

    def test_checkpoint_discovery_prefers_newest_mid_task_and_completed_snapshot(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            task_zero = os.path.join(checkpoint_dir, "task_0.pkl")
            task_one_mid = os.path.join(checkpoint_dir, "task_1_mid.pkl")
            task_one = os.path.join(checkpoint_dir, "task_1.pkl")
            for path in (task_zero, task_one_mid):
                with open(path, "wb"):
                    pass

            self.assertEqual(
                find_latest_checkpoint(checkpoint_dir), task_one_mid
            )
            with open(task_one, "wb"):
                pass
            self.assertEqual(find_latest_checkpoint(checkpoint_dir), task_one)

    def test_task_zero_model_initialization_and_forward_match_baseline(self):
        torch.manual_seed(31)
        baseline = tiny_vit("baseline")
        baseline_rng = torch.get_rng_state().clone()
        torch.manual_seed(31)
        experimental = tiny_vit("keeplora_m_only")
        experimental_rng = torch.get_rng_state().clone()

        self.assertTrue(torch.equal(baseline_rng, experimental_rng))
        baseline_state = baseline.state_dict()
        experimental_state = experimental.state_dict()
        self.assertEqual(list(baseline_state), list(experimental_state))
        for key in baseline_state:
            self.assertTrue(torch.equal(baseline_state[key], experimental_state[key]))

        inputs = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            baseline_output = baseline.forward_features(inputs)
            experimental_output = experimental.forward_features(inputs)
        torch.testing.assert_close(experimental_output, baseline_output, rtol=0, atol=0)

    def test_adapter_feature_capture_uses_each_linear_input_dimension(self):
        adapter = Adapter(
            adapter_config("keeplora_m_only"),
            dropout=0.1,
            bottleneck=4,
            init_option="lora",
            adapter_scalar="0.1",
            adapter_layernorm_option="none",
        ).eval()
        adapter.start_m_only_feature_collection(rows_per_batch=4)
        with torch.no_grad():
            adapter(torch.randn(2, 5, 8))
        moments = adapter.finish_m_only_feature_collection()

        self.assertEqual(moments["down"].shape, (8, 8))
        self.assertEqual(moments["up"].shape, (4, 4))
        self.assertGreater(torch.trace(moments["down"]).item(), 0)
        self.assertGreater(torch.trace(moments["up"]).item(), 0)

    def test_feature_basis_extends_only_in_the_residual_subspace(self):
        previous = torch.eye(6)[:, :2]
        moment = torch.diag(torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 1.0]))
        basis, stats = update_feature_basis(moment, previous, 0.8)

        self.assertEqual(basis.shape[0], 6)
        self.assertEqual(stats["previous_rank"], 2)
        self.assertGreater(stats["new_rank"], 0)
        self.assertLessEqual(stats["rank"], 6)
        self.assertLess(stats["cross_orthogonality_error"], 1e-5)
        torch.testing.assert_close(
            basis.transpose(0, 1) @ basis,
            torch.eye(basis.shape[1]),
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertEqual(stats["residual_dimension"], 6 - basis.shape[1])

    def test_basis_does_not_grow_if_previous_basis_covers_threshold(self):
        moment = torch.diag(torch.tensor([99.5, 0.4, 0.1]))
        previous = torch.eye(3)[:, :1]
        basis, stats = update_feature_basis(moment, previous, 0.99)
        torch.testing.assert_close(basis, previous)
        self.assertEqual(stats["new_rank"], 0)
        self.assertAlmostEqual(stats["captured_energy"], 0.995)

    def test_protected_gradient_roundoff_does_not_initialize_a_factor(self):
        generator = torch.Generator().manual_seed(5)
        basis = torch.linalg.qr(torch.randn(8, 7, generator=generator)).Q
        gradient = (basis @ torch.randn(7, 4, generator=generator)).T
        with self.assertWarnsRegex(RuntimeWarning, "no update"):
            factor, stats = MOnlyKeepLoRA.from_gradient(
                gradient, basis, 4, "fixed_effective_scaling", 0.0625, 2.0, "no_update"
            )
        self.assertIsNone(factor)
        self.assertEqual(stats["actual_rank"], 0)

    def test_gradient_factorization_is_m_orthogonal_and_uses_effective_scaling(self):
        basis = torch.eye(6)[:, :2]
        safe = torch.randn(6, 4)
        safe[:2] = 0
        weight_gradient = safe.transpose(0, 1)
        factor, stats = MOnlyKeepLoRA.from_gradient(
            weight_gradient,
            basis,
            rank_cap=3,
            scaling_mode="fixed_effective_scaling",
            target_scaling=0.0625,
            lora_alpha=2,
            empty_residual_policy="error",
        )

        self.assertEqual(factor.lora_A.shape, (6, stats["actual_rank"]))
        self.assertEqual(factor.lora_B.shape, (stats["actual_rank"], 4))
        self.assertFalse(factor.lora_A.requires_grad)
        self.assertTrue(factor.lora_B.requires_grad)
        self.assertAlmostEqual(factor.scaling, 0.0625)
        self.assertAlmostEqual(stats["alpha"], 0.0625 * stats["actual_rank"])
        self.assertLess(stats["a_orthogonality_error"], 1e-5)

    def test_factorization_and_merge_preserve_adapter_forward(self):
        torch.manual_seed(43)
        adapter = Adapter(
            adapter_config("keeplora_m_only"),
            dropout=0.1,
            bottleneck=4,
            init_option="lora",
            adapter_scalar="0.1",
            adapter_layernorm_option="none",
        ).eval()
        inputs = torch.randn(3, 7, 8)
        with torch.no_grad():
            original = adapter(inputs)

        factors = {}
        for name, linear in (("down", adapter.down_proj), ("up", adapter.up_proj)):
            gradient = torch.randn_like(linear.weight)
            factor, _ = MOnlyKeepLoRA.from_gradient(
                gradient,
                torch.empty(linear.in_features, 0),
                rank_cap=2,
                scaling_mode="fixed_effective_scaling",
                target_scaling=0.0625,
                lora_alpha=2,
                empty_residual_policy="error",
            )
            linear.weight.data.sub_(factor.get_delta_weight())
            factors[name] = factor
        adapter.attach_m_only_factors(factors["down"], factors["up"])

        with torch.no_grad():
            initialized = adapter(inputs)
        torch.testing.assert_close(initialized, original, rtol=1e-5, atol=1e-6)

        with torch.no_grad():
            factors["down"].lora_B.add_(torch.randn_like(factors["down"].lora_B) * 0.01)
            factors["up"].lora_B.add_(torch.randn_like(factors["up"].lora_B) * 0.01)
            before_merge = adapter(inputs)
            factors["down"].merge_into(adapter.down_proj.weight)
            factors["up"].merge_into(adapter.up_proj.weight)
            adapter.remove_m_only_factors()
            after_merge = adapter(inputs)
        torch.testing.assert_close(after_merge, before_merge, rtol=1e-5, atol=1e-6)

    def test_zero_rank_residual_obeys_explicit_no_update_policy(self):
        with self.assertWarns(RuntimeWarning):
            factor, stats = MOnlyKeepLoRA.from_gradient(
                torch.randn(3, 4),
                torch.eye(4),
                rank_cap=2,
                scaling_mode="fixed_effective_scaling",
                target_scaling=0.0625,
                lora_alpha=2,
                empty_residual_policy="no_update",
            )
        self.assertIsNone(factor)
        self.assertEqual(stats["actual_rank"], 0)

    def test_zero_rank_residual_can_be_configured_to_fail(self):
        with self.assertRaisesRegex(RuntimeError, "no useful residual gradient"):
            MOnlyKeepLoRA.from_gradient(
                torch.randn(3, 4),
                torch.eye(4),
                rank_cap=2,
                scaling_mode="fixed_effective_scaling",
                target_scaling=0.0625,
                lora_alpha=2,
                empty_residual_policy="error",
            )

    def test_task_zero_baseline_checkpoint_is_accepted_by_both_optimizer_options(self):
        baseline_args = {
            "device": ["cpu"],
            "dataset": "cifar224",
            "seed": 1993,
            "init_cls": 10,
            "increment": 10,
            "model_name": "adapter",
            "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
            "ffn_num": 64,
            "optimizer": "sgd",
            "classifier_optimizer": "main",
            "rae_optimizer": "main",
            "shared_adapter_update_mode": "baseline",
        }
        saved_metadata = BaseLearner(baseline_args)._checkpoint_run_metadata()
        for optimizer in ("adamw", "sgd"):
            experimental_args = {
                **baseline_args,
                "shared_adapter_update_mode": "keeplora_m_only",
                "incremental_optimizer": optimizer,
                "rank_down_max": 32,
                "rank_up_max": 32,
                "feature_threshold": 0.99,
                "scaling_mode": "fixed_effective_scaling",
                "target_scaling": 0.0625,
                "lora_alpha": 2.0,
                "ae_init_lr": 5e-4,
                "ae_weight_decay": 0.002,
            }
            learner = BaseLearner(experimental_args)
            learner._validate_checkpoint(
                {
                    "cur_task": 0,
                    "run_metadata": saved_metadata,
                }
            )

    def test_optimizer_options_cannot_resume_each_others_incremental_checkpoint(self):
        args = {
            "device": ["cpu"],
            "dataset": "cifar224",
            "seed": 1993,
            "init_cls": 10,
            "increment": 10,
            "model_name": "adapter",
            "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
            "ffn_num": 64,
            "optimizer": "sgd",
            "classifier_optimizer": "main",
            "rae_optimizer": "main",
            "shared_adapter_update_mode": "keeplora_m_only",
            "incremental_optimizer": "adamw",
            "rank_down_max": 32,
            "rank_up_max": 32,
            "feature_threshold": 0.99,
            "scaling_mode": "fixed_effective_scaling",
            "target_scaling": 0.0625,
            "lora_alpha": 2.0,
            "ae_init_lr": 5e-4,
            "ae_weight_decay": 0.002,
        }
        sgd_metadata = BaseLearner(
            {
                **args,
                "incremental_optimizer": "sgd",
            }
        )._checkpoint_run_metadata()
        with self.assertRaisesRegex(ValueError, "incremental_optimizer"):
            BaseLearner(args)._validate_checkpoint(
                {"cur_task": 1, "run_metadata": sgd_metadata}
            )


if __name__ == "__main__":
    unittest.main()
