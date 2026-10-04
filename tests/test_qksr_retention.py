"""Source-only retention checks; all generated files stay in this checkout."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from data.data_manager import DataManager
from models.RSIAT_adapter import Learner
from network.classifier import SimpleContinualLinear
from network.vision_transformer_adapter import validate_backbone_load
from utils.loss import AngularPenaltySMLoss, prototype_relation_kl
from utils.qksr_profiles import RETENTION_OPTIONS
from utils.statistics_transport import (
    _fit_residual_map, estimate_gaussian_statistics, transport_gaussian_statistics,
)
from utils.toolkit import AutoencoderSigmoid, SignedResidualAutoencoder

ROOT = Path(__file__).resolve().parents[1]


class OfflineConvNet(nn.Module):
    out_dim = 5

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 5))

    def forward(self, inputs):
        return self.blocks(inputs.flatten(1))


class OfflineNetwork(nn.Module):
    feature_dim = 5

    def __init__(self):
        super().__init__()
        self.convnet, self.fc = OfflineConvNet(), None

    def update_fc(self, size):
        if self.fc is None:
            self.fc = SimpleContinualLinear(self.feature_dim, size)
        else:
            self.fc.update(size, freeze_old=False)

    def extract_vector(self, inputs):
        return self.convnet(inputs)

    def forward(self, inputs):
        return self.fc(self.extract_vector(inputs))

    def ca_forward(self, features):
        return self.fc(features)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        self.requires_grad_(False)
        return self.eval()


def toy_manager():
    manager = DataManager.__new__(DataManager)
    manager._train_data = np.arange(6 * 8 * 4, dtype=np.uint8).reshape(48, 2, 2)
    manager._train_targets = np.repeat(np.arange(6), 8)
    manager._test_data = manager._train_data.copy()
    manager._test_targets = manager._train_targets.copy()
    manager._train_trsf = manager._test_trsf = [transforms.ToTensor()]
    manager._common_trsf = []
    manager._increments, manager._class_order = [2, 2, 2], list(range(6))
    manager.use_path = False
    return manager


def make_learner(**overrides):
    args = {
        "device": [torch.device("cpu")], "dataset": "toy", "seed": 1993,
        "init_cls": 2, "increment": 2, "convnet_type": "offline_adapter", "model_name": "adapter",
        "batch_size": 5, "num_workers": 0, "stats_num_workers": 0, "pin_memory": False,
        "init_lr": 0.003, "weight_decay": 0.0, "min_lr": 0.0,
        "init_epochs": 2, "inc_epochs": 2, "ca_epochs": 1, "warmup_epoch": 1,
        "optimizer": "sgd", "alpha": 0.5, "rs_margin": 0.5,
        "lambda_rs": 0.5, "beta": 1.5, "gamma": 0.75, "scale": 5.0, "margin": 0.2,
        "ae_code_dims": 8, "ae_init_lr": 0.01, "ae_weight_decay": 0.0,
        "ssca": True, "ca": True, "use_quantum_kernel_base": True, "use_quantum_kernel_inc": True,
        **RETENTION_OPTIONS, "q_num_qubits": 3, "q_num_layers": 1,
        "q_calib_samples": 8, "transport_rank": 3, "val_ratio": 0.25,
        **overrides,
    }
    with patch("models.RSIAT_adapter.SimpleVitNet", return_value=OfflineNetwork()):
        learner = Learner(args)
    learner.class_order = list(range(6))
    return learner


class RetentionUnitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1993)

    def test_signed_projector_starts_identity_with_equal_parameter_count(self):
        signed, legacy = SignedResidualAutoencoder(5, 8), AutoencoderSigmoid(5, 8)
        features = torch.randn(7, 5)
        torch.testing.assert_close(signed(features), features, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in signed.parameters()), sum(p.numel() for p in legacy.parameters()))
        self.assertTrue(((legacy(features) - features) > 0).all())

    def test_signed_projector_can_learn_negative_shift(self):
        signed = SignedResidualAutoencoder(5, 8)
        features = torch.randn(8, 5)
        target = features - 0.5
        optimizer = torch.optim.SGD(signed.parameters(), lr=0.1)
        initial = nn.functional.mse_loss(signed(features), target).detach()
        for _ in range(10):
            optimizer.zero_grad()
            nn.functional.mse_loss(signed(features), target).backward()
            optimizer.step()
        self.assertLess(float(nn.functional.mse_loss(signed(features), target).detach()), float(initial))
        self.assertLess(float((signed(features) - features).mean().detach()), 0.0)

    def test_projector_initialization_does_not_move_training_rng(self):
        learner = make_learner()
        learner._cur_task = 1
        before = torch.get_rng_state().clone()
        learner._new_drift_projector()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_relation_retention_matches_identity_and_only_updates_student(self):
        previous, prototypes = torch.randn(4, 5), torch.randn(3, 5)
        self.assertLess(abs(float(prototype_relation_kl(previous, previous, prototypes, prototypes))), 1e-7)
        current = (previous + torch.randn_like(previous) * 0.2).requires_grad_()
        teacher = previous.clone().requires_grad_()
        anchors = prototypes.clone().requires_grad_()
        loss = prototype_relation_kl(current, teacher, anchors, anchors)
        loss.backward()
        self.assertGreater(float(current.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(anchors.grad)

    def test_detached_prototypes_remove_repulsion_escape_path(self):
        learner = make_learner(beta=0.0, relation_distill_weight=0.0, rs_margin_inc=0.0)
        learner._class_means = torch.randn(3, 5).numpy()
        learner.old_ae = SignedResidualAutoencoder(5, 8)
        learner.quantum_kernel.set_inc_mode("frozen")
        current = torch.randn(4, 5, requires_grad=True)
        learner._inc_loss(current, torch.randn(4, 5)).backward()
        self.assertGreater(float(current.grad.abs().sum()), 0.0)
        self.assertTrue(all(p.grad is None or not p.grad.any() for p in learner.old_ae.parameters()))

    def test_covariance_singleton_and_shrinkage_are_finite_positive_definite(self):
        for features in (torch.ones(1, 5), torch.ones(3, 5), torch.randn(8, 5)):
            mean, cov = estimate_gaussian_statistics(features, shrinkage=0.1)
            self.assertTrue(torch.isfinite(mean).all() and torch.isfinite(cov).all())
            self.assertGreater(float(torch.linalg.eigvalsh(cov).min()), 0.0)
        with self.assertRaises(ValueError):
            estimate_gaussian_statistics(torch.tensor([[float("nan"), 0.0]]))

    def test_covariance_defaults_match_legacy_for_multiple_samples(self):
        features = torch.randn(7, 5, dtype=torch.float64)
        mean, cov = estimate_gaussian_statistics(features)
        torch.testing.assert_close(mean, features.mean(0))
        torch.testing.assert_close(cov, torch.cov(features.T) + torch.eye(5) * 1e-3)

    def test_transport_updates_covariance_by_same_affine_map_as_mean(self):
        old = torch.randn(80, 5, dtype=torch.float64)
        new = old @ (torch.eye(5, dtype=torch.float64) * 1.1) + 0.2
        mean = old[:1].clone()
        cov = torch.eye(5, dtype=torch.float64)[None]
        kwargs = dict(rank=5, ridge=0.001, max_change=0.5, jitter=1e-4)
        mapped_mean, mapped_cov, info = transport_gaussian_statistics(mean, cov, old, new, **kwargs)
        self.assertTrue(info["accepted"])
        center, shift, basis, coefficient = _fit_residual_map(old, new, 5, 0.001, 0.5)
        trust = info["trust_mean"]  # Single class is exactly on a training anchor.
        matrix = torch.eye(5, dtype=torch.float64) + trust * basis @ coefficient
        expected_cov = matrix.T @ cov[0] @ matrix + torch.eye(5, dtype=torch.float64) * 1e-4
        torch.testing.assert_close(mapped_cov[0], expected_cov)
        expected_mean = mean + trust * (shift + (mean - center) @ basis @ coefficient)
        torch.testing.assert_close(torch.from_numpy(mapped_mean), expected_mean)
        self.assertGreater(float(torch.linalg.eigvalsh(mapped_cov).min()), 0.0)

    def test_transport_reduces_known_affine_drift_error_in_synthetic_case(self):
        old = torch.randn(100, 5, dtype=torch.float64)
        rotation = torch.eye(5, dtype=torch.float64)
        angle = torch.tensor(0.2, dtype=torch.float64)
        rotation[:2, :2] = torch.stack((torch.stack((angle.cos(), -angle.sin())), torch.stack((angle.sin(), angle.cos()))))
        matrix = rotation * 1.1
        new = old @ matrix + 0.2
        means = torch.randn(3, 5, dtype=torch.float64) * 0.2
        covariance = torch.eye(5, dtype=torch.float64).repeat(3, 1, 1)
        mapped, cov, info = transport_gaussian_statistics(
            means, covariance, old, new, rank=5, ridge=1e-5, max_change=0.5, support_scale=100.0,
        )
        oracle_mean, oracle_cov = means @ matrix + 0.2, matrix.T @ covariance @ matrix
        self.assertTrue(info["accepted"])
        self.assertLess(float((torch.from_numpy(mapped) - oracle_mean).square().mean()),
                        float((means - oracle_mean).square().mean()) * 0.01)
        self.assertLess(float((cov - oracle_cov).square().mean()), float((covariance - oracle_cov).square().mean()) * 0.01)

    def test_transport_rejects_bad_holdout_and_unsupported_classes_without_rng_drift(self):
        old = torch.randn(30, 5, dtype=torch.float64)
        generator = torch.Generator().manual_seed(17)
        holdout = torch.randperm(30, generator=generator)[:6]
        new = old + 1.0
        new[holdout] = old[holdout] - 1.0
        means, covariance = old[:2].clone(), torch.eye(5, dtype=torch.float64).repeat(2, 1, 1)
        before = torch.get_rng_state().clone()
        mapped, cov, info = transport_gaussian_statistics(means, covariance, old, new, seed=17)
        self.assertFalse(info["accepted"])
        np.testing.assert_array_equal(mapped, means.numpy())
        torch.testing.assert_close(cov, covariance)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        mapped, cov, info = transport_gaussian_statistics(means + 1e6, covariance, old, old + 0.1)
        self.assertEqual(info["supported_classes"], 0)
        np.testing.assert_array_equal(mapped, (means + 1e6).numpy())
        torch.testing.assert_close(cov, covariance)

    def test_transport_no_drift_and_small_datasets_are_noops(self):
        old = torch.randn(5, 5)
        means, covariance = old[:1], torch.eye(5)[None]
        for before, after in ((old, old), (old[:3], old[:3] + 1)):
            mapped, cov, info = transport_gaussian_statistics(means, covariance, before, after)
            self.assertFalse(info["accepted"])
            np.testing.assert_array_equal(mapped, means.double().numpy())
            torch.testing.assert_close(cov, covariance)

    def test_stable_cosface_matches_cross_entropy_and_high_scale_stays_finite(self):
        scores = torch.randn(4, 3).clamp(-1, 1).requires_grad_()
        labels = torch.tensor([0, 1, 2, 0])
        for scale in (5.0, 1000.0):
            adjusted = scores.clone()
            adjusted[torch.arange(4), labels] -= 0.2
            expected = nn.functional.cross_entropy(adjusted * scale, labels)
            actual = AngularPenaltySMLoss(s=scale, m=0.2)(scores, labels)
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.isfinite(actual))
        actual.backward()
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_validation_evaluation_never_reads_test_when_holdout_exists(self):
        learner = make_learner()
        learner.seen_val_loader, learner.test_loader = object(), object()
        self.assertIs(learner._evaluation_loader(), learner.seen_val_loader)
        learner.seen_val_loader = None
        self.assertIs(learner._evaluation_loader(), learner.test_loader)

    def test_configurable_base_lr_reaches_actual_optimizer_groups(self):
        learner = make_learner(base_adapter_lr=0.004, use_quantum_kernel_base=False, use_quantum_kernel_inc=False)
        learner._cur_task = 0
        learner._network.update_fc(2)
        captured = []
        learner._init_train = lambda train, test, opt, sched, warmup: captured.extend(opt.param_groups)
        learner._train(None, None)
        self.assertEqual([group["lr"] for group in captured], [0.004] * 3)

    def test_new_protocol_is_rejected_when_loading_legacy_metadata(self):
        learner = make_learner()
        old_metadata = {"dataset": "toy", "seed": 1993, "init_cls": 2, "increment": 2}
        with self.assertRaisesRegex(ValueError, "ae_type"):
            learner._validate_checkpoint({"run_metadata": old_metadata})

    def test_partial_pretrained_load_cannot_silently_create_random_backbone(self):
        validate_backbone_load(SimpleNamespace(missing_keys=["blocks.0.adaptmlp.up_proj.weight"], unexpected_keys=[]))
        for missing, unexpected in ((["norm.weight"], []), ([], ["blocks.0.wrong_key"])):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                validate_backbone_load(SimpleNamespace(missing_keys=missing, unexpected_keys=unexpected))

    def test_shared_adam_retention_control_also_optimizes_projector(self):
        learner = make_learner(optimizer="adam", use_quantum_kernel_base=False, use_quantum_kernel_inc=False)
        learner._cur_task = 1
        learner._network.update_fc(2)
        learner.old_ae = learner._new_drift_projector()
        groups = []
        learner._init_train = lambda train, test, opt, sched, warmup: groups.extend(opt.param_groups)
        learner._train(None, None)
        optimized = {id(parameter) for group in groups for parameter in group["params"]}
        self.assertTrue(all(id(parameter) in optimized for parameter in learner.old_ae.parameters()))

    def test_retention_rejects_incompatible_or_nonfinite_options(self):
        for options in ({"compact_diagonal_checkpoint": True}, {"ssca_feature_mode": "legacy"},
                        {"ssca": False}, {"relation_temperature": 0.0},
                        {"stats_cov_shrinkage": 1.1}, {"transport_max_change": 1.0},
                        {"q_inc_pair": "old_proj"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_learner(**options)
        old = torch.randn(5, 5)
        for options in ({"jitter": float("nan")}, {"rank": 1.5}, {"chunk_size": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                transport_gaussian_statistics(old[:1], torch.eye(5)[None], old, old + 0.1, **options)

    def test_incomplete_task_checkpoint_is_rejected_instead_of_random_weights(self):
        learner = make_learner(use_quantum_kernel_base=False, use_quantum_kernel_inc=False)
        learner._cur_task = 0
        learner._known_classes = learner._total_classes = 2
        learner.task_sizes = [2]
        learner._network.update_fc(2)
        with tempfile.TemporaryDirectory(prefix=".qksr_test_checkpoint_", dir=ROOT) as directory:
            path = str(Path(directory) / "partial.pkl")
            learner.save_checkpoint(path)
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["model_state_dict"].pop("convnet.blocks.0.weight")
            torch.save(checkpoint, path)
            restored = make_learner(use_quantum_kernel_base=False, use_quantum_kernel_inc=False)
            with self.assertRaisesRegex(ValueError, "incompatible network weights"):
                restored.load_checkpoint(path)


class RetentionPipelineTests(unittest.TestCase):
    def test_three_task_pipeline_and_checkpoint_resume_match_without_downloads(self):
        torch.manual_seed(123)
        manager, learner = toy_manager(), make_learner()
        for _ in range(2):
            learner.incremental_train(manager)
            learner.after_task()
        self.assertEqual(len(learner.seen_val_loader.dataset), 8)
        self.assertEqual(len(learner.stage_metrics), 4)
        self.assertTrue(torch.linalg.cholesky_ex(learner._class_covs).info.eq(0).all())
        projector_before = learner.old_ae
        with tempfile.TemporaryDirectory(prefix=".qksr_test_retention_", dir=ROOT) as directory:
            checkpoint = str(Path(directory) / "task_1.pkl")
            learner.save_checkpoint(checkpoint)
            # Uninterrupted task 2.
            learner.incremental_train(manager)
            learner.after_task()
            expected_weights = {key: value.clone() for key, value in learner._network.state_dict().items()}
            expected_cov, expected_mean = learner._class_covs.clone(), learner._class_means.copy()
            self.assertIsNot(learner.old_ae, projector_before)
            # Same boundary checkpoint, reconstructed modules then restored RNG.
            restored = make_learner()
            restored.load_checkpoint(checkpoint)
            restored.incremental_train(manager)
            restored.after_task()
            for key, value in restored._network.state_dict().items():
                torch.testing.assert_close(value, expected_weights[key], rtol=0, atol=0)
            torch.testing.assert_close(restored._class_covs, expected_cov, rtol=0, atol=0)
            np.testing.assert_array_equal(restored._class_means, expected_mean)
            self.assertEqual(len(restored.stage_metrics), 6)
            self.assertEqual(len(restored.seen_val_loader.dataset), 12)
            self.assertTrue(torch.linalg.cholesky_ex(restored._class_covs).info.eq(0).all())


if __name__ == "__main__":
    unittest.main()
