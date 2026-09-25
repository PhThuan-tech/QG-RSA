import random
import json
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.RSIAT_adapter import Learner
from models.base import BaseLearner
from trainer import _finalize_ca_rescue

from utils.bicyc_transport import (
    analytic_affine_gaussian_transport,
    bicyc_loss_terms,
    make_deterministic_affine,
    module_state_sha256,
    same_input_feature_pair,
    weighted_bicyc_loss,
)
from utils.ca_rescue import (
    affine_distribution_diagnostic,
    capture_rng_state,
    restore_rng_state,
    sample_source_then_affine,
)


class Recorder(nn.Module):
    def __init__(self, shift):
        super().__init__()
        self.shift = shift
        self.seen_ptr = None
        self.seen_value = None

    def forward(self, x):
        self.seen_ptr = x.data_ptr()
        self.seen_value = x.detach().clone()
        return x + self.shift


class TinyCANetwork(nn.Module):
    def __init__(self, dimension, classes):
        super().__init__()
        self.fc = nn.Linear(dimension, classes, bias=False)

    def ca_forward(self, features):
        return {"logits": self.fc(F.normalize(features, p=2, dim=1))}


class BiCycTransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_same_input_tensor_is_used_for_pair(self):
        current, old = Recorder(1.0), Recorder(-1.0)
        x = torch.randn(7, 5)
        z_new, z_old = same_input_feature_pair(current, old, x)
        self.assertEqual(current.seen_ptr, old.seen_ptr)
        self.assertTrue(torch.equal(current.seen_value, old.seen_value))
        self.assertTrue(torch.equal(z_new, x + 1.0))
        self.assertTrue(torch.equal(z_old, x - 1.0))

    def _leaves(self):
        return (torch.randn(8, 6, requires_grad=True),
                torch.randn(8, 6, requires_grad=True))

    @staticmethod
    def _has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    def test_forward_gradient_routing(self):
        z_new, z_old = self._leaves()
        A = nn.Linear(6, 6)
        loss = bicyc_loss_terms(z_new, z_old, A, None, "forward")["loss_a"]
        loss.backward()
        self.assertTrue(self._has_grad(A))
        self.assertIsNone(z_new.grad)
        self.assertIsNone(z_old.grad)

    def test_backward_gradient_routing(self):
        z_new, z_old = self._leaves()
        A, D = nn.Linear(6, 6), nn.Linear(6, 6)
        loss = bicyc_loss_terms(z_new, z_old, A, D, "bidirectional")["loss_d"]
        loss.backward()
        self.assertFalse(self._has_grad(A))
        self.assertTrue(self._has_grad(D))
        self.assertIsNotNone(z_new.grad)
        self.assertGreater(float(z_new.grad.abs().sum()), 0.0)
        self.assertIsNone(z_old.grad)

    def test_cycle_new_gradient_routing(self):
        z_new, z_old = self._leaves()
        A, D = nn.Linear(6, 6), nn.Linear(6, 6)
        loss = bicyc_loss_terms(z_new, z_old, A, D, "cycle")["cycle_new"]
        loss.backward()
        self.assertTrue(self._has_grad(A))
        self.assertFalse(self._has_grad(D))
        self.assertIsNone(z_new.grad)
        self.assertIsNone(z_old.grad)

    def test_cycle_old_gradient_routing(self):
        z_new, z_old = self._leaves()
        A, D = nn.Linear(6, 6), nn.Linear(6, 6)
        loss = bicyc_loss_terms(z_new, z_old, A, D, "cycle")["cycle_old"]
        loss.backward()
        self.assertTrue(self._has_grad(D))
        self.assertFalse(self._has_grad(A))
        self.assertIsNone(z_new.grad)
        self.assertIsNone(z_old.grad)

    def test_analytic_gaussian_transport(self):
        A = nn.Linear(3, 3, bias=True)
        with torch.no_grad():
            A.weight.copy_(torch.tensor([[1., 2., 0.], [0., -1., 1.], [2., 0., 1.]]))
            A.bias.copy_(torch.tensor([0.5, -0.25, 1.]))
        means = torch.tensor([[1., 2., 3.], [-1., 0., 2.]], dtype=torch.float64)
        covs = torch.stack([torch.eye(3, dtype=torch.float64),
                            torch.diag(torch.tensor([1., 2., 4.], dtype=torch.float64))])
        got_mean, got_cov = analytic_affine_gaussian_transport(means, covs, A)
        W, b = A.weight.double(), A.bias.double()
        self.assertTrue(torch.allclose(got_mean, means @ W.T + b))
        self.assertTrue(torch.allclose(got_cov, W[None] @ covs @ W.T[None]))
        self.assertTrue(torch.isfinite(got_cov).all())
        self.assertTrue(torch.allclose(got_cov, got_cov.transpose(-1, -2)))

    def test_rng_preservation_and_matched_initial_hashes(self):
        random.seed(9); np.random.seed(9); torch.manual_seed(9)
        py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
        a1 = make_deterministic_affine(16, 199301)
        d1 = make_deterministic_affine(16, 199302)
        self.assertEqual(py_state, random.getstate())
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        a2 = make_deterministic_affine(16, 199301)
        d2 = make_deterministic_affine(16, 199302)
        self.assertEqual(module_state_sha256(a1), module_state_sha256(a2))
        self.assertEqual(module_state_sha256(d1), module_state_sha256(d2))
        self.assertNotEqual(module_state_sha256(a1), module_state_sha256(d1))

    def test_official_mode_is_exact_noop(self):
        base = torch.tensor(3.25, requires_grad=True)
        z = torch.randn(4, 3)
        terms = bicyc_loss_terms(z, z, None, None, "official")
        total = base + weighted_bicyc_loss(terms, "official")
        self.assertEqual(float(total.detach()), float(base.detach()))
        total.backward()
        self.assertEqual(float(base.grad), 1.0)

    def test_original_rsiat_P_loss_is_numerically_unchanged(self):
        learner = Learner.__new__(Learner)
        learner.old_ae = nn.Linear(4, 4)
        learner._class_means = np.random.RandomState(3).randn(5, 4)
        learner._device = torch.device("cpu")
        learner.args = {"beta": 1.7, "gamma": 0.4}
        z_new = torch.randn(6, 4)
        z_old = torch.randn(6, 4)
        mapped = learner.old_ae(z_old)
        expected_align = F.mse_loss(z_new, mapped)
        expected_old_norm = F.normalize(mapped, p=2, dim=1)
        protos = learner.old_ae(torch.from_numpy(learner._class_means).float())
        protos = F.normalize(protos, p=2, dim=1)
        expected_orth = (protos @ expected_old_norm.T).mean()
        expected = 1.7 * expected_align + 0.4 * expected_orth
        got, got_align, got_orth = learner._inc_loss_components(z_new, z_old)
        self.assertTrue(torch.equal(got_align, expected_align))
        self.assertTrue(torch.equal(got_orth, expected_orth))
        self.assertTrue(torch.equal(got, expected))

    def test_minimal_pd_covariance_repairs_singular_psd_with_tiny_jitter(self):
        covariance = torch.diag(torch.tensor([1.0, 0.25, 0.0]))
        repaired, diagnostic = BaseLearner._minimal_pd_covariance(covariance)
        _, info = torch.linalg.cholesky_ex(repaired, check_errors=False)
        self.assertEqual(int(info.item()), 0)
        self.assertTrue(diagnostic["required"])
        self.assertGreater(diagnostic["jitter"], 0.0)
        self.assertLessEqual(diagnostic["relative_jitter"], 1e-3)
        self.assertTrue(torch.equal(covariance, torch.diag(torch.tensor([1.0, 0.25, 0.0]))))

    def test_pd_covariance_fallback_refuses_substantive_repair(self):
        covariance = torch.diag(torch.tensor([1.0, 0.25, -1.0]))
        with self.assertRaises(RuntimeError):
            BaseLearner._minimal_pd_covariance(covariance)

    def test_checkpoint_payload_roundtrip_includes_A_and_D(self):
        source = Learner.__new__(Learner)
        source.bicyc_mode = "cycle"
        source._cur_task = 1
        source._device = torch.device("cpu")
        source._network = SimpleNamespace(feature_dim=8)
        source.transport_init_seed = 1993000
        source.forward_transport = make_deterministic_affine(8, source._transport_seed(1, "A"))
        source.backward_transport = make_deterministic_affine(8, source._transport_seed(1, "D"))
        source.transport_initial_hashes = {"A": {"sha256": module_state_sha256(source.forward_transport)},
                                           "D": {"sha256": module_state_sha256(source.backward_transport)}}
        source.transport_history = [{"task": 1}]
        source.pre_ca_metrics = {"1": {"metrics": {"top1": 1.0}}}
        source.experiment_records = [{"task": 0}]
        payload = source._checkpoint_extra_state()
        self.assertIn("forward_transport_state_dict", payload)
        self.assertIn("backward_transport_state_dict", payload)

        restored = Learner.__new__(Learner)
        restored.bicyc_mode = "cycle"
        restored._cur_task = 1
        restored._device = torch.device("cpu")
        restored._network = SimpleNamespace(feature_dim=8)
        restored.transport_init_seed = 1993000
        restored._restore_transport_state(payload)
        self.assertEqual(module_state_sha256(source.forward_transport),
                         module_state_sha256(restored.forward_transport))
        self.assertEqual(module_state_sha256(source.backward_transport),
                         module_state_sha256(restored.backward_transport))
        self.assertEqual(restored.transport_history, source.transport_history)

    def test_sample_source_then_affine_preserves_exact_ca_target_distribution(self):
        affine = nn.Linear(3, 3, bias=True)
        with torch.no_grad():
            affine.weight.copy_(torch.tensor([
                [1.0, 0.2, -0.1], [0.0, 0.8, 0.3], [0.1, -0.2, 1.1]
            ]))
            affine.bias.copy_(torch.tensor([0.4, -0.2, 0.1]))
        source_mean = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
        source_covariance = torch.tensor([
            [1.0, 0.1, 0.0], [0.1, 0.6, 0.05], [0.0, 0.05, 0.8]
        ], dtype=torch.float64)
        alpha = 0.9333333333333333
        analytic_mean = source_mean @ affine.weight.double().T + affine.bias.double()
        target_mean = alpha * analytic_mean
        torch.manual_seed(23)
        result = sample_source_then_affine(
            source_mean, source_covariance, target_mean, affine, 50000)
        source = result["source_samples"]
        raw = result["raw_transformed"]
        ca_samples = result["ca_samples"]
        self.assertTrue(torch.equal(raw, affine(source)))
        expected_samples = target_mean.float()[None] + F.linear(
            source - source_mean.float()[None], affine.weight, bias=None)
        self.assertTrue(torch.equal(ca_samples, expected_samples))
        self.assertTrue(torch.isfinite(ca_samples).all())
        expected_covariance = (
            affine.weight.double() @ source_covariance @ affine.weight.double().T)
        self.assertTrue(torch.allclose(
            ca_samples.double().mean(0), target_mean, atol=0.02, rtol=0.02))
        self.assertTrue(torch.allclose(
            torch.cov(ca_samples.double().T), expected_covariance,
            atol=0.03, rtol=0.06))

    def test_direct_affine_sample_is_not_historical_ca_mean_when_alpha_not_one(self):
        affine = nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            affine.weight.copy_(torch.eye(2))
            affine.bias.copy_(torch.tensor([0.4, -0.2]))
        source_mean = torch.tensor([1.0, 2.0], dtype=torch.float64)
        alpha = 0.9
        raw_mean = source_mean @ affine.weight.double().T + affine.bias.double()
        historical_target = alpha * raw_mean
        self.assertFalse(torch.equal(raw_mean, historical_target))
        self.assertTrue(torch.allclose(raw_mean - historical_target,
                                       (1.0 - alpha) * raw_mean))

    def test_affine_distribution_diagnostic_reports_finite_empirical_errors(self):
        affine = nn.Linear(2, 2)
        source_mean = torch.tensor([0.2, -0.4], dtype=torch.float64)
        source_covariance = torch.tensor(
            [[0.7, 0.1], [0.1, 0.5]], dtype=torch.float64)
        torch.manual_seed(31)
        target = source_mean @ affine.weight.double().T + affine.bias.double()
        result = sample_source_then_affine(
            source_mean, source_covariance, target, affine, 4096)
        diagnostic = affine_distribution_diagnostic(
            result["source_samples"], result["raw_transformed"],
            source_mean, source_covariance, affine)
        self.assertTrue(all(np.isfinite(value) for value in diagnostic.values()))
        self.assertLess(diagnostic[
            "sample_transform_max_abs_error_float64_recompute"], 1e-6)

    def test_rng_snapshot_roundtrip(self):
        random.seed(41); np.random.seed(41); torch.manual_seed(41)
        snapshot = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng_state(snapshot)
        observed = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(expected[0], observed[0])
        self.assertEqual(expected[1], observed[1])
        self.assertTrue(torch.equal(expected[2], observed[2]))

    def test_rescue_ca_loop_keeps_loss_gradients_and_classifier_finite(self):
        learner = Learner.__new__(Learner)
        learner.ca_sampling_mode = "sample_source_then_affine"
        learner._device = torch.device("cpu")
        learner._multiple_gpus = []
        learner._cur_task = 2
        learner._known_classes = 2
        learner._total_classes = 3
        learner.task_sizes = [1, 1, 1]
        learner.init_lr = 0.01
        learner.weight_decay = 0.0
        learner.logit_norm = None
        learner.args = {"scale": 4.0, "ca_epochs": 2,
                        "ca_covariance_pd_fallback": False}
        learner._network = TinyCANetwork(4, 3)
        learner.forward_transport = nn.Linear(4, 4)
        with torch.no_grad():
            learner.forward_transport.weight.copy_(torch.eye(4))
            learner.forward_transport.bias.copy_(torch.tensor([0.1, -0.2, 0.0, 0.3]))
        learner._ca_source_old_means = np.array([
            [0.5, 0.0, -0.3, 0.2], [-0.2, 0.4, 0.1, -0.5]
        ], dtype=np.float64)
        learner._ca_source_old_covs = torch.stack([
            torch.eye(4) * 0.2, torch.eye(4) * 0.3])
        source_means = torch.from_numpy(learner._ca_source_old_means).double()
        weight = learner.forward_transport.weight.detach().double()
        bias = learner.forward_transport.bias.detach().double()
        transported = source_means @ weight.T + bias
        learner._class_means = np.zeros((3, 4), dtype=np.float64)
        learner._class_means[:2] = transported.numpy()
        learner._class_means[2] = np.array([0.1, 0.2, -0.1, 0.3])
        learner._class_covs = torch.stack([
            torch.eye(4) * 0.2, torch.eye(4) * 0.3, torch.eye(4) * 0.25])
        learner.ca_rescue_diagnostics = {"epochs": []}
        learner._write_ca_rescue_report = lambda: None
        learner._write_progress = lambda *args, **kwargs: None
        torch.manual_seed(53)
        learner._stage2_compact_classifier(task_size=1, ca_epochs=2)
        self.assertEqual(len(learner.ca_rescue_diagnostics["epochs"]), 2)
        for epoch in learner.ca_rescue_diagnostics["epochs"]:
            self.assertTrue(epoch["loss_finite"])
            self.assertEqual(epoch["synthetic_sample_finite_fraction"], 1.0)
            self.assertEqual(epoch["gradient_finite_fraction_min"], 1.0)
            self.assertEqual(epoch["classifier_finite_fraction_after_epoch"], 1.0)

    def test_final_rescue_gate_is_fail_closed_and_requires_complete_protocol(self):
        with tempfile.TemporaryDirectory() as root:
            model = SimpleNamespace()
            model.args = {
                "ca_epochs": 2, "metrics_output": root + "/metrics.json",
                "arm": "B1_forward_ca_rescue"}
            model._total_classes = 3
            model._cur_task = 2
            model.bicyc_mode = "forward"
            model.ca_covariance_stabilization = {
                "enabled": False, "fallback_count": 0}
            model._current_transport_record = {}
            model._write_ca_rescue_report = lambda: None
            complete_epoch = {
                "batch_count": 3,
                "synthetic_sample_finite_fraction": 1.0,
                "loss_finite": True,
                "gradient_finite_fraction_min": 1.0,
                "classifier_finite_fraction_min": 1.0,
                "classifier_finite_fraction_after_epoch": 1.0,
            }
            model.ca_rescue_diagnostics = {
                "epochs": [dict(complete_epoch), dict(complete_epoch)],
                "ca_protocol": {"epochs": 2, "batches_per_epoch": 3},
                "pre_optimization_gate": {"all_samples_finite": True},
            }
            pre = {"metrics": {"top1": 96.0, "top5": 99.0,
                               "grouped": {"old": 95.0, "new": 98.0}}}
            post = {"metrics": {"top1": 95.0, "top5": 99.0,
                                "grouped": {"old": 94.0, "new": 97.0}}}
            self.assertTrue(_finalize_ca_rescue(model, pre, post))
            self.assertEqual(model.ca_rescue_diagnostics["status"], "RESCUE_PASS")
            model.ca_rescue_diagnostics["epochs"] = [dict(complete_epoch)]
            post["metrics"]["top1"] = 3.33
            self.assertFalse(_finalize_ca_rescue(model, pre, post))
            with open(root + "/metrics.json") as handle:
                payload = json.load(handle)
            self.assertEqual(payload["status"], "FAIL")
            self.assertFalse(payload["checkpoint_saved"])


if __name__ == "__main__":
    unittest.main()
