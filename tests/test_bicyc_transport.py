import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.RSIAT_adapter import Learner
from models.base import BaseLearner

from utils.bicyc_transport import (
    analytic_affine_gaussian_transport,
    bicyc_loss_terms,
    make_deterministic_affine,
    module_state_sha256,
    same_input_feature_pair,
    weighted_bicyc_loss,
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


if __name__ == "__main__":
    unittest.main()
