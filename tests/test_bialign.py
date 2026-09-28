import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from models.RSIAT_adapter import BIALIGN_MODE, Learner
from utils.bialign import (
    SignedResidualProjector,
    bialign_loss_terms,
    make_identity_reverse_projector,
    module_gradient_record,
)
from utils.bicyc_transport import module_state_sha256
from utils.toolkit import AutoencoderSigmoid


ROOT = Path(__file__).resolve().parents[1]


def has_nonzero_gradient(module):
    return any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in module.parameters()
    )


def all_gradients_finite(module):
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
    )


class BiAlignTests(unittest.TestCase):
    def setUp(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)

    @staticmethod
    def feature_modules(dimension=6):
        current = nn.Linear(dimension, dimension, bias=False)
        old = nn.Linear(dimension, dimension, bias=False)
        old.requires_grad_(False)
        projector = nn.Linear(dimension, dimension)
        reverse = make_identity_reverse_projector(
            dimension, hidden_dim=4, seed=1993001)
        inputs = torch.randn(8, dimension)
        z_new = current(inputs)
        with torch.no_grad():
            z_old = old(inputs)
        return current, old, projector, reverse, z_new, z_old

    def test_existing_P_is_one_sided_while_D_is_signed_lightweight_identity(self):
        projector = AutoencoderSigmoid(input_dims=8, code_dims=4)
        values = torch.randn(5, 8)
        residual = projector(values) - values
        self.assertTrue(torch.all(residual >= 0.0))
        self.assertTrue(torch.all(residual <= 1.0))

        production_reverse = SignedResidualProjector(input_dim=768, hidden_dim=64)
        self.assertEqual(tuple(production_reverse.down.weight.shape), (64, 768))
        self.assertEqual(tuple(production_reverse.up.weight.shape), (768, 64))
        self.assertEqual(
            sum(p.numel() for p in production_reverse.parameters()), 99136)

        reverse = SignedResidualProjector(input_dim=8, hidden_dim=4)
        self.assertTrue(torch.equal(reverse(values), values))
        self.assertIsInstance(reverse.activation, nn.GELU)
        self.assertIsInstance(reverse.up, nn.Linear)
        self.assertLess(sum(p.numel() for p in reverse.parameters()), 8 * 8 * 3)
        with torch.no_grad():
            reverse.up.weight[0, 0] = -1.0
        self.assertLess(
            float((reverse(values) - values).detach().min()), 0.0)

    def test_L_fwd_gradient_routing(self):
        current, old, projector, reverse, z_new, z_old = self.feature_modules()
        terms = bialign_loss_terms(z_new, z_old, projector, reverse)
        terms["loss_fwd"].backward()
        self.assertTrue(has_nonzero_gradient(projector))
        self.assertFalse(has_nonzero_gradient(current))
        self.assertFalse(has_nonzero_gradient(reverse))
        self.assertFalse(has_nonzero_gradient(old))
        self.assertIsNone(z_old.grad)

    def test_L_back_gradient_routing(self):
        current, old, projector, reverse, z_new, z_old = self.feature_modules()
        terms = bialign_loss_terms(z_new, z_old, projector, reverse)
        terms["loss_back"].backward()
        self.assertFalse(has_nonzero_gradient(projector))
        self.assertTrue(has_nonzero_gradient(current))
        self.assertTrue(has_nonzero_gradient(reverse))
        self.assertFalse(has_nonzero_gradient(old))
        self.assertIsNone(z_old.grad)

    def test_D_identity_initialization_is_exact_and_rng_preserving(self):
        torch.manual_seed(23)
        before = torch.get_rng_state().clone()
        reverse = make_identity_reverse_projector(12, 5, seed=1993001)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        values = torch.randn(7, 12)
        self.assertTrue(torch.equal(reverse(values), values))
        self.assertTrue(torch.count_nonzero(reverse.up.weight) == 0)
        self.assertTrue(torch.count_nonzero(reverse.up.bias) == 0)

    def test_one_minibatch_forward_backward_optimizer_step_is_finite(self):
        current, old, projector, reverse, z_new, z_old = self.feature_modules()
        classifier = nn.Linear(6, 3)
        targets = torch.arange(z_new.shape[0]) % 3
        terms = bialign_loss_terms(z_new, z_old, projector, reverse)
        loss_cos = F.cross_entropy(classifier(z_new), targets)
        loss_orth = projector(z_old).mean()
        total = loss_cos + terms["loss_bialign"] + 0.4 * loss_orth
        self.assertTrue(torch.isfinite(total))
        optimizer = optim.SGD([
            {"params": current.parameters(), "lr": 0.01},
            {"params": classifier.parameters(), "lr": 0.01},
            {"params": projector.parameters(), "lr": 5.611742230603744e-4},
            {"params": reverse.parameters(), "lr": 5.611742230603744e-4},
        ], momentum=0.9)
        optimizer.zero_grad()
        total.backward()
        for module in (current, classifier, projector, reverse):
            self.assertTrue(all_gradients_finite(module))
        self.assertTrue(module_gradient_record(current)["has_gradient"])
        self.assertTrue(module_gradient_record(projector)["has_gradient"])
        self.assertTrue(module_gradient_record(reverse)["has_gradient"])
        optimizer.step()
        for module in (current, classifier, projector, reverse):
            self.assertTrue(all(
                bool(torch.isfinite(parameter).all())
                for parameter in module.parameters()))

    def test_bialign_checkpoint_roundtrip_and_missing_D_fail_closed(self):
        source = Learner.__new__(Learner)
        source.bicyc_mode = BIALIGN_MODE
        source._cur_task = 1
        source._device = torch.device("cpu")
        source._network = SimpleNamespace(feature_dim=8)
        source.transport_init_seed = 1993000
        source.bialign_init_seed = 1993000
        source.bialign_reverse_hidden_dim = 4
        source.forward_transport = None
        source.backward_transport = None
        source.reverse_projector = make_identity_reverse_projector(8, 4, 1993001)
        with torch.no_grad():
            source.reverse_projector.up.weight[0, 0] = 0.25
        source.transport_initial_hashes = {"D_t": {"seed": 1993001}}
        source.transport_history = [{"task": 1, "mode": BIALIGN_MODE}]
        source.pre_ca_metrics = {"1": {"metrics": {"top1": 1.0}}}
        source.experiment_records = [{"task": 0}]
        source.ca_rescue_diagnostics = None
        payload = source._checkpoint_extra_state()
        self.assertIn("bialign_reverse_projector_state_dict", payload)

        restored = Learner.__new__(Learner)
        restored.bicyc_mode = BIALIGN_MODE
        restored._cur_task = 1
        restored._device = torch.device("cpu")
        restored._network = SimpleNamespace(feature_dim=8)
        restored.transport_init_seed = 1993000
        restored.bialign_init_seed = 1993000
        restored.bialign_reverse_hidden_dim = 4
        restored._restore_transport_state(payload)
        self.assertEqual(
            module_state_sha256(source.reverse_projector),
            module_state_sha256(restored.reverse_projector))
        self.assertEqual(restored.transport_history, source.transport_history)

        missing = dict(payload)
        missing.pop("bialign_reverse_projector_state_dict")
        with self.assertRaisesRegex(ValueError, "missing reverse projector"):
            restored._restore_transport_state(missing)

    def test_D_is_reset_to_identity_for_each_transition(self):
        learner = Learner.__new__(Learner)
        learner.bicyc_mode = BIALIGN_MODE
        learner._device = torch.device("cpu")
        learner._network = SimpleNamespace(feature_dim=6)
        learner.transport_init_seed = 1993000
        learner.bialign_init_seed = 1993000
        learner.bialign_reverse_hidden_dim = 4
        learner._cur_task = 1
        learner._initialize_transition_maps()
        first_down = learner.reverse_projector.down.weight.detach().clone()
        with torch.no_grad():
            learner.reverse_projector.up.weight.fill_(1.0)
        learner._cur_task = 2
        learner._initialize_transition_maps()
        values = torch.randn(3, 6)
        self.assertTrue(torch.equal(learner.reverse_projector(values), values))
        self.assertFalse(torch.equal(
            first_down, learner.reverse_projector.down.weight.detach()))
        self.assertEqual(
            learner.transport_initial_hashes["D_t"]["lifecycle"],
            "identity_reset_each_incremental_transition")

    def test_official_loss_and_statistics_path_regression(self):
        learner = Learner.__new__(Learner)
        learner.old_ae = nn.Linear(4, 4)
        learner._class_means = np.random.RandomState(3).randn(5, 4)
        learner._device = torch.device("cpu")
        learner.args = {"beta": 1.7, "gamma": 0.4}
        learner.bicyc_mode = "official"
        learner._cur_task = 1
        learner.forward_transport = object()
        learner.backward_transport = object()
        learner.reverse_projector = object()
        learner.transport_initial_hashes = {"sentinel": True}
        learner._initialize_transition_maps()
        self.assertIsNone(learner.forward_transport)
        self.assertIsNone(learner.backward_transport)
        self.assertIsNone(learner.reverse_projector)
        self.assertEqual(learner.transport_initial_hashes, {})
        z_new = torch.randn(6, 4)
        z_old = torch.randn(6, 4)
        mapped = learner.old_ae(z_old)
        expected_align = F.mse_loss(z_new, mapped)
        old_norm = F.normalize(mapped, p=2, dim=1)
        prototypes = learner.old_ae(
            torch.from_numpy(learner._class_means).float())
        expected_orth = (F.normalize(prototypes, p=2, dim=1) @ old_norm.T).mean()
        expected = 1.7 * expected_align + 0.4 * expected_orth
        got, got_align, got_orth = learner._inc_loss_components(z_new, z_old)
        self.assertTrue(torch.equal(got, expected))
        self.assertTrue(torch.equal(got_align, expected_align))
        self.assertTrue(torch.equal(got_orth, expected_orth))
        self.assertTrue(learner._uses_official_statistics_path())
        learner.bicyc_mode = BIALIGN_MODE
        self.assertTrue(learner._uses_official_statistics_path())
        learner.bicyc_mode = "forward"
        self.assertFalse(learner._uses_official_statistics_path())

    def test_bialign_loss_scale_is_beta_times_sum_without_lambda_or_division(self):
        learner = Learner.__new__(Learner)
        learner.old_ae = nn.Linear(4, 4)
        learner.reverse_projector = make_identity_reverse_projector(4, 3, 5)
        learner._class_means = np.random.RandomState(7).randn(5, 4)
        learner._device = torch.device("cpu")
        learner.args = {"beta": 1.0, "gamma": 0.4}
        z_new = torch.randn(6, 4, requires_grad=True)
        z_old = torch.randn(6, 4)
        loss, terms, orth = learner._bialign_loss_components(z_new, z_old)
        expected = terms["loss_fwd"] + terms["loss_back"] + 0.4 * orth
        self.assertTrue(torch.equal(loss, expected))
        self.assertTrue(torch.equal(
            terms["loss_bialign"], terms["loss_fwd"] + terms["loss_back"]))

    def test_full_configs_are_matched_except_mode_and_output_identity(self):
        official = json.loads(
            (ROOT / "exps" / "RSIAT_BiAlign_official.json").read_text())
        bialign = json.loads(
            (ROOT / "exps" / "RSIAT_BiAlign.json").read_text())
        ignored = {
            "arm", "bicyc_mode", "metrics_output", "output_root",
            "per_class_output", "prefix", "progress_path", "scientific_label",
            "task_metrics_output", "transport_metrics_output",
        }
        self.assertEqual(
            {k: v for k, v in official.items() if k not in ignored},
            {k: v for k, v in bialign.items() if k not in ignored})
        self.assertEqual(official["bicyc_mode"], "official")
        self.assertEqual(bialign["bicyc_mode"], BIALIGN_MODE)
        self.assertEqual(official["stats_batch_size"], 32)
        self.assertEqual(bialign["stats_batch_size"], 32)
        self.assertTrue(official["ssca"] and bialign["ssca"])
        self.assertTrue(official["ca"] and bialign["ca"])
        self.assertEqual(official["ca_sampling_mode"], "analytic_transport")
        self.assertEqual(bialign["ca_sampling_mode"], "analytic_transport")
        self.assertFalse(official["ca_covariance_pd_fallback"])
        self.assertFalse(bialign["ca_covariance_pd_fallback"])
        self.assertNotIn("lambda_bi", bialign)
        self.assertNotIn("lambda_cycle", bialign)


if __name__ == "__main__":
    unittest.main()
