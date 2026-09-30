import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.RSIAT_adapter import (
    BIALIGN_CYCLE_MODE,
    BIALIGN_MODE,
    BIALIGN_MODES,
    TRACK_B_TRANSPORT_MODES,
    VALID_EXPERIMENT_MODES,
    Learner,
)
from utils.bialign import (
    bialign_cycle_loss_terms,
    bialign_loss_terms,
    make_identity_reverse_projector,
)
from utils.bicyc_transport import module_state_sha256


ROOT = Path(__file__).resolve().parents[1]


def has_nonzero_gradient(module):
    return any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in module.parameters()
    )


class BiAlignCycleTests(unittest.TestCase):
    def setUp(self):
        random.seed(29)
        np.random.seed(29)
        torch.manual_seed(29)

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

    def assert_routes(self, loss_name, expected):
        current, old, projector, reverse, z_new, z_old = self.feature_modules()
        terms = bialign_cycle_loss_terms(
            z_new, z_old, projector, reverse)
        terms[loss_name].backward()
        observed = {
            "P_t": has_nonzero_gradient(projector),
            "D_t": has_nonzero_gradient(reverse),
            "current": has_nonzero_gradient(current),
            "old": has_nonzero_gradient(old),
        }
        self.assertEqual(observed, expected)
        self.assertIsNone(z_old.grad)

    def test_L_fwd_gradient_routing(self):
        self.assert_routes("loss_fwd", {
            "P_t": True, "D_t": False, "current": False, "old": False})

    def test_L_back_gradient_routing(self):
        self.assert_routes("loss_back", {
            "P_t": False, "D_t": True, "current": True, "old": False})

    def test_L_cycle_new_gradient_routing(self):
        self.assert_routes("cycle_new", {
            "P_t": True, "D_t": True, "current": True, "old": False})

    def test_L_cycle_old_gradient_routing(self):
        self.assert_routes("cycle_old", {
            "P_t": True, "D_t": True, "current": False, "old": False})

    def test_combined_cycle_optimizer_step_is_finite(self):
        current, old, projector, reverse, z_new, z_old = self.feature_modules()
        classifier = nn.Linear(6, 3)
        targets = torch.arange(z_new.shape[0]) % 3
        terms = bialign_cycle_loss_terms(
            z_new, z_old, projector, reverse)
        loss_cos = F.cross_entropy(classifier(z_new), targets)
        loss_orth = projector(z_old).mean()
        total = (
            loss_cos + terms["loss_bialign"] + 0.4 * loss_orth
            + terms["loss_cycle"])
        self.assertTrue(torch.isfinite(total))
        optimizer = torch.optim.SGD([
            {"params": current.parameters(), "lr": 0.01},
            {"params": classifier.parameters(), "lr": 0.01},
            {"params": projector.parameters(), "lr": 5.611742230603744e-4},
            {"params": reverse.parameters(), "lr": 5.611742230603744e-4},
        ], momentum=0.9)
        optimizer.zero_grad()
        total.backward()
        for module in (current, classifier, projector, reverse):
            self.assertTrue(has_nonzero_gradient(module))
            self.assertTrue(all(
                parameter.grad is None
                or bool(torch.isfinite(parameter.grad).all())
                for parameter in module.parameters()))
        self.assertFalse(has_nonzero_gradient(old))
        optimizer.step()
        for module in (current, classifier, projector, reverse):
            self.assertTrue(all(
                bool(torch.isfinite(parameter).all())
                for parameter in module.parameters()))

    def test_cycle_sum_and_total_loss_are_exact(self):
        learner = Learner.__new__(Learner)
        learner.old_ae = nn.Linear(4, 4)
        learner.reverse_projector = make_identity_reverse_projector(4, 3, 5)
        learner._class_means = np.random.RandomState(7).randn(5, 4)
        learner._device = torch.device("cpu")
        learner.args = {"beta": 1.3, "gamma": 0.4}
        learner.lambda_cycle = 0.7
        z_new = torch.randn(6, 4, requires_grad=True)
        z_old = torch.randn(6, 4)
        loss, terms, orth = learner._bialign_cycle_loss_components(z_new, z_old)
        self.assertTrue(torch.equal(
            terms["loss_bialign"], terms["loss_fwd"] + terms["loss_back"]))
        self.assertTrue(torch.equal(
            terms["loss_cycle"], terms["cycle_new"] + terms["cycle_old"]))
        expected = (
            1.3 * (terms["loss_fwd"] + terms["loss_back"])
            + 0.4 * orth
            + 0.7 * (terms["cycle_new"] + terms["cycle_old"]))
        self.assertTrue(torch.equal(loss, expected))

    def test_plain_bialign_loss_is_numerically_unchanged_and_cycle_free(self):
        learner = Learner.__new__(Learner)
        learner.old_ae = nn.Linear(4, 4)
        learner.reverse_projector = make_identity_reverse_projector(4, 3, 5)
        learner._class_means = np.random.RandomState(11).randn(5, 4)
        learner._device = torch.device("cpu")
        learner.args = {"beta": 1.0, "gamma": 0.4}
        learner.lambda_cycle = 12345.0
        z_new = torch.randn(6, 4, requires_grad=True)
        z_old = torch.randn(6, 4)
        direct = bialign_loss_terms(
            z_new, z_old, learner.old_ae, learner.reverse_projector)
        loss, terms, orth = learner._bialign_loss_components(z_new, z_old)
        self.assertEqual(set(terms), {
            "mapped_old", "loss_fwd", "loss_back", "loss_bialign"})
        self.assertTrue(torch.equal(terms["loss_fwd"], direct["loss_fwd"]))
        self.assertTrue(torch.equal(terms["loss_back"], direct["loss_back"]))
        self.assertTrue(torch.equal(
            loss, terms["loss_fwd"] + terms["loss_back"] + 0.4 * orth))

    def test_D_uses_the_exact_same_optimizer_scale_as_P(self):
        class TinyNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.convnet = nn.Linear(4, 4)
                self.fc = nn.Linear(4, 3)

        learner = Learner.__new__(Learner)
        learner._network = TinyNetwork()
        learner._device = torch.device("cpu")
        learner._cur_task = 1
        learner.old_ae = nn.Linear(4, 4)
        learner.reverse_projector = make_identity_reverse_projector(4, 3, 5)
        learner.forward_transport = None
        learner.backward_transport = None
        learner.init_lr = 0.018
        learner.weight_decay = 0.00017
        learner.min_lr = 0.0
        learner.args = {
            "inc_epochs": 30,
            "init_lr": 0.018,
            "weight_decay": 0.00017,
            "ae_init_lr": 0.00056,
            "ae_weight_decay": 0.0029,
            "optimizer": "sgd",
            "warmup_epoch": 7,
        }
        captured = {}

        def capture(_train, _test, optimizer, _scheduler, _warmup):
            captured["groups"] = optimizer.param_groups

        learner._init_train = capture
        learner._train(None, None)
        groups = captured["groups"]
        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[2]["lr"], groups[3]["lr"])
        self.assertEqual(groups[2]["weight_decay"], groups[3]["weight_decay"])
        self.assertEqual(groups[3]["name"], "D_t")

    def test_plain_bialign_cycle_metrics_are_zero_and_cycle_mode_is_active(self):
        class TinyExtractor(nn.Module):
            def __init__(self, dimension, classes):
                super().__init__()
                self.encoder = nn.Linear(dimension, dimension, bias=False)
                self.head = nn.Linear(dimension, classes)

            def extract_vector(self, values):
                return self.encoder(values)

            def fc(self, values):
                return {"logits": self.head(values)}

        def make_learner(mode):
            learner = Learner.__new__(Learner)
            learner.bicyc_mode = mode
            learner._cur_task = 1
            learner._known_classes = 0
            learner._device = torch.device("cpu")
            learner.args = {
                "scale": 1.0, "margin": 0.0, "beta": 1.0, "gamma": 0.4}
            learner.lambda_cycle = 1.0
            learner._class_means = np.random.RandomState(13).randn(5, 4)
            learner._network_module_ptr = TinyExtractor(4, 3)
            learner.old_network_module_ptr = TinyExtractor(4, 3).requires_grad_(False)
            learner.old_ae = nn.Linear(4, 4)
            learner.reverse_projector = make_identity_reverse_projector(4, 3, 5)
            return learner

        inputs = torch.randn(6, 4)
        targets = torch.arange(6) % 3
        plain = make_learner(BIALIGN_MODE)
        _, _, _, plain_details = plain._compute_rt_loss(inputs, targets)
        self.assertEqual(float(plain_details["cycle_new"]), 0.0)
        self.assertEqual(float(plain_details["cycle_old"]), 0.0)
        self.assertEqual(float(plain_details["cycle"]), 0.0)
        cycle = make_learner(BIALIGN_CYCLE_MODE)
        _, _, _, cycle_details = cycle._compute_rt_loss(inputs, targets)
        self.assertGreater(float(cycle_details["cycle_new"].detach()), 0.0)
        self.assertGreater(float(cycle_details["cycle_old"].detach()), 0.0)
        self.assertTrue(torch.equal(
            cycle_details["cycle"],
            cycle_details["cycle_new"] + cycle_details["cycle_old"]))
        self.assertEqual(float(cycle_details["transport"]), 0.0)

    def test_lambda_cycle_zero_matches_plain_bialign_for_same_state(self):
        learner = Learner.__new__(Learner)
        learner.old_ae = nn.Linear(4, 4)
        learner.reverse_projector = make_identity_reverse_projector(4, 3, 5)
        learner._class_means = np.random.RandomState(19).randn(5, 4)
        learner._device = torch.device("cpu")
        learner.args = {"beta": 1.3, "gamma": 0.4}
        learner.lambda_cycle = 0.0
        z_new = torch.randn(6, 4, requires_grad=True)
        z_old = torch.randn(6, 4)
        plain_loss, plain_terms, plain_orth = learner._bialign_loss_components(
            z_new, z_old)
        cycle_loss, cycle_terms, cycle_orth = (
            learner._bialign_cycle_loss_components(z_new, z_old))
        self.assertTrue(torch.equal(plain_loss, cycle_loss))
        self.assertTrue(torch.equal(plain_orth, cycle_orth))
        for key in ("mapped_old", "loss_fwd", "loss_back", "loss_bialign"):
            self.assertTrue(torch.equal(plain_terms[key], cycle_terms[key]))

    def test_bialign_and_cycle_modes_initialize_identical_D_state(self):
        hashes = {}
        for mode in (BIALIGN_MODE, BIALIGN_CYCLE_MODE):
            learner = Learner.__new__(Learner)
            learner.bicyc_mode = mode
            learner._device = torch.device("cpu")
            learner._network = SimpleNamespace(feature_dim=6)
            learner.transport_init_seed = 1993000
            learner.bialign_init_seed = 1993000
            learner.bialign_reverse_hidden_dim = 4
            learner._cur_task = 3
            learner._initialize_transition_maps()
            hashes[mode] = module_state_sha256(learner.reverse_projector)
            self.assertIsNone(learner.forward_transport)
            self.assertIsNone(learner.backward_transport)
        self.assertEqual(hashes[BIALIGN_MODE], hashes[BIALIGN_CYCLE_MODE])

    def test_mode_uses_bialign_lifecycle_and_official_statistics_only(self):
        self.assertIn(BIALIGN_CYCLE_MODE, BIALIGN_MODES)
        self.assertIn(BIALIGN_CYCLE_MODE, VALID_EXPERIMENT_MODES)
        self.assertNotIn(BIALIGN_CYCLE_MODE, TRACK_B_TRANSPORT_MODES)
        learner = Learner.__new__(Learner)
        learner.bicyc_mode = BIALIGN_CYCLE_MODE
        learner._device = torch.device("cpu")
        learner._network = SimpleNamespace(feature_dim=6)
        learner.transport_init_seed = 1993000
        learner.bialign_init_seed = 1993000
        learner.bialign_reverse_hidden_dim = 4
        learner._cur_task = 1
        learner._initialize_transition_maps()
        self.assertIsNotNone(learner.reverse_projector)
        self.assertIsNone(learner.forward_transport)
        self.assertIsNone(learner.backward_transport)
        self.assertTrue(learner._uses_official_statistics_path())
        self.assertEqual(learner._fine_tune_forward_transport(), [])
        first_down = learner.reverse_projector.down.weight.detach().clone()
        with torch.no_grad():
            learner.reverse_projector.up.weight.fill_(1.0)
        learner._cur_task = 2
        learner._initialize_transition_maps()
        values = torch.randn(3, 6)
        self.assertTrue(torch.equal(learner.reverse_projector(values), values))
        self.assertFalse(torch.equal(
            first_down, learner.reverse_projector.down.weight.detach()))

    def test_checkpoint_roundtrip_and_missing_D_fail_closed(self):
        source = Learner.__new__(Learner)
        source.bicyc_mode = BIALIGN_CYCLE_MODE
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
        source.transport_history = [{"task": 1, "mode": BIALIGN_CYCLE_MODE}]
        source.pre_ca_metrics = {}
        source.experiment_records = []
        source.ca_rescue_diagnostics = None
        payload = source._checkpoint_extra_state()
        self.assertIn("bialign_reverse_projector_state_dict", payload)
        self.assertNotIn("forward_transport_state_dict", payload)
        self.assertNotIn("backward_transport_state_dict", payload)

        restored = Learner.__new__(Learner)
        restored.bicyc_mode = BIALIGN_CYCLE_MODE
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
        self.assertIsNone(restored.forward_transport)
        self.assertIsNone(restored.backward_transport)

        missing = dict(payload)
        missing.pop("bialign_reverse_projector_state_dict")
        with self.assertRaisesRegex(ValueError, "missing reverse projector"):
            restored._restore_transport_state(missing)

        malformed = dict(payload)
        malformed["forward_transport_state_dict"] = {
            "weight": torch.eye(8), "bias": torch.zeros(8)}
        with self.assertRaisesRegex(ValueError, "unexpected Track-B A/D state"):
            restored._restore_transport_state(malformed)
        self.assertIsNone(restored.forward_transport)
        self.assertIsNone(restored.backward_transport)

    def test_config_is_minimal_matched_extension(self):
        bialign = json.loads(
            (ROOT / "exps" / "RSIAT_BiAlign.json").read_text())
        cycle = json.loads(
            (ROOT / "exps" / "RSIAT_BiAlign_Cycle.json").read_text())
        ignored = {
            "arm", "bicyc_mode", "lambda_cycle", "metrics_output",
            "output_root", "per_class_output", "prefix", "progress_path",
            "scientific_label", "task_metrics_output",
            "transport_metrics_output",
        }
        self.assertEqual(
            {k: v for k, v in bialign.items() if k not in ignored},
            {k: v for k, v in cycle.items() if k not in ignored})
        self.assertEqual(bialign["bicyc_mode"], BIALIGN_MODE)
        self.assertEqual(cycle["bicyc_mode"], BIALIGN_CYCLE_MODE)
        self.assertNotIn("lambda_cycle", bialign)
        self.assertEqual(cycle["lambda_cycle"], 1.0)
        self.assertNotIn("lambda_bi", cycle)
        self.assertTrue(cycle["ssca"] and cycle["ca"])
        self.assertEqual(cycle["ca_sampling_mode"], "analytic_transport")
        self.assertFalse(cycle["ca_covariance_pd_fallback"])

        bialign_smoke = json.loads(
            (ROOT / "exps" / "RSIAT_BiAlign_smoke.json").read_text())
        cycle_smoke = json.loads(
            (ROOT / "exps" / "RSIAT_BiAlign_Cycle_smoke.json").read_text())
        self.assertEqual(
            {k: v for k, v in bialign_smoke.items() if k not in ignored},
            {k: v for k, v in cycle_smoke.items() if k not in ignored})
        self.assertEqual(cycle_smoke["bicyc_mode"], BIALIGN_CYCLE_MODE)
        self.assertEqual(cycle_smoke["lambda_cycle"], 1.0)
        self.assertEqual(cycle_smoke["inc_epochs"], 1)
        self.assertEqual(cycle_smoke["ca_epochs"], 1)
        self.assertEqual(cycle_smoke["max_tasks_per_run"], 1)
        self.assertEqual(cycle_smoke["stop_after_task"], 1)


if __name__ == "__main__":
    unittest.main()
