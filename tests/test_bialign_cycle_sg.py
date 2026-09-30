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
    BIALIGN_CYCLE_MODES,
    BIALIGN_CYCLE_SG_MODE,
    BIALIGN_MODE,
    BIALIGN_MODES,
    TRACK_B_TRANSPORT_MODES,
    VALID_EXPERIMENT_MODES,
    Learner,
)
from utils.bialign import (
    bialign_cycle_loss_terms,
    loss_gradient_records,
    make_identity_reverse_projector,
)
from utils.bicyc_transport import module_state_sha256
from utils.toolkit import AutoencoderSigmoid


ROOT = Path(__file__).resolve().parents[1]


def has_nonzero_gradient(module):
    return any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in module.parameters()
    )


class BiAlignCycleSGTests(unittest.TestCase):
    def setUp(self):
        random.seed(37)
        np.random.seed(37)
        torch.manual_seed(37)

    @staticmethod
    def feature_modules(dimension=6):
        current = nn.Linear(dimension, dimension, bias=False)
        old = nn.Linear(dimension, dimension, bias=False).requires_grad_(False)
        projector = nn.Linear(dimension, dimension)
        reverse = make_identity_reverse_projector(
            dimension, hidden_dim=4, seed=1993001)
        inputs = torch.randn(8, dimension)
        z_new = current(inputs)
        with torch.no_grad():
            z_old = old(inputs)
        return current, old, projector, reverse, z_new, z_old

    def assert_sg_routes(self, loss_name, expected):
        current, old, projector, reverse, z_new, z_old = self.feature_modules()
        terms = bialign_cycle_loss_terms(
            z_new, z_old, projector, reverse,
            stop_gradient_cycle_new=True)
        terms[loss_name].backward()
        observed = {
            "P_t": has_nonzero_gradient(projector),
            "D_t": has_nonzero_gradient(reverse),
            "current": has_nonzero_gradient(current),
            "old": has_nonzero_gradient(old),
        }
        self.assertEqual(observed, expected)
        self.assertIsNone(z_old.grad)

    def test_existing_cycle_default_path_and_values_are_unchanged(self):
        current, _, projector, reverse, z_new, z_old = self.feature_modules()
        terms = bialign_cycle_loss_terms(z_new, z_old, projector, reverse)
        manual_new = F.mse_loss(
            projector(reverse(z_new)), z_new.detach())
        manual_old = F.mse_loss(
            reverse(projector(z_old.detach())), z_old.detach())
        self.assertTrue(torch.equal(terms["cycle_new"], manual_new))
        self.assertTrue(torch.equal(terms["cycle_old"], manual_old))
        terms["cycle_new"].backward()
        self.assertTrue(has_nonzero_gradient(current))

    def test_sg_cycle_values_match_cycle_and_manual_detached_reference(self):
        _, _, projector, reverse, z_new, z_old = self.feature_modules()
        cycle = bialign_cycle_loss_terms(
            z_new, z_old, projector, reverse)
        sg = bialign_cycle_loss_terms(
            z_new, z_old, projector, reverse,
            stop_gradient_cycle_new=True)
        manual = F.mse_loss(
            projector(reverse(z_new.detach())), z_new.detach())
        for key in ("loss_fwd", "loss_back", "loss_bialign",
                    "cycle_new", "cycle_old", "loss_cycle"):
            self.assertTrue(torch.equal(cycle[key], sg[key]), key)
        self.assertTrue(torch.equal(sg["cycle_new"], manual))

    def test_L_fwd_gradient_routing(self):
        self.assert_sg_routes("loss_fwd", {
            "P_t": True, "D_t": False, "current": False, "old": False})

    def test_L_back_gradient_routing(self):
        self.assert_sg_routes("loss_back", {
            "P_t": False, "D_t": True, "current": True, "old": False})

    def test_cycle_sg_new_gradient_routing(self):
        self.assert_sg_routes("cycle_new", {
            "P_t": True, "D_t": True, "current": False, "old": False})

    def test_cycle_sg_old_gradient_routing(self):
        self.assert_sg_routes("cycle_old", {
            "P_t": True, "D_t": True, "current": False, "old": False})

    def test_combined_cycle_sg_gradient_routing(self):
        self.assert_sg_routes("loss_cycle", {
            "P_t": True, "D_t": True, "current": False, "old": False})

    def test_cycle_only_diagnostic_has_no_current_gradient(self):
        current, old, projector, reverse, z_new, z_old = self.feature_modules()
        terms = bialign_cycle_loss_terms(
            z_new, z_old, projector, reverse,
            stop_gradient_cycle_new=True)
        records = loss_gradient_records(terms["loss_cycle"], {
            "current_adapter": current,
            "P_t": projector,
            "D_t": reverse,
            "old_model": old,
        })
        self.assertEqual(records["current_adapter"], {
            "has_gradient": False, "l2_norm": 0.0, "all_finite": True})
        self.assertEqual(records["old_model"], {
            "has_gradient": False, "l2_norm": 0.0, "all_finite": True})
        for name in ("P_t", "D_t"):
            self.assertTrue(records[name]["has_gradient"])
            self.assertTrue(records[name]["all_finite"])
            self.assertGreater(records[name]["l2_norm"], 0.0)

    def test_cycle_addition_does_not_change_current_full_objective_gradient(self):
        current, old, projector, reverse, z_new, z_old = self.feature_modules()
        classifier = nn.Linear(6, 3)
        targets = torch.arange(z_new.shape[0]) % 3
        terms = bialign_cycle_loss_terms(
            z_new, z_old, projector, reverse,
            stop_gradient_cycle_new=True)
        classification = F.cross_entropy(classifier(z_new), targets)
        orth = projector(z_old.detach()).mean()
        without_cycle = classification + terms["loss_bialign"] + 0.4 * orth
        with_cycle = without_cycle + terms["loss_cycle"]
        parameter = current.weight
        gradient_without = torch.autograd.grad(
            without_cycle, parameter, retain_graph=True)[0]
        gradient_with = torch.autograd.grad(
            with_cycle, parameter, retain_graph=True)[0]
        self.assertTrue(torch.equal(gradient_without, gradient_with))
        self.assertGreater(float(gradient_with.abs().sum()), 0.0)
        projector_gradient, reverse_gradient = torch.autograd.grad(
            with_cycle, (projector.weight, reverse.up.weight))
        self.assertGreater(float(projector_gradient.abs().sum()), 0.0)
        self.assertGreater(float(reverse_gradient.abs().sum()), 0.0)
        self.assertFalse(has_nonzero_gradient(old))

    def test_lambda_zero_matches_plain_bialign(self):
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
        sg_loss, sg_terms, sg_orth = learner._bialign_cycle_loss_components(
            z_new, z_old, stop_gradient_cycle_new=True)
        self.assertTrue(torch.equal(plain_loss, sg_loss))
        self.assertTrue(torch.equal(plain_orth, sg_orth))
        for key in ("mapped_old", "loss_fwd", "loss_back", "loss_bialign"):
            self.assertTrue(torch.equal(plain_terms[key], sg_terms[key]))

    def test_cycle_and_sg_initialize_identical_P_and_D_state(self):
        d_hashes = {}
        p_hashes = {}
        for mode in (BIALIGN_CYCLE_MODE, BIALIGN_CYCLE_SG_MODE):
            learner = Learner.__new__(Learner)
            learner.bicyc_mode = mode
            learner._device = torch.device("cpu")
            learner._network = SimpleNamespace(feature_dim=6)
            learner.transport_init_seed = 1993000
            learner.bialign_init_seed = 1993000
            learner.bialign_reverse_hidden_dim = 4
            learner._cur_task = 3
            torch.manual_seed(73)
            learner.old_ae = AutoencoderSigmoid(
                input_dims=6, code_dims=3, residual_mode="sigmoid")
            p_hashes[mode] = module_state_sha256(learner.old_ae)
            learner._initialize_transition_maps()
            d_hashes[mode] = module_state_sha256(learner.reverse_projector)
            values = torch.randn(3, 6)
            self.assertTrue(torch.equal(learner.reverse_projector(values), values))
            self.assertIsNone(learner.forward_transport)
            self.assertIsNone(learner.backward_transport)
        self.assertEqual(p_hashes[BIALIGN_CYCLE_MODE],
                         p_hashes[BIALIGN_CYCLE_SG_MODE])
        self.assertEqual(d_hashes[BIALIGN_CYCLE_MODE],
                         d_hashes[BIALIGN_CYCLE_SG_MODE])

    def test_sg_mode_uses_official_path_and_resets_D(self):
        self.assertIn(BIALIGN_CYCLE_SG_MODE, BIALIGN_CYCLE_MODES)
        self.assertIn(BIALIGN_CYCLE_SG_MODE, BIALIGN_MODES)
        self.assertIn(BIALIGN_CYCLE_SG_MODE, VALID_EXPERIMENT_MODES)
        self.assertNotIn(BIALIGN_CYCLE_SG_MODE, TRACK_B_TRANSPORT_MODES)
        learner = Learner.__new__(Learner)
        learner.bicyc_mode = BIALIGN_CYCLE_SG_MODE
        learner._device = torch.device("cpu")
        learner._network = SimpleNamespace(feature_dim=6)
        learner.transport_init_seed = 1993000
        learner.bialign_init_seed = 1993000
        learner.bialign_reverse_hidden_dim = 4
        learner._cur_task = 1
        learner._initialize_transition_maps()
        self.assertTrue(learner._uses_official_statistics_path())
        self.assertEqual(learner._fine_tune_forward_transport(), [])
        with torch.no_grad():
            learner.reverse_projector.up.weight.fill_(1.0)
        learner._cur_task = 2
        learner._initialize_transition_maps()
        values = torch.randn(3, 6)
        self.assertTrue(torch.equal(learner.reverse_projector(values), values))
        self.assertIsNone(learner.forward_transport)
        self.assertIsNone(learner.backward_transport)

    def test_checkpoint_roundtrip_and_fail_closed_state(self):
        source = Learner.__new__(Learner)
        source.bicyc_mode = BIALIGN_CYCLE_SG_MODE
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
        source.transport_history = [{"task": 1, "mode": BIALIGN_CYCLE_SG_MODE}]
        source.pre_ca_metrics = {}
        source.experiment_records = []
        source.ca_rescue_diagnostics = None
        payload = source._checkpoint_extra_state()

        restored = Learner.__new__(Learner)
        restored.bicyc_mode = BIALIGN_CYCLE_SG_MODE
        restored._cur_task = 1
        restored._device = torch.device("cpu")
        restored._network = SimpleNamespace(feature_dim=8)
        restored.transport_init_seed = 1993000
        restored.bialign_init_seed = 1993000
        restored.bialign_reverse_hidden_dim = 4
        restored._restore_transport_state(payload)
        self.assertEqual(module_state_sha256(source.reverse_projector),
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

    def test_one_optimizer_step_is_finite_with_expected_groups(self):
        current, old, projector, reverse, z_new, z_old = self.feature_modules()
        classifier = nn.Linear(6, 3)
        targets = torch.arange(z_new.shape[0]) % 3
        terms = bialign_cycle_loss_terms(
            z_new, z_old, projector, reverse,
            stop_gradient_cycle_new=True)
        total = (F.cross_entropy(classifier(z_new), targets)
                 + terms["loss_bialign"] + 0.4 * projector(z_old).mean()
                 + terms["loss_cycle"])
        groups = [
            {"params": current.parameters(), "lr": 0.01},
            {"params": classifier.parameters(), "lr": 0.01},
            {"params": projector.parameters(), "lr": 5.611742230603744e-4},
            {"params": reverse.parameters(), "lr": 5.611742230603744e-4},
        ]
        optimizer = torch.optim.SGD(groups, momentum=0.9)
        self.assertEqual(len(optimizer.param_groups), 4)
        self.assertEqual(optimizer.param_groups[2]["lr"],
                         optimizer.param_groups[3]["lr"])
        optimizer.zero_grad()
        total.backward()
        for module in (current, classifier, projector, reverse):
            self.assertTrue(has_nonzero_gradient(module))
            self.assertTrue(all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in module.parameters()))
        self.assertFalse(has_nonzero_gradient(old))
        optimizer.step()
        for module in (current, classifier, projector, reverse):
            self.assertTrue(all(bool(torch.isfinite(parameter).all())
                                for parameter in module.parameters()))

    def test_sg_uses_P_and_D_optimizer_groups_without_track_b_groups(self):
        class TinyNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.convnet = nn.Linear(4, 4)
                self.fc = nn.Linear(4, 3)

        learner = Learner.__new__(Learner)
        learner.bicyc_mode = BIALIGN_CYCLE_SG_MODE
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
        learner.transport_stage1_lr = 0.05
        learner.transport_stage1_weight_decay = 0.0001
        learner.args = {
            "inc_epochs": 30, "init_lr": 0.018,
            "weight_decay": 0.00017, "ae_init_lr": 0.00056,
            "ae_weight_decay": 0.0029, "optimizer": "sgd",
            "warmup_epoch": 7,
        }
        captured = {}
        learner._init_train = lambda _a, _b, optimizer, _c, _d: (
            captured.update(groups=optimizer.param_groups))
        learner._train(None, None)
        groups = captured["groups"]
        self.assertEqual(len(groups), 4)
        self.assertEqual(groups[2]["lr"], groups[3]["lr"])
        self.assertEqual(groups[2]["weight_decay"], groups[3]["weight_decay"])
        self.assertEqual(groups[3]["name"], "D_t")
        self.assertNotIn("A", [group.get("name") for group in groups])
        self.assertNotIn("D", [group.get("name") for group in groups])

    def test_checkpoint_metadata_preserves_sg_mode_and_rejects_cycle_mode(self):
        learner = Learner.__new__(Learner)
        learner.args = {
            "dataset": "cifar224", "seed": 1993, "init_cls": 10,
            "increment": 10, "model_name": "adapter",
            "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
            "ae_residual_mode": "sigmoid",
            "bicyc_mode": BIALIGN_CYCLE_SG_MODE,
            "lambda_cycle": 1.0,
            "bialign_reverse_hidden_dim": 64,
            "bialign_init_seed": 1993000,
            "stats_batch_size": 32,
            "ca_covariance_pd_fallback": False,
        }
        learner.class_order = None
        metadata = learner._checkpoint_run_metadata()
        self.assertEqual(metadata["bicyc_mode"], BIALIGN_CYCLE_SG_MODE)
        self.assertEqual(metadata["lambda_cycle"], 1.0)
        wrong = dict(metadata)
        wrong["bicyc_mode"] = BIALIGN_CYCLE_MODE
        with self.assertRaisesRegex(ValueError, "bicyc_mode"):
            learner._validate_checkpoint({"run_metadata": wrong})

    def test_config_diff_is_identity_and_mode_only(self):
        cycle = json.loads(
            (ROOT / "exps" / "RSIAT_BiAlign_Cycle.json").read_text())
        sg = json.loads(
            (ROOT / "exps" / "RSIAT_BiAlign_Cycle_SG.json").read_text())
        expected_differences = {
            "arm", "bicyc_mode", "metrics_output", "output_root",
            "per_class_output", "prefix", "progress_path", "scientific_label",
            "task_metrics_output", "transport_metrics_output",
        }
        differences = {key for key in set(cycle) | set(sg)
                       if cycle.get(key) != sg.get(key)}
        self.assertEqual(differences, expected_differences)
        self.assertEqual(sg["bicyc_mode"], BIALIGN_CYCLE_SG_MODE)
        self.assertEqual(cycle["lambda_cycle"], sg["lambda_cycle"])
        self.assertEqual(sg["lambda_cycle"], 1.0)
        self.assertIn("[IMPLEMENTATION ABLATION]", sg["scientific_label"])


if __name__ == "__main__":
    unittest.main()
