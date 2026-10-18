"""CPU fixtures validate contracts; these are not dataset benchmark results."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.RSIAT_adapter import Learner, RS_Loss
from network.classifier import SimpleContinualLinear
from utils.loss import AngularPenaltySMLoss
from utils.quantum_kernel import QuantumKernelModule
from utils.qbridge import (BridgeRuntime, MonotoneCalibrator, config_from_args, cross_margin_loss,
                           guard_loss, id_positions, margin_violations, substitute_active_risk)
from utils.qbridge_calibration import binary_metrics, fit_calibrator, stratified_split


def metric():
    model = QuantumKernelModule(input_dim=4, num_qubits=2, num_layers=1, init_seed=71)
    model.calibrate_gamma(torch.tensor([[1., 2, 3, 4], [4., 1, 2, 3], [2., 4, 1, 3]]))
    model.set_inc_mode("frozen")
    return model


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Linear(4, 4)
        self.fc = SimpleContinualLinear(4, 2)
        self.fc.update(2, freeze_old=False)

    def extract_vector(self, x):
        return self.convnet(x)

    def forward(self, x):
        return self.fc(self.extract_vector(x))

    @property
    def feature_dim(self):
        return 4

    def update_fc(self, count):
        if self.fc is None:
            self.fc = SimpleContinualLinear(4, count)
        else:
            self.fc.update(count, freeze_old=False)

    def ca_forward(self, features):
        return self.fc(features)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        return self.requires_grad_(False).eval()


def learner_fixture(enabled=True, variant="custom", **overrides):
    torch.manual_seed(73)
    learner = Learner.__new__(Learner)
    learner.args = {"beta": 1.3, "gamma": .4, "scale": 20., "margin": .1,
                    "lambda_rs": .2, "q_inc_pair": "old_proj", "inc_loss_mode": "mean",
                    "use_quantum_kernel_base": True, "use_quantum_kernel_inc": not enabled}
    qb = dict(enabled=enabled, variant=variant, risk_mode="raw_pqk", lambda_cross=.1,
              lambda_guard=.1, risk_gain=.5, margin_x=.5, margin_old=.5)
    qb.update(overrides)
    learner.args["qbridge"] = qb
    learner.qbridge_config = config_from_args(learner.args)
    learner._network = TinyNet()
    learner._network_module_ptr = learner._network
    learner.old_network_module_ptr = copy.deepcopy(learner._network).requires_grad_(False)
    learner.old_ae = nn.Linear(4, 4)
    learner.quantum_kernel = metric()
    learner.use_quantum_kernel_base = True
    learner.use_quantum_kernel_inc = not enabled
    learner.rs_loss_func = RS_Loss(.5, .3)
    learner._class_means = np.array([[1., 0, 0, 0], [0., 1, 0, 0]], dtype=np.float32)
    learner._device = torch.device("cpu")
    learner._cur_task = 1
    learner._known_classes = 2
    learner._total_classes = 4
    learner.class_order = [70, 20, 50, 10]
    learner._qb_seen_ids = learner.class_order
    learner._qb_old_ids = [70, 20]
    learner._qb_new_ids = [50, 10]
    learner.qbridge_runtime = None
    if enabled:
        learner.qbridge_runtime = BridgeRuntime(learner.qbridge_config, learner.quantum_kernel, "base", 3, "cpu")
        learner.qbridge_runtime.start_task(torch.from_numpy(learner._class_means), [70, 20], "s0", "s0")
    return learner


class QBridgeMathTests(unittest.TestCase):
    def test_fixed_denominator_is_continuous(self):
        for epsilon in (1e-3, 1e-6, 0.):
            violations = torch.tensor([[.2, epsilon]], dtype=torch.float64, requires_grad=True)
            loss = cross_margin_loss(violations)
            self.assertAlmostEqual(loss.item(), (.04 + epsilon**2)/2)
            loss.backward()
            self.assertAlmostEqual(violations.grad[0, 0].item(), .2)

    def test_empty_and_inactive_pairs_have_zero_gradient(self):
        for count in (0, 1, 200):
            v = torch.zeros(1, count, requires_grad=True)
            loss = cross_margin_loss(v)
            loss.backward()
            self.assertEqual(loss.item(), 0)
            self.assertEqual(v.grad.abs().sum().item(), 0)

    def test_zero_gain_and_risk_detachment(self):
        v = torch.tensor([[.1, .3]], requires_grad=True)
        r = torch.tensor([[.2, .9]], requires_grad=True)
        torch.testing.assert_close(cross_margin_loss(v, r, 0), cross_margin_loss(v))
        cross_margin_loss(v, r, .5).backward()
        self.assertIsNone(r.grad)

    def test_permuted_global_ids_and_raw_cosine(self):
        head = SimpleContinualLinear(4, 2)
        head.update(2, freeze_old=False)
        x = torch.randn(3, 4, requires_grad=True)
        ids = [40, 2, 90, 7]
        score = head.qbridge_scores(x, ids, [90, 40])
        torch.testing.assert_close(score, head(x)["logits"])
        self.assertTrue((score.abs() <= 1+1e-6).all())
        actual = margin_violations(score, torch.tensor([7, 2, 7]), ids, [90, 40], .1)
        expected = (.1 + score[:, [2, 0]] - score[torch.arange(3), [3, 1, 3]][:, None]).relu()
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            id_positions([999], ids, "cpu")

    def test_shuffle_and_constant_active_only(self):
        risk = torch.arange(10.).reshape(2, 5)/10
        active = risk > .3
        result = substitute_active_risk(risk, active, "shuffled_raw_pqk", torch.Generator().manual_seed(3))
        torch.testing.assert_close(result[active].sort().values, risk[active].sort().values)
        self.assertFalse(torch.equal(result[active], risk[active]))
        torch.testing.assert_close(result[~active], risk[~active])
        repeat = substitute_active_risk(risk, active, "shuffled_raw_pqk", torch.Generator().manual_seed(3))
        torch.testing.assert_close(result, repeat)
        const = substitute_active_risk(risk, active, "constant_raw_pqk", torch.Generator())
        torch.testing.assert_close(const[active], risk[active].mean().expand_as(risk[active]))
        for mask in (torch.zeros_like(active), torch.arange(10).reshape(2, 5) == 0):
            single = substitute_active_risk(risk, mask, "shuffled_raw_pqk", torch.Generator())
            torch.testing.assert_close(single, risk)

    def test_monotone_preserves_rank_metrics(self):
        raw = torch.tensor([.1, .2, .3, .4, .7, .8], dtype=torch.float64)
        labels = torch.tensor([0, 1, 0, 1, 1, 0])
        cal = MonotoneCalibrator(1., -2.)
        transformed = cal(raw)
        a, b = binary_metrics(raw, labels), binary_metrics(transformed, labels)
        for key in ("auroc", "auprc", "spearman"):
            self.assertAlmostEqual(a[key], b[key])
        self.assertTrue(math.isfinite(b["brier"]) and math.isfinite(b["nll"]))


class QBridgeGradientTests(unittest.TestCase):
    def test_cross_reaches_adapter_and_new_head_only(self):
        learner = learner_fixture()
        x = torch.randn(3, 4)
        z = learner._network.extract_vector(x)
        score = learner._network.fc.qbridge_scores(z, learner.class_order, [70, 20])
        v = margin_violations(score, torch.tensor([50, 10, 50]), learner.class_order, [70, 20], 2.)
        cross_margin_loss(v, torch.full_like(v, .4), .5).backward()
        self.assertGreater(learner._network.convnet.weight.grad.norm(), 0)
        self.assertGreater(learner._network.fc.heads[1][0].weight.grad.norm(), 0)
        old_grad = learner._network.fc.heads[0][0].weight.grad
        self.assertTrue(old_grad is None or old_grad.abs().sum() == 0)
        self.assertTrue(all(p.grad is None for p in learner.old_ae.parameters()))
        self.assertTrue(all(p.grad is None for p in learner.quantum_kernel.parameters()))

    def test_guard_only_updates_new_head_and_refreshes_anchor(self):
        learner = learner_fixture()
        z = learner._network.extract_vector(torch.randn(2, 4))
        anchor = learner.old_ae(z)
        guard_loss(learner._network.fc, anchor, learner.class_order, [70, 20], [50, 10], 2.).backward()
        self.assertIsNone(learner._network.convnet.weight.grad)
        self.assertIsNone(learner.old_ae.weight.grad)
        self.assertGreater(learner._network.fc.heads[1][0].weight.grad.norm(), 0)
        before = learner.qbridge_runtime.guard_metrics(learner._network.fc, learner.old_ae, learner.class_order, [50, 10])
        with torch.no_grad():
            learner.old_ae.weight.mul_(-1)
        after = learner.qbridge_runtime.guard_metrics(learner._network.fc, learner.old_ae, learner.class_order, [50, 10])
        self.assertNotEqual(before, after)

    def test_alignment_gradient_ownership(self):
        learner = learner_fixture(lambda_cross=0, lambda_guard=0)
        learner._inc_loss = Mock(side_effect=AssertionError("legacy separation evaluated"))
        _, _, loss = learner._compute_rt_loss(torch.randn(3, 4), torch.tensor([2, 3, 2]), 0)
        loss.backward()
        self.assertGreater(learner.old_ae.weight.grad.norm(), 0)
        self.assertGreater(learner._network.convnet.weight.grad.norm(), 0)
        self.assertTrue(all(p.grad is None for p in learner.old_network_module_ptr.parameters()))
        learner._inc_loss.assert_not_called()

    def test_old_row_post_step_matches_zero_grad_decay_policy(self):
        head = SimpleContinualLinear(4, 2)
        head.update(2, freeze_old=False)
        reference = copy.deepcopy(head)
        opt = torch.optim.SGD(head.parameters(), lr=.1, weight_decay=.2, momentum=.9)
        refopt = torch.optim.SGD(reference.parameters(), lr=.1, weight_decay=.2, momentum=.9)
        z = torch.randn(3, 4, requires_grad=True)
        score = head.qbridge_scores(z, [0, 1, 2, 3], [0, 1])
        score.square().sum().backward()
        before = head.heads[0][0].weight.detach().clone()
        for ref, actual in zip(reference.parameters(), head.parameters()):
            ref.grad = None if actual.grad is None else actual.grad.detach().clone()
        opt.step(); refopt.step()
        torch.testing.assert_close(head.heads[0][0].weight, reference.heads[0][0].weight)
        self.assertFalse(torch.equal(before, head.heads[0][0].weight))


class QBridgeIntegrationTests(unittest.TestCase):
    def test_base_checkpoint_compatibility_is_selective(self):
        learner = learner_fixture()
        learner.args.update(_qbridge_restore_only=True, dataset="toy", seed=3,
                            init_cls=2, increment=2, model_name="adapter", convnet_type="toy")
        metadata = learner._checkpoint_run_metadata()
        metadata["use_quantum_kernel_inc"] = True
        checkpoint = {"cur_task": 0, "run_metadata": metadata, "class_order": learner.class_order,
                      "class_means": torch.zeros(2, 4), "class_covs": torch.eye(4).repeat(2, 1, 1),
                      "quantum_kernel_state_dict": {}}
        learner._validate_checkpoint(checkpoint)
        checkpoint["run_metadata"]["seed"] = 4
        with self.assertRaisesRegex(ValueError, "seed"):
            learner._validate_checkpoint(checkpoint)
        checkpoint["run_metadata"]["seed"] = 3
        checkpoint["class_order"] = list(reversed(learner.class_order))
        with self.assertRaisesRegex(ValueError, "class order"):
            learner._validate_checkpoint(checkpoint)

    def test_local_base_restore_does_not_request_timm_weights(self):
        from utils.inc_net import get_convnet
        class StubVit(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.adaptmlp = nn.Linear(4, 4)
                self.backbone = nn.Linear(4, 4)
        with patch("network.vision_transformer_adapter.VisionTransformer", StubVit), patch(
                "timm.create_model", side_effect=AssertionError("pretrained download requested")):
            model = get_convnet({"convnet_type": "pretrained_vit_b16_224_in21k_adapter",
                                 "model_name": "adapter", "ffn_num": 64, "_qbridge_restore_only": True})
        self.assertTrue(model.adaptmlp.weight.requires_grad)
        self.assertFalse(model.backbone.weight.requires_grad)

    def test_uploaded_local_pretrained_never_requests_timm_weights(self):
        from utils.inc_net import get_convnet
        from network import vision_transformer_adapter
        from types import SimpleNamespace
        tuning = SimpleNamespace(ffn_adapt=True, ffn_option="parallel",
            ffn_adapter_layernorm_option="none", ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1", ffn_num=4, d_model=768, vpt_on=False, vpt_num=0)
        source = vision_transformer_adapter.VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=lambda dim: nn.LayerNorm(dim, eps=1e-6),
            num_classes=0, global_pool=False, drop_path_rate=0.0, tuning_config=tuning)
        state = source.state_dict()
        for key in list(state):
            if key.endswith("q_proj.weight"):
                stem = key[:-len("q_proj.weight")]
                state[stem+"qkv.weight"] = torch.cat([state.pop(stem+n+"_proj.weight") for n in "qkv"])
            elif key.endswith("q_proj.bias"):
                stem = key[:-len("q_proj.bias")]
                state[stem+"qkv.bias"] = torch.cat([state.pop(stem+n+"_proj.bias") for n in "qkv"])
            elif ".fc" in key and ".adaptmlp." not in key:
                state[key.replace(".fc", ".mlp.fc")] = state.pop(key)
        state = {key: value for key, value in state.items() if "adaptmlp" not in key}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"vit.pt"
            torch.save(state, path)
            with patch("timm.create_model", side_effect=AssertionError("network requested")):
                model = get_convnet({"convnet_type": "pretrained_vit_b16_224_in21k_adapter",
                                     "model_name": "adapter", "ffn_num": 4,
                                     "pretrained_path": str(path)})
            self.assertEqual(model.out_dim, 768)
            self.assertTrue(any(p.requires_grad for n, p in model.named_parameters() if "adaptmlp" in n))
            self.assertTrue(all(not p.requires_grad for n, p in model.named_parameters() if "adaptmlp" not in n))

    def test_preflight_metadata_and_seed_templates(self):
        from utils.qbridge_experiment import preflight_base
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "base_3.pt"
            args = {"dataset": "cifar224", "seed": 3, "init_cls": 2, "increment": 2,
                    "model_name": "adapter", "convnet_type": "pretrained_vit_b16_224_in21k_adapter",
                    "use_quantum_kernel_base": True, "q_kernel_type": "pqk",
                    "qbridge": {"enabled": False, "base_checkpoint_path": str(Path(folder)/"base_{seed}.pt")}}
            metadata = {key: value for key, value in args.items() if key != "qbridge"}
            checkpoint = {"cur_task": 0, "known_classes": 2, "total_classes": 2,
                          "run_metadata": metadata, "class_order": [7, 3], "model_state_dict": {},
                          "quantum_kernel_state_dict": {}, "class_means": torch.zeros(2, 768),
                          "class_covs": torch.eye(768).repeat(2, 1, 1)}
            torch.save(checkpoint, path)
            preflight_base(args)
            self.assertEqual(args["qbridge"]["base_checkpoint_path"], str(path))
            self.assertEqual(len(args["_qbridge_base_hash"]), 64)
            checkpoint["cur_task"] = 1
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "task-0"):
                preflight_base(args)

    def test_real_incremental_loop_ssca_ca_and_reports_on_toy_data(self):
        from torch.utils.data import Dataset
        class ToyDataset(Dataset):
            def __init__(self, ids):
                self.labels = np.repeat(np.asarray(ids), 4)
                g = torch.Generator().manual_seed(10)
                self.images = torch.randn(len(self.labels), 4, generator=g)
            def __len__(self): return len(self.labels)
            def __getitem__(self, i): return i, self.images[i], int(self.labels[i])
        class Manager:
            _class_order = [70, 20, 50, 10]
            def get_task_size(self, task): return 2
            def get_dataset(self, ids, source, mode, ret_data=False):
                ds = ToyDataset(ids)
                return (ds.images, ds.labels, ds) if ret_data else ds
        with tempfile.TemporaryDirectory() as folder:
            learner = learner_fixture()
            learner._cur_task = 0
            learner._total_classes = 2
            learner._network.fc.heads = nn.ModuleList([learner._network.fc.heads[0]])
            learner._old_network = learner._network.copy().freeze()
            learner.old_network_module_ptr = learner._old_network
            learner.task_sizes = [2]
            learner._class_covs = torch.eye(4).repeat(2, 1, 1)
            learner.args.update(qbridge_experiment=True, _qbridge_base_hash="fixture",
                                _qbridge_report_dir=folder, seed=3, ae_code_dims=2,
                                ssca=True, ca=True, ca_epochs=1, inc_epochs=1, optimizer="sgd",
                                ae_init_lr=.001, ae_weight_decay=0, warmup_epoch=1,
                                stats_num_workers=0, eval_interval=0)
            learner.batch_size = 4
            learner.num_workers = 0
            learner.pin_memory = learner.persistent_workers = False
            learner._multiple_gpus = [torch.device("cpu")]
            learner.init_lr = .001
            learner.weight_decay = learner.min_lr = 0
            learner.logit_norm = None
            learner.topk = 2
            learner.qbridge_stage_records = []
            learner._qb_snapshot_id = learner._qb_means_id = "fixture:task0"
            with patch("models.RSIAT_adapter.AutoencoderSigmoid", side_effect=lambda **kw: nn.Linear(4, 4)):
                learner.incremental_train(Manager())
            self.assertEqual([r["stage"] for r in learner.qbridge_stage_records], ["pre_ca", "post_ca"])
            self.assertEqual(learner._class_covs.shape, (4, 4, 4))
            self.assertIsNone(learner.qbridge_runtime.cache)
            learner.after_task()
            self.assertEqual(learner._qb_snapshot_id, learner._qb_means_id)
            epochs = [json.loads(line) for line in (Path(folder)/"epochs.jsonl").read_text().splitlines()]
            self.assertEqual(epochs[0]["device"], "cpu")
            self.assertIn("cross_adapter_gradient_norm", epochs[0]["first_batch"])
            self.assertIn("guard", json.loads((Path(folder)/"stages.json").read_text())[1])

    def test_disabled_legacy_loss_and_gradients(self):
        for quantum in (False, True):
            learner = learner_fixture(enabled=False)
            learner.use_quantum_kernel_inc = quantum
            x = torch.randn(3, 4)
            y = torch.tensor([2, 3, 2])
            _, cls, inc = learner._compute_rt_loss(x, y, 0)
            z = learner._network.extract_vector(x)
            teacher = learner.old_network_module_ptr.extract_vector(x)
            projected = learner.old_ae(teacher)
            protos = learner.old_ae(torch.from_numpy(learner._class_means))
            similarity = learner.quantum_kernel(protos, projected) if quantum else F.normalize(protos, dim=1) @ F.normalize(projected, dim=1).T
            expected = learner.args["beta"]*F.mse_loss(z, projected)+learner.args["gamma"]*similarity.mean()
            torch.testing.assert_close(inc, expected)
            params = tuple(learner._network.parameters())+tuple(learner.old_ae.parameters())
            g1 = torch.autograd.grad(inc, params, retain_graph=True, allow_unused=True)
            g2 = torch.autograd.grad(expected, params, allow_unused=True)
            for a, b in zip(g1, g2):
                if a is None: self.assertIsNone(b)
                else: torch.testing.assert_close(a, b)

    def test_base_task_unchanged_with_flag(self):
        a, b = learner_fixture(False), learner_fixture(True)
        a._cur_task = b._cur_task = 0
        a._known_classes = b._known_classes = 0
        x, y = torch.randn(4, 4), torch.tensor([0, 1, 2, 3])
        for left, right in zip(a._compute_rt_loss(x, y, 3), b._compute_rt_loss(x, y, 3)):
            torch.testing.assert_close(left, right)

    def test_enabled_zero_weights_is_classification_alignment_without_separation(self):
        learner = learner_fixture(lambda_cross=0, lambda_guard=0)
        learner._inc_loss = Mock(side_effect=AssertionError("legacy separation called"))
        x, y = torch.randn(3, 4), torch.tensor([2, 3, 2])
        _, classification, regularizer = learner._compute_rt_loss(x, y, 0)
        z = learner._network.extract_vector(x)
        logits = learner._network.fc(z)["logits"]
        expected_cls = AngularPenaltySMLoss(loss_type="cosface", s=20, m=.1)(logits[:, 2:], y-2)
        expected_align = learner.args["beta"] * F.mse_loss(z, learner.old_ae(
            learner.old_network_module_ptr.extract_vector(x)))
        torch.testing.assert_close(classification, expected_cls)
        torch.testing.assert_close(regularizer, expected_align)

    def test_r1_replaces_classification_and_skips_selector(self):
        learner = learner_fixture(variant="R1", risk_mode="uniform", classification_mode="all_seen_cosface",
                                  risk_gain=0, lambda_cross=0, lambda_guard=0)
        learner.qbridge_runtime.risk = Mock(side_effect=AssertionError("selector called"))
        learner._inc_loss = Mock(side_effect=AssertionError("legacy loss called"))
        x, y = torch.randn(3, 4), torch.tensor([2, 3, 2])
        _, cls, reg = learner._compute_rt_loss(x, y, 0)
        z = learner._network.extract_vector(x)
        scores = learner._network.fc.qbridge_scores(z, learner.class_order, [70, 20])
        expected = AngularPenaltySMLoss(loss_type="cosface", s=20, m=.1)(scores, y)
        torch.testing.assert_close(cls, expected)
        cls.backward()
        self.assertGreater(learner._network.convnet.weight.grad.norm(), 0)
        self.assertEqual(learner._network.fc.heads[0][0].weight.grad.abs().sum(), 0)

    def test_cached_risk_and_gamma_across_tasks(self):
        learner = learner_fixture()
        runtime = learner.qbridge_runtime
        x = torch.randn(3, 4)
        gamma = learner.quantum_kernel.gamma.detach().clone()
        for task in (1, 5, 9):
            runtime.start_task(torch.from_numpy(learner._class_means), [70, 20], str(task), str(task))
            actual = runtime.risk(x)
            expected = learner.quantum_kernel(x, torch.from_numpy(learner._class_means))
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(gamma, learner.quantum_kernel.gamma)
            self.assertEqual(runtime.encode_count, 5)
        with self.assertRaises(ValueError):
            runtime.start_task(torch.from_numpy(learner._class_means), [70, 20], "1", "0")

    def test_risk_zero_gain_bypasses_circuit(self):
        for kwargs in ({"risk_gain": 0}, {"lambda_cross": 0}, {"risk_mode": "uniform"}):
            learner = learner_fixture(**kwargs)
            with patch.object(learner.quantum_kernel, "encode", side_effect=AssertionError("unused circuit")):
                learner._compute_rt_loss(torch.randn(3, 4), torch.tensor([2, 3, 2]), 0)

    def test_inference_has_no_auxiliary_call(self):
        learner = learner_fixture()
        learner.quantum_kernel.forward = Mock(side_effect=AssertionError("inference circuit"))
        learner.old_ae.forward = Mock(side_effect=AssertionError("inference projector"))
        self.assertEqual(learner._network(torch.randn(2, 4))["logits"].shape, (2, 4))

    def test_full_r6_fails_missing_artifact(self):
        with self.assertRaisesRegex(ValueError, "artifact"):
            learner_fixture(variant="R6", risk_mode="calibrated_pqk", calibration={"enabled": True},
                            calibrator_artifact_path="/missing/calibrator.pt")

    def test_disabled_config_ignores_qbridge_only_artifacts(self):
        self.assertIsNone(config_from_args({"qbridge": {"enabled": False, "risk_mode": "wrong"}}))
        with self.assertRaisesRegex(ValueError, "separation"):
            config_from_args({"use_quantum_kernel_inc": True, "qbridge": {"enabled": True}})


class QBridgeCalibrationTests(unittest.TestCase):
    def test_summary_rejects_mismatched_budgets(self):
        from scripts.summarize_qbridge import summarize
        from utils.qbridge import write_json
        with tempfile.TemporaryDirectory() as folder:
            for variant, epochs in (("R3", 1), ("R5", 2)):
                path = Path(folder) / variant
                write_json(path / "manifest.json", {
                    "schema": "qbridge-v1.1-run", "status": "completed", "last_completed_task": 1,
                    "base_checkpoint_sha256": "same", "config": {"dataset": "fixture", "seed": 3,
                    "tuned_epochs": epochs, "qbridge": {"variant": variant}}})
                write_json(path / "task_1.json", {"aia": 20., "forgetting": 1.,
                           "accuracy": {"top1": 20., "grouped": {"old": 20., "new": 20.}}})
            with self.assertRaisesRegex(ValueError, "budgets"):
                summarize(folder)

    def test_calibration_rejects_policy_and_fraction_mismatch(self):
        with self.assertRaisesRegex(ValueError, "calibration policy"):
            learner_fixture(variant="R6", risk_mode="calibrated_pqk",
                            calibration={"enabled": True, "fit_source": "test"})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cal.pt"
            torch.save({"kind": "calibrator", "source": "base_train_only", "status": "fitted",
                        "gate_passed": True, "base_checkpoint_hash": "base", "parameters": {"u": 1., "b": -1.},
                        "fit_id": "fixture", "fractions": [.5, .25, .25]}, path)
            with self.assertRaisesRegex(ValueError, "fractions"):
                learner_fixture(variant="R6", risk_mode="calibrated_pqk",
                                calibration={"enabled": True, "prototype_fit_fraction": .6},
                                calibrator_artifact_path=str(path))

    def test_fit_success_and_classical_selector_artifacts(self):
        from utils.qbridge_calibration import fit_classical_selector
        labels = torch.arange(40) % 2
        split = stratified_split(labels.numpy(), 5)
        scores = torch.zeros(40, 2)
        for part in ("F", "V"):
            for i, row in enumerate(split[part]):
                scores[row, labels[row]] = .8 if i % 2 else -.8
        cache = {"kind": "base_feature_cache", "source": "base_train_only", "base_checkpoint_hash": "b",
                 "features": torch.randn(40, 4), "labels": labels, "scores": scores,
                 "sample_ids": list(range(40))}
        pqk = metric()
        artifact = fit_calibrator(cache, pqk, "b", seed=5, steps=20)
        self.assertEqual(artifact["status"], "fitted")
        self.assertTrue(artifact["ranking_invariant"])
        self.assertFalse(set(artifact["split_ids"]["P"]) & set(artifact["split_ids"]["V"]))
        for kind in ("rbf_proj", "mlp_small"):
            config = dict(input_dim=4, num_qubits=2, num_layers=1, kernel_type=kind, init_seed=3)
            fitted = fit_classical_selector(cache, pqk, config, "b", {"alpha": .5, "rs_margin": .3},
                                           epochs=1, batch_size=8)
            self.assertEqual(fitted["status"], "fitted")
            if kind == "rbf_proj":
                for key, value in pqk.projector.state_dict().items():
                    torch.testing.assert_close(value, fitted["state_dict"]["projector."+key])

    def test_locked_config_generation(self):
        from scripts.generate_qbridge_configs import generate
        with tempfile.TemporaryDirectory() as folder:
            generate(folder)
            for i in range(11):
                cfg = json.loads((Path(folder)/("R%d.json" % i)).read_text())
                normalized = config_from_args(cfg)
                if i == 0:
                    self.assertIsNone(normalized)
                else:
                    self.assertEqual(normalized["variant"], "R%d" % i)
            r6 = json.loads((Path(folder)/"R6.json").read_text())["qbridge"]
            self.assertGreater(r6["lambda_guard"], 0)
            self.assertEqual(r6["calibration"]["split_seed"], "{seed}")

    def test_stratified_three_way_disjoint_split(self):
        labels = np.repeat(np.arange(4), 10)
        split = stratified_split(labels, 73)
        self.assertEqual(split, stratified_split(labels, 73))
        self.assertFalse(set(split["P"]) & set(split["F"]))
        self.assertFalse(set(split["F"]) & set(split["V"]))
        self.assertFalse(set(split["P"]) & set(split["V"]))
        self.assertEqual(sorted(sum(split.values(), [])), list(range(40)))
        with self.assertRaises(ValueError): stratified_split([0, 0, 1], 3)

    def test_calibration_gate_on_single_target_class(self):
        labels = torch.arange(20) % 2
        scores = F.one_hot(labels, 2).float()
        cache = {"kind": "base_feature_cache", "source": "base_train_only", "base_checkpoint_hash": "b",
                 "features": torch.randn(20, 4), "labels": labels, "scores": scores,
                 "sample_ids": list(range(20))}
        artifact = fit_calibrator(cache, metric(), "b", steps=2)
        self.assertFalse(artifact["gate_passed"])
        self.assertEqual(artifact["status"], "gate_failed")
        cache["source"] = "test"
        with self.assertRaises(ValueError): fit_calibrator(cache, metric(), "b", steps=2)

    def test_fitted_calibrator_is_frozen_and_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cal.pt"
            torch.save({"kind": "calibrator", "source": "base_train_only", "status": "fitted",
                        "gate_passed": True, "base_checkpoint_hash": "base", "parameters": {"u": 1., "b": -1.},
                        "fit_id": "fixture"}, path)
            learner = learner_fixture(variant="R6", risk_mode="calibrated_pqk", calibration={"enabled": True},
                                      calibrator_artifact_path=str(path))
            self.assertTrue(all(not p.requires_grad for p in learner.qbridge_runtime.calibrator.parameters()))
            cfg = learner.qbridge_config
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                BridgeRuntime(cfg, metric(), "other", 3, "cpu")


if __name__ == "__main__":
    unittest.main()
