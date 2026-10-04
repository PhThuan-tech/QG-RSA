import unittest
import ast
import json
import io
from contextlib import redirect_stdout
import random
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from scripts.generate_qksr_ablation_configs import generate

from data.data_manager import DataManager
from models.RSIAT_adapter import Learner, RS_Loss
from models.base import BaseLearner
from utils.quantum_kernel import QuantumKernelModule
from utils.qksr_statistics import holm_adjust, paired_comparison


DTYPE = torch.float64


def _reference_rdm(state, num_qubits, selected):
    """Slow, independent partial trace used only for q <= 4 tests."""
    batch_size = state.shape[0]
    selected = tuple(selected)
    remaining = tuple(q for q in range(num_qubits) if q not in selected)
    subsystem_dim = 1 << len(selected)
    rest_dim = 1 << len(remaining)
    rho = torch.zeros(
        batch_size, subsystem_dim, subsystem_dim, dtype=state.dtype
    )

    def global_index(rest_value, selected_value):
        bits = [0] * num_qubits
        for offset, qubit in enumerate(remaining):
            shift = len(remaining) - 1 - offset
            bits[qubit] = (rest_value >> shift) & 1
        for offset, qubit in enumerate(selected):
            shift = len(selected) - 1 - offset
            bits[qubit] = (selected_value >> shift) & 1
        index = 0
        for bit in bits:
            index = (index << 1) | bit
        return index

    for rest_value in range(rest_dim):
        for row in range(subsystem_dim):
            row_index = global_index(rest_value, row)
            for column in range(subsystem_dim):
                column_index = global_index(rest_value, column)
                rho[:, row, column] += (
                    state[:, row_index] * state[:, column_index].conj()
                )
    return rho


class QuantumCircuitTests(unittest.TestCase):
    def make_module(self, **kwargs):
        defaults = dict(
            input_dim=6,
            num_qubits=3,
            num_layers=2,
            kernel_type="pqk",
            kernel_order=2,
            gamma_mode="bounded_learned",
            dtype=DTYPE,
            init_seed=17,
        )
        defaults.update(kwargs)
        return QuantumKernelModule(**defaults)

    def test_cnot_truth_table_msb_control(self):
        module = self.make_module(num_qubits=2, num_layers=1, kernel_order=1)
        encoder = module.encoder
        state = torch.eye(4, dtype=DTYPE)
        result = encoder._apply_cnot(state, control=0)
        expected_destinations = [0, 1, 3, 2]
        for source, destination in enumerate(expected_destinations):
            self.assertEqual(int(result[source].argmax()), destination)

    def test_statevector_norm_after_every_layer(self):
        module = self.make_module()
        features = torch.randn(5, 6, dtype=DTYPE)
        angles = module.projector(features)
        _, history = module.encoder(angles, return_intermediate=True)
        for state in history:
            torch.testing.assert_close(
                state.square().sum(dim=1), torch.ones(5, dtype=DTYPE),
                rtol=1e-10, atol=1e-10,
            )

    def test_rdm_matches_full_reference(self):
        for num_qubits in (3, 4):
            module = self.make_module(
                input_dim=7, num_qubits=num_qubits, kernel_order=2
            )
            state = module.encoder(module.projector(torch.randn(2, 7, dtype=DTYPE)))
            rdms = module.extractor(state)
            for qubit in range(num_qubits):
                reference_one = _reference_rdm(state, num_qubits, (qubit,))
                torch.testing.assert_close(
                    rdms["one"][:, qubit], reference_one, rtol=1e-10, atol=1e-10
                )
                pair = (qubit, (qubit + 1) % num_qubits)
                reference_two = _reference_rdm(state, num_qubits, pair)
                torch.testing.assert_close(
                    rdms["two"][:, qubit], reference_two, rtol=1e-10, atol=1e-10
                )

    def test_rdm_physical_properties_and_shapes(self):
        module = self.make_module(kernel_order=2)
        representation = module.encode(torch.randn(4, 6, dtype=DTYPE))
        self.assertEqual(representation["one"].shape, (4, 3, 2, 2))
        self.assertEqual(representation["two"].shape, (4, 3, 4, 4))
        for key in ("one", "two"):
            rho = representation[key]
            torch.testing.assert_close(rho, rho.transpose(-1, -2).conj())
            torch.testing.assert_close(
                rho.diagonal(dim1=-2, dim2=-1).sum(-1),
                torch.ones(rho.shape[:-2], dtype=DTYPE),
                rtol=1e-10,
                atol=1e-10,
            )
            self.assertGreaterEqual(
                float(torch.linalg.eigvalsh(rho).min().detach()), -1e-8
            )


class QuantumKernelTests(unittest.TestCase):
    def make_module(self, **kwargs):
        defaults = dict(
            input_dim=5,
            num_qubits=3,
            num_layers=2,
            kernel_type="pqk",
            kernel_order=1,
            gamma_mode="bounded_learned",
            dtype=DTYPE,
            init_seed=23,
        )
        defaults.update(kwargs)
        return QuantumKernelModule(**defaults)

    def test_kernel_identity_symmetry_range_and_psd(self):
        for kernel_type in sorted(QuantumKernelModule.KERNEL_TYPES):
            module = self.make_module(kernel_type=kernel_type)
            x = torch.randn(5, 5, dtype=DTYPE)
            y = torch.randn(3, 5, dtype=DTYPE)
            gram = module(x)
            cross = module(x, y)
            reverse = module(y, x)
            torch.testing.assert_close(torch.diag(gram), torch.ones(5, dtype=DTYPE))
            torch.testing.assert_close(cross, reverse.T, rtol=1e-10, atol=1e-10)
            self.assertTrue(bool(((gram > 0) & (gram <= 1)).all()))
            self.assertGreaterEqual(
                float(torch.linalg.eigvalsh(gram).min().detach()), -1e-8
            )

    def test_gradcheck_and_nonzero_parameter_gradients(self):
        module = self.make_module(num_qubits=2, input_dim=4, num_layers=2)
        x = torch.randn(3, 4, dtype=DTYPE, requires_grad=True)
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda value: module(value), (x,), eps=1e-6, atol=1e-4, rtol=1e-3
            )
        )
        module.zero_grad(set_to_none=True)
        loss = module(x)[0, 1] + module(x)[1, 2]
        loss.backward()
        self.assertGreater(float(module.projector.linear.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.encoder.theta.grad.abs().sum()), 0.0)
        self.assertGreater(float(module.gamma_param.grad.abs().sum()), 0.0)

    def test_frozen_metric_passes_gradient_to_inputs(self):
        module = self.make_module()
        module.set_inc_mode("frozen")
        old_ae = nn.Linear(5, 5, dtype=DTYPE)
        inputs_a = old_ae(torch.randn(4, 5, dtype=DTYPE))
        inputs_b = old_ae(torch.randn(3, 5, dtype=DTYPE))
        module(inputs_a, inputs_b).mean().backward()
        self.assertGreater(float(old_ae.weight.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

    def test_param_groups_cover_all_variants_without_buffers(self):
        for kernel_type in sorted(QuantumKernelModule.KERNEL_TYPES):
            for gamma_mode in sorted(QuantumKernelModule.GAMMA_MODES):
                module = self.make_module(
                    kernel_type=kernel_type, gamma_mode=gamma_mode
                )
                groups = module.param_groups(0.01, 1e-4, None)
                grouped = [parameter for group in groups for parameter in group["params"]]
                expected = [parameter for parameter in module.parameters() if parameter.requires_grad]
                self.assertEqual({id(p) for p in grouped}, {id(p) for p in expected})
                self.assertTrue(all(parameter.requires_grad for parameter in grouped))
                module.set_inc_mode("frozen")
                self.assertEqual(module.param_groups(0.01, 1e-4, 0.1), [])

    def test_gamma_self_cross_idempotence_and_fallback(self):
        module = self.make_module(gamma_mode="bounded_learned")
        x = torch.randn(8, 5, dtype=DTYPE)
        first = module.calibrate_gamma(x)
        second = module.calibrate_gamma(torch.randn(8, 5, dtype=DTYPE))
        self.assertEqual(first, second)
        self.assertTrue(bool(module.gamma_initialized.item()))
        self.assertIn("gamma0", module.state_dict())

        cross_module = self.make_module(gamma_mode="median_fixed")
        gamma = cross_module.calibrate_gamma(x[:3], x[3:], exclude_diagonal=False)
        self.assertGreater(gamma, 0.0)

        fallback = self.make_module(kernel_type="rbf_proj")
        zeros = torch.zeros(4, 5, dtype=DTYPE)
        self.assertEqual(fallback.calibrate_gamma(zeros), 1.0)

    def test_module_initialization_preserves_rng_and_is_repeatable(self):
        torch.manual_seed(101)
        state_before = torch.get_rng_state().clone()
        first = self.make_module(init_seed=999)
        state_after = torch.get_rng_state().clone()
        self.assertTrue(torch.equal(state_before, state_after))
        second = self.make_module(init_seed=999)
        for left, right in zip(first.state_dict().values(), second.state_dict().values()):
            torch.testing.assert_close(left, right)

    def test_random_frozen_theta_is_buffer(self):
        module = self.make_module(kernel_type="pqk_random_frozen")
        parameter_names = dict(module.named_parameters())
        buffer_names = dict(module.named_buffers())
        self.assertNotIn("encoder.theta", parameter_names)
        self.assertIn("encoder.theta", buffer_names)

    def test_state_dict_roundtrip_preserves_calibration_and_metric(self):
        source = self.make_module(kernel_type="pqk_random_frozen")
        features = torch.randn(6, 5, dtype=DTYPE)
        source.calibrate_gamma(features)
        expected = source(features)

        restored = self.make_module(kernel_type="pqk_random_frozen", init_seed=999)
        restored.load_state_dict(source.state_dict())
        actual = restored(features)
        torch.testing.assert_close(actual, expected)
        self.assertTrue(bool(restored.gamma_initialized.item()))
        self.assertEqual(restored.parameter_counts()["projector"], 18)

    def test_rs_loss_baseline_parity_and_quantum_gradient(self):
        features = torch.randn(6, 5, dtype=DTYPE, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss_module = RS_Loss(lamda=0.7, margin=0.5)
        actual = loss_module(features, labels)

        normalized = torch.nn.functional.normalize(features, p=2, dim=1)
        similarity = normalized @ normalized.T
        mask = labels[:, None].eq(labels[None, :]).to(DTYPE)
        positive = mask - torch.eye(len(labels), dtype=DTYPE)
        negative = 1.0 - mask
        expected = (
            (torch.relu(1.0 - similarity) * positive).sum() / (positive.sum() + 1e-6)
            + 0.7 * (torch.relu(similarity - 0.5) * negative).sum()
            / (negative.sum() + 1e-6)
        )
        torch.testing.assert_close(actual, expected)

        metric = self.make_module(input_dim=5)
        quantum_loss = loss_module(
            features,
            labels,
            quantum_kernel_module=metric,
            margin_override=0.3,
            margin_mode="quantile",
            margin_quantile=0.9,
        )
        quantum_loss.backward()
        self.assertGreater(float(metric.projector.linear.weight.grad.abs().sum()), 0.0)

    def test_incremental_integration_keeps_frozen_metric_and_updates_autoencoder(self):
        learner = Learner.__new__(Learner)
        learner.old_ae = nn.Linear(5, 5)
        learner._class_means = torch.randn(4, 5).numpy()
        learner._device = torch.device("cpu")
        learner.use_quantum_kernel_inc = True
        learner.quantum_kernel = self.make_module(input_dim=5, dtype=torch.float32)
        learner.quantum_kernel.set_inc_mode("frozen")
        learner.args = {
            "beta": 1.5,
            "gamma": 0.75,
            "q_inc_pair": "old_proj",
            "inc_loss_mode": "mean",
        }
        current = torch.randn(3, 5, requires_grad=True)
        previous = torch.randn(3, 5)
        Learner._inc_loss(learner, current, previous).backward()
        self.assertGreater(float(learner.old_ae.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(current.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in learner.quantum_kernel.parameters())
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_module_initialization_preserves_cuda_rng(self):
        before = torch.cuda.get_rng_state().clone()
        self.make_module(init_seed=123)
        after = torch.cuda.get_rng_state().clone()
        self.assertTrue(torch.equal(before, after))


class QKSRImprovementTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1993)

    def make_learner(self, **overrides):
        learner = Learner.__new__(Learner)
        learner.old_ae = nn.Linear(5, 5)
        learner._class_means = torch.randn(4, 5).numpy()
        learner._device = torch.device("cpu")
        learner.use_quantum_kernel_inc = True
        learner.quantum_kernel = QuantumKernelModule(
            input_dim=5, num_qubits=3, num_layers=1, dtype="float32"
        )
        learner.quantum_kernel.set_inc_mode("frozen")
        learner.args = {
            "beta": 0.0, "gamma": 0.75,
            "q_inc_pair": "old_proj", "inc_loss_mode": "mean",
            **overrides,
        }
        return learner

    def test_legacy_incremental_loss_is_unchanged(self):
        learner = self.make_learner(beta=1.5)
        current, previous = torch.randn(3, 5), torch.randn(3, 5)
        projected = learner.old_ae(previous)
        prototypes = learner.old_ae(torch.from_numpy(learner._class_means))
        expected = 1.5 * nn.functional.mse_loss(current, projected)
        expected += 0.75 * learner.quantum_kernel(prototypes, projected).mean()
        torch.testing.assert_close(learner._inc_loss(current, previous), expected)

    def test_current_pair_has_direct_adapter_gradient_old_proj_does_not(self):
        for pair_mode in ("old_proj", "current"):
            with self.subTest(pair_mode=pair_mode):
                learner = self.make_learner(q_inc_pair=pair_mode)
                current = torch.randn(3, 5, requires_grad=True)
                learner._inc_loss(current, torch.randn(3, 5)).backward()
                magnitude = float(current.grad.abs().sum())
                if pair_mode == "current":
                    self.assertGreater(magnitude, 0.0)
                else:
                    self.assertEqual(magnitude, 0.0)
                self.assertTrue(all(p.grad is None for p in learner.quantum_kernel.parameters()))

    def test_margin_stops_repulsion_and_weight_only_scales_quantum_term(self):
        learner = self.make_learner(q_inc_pair="current", inc_loss_mode="margin")
        current, previous = torch.randn(3, 5, requires_grad=True), torch.randn(3, 5)
        learner.args["rs_margin_inc"] = 1.0
        loss = learner._inc_loss(current, previous)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertEqual(float(current.grad.abs().sum()), 0.0)
        learner.args["rs_margin_inc"] = 0.0
        full = learner._inc_loss(current, previous)
        learner.args["q_inc_weight"] = 0.25
        torch.testing.assert_close(learner._inc_loss(current, previous), full * 0.25)
        learner.args.update(beta=1.5, q_inc_weight=0.0)
        expected_alignment = 1.5 * nn.functional.mse_loss(current, learner.old_ae(previous))
        torch.testing.assert_close(learner._inc_loss(current, previous), expected_alignment)
        # Controls must retain original RSIAT even if Q-only knobs are present.
        learner.use_quantum_kernel_inc = False
        baseline = learner._inc_loss(current, previous)
        learner.args["q_inc_weight"] = 999
        torch.testing.assert_close(learner._inc_loss(current, previous), baseline)

    def test_incremental_warmup_schedule(self):
        learner = self.make_learner(q_inc_weight=0.25, q_inc_warmup_epochs=3)
        self.assertAlmostEqual(learner._quantum_inc_weight(0), 0.25 / 3)
        self.assertAlmostEqual(learner._quantum_inc_weight(1), 0.25 * 2 / 3)
        self.assertEqual(learner._quantum_inc_weight(2), 0.25)
        self.assertEqual(learner._quantum_inc_weight(20), 0.25)

    def test_inc_only_calibration_uses_selected_feature_branch(self):
        for pair in ("current", "old_proj"):
            learner = self.make_learner(q_inc_pair=pair)
            learner._cur_task, learner._known_classes = 1, 4
            learner.use_quantum_kernel_base = False
            learner._network = nn.Linear(5, 5)
            inputs = torch.randn(4, 5)
            learner._network_module_ptr = Mock()
            learner._network_module_ptr.extract_vector.side_effect = lambda x: x * 2
            learner.old_network_module_ptr = Mock()
            learner.old_network_module_ptr.extract_vector.side_effect = lambda x: x * 3
            learner.q_calibration_loader = [(None, inputs, None)]
            # Keep metric real, inspect inputs actually used to calibrate it.
            calibrate = Mock(wraps=learner.quantum_kernel.calibrate_gamma)
            learner.quantum_kernel.calibrate_gamma = calibrate
            learner._calibrate_quantum_kernel()
            expected = inputs * 2 if pair == "current" else learner.old_ae(inputs * 3)
            torch.testing.assert_close(calibrate.call_args.args[1], expected)
            self.assertTrue(learner._network.training)
            self.assertTrue(learner.old_ae.training)
            learner._calibrate_quantum_kernel()
            self.assertEqual(calibrate.call_count, 1)

    def test_epoch_diagnostics_pool_pairs_ignore_self_diagonal_and_singleton(self):
        module = QuantumKernelModule(input_dim=5, num_qubits=3, num_layers=1)
        module.reset_epoch_diagnostics()
        features = torch.randn(3, 5)
        before = torch.get_rng_state().clone()
        first = module(features)
        second = module(features[:2], features[:2])  # Cross diagonal IS relevant.
        module(features[:1])                       # No informative self pairs.
        expected = torch.cat((first[~torch.eye(3, dtype=torch.bool)], second.flatten())).detach()
        stats = module.epoch_kernel_stats()
        self.assertEqual(stats["pairs"], 10)
        self.assertEqual(stats["empty_batches"], 1)
        self.assertEqual(stats["batches"], 3)
        self.assertAlmostEqual(stats["mean"], float(expected.double().mean()), places=7)
        self.assertAlmostEqual(stats["std"], float(expected.double().std(unbiased=False)), places=7)
        self.assertEqual(sum(stats["histogram_10_bins_0_1"]), 10)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        module.reset_epoch_diagnostics()
        module(features[:1])
        self.assertEqual(module.epoch_kernel_stats()["pairs"], 0)
        self.assertNotIn("mean", module.epoch_kernel_stats())

    def test_epoch_gradient_summary_not_last_batch_and_does_not_change_gradients(self):
        module = QuantumKernelModule(input_dim=5, num_qubits=3, num_layers=1)
        module.reset_epoch_diagnostics()
        expected = {}
        for scale in (1.0, 0.0):
            module.zero_grad()
            (module(torch.randn(3, 5)).mean() * scale).backward()
            snapshots = {name: p.grad.clone() for name, p in module.named_parameters() if p.grad is not None}
            for name, gradient in snapshots.items():
                expected[name] = expected.get(name, 0.0) + float(gradient.abs().mean()) / 2
            module.record_gradient_health()
            for name, p in module.named_parameters():
                if name in snapshots:
                    torch.testing.assert_close(p.grad, snapshots[name])
        actual = module.epoch_gradient_health()
        self.assertGreater(actual["projector.linear.weight"], 0.0)
        self.assertEqual(module.gradient_health()["projector.linear.weight"], 0.0)
        for name in expected:
            self.assertAlmostEqual(actual[name], expected[name], places=7)

    def test_ssca_loader_matches_images_views_subset_and_preserves_rng(self):
        manager = DataManager.__new__(DataManager)
        manager._train_data = np.arange(20 * 4, dtype=np.uint8).reshape(20, 2, 2)
        manager._train_targets = np.array([0] * 10 + [1] * 10)
        manager._train_trsf = [transforms.RandomHorizontalFlip()]
        manager._test_trsf = [transforms.ToTensor()]
        manager._common_trsf = []
        manager.use_path = False
        train, _ = manager.get_dataset_with_validation([0, 1], 0.2, 99)
        learner = Learner.__new__(Learner)
        learner.args = {"ssca_feature_mode": "paired_eval", "seed": 1993}
        learner._cur_task, learner.batch_size, learner.pin_memory = 1, 5, False
        before = torch.get_rng_state().clone()
        loader = learner._build_ssca_loader(manager, train)
        first, second = list(loader), list(loader)
        for a, b in zip(first, second):
            for left, right in zip(a, b):
                torch.testing.assert_close(left, right)
        self.assertEqual(sum(len(batch[0]) for batch in first), len(train))
        self.assertTrue(np.array_equal(loader.dataset.images, train.images))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        learner.args["ssca_feature_mode"] = "legacy"
        learner.train_loader = object()
        self.assertIs(learner._build_ssca_loader(manager, train), learner.train_loader)

    def test_actual_optimizer_loop_base_and_incremental_on_cpu(self):
        class ToyHead(nn.Linear):
            def forward(self, features):
                return {"logits": super().forward(features)}

        class ToyNetwork(nn.Module):
            def __init__(self, classes):
                super().__init__()
                self.adapter = nn.Linear(5, 5)
                self.fc = ToyHead(5, classes)

            def extract_vector(self, inputs):
                return self.adapter(inputs)

        for task in (0, 1):
            with self.subTest(task=task):
                learner = self.make_learner(
                    beta=1.5, q_inc_pair="current", inc_loss_mode="margin",
                    rs_margin_inc=0.3, q_inc_weight=0.25, q_inc_warmup_epochs=3,
                    scale=30.0, margin=0.3, lambda_rs=1.0,
                )
                learner._cur_task, learner._known_classes = task, 2 * task
                learner.use_quantum_kernel_base = True
                learner._network = ToyNetwork(2 + learner._known_classes)
                learner._network_module_ptr = learner._network
                learner.old_network_module_ptr = ToyNetwork(2).requires_grad_(False).eval()
                learner.rs_loss_func = RS_Loss()
                learner.tuned_epochs = 2
                learner._compute_accuracy = lambda *_: 0.0
                labels = torch.tensor([0, 0, 1, 1, 0, 1, 0]) + learner._known_classes
                loader = DataLoader(TensorDataset(torch.arange(7), torch.randn(7, 5), labels), batch_size=3)
                learner.quantum_kernel.set_inc_mode("trainable" if task == 0 else "frozen")
                parameters = list(learner._network.parameters()) + list(learner.old_ae.parameters())
                parameters += [p for p in learner.quantum_kernel.parameters() if p.requires_grad]
                optimizer = torch.optim.SGD(parameters, lr=0.001)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
                before = learner._network.adapter.weight.detach().clone()
                learner._init_train(loader, loader, optimizer, scheduler, warmup_epoch=1)
                self.assertFalse(torch.equal(before, learner._network.adapter.weight))
                self.assertTrue(all(torch.isfinite(p).all() for p in parameters))
                if task == 1:
                    self.assertTrue(all(p.grad is None for p in learner.quantum_kernel.parameters()))
                else:
                    self.assertGreater(learner.quantum_kernel.epoch_gradient_health()["projector.linear.weight"], 0.0)

    def test_generator_enables_quantum_from_original_imagenet_config(self):
        with tempfile.TemporaryDirectory(prefix=".qksr_test_", dir=Path(__file__).resolve().parents[1]) as directory:
            output = Path(directory)
            with redirect_stdout(io.StringIO()):
                generate(Path(__file__).resolve().parents[1] / "exps" / "adapter_imageneta.json", output)
            baseline = json.loads((output / "A_rsiat.json").read_text(encoding="utf-8"))
            legacy = json.loads((output / "B_qksr.json").read_text(encoding="utf-8"))
            improved = json.loads((output / "F5_current_margin_warmup.json").read_text(encoding="utf-8"))
            retention = json.loads((output / "G6_cov_shrinkage.json").read_text(encoding="utf-8"))
            rbf = json.loads((output / "G7_retention_rbf.json").read_text(encoding="utf-8"))
            shared = json.loads((output / "G8_rsiat_shared_retention.json").read_text(encoding="utf-8"))
            self.assertFalse(baseline["use_quantum_kernel_base"])
            self.assertTrue(legacy["use_quantum_kernel_inc"])
            self.assertEqual(legacy["q_inc_pair"], "old_proj")
            self.assertEqual(improved["q_inc_weight"], 0.25)
            self.assertEqual(improved["q_inc_pair"], "current")
            self.assertEqual(retention["statistics_transport"], "guarded_ridge")
            self.assertEqual(retention["ae_type"], "signed_residual")
            self.assertTrue(retention["q_detach_prototypes"])
            self.assertFalse(retention["keep_last_checkpoint"])
            self.assertEqual(rbf["q_kernel_type"], "rbf_proj")
            self.assertFalse(shared["use_quantum_kernel_inc"])
            for key in ("ae_type", "ae_reset_each_task", "statistics_transport", "ssca_feature_mode",
                        "relation_distill_weight", "stats_cov_shrinkage"):
                self.assertEqual(retention[key], rbf[key])
                self.assertEqual(retention[key], shared[key])
            steps = [json.loads(next(output.glob("G{}_*.json".format(index))).read_text(encoding="utf-8"))
                     for index in range(7)]
            for previous, current in zip(steps, steps[1:]):
                changed = {key for key in previous.keys() | current.keys()
                           if key != "prefix" and previous.get(key) != current.get(key)}
                self.assertIn(changed, ({"ae_type"}, {"ae_reset_each_task"}, {"q_detach_prototypes"},
                                      {"relation_distill_weight", "relation_temperature"},
                                      {"statistics_transport"}, {"stats_cov_shrinkage"}))
            for key in ("seed", "init_cls", "increment", "batch_size", "gamma", "beta", "init_lr"):
                self.assertEqual(baseline[key], improved[key])

    def test_notebook_cells_compile_and_profiles_create_distinct_configs(self):
        project = Path(__file__).resolve().parents[1]
        notebook_path = project / "QKSR_Kaggle_ImageNetR.ipynb"
        if not notebook_path.is_file():
            self.skipTest("Notebook not included in this source archive")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
        for name, source in cells.items():
            # %pip is an IPython magic, not Python syntax. Validate the
            # remaining Python without adding IPython as a test dependency.
            lines = []
            for line in source.splitlines():
                if line.lstrip().startswith("%"):
                    self.assertTrue(line.startswith("%pip install "))
                    lines.append("pass")
                else:
                    lines.append(line)
            compile("\n".join(lines), name, "exec")
        with tempfile.TemporaryDirectory(prefix=".qksr_test_", dir=Path(__file__).resolve().parents[1]) as directory:
            for dataset in ("imageneta", "imagenetr"):
                for profile, use_q in (("legacy", True), ("current_margin", True), ("legacy", False),
                                       ("retention", True), ("retention", False)):
                    for ssca in ("legacy", "paired_eval"):
                        for run_profile in ("smoke", "full"):
                            namespace = {
                                "PROJECT_DIR": project, "WORK_ROOT": Path(directory),
                                "OUTPUT_ROOT": Path(directory), "DATASET_NAME": dataset,
                                "CONFIG_FILENAME": "adapter_{}.json".format(dataset),
                                "SEED": 1993, "USE_QKSR": use_q, "QKSR_PROFILE": profile,
                                "SSCA_FEATURE_MODE": ssca, "RUN_PROFILE": run_profile,
                                "SMOKE_TASKS": 2,
                            }
                            with redirect_stdout(io.StringIO()):
                                exec(cells["tao-cau-hinh"], namespace)
                            config = namespace["config"]
                            self.assertEqual(config["dataset"], dataset)
                            self.assertEqual(config["use_quantum_kernel_inc"], use_q)
                            expected_ssca = "paired_eval" if profile == "retention" else ssca
                            self.assertEqual(config["ssca_feature_mode"], expected_ssca)
                            self.assertEqual(config["init_cls"], 20)
                            self.assertEqual(config["increment"], 20)
                            if profile == "retention":
                                self.assertEqual(config["ae_type"], "signed_residual")
                                self.assertEqual(config["statistics_transport"], "guarded_ridge")
                                self.assertIn("retention", config["prefix"])
                                self.assertFalse(config["keep_last_checkpoint"])
                                self.assertEqual(config.get("q_detach_prototypes", False), use_q)
                            elif use_q and profile == "current_margin":
                                self.assertEqual(config["q_inc_weight"], 0.25)
                                self.assertIn("current_margin", config["prefix"])
                            elif use_q:
                                self.assertEqual(config["q_inc_pair"], "old_proj")
                            if expected_ssca == "paired_eval":
                                self.assertIn("paired_eval", config["prefix"])
                            if run_profile == "smoke":
                                self.assertEqual(config["max_tasks_per_run"], 2)
                                self.assertFalse(config["resume"])
                            else:
                                self.assertNotIn("max_tasks_per_run", config)
                                self.assertTrue(config["resume"])

    def test_dedicated_ablation_notebook_has_independent_config_cells(self):
        project = Path(__file__).resolve().parents[1]
        notebook_path = project / "QKSR_Kaggle_Ablations.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = {
            cell["id"]: "".join(cell["source"])
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        }
        for name, source in code_cells.items():
            lines = ["pass" if line.lstrip().startswith("%") else line
                     for line in source.splitlines()]
            compile("\n".join(lines), name, "exec")

        expected = [
            "A_rsiat", "B_qksr", "G0_paired_current_margin",
            "G1_signed_projector", "G2_reset_projector",
            "G3_detached_prototypes", "G4_relation_retention",
            "G5_mean_cov_transport", "G6_cov_shrinkage",
            "G7_retention_rbf", "G8_rsiat_shared_retention",
        ]
        run_cells = {key: value for key, value in code_cells.items() if key.startswith("run-")}
        self.assertEqual(len(run_cells), len(expected))
        for experiment in expected:
            cell_id = "run-" + experiment.lower().replace("_", "-")
            self.assertEqual(run_cells[cell_id].strip(), "run_experiment({!r})".format(experiment))

        settings = code_cells["ablation-settings"]
        self.assertIn("'imageneta'", settings)
        self.assertIn("'imagenetr'", settings)
        self.assertIn("'cifar224'", settings)
        self.assertIn("RUN_MODE = 'smoke'", settings)
        runner = code_cells["ablation-runner"]
        self.assertIn("'max_tasks_per_run': SMOKE_TASKS", runner)
        self.assertIn("'ca_epochs': 1", runner)
        self.assertIn("'resume': RUN_MODE == 'full' and FULL_RESUME", runner)
        self.assertIn("f'{DATASET_NAME}_{experiment_name}_{mode_suffix}", runner)

        # The outer f-string injects only the local model path. Braces for the
        # inner runtime error must survive until the patched adapter executes.
        dependency_source = code_cells["ablation-dependencies"].replace(
            "%pip install -q --upgrade-strategy only-if-needed -r requirements-colab.txt", "pass"
        )
        tree = ast.parse(dependency_source)
        assignments = [
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in {"online_line", "local_block"}
                    for target in node.targets)
        ]
        namespace = {"pretrained_path": Path("/kaggle/input/model/model.safetensors")}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), "local-model-patch", "exec"), namespace)
        self.assertIn("{local_load}", namespace["local_block"])
        compile("def patched_factory():\n" + namespace["local_block"], "patched-adapter-block", "exec")

        with tempfile.TemporaryDirectory(prefix=".qksr_test_ablation_", dir=project) as directory:
            experiment_names = expected
            specs = {
                "imageneta": ("adapter_imageneta.json", 20),
                "imagenetr": ("adapter_imagenetr.json", 20),
                "cifar224": ("adapter_cifar224.json", 10),
            }
            for dataset, (filename, increment) in specs.items():
                for mode in ("smoke", "full"):
                    namespace = {
                        "EXPERIMENTS": experiment_names,
                        "PROJECT_DIR": project,
                        "CONFIG_FILENAME": filename,
                        "CONFIG_ROOT": Path(directory),
                        "OUTPUT_ROOT": Path(directory),
                        "DATASET_NAME": dataset,
                        "RUN_MODE": mode,
                        "VAL_RATIO": 0.0,
                        "SEEDS": [1993, 2024],
                        "FULL_RESUME": True,
                        "SMOKE_TASKS": 2,
                    }
                    with redirect_stdout(io.StringIO()):
                        exec(runner, namespace)
                    configs = {
                        name: namespace["build_experiment_config"](name)
                        for name in experiment_names
                    }
                    self.assertTrue(all(config["increment"] == increment for config in configs.values()))
                    self.assertTrue(all(config["seed"] == [1993, 2024] for config in configs.values()))
                    self.assertEqual(configs["G6_cov_shrinkage"]["q_kernel_type"], "pqk")
                    self.assertEqual(configs["G7_retention_rbf"]["q_kernel_type"], "rbf_proj")
                    self.assertFalse(configs["G8_rsiat_shared_retention"]["use_quantum_kernel_inc"])
                    if mode == "smoke":
                        self.assertTrue(all(config["max_tasks_per_run"] == 2 for config in configs.values()))
                        self.assertTrue(all(config["ca_epochs"] == 1 for config in configs.values()))
                        self.assertTrue(all(not config["resume"] for config in configs.values()))
                    else:
                        self.assertTrue(all("max_tasks_per_run" not in config for config in configs.values()))
                        self.assertTrue(all(config["resume"] for config in configs.values()))


class TinyCheckpointNetwork(nn.Module):
    """Offline stand-in for a dynamic classifier; no pretrained download."""
    def __init__(self):
        super().__init__()
        self.fc = None

    def update_fc(self, size):
        if self.fc is None:
            self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(5, size))

    def freeze(self):
        self.requires_grad_(False)
        return self.eval()


class TinyCheckpointLearner(BaseLearner):
    def _after_load_checkpoint(self, checkpoint):
        # Simulate the random draws during autoencoder/metric reconstruction.
        random.random()
        np.random.random()
        torch.rand(3)


class CheckpointReproducibilityTests(unittest.TestCase):
    def make_learner(self, **overrides):
        learner = TinyCheckpointLearner({
            "device": [torch.device("cpu")], "dataset": "toy", "seed": 1993,
            "init_cls": 2, "increment": 2, **overrides,
        })
        learner._network = TinyCheckpointNetwork()
        learner._network.update_fc(2)
        learner._cur_task, learner._known_classes, learner._total_classes = 0, 2, 2
        learner.task_sizes, learner.class_order = [2], [0, 1, 2, 3]
        return learner

    def test_checkpoint_restores_rng_after_model_reconstruction(self):
        source = self.make_learner()
        with tempfile.TemporaryDirectory(prefix=".qksr_test_", dir=Path(__file__).resolve().parents[1]) as directory:
            path = str(Path(directory) / "task_0.pkl")
            source.save_checkpoint(path)
            expected = (random.random(), np.random.random(), torch.rand(5))
            restored = self.make_learner()
            restored.load_checkpoint(path)
            actual = (random.random(), np.random.random(), torch.rand(5))
            self.assertEqual(expected[:2], actual[:2])
            torch.testing.assert_close(actual[2], expected[2])
            for left, right in zip(source._network.parameters(), restored._network.parameters()):
                torch.testing.assert_close(left, right)

    def test_legacy_checkpoint_cannot_silently_change_new_protocol(self):
        legacy = self.make_learner()._checkpoint_run_metadata()
        for key in BaseLearner._CHECKPOINT_PROTOCOL_DEFAULTS:
            legacy.pop(key)
        self.make_learner()._validate_checkpoint({"run_metadata": legacy})
        for override in ({"q_inc_weight": 0.25}, {"q_inc_warmup_epochs": 3}, {"ssca_feature_mode": "paired_eval"}):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    self.make_learner(**override)._validate_checkpoint({"run_metadata": legacy})


class StatisticalProtocolTests(unittest.TestCase):
    def test_paired_bootstrap_and_holm(self):
        summary = paired_comparison(
            [90.8, 91.1, 90.9, 91.0, 91.2],
            [90.0, 90.2, 90.1, 90.0, 90.3],
            bootstrap_resamples=2000,
            seed=7,
            minimum_effect=0.3,
        )
        self.assertTrue(summary["supports_improvement"])
        self.assertGreater(summary["ci"][0], 0.0)
        adjusted = holm_adjust({"E1": 0.01, "E2": 0.04, "E3": 0.03})
        self.assertAlmostEqual(adjusted["E1"], 0.03)
        self.assertAlmostEqual(adjusted["E3"], 0.06)
        self.assertAlmostEqual(adjusted["E2"], 0.06)


class ValidationSplitTests(unittest.TestCase):
    def test_class_statistics_do_not_leak_validation_images(self):
        manager = DataManager.__new__(DataManager)
        manager._train_data = np.arange(20 * 4, dtype=np.uint8).reshape(20, 2, 2)
        manager._train_targets = np.array([0] * 10 + [1] * 10)
        manager._train_trsf, manager._test_trsf, manager._common_trsf = [], [], []
        manager.use_path = False
        learner = BaseLearner({"device": [torch.device("cpu")], "val_ratio": 0.2})
        train, validation = manager.get_dataset_with_validation([0, 1], 0.2, 99)
        learner.train_dataset = train
        for class_idx in (0, 1):
            dataset = learner._class_statistics_dataset(manager, class_idx)
            selected = dataset.dataset.images[dataset.indices]
            self.assertEqual(len(dataset), 8)
            self.assertTrue(np.array_equal(selected, train.images[train.labels == class_idx]))
            held_out = {image.tobytes() for image in validation.images}
            self.assertTrue(all(image.tobytes() not in held_out for image in selected))
        learner.args["val_ratio"] = 0.0
        self.assertEqual(len(learner._class_statistics_dataset(manager, 0)), 10)

    def test_validation_split_is_stratified_and_repeatable(self):
        manager = DataManager.__new__(DataManager)
        manager._train_data = np.arange(20 * 4, dtype=np.uint8).reshape(20, 2, 2)
        manager._train_targets = np.array([0] * 10 + [1] * 10)
        manager._train_trsf = []
        manager._test_trsf = []
        manager._common_trsf = []
        manager.use_path = False

        train_a, val_a = manager.get_dataset_with_validation([0, 1], 0.2, 99)
        train_b, val_b = manager.get_dataset_with_validation([0, 1], 0.2, 99)
        self.assertEqual(len(train_a), 16)
        self.assertEqual(len(val_a), 4)
        self.assertEqual(np.bincount(val_a.labels).tolist(), [2, 2])
        self.assertTrue(np.array_equal(train_a.images, train_b.images))
        self.assertTrue(np.array_equal(val_a.images, val_b.images))
        eval_view = manager.get_eval_view(train_a)
        self.assertTrue(np.array_equal(eval_view.images, train_a.images))


if __name__ == "__main__":
    unittest.main()
