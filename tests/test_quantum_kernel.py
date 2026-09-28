import unittest

import numpy as np
import torch
from torch import nn

from data.data_manager import DataManager
from models.RSIAT_adapter import Learner, RS_Loss
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
