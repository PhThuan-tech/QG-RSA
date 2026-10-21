import json
import logging
import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils import ca_heads as ch
from utils.quantum_kernel import LocalRDMExtractor, QuantumFeatureEncoder


def small_args(**extra):
    args = {"ca_heads": list(ch.HEADS), "ca_head_dim": 64, "ca_head_samples": 60, "ca_head_qubits": 3,
            "ca_head_layers": 1, "ca_head_gate_samples": 40, "ca_head_seed": 7}
    args.update(extra)
    return args


class SettingsTest(unittest.TestCase):
    def test_absent_means_disabled(self):
        self.assertIsNone(ch.head_settings({}))
        self.assertIsNone(ch.head_settings({"ca_heads": []}))

    def test_linear_first_and_validation(self):
        cfg = ch.head_settings({"ca_heads": ["pqk_ridge", "qda"]})
        self.assertEqual(cfg["heads"], ["linear", "qda", "pqk_ridge"])
        with self.assertRaises(ValueError):
            ch.head_settings({"ca_heads": ["svm"]})
        with self.assertRaises(ValueError):
            ch.head_settings({"ca_heads": "qda"})
        with self.assertRaises(ValueError):
            ch.head_settings({"ca_heads": ["qda"], "ca_head_qubits": 20})

    def test_mean_scale_matches_ca(self):
        task_sizes, cur = [20] * 4, 3
        expected = [0.9 + (c // 20 + 1) / (cur + 1) * 0.1 for c in range(80)]
        np.testing.assert_allclose(ch.ca_mean_scale(task_sizes, cur), expected)


class PQKFeatureTest(unittest.TestCase):
    def test_bloch_coordinates_of_product_state(self):
        q = 3
        enc = QuantumFeatureEncoder(q, 1, torch.float64, entangle=False, reupload=False, frozen_random=True)
        theta = torch.tensor([[[0.3, -1.1, 2.0]]], dtype=torch.float64)
        feats = ch.pqk_features(theta, torch.zeros(1, 1, q, dtype=torch.float64), enc, LocalRDMExtractor(q, 1), 1)
        z, x = feats.reshape(q, 2).T
        torch.testing.assert_close(z, torch.cos(theta[0, 0]))
        torch.testing.assert_close(x, torch.sin(theta[0, 0]))

    def test_entangled_features_are_valid_rdm_entries(self):
        fmap = ch.FeatureMap("pqk", 16, 200, 3, "cpu", qubits=4, layers=2, reupload=True, order=2)
        x = torch.randn(50, 16)
        phi = fmap.fit_scale(x, torch.Generator().manual_seed(0))(x)
        self.assertEqual(phi.shape, (50, 200))
        self.assertTrue(bool((phi.abs() <= 1 + 1e-5).all()))
        # Two-qubit RDM diagonal (upper-triangle positions 0, 4, 7, 9) has unit trace.
        two = phi[:, 8:8 + 10]
        torch.testing.assert_close(two[:, [0, 4, 7, 9]].sum(1), torch.ones(50), atol=1e-5, rtol=0)

    def test_maps_are_deterministic_and_pqk_variants_share_projection(self):
        x = torch.randn(30, 16)
        a = ch.FeatureMap("rff", 16, 32, 5, "cpu").fit_scale(x, torch.Generator().manual_seed(1))(x)
        b = ch.FeatureMap("rff", 16, 32, 5, "cpu").fit_scale(x, torch.Generator().manual_seed(1))(x)
        torch.testing.assert_close(a, b)
        p = ch.FeatureMap("pqk", 16, 64, 9, "cpu", qubits=3, layers=1)
        n = ch.FeatureMap("pqk_noent", 16, 64, 9, "cpu", qubits=3, layers=1)
        torch.testing.assert_close(p.weight, n.weight)
        torch.testing.assert_close(p.angles, n.angles)


class HeadTest(unittest.TestCase):
    def test_ridge_on_rff_solves_xor(self):
        gen = torch.Generator().manual_seed(0)
        signs = torch.randint(0, 2, (800, 2), generator=gen) * 2 - 1
        x = torch.cat([signs * 2.0 + 0.3 * torch.randn(800, 2, generator=gen),
                       0.1 * torch.randn(800, 6, generator=gen)], 1)
        y = (signs[:, 0] * signs[:, 1] > 0).long()
        fmap = ch.FeatureMap("rff", 8, 256, 2, "cpu").fit_scale(x, gen)
        hold = ch.holdout_mask(1, 800, 0.2, "cpu")
        weight, info = ch.fit_ridge(fmap, x, y, 2, hold)
        accuracy = float(((fmap(x) @ weight).argmax(1) == y).float().mean())
        self.assertGreater(accuracy, 0.9)
        self.assertIn(info["ridge_factor"], ch.RIDGE_GRID)

    def test_qda_separates_classes_by_covariance(self):
        gen = torch.Generator().manual_seed(0)
        d = 6
        means = torch.zeros(2, d, dtype=torch.float64)
        covs = torch.stack([torch.eye(d), 9 * torch.eye(d)]).double()
        x, y = ch.sample_pseudo(means, covs, 300, gen, "cpu")
        accuracy = float((ch.qda_predict(means, covs, x) == y).float().mean())
        self.assertGreater(accuracy, 0.8)

    def test_gates_are_finite(self):
        gen = torch.Generator().manual_seed(0)
        x, y = torch.randn(60, 16, generator=gen), torch.arange(3).repeat_interleave(20)
        maps = {k: ch.FeatureMap(k, 16, 48, 1, "cpu", qubits=3, layers=1).fit_scale(x, gen)
                for k in ("rff", "pqk")}
        gates = ch.kernel_gates(maps, x, y, 3)
        self.assertTrue(all(-1 <= v <= 1 for v in gates["alignment"].values()))
        self.assertTrue(math.isfinite(gates["geometric_difference"]["rff"]))


class FakeLearner:
    def __init__(self, tmp):
        gen = np.random.default_rng(0)
        self.args = small_args(_run_log_prefix=str(Path(tmp) / "run"))
        self._device = torch.device("cpu")
        self._network = torch.nn.Linear(2, 2)
        self.task_sizes = [2, 2]
        self._known_classes, self._total_classes = 2, 4
        self._class_means = gen.normal(size=(4, 12)) * 3
        self._class_covs = torch.stack([torch.eye(12) * (1 + c) for c in range(4)]).float()
        self._test = (np.concatenate([gen.normal(self._class_means[c], 1 + c, size=(30, 12)) for c in range(4)]
                                     ).astype(np.float32), np.repeat(np.arange(4), 30))
        self.test_loader = None

    def _extract_vectors(self, loader):
        return self._test


class DriverTest(unittest.TestCase):
    def test_evaluate_heads_preserves_rng_and_logs_curves(self):
        with tempfile.TemporaryDirectory() as tmp:
            learner = FakeLearner(tmp)
            linear = np.repeat(np.arange(4), 30)
            torch.manual_seed(123); np.random.seed(5); random.seed(9)
            before = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
            with self.assertLogs(level=logging.INFO) as logs:
                result = ch.evaluate_heads(learner, 1, linear, str(Path(tmp) / "stats"))
            self.assertTrue(torch.equal(before[0], torch.get_rng_state()))
            self.assertTrue(np.array_equal(before[1], np.random.get_state()[1]))
            self.assertEqual(before[2], random.getstate())
            self.assertEqual(set(result["heads"]), set(ch.HEADS))
            self.assertEqual(result["heads"]["linear"]["top1"], 100.0)
            self.assertTrue(any("CAHEAD curve head=pqk_ridge" in line for line in logs.output))
            self.assertTrue((Path(tmp) / "stats" / "cahead_task_1.json").is_file())
            with np.load(Path(tmp) / "stats" / "cahead_predictions_task_1.npz") as saved:
                self.assertEqual(set(saved.files), set(ch.HEADS) | {"target"})
            again = ch.evaluate_heads(learner, 1, linear, str(Path(tmp) / "stats2"))
            self.assertEqual(result["heads"], again["heads"])  # deterministic given the seed


class SummaryTest(unittest.TestCase):
    def write_log(self, root, prefix, linear, pqk, rff):
        folder = Path(root) / "adapter" / "imagenetr" / "0" / "20"
        folder.mkdir(parents=True, exist_ok=True)
        lines = []
        for head, curve in (("linear", linear), ("pqk_ridge", pqk), ("rff_ridge", rff)):
            lines.append("x => CAHEAD tasks head={}: {}".format(head, json.dumps([curve[-1], curve[-1]])))
            lines.append("x => CAHEAD curve head={}: {}".format(head, json.dumps(curve)))
        lines.append("x => CNN top1 curve: {}".format(json.dumps(linear)))
        (folder / "{}_1993_pretrained_vit_b16_224_in21k_adapter.log".format(prefix)).write_text("\n".join(lines))

    def test_rules(self):
        from scripts.summarize_ca_heads import collect, render
        with tempfile.TemporaryDirectory() as tmp:
            for k, prefix in enumerate(("qrfca", "qrfca_rep1", "qrfca_rep2")):
                self.write_log(tmp, prefix, [90.0, 80.0 + k * 0.1], [91.0, 81.0 + k * 0.1 + 0.05 * k],
                               [90.0, 80.0 + k * 0.1])
            runs = collect([tmp])
            self.assertEqual(len(runs), 3)
            text, summary = render(runs, {})
            primary = summary["imagenetr|qrfca|1993"]["primary"]
            self.assertTrue(primary["pqk_ridge"]["verdict"].startswith("ĐẠT"))
            self.assertIn("head linear = CNN curve", text)
            quantum = summary["imagenetr|qrfca|1993"]["quantum"]
            self.assertIn("chưa chứng minh", quantum["verdict"])  # qda / rp / pqk_noent missing


if __name__ == "__main__":
    unittest.main()
