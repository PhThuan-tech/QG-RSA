"""QKSR-v2 P2: projector refit from class statistics and bandwidth target (utils/quantum_kernel.py,
models/RSIAT_adapter.py::_qksr_v2_refit)."""
import importlib.util
import json
import logging
import math
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils.quantum_kernel import QuantumKernelModule

REPO = Path(__file__).resolve().parents[1]


class KernelRefitTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # 6 classes in 10-d, separated along the first two axes only; noise elsewhere is larger.
        self.dim, self.classes = 10, 6
        centers = torch.zeros(self.classes, self.dim)
        centers[:, 0] = torch.linspace(-3, 3, self.classes)
        centers[:, 1] = torch.tensor([1., -1., 1., -1., 1., -1.]) * 2
        self.centers = centers
        noise = torch.ones(self.dim)
        noise[2:] = 4.0
        self.noise = noise
        self.x = torch.cat([centers[c] + noise * torch.randn(200, self.dim) for c in range(self.classes)])
        self.y = torch.arange(self.classes).repeat_interleave(200)

    def stats(self):
        means = torch.stack([self.x[self.y == c].mean(0) for c in range(self.classes)])
        within = sum(torch.cov(self.x[self.y == c].T) for c in range(self.classes)) / self.classes
        return means, within

    def test_lda_finds_class_directions_and_scales_angles(self):
        module = QuantumKernelModule(input_dim=self.dim, num_qubits=2, num_layers=1, init_seed=1)
        keys = set(module.state_dict())
        means, within = self.stats()
        info = module.refit_projector_lda(means, within, angle_std=0.8)
        self.assertTrue(module.projector.skip_norm)
        self.assertEqual(set(module.state_dict()), keys)  # checkpoint layout unchanged
        w = module.projector.linear.weight.detach()
        # rows live (almost) in the span of the two informative axes
        share = (w[:, :2].square().sum() / w.square().sum()).item()
        self.assertGreater(share, 0.95)
        with torch.no_grad():
            pre = module.projector.linear(self.x)
        for k in range(2):
            self.assertAlmostEqual(pre[:, k].std().item(), 0.8 / math.pi, delta=0.03)
        self.assertGreater(info["fisher_share"], 0.9)

    def test_gamma_target_sets_the_median_kernel(self):
        module = QuantumKernelModule(input_dim=self.dim, num_qubits=3, num_layers=2, init_seed=1)
        a, b = self.x[:50], self.x[600:700]
        gamma = module.set_gamma_target(a, b, 0.2)
        with torch.no_grad():
            k = module(a, b)
        self.assertAlmostEqual(float(k.median()), 0.2, delta=0.02)
        self.assertAlmostEqual(float(module.gamma), gamma, places=5)
        with self.assertRaises(ValueError):
            module.set_gamma_target(a, b, 1.5)


class LearnerRefitTest(unittest.TestCase):
    """The real incremental loop (task 1) on the toy fixtures, as tests/test_qksr_probe.py."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("_sd_fixtures", Path(__file__).with_name("test_stats_diagnostics.py"))
        cls.fx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fx)

    def run_mode(self, folder, **options):
        learner = self.fx.legacy_learner(folder)
        torch.manual_seed(7)
        kernel = QuantumKernelModule(input_dim=4, num_qubits=3, num_layers=2, init_seed=3)
        kernel.calibrate_gamma(torch.randn(16, 4))
        kernel.set_inc_mode("frozen")
        learner.quantum_kernel = kernel
        learner.use_quantum_kernel_inc = True
        learner.qksr_probe = None
        learner.args.update(q_inc_pair="current", inc_loss_mode="mean", q_inc_train_mode="frozen",
                            q_calib_samples=8, **options)
        torch.manual_seed(21)
        np.random.seed(22)
        random.seed(23)
        with self.assertLogs(level=logging.INFO) as logs:
            self.fx.run_task(learner)
        return learner, logs.output

    def test_default_off_is_plain_qksr(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain, _ = self.run_mode(tmp)
            off, logs = self.run_mode(tmp, q_proj_refit="none", q_gamma_refit="none")
        for a, b in zip(plain._network.state_dict().values(), off._network.state_dict().values()):
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(any("QKSR refit" in line for line in logs))

    def test_refit_runs_and_changes_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain, _ = self.run_mode(tmp)
            lda, logs = self.run_mode(tmp, q_proj_refit="lda", q_gamma_refit="target", q_gamma_target=0.2)
            gamma, glogs = self.run_mode(tmp, q_gamma_refit="target", q_gamma_target=0.2)
        line = [l for l in logs if "QKSR refit task=1" in l][0]
        self.assertIn('"lda"', line)
        self.assertIn('"gamma"', line)
        self.assertTrue(lda.quantum_kernel.projector.skip_norm)
        self.assertFalse(gamma.quantum_kernel.projector.skip_norm)
        self.assertTrue(any("QKSR refit task=1" in l and '"lda"' not in l for l in glogs))
        self.assertFalse(torch.equal(plain._network.convnet.weight, lda._network.convnet.weight))
        for name, learner in (("lda", lda), ("gamma", gamma)):
            self.assertTrue(all(torch.isfinite(p).all() for p in learner._network.parameters()), name)

    def test_invalid_options_fail_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.run_mode(tmp, q_proj_refit="pca")


class OverlayTest(unittest.TestCase):
    def test_p2_overlays_differ_from_p1_only_in_the_refit(self):
        allowed = {"prefix", "q_kernel_type", "q_proj_refit", "q_gamma_refit", "q_gamma_target", "q_angle_std",
                   "q_lda_ridge"}
        for k in range(3):
            ref = json.loads((REPO / "exps" / "qksr2" / "p1_qksr_cur_rep{}.json".format(k)).read_text())
            for fam, kind, proj in (("p2_lda", "pqk", "lda"), ("p2_lda_rbf", "rbf_proj", "lda"), ("p2_gamma", "pqk", "none")):
                got = json.loads((REPO / "exps" / "qksr2" / "{}_rep{}.json".format(fam, k)).read_text())
                self.assertLessEqual({key for key in set(ref) | set(got) if ref.get(key) != got.get(key)}, allowed)
                self.assertEqual((got["q_kernel_type"], got["q_proj_refit"], got["q_gamma_refit"]), (kind, proj, "target"))
                self.assertEqual(got["rng_replicate"], k)


if __name__ == "__main__":
    unittest.main()
