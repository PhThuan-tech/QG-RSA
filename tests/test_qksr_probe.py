"""QKSR P0 probe (utils/qksr_probe.py): read-only, and its statistics are correct."""
import importlib.util
import logging
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from utils import qksr_probe as qp
from utils.quantum_kernel import QuantumKernelModule


class HelperTest(unittest.TestCase):
    def test_settings(self):
        self.assertIsNone(qp.probe_settings({}))
        self.assertEqual(qp.probe_settings({"qksr_probe": True})["qksr_probe_every"], 10)
        with self.assertRaises(ValueError):
            qp.probe_settings({"qksr_probe": True, "qksr_probe_every": 0})
        with self.assertRaises(ValueError):
            qp.probe_settings({"qksr_probe": True, "resume": True})

    def test_auc(self):
        self.assertAlmostEqual(qp._auc(torch.tensor([0., 1.]), torch.tensor([2., 3.])), 1.0)
        self.assertAlmostEqual(qp._auc(torch.tensor([2., 3.]), torch.tensor([0., 1.])), 0.0)
        self.assertAlmostEqual(qp._auc(torch.tensor([1., 1.]), torch.tensor([1., 1.])), 0.5)
        self.assertAlmostEqual(qp._auc(torch.tensor([0., 2.]), torch.tensor([1., 3.])), 0.75)

    def test_cka(self):
        x = torch.randn(30, 3, dtype=torch.float64)
        k = x @ x.T
        self.assertAlmostEqual(qp._cka(k, 2 * k), 1.0, places=10)
        self.assertLess(qp._cka(k, torch.eye(30, dtype=torch.float64)), 0.9)


class LearnerIntegrationTest(unittest.TestCase):
    """The real incremental loop (task 1) on the toy fixtures, with and without the probe."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("_sd_fixtures",
                                                      Path(__file__).with_name("test_stats_diagnostics.py"))
        cls.fx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fx)

    def run_mode(self, folder, qksr, probe, kernel_type="pqk"):
        learner = self.fx.legacy_learner(folder)
        if qksr:
            torch.manual_seed(7)
            kernel = QuantumKernelModule(input_dim=4, num_qubits=3, num_layers=2, init_seed=3,
                                         kernel_type=kernel_type)
            kernel.calibrate_gamma(torch.randn(16, 4))
            kernel.set_inc_mode("frozen")
            learner.quantum_kernel = kernel
            learner.use_quantum_kernel_inc = True
            learner.args.update(q_inc_pair="current", inc_loss_mode="mean", q_inc_train_mode="frozen",
                                q_calib_samples=8)
        learner.qksr_probe = None
        if probe:
            cfg = qp.probe_settings({"qksr_probe": True, "qksr_probe_every": 1, "qksr_probe_per_class": 4,
                                     "qksr_probe_topk": 1})
            learner.qksr_probe = qp.QKSRProbe(cfg, learner._device)
        torch.manual_seed(21)
        np.random.seed(22)
        random.seed(23)
        with self.assertLogs(level=logging.INFO) as logs:
            self.fx.run_task(learner)
        return learner, logs.output

    def check_identical(self, qksr):
        with tempfile.TemporaryDirectory() as tmp:
            plain, _ = self.run_mode(tmp, qksr, probe=False)
            probed, logs = self.run_mode(tmp, qksr, probe=True)
        for (name, a), b in zip(plain._network.state_dict().items(), probed._network.state_dict().values()):
            self.assertTrue(torch.equal(a, b), name)
        for a, b in zip(plain.old_ae.state_dict().values(), probed.old_ae.state_dict().values()):
            self.assertTrue(torch.equal(a, b))
        np.testing.assert_array_equal(plain._class_means, probed._class_means)
        return logs

    def test_rsiat_unchanged_and_logged(self):
        logs = self.check_identical(qksr=False)
        self.assertTrue(any("QKSRPROBE grad task=1 epoch=1" in line and "cos_c_kd" in line for line in logs))
        end = [line for line in logs if "QKSRPROBE end task=1" in line]
        self.assertTrue(end and "twin_euclid_z" in end[0] and "pqk" not in end[0])

    def test_qksr_unchanged_and_measured(self):
        logs = self.check_identical(qksr=True)
        grad = [line for line in logs if "QKSRPROBE grad task=1 epoch=1" in line]
        self.assertTrue(grad and "cos_c_sep" in grad[0] and "pcgrad_keep_sep" in grad[0])
        end = [line for line in logs if "QKSRPROBE end task=1" in line][0]
        for key in ("w1", "angles", "pqk", "cka_pqk_vs", "loss_pairs", "twin_euclid_theta"):
            self.assertIn('"%s"' % key, end)
        # The summary script reads these lines back and applies every reading rule.
        spec = importlib.util.spec_from_file_location(
            "_summ", Path(__file__).resolve().parents[1] / "scripts" / "summarize_qksr_probe.py")
        summ = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(summ)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p0_qksr_cur_1993_toy.log"
            path.write_text("\n".join(line.split(":", 2)[-1] for line in logs)
                            + "\nCNN top1 curve: [90.0, 80.0]\n", encoding="utf-8")
            s = summ.summarise("p0_qksr_cur", path)
        self.assertIsNotNone(s["grad"]["cos_c_sep"])
        self.assertIsNotNone(s["end"]["auc_pqk"])
        text = summ.render([s])
        for rule in ("W6:", "W7: AUC", "W1: AUC", "W1: tán xạ", "W2:", "W7: CKA", "W4:"):
            self.assertIn(rule, text)

    def test_p1_twins_train_with_a_live_separation_gradient(self):
        """P1 swaps only q_kernel_type: RBF on the same angles and the PQK without CNOTs both train."""
        finals = {}
        for kernel_type in ("pqk", "rbf_proj", "pqk_no_cnot"):
            with tempfile.TemporaryDirectory() as tmp:
                learner, logs = self.run_mode(tmp, True, True, kernel_type=kernel_type)
            grad = [line for line in logs if "QKSRPROBE grad task=1 epoch=1" in line][0]
            self.assertIn('"norm_sep"', grad)
            self.assertNotIn('"norm_sep": {"median": 0.0', grad)
            finals[kernel_type] = learner._network.convnet.weight.detach().clone()
        self.assertFalse(torch.equal(finals["pqk"], finals["rbf_proj"]))
        self.assertFalse(torch.equal(finals["pqk"], finals["pqk_no_cnot"]))
        overlays = Path(__file__).resolve().parents[1] / "exps" / "qksr2"
        import json
        for k in range(3):
            ref = json.loads((Path(__file__).resolve().parents[1] / "exps" / "qksr" /
                              "qksr_cur_rep{}.json".format(k)).read_text())
            for fam, kind in (("p1_qksr_cur", "pqk"), ("p1_qksr_rbf", "rbf_proj"), ("p1_qksr_nocnot", "pqk_no_cnot")):
                got = json.loads((overlays / "{}_rep{}.json".format(fam, k)).read_text())
                diff = {key for key in set(ref) | set(got) if ref.get(key) != got.get(key)}
                self.assertEqual(got["q_kernel_type"], kind)
                self.assertLessEqual(diff, {"prefix", "offline", "q_kernel_type"})


if __name__ == "__main__":
    unittest.main()
