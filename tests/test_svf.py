import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from utils import qsd

ROOT = Path(__file__).resolve().parents[1]
# Loaded by path: on Kaggle a site-packages "tests" package shadows this folder, so
# "from tests.test_qsd import ..." fails there.
_spec = importlib.util.spec_from_file_location("_qsd_fixtures", Path(__file__).with_name("test_qsd.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
FakeLearner = _fixtures.FakeLearner
SVF_HEADS = ["linear", "ridge_stats", "snap_ridge"]


def run_sequence(tmp, tasks=2):
    torch.manual_seed(0)  # the fake heads are initialised from the global RNG
    learner, results = FakeLearner(), []
    learner.args.update(qsd_heads=SVF_HEADS, qsd_fuse=True, qsd_save_scores=True)
    for task in range(tasks):
        learner.start_task(task)
        linear = np.asarray(learner.test_y[learner.test_y < learner._total_classes])
        before = (torch.get_rng_state(), np.random.get_state()[1].copy())
        results.append(qsd.evaluate_qsd(learner, task, linear, str(Path(tmp) / "stats")))
        assert torch.equal(before[0], torch.get_rng_state())
        assert np.array_equal(before[1], np.random.get_state()[1])
    return learner, results


class SettingsTest(unittest.TestCase):
    def test_heads_and_overlay(self):
        cfg = qsd.qsd_settings({"qsd_heads": SVF_HEADS, "qsd_fuse": True})
        for w in qsd.FUSE_WEIGHTS:
            self.assertIn(qsd.fuse_name("snap_ridge", w), cfg["heads"])
            self.assertIn(qsd.fuse3_name(w), cfg["heads"])
        # Without ridge_stats there is no three-way fusion; without qsd_fuse no fusion at all.
        self.assertEqual(qsd.qsd_settings({"qsd_heads": ["snap_ridge"], "qsd_fuse": True})["fuse3"], [])
        self.assertEqual(qsd.qsd_settings({"qsd_heads": SVF_HEADS})["heads"], SVF_HEADS)
        # Old overlays keep exactly their heads (SVF adds nothing to A2 / SCF runs).
        scf = json.loads((ROOT / "exps/scf/scf.json").read_text(encoding="utf-8"))
        self.assertFalse(any("snap" in h for h in qsd.qsd_settings(scf)["heads"]))
        svf = json.loads((ROOT / "exps/svf/svf.json").read_text(encoding="utf-8"))
        changed = {k for k in set(svf) | set(scf) if svf.get(k) != scf.get(k)}
        self.assertEqual({k for k in changed if not k.startswith("qsd")}, {"prefix"})  # evaluation only


class DriverTest(unittest.TestCase):
    def test_snapshot_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs(level="INFO"):
                learner, results = run_sequence(tmp)
            state = learner._qsd_state
            # The snapshot is the network after task 0, frozen: it equals the old network at task 1
            # while the plastic network has moved.
            self.assertTrue(torch.equal(state["snapshot"].weight, learner._old_network.weight))
            self.assertFalse(torch.equal(state["snapshot"].weight, learner._network.weight))
            self.assertFalse(any(p.requires_grad for p in state["snapshot"].parameters()))
            # At task 0 the snapshot view is the plastic view and the fake stored statistics are exact,
            # so the snapshot ridge reproduces ridge_stats.
            np.testing.assert_array_equal(results[0]["predictions"]["snap_ridge"],
                                          results[0]["predictions"]["ridge_stats"])
            # Statistics are appended once per task and never rewritten.
            self.assertEqual([m.shape[0] for m in state["snap_means"]], [2, 2])
            feats = learner.train_x[learner.train_y < 2] @ state["snapshot"].weight.detach()
            np.testing.assert_allclose(state["snap_means"][0].numpy(),
                                       np.stack([feats[learner.train_y[learner.train_y < 2] == c].mean(0).numpy()
                                                 for c in range(2)]), atol=1e-5)
            self.assertIn("snap_ridge", results[1]["diagnostics"]["fuse_scale"])
            for w in qsd.FUSE_WEIGHTS:
                self.assertIn(qsd.fuse3_name(w), results[1]["heads"])
            with np.load(Path(tmp) / "stats" / "qsd_scores_task_1.npz") as saved:
                self.assertEqual(saved["snap_ridge"].shape, (80, 4))
            # Deterministic given the seed.
            _, again = run_sequence(tmp)
            self.assertEqual(results[1]["heads"], again[1]["heads"])

    def test_vanishing_weight_recovers_the_other_heads(self):
        with mock.patch.object(qsd, "FUSE_WEIGHTS", (0.25, 1e-9)):
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertLogs(level="INFO"):
                    _, results = run_sequence(tmp)
        p = results[1]["predictions"]
        np.testing.assert_array_equal(p[qsd.fuse3_name(1e-9)], p["fuse_ridge_stats_b0p25"])
        np.testing.assert_array_equal(p[qsd.fuse_name("snap_ridge", 1e-9)], p[qsd.fuse_name("ridge_stats", 1e-9)])


class SummaryTest(unittest.TestCase):
    def write_run(self, root, dataset, seed, gain, tasks=3):
        folder = root / "adapter" / dataset / "0" / "20"
        folder.mkdir(parents=True, exist_ok=True)
        base_rows = [[90.0 - 5 * (t - j) for j in range(t + 1)] for t in range(tasks)]
        heads = {"linear": 0.0, "fuse_ridge_stats_b0p25": 0.4, "snap_ridge": -2.0}
        for w in qsd.FUSE_WEIGHTS:  # weight 1 is best on every run
            heads[qsd.fuse_name("snap_ridge", w)] = gain - abs(np.log2(w))
            heads[qsd.fuse3_name(w)] = gain + 0.3 - abs(np.log2(w))
        lines = []
        for t in range(tasks):
            for head, g in heads.items():
                lines.append("x => QSD tasks head={}: {}".format(head, json.dumps([a + g for a in base_rows[t]])))
        for head, g in heads.items():
            lines.append("x => QSD mech head={}: {}".format(head, json.dumps({"task_prediction": 90.0,
                                                                             "pair_ratio": 1.2})))
            lines.append("x => QSD curve head={}: {}".format(head, json.dumps([sum(r) / len(r) + g
                                                                               for r in base_rows])))
        lines.append("x => CNN top1 curve: {}".format(json.dumps([sum(r) / len(r) for r in base_rows])))
        name = "svf_{}_pretrained_vit_b16_224_in21k_adapter.log".format(seed)
        (folder / name).write_text("\n".join(lines), encoding="utf-8")

    def test_weight_frozen_on_tuning_run_and_rule(self):
        from scripts.summarize_svf import collect, render
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root, "imagenetr", 1993, 5.0)  # tuning run: excluded from the rule
            for i, seed in enumerate((2015, 2026)):
                self.write_run(root, "imagenetr", seed, 1.2 + 0.1 * i)
            text, decision = render(collect([tmp]))
            self.assertIsNone(decision["holds"])  # ImageNet-A missing
            self.assertIn("chưa đủ run (2/5)", text)
            # Tuning run: fuse3 at weight 1 is 5.3 against SCF 0.4 -> go; weight 4 is 2 below its peak.
            self.assertTrue(decision["screen"]["go"])
            self.assertAlmostEqual(decision["screen"]["gain"], 4.9)
            self.assertAlmostEqual(decision["screen"]["by_weight"]["fuse3_snap_b4"], 2.9)
            self.assertIn("đi tiếp", text)
            for i, seed in enumerate((1993, 2015, 2026)):
                self.write_run(root, "imageneta", seed, 1.1 + 0.2 * i)
            text, decision = render(collect([tmp]))
        self.assertEqual(decision["chosen"], {"snap": "fuse_snap_ridge_b1", "fuse3": "fuse3_snap_b1"})
        self.assertEqual(decision["gain"]["n"], 5)
        self.assertTrue(decision["holds"])
        self.assertIn("ĐẠT", text)
        self.assertNotIn("≠", text)

    def test_stopping_rule_on_tuning_run(self):
        from scripts.summarize_svf import collect, render
        with tempfile.TemporaryDirectory() as tmp:
            self.write_run(Path(tmp), "imagenetr", 1993, 0.3)  # fuse3 at w=1: 0.6, SCF ridge 0.4 -> +0.2 < 0.3
            text, decision = render(collect([tmp]))
        self.assertFalse(decision["screen"]["go"])
        self.assertIn("DỪNG", text)


if __name__ == "__main__":
    unittest.main()
