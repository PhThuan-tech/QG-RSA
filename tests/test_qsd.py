import copy
import json
import logging
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from network.classifier import SimpleContinualLinear
from utils import qsd


def random_states(num_classes=4, dim=9, rank=5, seed=0):
    gen = torch.Generator().manual_seed(seed)
    bank, psis, labels = qsd.StateBank(), [], []
    for c in range(num_classes):
        center = torch.randn(dim, generator=gen) * 3
        rows = center + torch.randn(rank * 4, dim, generator=gen)
        rows = rows / rows.norm(dim=1, keepdim=True)
        psis.append(rows)
        labels.append(torch.full((rows.shape[0],), c))
    psi, y = torch.cat(psis), torch.cat(labels)
    bank.add(psi, y)
    return bank, psi, y


class SettingsTest(unittest.TestCase):
    def test_settings(self):
        self.assertIsNone(qsd.qsd_settings({}))
        cfg = qsd.qsd_settings({"qsd_heads": ["pgm_channel"]})
        self.assertEqual(cfg["heads"], ["linear", "pgm_channel"])
        with self.assertRaises(ValueError):
            qsd.qsd_settings({"qsd_heads": ["svm"]})
        with self.assertRaises(ValueError):
            qsd.qsd_settings({"qsd_heads": ["pgm_channel"], "qsd_delta": -1})


class EncodingTest(unittest.TestCase):
    def test_unit_norm_and_two_copy_block(self):
        gen = torch.Generator().manual_seed(1)
        z = torch.randn(50, 12, generator=gen) + 5
        enc = qsd.Encoder.from_base(z, 4, gen)
        lin, nl = enc.linear(z), enc.nonlinear(z)
        self.assertEqual(lin.shape, (50, 13))
        self.assertEqual(nl.shape, (50, 13 + 16))
        torch.testing.assert_close(lin.norm(dim=1), torch.ones(50), atol=1e-5, rtol=0)
        torch.testing.assert_close(nl.norm(dim=1), torch.ones(50), atol=1e-5, rtol=0)
        self.assertEqual(qsd.Encoder.nominal_dim(769), 1024)
        self.assertEqual(qsd.Encoder.nominal_dim(1793), 2048)


class MeasurementTheoryTest(unittest.TestCase):
    def test_unitary_invariance(self):
        bank, psi, _ = random_states()
        factors, owner = bank.stacked(4)
        q, _ = torch.linalg.qr(torch.randn(9, 9, generator=torch.Generator().manual_seed(3)))
        probs, _ = qsd.pgm(psi, factors, owner, 4, 1e-3)
        rotated, _ = qsd.pgm(psi @ q.T, q @ factors, owner, 4, 1e-3)
        torch.testing.assert_close(probs, rotated, atol=1e-4, rtol=1e-4)

    def test_shrinkage_towards_pooled_state_is_inert(self):
        bank, psi, _ = random_states()
        factors, owner = bank.stacked(4)
        eps = 0.3
        pooled = factors / 2.0  # sqrt(1/C) * L_all: rho_bar = pooled pooled^T
        shrunk = torch.cat([torch.cat([factors[:, owner == c] * (1 - eps) ** 0.5, pooled * eps ** 0.5], 1)
                            for c in range(4)], 1)
        shrunk_owner = torch.cat([torch.full(((owner == c).sum() + pooled.shape[1],), c) for c in range(4)])
        base = qsd.pgm(psi, factors, owner, 4, 1e-3)[0].argmax(1)
        other = qsd.pgm(psi, shrunk, shrunk_owner, 4, 1e-3)[0].argmax(1)
        self.assertTrue(torch.equal(base, other))

    def test_povm_completeness_on_support(self):
        bank, psi, _ = random_states()
        factors, owner = bank.stacked(4)
        rho_bar = factors @ factors.T / 4
        whitening, _ = qsd._inv_sqrt(rho_bar, 1e-12)
        total = qsd.class_scores(psi.double(), factors.double(), owner, 4, whitening).sum(1) / 4
        torch.testing.assert_close(total, torch.ones_like(total), atol=1e-6, rtol=0)

    def test_pgm_and_hierarchy_classify_separable_states(self):
        bank, psi, y = random_states()
        factors, owner = bank.stacked(4)
        probs, info = qsd.pgm(psi, factors, owner, 4, 1e-3)
        self.assertGreater(float((probs.argmax(1) == y).float().mean()), 0.95)
        self.assertGreater(info["effective_rank"], 1.0)
        hier = qsd.pgm_hierarchical(psi, factors, owner, torch.tensor([0, 0, 1, 1]), 4, 1e-3)
        torch.testing.assert_close(hier.sum(1), torch.ones(psi.shape[0], dtype=hier.dtype), atol=1e-6, rtol=0)
        self.assertGreater(float((hier.argmax(1) == y).float().mean()), 0.95)


class ChannelTest(unittest.TestCase):
    def test_identity_and_known_rotation(self):
        bank, psi, y = random_states(dim=6, rank=10)
        a, info = qsd.fit_channel(psi, psi, 0.1)
        self.assertLess(info["deviation"], 1e-5)
        q, _ = torch.linalg.qr(torch.randn(6, 6, generator=torch.Generator().manual_seed(5)))
        a, info = qsd.fit_channel(psi, psi @ q.T, 1e-8)
        torch.testing.assert_close(a, q, atol=1e-4, rtol=0)
        moved = copy.deepcopy(bank)
        moved.transport(a)
        fresh = qsd.StateBank()
        fresh.add(psi @ q.T, y)
        for c in range(4):
            torch.testing.assert_close(moved.factors[c] @ moved.factors[c].T,
                                       fresh.factors[c] @ fresh.factors[c].T, atol=1e-4, rtol=0)


class HeadTest(unittest.TestCase):
    def test_ridge_stats_and_metrics(self):
        gen = torch.Generator().manual_seed(0)
        means = torch.randn(3, 5, generator=gen) * 4
        covs = torch.eye(5).repeat(3, 1, 1)
        z = torch.cat([means[c] + torch.randn(30, 5, generator=gen) for c in range(3)])
        y = torch.arange(3).repeat_interleave(30)
        self.assertGreater(float((qsd.ridge_stats(means, covs, z, 0.1) == y).float().mean()), 0.9)
        target = np.array([0, 1, 2, 3, 0, 2])
        pred = np.array([0, 0, 2, 0, 1, 2])  # two within-task errors (1->0, 0->1), one cross-task (3->0)
        m = qsd.head_metrics(pred, target, [2, 2], 2)
        self.assertEqual(m["task_prediction"], np.around(5 / 6 * 100, 2))
        self.assertAlmostEqual(m["pair_ratio"], (1 / 6 / 2) / (2 / 6 / 1))


class StatsStatesTest(unittest.TestCase):
    """Round A2: states from stored statistics and calibrated fusion."""

    def test_stats_state_is_the_norm_weighted_mixture(self):
        gen = torch.Generator().manual_seed(4)
        z = torch.randn(60, 5, generator=gen, dtype=torch.float64) + 2
        enc = qsd.Encoder(torch.zeros(5, dtype=torch.float64) + 0.5, torch.tensor(1.7, dtype=torch.float64), None)
        v = torch.cat([(z - enc.center) / enc.scale, torch.ones(60, 1, dtype=torch.float64)], 1)
        mixture = v.T @ v / v.square().sum()
        cov = torch.cov(z.T, correction=0)
        rho = qsd.stats_states(enc, z.mean(0, keepdim=True), cov[None]).double()[0]
        torch.testing.assert_close(rho, mixture, atol=1e-6, rtol=0)
        self.assertAlmostEqual(float(torch.trace(rho)), 1.0, places=5)
        self.assertGreater(float(torch.linalg.eigvalsh(rho).min()), -1e-6)

    def test_dense_measurement_matches_factor_measurement(self):
        bank, psi, _ = random_states()
        factors, owner = bank.stacked(4)
        rhos = torch.stack([factors[:, owner == c] @ factors[:, owner == c].T for c in range(4)])
        torch.testing.assert_close(qsd.pgm_dense(psi, rhos, 1e-3), qsd.pgm(psi, factors, owner, 4, 1e-3)[0],
                                   atol=1e-5, rtol=1e-4)
        tasks = torch.tensor([0, 0, 1, 1])
        torch.testing.assert_close(qsd.pgm_dense_hierarchical(psi, rhos, tasks, 1e-3),
                                   qsd.pgm_hierarchical(psi, factors, owner, tasks, 4, 1e-3), atol=1e-5, rtol=1e-4)

    def test_class_statistics_match_base_learner(self):
        gen = torch.Generator().manual_seed(6)
        z, y = torch.randn(40, 3, generator=gen), torch.arange(2).repeat_interleave(20)
        means, covs = qsd.class_statistics(z, y, 2)
        rows = z[y == 1].double().numpy()
        np.testing.assert_allclose(means[1].numpy(), rows.mean(0), atol=1e-10)
        np.testing.assert_allclose(covs[1].numpy(), np.cov(rows.T) + 1e-3 * np.eye(3), atol=1e-6)

    def test_fusion_settings_and_scale(self):
        cfg = qsd.qsd_settings({"qsd_heads": ["pgm_stats", "ridge_stats"], "qsd_fuse": True})
        self.assertIn("fuse_pgm_stats_b0p25", cfg["heads"])
        self.assertEqual(len(cfg["fuse"]), 2 * len(qsd.FUSE_WEIGHTS))
        self.assertTrue(all(name.replace("_", "a").isalnum() for name in cfg["heads"]))  # \w-safe for the log parser
        with self.assertRaises(ValueError):
            qsd.qsd_settings({"qsd_heads": ["pgm_stale"], "qsd_fuse": True})
        with self.assertRaises(ValueError):
            qsd.qsd_settings({"qsd_heads": ["pgm_stats"], "qsd_save_scores": "yes"})
        a = torch.randn(10, 4, generator=torch.Generator().manual_seed(7))
        self.assertAlmostEqual(qsd.fusion_scale(a, a / 8), 8.0, places=5)


class TensorSet(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return i, self.x[i], int(self.y[i])


class Net(torch.nn.Module):
    def __init__(self, weight, fc):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.fc = fc

    def extract_vector(self, x):
        return x @ self.weight


class FakeManager:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def get_dataset(self, indices, source, mode):
        mask = np.isin(self.y.numpy(), indices)
        return TensorSet(self.x[mask], self.y[mask])


class FakeLearner:
    def __init__(self):
        gen = torch.Generator().manual_seed(0)
        d = 12
        centers = torch.randn(4, d, generator=gen) * 3
        self.train_x = torch.cat([centers[c] + torch.randn(40, d, generator=gen) for c in range(4)])
        self.train_y = torch.arange(4).repeat_interleave(40)
        self.test_x = torch.cat([centers[c] + torch.randn(20, d, generator=gen) for c in range(4)])
        self.test_y = torch.arange(4).repeat_interleave(20)
        self.args = {"qsd_heads": list(qsd.HEADS), "qsd_nl_dim": 4, "qsd_seed": 3, "scale": 10.0,
                     "init_lr": 0.05, "weight_decay": 0.0, "ca_epochs": 2, "ca": True, "stats_num_workers": 0}
        self._device = torch.device("cpu")
        self._network = Net(torch.eye(d), SimpleContinualLinear(d, 2))
        self._old_network = None
        self.data_manager = FakeManager(self.train_x, self.train_y)

    def start_task(self, task):
        self._cur_task = task
        self._known_classes, self._total_classes = 2 * task, 2 * task + 2
        self.task_sizes = [2] * (task + 1)
        if task == 1:
            self._old_network = copy.deepcopy(self._network)
            with torch.no_grad():
                self._network.weight.add_(0.05 * torch.randn(12, 12, generator=torch.Generator().manual_seed(9)))
            self._network.fc.update(2)
        self._network.fc.backup()
        with torch.no_grad():
            feats = self._network.extract_vector(self.train_x)
        self._class_means = np.stack([feats[self.train_y == c].mean(0).numpy() for c in range(self._total_classes)])
        self._class_covs = torch.stack([torch.cov(feats[self.train_y == c].T) + 1e-3 * torch.eye(12)
                                        for c in range(self._total_classes)])
        mask = self.test_y < self._total_classes
        self.test_loader = DataLoader(TensorSet(self.test_x[mask], self.test_y[mask]), batch_size=16)

    def _extract_vectors(self, loader):
        xs, ys = [], []
        for _, x, y in loader:
            xs.append(self._network.extract_vector(x).detach().numpy())
            ys.append(np.asarray(y))
        return np.concatenate(xs), np.concatenate(ys)


class DriverTest(unittest.TestCase):
    def run_sequence(self, tmp, extra=None):
        torch.manual_seed(0)  # the fake heads are initialised from the global RNG
        learner, results = FakeLearner(), []
        learner.args.update(extra or {})
        for task in range(2):
            learner.start_task(task)  # builds heads with trunc_normal_ (global RNG): outside the check
            linear = np.asarray(learner.test_y[learner.test_y < learner._total_classes])
            torch.manual_seed(11 + task); np.random.seed(12); random.seed(13)
            before = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
            results.append(qsd.evaluate_qsd(learner, task, linear, str(Path(tmp) / "stats")))
            self.assertTrue(torch.equal(before[0], torch.get_rng_state()))
            self.assertTrue(np.array_equal(before[1], np.random.get_state()[1]))
            self.assertEqual(before[2], random.getstate())
        return learner, results

    def test_two_tasks_preserve_rng_and_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs(level=logging.INFO) as logs:
                learner, results = self.run_sequence(tmp)
            self.assertEqual(set(results[1]["heads"]), set(qsd.HEADS))
            self.assertEqual(len(learner._qsd_curves["pgm_channel"]), 2)
            self.assertGreater(results[1]["diagnostics"]["channel"]["deviation"], 0)
            self.assertTrue(any("QSD curve head=pgm_nonlinear" in line for line in logs.output))
            self.assertTrue((Path(tmp) / "stats" / "qsd_task_1.json").is_file())
            with np.load(Path(tmp) / "stats" / "qsd_predictions_task_1.npz") as saved:
                self.assertEqual(set(saved.files), set(qsd.HEADS) | {"target"})
            self.assertGreater(results[1]["heads"]["pgm_channel"]["top1"], 80)
            # Deterministic given the seed.
            _, again = self.run_sequence(tmp)
            self.assertEqual(results[1]["heads"], again[1]["heads"])

    def test_round_a2_fusion_scores_and_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertLogs(level=logging.INFO):
                learner, results = self.run_sequence(tmp, {"qsd_fuse": True, "qsd_save_scores": True})
            fused = [qsd.fuse_name(s, w) for s in qsd.FUSE_SOURCES for w in qsd.FUSE_WEIGHTS]
            self.assertTrue(set(fused) <= set(results[1]["heads"]))
            self.assertEqual(set(results[1]["diagnostics"]["fuse_scale"]), set(qsd.FUSE_SOURCES))
            # The fake learner recomputes every class mean with the current network, so its stored
            # statistics are the oracle ones: the oracle control must reproduce pgm_stats exactly.
            for result in results:
                np.testing.assert_array_equal(result["predictions"]["pgm_stats_oracle"],
                                              result["predictions"]["pgm_stats"])
            np.testing.assert_array_equal(results[0]["predictions"]["pgm_stats_stale"],
                                          results[0]["predictions"]["pgm_stats"])
            with np.load(Path(tmp) / "stats" / "qsd_scores_task_1.npz") as saved:
                self.assertEqual(set(saved.files), set(qsd.FUSE_SOURCES) | {"linear", "target"})
                self.assertEqual(saved["linear"].shape, (80, 4))
                self.assertEqual(saved["pgm_stats"].dtype, np.float16)
                # Argmax of the saved source scores reproduces the logged head.
                np.testing.assert_array_equal(saved["pgm_stats"].astype(np.float32).argmax(1),
                                              results[1]["predictions"]["pgm_stats"])
                # The logged fusion head is argmax(linear + w k source); float16 copies may flip near-ties.
                k = results[1]["diagnostics"]["fuse_scale"]["pgm_stats"]
                redo = (saved["linear"].astype(np.float64) + 2.0 * k * saved["pgm_stats"].astype(np.float64)).argmax(1)
                agree = (redo == results[1]["predictions"]["fuse_pgm_stats_b2"]).mean()
                self.assertGreaterEqual(agree, 0.95)


class SummaryTest(unittest.TestCase):
    def test_rules(self):
        from scripts.summarize_qsd import collect, render
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "adapter" / "imagenetr" / "0" / "20"
            folder.mkdir(parents=True)
            for k, prefix in enumerate(("qsd", "qsd_rep1", "qsd_rep2")):
                base = [90.0, 80.0 + 0.2 * k]
                heads = {"linear": base, "ridge_stats": [90.0, base[1] + 0.1], "pgm_stale": [90.0, base[1] + 0.2],
                         "pgm_channel": [91.0, base[1] + 1.0 + 0.05 * k]}
                lines = []
                for head, curve in heads.items():
                    lines.append("x => QSD tasks head={}: {}".format(head, json.dumps([curve[-1], curve[-1]])))
                    lines.append("x => QSD mech head={}: {}".format(head, json.dumps({"task_prediction": 90.0,
                                                                                        "pair_ratio": 1.2})))
                    lines.append("x => QSD curve head={}: {}".format(head, json.dumps(curve)))
                lines.append("x => CNN top1 curve: {}".format(json.dumps(base)))
                (folder / "{}_1993_pretrained_vit_b16_224_in21k_adapter.log".format(prefix)).write_text("\n".join(lines))
            runs = collect([tmp])
            text, summary = render(runs, {})
            data = summary["imagenetr|qsd|1993"]
            self.assertTrue(data["primary"]["pgm_channel"]["verdict"].startswith("ĐẠT"))
            self.assertTrue(data["claims"]["pgm_channel − pgm_stale"]["holds"])
            self.assertIn("head linear = CNN curve", text)


class RoundA2SummaryTest(unittest.TestCase):
    def test_weights_are_tuned_once_and_frozen(self):
        from scripts.summarize_qsd_a2 import collect, render
        gains = {("ridge_stats", 0.5): 0.2, ("pgm_stats", 1.0): 0.6, ("pgm_stats", 4.0): 0.1}
        with tempfile.TemporaryDirectory() as tmp:
            runs = [("imagenetr", 1993, k) for k in range(3)] + [("imagenetr", s, 0) for s in (2015, 2026, 2042)] \
                + [("imageneta", 1993, k) for k in range(3)]
            for i, (ds, seed, k) in enumerate(runs):
                folder = Path(tmp) / ds / "logs" / "adapter" / ds / "0" / "20"
                folder.mkdir(parents=True, exist_ok=True)
                base = [90.0, 80.0 + 0.1 * i]
                heads = {"linear": base, "ridge_stats": [89.5, base[1] - 1], "pgm_stats_stale": [89.0, base[1] - 1.2],
                         "pgm_stats": [89.5, base[1] - 1], "pgm_stats_oracle": [89.5, base[1] + 0.5]}
                for s in qsd.FUSE_SOURCES[:1] + ("pgm_stats",):
                    for w in qsd.FUSE_WEIGHTS:
                        g = gains.get((s, w), 0.0) + 0.01 * (i % 3)
                        heads[qsd.fuse_name(s, w)] = [90.0 + g, base[1] + g]
                lines = ["x => QSD curve head={}: {}".format(h, json.dumps(c)) for h, c in heads.items()]
                lines.append("x => CNN top1 curve: {}".format(json.dumps(base)))
                prefix = "qsd2" if k == 0 else "qsd2_rep{}".format(k)
                (folder / "{}_{}_pretrained_vit_b16_224_in21k_adapter.log".format(prefix, seed)).write_text("\n".join(lines))
            text, d = render([r for r in collect([tmp]) if r["family"] == "qsd2"], [])
        self.assertEqual(d["weights"]["pgm_stats"], 1.0)
        self.assertEqual(d["weights"]["ridge_stats"], 0.5)
        self.assertEqual(d["n_tune"], 3)
        self.assertEqual(d["n_heldout"], 6)
        self.assertEqual(d["method"], "ĐẠT")
        self.assertTrue(d["quantum"]["holds"])
        self.assertFalse(d["claims"]["pgm_stats − ridge_stats"]["holds"])  # equal AIA by construction
        self.assertTrue(d["claims"]["pgm_stats − pgm_stats_stale"]["holds"])
        self.assertIn("không có run giai đoạn A", text)


if __name__ == "__main__":
    unittest.main()
