"""Holevo-CIL (exps/holevo/README.md) and Holevo-POVM (exps/holevo_povm/README.md): unit tests on CPU."""
import json
import logging
import math
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from utils import holevo, qsd

ROOT = Path(__file__).resolve().parents[1]
A_OVERLAYS = ("holevo_vn", "holevo_mcr2", "holevo_spread")
B_OVERLAYS = ("hp_full", "hp_povm", "hp_twin", "hp_twin_povm", "hp_diag")


def overlay(name):
    folder = "holevo" if name.startswith("holevo_") else "holevo_povm"
    return json.loads((ROOT / "exps" / folder / (name + ".json")).read_text(encoding="utf-8"))


def random_state(dim, gen, rank=None):
    a = torch.randn(dim, rank or dim, generator=gen, dtype=torch.float64)
    rho = a @ a.T
    return rho / torch.trace(rho)


# ----------------------------------------------------------------------------- fakes
class TensorSet(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return i, self.x[i], self.y[i]


class FakeManager:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def get_dataset(self, classes, source, mode):
        mask = torch.isin(self.y, torch.as_tensor(np.asarray(classes)))
        return TensorSet(self.x[mask], self.y[mask])


class FakeNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.convnet = nn.Linear(dim, dim)

    def extract_vector(self, x):
        return self.convnet(x)


class FakeLearner:
    """Two tasks of two classes; features = a linear 'adapter' of fixed inputs."""

    def __init__(self, dim=12, per_class=30):
        gen = torch.Generator().manual_seed(0)
        centers = torch.randn(4, dim, generator=gen) * 3
        self.x = torch.cat([centers[c] + torch.randn(per_class, dim, generator=gen) for c in range(4)])
        self.y = torch.arange(4).repeat_interleave(per_class)
        torch.manual_seed(1)
        self._network = FakeNet(dim)
        self._network_module_ptr = self._network
        self._device = torch.device("cpu")
        self.args = {"stats_num_workers": 0}
        self.data_manager = FakeManager(self.x, self.y)

    def set_task(self, task):
        self._cur_task = task
        self._known_classes, self._total_classes = 2 * task, 2 * task + 2
        with torch.no_grad():
            feats = self._network.extract_vector(self.x)
        self._class_means = np.stack([feats[self.y == c].mean(0).numpy() for c in range(self._total_classes)])
        self._class_covs = torch.stack([torch.cov(feats[self.y == c].T) + 1e-3 * torch.eye(feats.shape[1])
                                        for c in range(self._total_classes)])

    def task_data(self):
        mask = (self.y >= self._known_classes) & (self.y < self._total_classes)
        with torch.no_grad():
            return self._network.extract_vector(self.x[mask]), self.y[mask]


def runtime(mode_key, mode, **extra):
    args = {mode_key: mode, "holevo_calib_steps": 3, "holevo_log_every": 1}
    args.update(extra)
    cfg = holevo.holevo_settings(args)
    return holevo.HolevoRuntime(cfg, warmup_epoch=4, device=torch.device("cpu"))


def started(mode_key, mode, **extra):
    learner = FakeLearner()
    learner.set_task(0)
    rt = runtime(mode_key, mode, **extra)
    rt.fix_encoder(learner)
    learner.set_task(1)  # stored statistics of task 0 stay from the same (unchanged) network
    feats, labels = learner.task_data()
    with torch.no_grad():
        learner._network.convnet.weight.add_(0.05 * torch.randn(12, 12, generator=torch.Generator().manual_seed(4)))
    rt.start_task(learner, feats, labels)
    return learner, rt


def batch(learner, n=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    mask = (learner.y >= learner._known_classes) & (learner.y < learner._total_classes)
    idx = torch.nonzero(mask).squeeze(1)[torch.randperm(int(mask.sum()), generator=gen)[:n]]
    feats = learner._network.extract_vector(learner.x[idx])
    return feats, learner.y[idx]


# ----------------------------------------------------------------------------- settings
class SettingsTest(unittest.TestCase):
    def test_off_and_errors(self):
        self.assertIsNone(holevo.holevo_settings({}))
        self.assertIsNone(holevo.holevo_settings({"holevo_mode": "off", "hp_mode": "off"}))
        for bad in ({"holevo_mode": "vn", "hp_mode": "full"}, {"holevo_mode": "x"}, {"hp_mode": "x"},
                    {"holevo_mode": "vn", "holevo_lambda": -1}, {"holevo_mode": "vn", "holevo_lambda": "big"},
                    {"holevo_mode": "vn", "holevo_queue_cap": 1}, {"hp_mode": "full", "hp_tau": 0},
                    {"holevo_mode": "vn", "resume": True}, {"holevo_mode": "vn", "holevo_oracle_diag": 1}):
            with self.assertRaises(ValueError, msg=bad):
                holevo.holevo_settings(bad)

    def test_terms_and_budget(self):
        cfg = holevo.holevo_settings({"holevo_mode": "vn"})
        self.assertEqual((cfg["enc"], cfg["dec"], cfg["target"]), ("vn", None, {"enc": 0.5}))
        cfg = holevo.holevo_settings({"hp_mode": "full"})
        self.assertEqual(cfg["target"], {"enc": 0.25, "dec": 0.25})  # same total budget (spec B 2.5)
        self.assertEqual(holevo.holevo_settings({"hp_mode": "povm"})["target"], {"dec": 0.5})
        twin = holevo.holevo_settings({"hp_mode": "twin"})
        self.assertEqual((twin["enc"], twin["dec"], twin["objective"]), ("mcr2", "gauss", "mcr2"))
        self.assertTrue(holevo.holevo_settings({"hp_mode": "diag"})["diag"])

    def test_overlays_change_only_the_method(self):
        noise = json.loads((ROOT / "exps/measurement/noise_rep0.json").read_text(encoding="utf-8"))
        for name in A_OVERLAYS + B_OVERLAYS:
            cfg = overlay(name)
            self.assertEqual(cfg["prefix"], name)
            self.assertIsNotNone(holevo.holevo_settings(cfg))
            rest = {k: v for k, v in cfg.items() if not k.startswith(("holevo_", "hp_", "qsd_"))}
            rest.pop("offline")
            self.assertEqual({k: v for k, v in rest.items() if k != "prefix"},
                             {k: v for k, v in noise.items() if k != "prefix"}, name)
            self.assertEqual(cfg["holevo_lambda"], "auto")
            self.assertTrue(cfg["holevo_oracle_diag"])
        for name in B_OVERLAYS:  # PGM head at delta = 1/769 (qsd_delta is relative to 1024)
            cfg = overlay(name)
            self.assertAlmostEqual(cfg["qsd_delta"] / 1024, cfg["hp_tau"] / 769, places=7)
            qsd.qsd_settings(cfg)


# ----------------------------------------------------------------------------- spectral functions
class SpectralTest(unittest.TestCase):
    def test_surrogate_gradient_is_exact(self):
        gen = torch.Generator().manual_seed(2)
        for kind, rate in (("vn", None), ("rate", 3.0)):
            rho = random_state(8, gen).requires_grad_(True)
            lam = torch.linalg.eigvalsh(rho)
            exact = -(lam * torch.log(lam)).sum() if kind == "vn" else 0.5 * torch.log1p(rate * lam).sum()
            g_exact, = torch.autograd.grad(exact, rho)
            value = holevo.spectral_trace(rho, kind, eps=1e-15, rate=rate)
            g_sur, = torch.autograd.grad(value, rho)
            self.assertAlmostEqual(float(value), float(exact), places=10)
            # vn: the -I term of the gradient is orthogonal to trace-preserving directions only;
            # compare on the traceless part (rho stays trace one in every use).
            eye = torch.eye(8, dtype=torch.float64)
            proj = lambda g: g - torch.trace(g) / 8 * eye
            torch.testing.assert_close(proj(g_sur), proj(g_exact), atol=1e-8, rtol=1e-6)

    def test_gram_equals_full(self):
        gen = torch.Generator().manual_seed(3)
        f = torch.randn(9, 4, generator=gen, dtype=torch.float64)
        f = f / f.norm()
        for kind, rate in (("vn", None), ("rate", 2.0)):
            self.assertAlmostEqual(float(holevo.spectral_trace(f @ f.T, kind, rate=rate)),
                                   float(holevo.spectral_trace(f.T @ f, kind, rate=rate)), places=10)

    def test_degenerate_spectrum_is_safe(self):
        rho = (torch.eye(6, dtype=torch.float64) / 6).requires_grad_(True)
        value = holevo.spectral_trace(rho, "vn")
        value.backward()
        self.assertAlmostEqual(float(value.detach()), math.log(6), places=10)
        self.assertTrue(torch.isfinite(rho.grad).all())


# ----------------------------------------------------------------------------- encoder, states, queue
class StateTest(unittest.TestCase):
    def test_stats_state_matches_qsd_and_samples(self):
        gen = torch.Generator().manual_seed(5)
        rows = torch.randn(40, 6, generator=gen, dtype=torch.float64) * 2 + 1
        enc = holevo.StatsEncoder(torch.randn(6, generator=gen, dtype=torch.float64), 1.7)
        mean, cov = rows.mean(0), torch.cov(rows.T) + 1e-3 * torch.eye(6, dtype=torch.float64)
        ref = qsd.stats_states(qsd.Encoder(enc.center, torch.tensor(enc.scale, dtype=torch.float64), None),
                               mean[None], cov[None])[0].double()
        torch.testing.assert_close(enc.stats_state(mean, cov), ref, atol=1e-6, rtol=1e-5)
        # With the biased covariance the stats state is exactly the ||v||^2-weighted sample mixture.
        biased = (rows - mean).T @ (rows - mean) / rows.shape[0]
        f = enc.factor(rows)
        torch.testing.assert_close(enc.stats_state(mean, biased), f @ f.T, atol=1e-12, rtol=1e-10)
        self.assertAlmostEqual(float(torch.trace(f @ f.T)), 1.0, places=12)

    def test_encoder_from_stats(self):
        means = torch.tensor([[0.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
        covs = torch.eye(2, dtype=torch.float64).repeat(2, 1, 1)
        enc = holevo.StatsEncoder.from_stats(means, covs)
        torch.testing.assert_close(enc.center, torch.tensor([1.0, 0.0], dtype=torch.float64))
        self.assertAlmostEqual(enc.scale, math.sqrt(3.0))  # mean of tr(cov) + ||mu - m0||^2 = 2 + 1

    def test_queue_fifo(self):
        feats = torch.arange(10, dtype=torch.float32)[:, None].repeat(1, 3)
        labels = torch.tensor([5] * 6 + [6] * 4)
        q = holevo.FeatureQueue(feats, labels, [5, 6], cap=4, device=torch.device("cpu"))
        self.assertEqual({c: b.shape[0] for c, b in q.buf.items()}, {5: 4, 6: 4})  # min(cap, images)
        z = torch.full((3, 3), 100.0, requires_grad=True)
        y = torch.tensor([5, 5, 6])
        rows = q.view(z, y)
        self.assertEqual(rows[5].shape[0], 4)
        self.assertEqual(int((rows[5][:, 0] == 100).sum()), 2)  # two oldest replaced by the batch
        self.assertTrue(rows[5].requires_grad and not q.buf[5].requires_grad)
        q.commit(z, y)
        self.assertEqual(sorted(q.buf[5][:, 0].tolist()), [4.0, 5.0, 100.0, 100.0])
        self.assertEqual(q.ptr[5], 2)
        q.commit(torch.full((3, 3), 200.0), torch.tensor([5, 5, 5]))  # wraps around
        self.assertEqual(sorted(q.buf[5][:, 0].tolist()), [100.0, 200.0, 200.0, 200.0])
        with self.assertRaises(ValueError):
            holevo.FeatureQueue(feats, labels, [7], cap=4, device=torch.device("cpu"))


# ----------------------------------------------------------------------------- objectives
class ObjectiveTest(unittest.TestCase):
    def test_chi_bounds_and_rotation(self):
        learner, rt = started("holevo_mode", "vn")
        feats, labels = batch(learner)
        rows = rt.queue.view(feats.detach(), labels)
        factors = {c: rt.encoder.factor(rows[c]) for c in rt.queue.classes}
        chi = float(rt.encoder_objective(factors, "vn"))
        self.assertGreaterEqual(chi, 0.0)
        self.assertLessEqual(chi, 1.0)  # chi <= H(p) = log C, normalised
        q, _ = torch.linalg.qr(torch.randn(13, 13, generator=torch.Generator().manual_seed(8), dtype=torch.float64))
        rt.old["rho_bar"] = q @ rt.old["rho_bar"] @ q.T
        rotated = float(rt.encoder_objective({c: q @ f for c, f in factors.items()}, "vn"))
        self.assertAlmostEqual(chi, rotated, places=8)  # unitary invariance

    def test_povm_completeness_and_residual(self):
        learner, rt = started("hp_mode", "full")
        old = rt.old
        eye = torch.eye(13, dtype=torch.float64)
        torch.testing.assert_close(old["R_half"] @ old["R_half"],
                                   old["delta"] * torch.linalg.inv(old["rho_bar"] + old["delta"] * eye))
        torch.testing.assert_close(old["P_old"] + old["R_half"] @ old["R_half"], eye)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(old["P_old"]).min()), -1e-12)
        feats, labels = batch(learner)
        rows = rt.queue.view(feats, labels)
        factors = {c: rt.encoder.factor(rows[c]) for c in rt.queue.classes}
        psi = rt.encoder.pure(feats)
        p_new, p_old, p_null = rt.povm(psi, factors)
        torch.testing.assert_close(p_new.sum(1) + p_old + p_null, torch.ones(len(labels), dtype=torch.float64))
        self.assertTrue((p_new >= 0).all() and (p_old >= 0).all() and (p_null >= 0).all())
        # Gradient reaches the features (through psi and the batch rows), not through the whitening.
        loss = -torch.log(p_new.gather(1, (labels - 2)[:, None]) + 1e-12).mean()
        grad, = torch.autograd.grad(loss, feats)
        self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)

    def test_residual_limits(self):
        learner, rt = started("hp_mode", "povm", hp_tau=1e9)  # delta -> inf: R -> I, old elements vanish
        feats, labels = batch(learner)
        factors = {c: rt.encoder.factor(rt.queue.buf[c]) for c in rt.queue.classes}
        _, p_old, _ = rt.povm(rt.encoder.pure(feats.detach()), factors)
        self.assertLess(float(p_old.max()), 1e-6)

    def test_gauss_twin(self):
        learner, rt = started("hp_mode", "twin")
        feats, labels = batch(learner)
        rows = rt.queue.view(feats, labels)
        logits = rt.gauss(feats, rows)
        self.assertEqual(tuple(logits.shape), (len(labels), 4))
        loss = torch.nn.functional.cross_entropy(logits, labels)
        grad, = torch.autograd.grad(loss, feats)
        self.assertTrue(torch.isfinite(loss) and torch.isfinite(grad).all())

    def test_low_rank_queue_is_safe(self):
        learner, rt = started("hp_mode", "full", holevo_queue_cap=3)
        feats, labels = batch(learner)
        terms, info = rt.compute(feats, labels)
        for value in terms.values():
            grad, = torch.autograd.grad(value, feats, retain_graph=True)
            self.assertTrue(torch.isfinite(value) and torch.isfinite(grad).all())
        self.assertLess(info["completeness_error"], 1e-8)


# ----------------------------------------------------------------------------- training step and diagnostics
class RuntimeTest(unittest.TestCase):
    def loss_c(self, learner, feats):
        return feats.square().mean()

    def test_calibration_leaves_epoch_zero_unchanged(self):
        for key, mode, names in (("holevo_mode", "vn", {"enc"}), ("holevo_mode", "mcr2", {"enc"}),
                                 ("hp_mode", "full", {"enc", "dec"}), ("hp_mode", "twin", {"enc", "dec"})):
            learner, rt = started(key, mode)
            with self.assertLogs(level=logging.INFO) as logs:
                for step in range(3):  # epoch 0: ramp weight 0, calibrate only
                    feats, labels = batch(learner, seed=step)
                    out = rt.step_loss(learner, feats, labels, self.loss_c(learner, feats), epoch=0)
                    self.assertIsNone(out)
                    self.assertTrue(all(p.grad is None for p in learner._network.parameters()))
                rt.end_epoch(1, 0)
            self.assertEqual(set(rt.lambdas), names)
            self.assertTrue(all(v is not None and v > 0 for v in rt.lambdas.values()))
            self.assertTrue(any("HOLEVO lambda" in line for line in logs.output))
            feats, labels = batch(learner, seed=9)
            out = rt.step_loss(learner, feats, labels, self.loss_c(learner, feats), epoch=2)
            self.assertTrue(out.requires_grad and torch.isfinite(out))
            # With the calibrated lambda the auxiliary gradient is r* of CosFace's at full weight.
            ratios, _ = rt._grad_ratios(learner, self.loss_c(learner, feats), rt.compute(feats, labels)[0],
                                        {n: rt.lambdas[n] for n in rt.lambdas})
            for name, r in ratios.items():
                self.assertGreater(r, 0)

    def test_diag_never_changes_training(self):
        learner, rt = started("hp_mode", "diag")
        self.assertEqual(rt.lambdas, {"enc": None, "dec": None})
        with self.assertLogs(level=logging.INFO):
            for epoch in (0, 3):
                feats, labels = batch(learner, seed=epoch)
                self.assertIsNone(rt.step_loss(learner, feats, labels, self.loss_c(learner, feats), epoch))
                rt.end_epoch(1, epoch)

    def test_uncalibrated_later_task_fails(self):
        learner, rt = started("holevo_mode", "vn")
        rt.task = 2
        feats, labels = batch(learner)
        with self.assertRaises(RuntimeError):
            rt.step_loss(learner, feats, labels, self.loss_c(learner, feats), epoch=3)

    def test_fixed_lambda(self):
        learner, rt = started("holevo_mode", "spread", holevo_lambda=0.7)
        feats, labels = batch(learner)
        terms, _ = rt.compute(feats, labels)  # view only: the queue is written by step_loss
        out = rt.step_loss(learner, feats, labels, self.loss_c(learner, feats), epoch=4)
        self.assertAlmostEqual(float(out.detach()), 0.7 * float(terms["enc"].detach()), places=6)

    def test_start_needs_encoder(self):
        learner = FakeLearner()
        learner.set_task(1)
        rt = runtime("holevo_mode", "vn")
        with self.assertRaises(RuntimeError):
            rt.start_task(learner, *learner.task_data())

    def test_oracle_preserves_rng_and_state(self):
        for key, mode in (("holevo_mode", "vn"), ("hp_mode", "diag"), ("hp_mode", "twin")):
            learner, rt = started(key, mode, holevo_oracle_diag=True)
            learner._network.train()
            weights = {k: v.clone() for k, v in learner._network.state_dict().items()}
            queue = {c: b.clone() for c, b in rt.queue.buf.items()}
            torch.manual_seed(11); np.random.seed(12); random.seed(13)
            before = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
            with self.assertLogs(level=logging.INFO) as logs:
                report = rt.oracle(learner, "end")
            self.assertTrue(torch.equal(before[0], torch.get_rng_state()))
            self.assertTrue(np.array_equal(before[1], np.random.get_state()[1]))
            self.assertEqual(before[2], random.getstate())
            self.assertTrue(learner._network.training)
            self.assertTrue(all(torch.equal(v, learner._network.state_dict()[k]) for k, v in weights.items()))
            self.assertTrue(all(torch.equal(b, rt.queue.buf[c]) for c, b in queue.items()))
            self.assertTrue(any("HOLEVO oracle task=1 phase=end" in line for line in logs.output))
            for kind in ("vn", "mcr2", "spread"):
                self.assertTrue(math.isfinite(report["oracle"][kind]) and math.isfinite(report["stale"][kind]))
            if mode != "vn":
                self.assertTrue(0.0 <= report["I_orc"] <= 1.0)
            self.assertGreaterEqual(report["trace_distance"]["mean"], 0.0)

    def test_oracle_equals_stale_when_statistics_are_exact(self):
        # With an unchanged network, stats of old classes = real features and a full queue of the new
        # classes (eval features): chi_oracle equals the objective up to unbiased vs biased covariance.
        learner = FakeLearner(per_class=200)
        learner.set_task(0)
        rt = runtime("holevo_mode", "vn", holevo_queue_cap=1000)
        rt.fix_encoder(learner)
        learner.set_task(1)
        rt.start_task(learner, *learner.task_data())
        with self.assertLogs(level=logging.INFO):
            report = rt.oracle(learner, "start")
        self.assertAlmostEqual(report["oracle"]["vn"], report["stale"]["vn"], places=2)


class LearnerIntegrationTest(unittest.TestCase):
    """The real RSIAT incremental loop (task 1) on the toy fixtures of test_stats_diagnostics."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        # Loaded by path: on Kaggle a site-packages "tests" package shadows this folder.
        spec = importlib.util.spec_from_file_location("_sd_fixtures",
                                                      Path(__file__).with_name("test_stats_diagnostics.py"))
        cls.fx = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.fx)

    def run_mode(self, folder, args=None):
        learner = self.fx.legacy_learner(folder)
        if args is not None:
            cfg = holevo.holevo_settings(dict(args, holevo_oracle_diag=True, holevo_log_every=1))
            learner.holevo = holevo.HolevoRuntime(cfg, learner.args["warmup_epoch"], learner._device)
            learner.holevo.fix_encoder(learner)  # as at the end of task 0
        torch.manual_seed(21)
        np.random.seed(22)
        random.seed(23)
        with self.assertLogs(level=logging.INFO) as logs:
            self.fx.run_task(learner)
        return learner, logs.output

    def test_diag_is_identical_to_rsiat(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            plain, _ = self.run_mode(tmp)
            diag, logs = self.run_mode(tmp, {"hp_mode": "diag"})
        for (name, a), b in zip(plain._network.state_dict().items(), diag._network.state_dict().values()):
            self.assertTrue(torch.equal(a, b), name)
        np.testing.assert_array_equal(plain._class_means, diag._class_means)
        self.assertTrue(any("HOLEVO oracle task=1 phase=start" in line for line in logs))
        self.assertTrue(any("HOLEVO oracle task=1 phase=end" in line for line in logs))
        self.assertTrue(any("HOLEVO epoch task=1 epoch=2" in line for line in logs))

    def test_methods_train_and_change_the_adapter(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            plain, _ = self.run_mode(tmp)
            for args in ({"holevo_mode": "vn"}, {"holevo_mode": "mcr2"}, {"holevo_mode": "spread"},
                         {"hp_mode": "full"}, {"hp_mode": "povm"}, {"hp_mode": "twin"}):
                learner, logs = self.run_mode(tmp, args)
                self.assertTrue(all(v is not None and v > 0 for v in learner.holevo.lambdas.values()), args)
                self.assertTrue(any("HOLEVO lambda" in line for line in logs))
                weight = learner._network.convnet.weight
                self.assertTrue(torch.isfinite(weight).all())
                self.assertFalse(torch.equal(weight, plain._network.convnet.weight), args)
                self.assertIsNone(learner.holevo.queue)  # released at the end of the task


class SummaryTest(unittest.TestCase):
    def write_run(self, root, family, dataset, seed, final_gain, oracle_gain=0.01, stale_gain=0.01, ratio_good=True,
                  last3_gain=0.0, i_orc=None, pgm=None, tasks=3):
        import tempfile  # noqa: F401  (root is a tmp dir)
        from scripts.summarize_holevo import BASE
        folder = root / "adapter" / dataset / "0" / "20"
        folder.mkdir(parents=True, exist_ok=True)
        stem = "{}_{}_pretrained_vit_b16_224_in21k_adapter".format(family, seed)
        curve = [BASE["task0"]] + [88.0] * (tasks - 2) + [BASE["final"] + final_gain]
        row = [80.0] * (tasks - 3) + [BASE["last3"] + last3_gain] * 3
        lines = ["x => Continual metrics after task {}: BWT=-1.0, forgetting=5.0, task_acc={}".format(
            tasks - 1, json.dumps(row)), "x => CNN top1 curve: {}".format(json.dumps(curve))]
        for task in range(1, tasks):
            for phase, step in (("start", 0.0), ("end", 1.0)):
                report = {"oracle": {k: 0.5 + step * oracle_gain for k in ("vn", "mcr2", "spread")},
                          "stale": {k: 0.5 + step * stale_gain for k in ("vn", "mcr2", "spread")}}
                if i_orc is not None and phase == "end":
                    report["I_orc"] = i_orc
                lines.append("x => HOLEVO oracle task={} phase={} {}".format(task, phase, json.dumps(report)))
        lines.append('x => HOLEVO lambda {"enc": {"lambda": 0.3}}')
        if pgm is not None:
            lines.append("x => QSD curve head=pgm_stats: {}".format(json.dumps([pgm])))
            lines.append("x => QSD curve head=ridge_stats: {}".format(json.dumps([pgm - 0.2])))
        (folder / (stem + ".log")).write_text("\n".join(lines), encoding="utf-8")
        # predictions: 3 tasks of 20 classes; inter-task errors set the pair ratio
        stats = folder / (stem + "_stats")
        stats.mkdir(exist_ok=True)
        target = np.repeat(np.arange(60), 10)
        pred = target.copy()
        wrong = np.arange(0, 600, 7)
        pred[wrong] = (target[wrong] + (1 if ratio_good else 20)) % 60
        within = wrong[::2] if ratio_good else wrong[::10]  # within-task errors: few of them -> high pair ratio
        pred[within] = (target[within] // 20) * 20 + (target[within] + 1) % 20
        np.savez(stats / "predictions_task_{}.npz".format(tasks - 1), pred=pred, target=target, known_classes=40)

    def test_screening_rules(self):
        import tempfile
        from scripts.summarize_holevo import collect, render
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root, "holevo_vn", "imagenetr", 1993, 0.8)
            self.write_run(root, "holevo_mcr2", "imagenetr", 1993, 0.2, oracle_gain=-0.01, stale_gain=0.02,
                           ratio_good=False)
            self.write_run(root, "holevo_spread", "imagenetr", 1993, 0.9, last3_gain=-2.0)
            text, d = render(collect([tmp]))
        s = d["A (Holevo-CIL)"]["screen"]
        self.assertTrue(s["holevo_vn"]["R"] and s["holevo_vn"]["G"] and s["holevo_vn"]["O"] and s["holevo_vn"]["P"])
        self.assertTrue(s["holevo_vn"]["M"])  # within-task-heavy errors: low pair ratio
        self.assertTrue(s["holevo_vn"]["go"])
        self.assertTrue(s["holevo_mcr2"]["G_detail"]["gamed"])  # stale up, oracle down
        self.assertFalse(s["holevo_mcr2"]["G"] or s["holevo_mcr2"]["M"] or s["holevo_mcr2"]["O"])
        self.assertFalse(s["holevo_spread"]["P"])
        self.assertIn("BỊ LÁCH", text)
        self.assertIn("đi tiếp sang held-out", text)

    def test_decoder_and_held_out(self):
        import tempfile
        from scripts.summarize_holevo import BASE, collect, render
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_run(root, "hp_full", "imagenetr", 1993, 0.7, i_orc=0.05, pgm=81.0)
            self.write_run(root, "hp_diag", "imagenetr", 1993, 0.0, i_orc=0.10, pgm=80.2)
            for i, (ds, seed) in enumerate((("imagenetr", 2015), ("imagenetr", 2026), ("imageneta", 1993),
                                            ("imageneta", 2015), ("imageneta", 2026))):
                self.write_run(root, "hp_full", ds, seed, 0.9 + 0.1 * i, pgm=80.0 + i)
                self.write_run(root, "hp_twin", ds, seed, 0.5 + 0.1 * i)
                self.write_run(root, "qsd2", ds, seed, 0.0)
            text, d = render(collect([tmp]))
        b = d["A+B (Holevo-POVM)"]
        self.assertAlmostEqual(b["screen"]["hp_full"]["I_orc_vs_diag"], -0.05)
        self.assertTrue(b["screen"]["hp_full"]["G"])
        self.assertTrue(b["screen"]["hp_full"]["D"])  # 81.0 - 80.2 >= 0.5
        h = b["held_out"]
        self.assertEqual(h["H"]["n"], 5)
        self.assertTrue(h["H_holds"] and h["Q_holds"] and h["Q_dec_holds"])
        self.assertAlmostEqual(h["Q"]["mean"], 0.4)
        self.assertAlmostEqual(h["H"]["mean"], 1.1)
        self.assertIn("task 0 {:.2f}".format(BASE["task0"]), text)


if __name__ == "__main__":
    unittest.main()
