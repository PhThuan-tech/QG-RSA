"""CPU fixtures for paired SSCA and oracle diagnostics; not benchmark results."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from models.RSIAT_adapter import Learner, RS_Loss
from network.classifier import SimpleContinualLinear
from utils import stats_diagnostics as sd


class ToyDataset(Dataset):
    """Fixed images; train mode adds global-RNG noise like a random augmentation."""

    def __init__(self, ids, augment):
        self.labels = np.repeat(np.asarray(ids), 6)
        generator = torch.Generator().manual_seed(10 + int(min(ids)))
        self.images = torch.randn(len(self.labels), 4, generator=generator) + torch.as_tensor(
            self.labels, dtype=torch.float32)[:, None]
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = self.images[i]
        if self.augment:
            image = image + 0.1 * torch.randn(4)
        return i, image, int(self.labels[i])


class ToyManager:
    def get_task_size(self, task):
        return 2

    def get_dataset(self, ids, source, mode, ret_data=False):
        dataset = ToyDataset(ids, augment=(source == "train" and mode == "train"))
        return (dataset.images, dataset.labels, dataset) if ret_data else dataset

    def get_eval_view(self, dataset):
        view = copy.copy(dataset)
        view.augment = False
        return view


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Linear(4, 4)
        self.fc = SimpleContinualLinear(4, 2)

    def extract_vector(self, x):
        return self.convnet(x)

    def forward(self, x):
        return self.fc(self.extract_vector(x))

    @property
    def feature_dim(self):
        return 4

    def update_fc(self, count):
        self.fc.update(count, freeze_old=False)

    def ca_forward(self, features):
        return self.fc(features)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        return self.requires_grad_(False).eval()


def legacy_learner(folder, **stats):
    """A classical RSIAT learner after task 0, before incremental task 1."""
    torch.manual_seed(5)
    learner = Learner.__new__(Learner)
    learner.args = {"beta": 1.0, "gamma": 0.4, "scale": 20.0, "margin": 0.1, "lambda_rs": 0.2,
                    "seed": 3, "ae_code_dims": 2, "ssca": True, "ca": True, "ca_epochs": 1,
                    "inc_epochs": 2, "optimizer": "sgd", "ae_init_lr": 1e-3, "ae_weight_decay": 0,
                    "warmup_epoch": 1, "stats_num_workers": 0, "eval_interval": 0,
                    "_run_log_prefix": str(Path(folder) / "run")}
    learner.args.update(stats)
    learner.qbridge_config = None
    learner.qbridge_runtime = None
    learner.quantum_kernel = None
    learner.use_quantum_kernel_base = learner.use_quantum_kernel_inc = False
    learner.rs_loss_func = RS_Loss(0.5, 0.3)
    learner._network = TinyNet()
    learner._network_module_ptr = learner._network
    learner._old_network = learner._network.copy().freeze()
    learner.old_network_module_ptr = learner._old_network
    learner.old_ae = None
    learner._device = torch.device("cpu")
    learner._cur_task = 0
    learner._known_classes = learner._total_classes = 2
    learner.class_order = [0, 1, 2, 3]
    learner.task_sizes = [2]
    with torch.no_grad():
        base = ToyDataset([0, 1], augment=False)
        features = learner._network.extract_vector(base.images).numpy()
    learner._class_means = np.stack([features[base.labels == c].mean(0) for c in (0, 1)]).astype(np.float64)
    learner._class_covs = torch.eye(4).repeat(2, 1, 1)
    learner.batch_size = 4
    learner.num_workers = 0
    learner.pin_memory = learner.persistent_workers = False
    learner._multiple_gpus = [torch.device("cpu")]
    learner.init_lr = 0.05
    learner.weight_decay = learner.min_lr = 0
    learner.logit_norm = sd.ca_settings(learner.args)[0]  # as Learner.__init__
    learner.topk = 2
    return learner


def run_task(learner):
    with patch("models.RSIAT_adapter.AutoencoderSigmoid", side_effect=lambda **kw: nn.Linear(4, 4)):
        learner.incremental_train(ToyManager())
    return learner


class SettingsTests(unittest.TestCase):
    def test_defaults_are_legacy_and_inactive(self):
        cfg = sd.stats_settings({"ssca": True})
        self.assertEqual((cfg["mode"], cfg["sigma"], cfg["ca_stats_source"]), ("legacy", 4.0, "estimated"))
        self.assertFalse(cfg["needs_paired"] or cfg["needs_oracle"])

    def test_invalid_settings_fail_loudly(self):
        for bad in ({"ssca_mode": "pairs", "ssca": True}, {"ssca_mode": "paired", "ssca": False},
                    {"ca_stats_source": "oracle"}, {"ssca_sigma": 0}, {"ssca_sigma": float("nan")}):
            with self.assertRaises(ValueError):
                sd.stats_settings(bad)


class EstimatorTests(unittest.TestCase):
    def test_matrix_form_matches_legacy_displacement(self):
        rng = np.random.default_rng(0)
        Y1 = torch.from_numpy(rng.normal(size=(30, 6)).astype(np.float32))
        Y2 = Y1 + torch.from_numpy(rng.normal(size=(30, 6)).astype(np.float32))
        means = rng.normal(size=(4, 6))
        legacy = Learner.__new__(Learner).displacement(Y1, Y2, means, 1.5)
        # Legacy subtracts the float32 tensors before promoting to float64.
        np.testing.assert_allclose(sd.kernel_displacement(Y1, Y2, means, 1.5), legacy, rtol=1e-6, atol=1e-7)

    def test_pairing_recovers_class_specific_drift_and_shuffling_does_not(self):
        rng = np.random.default_rng(1)
        means = np.array([[0.0, 0.0], [10.0, 0.0]])
        drift = np.array([[1.0, 0.0], [0.0, -2.0]])
        labels = np.repeat([0, 1], 50)
        Y1 = means[labels] + 0.3 * rng.normal(size=(100, 2))
        Y2 = Y1 + drift[labels]
        paired = means + sd.kernel_displacement(Y1, Y2, means, 1.0)
        shuffled = means + sd.kernel_displacement(Y1, Y2[rng.permutation(100)], means, 1.0)
        oracle = means + drift
        # The 1e-5 weight floor leaks a tiny share of the other class's drift.
        np.testing.assert_allclose(paired, oracle, atol=1e-3)
        self.assertGreater(np.abs(shuffled - oracle).max(), 2.0)
        errors = sd.mean_errors(paired, oracle, means)
        np.testing.assert_allclose(errors["relative"], 0.0, atol=1e-3)
        np.testing.assert_allclose(sd.mean_errors(means, oracle, means)["relative"], 1.0)

    def test_kernel_summary_detects_floor_dominated_rows(self):
        Y1 = np.zeros((5, 3))
        local = sd.kernel_summary(Y1, np.zeros((1, 3)), 1.0)
        far = sd.kernel_summary(Y1, np.full((1, 3), 100.0), 1.0)
        self.assertAlmostEqual(local["effective_sample_size_mean"], 5.0)
        self.assertEqual(far["fraction_rows_floor_dominated"], 1.0)

    def test_report_replaces_nan_with_null(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "r.json"
            sd.write_report(str(path), {"x": float("nan"), "y": np.float64(2.0), "z": [np.int64(3)]})
            self.assertEqual(json.loads(path.read_text()), {"x": None, "y": 2.0, "z": [3]})

    def test_protocol_hash_ignores_diagnostics_but_not_ssca_mode(self):
        from scripts.summarize_qbridge import protocol_hash
        base = {"dataset": "toy", "ssca_mode": "legacy"}
        self.assertEqual(protocol_hash(base), protocol_hash(dict(base, stats_diagnostics=True)))
        self.assertNotEqual(protocol_hash(base), protocol_hash(dict(base, ssca_mode="paired")))


class IncrementalIntegrationTests(unittest.TestCase):
    def test_diagnostics_do_not_change_training_or_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            plain = run_task(legacy_learner(folder))
            plain_rng = torch.rand(3)
            observed = run_task(legacy_learner(folder, stats_diagnostics=True))
            observed_rng = torch.rand(3)
            np.testing.assert_array_equal(plain._class_means, observed._class_means)
            self.assertTrue(torch.equal(plain._class_covs, observed._class_covs))
            for (name, a), (_, b) in zip(plain._network.state_dict().items(),
                                         observed._network.state_dict().items()):
                self.assertTrue(torch.equal(a, b), name)
            self.assertTrue(torch.equal(plain_rng, observed_rng))
            report = json.loads((Path(folder) / "run_stats" / "stats_task_1.json").read_text())
            errors = report["estimator_mean_errors"]
            self.assertEqual(set(errors), {"stale", "legacy", "paired", "applied"})
            self.assertEqual(errors["applied"], errors["legacy"])
            self.assertAlmostEqual(errors["stale"]["all"]["relative"]["median"], 1.0)
            self.assertIn("0", errors["paired"]["by_origin_task"])
            self.assertIsNotNone(report["ssca_kernel"]["effective_sample_size_mean"])

    def test_paired_mode_applies_paired_estimate_without_changing_adaptation(self):
        with tempfile.TemporaryDirectory() as folder:
            legacy = run_task(legacy_learner(folder))
            paired = run_task(legacy_learner(folder, ssca_mode="paired"))
            # The adapter phase is identical; only the drift estimate differs.
            self.assertTrue(torch.equal(legacy._network.convnet.weight, paired._network.convnet.weight))
            self.assertFalse(np.array_equal(legacy._class_means[:2], paired._class_means[:2]))
            np.testing.assert_array_equal(legacy._class_means[2:], paired._class_means[2:])

    def test_paired_mode_report_marks_paired_as_applied(self):
        with tempfile.TemporaryDirectory() as folder:
            run_task(legacy_learner(folder, ssca_mode="paired", stats_diagnostics=True))
            errors = json.loads((Path(folder) / "run_stats" / "stats_task_1.json").read_text())[
                "estimator_mean_errors"]
            self.assertEqual(errors["applied"], errors["paired"])

    def test_oracle_mean_cov_replaces_old_statistics_before_ca(self):
        with tempfile.TemporaryDirectory() as folder:
            learner = legacy_learner(folder, ca_stats_source="oracle_mean_cov", ca_epochs=0)
            run_task(learner)
            old = ToyDataset([0, 1], augment=False)
            with torch.no_grad():
                features = learner._network.extract_vector(old.images).numpy()
            for c in (0, 1):
                mean, cov = sd.class_statistics(features[old.labels == c])
                np.testing.assert_allclose(learner._class_means[c], mean, rtol=1e-6)
                torch.testing.assert_close(learner._class_covs[c], cov.float())
            report = json.loads((Path(folder) / "run_stats" / "stats_task_1.json").read_text())
            self.assertFalse(report["oracle_is_exemplar_free"])
            self.assertEqual(report["ca_stats_source"], "oracle_mean_cov")


class SummaryScriptTests(unittest.TestCase):
    def test_summary_pairs_seeds_and_reads_reports(self):
        from scripts.summarize_stats_diagnostics import collect, render
        with tempfile.TemporaryDirectory() as folder:
            for prefix, curve in (("diag_legacy", [90.0, 80.0]), ("diag_paired", [90.0, 81.5])):
                stem = Path(folder) / "{}_1993_vit".format(prefix)
                run_task(legacy_learner(folder, _run_log_prefix=str(stem), stats_diagnostics=True))
                stem.with_suffix(".log").write_text(
                    "x => CNN: {'total': 80.0, 'old': 79.0, 'new': 81.0}\n"
                    "x => CNN top1 curve: %s\n" % curve, encoding="utf-8")
                # Final-task predictions: paired fixes 3 images and breaks 1.
                target = np.arange(20) % 4
                pred = target.copy()
                if prefix == "diag_legacy":
                    pred[:3] = (pred[:3] + 1) % 4
                else:
                    pred[3] = (pred[3] + 1) % 4
                sd.save_predictions(str(stem) + "_stats", 1, pred, target, 2)
            runs = collect(folder)
            self.assertEqual([r["prefix"] for r in runs], ["diag_legacy", "diag_paired"])
            self.assertEqual(runs[0]["tasks"][0]["task"], 1)
            self.assertIsNotNone(runs[0]["tasks"][0]["pre_total"])
            self.assertEqual(runs[0]["tasks"][0]["post_total"], None)  # log has a single (task-0) line
            text, scenario = render(runs, "diag_legacy")
            self.assertIn("| diag_paired | 1993 | 0.750 | 1.500 | 3 / 1 | 0.6250 |", text)
            self.assertIsNone(scenario)  # no oracle runs yet
            self.assertIn("Chưa đủ run oracle", text)

    def test_verdict_applies_preregistered_scenarios(self):
        from scripts.summarize_stats_diagnostics import verdict
        def c(aia, p=None, per_seed=None):
            test = None if p is None else {"p_value": p}
            return {"aia": aia, "mcnemar": test, "per_seed_aia": per_seed or [aia]}
        cases = [({"oracle_mean_cov": c(3.0), "paired": c(1.0, .01), "replicates": [c(0.1)]}, "A"),
                 ({"oracle_mean_cov": c(3.0), "paired": c(0.15, .5), "replicates": [c(0.1)]}, "B"),
                 ({"oracle_mean_cov": c(1.0), "paired": c(0.2)}, "C"),
                 ({"oracle_mean_cov": c(0.2), "oracle_mean": c(0.1)}, "D")]
        for contrasts, expected in cases:
            self.assertEqual(verdict(contrasts)[1], expected)
        # A large noise floor demotes an otherwise large paired gain.
        noisy = {"oracle_mean_cov": c(3.0), "paired": c(0.5), "replicates": [c(0.4, per_seed=[0.4, -0.4])]}
        self.assertEqual(verdict(noisy)[1], "B")


class PairedTestAndReplicateTests(unittest.TestCase):
    def test_exact_mcnemar(self):
        self.assertEqual(sd.binomial_two_sided(0, 0), 1.0)
        self.assertAlmostEqual(sd.binomial_two_sided(0, 5), 0.0625)
        self.assertAlmostEqual(sd.binomial_two_sided(4, 4), 1.0)
        result = sd.mcnemar([1, 1, 0, 0, 1], [0, 1, 1, 0, 1])
        self.assertEqual((result["only_a_correct"], result["only_b_correct"]), (1, 1))
        with self.assertRaises(ValueError):
            sd.mcnemar([1, 0], [1])

    def test_replicate_validation(self):
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                sd.stats_settings({"rng_replicate": bad})

    def test_replicate_keeps_base_and_changes_incremental_randomness(self):
        with tempfile.TemporaryDirectory() as folder:
            base = legacy_learner(folder)
            replicate = legacy_learner(folder, rng_replicate=1)
            self.assertTrue(torch.equal(base._network.convnet.weight, replicate._network.convnet.weight))
            run_task(base)
            run_task(replicate)
            self.assertFalse(torch.equal(base._network.convnet.weight, replicate._network.convnet.weight))

    def test_pre_ca_accuracy_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            run_task(legacy_learner(folder, stats_diagnostics=True))
            pre = json.loads((Path(folder) / "run_stats" / "stats_task_1.json").read_text())["pre_ca_accuracy"]
            self.assertTrue({"total", "old", "new", "old_to_new_error_rate", "new_to_old_error_rate"} <= set(pre))


class RoundTwoTests(unittest.TestCase):
    def test_auto_sigma_gives_nearest_sample_weight_e_minus_one(self):
        rng = np.random.default_rng(3)
        Y1, means = rng.normal(size=(40, 5)) * 30, rng.normal(size=(7, 5)) * 30  # odd: exact median
        sigma = sd.resolve_sigma("auto_nn", Y1, means)
        _, sq = sd.kernel_weights(Y1, means, 1.0)
        self.assertAlmostEqual(np.median(np.exp(-sq.min(axis=1) / (2 * sigma ** 2))), np.exp(-1), places=6)
        self.assertEqual(sd.resolve_sigma(4.0, Y1, means), 4.0)
        self.assertEqual(sd.stats_settings({"ssca_sigma": "auto_nn"})["sigma"], "auto_nn")
        with self.assertRaises(ValueError):
            sd.stats_settings({"ssca_sigma": "auto"})

    def test_auto_sigma_is_local_and_independent_of_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            plain = run_task(legacy_learner(folder, ssca_mode="paired", ssca_sigma="auto_nn"))
            observed = run_task(legacy_learner(folder, ssca_mode="paired", ssca_sigma="auto_nn",
                                               stats_diagnostics=True))
            np.testing.assert_array_equal(plain._class_means, observed._class_means)
            report = json.loads((Path(folder) / "run_stats" / "stats_task_1.json").read_text())
            self.assertGreater(report["ssca_sigma_used"], 0)
            self.assertEqual(report["ssca_kernel"]["sigma"], report["ssca_sigma_used"])
            self.assertEqual(report["ssca_kernel"]["fraction_rows_floor_dominated"], 0.0)

    def test_ca_options_default_off_and_change_classifier_when_on(self):
        with tempfile.TemporaryDirectory() as folder:
            explicit_off = run_task(legacy_learner(folder, ca_task_margin=0.0, ca_logit_norm=None))
            default = run_task(legacy_learner(folder))
            heads = lambda l: torch.cat([h[0].weight for h in l._network.fc.heads]).detach()
            self.assertTrue(torch.equal(heads(explicit_off), heads(default)))
            for options in ({"ca_task_margin": 0.1}, {"ca_logit_norm": 0.1}):
                changed = run_task(legacy_learner(folder, **options))
                self.assertFalse(torch.equal(heads(changed), heads(default)), options)
                self.assertTrue(torch.isfinite(heads(changed)).all())
            # Adaptation is untouched: CA only refits the head.
            self.assertTrue(torch.equal(changed._network.convnet.weight, default._network.convnet.weight))

    def test_invalid_ca_options(self):
        for bad in ({"ca_logit_norm": 0}, {"ca_task_margin": -0.1}):
            with self.assertRaises(ValueError):
                sd.ca_settings(bad)
        self.assertEqual(sd.ca_settings({}), (None, 0.0))

    def test_error_decomposition_and_candidate_judgment(self):
        from scripts.summarize_stats_diagnostics import error_decomposition, judge
        with tempfile.TemporaryDirectory() as folder:
            target = np.array([0, 1, 2, 3, 4, 5])
            pred = np.array([1, 3, 2, 0, 4, 5])   # within, to-newer, ok, to-older, ok, ok
            sd.save_predictions(folder, 0, pred[:2], target[:2], 0)
            sd.save_predictions(folder, 1, pred[:4], target[:4], 2)
            path = sd.save_predictions(folder, 2, pred, target, 4)
            files = [str(Path(folder) / f"predictions_task_{t}.npz") for t in range(3)]
            e = error_decomposition(files)
            self.assertAlmostEqual(e["within_task"], 100 / 6)
            self.assertAlmostEqual(e["to_newer_task"], 100 / 6)
            self.assertAlmostEqual(e["to_older_task"], 100 / 6)
        self.assertIn("McNemar p<0.05", judge({"aia": 0.5, "mcnemar": {"p_value": 0.01}}, 0.4))
        self.assertEqual(judge({"aia": 0.2, "mcnemar": None}, 0.4), "trong sàn nhiễu")


if __name__ == "__main__":
    unittest.main()
