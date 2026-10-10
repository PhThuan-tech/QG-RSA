"""Synthetic tests only: no downloads, pretrained weights or historical outputs."""

import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

from data.data_manager import DataManager
from models.base import BaseLearner
from models.RSIAT_adapter import Learner, RS_Loss
from utils.experimental_integrity import (
    capture_rng_state, restore_rng_state, preserve_rng_state, make_generator,
    seed_worker, pair_features, dataset_manifest, snapshot_moments,
    memory_drift_diagnostics,
)
from utils.quantum_kernel import QuantumKernelModule
from utils.toolkit import AutoencoderSigmoid


class RandomView:
    def __call__(self, value):
        return value + 0.03 * (random.random() + np.random.rand() + torch.rand(()))


class TinyConvnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(nn.Linear(6, 6), nn.Linear(6, 6))

    def forward(self, images):
        features = images.flatten(1)[:, :6]
        return self.blocks(F.dropout(features, 0.1, training=self.training))


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList()

    def forward(self, features):
        return {"logits": torch.cat([
            F.linear(F.normalize(features, dim=1), F.normalize(head.weight, dim=1))
            for head in self.heads
        ], dim=1)}

    def backup(self):
        self.old_state_dict = copy.deepcopy(self.state_dict())


class TinyNetwork(nn.Module):
    feature_dim = 6

    def __init__(self):
        super().__init__()
        self.convnet = TinyConvnet()
        self.fc = None

    def update_fc(self, size):
        if self.fc is None:
            self.fc = TinyHead()
        self.fc.heads.append(nn.Linear(6, size, bias=False))

    def extract_vector(self, images):
        return self.convnet(images)

    def forward(self, images):
        return self.fc(self.extract_vector(images))

    def ca_forward(self, features):
        return self.fc(features)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        self.requires_grad_(False)
        return self.eval()


class QuietProgress:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_description(self, description):
        pass


def small_autoencoder(input_dims, code_dims):
    return AutoencoderSigmoid(input_dims=6, code_dims=6)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_manager():
    manager = DataManager.__new__(DataManager)
    generator = np.random.default_rng(49)
    manager._train_data = generator.integers(0, 256, (48, 2, 2, 3), dtype=np.uint8)
    manager._train_targets = np.repeat(np.arange(6), 8)
    manager._test_data = generator.integers(0, 256, (24, 2, 2, 3), dtype=np.uint8)
    manager._test_targets = np.repeat(np.arange(6), 4)
    manager._train_trsf = [transforms.ToTensor(), RandomView()]
    manager._test_trsf = [transforms.ToTensor()]
    manager._common_trsf = []
    manager.use_path = False
    manager._increments = [2, 2, 2]
    manager._class_order = list(range(6))
    return manager


def make_learner(quantum=False, workers=0, val_ratio=0.25, optimizer="adam"):
    args = {
        "seed": 1993, "dataset": "synthetic", "init_cls": 2, "increment": 2,
        "device": [torch.device("cpu")], "convnet_type": "synthetic_adapter",
        "optimizer": optimizer, "batch_size": 4, "init_lr": 0.02,
        "weight_decay": 1e-4, "ae_init_lr": 0.03, "ae_weight_decay": 2e-4,
        "init_epochs": 2, "inc_epochs": 2, "ca_epochs": 1, "warmup_epoch": 1,
        "min_lr": 0, "scale": 4.0, "margin": 0.1, "alpha": 1.0,
        "beta": 1.5, "gamma": 0.75, "lambda_rs": 0.2, "rs_margin": 0.5,
        "ae_code_dims": 6, "ssca": True, "ca": True, "val_ratio": val_ratio,
        "num_workers": workers, "stats_num_workers": 0,
        "persistent_workers": True, "pin_memory": False,
        "q_inc_train_mode": "frozen", "q_inc_pair": "old_proj",
        "inc_loss_mode": "mean", "q_calib_samples": 8, "q_init_seed": 1234,
        "use_quantum_kernel_base": quantum, "use_quantum_kernel_inc": quantum,
        "q_gamma_mode": "bounded_learned", "eval_interval": 0, "ca_eval_interval": 0,
    }
    learner = Learner.__new__(Learner)
    BaseLearner.__init__(learner, args)
    learner._network = TinyNetwork()
    learner.batch_size = args["batch_size"]
    learner.num_workers = workers
    learner.pin_memory = False
    learner.persistent_workers = workers > 0
    learner.init_lr = args["init_lr"]
    learner.weight_decay = args["weight_decay"]
    learner.min_lr = 0
    learner.logit_norm = None
    learner.task_sizes = []
    learner.rs_loss_func = RS_Loss(1.0, 0.5)
    learner.old_ae = None
    learner.use_quantum_kernel_base = quantum
    learner.use_quantum_kernel_inc = quantum
    learner.quantum_kernel = QuantumKernelModule(
        input_dim=6, num_qubits=3, num_layers=2, init_seed=1234,
    ) if quantum else None
    learner.q_calibration_loader = None
    learner.class_order = list(range(6))
    return learner


def shutdown_loaders(learner):
    for name in ("train_loader", "val_loader", "test_loader"):
        loader = getattr(learner, name, None)
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()


class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_count = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.thread_count)

    def assert_rng_equal(self, left, right):
        self.assertEqual(left["python"], right["python"])
        self.assertEqual(left["numpy"][0], right["numpy"][0])
        np.testing.assert_array_equal(left["numpy"][1], right["numpy"][1])
        self.assertEqual(left["numpy"][2:], right["numpy"][2:])
        self.assertTrue(torch.equal(left["torch_cpu"], right["torch_cpu"]))
        self.assertEqual(len(left["torch_cuda"]), len(right["torch_cuda"]))
        for a, b in zip(left["torch_cuda"], right["torch_cuda"]):
            self.assertTrue(torch.equal(a, b))

    def test_drift_reuses_same_ids_and_deterministic_views(self):
        manager = make_manager()
        learner = make_learner()
        learner._network.update_fc(2)
        train, validation = manager.get_dataset_with_validation([0, 1], 0.25, 1993)
        drift = learner._make_drift_loader(manager, train)
        first_inputs = list(drift)
        second_inputs = list(drift)
        for left, right in zip(first_inputs, second_inputs):
            for a, b in zip(left, right):
                self.assertTrue(torch.equal(a, b))
        rng = capture_rng_state()
        sampler_state = learner._loader_generator("train_sampler").get_state().clone()
        before = learner.extract_features(drift, learner._network, return_ids=True)
        after = learner.extract_features(drift, learner._network, return_ids=True)
        a, b, ids = pair_features(before, after)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(set(ids.tolist()), set(train.sample_ids.tolist()))
        self.assertFalse(set(ids.tolist()) & set(validation.sample_ids.tolist()))
        self.assertTrue(learner._network.training)
        self.assert_rng_equal(rng, capture_rng_state())
        self.assertTrue(torch.equal(sampler_state, learner._loader_generator("train_sampler").get_state()))

    def test_pairing_aligns_ids_and_rejects_missing_duplicate_or_changed_labels(self):
        before = (torch.tensor([9, 2]), torch.tensor([[1.0], [2.0]]), torch.tensor([0, 1]))
        after = (torch.tensor([2, 9]), torch.tensor([[3.0], [2.0]]), torch.tensor([1, 0]))
        a, b, ids = pair_features(before, after)
        self.assertEqual(ids.tolist(), [2, 9])
        torch.testing.assert_close(b - a, torch.ones_like(a))
        for bad_ids, bad_labels in [([9, 9], [0, 0]), ([9, 3], [0, 1]), ([9, 2], [1, 1])]:
            with self.subTest(ids=bad_ids, labels=bad_labels), self.assertRaises(ValueError):
                pair_features(before, (torch.tensor(bad_ids), after[1], torch.tensor(bad_labels)))

    def test_statistics_use_only_training_subset_and_preserve_old_covariance(self):
        manager, learner = make_manager(), make_learner()
        learner._network.update_fc(2)
        learner._cur_task = 0
        learner._total_classes = 2
        train, val = manager.get_dataset_with_validation([0, 1], 0.25, 1993)
        view = manager.get_eval_view(train)
        loader = DataLoader(view, batch_size=32)
        features, labels = learner._extract_vectors(loader)
        with patch.object(manager, "get_dataset", side_effect=AssertionError("full train leakage")):
            learner._compute_class_mean(manager, training_dataset=train)
        for cls in range(2):
            vectors = features[labels == cls]
            np.testing.assert_allclose(learner._class_means[cls], vectors.mean(0))
            expected = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(6) * 1e-3
            torch.testing.assert_close(learner._class_covs[cls], expected.float())
        self.assertFalse(set(view.sample_ids) & set(val.sample_ids))
        old_covariances = learner._class_covs.clone()
        learner._known_classes = 2
        learner._total_classes = 4
        learner._cur_task = 1
        learner._network.update_fc(2)
        train, _ = manager.get_dataset_with_validation([2, 3], 0.25, 1994)
        learner._compute_class_mean(manager, training_dataset=train)
        self.assertTrue(torch.equal(old_covariances, learner._class_covs[:2]))

    def test_statistics_fail_closed_without_tuning_subset(self):
        with self.assertRaisesRegex(ValueError, "explicit training subset"):
            make_learner()._compute_class_mean(make_manager())

    def test_validation_ids_are_stable_and_all_seen_validation_is_disjoint(self):
        manager = make_manager()
        seen = manager.get_seen_validation_dataset([2, 2], 0.25, 1993)
        expected_ids, training_ids = [], []
        for task, classes in enumerate(([0, 1], [2, 3])):
            train, val = manager.get_dataset_with_validation(classes, 0.25, 1993 + task)
            expected_ids.extend(val.sample_ids.tolist())
            training_ids.extend(train.sample_ids.tolist())
            view = manager.get_eval_view(train)
            np.testing.assert_array_equal(view.sample_ids, train.sample_ids)
            self.assertEqual(view[0][0], int(train.sample_ids[0]))
        self.assertEqual(seen.sample_ids.tolist(), expected_ids)
        self.assertFalse(set(training_ids) & set(seen.sample_ids))

    def test_rng_roundtrip_and_probe_exception_restore_all_cpu_rngs(self):
        set_seed(78)
        initial = capture_rng_state()
        expected = (random.random(), np.random.rand(3), torch.rand(4))
        restore_rng_state(initial)
        actual = (random.random(), np.random.rand(3), torch.rand(4))
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        initial = capture_rng_state()
        with self.assertRaises(RuntimeError), preserve_rng_state():
            random.random(), np.random.rand(), torch.rand(2)
            raise RuntimeError("probe failure")
        self.assert_rng_equal(initial, capture_rng_state())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_rng_roundtrip(self):
        state = capture_rng_state()
        expected = [torch.rand(3, device=i).cpu() for i in range(torch.cuda.device_count())]
        restore_rng_state(state)
        actual = [torch.rand(3, device=i).cpu() for i in range(torch.cuda.device_count())]
        for a, b in zip(actual, expected):
            self.assertTrue(torch.equal(a, b))

    def test_checkpoint_restores_rng_after_random_head_and_ae_reconstruction(self):
        learner = make_learner(quantum=True)
        learner._network.update_fc(2)
        learner._network.update_fc(2)
        learner._cur_task = 1
        learner._known_classes = learner._total_classes = 4
        learner.task_sizes = [2, 2]
        learner.old_ae = small_autoencoder(6, 6)
        learner.quantum_kernel.set_inc_mode("frozen")
        list(RandomSampler(range(13), generator=learner._loader_generator("train_sampler")))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "task_1.pkl")
            learner.save_checkpoint(path)
            state = capture_rng_state()
            expected_order = list(RandomSampler(range(13), generator=learner._loader_generator("train_sampler")))
            set_seed(999)
            restored = make_learner(quantum=True)
            with patch("models.RSIAT_adapter.AutoencoderSigmoid", small_autoencoder):
                restored.load_checkpoint(path)
            self.assert_rng_equal(state, capture_rng_state())
            actual_order = list(RandomSampler(range(13), generator=restored._loader_generator("train_sampler")))
            self.assertEqual(expected_order, actual_order)
            self.assertTrue(restored.resume_reproducible)
            for a, b in zip(learner.old_ae.parameters(), restored.old_ae.parameters()):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(all(not p.requires_grad for p in restored.quantum_kernel.parameters()))

    def test_resume_rejects_changed_hyperparameters_and_split_content(self):
        learner = make_learner()
        metadata = learner._checkpoint_run_metadata()
        for key, changed in [("beta", 9.0), ("optimizer", "sgd"), ("val_ratio", 0.1), ("num_workers", 1)]:
            with self.subTest(key=key):
                learner.args[key] = changed
                with self.assertRaisesRegex(ValueError, key):
                    learner._validate_checkpoint({"run_metadata": metadata})
                learner.args[key] = metadata[key]
        learner.args.pop("beta")
        with self.assertRaisesRegex(ValueError, "beta"):
            learner._validate_checkpoint({"run_metadata": metadata})
        manager = make_manager()
        train, val = learner._task_partition(manager, [0, 1], 0)
        learner.task_sizes = [2]
        learner.split_manifests = {"0": learner._partition_manifest(train, val)}
        learner._validate_saved_partitions(manager)
        manager._train_data[0, 0, 0, 0] ^= 255
        with self.assertRaisesRegex(ValueError, "manifest"):
            learner._validate_saved_partitions(manager)

    def test_malformed_integrity_checkpoint_is_rejected(self):
        learner = make_learner()
        learner._network.update_fc(2)
        learner._cur_task = 0
        learner._known_classes = learner._total_classes = 2
        learner.task_sizes = [2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.pkl"
            learner.save_checkpoint(str(path))
            checkpoint = torch.load(path, weights_only=False)
            checkpoint.pop("loader_generator_states")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "reproducibility state"):
                make_learner().load_checkpoint(str(path))

    def test_diagonal_checkpoint_is_explicitly_approximate(self):
        learner = make_learner()
        learner._network.update_fc(2)
        learner._cur_task = 0
        learner._known_classes = learner._total_classes = 2
        learner.task_sizes = [2]
        learner.args["compact_diagonal_checkpoint"] = True
        learner._class_covs = torch.eye(6).repeat(2, 1, 1)
        learner._class_covs[:, 0, 1] = learner._class_covs[:, 1, 0] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagonal.pkl"
            learner.save_checkpoint(str(path))
            restored = make_learner()
            restored.args["compact_diagonal_checkpoint"] = True
            with self.assertLogs(level="WARNING"):
                restored.load_checkpoint(str(path))
            self.assertFalse(restored.resume_reproducible)
            self.assertTrue(torch.equal(restored._class_covs, torch.eye(6).repeat(2, 1, 1)))

    def test_legacy_checkpoint_warns_and_is_not_marked_exact(self):
        learner = make_learner()
        learner._network.update_fc(2)
        learner._cur_task = 0
        learner._known_classes = learner._total_classes = 2
        learner.task_sizes = [2]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pkl"
            learner.save_checkpoint(str(path))
            checkpoint = torch.load(path, weights_only=False)
            for key in ("rng_state", "loader_generator_states", "experimental_integrity_version", "split_manifests"):
                checkpoint.pop(key)
            torch.save(checkpoint, path)
            restored = make_learner()
            with self.assertLogs(level="WARNING") as logs:
                restored.load_checkpoint(str(path))
            self.assertFalse(restored.resume_reproducible)
            self.assertTrue(any("Legacy checkpoint" in message for message in logs.output))
            newer_path = Path(directory) / "new_from_legacy.pkl"
            restored.save_checkpoint(str(newer_path))
            next_restored = make_learner()
            with self.assertLogs(level="WARNING"):
                next_restored.load_checkpoint(str(newer_path))
            self.assertFalse(next_restored.resume_reproducible)

    def test_source_ids_and_manifest_digest_follow_dataset_content(self):
        manager = make_manager()
        dataset = manager.get_dataset([2, 4], source="train", mode="test")
        expected_ids = np.flatnonzero(np.isin(manager._train_targets, [2, 4]))
        np.testing.assert_array_equal(dataset.sample_ids, expected_ids)
        np.testing.assert_array_equal(dataset.images, manager._train_data[expected_ids])
        before = dataset_manifest(dataset)
        self.assertEqual(before, dataset_manifest(manager.get_eval_view(dataset)))
        dataset.images[0, 0, 0, 0] ^= 255
        self.assertNotEqual(before['sha256'], dataset_manifest(dataset)['sha256'])

    def test_extra_diagnostics_do_not_change_sampler_or_model_updates(self):
        outcomes = []
        for extra_probe in (False, True):
            set_seed(1993)
            learner, manager = make_learner(quantum=True), make_manager()
            learner.topk = 2
            with patch("models.RSIAT_adapter.AutoencoderSigmoid", small_autoencoder), \
                    patch("models.RSIAT_adapter.tqdm", QuietProgress), redirect_stdout(io.StringIO()):
                learner.incremental_train(manager)
                if extra_probe:
                    learner._record_classifier_alignment("extra_probe")
                learner.eval_task()
                learner.after_task()
                learner.incremental_train(manager)
                outcomes.append((copy.deepcopy(learner._network.state_dict()), capture_rng_state()))
        for key, value in outcomes[0][0].items():
            self.assertTrue(torch.equal(value, outcomes[1][0][key]), key)
        self.assert_rng_equal(outcomes[0][1], outcomes[1][1])

    def test_optimizer_groups_match_nonmetric_parameters_and_existing_rates(self):
        for name in ("sgd", "adam"):
            for task in (0, 1):
                summaries = []
                for quantum in (False, True):
                    learner = make_learner(quantum=quantum, optimizer=name)
                    learner._network.update_fc(2)
                    learner._cur_task = task
                    if task:
                        learner.old_ae = small_autoencoder(6, 6)
                        if quantum:
                            learner.quantum_kernel.set_inc_mode("frozen")
                    optimizer = learner._build_optimizer()
                    expected = list(learner._network.parameters())
                    if task:
                        expected += list(learner.old_ae.parameters())
                    nonmetric = [g for g in optimizer.param_groups if not g['group_name'].startswith("metric_")]
                    ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
                    self.assertEqual(len(ids), len(set(ids)))
                    self.assertEqual({id(p) for g in nonmetric for p in g["params"]}, {id(p) for p in expected})
                    if task:
                        group = next(g for g in nonmetric if g["group_name"] == "old_ae")
                        self.assertEqual(group["lr"], learner.args["ae_init_lr"])
                        self.assertEqual(group["weight_decay"], learner.args["ae_weight_decay"])
                    else:
                        for g in nonmetric:
                            self.assertEqual(g['lr'], 0.01 if name == 'sgd' else learner.init_lr)
                    summaries.append([(g['group_name'], g['lr'], g['weight_decay'], sum(p.numel() for p in g['params'])) for g in nonmetric])
                self.assertEqual(summaries[0], summaries[1])

    def test_adamw_updates_baseline_ae_and_frozen_metric_stays_unchanged(self):
        for quantum in (False, True):
            learner = make_learner(quantum=quantum)
            learner._network.update_fc(2)
            learner._cur_task = 1
            learner.old_ae = small_autoencoder(6, 6)
            learner._class_means = np.random.default_rng(0).normal(size=(2, 6))
            if quantum:
                learner.quantum_kernel.set_inc_mode("frozen")
                metric_before = copy.deepcopy(learner.quantum_kernel.state_dict())
            before = [p.detach().clone() for p in learner.old_ae.parameters()]
            optimizer = learner._build_optimizer()
            optimizer.zero_grad()
            learner._inc_loss(torch.randn(3, 6, requires_grad=True), torch.randn(3, 6)).backward()
            optimizer.step()
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, learner.old_ae.parameters())))
            if quantum:
                for key, value in metric_before.items():
                    self.assertTrue(torch.equal(value, learner.quantum_kernel.state_dict()[key]))

    def test_trainable_incremental_metric_groups_are_complete(self):
        learner = make_learner(quantum=True)
        learner._network.update_fc(2)
        learner._cur_task = 1
        learner.old_ae = small_autoencoder(6, 6)
        learner.args["q_inc_train_mode"] = "trainable"
        learner.quantum_kernel.set_inc_mode("trainable")
        optimizer = learner._build_optimizer()
        actual = {id(p) for g in optimizer.param_groups for p in g['params']}
        expected = {id(p) for module in (learner._network, learner.old_ae, learner.quantum_kernel) for p in module.parameters() if p.requires_grad}
        self.assertEqual(actual, expected)

    def test_memory_diagnostics_measure_stored_updates_without_mutation(self):
        means = np.array([[1., 2.], [3., 4.]])
        covs = torch.eye(2).repeat(2, 1, 1)
        before = snapshot_moments(means, covs, 2)
        means[0] += [3, 4]
        covs[1] *= 2
        expected_means, expected_covs = means.copy(), covs.clone()
        diagnostic = memory_drift_diagnostics(before, means, covs)
        self.assertEqual(diagnostic["old_memory_mean_update_l2"], [5.0, 0.0])
        self.assertAlmostEqual(diagnostic["old_memory_covariance_update_fro"][1], 2 ** 0.5, places=6)
        np.testing.assert_array_equal(means, expected_means)
        self.assertTrue(torch.equal(covs, expected_covs))

    def test_pre_post_ca_evaluation_is_read_only_and_uses_validation(self):
        learner = make_learner()
        learner._cur_task = 1
        learner._known_classes = 2
        learner._total_classes = 4
        learner.task_sizes = [2, 2]
        learner._network.update_fc(2)
        validation_loader = object()
        learner.evaluation_loader = validation_loader
        learner.test_loader = object()
        learner.topk = 2
        def predictions(loader):
            self.assertIs(loader, validation_loader)
            random.random(), np.random.rand(), torch.rand(1)
            learner._network.eval()
            return np.array([[0, 1], [3, 2]]), np.array([0, 2])
        weights = copy.deepcopy(learner._network.state_dict())
        state = capture_rng_state()
        with patch.object(learner, "_eval_cnn", side_effect=predictions):
            pre = learner._record_classifier_alignment("pre_ca")
            post = learner._record_classifier_alignment("post_ca")
            learner.eval_task()
        self.assertEqual(pre["grouped"]["old"], 100.0)
        self.assertEqual(post["grouped"]["new"], 0.0)
        self.assertEqual(pre["split"], "validation")
        self.assert_rng_equal(state, capture_rng_state())
        for key, value in weights.items():
            self.assertTrue(torch.equal(value, learner._network.state_dict()[key]))

    def _assert_end_to_end_resume(self, quantum, workers, val_ratio):
        manager = make_manager()
        set_seed(1993)
        continuous = make_learner(quantum=quantum, workers=workers, val_ratio=val_ratio)
        continuous.topk = 2
        with tempfile.TemporaryDirectory() as directory, \
                patch("models.RSIAT_adapter.AutoencoderSigmoid", small_autoencoder), \
                patch("models.RSIAT_adapter.tqdm", QuietProgress), redirect_stdout(io.StringIO()):
            path = str(Path(directory) / "task_0.pkl")
            for task in range(3):
                continuous.incremental_train(manager)
                continuous.eval_task()
                continuous.after_task()
                if task == 0:
                    continuous.save_checkpoint(path)
            final_rng = capture_rng_state()
            shutdown_loaders(continuous)
            set_seed(867)
            resumed = make_learner(quantum=quantum, workers=workers, val_ratio=val_ratio)
            resumed.topk = 2
            resumed.load_checkpoint(path)
            try:
                for task in range(1, 3):
                    resumed.incremental_train(manager)
                    resumed.eval_task()
                    resumed.after_task()
                self.assert_rng_equal(final_rng, capture_rng_state())
                for module_a, module_b in [(continuous._network, resumed._network), (continuous.old_ae, resumed.old_ae)]:
                    for key, value in module_a.state_dict().items():
                        self.assertTrue(torch.equal(value, module_b.state_dict()[key]), key)
                if quantum:
                    for key, value in continuous.quantum_kernel.state_dict().items():
                        self.assertTrue(torch.equal(value, resumed.quantum_kernel.state_dict()[key]), key)
                np.testing.assert_array_equal(continuous._class_means, resumed._class_means)
                self.assertTrue(torch.equal(continuous._class_covs, resumed._class_covs))
                self.assertEqual(continuous.split_manifests, resumed.split_manifests)
                self.assertEqual(continuous.task_diagnostics, resumed.task_diagnostics)
                self.assertTrue(resumed.resume_reproducible)
                for role, generator in continuous._loader_generators.items():
                    self.assertTrue(torch.equal(generator.get_state(), resumed._loader_generators[role].get_state()), role)
                for diagnostic in resumed.task_diagnostics:
                    self.assertEqual(diagnostic['memory']['old_memory_covariance_update_fro'], [0.] * diagnostic['memory']['old_class_count'])
            finally:
                shutdown_loaders(resumed)

    def test_end_to_end_resume_matches_continuous_baseline_and_qksr(self):
        for quantum in (False, True):
            for ratio in (0.0, 0.25):
                with self.subTest(quantum=quantum, val_ratio=ratio):
                    self._assert_end_to_end_resume(quantum, workers=0, val_ratio=ratio)

    def test_end_to_end_resume_matches_with_persistent_worker(self):
        self._assert_end_to_end_resume(quantum=True, workers=1, val_ratio=0.25)

    def test_tuning_pipeline_never_fetches_test_or_validation_for_training_statistics(self):
        learner, manager = make_learner(quantum=True), make_manager()
        learner.topk = 2
        original = manager.get_dataset
        def protected(classes, source, mode, **kwargs):
            self.assertNotEqual(source, "test")
            return original(classes, source, mode, **kwargs)
        with patch.object(manager, "get_dataset", side_effect=protected), \
                patch("models.RSIAT_adapter.AutoencoderSigmoid", small_autoencoder), \
                patch("models.RSIAT_adapter.tqdm", QuietProgress), redirect_stdout(io.StringIO()):
            for task in range(2):
                learner.incremental_train(manager)
                train_ids = set(learner.train_dataset.sample_ids)
                val_ids = set(learner.val_loader.dataset.sample_ids)
                self.assertFalse(train_ids & val_ids)
                self.assertIsNone(learner.test_loader)
                self.assertTrue(set(learner.q_calibration_loader.dataset.dataset.sample_ids).issubset(train_ids))
                learner.after_task()
            diagnostic = learner.task_diagnostics[0]
            self.assertTrue(set(diagnostic['drift_paired_sample_ids']).issubset(train_ids))
            self.assertEqual(diagnostic['statistics_train_count'], len(train_ids))
            self.assertEqual(diagnostic['pre_ca']['split'], 'validation')
            self.assertEqual(diagnostic['post_ca']['split'], 'validation')


if __name__ == "__main__":
    unittest.main()
