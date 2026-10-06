import copy
import random
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from utils.semantic_drift import SemanticDriftObserver, _pair_metrics


class ProbeDataset(Dataset):
    def __init__(self):
        self.labels = np.array([0, 0, 1, 1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        value = torch.tensor([float(index), 1.0])
        return index, value, int(self.labels[index])


class IdentityModel(nn.Module):
    def extract_vector(self, inputs):
        return inputs


class StatefulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def extract_vector(self, inputs):
        return inputs * self.scale


def make_learner(enabled=True):
    output_root = tempfile.mkdtemp()
    learner = SimpleNamespace(
        args={
            "semantic_drift": {"enabled": enabled},
            "prefix": "test_semantic_drift",
            "output_root": output_root,
        },
        _cur_task=1,
        _known_classes=1,
        _total_classes=2,
        batch_size=2,
        _device=torch.device("cpu"),
    )
    return learner


def make_flagged_learner(**flags):
    learner = make_learner()
    learner.args["semantic_drift"].update(flags)
    return learner


class SemanticDriftTests(unittest.TestCase):
    def test_pair_metrics_identity_orthogonal_opposite(self):
        identity = _pair_metrics(torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 0.0]]))
        orthogonal = _pair_metrics(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
        opposite = _pair_metrics(torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]]))
        self.assertAlmostEqual(float(identity["cosine"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(orthogonal["cosine"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(opposite["cosine"][0]), 2.0, places=6)
        self.assertAlmostEqual(float(identity["norm_ratio"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(identity["l2"][0]), 0.0, places=6)
        self.assertAlmostEqual(float(orthogonal["cosine"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(opposite["cosine"][0]), 2.0, places=6)

    def test_pairing_mismatch_raises(self):
        observer = SemanticDriftObserver(make_learner())
        with self.assertRaises(RuntimeError):
            observer._validate_pair(
                torch.tensor([0]), torch.tensor([1]),
                torch.tensor([0]), torch.tensor([0]),
            )

    def test_observer_preserves_rng_and_model_state(self):
        learner = make_learner()
        observer = SemanticDriftObserver(learner)
        dataset = ProbeDataset()
        model = IdentityModel()
        model.train()
        torch_state = torch.get_rng_state().clone()
        numpy_state = copy.deepcopy(np.random.get_state())
        python_state = random.getstate()
        observer.prepare(dataset, model)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(numpy_state[0], np.random.get_state()[0])
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(model.training)

    def test_disabled_observer_is_noop(self):
        learner = make_learner(enabled=False)
        observer = SemanticDriftObserver(learner)
        self.assertFalse(observer.enabled)
        observer.prepare(ProbeDataset(), IdentityModel())
        self.assertIsNone(observer._pre)

    def test_measurement_writes_task_metrics_without_mutating_model(self):
        learner = make_learner()
        observer = SemanticDriftObserver(learner)
        dataset = ProbeDataset()
        model = IdentityModel()
        observer.prepare(dataset, model)
        observer.measure_post(dataset, model)
        self.assertAlmostEqual(
            observer._task_result["old_sample_drift"]["cosine"]["mean"], 0.0, places=5
        )
        observer.record_ssca(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )
        self.assertIn("ssca_diagnostic", observer._task_result)

    def test_new_class_separation_uses_cosine_similarity(self):
        learner = make_learner()
        observer = SemanticDriftObserver(learner)
        dataset = ProbeDataset()
        observer.prepare(dataset, IdentityModel())
        observer.measure_post(dataset, IdentityModel())
        separation = observer._task_result["old_new_separation"]
        self.assertEqual(separation["pair_count"], 1)
        self.assertIn("pre_pairwise_cosine", separation)
        self.assertEqual(
            separation["interpretation"],
            "higher cosine similarity means less separation",
        )

    def test_observer_does_not_mutate_parameters_or_call_training_updates(self):
        learner = make_learner()
        observer = SemanticDriftObserver(learner)
        dataset = ProbeDataset()
        model = StatefulModel()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        backward_calls = []
        step_calls = []
        original_backward = torch.Tensor.backward
        original_step = torch.optim.Optimizer.step
        try:
            torch.Tensor.backward = lambda *args, **kwargs: backward_calls.append(True)
            torch.optim.Optimizer.step = lambda *args, **kwargs: step_calls.append(True)
            observer.prepare(dataset, model)
            observer.measure_post(dataset, model)
        finally:
            torch.Tensor.backward = original_backward
            torch.optim.Optimizer.step = original_step
        after = {key: value.detach() for key, value in model.state_dict().items()}
        self.assertEqual(backward_calls, [])
        self.assertEqual(step_calls, [])
        for key in before:
            self.assertTrue(torch.equal(before[key], after[key]))
        self.assertTrue(model.training)

    def test_observer_preserves_class_statistics_and_rae_like_state(self):
        learner = make_learner()
        learner._class_means = np.ones((2, 2), dtype=np.float32)
        learner._class_covs = torch.eye(2).repeat(2, 1, 1)
        learner.old_ae = nn.Linear(2, 2)
        observer = SemanticDriftObserver(learner)
        means_before = learner._class_means.copy()
        covs_before = learner._class_covs.clone()
        ae_before = {key: value.detach().clone() for key, value in learner.old_ae.state_dict().items()}
        observer.prepare(ProbeDataset(), IdentityModel())
        observer.measure_post(ProbeDataset(), IdentityModel())
        self.assertTrue(np.array_equal(means_before, learner._class_means))
        self.assertTrue(torch.equal(covs_before, learner._class_covs))
        for key, value in ae_before.items():
            self.assertTrue(torch.equal(value, learner.old_ae.state_dict()[key]))

    def test_memory_and_evaluation_diagnostics_are_recorded(self):
        learner = make_learner()
        learner.args["semantic_drift"]["compute_full_covariance"] = True
        learner.args["increment"] = 1
        learner._class_covs = torch.eye(2).repeat(2, 1, 1)
        observer = SemanticDriftObserver(learner)
        observer.prepare(ProbeDataset(), IdentityModel())
        observer.measure_post(ProbeDataset(), IdentityModel())
        observer.record_ssca(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )
        observer.record_evaluation(
            {
                "top1": 50.0,
                "top5": 75.0,
                "grouped": {"old": 40.0, "new": 60.0},
            }
        )
        memory = observer._task_result["prototype_memory_diagnostic"]
        self.assertIn("stored_covariance_vs_empirical_post_relative_frobenius", memory)
        self.assertIn("effective_ca_mean_vs_empirical_post", memory)
        self.assertEqual(observer._task_result["evaluation"]["old_class_accuracy"], 40.0)

    def test_metric_flags_gate_computation_and_serialization(self):
        learner = make_flagged_learner(
            compute_sample_metrics=False,
            compute_prototype_metrics=False,
            compute_distribution_metrics=False,
            compute_separation_metrics=False,
            compute_relational_metrics=False,
            compute_ssca_diagnostic=False,
            save_per_class=False,
            save_per_sample=False,
        )
        observer = SemanticDriftObserver(learner)
        dataset = ProbeDataset()
        observer.prepare(dataset, IdentityModel())
        observer.measure_post(dataset, IdentityModel())
        observer.record_ssca(
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
        )
        result = observer._task_result
        self.assertEqual(result["old_sample_drift"], {})
        self.assertEqual(result["new_task_shift"], {})
        self.assertEqual(result["old_prototype_drift"], {})
        self.assertEqual(result["old_distribution_drift"], {})
        self.assertEqual(result["old_relational_drift"], {})
        self.assertEqual(result["old_new_separation"], {})
        self.assertEqual(result["ssca_diagnostic"], {})

    def test_default_distribution_path_does_not_store_full_covariance(self):
        observer = SemanticDriftObserver(make_learner())
        dataset = ProbeDataset()
        observer.prepare(dataset, IdentityModel())
        observer.measure_post(dataset, IdentityModel())
        self.assertIn("old_distribution_drift", observer._task_result)
        self.assertNotIn("_old_probe_covariances", observer._task_result)


if __name__ == "__main__":
    unittest.main()
