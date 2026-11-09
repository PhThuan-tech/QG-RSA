import copy
import unittest
from unittest import mock

import numpy as np
import torch

import main
import trainer
from data.data_manager import DataManager
from models.base import BaseLearner
from utils.toolkit import accuracy


class AlgorithmDispatchTests(unittest.TestCase):
    def test_unknown_algorithm_is_rejected_before_dispatch(self):
        with self.assertRaisesRegex(ValueError, "Unsupported algorithm"):
            main.validate_algorithm_config({"algorithm": "typo"})

class SeedIsolationTests(unittest.TestCase):
    def test_each_seed_receives_independent_deep_copy(self):
        source = {
            "seed": [11, 29],
            "device": [-1],
            "nested": {"value": "original"},
        }
        seen = []

        def fake_train(run_args):
            seen.append(copy.deepcopy(run_args))
            run_args["nested"]["value"] = "mutated"
            return 1.0

        with mock.patch.object(trainer, "_train", side_effect=fake_train):
            trainer.RSIAT_train(source)

        self.assertEqual([item["seed"] for item in seen], [11, 29])
        self.assertEqual([item["train_seed"] for item in seen], [11, 29])
        self.assertEqual([item["class_order_seed"] for item in seen], [11, 29])
        self.assertEqual([item["partition_seed"] for item in seen], [11, 29])
        self.assertEqual([item["nested"]["value"] for item in seen], ["original"] * 2)
        self.assertEqual(source["nested"]["value"], "original")

    def test_explicit_seed_roles_are_preserved(self):
        result = trainer._prepare_seeded_args(
            {
                "seed": [7],
                "class_order_seed": 101,
                "train_seed": "{seed}",
                "partition_seed": 303,
            },
            7,
        )
        self.assertEqual(result["class_order_seed"], 101)
        self.assertEqual(result["train_seed"], 7)
        self.assertEqual(result["partition_seed"], 303)

    def test_single_integer_seed_is_supported(self):
        seen = []
        args = {"seed": 13, "device": [-1]}
        with mock.patch.object(
            trainer, "_train", side_effect=lambda run_args: seen.append(run_args) or 2.0
        ):
            trainer.RSIAT_train(args)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["seed"], 13)


class _FixtureData:
    train_data = np.arange(18, dtype=np.uint8).reshape(6, 3)
    train_targets = np.array([0, 0, 1, 1, 2, 2])
    test_data = train_data.copy()
    test_targets = train_targets.copy()
    train_trsf = []
    test_trsf = []
    common_trsf = []
    use_path = False
    class_order = [0, 1, 2]

    def download_data(self):
        return None


class DataOrderTests(unittest.TestCase):
    @mock.patch("data.data_manager._get_idata", return_value=_FixtureData())
    def test_class_order_seed_does_not_mutate_numpy_training_rng(self, _):
        np.random.seed(991)
        expected_next = np.random.random()
        np.random.seed(991)
        manager = DataManager("fixture", True, 17, 1, 1)
        actual_next = np.random.random()

        self.assertEqual(
            manager._class_order,
            np.random.RandomState(17).permutation(3).tolist(),
        )
        self.assertEqual(actual_next, expected_next)

    @mock.patch("data.data_manager._get_idata", return_value=_FixtureData())
    def test_explicit_class_order_is_authoritative(self, _):
        manager = DataManager(
            "fixture", True, 17, 1, 1, class_order=[2, 0, 1]
        )
        self.assertEqual(manager._class_order, [2, 0, 1])

    @mock.patch("data.data_manager._get_idata", return_value=_FixtureData())
    def test_invalid_explicit_class_order_fails(self, _):
        with self.assertRaisesRegex(ValueError, "permutation"):
            DataManager("fixture", True, 17, 1, 1, class_order=[0, 0, 2])


class MetricContractTests(unittest.TestCase):
    def test_grouped_accuracy_uses_uneven_task_sizes(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 0, 3, 0])
        result = accuracy(y_pred, y_true, nb_old=3, task_sizes=[3, 2])

        self.assertEqual(result["00-02"], 66.67)
        self.assertEqual(result["03-04"], 50.0)
        self.assertNotIn("00-09", result)

    def test_eval_caps_requested_topk_at_seen_class_count(self):
        class FixtureNetwork:
            def eval(self):
                return self

            def __call__(self, inputs):
                return {"logits": inputs}

        learner = BaseLearner({"device": [torch.device("cpu")]})
        learner._network = FixtureNetwork()
        learner.topk = 5
        loader = [
            (
                None,
                torch.tensor([[0.1, 0.9, 0.2], [0.8, 0.1, 0.2]]),
                torch.tensor([1, 0]),
            )
        ]
        predicted, targets = learner._eval_cnn(loader)
        learner._known_classes = 0
        learner.task_sizes = [3]
        metrics = learner._evaluate(predicted, targets)

        self.assertEqual(predicted.shape, (2, 3))
        self.assertEqual(metrics["topk_effective"], 3)
        self.assertEqual(metrics["top5"], 100.0)


if __name__ == "__main__":
    unittest.main()
