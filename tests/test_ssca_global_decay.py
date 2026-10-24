import json
import pathlib
import unittest

import numpy as np

from models.RSIAT_adapter import (
    _apply_ssca_gap,
    _ssca_scale_for_task,
    _validate_ssca_scale_config,
)


class SSCAGlobalDecayTests(unittest.TestCase):
    def test_task_zero_uses_unscaled_prototype_logic(self):
        self.assertEqual(
            _ssca_scale_for_task(
                {"ssca_scale_mode": "global_decay", "ssca_global_gamma": 0.9},
                task_index=0,
            ),
            1.0,
        )

    def test_config_without_mode_preserves_original_compensation(self):
        args = {"ssca": True}
        means = np.array([[1.0, 2.0], [3.0, 4.0], [8.0, 9.0]])
        original = means.copy()
        gap = np.array([[0.25, -0.5], [1.0, 0.5]])

        scale = _ssca_scale_for_task(args, task_index=5)
        _apply_ssca_gap(means, known_classes=2, gap=gap, enabled=True, scale=scale)

        expected = original.copy()
        expected[:2] += gap
        np.testing.assert_array_equal(means, expected)

    def test_task_one_keeps_full_ssca(self):
        self.assertEqual(
            _ssca_scale_for_task(
                {"ssca_scale_mode": "global_decay", "ssca_global_gamma": 0.9},
                task_index=1,
            ),
            1.0,
        )

    def test_task_two_uses_gamma(self):
        self.assertAlmostEqual(
            _ssca_scale_for_task(
                {"ssca_scale_mode": "global_decay", "ssca_global_gamma": 0.9},
                task_index=2,
            ),
            0.9,
        )

    def test_task_five_uses_gamma_to_fourth_power(self):
        self.assertAlmostEqual(
            _ssca_scale_for_task(
                {"ssca_scale_mode": "global_decay", "ssca_global_gamma": 0.9},
                task_index=5,
            ),
            0.9**4,
        )

    def test_one_scale_is_applied_to_every_old_class_row(self):
        means = np.zeros((4, 3), dtype=np.float64)
        gap = np.arange(1, 10, dtype=np.float64).reshape(3, 3)
        scale = 0.9**2

        _apply_ssca_gap(means, known_classes=3, gap=gap, enabled=True, scale=scale)

        np.testing.assert_allclose(means[:3], scale * gap)
        np.testing.assert_array_equal(means[3], np.zeros(3))

    def test_ssca_disabled_does_not_mutate_prototypes(self):
        means = np.array([[1.0, 2.0], [3.0, 4.0], [8.0, 9.0]])
        original = means.copy()
        gap = np.ones((2, 2))

        _apply_ssca_gap(means, known_classes=2, gap=gap, enabled=False, scale=0.9)

        np.testing.assert_array_equal(means, original)

    def test_gamma_one_matches_original_at_all_incremental_tasks(self):
        args = {"ssca_scale_mode": "global_decay", "ssca_global_gamma": 1.0}
        for task_index in range(1, 8):
            with self.subTest(task_index=task_index):
                self.assertEqual(_ssca_scale_for_task(args, task_index), 1.0)

    def test_invalid_mode_and_gamma_are_rejected(self):
        invalid_configs = [
            {"ssca_scale_mode": "unknown", "ssca_global_gamma": 0.9},
            {"ssca_scale_mode": "global_decay"},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": -0.01},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": 1.01},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": float("nan")},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": float("inf")},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": "not-a-number"},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": True},
        ]
        for args in invalid_configs:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    _validate_ssca_scale_config(args)

    def test_gamma_zero_keeps_first_increment_and_disables_later_shifts(self):
        args = {"ssca_scale_mode": "global_decay", "ssca_global_gamma": 0.0}
        self.assertEqual(_ssca_scale_for_task(args, task_index=1), 1.0)
        self.assertEqual(_ssca_scale_for_task(args, task_index=2), 0.0)
        self.assertEqual(_ssca_scale_for_task(args, task_index=5), 0.0)

    def test_experiment_config_only_adds_global_decay_and_isolated_prefix(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        source = json.loads(
            (root / "exps" / "keeplora_adamw.json").read_text(encoding="utf-8")
        )
        experiment = json.loads(
            (root / "exps" / "keeplora_adamw_ssca_global_g090.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(experiment["prefix"], "keeplora_adamw_ssca_global_g090")
        self.assertFalse(experiment["resume"])
        self.assertTrue(experiment["isolate_runs"])
        self.assertTrue(experiment["ssca"])
        self.assertEqual(experiment["ssca_scale_mode"], "global_decay")
        self.assertEqual(experiment["ssca_global_gamma"], 0.9)
        expected = source.copy()
        expected.update(
            {
                "prefix": experiment["prefix"],
                "ssca_scale_mode": "global_decay",
                "ssca_global_gamma": 0.9,
            }
        )
        self.assertEqual(experiment, expected)


if __name__ == "__main__":
    unittest.main()
