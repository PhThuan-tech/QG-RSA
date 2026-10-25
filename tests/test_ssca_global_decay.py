import json
import pathlib
import unittest

import numpy as np

from models.RSIAT_adapter import (
    _apply_ssca_gap,
    _ssca_scales_for_classes,
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

    def test_none_mode_preserves_original_compensation(self):
        mode, gamma, scales, groups = _ssca_scales_for_classes(
            {"ssca_scale_mode": "none"},
            task_index=5,
            task_sizes=[10, 10, 10, 10, 10, 10],
            known_classes=50,
            dtype=np.dtype(np.float32),
        )

        self.assertEqual(mode, "none")
        self.assertIsNone(gamma)
        np.testing.assert_array_equal(scales, np.ones((50, 1), dtype=np.float32))
        self.assertEqual(groups, [("all_old", 1.0)])

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
            {"ssca_scale_mode": "age_decay"},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": -0.01},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": 1.01},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": float("nan")},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": float("inf")},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": "not-a-number"},
            {"ssca_scale_mode": "global_decay", "ssca_global_gamma": True},
            {"ssca_scale_mode": "age_decay", "ssca_age_gamma": -0.01},
            {"ssca_scale_mode": "age_decay", "ssca_age_gamma": 1.01},
            {"ssca_scale_mode": "age_decay", "ssca_age_gamma": float("nan")},
            {"ssca_scale_mode": "age_decay", "ssca_age_gamma": True},
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

    def test_age_decay_matches_expected_origin_task_weights_at_task_five(self):
        mode, gamma, scales, groups = _ssca_scales_for_classes(
            {"ssca_scale_mode": "age_decay", "ssca_age_gamma": 0.9},
            task_index=5,
            task_sizes=[10, 10, 10, 10, 10, 10],
            known_classes=50,
            dtype=np.dtype(np.float32),
        )

        self.assertEqual(mode, "age_decay")
        self.assertEqual(gamma, 0.9)
        expected = np.repeat(
            np.array([0.9**4, 0.9**3, 0.9**2, 0.9, 1.0], dtype=np.float32),
            10,
        ).reshape(50, 1)
        np.testing.assert_allclose(scales, expected, rtol=1e-6)
        self.assertEqual(
            groups,
            [
                (0, float(np.float32(0.9**4))),
                (1, float(np.float32(0.9**3))),
                (2, float(np.float32(0.9**2))),
                (3, float(np.float32(0.9))),
                (4, 1.0),
            ],
        )

    def test_age_decay_first_incremental_task_keeps_all_old_classes(self):
        _, _, scales, _ = _ssca_scales_for_classes(
            {"ssca_scale_mode": "age_decay", "ssca_age_gamma": 0.9},
            task_index=1,
            task_sizes=[10, 10],
            known_classes=10,
            dtype=np.dtype(np.float64),
        )
        np.testing.assert_array_equal(scales, np.ones((10, 1), dtype=np.float64))

    def test_age_decay_initial_task_has_no_old_prototypes_to_scale(self):
        _, _, scales, groups = _ssca_scales_for_classes(
            {"ssca_scale_mode": "age_decay", "ssca_age_gamma": 0.9},
            task_index=0,
            task_sizes=[10],
            known_classes=0,
            dtype=np.dtype(np.float32),
        )
        self.assertEqual(scales.shape, (0, 1))
        self.assertEqual(groups, [])

    def test_age_decay_uses_actual_uneven_task_sizes(self):
        _, _, scales, groups = _ssca_scales_for_classes(
            {"ssca_scale_mode": "age_decay", "ssca_age_gamma": 0.9},
            task_index=2,
            task_sizes=[3, 2, 4],
            known_classes=5,
            dtype=np.dtype(np.float32),
        )
        expected = np.array(
            [[0.9], [0.9], [0.9], [1.0], [1.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(scales, expected)
        self.assertEqual(groups, [(0, float(np.float32(0.9))), (1, 1.0)])

    def test_age_decay_scales_gap_rows_without_changing_shape_or_dtype(self):
        means = np.arange(18, dtype=np.float32).reshape(6, 3)
        original = means.copy()
        gap = np.arange(15, dtype=np.float32).reshape(5, 3)
        _, _, scales, _ = _ssca_scales_for_classes(
            {"ssca_scale_mode": "age_decay", "ssca_age_gamma": 0.9},
            task_index=5,
            task_sizes=[1, 1, 1, 1, 1, 1],
            known_classes=5,
            dtype=gap.dtype,
        )

        _apply_ssca_gap(
            means, known_classes=5, gap=gap, enabled=True, scale=scales
        )

        self.assertEqual(means.shape, original.shape)
        self.assertEqual(means.dtype, original.dtype)
        np.testing.assert_allclose(means[:5], original[:5] + scales * gap)
        np.testing.assert_array_equal(means[5], original[5])

    def test_age_decay_rejects_task_sizes_that_do_not_match_old_classes(self):
        with self.assertRaisesRegex(ValueError, "do not match old-class"):
            _ssca_scales_for_classes(
                {"ssca_scale_mode": "age_decay", "ssca_age_gamma": 0.9},
                task_index=2,
                task_sizes=[10, 9, 10],
                known_classes=20,
                dtype=np.dtype(np.float32),
            )

    def test_age_decay_rejects_invalid_task_indices_and_task_sizes(self):
        args = {"ssca_scale_mode": "age_decay", "ssca_age_gamma": 0.9}
        for task_index in (-1, 1.5, True):
            with self.subTest(task_index=task_index):
                with self.assertRaisesRegex(ValueError, "task_index"):
                    _ssca_scales_for_classes(
                        args,
                        task_index=task_index,
                        task_sizes=[10, 10],
                        known_classes=10,
                        dtype=np.dtype(np.float32),
                    )

        with self.assertRaisesRegex(ValueError, "task_index"):
            _ssca_scale_for_task(
                {"ssca_scale_mode": "global_decay", "ssca_global_gamma": 0.9},
                task_index=-1,
            )
        with self.assertRaisesRegex(ValueError, "only supports"):
            _ssca_scale_for_task(
                {"ssca_scale_mode": "age_decay", "ssca_age_gamma": 0.9},
                task_index=1,
            )

        for task_sizes in ([], [10, 10.5], [10, True], [10, 0], [10, -1]):
            with self.subTest(task_sizes=task_sizes):
                with self.assertRaises(ValueError):
                    _ssca_scales_for_classes(
                        args,
                        task_index=1,
                        task_sizes=task_sizes,
                        known_classes=10,
                        dtype=np.dtype(np.float32),
                    )

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
                "semantic_drift": experiment["semantic_drift"],
            }
        )
        self.assertEqual(experiment, expected)

    def test_age_decay_experiment_changes_only_ssca_mode_gamma_and_prefix(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        global_experiment = json.loads(
            (root / "exps" / "keeplora_adamw_ssca_global_g090.json").read_text(
                encoding="utf-8"
            )
        )
        age_experiment = json.loads(
            (root / "exps" / "keeplora_adamw_ssca_age_g090.json").read_text(
                encoding="utf-8"
            )
        )

        expected = global_experiment.copy()
        expected.update(
            {
                "prefix": "keeplora_adamw_ssca_age_g090",
                "ssca_scale_mode": "age_decay",
                "ssca_age_gamma": 0.9,
            }
        )
        expected.pop("ssca_global_gamma")
        self.assertEqual(age_experiment, expected)
        self.assertFalse(age_experiment["resume"])
        self.assertTrue(age_experiment["isolate_runs"])
        self.assertTrue(age_experiment["ssca"])
        self.assertTrue(age_experiment["semantic_drift"]["enabled"])
        self.assertTrue(age_experiment["semantic_drift"]["compute_ssca_diagnostic"])

    def test_age_decay_mlp_experiment_matches_attention_recipe(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        attention = json.loads(
            (root / "exps" / "keeplora_adamw_ssca_age_g090.json").read_text(
                encoding="utf-8"
            )
        )
        mlp = json.loads(
            (root / "exps" / "keeplora_adamw_mlp_ssca_age_g090.json").read_text(
                encoding="utf-8"
            )
        )
        target_keys = {
            "prefix",
            "keeplora_target",
            "keeplora_rank",
            "keeplora_alpha",
            "keeplora_fc1_rank",
            "keeplora_fc2_rank",
            "keeplora_targets",
        }
        for key in set(attention) | set(mlp):
            if key not in target_keys:
                self.assertEqual(attention.get(key), mlp.get(key), key)
        self.assertEqual(mlp["prefix"], "keeplora_adamw_mlp_ssca_age_g090")
        self.assertEqual(mlp["keeplora_target"], "mlp")
        self.assertEqual(mlp["keeplora_fc1_rank"], 25)
        self.assertEqual(mlp["keeplora_fc2_rank"], 29)
        self.assertEqual(mlp["keeplora_targets"], [])
        self.assertTrue(mlp["semantic_drift"]["enabled"])
        self.assertTrue(mlp["semantic_drift"]["compute_ssca_diagnostic"])


if __name__ == "__main__":
    unittest.main()
