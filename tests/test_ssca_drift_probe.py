import json
import pathlib
import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler

from models.RSIAT_adapter import (
    Learner,
    _validate_ssca_drift_probe_mode,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


class ProbeDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode

    def __len__(self):
        return 12

    def __getitem__(self, index):
        return index, torch.tensor([float(index)]), index // 3


class ProbeDataManager:
    def __init__(self):
        self.requests = []

    def get_dataset(self, indices, source, mode):
        self.requests.append((indices.copy(), source, mode))
        return ProbeDataset(mode)


class IdentityFeatureModel:
    def eval(self):
        return self

    def extract_vector(self, data):
        return data


def _probe_learner(mode):
    return SimpleNamespace(
        args={
            "ssca_drift_probe_mode": mode,
            "seed": 1993,
            "stats_num_workers": 0,
        },
        _known_classes=0,
        _total_classes=4,
        _cur_task=1,
        batch_size=4,
        pin_memory=False,
        train_loader=object(),
    )


class SSCADriftProbeTests(unittest.TestCase):
    def test_absent_mode_keeps_the_existing_loader(self):
        learner = _probe_learner(None)
        learner.args.pop("ssca_drift_probe_mode")

        loader = Learner._build_ssca_drift_probe_loader(
            learner, ProbeDataManager()
        )

        self.assertIs(loader, learner.train_loader)

    def test_paired_loader_uses_deterministic_mode_and_stable_ids(self):
        learner = _probe_learner("paired_deterministic")
        data_manager = ProbeDataManager()
        loader = Learner._build_ssca_drift_probe_loader(learner, data_manager)

        first_ids = torch.cat([batch[0] for batch in loader])
        second_ids = torch.cat([batch[0] for batch in loader])

        self.assertIsInstance(loader.sampler, SequentialSampler)
        self.assertEqual(data_manager.requests[0][1:], ("train", "test"))
        torch.testing.assert_close(first_ids, second_ids)
        torch.testing.assert_close(first_ids, torch.arange(12))
        learner._device = torch.device("cpu")
        features, labels, indices = Learner.extract_features(
            learner, loader, IdentityFeatureModel(), None, return_indices=True
        )
        torch.testing.assert_close(features.flatten(), indices.float())
        torch.testing.assert_close(labels, indices // 3)

    def test_stochastic_control_uses_train_augmentation_and_shuffle(self):
        learner = _probe_learner("stochastic_unpaired")
        data_manager = ProbeDataManager()
        loader = Learner._build_ssca_drift_probe_loader(learner, data_manager)

        self.assertIsInstance(loader.sampler, RandomSampler)
        self.assertEqual(data_manager.requests[0][1:], ("train", "train"))
        self.assertFalse(loader.persistent_workers)

    def test_probe_extraction_preserves_global_rng_and_uses_private_rng(self):
        learner = _probe_learner("stochastic_unpaired")
        learner._network = object()
        learner.semantic_drift = SimpleNamespace(
            _rng_snapshot=lambda: (
                torch.get_rng_state(),
                None,
                np.random.get_state(),
                random.getstate(),
            ),
            _restore_rng=lambda state: (
                torch.set_rng_state(state[0]),
                np.random.set_state(state[2]),
                random.setstate(state[3]),
            ),
        )
        data_manager = ProbeDataManager()
        loader = Learner._build_ssca_drift_probe_loader(learner, data_manager)
        observed_random_values = []

        def fake_extract_features(*args, **kwargs):
            observed_random_values.append(
                (random.random(), float(np.random.random()), float(torch.rand(1)))
            )
            return torch.zeros((1, 2)), torch.zeros(1), torch.zeros(1, dtype=torch.long)

        learner.extract_features = fake_extract_features
        torch.manual_seed(47)
        np.random.seed(47)
        random.seed(47)
        before_torch = torch.get_rng_state().clone()
        before_numpy = np.random.get_state()
        before_python = random.getstate()

        Learner._extract_ssca_drift_probe_features(learner, loader)
        Learner._extract_ssca_drift_probe_features(learner, loader)

        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        self.assertNotEqual(observed_random_values[0], observed_random_values[1])

    def test_only_the_probe_mode_differs_within_each_config_pair(self):
        pairs = [
            (
                "adapter_sgd_ssca_probe_stochastic.json",
                "adapter_sgd_ssca_probe_paired.json",
            ),
            (
                "keeplora_adamw_mlp_sgd_rae_ssca_age_g090_probe_stochastic.json",
                "keeplora_adamw_mlp_sgd_rae_ssca_age_g090_probe_paired.json",
            ),
            (
                "keeplora_adamw_mlp_ssca_age_g090_probe_adamw_stochastic.json",
                "keeplora_adamw_mlp_ssca_age_g090_probe_adamw_paired.json",
            ),
        ]
        for stochastic_name, paired_name in pairs:
            with self.subTest(pair=stochastic_name):
                stochastic = json.loads(
                    (ROOT / "exps" / stochastic_name).read_text(encoding="utf-8")
                )
                paired = json.loads(
                    (ROOT / "exps" / paired_name).read_text(encoding="utf-8")
                )
                stochastic_mode = stochastic.pop("ssca_drift_probe_mode")
                paired_mode = paired.pop("ssca_drift_probe_mode")
                stochastic_prefix = stochastic.pop("prefix")
                paired_prefix = paired.pop("prefix")

                self.assertEqual(stochastic_mode, "stochastic_unpaired")
                self.assertEqual(paired_mode, "paired_deterministic")
                self.assertNotEqual(stochastic_prefix, paired_prefix)
                self.assertEqual(stochastic, paired)
                self.assertTrue(stochastic["semantic_drift"]["enabled"])
                if "probe_adamw" in stochastic_prefix:
                    self.assertEqual(stochastic["optimizer"], "adamw")
                    self.assertNotIn("rae_optimizer", stochastic)
                    self.assertEqual(
                        stochastic["adamw_rae_lr"],
                        stochastic["ae_init_lr"],
                    )
                    self.assertEqual(
                        stochastic["adamw_rae_weight_decay"],
                        stochastic["ae_weight_decay"],
                    )
                elif "keeplora" in stochastic["model_name"]:
                    self.assertEqual(stochastic["optimizer"], "adamw")
                    self.assertEqual(stochastic["rae_optimizer"], "sgd")
                    self.assertEqual(stochastic["ssca_scale_mode"], "age_decay")
                    self.assertEqual(stochastic["ssca_age_gamma"], 0.9)

    def test_adamw_probe_pair_matches_original_age_decay_recipe(self):
        original = json.loads(
            (ROOT / "exps" / "keeplora_adamw_mlp_ssca_age_g090.json")
            .read_text(encoding="utf-8")
        )
        original.pop("prefix")
        for mode, suffix in (
            ("stochastic_unpaired", "stochastic"),
            ("paired_deterministic", "paired"),
        ):
            config = json.loads(
                (
                    ROOT
                    / "exps"
                    / (
                        "keeplora_adamw_mlp_ssca_age_g090_probe_adamw_"
                        + suffix
                        + ".json"
                    )
                ).read_text(encoding="utf-8")
            )
            config.pop("prefix")
            self.assertEqual(config.pop("ssca_drift_probe_mode"), mode)
            self.assertEqual(config, original)

    def test_invalid_probe_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "ssca_drift_probe_mode"):
            _validate_ssca_drift_probe_mode(
                {"ssca_drift_probe_mode": "unknown"}
            )


if __name__ == "__main__":
    unittest.main()
