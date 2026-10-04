import json
import pathlib
import unittest

import torch

from utils.toolkit import AutoencoderSigmoid

try:
    from models.RSIAT_adapter import Learner
    HAS_LEARNER = True
except ModuleNotFoundError:
    Learner = None
    HAS_LEARNER = False


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RAEExperimentTests(unittest.TestCase):
    def test_zero_residual_is_exact_identity(self):
        torch.manual_seed(19)
        projector = AutoencoderSigmoid(
            input_dims=8, code_dims=4, zero_residual=True
        )
        inputs = torch.randn(5, 8)
        with torch.no_grad():
            outputs = projector(inputs)
        self.assertTrue(torch.equal(outputs, inputs))

    @unittest.skipUnless(HAS_LEARNER, "RSIAT learner dependencies are unavailable")
    def test_shared_lifecycle_reuses_projector(self):
        learner = object.__new__(Learner)
        learner._cur_task = 1
        learner.old_ae = None
        learner.rae_lifecycle = "shared"
        learner.rae_zero_init = True
        learner.rae_generation = 0
        learner.rae_task_id = None
        learner.args = {"ae_code_dims": 4}
        learner._device = torch.device("cpu")

        Learner._prepare_rae_for_task(learner)
        first = learner.old_ae
        Learner._prepare_rae_for_task(learner)
        self.assertIs(learner.old_ae, first)
        self.assertEqual(learner.rae_generation, 1)

    @unittest.skipUnless(HAS_LEARNER, "RSIAT learner dependencies are unavailable")
    def test_per_task_lifecycle_creates_fresh_projector(self):
        learner = object.__new__(Learner)
        learner._cur_task = 1
        learner.old_ae = None
        learner.rae_lifecycle = "per_task"
        learner.rae_zero_init = True
        learner.rae_generation = 0
        learner.rae_task_id = None
        learner.args = {"ae_code_dims": 4}
        learner._device = torch.device("cpu")

        Learner._prepare_rae_for_task(learner)
        first = learner.old_ae
        first.encoder[0].weight.data.fill_(0.25)
        learner._cur_task = 2
        Learner._prepare_rae_for_task(learner)
        self.assertIsNot(learner.old_ae, first)
        self.assertEqual(learner.rae_generation, 2)
        self.assertEqual(learner.rae_task_id, 2)
        self.assertNotEqual(
            first.state_dict()["decoder.2.weight"].data_ptr(),
            learner.old_ae.state_dict()["decoder.2.weight"].data_ptr(),
        )
        self.assertFalse(
            torch.equal(
                first.state_dict()["encoder.0.weight"],
                learner.old_ae.state_dict()["encoder.0.weight"],
            )
        )
        inputs = torch.randn(3, 768)
        with torch.no_grad():
            self.assertTrue(torch.equal(first(inputs), inputs))
            self.assertTrue(torch.equal(learner.old_ae(inputs), inputs))

    @unittest.skipUnless(HAS_LEARNER, "RSIAT learner dependencies are unavailable")
    def test_per_task_lifecycle_preserves_backbone_state(self):
        learner = object.__new__(Learner)
        learner._cur_task = 1
        learner.old_ae = None
        learner.rae_lifecycle = "per_task"
        learner.rae_zero_init = True
        learner.rae_generation = 0
        learner.rae_task_id = None
        learner.args = {"ae_code_dims": 4}
        learner._device = torch.device("cpu")
        backbone_state = {"keeplora": torch.randn(4, 4), "classifier": torch.randn(3, 4)}

        Learner._prepare_rae_for_task(learner)
        first_backbone_state = {
            key: value.clone() for key, value in backbone_state.items()
        }
        learner._cur_task = 2
        Learner._prepare_rae_for_task(learner)

        self.assertEqual(learner.rae_generation, 2)
        for key, value in backbone_state.items():
            self.assertTrue(torch.equal(value, first_backbone_state[key]))

    @unittest.skipUnless(HAS_LEARNER, "RSIAT learner dependencies are unavailable")
    def test_checkpoint_restore_keeps_task_projector_until_next_task(self):
        source = object.__new__(Learner)
        source._cur_task = 2
        source.old_ae = AutoencoderSigmoid(768, 4, zero_residual=True)
        source.old_ae.encoder[0].weight.data.fill_(0.25)
        checkpoint = {
            "old_ae_state_dict": {
                key: value.clone() for key, value in source.old_ae.state_dict().items()
            },
            "rae_generation": 2,
            "rae_task_id": 2,
        }

        restored = object.__new__(Learner)
        restored._cur_task = 2
        restored.old_ae = None
        restored.args = {"ae_code_dims": 4}
        restored.rae_zero_init = True
        restored.rae_lifecycle = "per_task"
        restored.rae_generation = 0
        restored.rae_task_id = None
        restored._device = torch.device("cpu")
        restored._network = object()
        restored._old_network = object()
        Learner._after_load_checkpoint(restored, checkpoint)
        active_task_projector = restored.old_ae

        self.assertEqual(restored.rae_generation, 2)
        self.assertEqual(restored.rae_task_id, 2)
        self.assertTrue(
            all(
                torch.equal(value, active_task_projector.state_dict()[key])
                for key, value in checkpoint["old_ae_state_dict"].items()
            )
        )

        restored._cur_task = 3
        Learner._prepare_rae_for_task(restored)
        self.assertIsNot(restored.old_ae, active_task_projector)
        self.assertEqual(restored.rae_generation, 3)
        self.assertEqual(restored.rae_task_id, 3)

    def test_experiment_configs_select_requested_modes(self):
        configs = {
            "keeplora_cifar224_fullinit.json": ("full_rsiat", "shared"),
            "keeplora_cifar224_fullinit_pertask_rae.json": (
                "full_rsiat",
                "per_task",
            ),
            "adapter_cifar224_zero_rae.json": (None, "shared"),
        }
        for filename, expected in configs.items():
            with self.subTest(filename=filename):
                config = json.loads((ROOT / "exps" / filename).read_text())
                self.assertTrue(config["rae_zero_init"])
                self.assertEqual(config["rae_lifecycle"], expected[1])
                if filename == "adapter_cifar224_zero_rae.json":
                    self.assertEqual(config["model_name"], "adapter")
                    self.assertNotIn("keeplora", config["convnet_type"].lower())
                if expected[0] is not None:
                    self.assertEqual(config["keeplora_init_mode"], expected[0])

    @unittest.skipUnless(HAS_LEARNER, "RSIAT learner dependencies are unavailable")
    def test_adapter_zero_shared_uses_baseline_mode(self):
        learner = object.__new__(Learner)
        learner.rae_zero_init = True
        learner.rae_lifecycle = "shared"
        learner._uses_keeplora = False
        self.assertEqual(Learner._rae_mode(learner), "zero_shared_baseline")


if __name__ == "__main__":
    unittest.main()
