import gc
import importlib.util
import sys
import types
import unittest
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from KeepLora.peft_modules import KeepLoRA


HAS_TIMM = importlib.util.find_spec("timm") is not None

if not HAS_TIMM:
    timm_stub = types.ModuleType("timm")
    timm_stub.__path__ = []
    timm_layers_stub = types.ModuleType("timm.layers")
    timm_layers_stub.trunc_normal_ = nn.init.trunc_normal_
    timm_layers_stub.DropPath = nn.Identity

    # Keep model-construction tests independent of optional timm installation.
    class TestPatchEmbed(nn.Module):
        def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
            super().__init__()
            self.grid_size = (img_size // patch_size, img_size // patch_size)
            self.num_patches = self.grid_size[0] * self.grid_size[1]
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
            )

        def forward(self, inputs):
            return self.proj(inputs).flatten(2).transpose(1, 2)

    timm_layers_stub.PatchEmbed = TestPatchEmbed
    timm_stub.layers = timm_layers_stub
    timm_models_stub = types.ModuleType("timm.models")
    timm_models_stub.__path__ = []
    timm_models_stub.register_model = lambda function: function
    sys.modules.setdefault("timm", timm_stub)
    sys.modules.setdefault("timm.layers", timm_layers_stub)
    sys.modules.setdefault("timm.models", timm_models_stub)

if importlib.util.find_spec("tqdm") is None:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda iterable, *args, **kwargs: iterable
    sys.modules.setdefault("tqdm", tqdm_stub)

if importlib.util.find_spec("scipy") is None:
    scipy_stub = types.ModuleType("scipy")
    scipy_stub.__path__ = []
    scipy_spatial_stub = types.ModuleType("scipy.spatial")
    scipy_spatial_stub.__path__ = []
    scipy_distance_stub = types.ModuleType("scipy.spatial.distance")
    scipy_distance_stub.cdist = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("scipy cdist is not used by these unit tests")
    )
    sys.modules.setdefault("scipy", scipy_stub)
    sys.modules.setdefault("scipy.spatial", scipy_spatial_stub)
    sys.modules.setdefault("scipy.spatial.distance", scipy_distance_stub)

try:
    from models.RSIAT_adapter import Learner, RS_Loss
    from models.base import BaseLearner
    HAS_LEARNER_IMPORT = True
except ImportError:
    Learner = None
    RS_Loss = None
    BaseLearner = None
    HAS_LEARNER_IMPORT = False

try:
    from network.vision_transformer_keeplora import KeepLoRAVisionTransformer
    HAS_MODEL_IMPORT = True
except ImportError:
    KeepLoRAVisionTransformer = None
    HAS_MODEL_IMPORT = False


class TinyNetwork:
    def __init__(self, class_count=4):
        self.classifier = nn.Linear(4, class_count)

    def extract_vector(self, inputs):
        return inputs

    def fc(self, features):
        return {"logits": self.classifier(features)}


class FixedCosineLoss:
    def __call__(self, logits, targets):
        return logits.square().mean() + targets.float().mean() * 0.0


@unittest.skipUnless(HAS_LEARNER_IMPORT, "the RSIAT learner dependencies are required")
class FullRsiatInitializationLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.inputs = torch.randn(4, 4, requires_grad=True)
        self.targets = torch.tensor([0, 0, 1, 1])
        self.network = TinyNetwork()
        self.cosine = FixedCosineLoss()

    def make_learner(self, task, mode="full_rsiat"):
        learner = SimpleNamespace(
            _known_classes=0 if task == 0 else 2,
            _cur_task=task,
            keeplora_init_mode=mode,
            args={
                "lambda_rs": 0.35,
                "beta": 1.7,
                "gamma": 0.6,
            },
            rs_loss_func=RS_Loss(lamda=0.5, margin=0.3),
            old_ae=nn.Linear(4, 4),
            old_network_module_ptr=SimpleNamespace(
                extract_vector=lambda x: x.detach() * 0.7
            ),
            _class_means=np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
            _device=torch.device("cpu"),
        )
        learner._inc_loss_components = MethodType(
            Learner._inc_loss_components, learner
        )
        return learner

    def test_task_zero_uses_full_cosine_plus_rs_loss(self):
        learner = self.make_learner(task=0)
        _, actual, components = Learner._keeplora_initialization_loss(
            learner, self.network, self.inputs, self.targets, self.cosine
        )
        logits = self.network.fc(self.inputs)["logits"]
        expected_cosine = self.cosine(logits, self.targets)
        expected_rs = learner.rs_loss_func(self.inputs, self.targets)
        expected = expected_cosine + learner.args["lambda_rs"] * expected_rs
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(components["L_RS"], expected_rs))
        self.assertTrue(torch.allclose(components["L_init"], expected))

    def test_incremental_task_uses_cosine_alignment_and_orthogonality(self):
        learner = self.make_learner(task=1)
        _, actual, components = Learner._keeplora_initialization_loss(
            learner, self.network, self.inputs, self.targets + 2, self.cosine
        )
        features = self.network.extract_vector(self.inputs)
        features_old = learner.old_network_module_ptr.extract_vector(self.inputs)
        align, orth = Learner._inc_loss_components(
            learner, features, features_old
        )
        expected_cosine = self.cosine(
            self.network.fc(features)["logits"][:, 2:], self.targets
        )
        expected = (
            expected_cosine
            + learner.args["beta"] * align
            + learner.args["gamma"] * orth
        )
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.allclose(components["L_align"], align))
        self.assertTrue(torch.allclose(components["L_orth"], orth))
        self.assertTrue(torch.allclose(components["L_init"], expected))

        orth_gradient = torch.autograd.grad(
            orth, features, allow_unused=True, retain_graph=True
        )[0]
        align_gradient = torch.autograd.grad(align, features, allow_unused=True)[0]
        self.assertIsNone(orth_gradient)
        self.assertIsNotNone(align_gradient)

    def test_cosine_mode_preserves_cosine_initialization(self):
        learner = self.make_learner(task=0, mode="cosine")
        _, actual, components = Learner._keeplora_initialization_loss(
            learner, self.network, self.inputs, self.targets, self.cosine
        )
        expected = self.cosine(
            self.network.fc(self.inputs)["logits"], self.targets
        )
        self.assertTrue(torch.allclose(actual, expected))
        self.assertNotIn("L_RS", components)

    def test_incremental_training_loss_remains_weighted_existing_components(self):
        learner = self.make_learner(task=1)
        features = self.inputs
        features_old = self.inputs.detach() * 0.7
        align, orth = Learner._inc_loss_components(
            learner, features, features_old
        )
        actual = Learner._inc_loss(learner, features, features_old)
        expected = learner.args["beta"] * align + learner.args["gamma"] * orth
        self.assertTrue(torch.allclose(actual, expected))

    def test_legacy_checkpoint_without_init_mode_remains_loadable(self):
        args = {
            "dataset": "cifar224",
            "seed": 1993,
            "init_cls": 10,
            "increment": 10,
            "model_name": "keeplora",
            "convnet_type": "pretrained_vit_b16_224_in21k_keeplora",
            "ffn_num": 64,
            "keeplora_rank": 32,
            "keeplora_alpha": 2,
            "keeplora_targets": ["q", "k", "v", "o"],
            "keeplora_weight_threshold": 0.85,
            "keeplora_feature_threshold": 0.99,
            "keeplora_feature_samples": 0,
            "keeplora_grad_batches": 0,
            "keeplora_feature_batches": 0,
            "keeplora_init_mode": "full_rsiat",
        }
        current = SimpleNamespace(args=args, class_order=None)
        current._checkpoint_run_metadata = MethodType(
            BaseLearner._checkpoint_run_metadata, current
        )
        legacy = SimpleNamespace(
            args={
                key: value
                for key, value in args.items()
                if key != "keeplora_init_mode"
            }
        )
        saved_metadata = BaseLearner._checkpoint_run_metadata(legacy)
        saved_metadata.pop("keeplora_init_mode")
        BaseLearner._validate_checkpoint(
            current, {"run_metadata": saved_metadata}
        )
        saved_metadata["keeplora_init_mode"] = "cosine"
        with self.assertRaisesRegex(ValueError, "keeplora_init_mode"):
            BaseLearner._validate_checkpoint(
                current, {"run_metadata": saved_metadata}
            )

    @unittest.skipUnless(HAS_MODEL_IMPORT, "ViT construction dependencies are required")
    def test_model_has_48_attention_keeplora_modules(self):
        config = SimpleNamespace(
            ffn_adapt=False,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=64,
            d_model=768,
            vpt_on=False,
            vpt_num=0,
            keeplora_targets=("q", "k", "v", "o"),
            keeplora_rank=32,
            keeplora_alpha=2,
        )
        model = KeepLoRAVisionTransformer(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            num_classes=0,
            tuning_config=config,
            keeplora_config=config,
        )
        try:
            adapters = [
                module for module in model.modules() if isinstance(module, KeepLoRA)
            ]
            self.assertEqual(len(adapters), 48)
        finally:
            del model
            gc.collect()


class KeepLoRAInitializationInvarianceTests(unittest.TestCase):
    def test_subtracting_delta_preserves_linear_forward(self):
        torch.manual_seed(11)
        adapter = KeepLoRA(in_dim=8, out_dim=8, r=4, lora_alpha=2)
        weight = nn.Parameter(torch.randn(8, 8), requires_grad=False)
        original_weight = weight.detach().clone()
        inputs = torch.randn(3, 5, 8)
        expected = F.linear(inputs, weight)

        adapter.initialize_from_gradient(torch.randn_like(weight))
        delta = adapter.get_delta_weight()
        weight.data.sub_(delta)
        actual = F.linear(inputs, weight) + adapter(inputs)

        self.assertTrue(torch.allclose(actual, expected, rtol=1e-4, atol=1e-5))
        self.assertTrue(
            torch.allclose(weight + delta, original_weight, rtol=1e-4, atol=1e-5)
        )


if __name__ == "__main__":
    unittest.main()
