import unittest
from types import SimpleNamespace

import torch
from torch import nn

from KeepLora.keeplora_controller import KeepLoRAController
from KeepLora.peft_modules import KeepLoRA
from models.base import BaseLearner
from network.vision_transformer_adapter import Adapter, Block, VisionTransformer
from network.vision_transformer_keeplora import KeepLoRAVisionTransformer


def make_config(target="mlp", ffn_adapt=False):
    config = SimpleNamespace(
        ffn_adapt=ffn_adapt,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",
        ffn_num=4,
        d_model=8,
        vpt_on=False,
        vpt_num=0,
        keeplora_rank=2,
        keeplora_alpha=2.0,
        keeplora_targets=("q", "k", "v", "o"),
        keeplora_fc1_rank=3,
        keeplora_fc2_rank=2,
        keeplora_weight_threshold=0.85,
        keeplora_feature_threshold=0.99,
        keeplora_feature_samples=0,
    )
    if target is not None:
        config.keeplora_target = target
    return config


def make_tiny_keeplora(target="mlp"):
    config = make_config(target=target)
    model = KeepLoRAVisionTransformer(
        img_size=32,
        patch_size=16,
        num_classes=0,
        embed_dim=8,
        depth=2,
        num_heads=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        tuning_config=config,
        keeplora_config=config,
    )
    model.eval()
    return model


class KeepLoRAMlpTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)

    def test_mlp_target_wraps_only_fc1_and_fc2_with_correct_input_bases(self):
        model = make_tiny_keeplora()
        targets = list(model.named_keeplora_targets())

        self.assertEqual(len(targets), 4)
        self.assertEqual(
            [name.rsplit(".", 1)[-1] for name, _, _ in targets],
            ["fc1", "fc2", "fc1", "fc2"],
        )
        self.assertEqual(model.keeplora_trainable_parameter_count(), 128)
        for block in model.blocks:
            self.assertFalse(hasattr(block.attn, "keeplora"))
            self.assertEqual(set(block.keeplora_mlp), {"fc1", "fc2"})

        model.initialize_keeplora_principal_subspaces()
        for _, weight, adapter in targets:
            self.assertEqual(adapter.principal_basis.shape[0], weight.shape[1])
            self.assertEqual(adapter.lora_A.shape[0], weight.shape[1])
            self.assertEqual(adapter.lora_B.shape[1], weight.shape[0])

    def test_zero_mlp_update_matches_plain_vit_forward(self):
        config = make_config()
        baseline = VisionTransformer(
            img_size=32,
            patch_size=16,
            num_classes=0,
            embed_dim=8,
            depth=2,
            num_heads=2,
            mlp_ratio=2.0,
            qkv_bias=True,
            tuning_config=config,
        ).eval()
        keeplora = make_tiny_keeplora()
        missing, unexpected = keeplora.load_state_dict(
            baseline.state_dict(), strict=False
        )

        self.assertTrue(missing)
        self.assertTrue(all("keeplora_mlp" in name for name in missing))
        self.assertFalse(unexpected)
        inputs = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected = baseline.forward_features(inputs)
            actual = keeplora.forward_features(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gradient_initialization_and_task_merge_preserve_effective_forward(self):
        model = make_tiny_keeplora()
        inputs = torch.randn(2, 3, 32, 32)
        targets = list(model.named_keeplora_targets())
        model.initialize_keeplora_principal_subspaces()

        with torch.no_grad():
            original_output = model.forward_features(inputs)
        gradients = {
            name: torch.randn_like(weight)
            for name, weight, _ in targets
        }
        model.initialize_keeplora_from_gradients(
            gradients, verify_invariance=True
        )
        with torch.no_grad():
            initialized_output = model.forward_features(inputs)
        torch.testing.assert_close(
            initialized_output, original_output, rtol=1e-4, atol=1e-5
        )

        controller = KeepLoRAController(model, lambda *args: None)
        controller.begin_feature_collection()
        with torch.no_grad():
            model.forward_features(inputs)
        controller.finish_task()
        with torch.no_grad():
            merged_output = model.forward_features(inputs)
        torch.testing.assert_close(
            merged_output, original_output, rtol=1e-4, atol=1e-5
        )
        for _, _, adapter in model.named_keeplora_targets():
            self.assertEqual(torch.count_nonzero(adapter.lora_B).item(), 0)
            self.assertEqual(adapter.feature_basis.shape[0], adapter.in_dim)

    def test_optimizer_updates_only_trainable_lora_b_factors(self):
        model = make_tiny_keeplora()
        targets = list(model.named_keeplora_targets())
        base_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if "keeplora_mlp" not in name
        ]
        base_snapshots = [parameter.detach().clone() for parameter in base_parameters]
        adapters = [adapter for _, _, adapter in targets]
        with torch.no_grad():
            for adapter in adapters:
                adapter.lora_A.normal_()
        optim = torch.optim.SGD(
            [adapter.lora_B for adapter in adapters], lr=0.1
        )

        optim.zero_grad(set_to_none=True)
        model.forward_features(torch.randn(2, 3, 32, 32)).square().mean().backward()
        self.assertTrue(
            any(
                adapter.lora_B.grad is not None
                and torch.count_nonzero(adapter.lora_B.grad).item() > 0
                for adapter in adapters
            )
        )
        factor_a_snapshots = [adapter.lora_A.clone() for adapter in adapters]
        factor_b_snapshots = [adapter.lora_B.detach().clone() for adapter in adapters]
        optim.step()

        self.assertTrue(
            any(
                not torch.equal(adapter.lora_B, before)
                for adapter, before in zip(adapters, factor_b_snapshots)
            )
        )
        for parameter, before in zip(base_parameters, base_snapshots):
            self.assertTrue(torch.equal(parameter, before))
        for adapter, before in zip(adapters, factor_a_snapshots):
            self.assertTrue(torch.equal(adapter.lora_A, before))

    def test_attention_default_and_explicit_target_are_equivalent(self):
        torch.manual_seed(101)
        default = make_tiny_keeplora(target=None)
        torch.manual_seed(101)
        explicit = make_tiny_keeplora(target="attention")
        inputs = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            default_output = default.forward_features(inputs)
            explicit_output = explicit.forward_features(inputs)

        torch.testing.assert_close(default_output, explicit_output, rtol=0, atol=0)
        self.assertEqual(
            len(list(default.named_keeplora_targets())),
            len(list(explicit.named_keeplora_targets())),
        )

    def test_adapter_parallel_forward_remains_the_existing_residual_path(self):
        config = make_config(ffn_adapt=True)
        block = Block(
            dim=8,
            num_heads=2,
            mlp_ratio=2.0,
            qkv_bias=True,
            config=config,
            layer_id=0,
        ).eval()
        with torch.no_grad():
            block.adaptmlp.up_proj.weight.normal_()
            block.adaptmlp.up_proj.bias.normal_()
        inputs = torch.randn(2, 5, 8)

        with torch.no_grad():
            after_attention = inputs + block.drop_path(
                block.attn(block.norm1(inputs))
            )
            adapt_update = block.adaptmlp(
                after_attention, add_residual=False
            )
            mlp_output = block.mlp_drop(
                block.act(block.fc1(block.norm2(after_attention)))
            )
            mlp_output = block.drop_path(
                block.mlp_drop(block.fc2(mlp_output))
            )
            expected = after_attention + (mlp_output + adapt_update)
            actual = block(inputs)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_parameter_budget_matches_adapter_within_one_percent(self):
        adapter_config = SimpleNamespace(d_model=768)
        adapter_modules = [
            Adapter(
                adapter_config,
                bottleneck=64,
                init_option="lora",
                adapter_scalar="0.1",
                adapter_layernorm_option="none",
            )
            for _ in range(12)
        ]
        keeplora_modules = [
            (
                KeepLoRA(768, 3072, r=25, lora_alpha=2),
                KeepLoRA(3072, 768, r=29, lora_alpha=2),
            )
            for _ in range(12)
        ]
        adapter_count = sum(
            parameter.numel()
            for module in adapter_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        keeplora_count = sum(
            parameter.numel()
            for pair in keeplora_modules
            for module in pair
            for parameter in module.parameters()
            if parameter.requires_grad
        )

        self.assertEqual(adapter_count, 1_189_632)
        self.assertEqual(keeplora_count, 1_188_864)
        self.assertLess(
            abs(keeplora_count - adapter_count) / adapter_count,
            0.01,
        )

    def test_legacy_attention_checkpoint_remains_compatible_but_target_mismatch_fails(self):
        attention_args = {
            "convnet_type": "pretrained_vit_b16_224_keeplora",
            "model_name": "keeplora",
            "keeplora_rank": 32,
            "keeplora_alpha": 2,
            "keeplora_targets": ["q", "k", "v", "o"],
        }
        attention_learner = BaseLearner.__new__(BaseLearner)
        attention_learner.args = attention_args
        legacy_metadata = attention_learner._checkpoint_run_metadata()
        for key in ("keeplora_target", "keeplora_fc1_rank", "keeplora_fc2_rank"):
            legacy_metadata.pop(key)
        attention_learner._validate_checkpoint(
            {"run_metadata": legacy_metadata}
        )

        mlp_learner = BaseLearner.__new__(BaseLearner)
        mlp_learner.args = {
            **attention_args,
            "keeplora_target": "mlp",
            "keeplora_fc1_rank": 25,
            "keeplora_fc2_rank": 29,
            "keeplora_targets": [],
        }
        with self.assertRaisesRegex(ValueError, "keeplora_target"):
            mlp_learner._validate_checkpoint(
                {"run_metadata": legacy_metadata}
            )


if __name__ == "__main__":
    unittest.main()
