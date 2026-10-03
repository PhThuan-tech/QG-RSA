"""RSIAT's ViT with KeepLoRA attention updates and no AdaptFormer branch."""

import logging

import torch
import torch.nn as nn
import timm

from KeepLora.peft_modules import KeepLoRA
from network.vision_transformer_adapter import Attention, VisionTransformer


class KeepLoRAAttention(Attention):
    def __init__(self, *args, keeplora_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        targets = tuple(keeplora_config.keeplora_targets)
        common = {
            "r": keeplora_config.keeplora_rank,
            "lora_alpha": keeplora_config.keeplora_alpha,
        }
        self.keeplora = nn.ModuleDict(
            {
                name: KeepLoRA(
                    self.q_proj.in_features,
                    self.q_proj.out_features,
                    **common,
                )
                for name in targets
            }
        )

    def forward(self, x):
        batch_size, sequence_length, channels = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        if "q" in self.keeplora:
            q = q + self.keeplora["q"](x)
        if "k" in self.keeplora:
            k = k + self.keeplora["k"](x)
        if "v" in self.keeplora:
            v = v + self.keeplora["v"](x)

        k = self._shape(k, -1, batch_size).view(batch_size * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, batch_size).view(batch_size * self.num_heads, -1, self.head_dim)
        q = self._shape(q, sequence_length, batch_size).view(
            batch_size * self.num_heads, -1, self.head_dim
        )
        attention = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attention = self.attn_drop(torch.nn.functional.softmax(attention, dim=-1))
        pre_projection = torch.bmm(attention, v).view(
            batch_size, self.num_heads, sequence_length, self.head_dim
        )
        pre_projection = pre_projection.transpose(1, 2).reshape(
            batch_size, sequence_length, channels
        )
        output = self.proj(pre_projection)
        if "o" in self.keeplora:
            output = output + self.keeplora["o"](pre_projection)
        return self.proj_drop(output)

    def named_keeplora_targets(self, prefix):
        projections = {"q": self.q_proj, "k": self.k_proj, "v": self.v_proj, "o": self.proj}
        for name, adapter in self.keeplora.items():
            yield f"{prefix}.{name}", projections[name].weight, adapter


class KeepLoRAVisionTransformer(VisionTransformer):
    def __init__(self, *args, keeplora_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.keeplora_config = keeplora_config
        for block in self.blocks:
            original = block.attn
            replacement = KeepLoRAAttention(
                original.q_proj.in_features,
                num_heads=original.num_heads,
                qkv_bias=original.q_proj.bias is not None,
                keeplora_config=keeplora_config,
            )
            # The original RSIAT attention has no KeepLoRA B/basis tensors.
            replacement.load_state_dict(original.state_dict(), strict=False)
            block.attn = replacement

    def named_keeplora_targets(self):
        for index, block in enumerate(self.blocks):
            yield from block.attn.named_keeplora_targets(f"blocks.{index}.attn")

    def initialize_keeplora_principal_subspaces(self):
        for _, weight, adapter in self.named_keeplora_targets():
            adapter.initialize_principal_subspace(
                weight, self.keeplora_config.keeplora_weight_threshold
            )

    def initialize_keeplora_from_gradients(self, gradients, verify_invariance=False):
        gradient_stats = {}
        for name, weight, adapter in self.named_keeplora_targets():
            gradient_norm, projected_gradient_norm = adapter.initialize_from_gradient(
                gradients.get(name)
            )
            # KeepLoRA starts from a non-zero SVD update; cancel it in W so
            # the first task forward pass is exactly the previous model.
            original_weight = weight.detach().clone() if verify_invariance else None
            adapter.subtract_from(weight)
            stats = {
                "gradient_norm_before_projection": gradient_norm,
                "gradient_norm_after_projection": projected_gradient_norm,
            }
            if verify_invariance:
                effective_weight = weight.detach() + adapter.get_delta_weight().to(
                    weight.device, dtype=weight.dtype
                )
                max_abs_error = (
                    effective_weight - original_weight
                ).abs().max().item()
                if not torch.allclose(
                    effective_weight, original_weight, rtol=1e-4, atol=1e-5
                ):
                    raise RuntimeError(
                        "KeepLoRA subtract initialization is not invariant for {} "
                        "(maximum absolute error {}).".format(name, max_abs_error)
                    )
                stats["weight_merge_max_abs_error"] = max_abs_error
            gradient_stats[name] = stats
        return gradient_stats

    def begin_keeplora_feature_collection(self):
        for _, _, adapter in self.named_keeplora_targets():
            adapter.start_feature_collection(self.keeplora_config.keeplora_feature_samples)

    def merge_keeplora_weights(self):
        for _, weight, adapter in self.named_keeplora_targets():
            adapter.merge_into(weight)
            adapter.reset_parameters()


def _make_keeplora_vit(pretrained_model_name, **kwargs):
    model = KeepLoRAVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs,
    )
    checkpoint_model = timm.create_model(pretrained_model_name, pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    for key in list(state_dict):
        if "qkv.weight" in key:
            qkv_weight = state_dict.pop(key)
            for index, projection in enumerate(("q", "k", "v")):
                state_dict[key.replace("qkv.weight", f"{projection}_proj.weight")] = qkv_weight[
                    index * 768 : (index + 1) * 768
                ]
        elif "qkv.bias" in key:
            qkv_bias = state_dict.pop(key)
            for index, projection in enumerate(("q", "k", "v")):
                state_dict[key.replace("qkv.bias", f"{projection}_proj.bias")] = qkv_bias[
                    index * 768 : (index + 1) * 768
                ]
        elif "mlp.fc" in key:
            state_dict[key.replace("mlp.", "")] = state_dict.pop(key)

    expected_missing_keeplora_keys = {
        name for name in model.state_dict() if ".attn.keeplora." in name
    }
    message = model.load_state_dict(state_dict, strict=False)
    missing_keys = set(message.missing_keys)
    unexpected_keys = set(message.unexpected_keys)
    actual_missing_keeplora_keys = missing_keys & expected_missing_keeplora_keys
    other_missing_keys = missing_keys - expected_missing_keeplora_keys

    logging.info(
        "Pretrained state_dict load: missing_keys_count=%d, unexpected_keys_count=%d",
        len(missing_keys),
        len(unexpected_keys),
    )
    logging.info(
        "Expected missing KeepLoRA keys (%d): %s",
        len(expected_missing_keeplora_keys),
        sorted(expected_missing_keeplora_keys),
    )
    logging.info(
        "Actual missing KeepLoRA keys (%d): %s",
        len(actual_missing_keeplora_keys),
        sorted(actual_missing_keeplora_keys),
    )
    logging.info(
        "Unexpected keys (%d): %s",
        len(unexpected_keys),
        sorted(unexpected_keys),
    )
    logging.info(
        "Other missing keys (%d): %s",
        len(other_missing_keys),
        sorted(other_missing_keys),
    )

    if (
        actual_missing_keeplora_keys != expected_missing_keeplora_keys
        or other_missing_keys
        or unexpected_keys
    ):
        raise RuntimeError(
            "Pretrained ViT checkpoint does not match the model. "
            "Expected missing KeepLoRA keys: {}. Actual missing KeepLoRA keys: {}. "
            "Other missing keys: {}. Unexpected keys: {}."
            .format(
                sorted(expected_missing_keeplora_keys),
                sorted(actual_missing_keeplora_keys),
                sorted(other_missing_keys),
                sorted(unexpected_keys),
            )
        )

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in message.missing_keys and "keeplora" in name)
    model.initialize_keeplora_principal_subspaces()
    return model


def vit_base_patch16_224_keeplora(**kwargs):
    return _make_keeplora_vit("vit_base_patch16_224", **kwargs)


def vit_base_patch16_224_in21k_keeplora(**kwargs):
    return _make_keeplora_vit("vit_base_patch16_224_in21k", **kwargs)
