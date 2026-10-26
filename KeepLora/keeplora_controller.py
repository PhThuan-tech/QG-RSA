import torch

from KeepLora.prin_subspace import PrinSubspace


class KeepLoRAController:
    """Bridge KeepLoRA's original task lifecycle to RSIAT's split Q/K/V/O ViT."""

    def __init__(self, convnet, log_txt):
        self.convnet = convnet
        self.targets = list(convnet.named_keeplora_targets())
        self.subspaces = PrinSubspace([name for name, _, _ in self.targets], log_txt)
        for name, _, adapter in self.targets:
            bases = [basis.detach().float().cpu().clone() for basis in (
                adapter.principal_basis,
                adapter.feature_basis,
            ) if basis.numel()]
            if bases:
                self.subspaces.prin_subspace_dict[name] = [
                    bases[0] if len(bases) == 1 else torch.cat(bases, dim=1)
                ]

    def initialize_from_gradients(self, gradients, verify_invariance=False):
        return self.convnet.initialize_keeplora_from_gradients(
            gradients, verify_invariance=verify_invariance
        )

    def begin_feature_collection(self):
        self.convnet.begin_keeplora_feature_collection()

    def finish_task(self):
        matrices = {}
        thresholds = {}
        for name, _, adapter in self.targets:
            matrix = adapter.take_feature_matrix()
            if matrix is None:
                matrix = adapter.feature_basis.new_zeros((adapter.in_dim, 1))
                thresholds[name] = 0.0
            else:
                thresholds[name] = self.convnet.keeplora_config.keeplora_feature_threshold
            matrices[name] = [matrix]
        self.subspaces.update_prin_subspace(matrices, thresholds)
        for name, _, adapter in self.targets:
            if self.subspaces.prin_subspace_dict[name]:
                combined_basis = self.subspaces.prin_subspace_dict[name][0]
                adapter.set_feature_basis(
                    combined_basis[:, adapter.principal_basis.shape[1] :]
                )
        self.convnet.merge_keeplora_weights()
