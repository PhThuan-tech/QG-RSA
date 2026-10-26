import torch


class PrinSubspace:
    """KeepLoRA's cumulative dominant-feature subspace tracker."""

    def __init__(self, prin_subspace_name_list, log_txt):
        self.prin_subspace_dict = {name: [] for name in prin_subspace_name_list}
        self.log_txt = log_txt

    def update_prin_subspace(self, mat_list_dict, threshold_dict):
        if set(mat_list_dict) != set(self.prin_subspace_dict):
            raise ValueError("Feature matrices and principal-subspace keys must match.")
        if set(threshold_dict) != set(self.prin_subspace_dict):
            raise ValueError("Feature thresholds and principal-subspace keys must match.")
        for name in self.prin_subspace_dict:
            threshold = threshold_dict[name]
            if threshold > 1e-8:
                self._update_prin_subspace(
                    mat_list_dict[name], self.prin_subspace_dict[name], threshold, name
                )

    def _update_prin_subspace(self, mat_list, subspaces, threshold, name):
        for index, activation in enumerate(mat_list):
            activation = activation.detach().float()
            if not subspaces:
                u, singular_values, _ = torch.linalg.svd(activation, full_matrices=False)
                if singular_values.square().sum() == 0:
                    subspaces.append(torch.empty(activation.shape[0], 0))
                    continue
                cumulative = torch.cumsum(singular_values.square(), dim=0)
                width = max(
                    int(torch.sum(cumulative < threshold * cumulative[-1]).item()), 1
                )
                subspaces.append(u[:, :width])
            else:
                previous = subspaces[index]
                residual = activation - previous @ (previous.transpose(0, 1) @ activation)
                u, singular_values, _ = torch.linalg.svd(residual, full_matrices=False)
                total_energy = activation.square().sum()
                if total_energy == 0 or singular_values.square().sum() == 0:
                    continue
                retained_energy = 1.0 - residual.square().sum() / total_energy
                width = 0
                for energy in singular_values.square() / total_energy:
                    if retained_energy >= threshold:
                        break
                    retained_energy += energy
                    width += 1
                available = previous.shape[0] - previous.shape[1]
                if available > 0 and width > 0:
                    subspaces[index] = torch.cat((previous, u[:, :min(width, available)]), dim=1)
            self.log_txt(
                "KeepLoRA %s feature basis: %d/%d",
                name,
                subspaces[index].shape[1],
                subspaces[index].shape[0],
            )
