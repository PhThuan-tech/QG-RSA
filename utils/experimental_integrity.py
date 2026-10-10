"""Task-boundary reproducibility and read-only experimental probes."""

from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [s.clone() for s in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    cuda_states = state["torch_cuda"]
    if cuda_states and len(cuda_states) != torch.cuda.device_count():
        raise ValueError("Exact RNG resume requires the same number of CUDA devices.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([s.cpu() for s in cuda_states])


@contextmanager
def preserve_rng_state():
    state = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(state)


def seed_worker(worker_id):
    # PyTorch seeds its own worker RNG before this callback. Use that seed for
    # the other augmentation RNGs as well; never use process-global hash().
    seed = torch.initial_seed()
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def make_generator(seed, role):
    digest = hashlib.sha256("{}:{}".format(int(seed), role).encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


def dataset_manifest(dataset):
    """IDs plus a content/label digest; no historical images in checkpoints."""
    digest = hashlib.sha256()
    for array in (dataset.sample_ids, dataset.labels):
        digest.update(np.asarray(array, dtype=np.int64).tobytes())
    if dataset.use_path:
        for path in dataset.images:
            value = str(path).replace("\\", "/").encode("utf-8")
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
    else:
        images = np.ascontiguousarray(dataset.images)
        digest.update(str((images.shape, images.dtype.str)).encode())
        digest.update(images.tobytes())
    return {"sample_ids": dataset.sample_ids.tolist(), "sha256": digest.hexdigest()}


def pair_features(before, after):
    """Align (IDs, features, labels) explicitly, rejecting missing/duplicate IDs."""
    paired = []
    for ids, features, labels in (before, after):
        ids = torch.as_tensor(ids, dtype=torch.long).cpu()
        if len(ids) != len(features) or len(ids) != len(labels):
            raise ValueError("Drift feature/ID/label counts differ.")
        if ids.unique().numel() != ids.numel():
            raise ValueError("Drift sample IDs must be unique.")
        order = ids.argsort()
        paired.append((ids[order], features[order], labels[order]))
    if not torch.equal(paired[0][0], paired[1][0]):
        raise ValueError("Before/after drift sample ID sets differ.")
    if not torch.equal(paired[0][2], paired[1][2]):
        raise ValueError("Before/after labels differ for the same drift sample ID.")
    return paired[0][1], paired[1][1], paired[0][0]


def snapshot_moments(means, covariances, old_count):
    return (
        np.asarray(means[:old_count]).copy(),
        covariances[:old_count].detach().cpu().clone(),
    )


def memory_drift_diagnostics(before, means, covariances):
    """Stored-memory updates, not oracle drift of unavailable old images."""
    old_means, old_covs = before
    count = len(old_means)
    new_covs = covariances[:count].detach().cpu().to(old_covs.dtype)
    mean_delta = np.linalg.norm(np.asarray(means[:count]) - old_means, axis=1)
    cov_delta = (new_covs - old_covs).flatten(1).norm(dim=1)
    return {
        "old_class_count": count,
        "old_memory_mean_update_l2": mean_delta.tolist(),
        "old_memory_covariance_update_fro": cov_delta.tolist(),
        "old_memory_covariance_trace_before": old_covs.diagonal(dim1=-2, dim2=-1).sum(-1).tolist(),
        "old_memory_covariance_trace_after": new_covs.diagonal(dim1=-2, dim2=-1).sum(-1).tolist(),
        "old_memory_covariance_policy": "retained_without_compensation",
    }
