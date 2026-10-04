"""Task-boundary process RNG state (not live DataLoader worker processes)."""

import logging
import random

import numpy as np
import torch


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if torch.cuda.is_available() and len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
        else:
            logging.warning(
                "CUDA RNG topology changed; CPU/Python/NumPy RNG restored, "
                "but this resume is not strictly paired on CUDA."
            )
