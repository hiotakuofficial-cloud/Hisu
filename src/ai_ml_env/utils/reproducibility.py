import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Ensure reproducibility across popular ML frameworks."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return an available compute device."""

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: Any) -> int:
    """Return the total number of trainable parameters."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad)
