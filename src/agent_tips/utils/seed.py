"""Make runs reproducible."""

from __future__ import annotations

import os
import random

import numpy as np

from agent_tips.utils.device import torch_available


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and (if installed) PyTorch; return a NumPy generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch_available():
        import torch

        torch.manual_seed(seed)
    return np.random.default_rng(seed)
