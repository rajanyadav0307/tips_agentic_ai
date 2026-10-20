"""Pick a compute device for optional model training.

The statistical analyses run on CPU. Training code may use a GPU when one is
available: CUDA (NVIDIA, Linux/Windows), MPS (Apple silicon), otherwise CPU.
PyTorch is optional (``pip install agent-tips[gpu]``); without it this always
returns ``"cpu"``.
"""

from __future__ import annotations

import importlib.util
from typing import Literal

Device = Literal["cpu", "cuda", "mps"]


def torch_available() -> bool:
    """Return True if PyTorch is installed."""
    return importlib.util.find_spec("torch") is not None


def select_device(prefer: str = "auto") -> Device:
    """Return the device to use.

    Parameters
    ----------
    prefer
        ``"auto"`` picks CUDA, then MPS, then CPU. ``"cpu"``, ``"cuda"`` or
        ``"mps"`` force a device; a forced GPU that is unavailable raises.

    Raises
    ------
    RuntimeError
        If a GPU is requested but not available.
    ValueError
        If ``prefer`` is not a known option.
    """
    if prefer not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unknown device preference: {prefer!r}")
    if prefer == "cpu":
        return "cpu"
    cuda = mps = False
    if torch_available():
        import torch

        cuda = bool(torch.cuda.is_available())
        mps = bool(torch.backends.mps.is_available())
    if prefer == "cuda" and not cuda:
        raise RuntimeError("CUDA requested but not available")
    if prefer == "mps" and not mps:
        raise RuntimeError("MPS requested but not available")
    if cuda:
        return "cuda"
    if mps:
        return "mps"
    return "cpu"
