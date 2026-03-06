"""
utils/device.py
───────────────
Device detection + global seed setting for reproducibility.
"""
from __future__ import annotations

import os
import random
import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = 0  # Ultralytics prefers 0 over 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def device_info() -> dict:
    """Return a dict with device metadata for logging."""
    device = get_device()
    info: dict = {"device": device}
    if device == 0 or device == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    return info
