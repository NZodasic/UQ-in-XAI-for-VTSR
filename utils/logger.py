"""
utils/logger.py
───────────────
Structured console + file logger for the TSR pipeline.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str = "TSR", log_file: Path | None = None) -> logging.Logger:
    """Return a configured logger that writes to console and optionally a file."""
    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on repeated calls
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section divider."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_dataset_description(
    dataset_name: str,
    total: int,
    num_classes: int,
    train: int,
    val: int,
    test: int,
) -> None:
    """Print standardised Dataset Description block."""
    print_section("Dataset Description")
    print(f"  Dataset       : {dataset_name}")
    print(f"  Total samples : {total}")
    print(f"  Classes       : {num_classes}")
    print(f"  Training      : {train}")
    print(f"  Validation    : {val}")
    print(f"  Testing       : {test}")
    print("=" * 60)


def print_experiment_setup(
    mode: str,
    device: str,
    batch: int,
    epochs: int,
    lr: float,
    optimizer: str,
    model_name: str,
    pretrained: bool,
    num_params: int | None = None,
    model_size_mb: float | None = None,
) -> None:
    """Print standardised Experimental Setup block."""
    print_section(f"Experimental Setup — {mode.upper()}")
    print(f"  Mode         : {'Pretrained (Transfer Learning)' if pretrained else 'From Scratch'}")
    print(f"  Device       : {device}")
    print(f"  Batch size   : {batch}")
    print(f"  Epochs       : {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Optimizer    : {optimizer}")
    print(f"  Model        : {model_name}")
    if num_params is not None:
        print(f"  Parameters   : {num_params:,}")
    if model_size_mb is not None:
        print(f"  Model size   : {model_size_mb:.2f} MB")
    print("=" * 60)
