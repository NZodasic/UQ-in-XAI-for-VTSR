"""
pruning/baseline.py
────────────────────
Magnitude-based L1 unstructured pruning on a trained YOLOv12 model.
Prunes Conv2d and Linear layers to the configured sparsity, then
evaluates and reports remaining parameters.

Usage:
    python pruning/baseline.py --mode pretrained --sparsity 0.3
    python pruning/baseline.py --mode scratch   --sparsity 0.5
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.model_utils import get_model_info, print_model_complexity
from utils.device import get_device, set_seed
from utils.logger import get_logger, print_section


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def get_best_weights(mode: str) -> Path | None:
    ref = ROOT / "EXPERIMENT" / f"latest_{mode}.txt"
    if ref.exists():
        p = Path(ref.read_text().strip()) / "weights" / "best.pt"
        return p if p.exists() else None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Pruning
# ─────────────────────────────────────────────────────────────────────────────

def apply_pruning(model: nn.Module, sparsity: float) -> nn.Module:
    """Apply L1 unstructured pruning to all Conv2d and Linear layers."""
    pruned = 0
    total  = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=sparsity)
            prune.remove(module, "weight")     # make permanent
            pruned += int((module.weight == 0).sum())
            total  += module.weight.numel()
    actual_sparsity = pruned / max(total, 1)
    print(f"  Pruned weights   : {pruned:,} / {total:,} = {actual_sparsity*100:.1f}%")
    return model


def count_nonzero_params(model: nn.Module) -> tuple[int, int]:
    """Return (nonzero_params, total_params)."""
    total = nz = 0
    for p in model.parameters():
        total += p.numel()
        nz    += int((p != 0).sum())
    return nz, total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_pruning(mode: str, sparsity: float | None = None) -> None:
    cfg    = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    logger = get_logger(f"PRUNE-{mode.upper()}")

    _sparsity = sparsity if sparsity is not None else cfg["pruning"]["sparsity"]
    fine_tune = cfg["pruning"]["fine_tune_epochs"]

    weights_path = get_best_weights(mode)
    if weights_path is None:
        logger.error(f"No trained weights for mode={mode}. Run trainer.py first.")
        return

    from ultralytics import YOLO
    yolo_model = YOLO(str(weights_path))
    torch_model: nn.Module = yolo_model.model.to(device)

    # ── Baseline complexity ──────────────────────────────────────────────────
    print_section(f"Pruning Baseline — {mode.upper()}")
    info_before = get_model_info(yolo_model)
    print("  [Before Pruning]")
    print_model_complexity(info_before, mode=f"{mode} (original)")

    nz, total = count_nonzero_params(torch_model)
    print(f"  Non-zero params  : {nz:,} / {total:,}")

    # ── Apply pruning ────────────────────────────────────────────────────────
    print(f"\n  Applying L1 unstructured pruning  (sparsity={_sparsity*100:.0f}%) …")
    pruned_model = apply_pruning(copy.deepcopy(torch_model), sparsity=_sparsity)

    nz_after, total_after = count_nonzero_params(pruned_model)
    remaining_pct = nz_after / max(total_after, 1) * 100

    print(f"  Non-zero params  : {nz_after:,} / {total_after:,}  ({remaining_pct:.1f}% remaining)")

    # ── Save pruned model ────────────────────────────────────────────────────
    run_dir = Path(
        (ROOT / "EXPERIMENT" / f"latest_{mode}.txt").read_text().strip()
    )
    pruned_save = run_dir / "weights" / f"pruned_{int(_sparsity*100)}pct.pt"
    yolo_model.model = pruned_model
    torch.save(pruned_model.state_dict(), pruned_save)
    logger.info(f"Pruned weights saved → {pruned_save}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print_section("Pruning Summary")
    print(f"  Mode             : {mode}")
    print(f"  Target sparsity  : {_sparsity*100:.0f}%")
    print(f"  Params (before)  : {total:,}")
    print(f"  Params (after)   : {nz_after:,}  ({remaining_pct:.1f}% non-zero)")
    print(f"  Size before      : {info_before['size_mb']:.2f} MB")
    print(f"  Saved pruned to  : {pruned_save}")
    print("=" * 60)
    print(
        "\n  [TIP] Fine-tune the pruned model by running:\n"
        f"  python training/trainer.py --mode {mode} --epochs {fine_tune}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Magnitude pruning for Vietnamese TSR")
    p.add_argument("--mode",     choices=["scratch", "pretrained"], default="pretrained")
    p.add_argument("--sparsity", type=float, default=None,
                   help="Fraction of weights to zero (default from config)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pruning(mode=args.mode, sparsity=args.sparsity)
