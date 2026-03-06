"""
training/trainer.py
────────────────────
Train a YOLOv12 model in two modes:
  --mode scratch     → random init, trains from scratch
  --mode pretrained  → COCO backbone, fine-tunes on Vietnamese TSR

Usage:
    python training/trainer.py --mode pretrained
    python training/trainer.py --mode scratch --epochs 50 --batch 8
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.data_loader import get_dataset_root, patch_dataset_yaml, validate_and_report
from models.model_utils import get_model_info, load_model, print_model_complexity
from utils.device import device_info, get_device, set_seed
from utils.logger import (
    get_logger,
    print_experiment_setup,
    print_section,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_run_dir(base: Path, mode: str) -> Path:
    """Create a timestamped run directory for a given mode."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / mode / f"run_{ts}"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    return run_dir


def load_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_yolo_metrics(results_csv: Path) -> dict[str, list[float]]:
    """Parse Ultralytics results.csv into a dict of metric lists."""
    import pandas as pd
    if not results_csv.exists():
        return {}
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    metrics: dict[str, list[float]] = {}
    col_map = {
        "train/box_loss": "box_loss",
        "train/cls_loss": "cls_loss",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map50_95",
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
    }
    for src, dst in col_map.items():
        if src in df.columns:
            metrics[dst] = df[src].dropna().tolist()
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training entry
# ─────────────────────────────────────────────────────────────────────────────

def train(mode: str, epochs: int | None = None, batch: int | None = None) -> Path:
    """
    Train YOLOv12 in the specified mode.

    Returns path to best.pt weights.
    """
    cfg = load_config()
    seed = cfg["seed"]
    set_seed(seed)

    dev = get_device()
    dinfo = device_info()
    train_cfg = cfg["training"][mode]

    # Override from CLI if provided
    _epochs = epochs if epochs is not None else train_cfg["epochs"]
    _batch  = batch  if batch  is not None else train_cfg["batch"]
    _lr0    = train_cfg["lr0"]
    _lrf    = train_cfg["lrf"]
    _opt    = train_cfg["optimizer"]
    _pretrained = train_cfg["pretrained"]

    # Dataset
    dataset_root = get_dataset_root()
    validate_and_report(dataset_root)
    data_yaml = patch_dataset_yaml(dataset_root)

    # Load model
    model = load_model(mode=mode)
    info  = get_model_info(model)

    # Logging
    exp_dir = ROOT / cfg["paths"]["experiment_dir"]
    run_dir = make_run_dir(exp_dir, mode)
    log_file = run_dir / "logs" / "train.log"
    logger = get_logger(f"TSR-{mode.upper()}", log_file=log_file)

    print_experiment_setup(
        mode=mode,
        device=f"{dev}" + (f" ({dinfo.get('gpu_name', '')})" if dev == "cuda" else ""),
        batch=_batch,
        epochs=_epochs,
        lr=_lr0,
        optimizer=_opt,
        model_name=cfg["model"]["name"],
        pretrained=_pretrained,
        num_params=info["total_params"],
        model_size_mb=info["size_mb"],
    )
    print_model_complexity(info, mode=mode)

    logger.info(f"Starting training | mode={mode} | epochs={_epochs} | batch={_batch}")
    t_start = time.perf_counter()

    # ── Ultralytics training ─────────────────────────────────────────────────
    yolo_run_dir = run_dir / "yolo_run"
    results = model.train(
        data=str(data_yaml),
        epochs=_epochs,
        batch=_batch,
        imgsz=cfg["model"]["input_size"],
        lr0=_lr0,
        lrf=_lrf,
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
        optimizer=_opt,
        warmup_epochs=train_cfg["warmup_epochs"],
        workers=train_cfg["workers"],
        patience=train_cfg["patience"],
        device=dev,
        seed=seed,
        project=str(yolo_run_dir),
        name="train",
        save=True,
        plots=True,
        exist_ok=True,
        pretrained=_pretrained,
        verbose=True,
    )
    # ────────────────────────────────────────────────────────────────────────

    elapsed = time.perf_counter() - t_start
    logger.info(f"Training complete in {elapsed/60:.1f} min")

    # Copy best weights to run_dir
    yolo_weights = yolo_run_dir / "train" / "weights" / "best.pt"
    best_dst = run_dir / "weights" / "best.pt"
    if yolo_weights.exists():
        shutil.copy(yolo_weights, best_dst)
        logger.info(f"Best weights saved → {best_dst}")

    # Save metrics JSON
    results_csv = yolo_run_dir / "train" / "results.csv"
    metrics = extract_yolo_metrics(results_csv)
    json_path = run_dir / "logs" / "metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved → {json_path}")

    # Print final epoch summary
    print_section(f"Training Summary — {mode.upper()}")
    if "map50" in metrics and metrics["map50"]:
        print(f"  Best mAP@0.5        : {max(metrics['map50']):.4f}")
    if "map50_95" in metrics and metrics["map50_95"]:
        print(f"  Best mAP@0.5:0.95   : {max(metrics['map50_95']):.4f}")
    if "precision" in metrics and metrics["precision"]:
        print(f"  Best Precision      : {max(metrics['precision']):.4f}")
    if "recall" in metrics and metrics["recall"]:
        print(f"  Best Recall         : {max(metrics['recall']):.4f}")
    print(f"  Training time       : {elapsed/60:.1f} min")
    print(f"  Run directory       : {run_dir}")
    print("=" * 60)

    # Save run_dir path for downstream scripts
    ref_path = exp_dir / f"latest_{mode}.txt"
    ref_path.write_text(str(run_dir))

    return best_dst


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Vietnamese TSR (YOLOv12)")
    p.add_argument("--mode", choices=["scratch", "pretrained"], default="pretrained")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch",  type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(mode=args.mode, epochs=args.epochs, batch=args.batch)
