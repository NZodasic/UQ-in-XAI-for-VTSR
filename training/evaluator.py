"""
training/evaluator.py
──────────────────────
Evaluate trained YOLOv12 models (scratch and/or pretrained) on the
test split and generate all required plots and metrics.

Usage:
    python training/evaluator.py --mode pretrained
    python training/evaluator.py --compare          # both modes side-by-side
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.data_loader import get_dataset_root, patch_dataset_yaml, CLASS_NAMES, NUM_CLASSES
from models.model_utils import load_model, get_model_info, print_model_complexity, mc_dropout_predict
from utils.device import get_device, set_seed
from utils.logger import get_logger, print_section
from utils.visualizer import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_curves,
    plot_calibration_curve,
    plot_uncertainty_histogram,
    plot_model_comparison,
)


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def get_latest_run_dir(mode: str) -> Path | None:
    """Read the latest run dir path written by trainer.py."""
    ref = ROOT / "EXPERIMENT" / f"latest_{mode}.txt"
    if ref.exists():
        p = Path(ref.read_text().strip())
        if p.exists():
            return p
    return None


def get_best_weights(mode: str) -> Path | None:
    run_dir = get_latest_run_dir(mode)
    if run_dir is None:
        return None
    w = run_dir / "weights" / "best.pt"
    return w if w.exists() else None


def load_metrics_json(mode: str) -> dict:
    run_dir = get_latest_run_dir(mode)
    if run_dir is None:
        return {}
    p = run_dir / "logs" / "metrics.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_mode(mode: str, save_dir: Path) -> dict:
    """
    Run YOLO validation on the test split for a given mode.
    Generates confusion matrix, ROC, calibration curve, uncertainty histogram.
    Returns a dict with summary metrics.
    """
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    logger = get_logger(f"EVAL-{mode.upper()}")

    weights = get_best_weights(mode)
    if weights is None:
        logger.error(f"No trained weights found for mode={mode}. Run trainer.py first.")
        return {}

    logger.info(f"Evaluating {mode} | weights={weights}")

    # Load model
    from ultralytics import YOLO
    model = YOLO(str(weights))
    info = get_model_info(model)
    print_model_complexity(info, mode=mode)

    # Dataset YAML
    dataset_root = get_dataset_root()
    data_yaml = patch_dataset_yaml(dataset_root)

    # ── YOLO validation ──────────────────────────────────────────────────────
    val_results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=cfg["model"]["input_size"],
        batch=cfg["training"][mode]["batch"],
        device=device,
        verbose=True,
        plots=True,
        save_json=True,
        project=str(save_dir / "yolo_val"),
        name=mode,
        exist_ok=True,
    )
    # ─────────────────────────────────────────────────────────────────────────

    # Extract scalar metrics
    map50    = float(val_results.box.map50)
    map50_95 = float(val_results.box.map)
    prec     = float(val_results.box.mp)
    recall   = float(val_results.box.mr)
    f1       = 2 * prec * recall / max(prec + recall, 1e-7)

    summary = {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "Precision": round(prec, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Params": info["total_params"],
        "SizeMB": info["size_mb"],
        "GFLOPs": info["gflops"],
        "LatencyMS": info["latency_ms"],
    }

    # Print metrics table
    print_section(f"Evaluation Results — {mode.upper()}")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<20s}: {v:.4f}")
        else:
            print(f"  {k:<20s}: {v:,}")
    print("=" * 60)

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    # Ultralytics saves confusion_matrix.csv in yolo_val/{mode}/
    cm_csv = save_dir / "yolo_val" / mode / "confusion_matrix.csv"
    if not cm_csv.exists():
        # Fall back: look for confusion_matrix.png path
        logger.warning("confusion_matrix.csv not found, skipping CM plot.")
    else:
        cm = pd.read_csv(cm_csv, index_col=0).values.astype(int)
        plot_confusion_matrix(
            cm=cm,
            class_names=CLASS_NAMES,
            save_path=save_dir / f"confusion_matrix_{mode}.png",
            title=f"Confusion Matrix — {mode.capitalize()}",
        )

    # ── Collect per-image predictions for ROC / calibration ─────────────────
    # We need class-level probabilities; use model.predict on test images
    test_img_dir = dataset_root / "images" / "test"
    test_label_dir = dataset_root / "labels" / "test"

    image_paths = sorted(test_img_dir.glob("*.jpg")) + sorted(test_img_dir.glob("*.png"))
    if len(image_paths) > 0:
        y_true_list, y_score_list = [], []

        for img_path in image_paths:
            label_path = test_label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue
            # Ground truth: take first annotation class
            lines = label_path.read_text().strip().splitlines()
            if not lines:
                continue
            gt_cls = int(lines[0].split()[0])

            # Predict
            pred_results = model.predict(
                source=str(img_path),
                imgsz=cfg["model"]["input_size"],
                device=device,
                verbose=False,
                conf=0.01,       # low threshold to get all scores
            )
            # Build a per-class score vector (max conf per class)
            scores = np.zeros(NUM_CLASSES)
            for r in pred_results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls.item())
                        conf   = float(box.conf.item())
                        if 0 <= cls_id < NUM_CLASSES:
                            scores[cls_id] = max(scores[cls_id], conf)

            y_true_list.append(gt_cls)
            y_score_list.append(scores)

        if len(y_true_list) >= 2:
            y_true   = np.array(y_true_list)
            y_scores = np.array(y_score_list)

            # ROC curve
            plot_roc_curves(
                y_true=y_true,
                y_scores=y_scores,
                class_names=CLASS_NAMES,
                save_path=save_dir / f"roc_curve_{mode}.png",
                title=f"ROC Curves — {mode.capitalize()}",
            )

            # Calibration (binary: correct vs incorrect)
            pred_classes = y_scores.argmax(axis=1)
            binary_true  = (pred_classes == y_true).astype(int)
            max_probs    = y_scores.max(axis=1)
            plot_calibration_curve(
                y_true=binary_true,
                y_prob=max_probs,
                save_path=save_dir / f"calibration_{mode}.png",
                label=mode.capitalize(),
            )

    # ── MC Dropout Uncertainty ───────────────────────────────────────────────
    n_unc = min(50, len(image_paths))
    if n_unc > 0:
        mc_cfg = cfg["model"]
        _, entropies = mc_dropout_predict(
            yolo_model=model,
            image_paths=[str(p) for p in image_paths[:n_unc]],
            n_passes=mc_cfg["mc_dropout_passes"],
            device=device,
        )
        plot_uncertainty_histogram(
            uncertainties=entropies,
            save_path=save_dir / f"uncertainty_{mode}.png",
            title=f"MC Dropout Uncertainty — {mode.capitalize()}",
        )
        summary["mean_entropy"] = round(float(entropies.mean()), 4)
        print(f"\n  MC Dropout uncertainty (mean entropy): {entropies.mean():.4f} bits")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Compare both modes
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_compare(save_dir: Path) -> None:
    """Evaluate both modes and plot comparison."""
    logger = get_logger("EVAL-COMPARE")

    results: dict[str, dict] = {}
    for mode in ["scratch", "pretrained"]:
        logger.info(f"\n{'─'*50}\nEvaluating mode: {mode}\n{'─'*50}")
        results[mode] = evaluate_mode(mode=mode, save_dir=save_dir)

    # Overlay training curves
    metrics_scratch    = load_metrics_json("scratch")
    metrics_pretrained = load_metrics_json("pretrained")
    if metrics_scratch or metrics_pretrained:
        plot_training_curves(
            metrics_scratch=metrics_scratch,
            metrics_pretrained=metrics_pretrained,
            save_path=save_dir / "training_curves_comparison.png",
        )

    # Comparison bar chart
    if results.get("scratch") and results.get("pretrained"):
        plot_model_comparison(
            scratch_metrics=results["scratch"],
            pretrained_metrics=results["pretrained"],
            save_path=save_dir / "model_comparison.png",
        )

    # Print side-by-side table
    print_section("Final Comparison")
    keys = ["mAP50", "mAP50_95", "Precision", "Recall", "F1"]
    print(f"  {'Metric':<20s} {'Scratch':>12s} {'Pretrained':>12s}")
    print("  " + "-" * 46)
    for k in keys:
        s = results.get("scratch", {}).get(k, "N/A")
        p = results.get("pretrained", {}).get(k, "N/A")
        print(f"  {k:<20s} {str(s):>12s} {str(p):>12s}")
    print("=" * 60)

    # Save JSON summary
    with open(save_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary JSON → {save_dir / 'comparison_results.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Vietnamese TSR models")
    p.add_argument("--mode",    choices=["scratch", "pretrained"], default="pretrained")
    p.add_argument("--compare", action="store_true",
                   help="Evaluate both modes and compare")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config()
    exp_dir = ROOT / cfg["paths"]["experiment_dir"]
    eval_dir = exp_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        evaluate_compare(save_dir=eval_dir)
    else:
        evaluate_mode(mode=args.mode, save_dir=eval_dir)
