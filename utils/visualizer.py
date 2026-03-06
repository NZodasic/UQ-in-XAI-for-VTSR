"""
utils/visualizer.py
───────────────────
Plotting helpers: confusion matrix, ROC, training curves,
calibration diagram, GradCAM overlays, uncertainty histograms.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")          # headless backend for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_plot = cm.astype(float) / row_sums
        fmt = ".2f"
        vmax = 1.0
    else:
        cm_plot = cm
        fmt = "d"
        vmax = cm.max()

    n = len(class_names)
    fig_size = max(12, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=vmax,
        ax=ax,
        linewidths=0.3,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] Confusion matrix → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# ROC Curve (multi-class, one-vs-rest)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: Sequence[str],
    save_path: Path,
    title: str = "ROC Curves (One-vs-Rest)",
) -> None:
    """Plot per-class ROC curves and macro-average."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr_macro, tpr_macro = [], []
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], y_scores[:, i])[0] for i in range(n_classes)
    ]))

    mean_tpr = np.zeros_like(all_fpr)
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab20", n_classes)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=cmap(i), lw=1.2, alpha=0.7,
                label=f"{class_names[i][:20]} (AUC={roc_auc:.2f})")
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, "k--", lw=2.5,
            label=f"Macro-avg (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "r:", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] ROC curve → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Training Curves (overlay two modes)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    metrics_scratch: dict,
    metrics_pretrained: dict,
    save_path: Path,
) -> None:
    """Plot loss and mAP@0.5 curves for both training modes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, ylabel in zip(
        axes,
        ["box_loss", "map50"],
        ["Box Loss", "mAP@0.5"],
    ):
        for data, label, color in [
            (metrics_scratch, "From Scratch", "#E74C3C"),
            (metrics_pretrained, "Pretrained", "#2ECC71"),
        ]:
            if key in data:
                ax.plot(data[key], label=label, color=color, lw=2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Training Curves — Scratch vs Pretrained", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] Training curves → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Calibration Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
    label: str = "Model",
    n_bins: int = 10,
) -> None:
    """Plot reliability diagram for model calibration."""
    fig, ax = plt.subplots(figsize=(7, 6))
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    ax.plot([0, 1], [0, 1], "k:", label="Perfect calibration", lw=1.5)
    ax.plot(mean_predicted_value, fraction_of_positives, "s-",
            label=label, color="#3498DB", lw=2, ms=6)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] Calibration curve → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty Histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_uncertainty_histogram(
    uncertainties: np.ndarray,
    save_path: Path,
    title: str = "MC Dropout — Predictive Entropy",
) -> None:
    """Histogram of per-sample predictive entropy."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(uncertainties, bins=40, color="#9B59B6", edgecolor="white", alpha=0.85)
    ax.axvline(uncertainties.mean(), color="red", lw=2, linestyle="--",
               label=f"Mean={uncertainties.mean():.4f}")
    ax.set_xlabel("Predictive Entropy (bits)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] Uncertainty histogram → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(
    scratch_metrics: dict,
    pretrained_metrics: dict,
    save_path: Path,
) -> None:
    """Bar chart comparing key metrics between Scratch and Pretrained."""
    metric_keys = ["mAP50", "mAP50_95", "Precision", "Recall", "F1"]
    scratch_vals = [scratch_metrics.get(k, 0.0) for k in metric_keys]
    pretrain_vals = [pretrained_metrics.get(k, 0.0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, scratch_vals, width, label="From Scratch",
                   color="#E74C3C", alpha=0.85)
    bars2 = ax.bar(x + width / 2, pretrain_vals, width, label="Pretrained",
                   color="#2ECC71", alpha=0.85)

    for bar in (*bars1, *bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Scratch vs. Pretrained — Performance Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] Model comparison → {save_path}")
