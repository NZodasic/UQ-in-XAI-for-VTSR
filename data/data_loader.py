"""
data/data_loader.py
────────────────────
Dataset validation, statistics reporting, and YAML patching
for the Vietnamese Traffic Sign dataset (YOLO format).

Usage (standalone):
    python data/data_loader.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

# ── allow running from project root ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.logger import print_dataset_description, print_section


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    "one way prohibition", "no parking", "no stopping and parking",
    "no turn left", "no turn right", "no u turn",
    "no u and left turn", "no u and right turn",
    "no motorbike entry/turning", "no car entry/turning",
    "no truck entry/turning", "other prohibition", "indication",
    "direction", "speed limit", "weight limit", "height limit",
    "pedestrian crossing", "intersection danger", "road danger",
    "pedestrian danger", "construction danger", "slow warning",
    "other warning", "vehicle permission lane",
    "vehicle and speed permission lane", "overpass route",
    "no more prohibition", "other",
]
NUM_CLASSES = 29


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_images(split_dir: Path) -> int:
    """Count images in a YOLO split folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not split_dir.exists():
        return 0
    return sum(1 for f in split_dir.iterdir() if f.suffix.lower() in exts)


def _count_annotations(label_dir: Path) -> dict[int, int]:
    """Count per-class annotation instances from .txt label files."""
    counts: dict[int, int] = {}
    if not label_dir.exists():
        return counts
    for txt in label_dir.glob("*.txt"):
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls = int(line.split()[0])
            counts[cls] = counts.get(cls, 0) + 1
    return counts


def get_dataset_root(config_path: Path | None = None) -> Path:
    """Resolve dataset root from config or default path."""
    if config_path is None:
        config_path = ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return ROOT / cfg["dataset"]["root"]


# ─────────────────────────────────────────────────────────────────────────────
# YAML patcher
# ─────────────────────────────────────────────────────────────────────────────

def patch_dataset_yaml(dataset_root: Path) -> Path:
    """
    Ensure custom_data.yaml exists with absolute paths so Ultralytics
    can locate splits regardless of cwd.
    Returns the path to the (possibly patched) yaml file.
    """
    yaml_path = dataset_root / "custom_data.yaml"
    train_img = dataset_root / "images" / "train"
    val_img   = dataset_root / "images" / "val"
    test_img  = dataset_root / "images" / "test"

    data = {
        "path": str(dataset_root),
        "train": str(train_img),
        "val":   str(val_img),
        "test":  str(test_img),
        "nc":    NUM_CLASSES,
        "names": CLASS_NAMES,
    }

    patched_path = dataset_root / "custom_data_abs.yaml"
    with open(patched_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    return patched_path


# ─────────────────────────────────────────────────────────────────────────────
# Main validation + statistics
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_report(dataset_root: Path) -> dict:
    """Validate dataset structure and print statistics. Returns counts dict."""
    img_root   = dataset_root / "images"
    label_root = dataset_root / "labels"

    splits = ["train", "val", "test"]
    counts: dict[str, int] = {}
    for split in splits:
        img_dir   = img_root / split
        label_dir = label_root / split

        n_img = _count_images(img_dir)
        counts[split] = n_img

        if not img_dir.exists():
            print(f"  [WARNING] Missing image dir: {img_dir}")
        if not label_dir.exists():
            print(f"  [WARNING] Missing label dir: {label_dir}")

    total = sum(counts.values())

    print_dataset_description(
        dataset_name="Vietnamese Traffic Signs (VTSR-29)",
        total=total,
        num_classes=NUM_CLASSES,
        train=counts.get("train", 0),
        val=counts.get("val", 0),
        test=counts.get("test", 0),
    )

    # Per-class annotation counts
    print_section("Per-class Instance Counts (train split)")
    label_train = label_root / "train"
    class_counts = _count_annotations(label_train)
    for cls_id in range(NUM_CLASSES):
        n = class_counts.get(cls_id, 0)
        bar = "█" * min(n // 5, 40)
        print(f"  [{cls_id:2d}] {CLASS_NAMES[cls_id][:30]:<30s}  {n:4d}  {bar}")
    print("=" * 60)

    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dataset_root = get_dataset_root()
    print(f"\n  Dataset root: {dataset_root}")
    validate_and_report(dataset_root)
    yaml_path = patch_dataset_yaml(dataset_root)
    print(f"\n  Absolute YAML written → {yaml_path}")
    print("\n  [OK] Dataset validation complete.\n")
