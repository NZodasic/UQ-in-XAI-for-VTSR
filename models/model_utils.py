"""
models/model_utils.py
──────────────────────
YOLOv12 model wrapper with:
  - model info (params, FLOPs, size, latency)
  - MC Dropout uncertainty wrapper
  - GradCAM hook for XAI
  - Temperature scaling for calibration

Usage (standalone):
    python models/model_utils.py
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from utils.device import get_device, set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(
    mode: str = "pretrained",   # "scratch" | "pretrained"
    weights_path: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> YOLO:
    """
    Load a YOLOv12 model.

    mode="scratch"    → architecture only (random weights, pretrained=False)
    mode="pretrained" → COCO-pretrained weights (transfer learning)
    weights_path      → load custom trained weights (overrides mode)
    """
    if config_path is None:
        config_path = ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]          # e.g. "yolo12n"
    pretrained_w = cfg["model"]["pretrained_weights"]  # e.g. "yolo12n.pt"

    if weights_path is not None:
        # Load fully trained custom weights
        model = YOLO(weights_path)
    elif mode == "pretrained":
        # Transfer learning from COCO
        model = YOLO(pretrained_w)
    else:
        # From scratch: load architecture yaml only
        model = YOLO(f"{model_name}.yaml")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model information
# ─────────────────────────────────────────────────────────────────────────────

def get_model_info(model: YOLO, img_size: int = 640) -> dict:
    """Return parameter count, approx FLOPs, model size, and latency."""
    device = get_device()
    torch_model: nn.Module = model.model

    # Parameters
    num_params = sum(p.numel() for p in torch_model.parameters())
    num_trainable = sum(p.numel() for p in torch_model.parameters() if p.requires_grad)

    # Model size on disk (if saved) or in memory
    tmp_path = Path("/tmp/tmp_model_size.pt")
    torch.save(torch_model.state_dict(), tmp_path)
    size_mb = tmp_path.stat().st_size / 1e6
    tmp_path.unlink(missing_ok=True)

    # FLOPs via thop
    try:
        from thop import profile
        dummy = torch.zeros(1, 3, img_size, img_size).to(device)
        torch_model.to(device).eval()
        flops, _ = profile(torch_model, inputs=(dummy,), verbose=False)
        gflops = flops / 1e9
    except Exception:
        gflops = -1.0

    # Latency
    torch_model.to(device).eval()
    dummy = torch.zeros(1, 3, img_size, img_size).to(device)
    with torch.no_grad():
        # warm-up
        for _ in range(3):
            torch_model(dummy)
        t0 = time.perf_counter()
        for _ in range(20):
            torch_model(dummy)
        latency_ms = (time.perf_counter() - t0) / 20 * 1000

    info = {
        "total_params": num_params,
        "trainable_params": num_trainable,
        "size_mb": round(size_mb, 2),
        "gflops": round(gflops, 3),
        "latency_ms": round(latency_ms, 2),
    }
    return info


def print_model_complexity(info: dict, mode: str = "") -> None:
    """Print model complexity metrics."""
    label = f" ({mode})" if mode else ""
    print(f"\n  Model Complexity{label}")
    print(f"  ─────────────────────────────────────")
    print(f"  Parameters (total)   : {info['total_params']:,}")
    print(f"  Parameters (trainable): {info['trainable_params']:,}")
    print(f"  Model size           : {info['size_mb']:.2f} MB")
    print(f"  GFLOPs               : {info['gflops']:.3f}")
    print(f"  Inference latency    : {info['latency_ms']:.2f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# MC Dropout — epistemic uncertainty
# ─────────────────────────────────────────────────────────────────────────────

def enable_mc_dropout(model: nn.Module) -> None:
    """Enable dropout layers at inference time for MC sampling."""
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()


def mc_dropout_predict(
    yolo_model: YOLO,
    image_paths: list[str],
    n_passes: int = 20,
    conf_thresh: float = 0.25,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run MC Dropout inference over a list of image paths.

    Returns:
        mean_confs   : (N,) mean max-confidence per image
        entropies    : (N,) predictive entropy per image
    """
    torch_model = yolo_model.model.to(device)
    enable_mc_dropout(torch_model)

    from ultralytics.data.augment import LetterBox
    import cv2
    import torchvision.transforms.functional as TF
    from PIL import Image

    all_confs = []  # shape: (N, n_passes)

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        # minimal pre-process: resize + to tensor
        img_resized = img.resize((640, 640))
        tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

        pass_confs = []
        with torch.no_grad():
            for _ in range(n_passes):
                preds = torch_model(tensor)
                # preds is a list; take first element's confidence column
                if hasattr(preds, "boxes") and preds.boxes is not None:
                    confs = preds.boxes.conf.cpu().numpy()
                    max_conf = float(confs.max()) if len(confs) > 0 else 0.0
                else:
                    # raw tensor output from backbone → use sigmoid max
                    raw = preds[0] if isinstance(preds, (list, tuple)) else preds
                    max_conf = float(raw.sigmoid().max().cpu())
                pass_confs.append(max_conf)

        all_confs.append(pass_confs)

    all_confs_np = np.array(all_confs)  # (N, n_passes)
    mean_confs = all_confs_np.mean(axis=1)

    # Predictive entropy: H = -p*log(p) - (1-p)*log(1-p)
    p = mean_confs.clip(1e-7, 1 - 1e-7)
    entropies = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    return mean_confs, entropies


# ─────────────────────────────────────────────────────────────────────────────
# Temperature Scaling (calibration)
# ─────────────────────────────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Learns a single temperature parameter T on a held-out calibration set.
    Apply:  calibrated_conf = conf / T
    """
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> None:
        """Optimise temperature using NLL loss on calibration data."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def _eval():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(_eval)
        print(f"  Temperature learned: T = {self.temperature.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(42)
    device = get_device()
    print(f"\n  Device: {device}")

    for mode in ["scratch", "pretrained"]:
        print(f"\n  Loading model  ({mode}) …")
        try:
            model = load_model(mode=mode)
            info = get_model_info(model)
            print_model_complexity(info, mode=mode)
        except Exception as e:
            print(f"  [ERROR] {e}")
