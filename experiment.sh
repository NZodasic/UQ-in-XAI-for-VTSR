#!/usr/bin/env bash
# ============================================================
# experiment.sh — Full Vietnamese TSR Pipeline
# Runs: data check → scratch training → pretrained training
#       → evaluation comparison → pruning
# ============================================================

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

EPOCHS=${1:-100}
BATCH=${2:-16}

echo "========================================"
echo "  Vietnamese TSR — Full Pipeline"
echo "  Epochs : $EPOCHS | Batch : $BATCH"
echo "========================================"

# ── Step 1: Validate dataset ────────────────────────────────
echo ""
echo "[1/5] Validating dataset …"
python data/data_loader.py

# ── Step 2: Train from scratch ──────────────────────────────
echo ""
echo "[2/5] Training from SCRATCH …"
python training/trainer.py --mode scratch --epochs "$EPOCHS" --batch "$BATCH"

# ── Step 3: Train pretrained (fine-tune) ────────────────────
echo ""
echo "[3/5] Training PRETRAINED (fine-tune) …"
python training/trainer.py --mode pretrained --epochs "$EPOCHS" --batch "$BATCH"

# ── Step 4: Evaluate both + compare ─────────────────────────
echo ""
echo "[4/5] Evaluating & comparing both modes …"
python training/evaluator.py --compare

# ── Step 5: Prune pretrained model ──────────────────────────
echo ""
echo "[5/5] Pruning pretrained model (sparsity=0.30) …"
python pruning/baseline.py --mode pretrained --sparsity 0.3

echo ""
echo "========================================"
echo "  Pipeline complete!"
echo "  Results → EXPERIMENT/"
echo "========================================"
