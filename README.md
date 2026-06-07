# UQ-in-XAI for VTSR

**Uncertainty Quantification in Explainable AI for Vietnamese Traffic Sign Recognition**

This repository provides a robust research framework for training, calibrating, and explaining deep learning models applied to the classification of Vietnamese traffic signs. The core contribution is the integration of **Monte Carlo Dropout** for uncertainty quantification with **Advanced Grad-CAM** variants to produce reliable, high-confidence explanations.

---

## Key Features

### 1. Explainability (XAI)
- **Advanced Grad-CAM**: Integrated support for `GradCAM++`, `EigenCAM`, and `HiResCAM` via `pytorch-grad-cam`.
- **UQ-Aware Saliency (Fusion)**: A novel visualization panel that combines mean saliency with epistemic uncertainty to highlight the most reliable decision features.

### 2. Uncertainty Quantification (UQ)
- **MC Dropout**: Bayesian approximation using stochastic forward passes to estimate predictive entropy and variance.
- **Robust Stochastic Control**: Intelligent hybrid training state that freezes BatchNorm layers while keeping Dropout active for consistent sampling.

### 3. Post-hoc Calibration
- **Temperature Scaling**: Learns a single scalar $T$ on the validation set to correct model overconfidence and produce well-calibrated probabilities.
- **Metric Suite**: Formal calculation of ECE (Expected Calibration Error), MCE, NLL, and Brier Score.

### 4. Experiment Management
- **Atomic Run Versioning**: Automatic isolated directory creation for every experiment (`run_001`, `run_002`, etc.).
- **Visual Diagnostics**: Automated plotting of training curves and 5-panel XAI-UQ heatmaps.

---

## Getting Started

### Installation

On Windows, the easiest path is the automated setup script:
```bat
setup_windows.bat -DatasetRoot "D:\path\to\data2-augment"
verify_windows.bat
experiment.bat
```

See `WINDOWS_SETUP.md` for CUDA, CPU-only, smoke-test, and benchmark commands.

1. Clone the repository:
   ```bash
   git clone https://github.com/NZodasic/UQ-in-XAI-for-VTSR.git
   cd UQ-in-XAI-for-VTSR
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Optional tools:
   ```bash
   pip install -r requirements-augmentation.txt  # offline augmentation script
   pip install -r requirements-api.txt           # API/server extras
   ```

### Data Preparation

The framework expects the **Vietnamese Traffic Sign Dataset** in a YOLO-style directory structure:
```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/ (classes as text files)
    ├── val/
    └── test/
```
Update the paths in `configs/config.yaml` to point to your local dataset directories.

### Running Experiments

Execute the full pipeline (training, calibration, UQ, and XAI):
```bash
python main.py --config configs/config.yaml
```

Run the supported comparison benchmark in one command:
```bash
python experiment_tools/run_full_benchmark.py --config configs/config.yaml --suite full
```

On Windows, `benchmark.bat` uses a stable fast comparison budget by default (`--epochs 2 --patience 1 --batch-size 192 --num-workers 0 --uncertainty-samples 5 --xai-mc-samples 3 --ig-steps 12 --skip-xai-metrics`) so the full suite avoids Windows DataLoader shared-memory limits:
```bat
benchmark.bat --suite full
```

For a quick smoke run or a custom benchmark budget:
```bash
python experiment_tools/run_full_benchmark.py --config configs/config.yaml --suite full --epochs 2 --batch-size 64
benchmark.bat --suite full --epochs 5 --patience 2 --batch-size 128
```

The benchmark writes:
- one combined table at `EXPERIMENT/benchmark_results.csv`, updated after each completed run
- named run folders such as `run_006_uq_efficientnet_b2_no_calibration`
- graphs in `EXPERIMENT/graphs/`
- per-run summaries in each `EXPERIMENT/run_*`

Override individual config values from the command line when running comparison experiments:
```bash
python main.py --config configs/config.yaml --set model.name=resnet50
python main.py --config configs/config.yaml --set model.name=mobilenet_v2
python main.py --config configs/config.yaml --set model.name=densenet121
python main.py --config configs/config.yaml --set calibration.temperature_scaling=false --set uncertainty.method=none
python main.py --config configs/config.yaml --set explainability.variant=eigencam
python main.py --config configs/config.yaml --set explainability.variant=hirescam
python main.py --config configs/config.yaml --set explainability.method=integrated_gradients
```

Each run writes `metrics_summary.json` and `metrics_summary.csv` inside its `EXPERIMENT/run_*` directory. After several runs, collect a single comparison table:
```bash
python dataset_tools/collect_experiment_summaries.py --experiment-dir EXPERIMENT --output EXPERIMENT/comparison_table.csv
```

### Dataset Utilities

Offline augmentation and dataset inspection scripts are grouped by purpose:
```bash
python augmentation/unified_augmentation.py --base-dir dataset --output-dir dataset-augment
python dataset_tools/check_distribution.py dataset-augment
python dataset_tools/check_classes.py dataset-augment/labels/train
python dataset_tools/calculate_mean_std.py dataset-augment
```

---

## Configuration

All settings are controlled via `configs/config.yaml`. Key sections include:
- `model`: Choose `resnet50`, `efficientnet_b2`, `mobilenet_v2`, or `densenet121`.
- `uncertainty`: Configure MC Dropout samples and method.
- `explainability`: Select Grad-CAM variants and target layers.

Recommended thesis comparison runs:
- Backbone table: fix UQ/calibration settings, then run `mobilenet_v2`, `resnet50`, `densenet121`, and `efficientnet_b2`.
- UQ table: fix `model.name=efficientnet_b2`, compare `uncertainty.method=none`, `calibration.temperature_scaling=true`, and `uncertainty.method=mc_dropout`.
- XAI table: fix the trained model/backbone, then compare `gradcam`, `gradcam++`, `eigencam`, `hirescam`, and `integrated_gradients` using the generated XAI timing field.

---

## Citation

If you use this framework in your research, please consider citing this project as part of the **Vietnamese Traffic Sign Recognition (VTSR) Research Series**.

---

## License
This project is licensed under the MIT License.
