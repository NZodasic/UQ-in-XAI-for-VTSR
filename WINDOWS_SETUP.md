# Windows Setup

Run these commands in CMD from the repository folder.

## Full setup

```bat
setup_windows.bat -DatasetRoot "D:\path\to\data2-augment"
```

The dataset folder should contain:

```text
images\train
images\val
images\test
labels\train
labels\val
labels\test
custom_data.yaml
```

The script installs/checks:

- Python 3.11
- Microsoft Visual C++ runtime
- Git, if missing
- NVIDIA CUDA Toolkit 12.8, if an NVIDIA GPU is detected
- `venv`
- PyTorch CUDA wheels
- all repository requirements

It also updates `configs\config.yaml` to use your dataset path and keeps the original first-run config at `configs\config.yaml.bak`.

PyTorch includes the CUDA runtime it needs, but the machine still needs a working NVIDIA driver. If setup finishes with `CUDA available: False`, reboot first, then update/install the NVIDIA display driver and run setup again.

## Train

```bat
verify_windows.bat
experiment.bat
```

## Benchmark

```bat
benchmark.bat --suite full
```

Benchmark results are combined into `EXPERIMENT\benchmark_results.csv`, and each run folder includes its group/model/method name.

Run folders are organized by model:

```text
EXPERIMENT\
  benchmark_results.csv
  graphs\
  model\
    resnet50\
      run_001_backbone_resnet50_resnet50\
      run_002_xai_resnet50_gradcam\
    efficientnet_b2\
      run_001_backbone_efficientnet_b2_efficientnet_b2\
      run_002_uq_efficientnet_b2_no_calibration\
```

`--suite full` runs the backbone table, the UQ table on EfficientNet-B2, and XAI runs for all supported models:

```text
resnet50: gradcam, gradcam++, eigencam, hirescam, saliency, integrated_gradients
efficientnet_b2: gradcam, gradcam++, eigencam, hirescam, saliency, integrated_gradients
mobilenet_v2: gradcam, gradcam++, eigencam, hirescam, saliency, integrated_gradients
densenet121: gradcam, gradcam++, eigencam, hirescam, saliency, integrated_gradients
```

`benchmark.bat` is tuned for stable faster comparison runs by default:

```text
--epochs 2 --patience 1 --batch-size 192 --num-workers 0 --uncertainty-samples 5 --xai-mc-samples 3 --ig-steps 12 --skip-xai-metrics
```

Override those values when you want a longer run:

```bat
benchmark.bat --suite full --epochs 8 --patience 3 --batch-size 128
```

To run the XAI comparison on only one model:

```bat
benchmark.bat --suite xai --xai-model efficientnet_b2
```

## Useful options

```bat
setup_windows.bat -DatasetRoot "D:\path\to\data2-augment" -RunSmokeTest
setup_windows.bat -TorchIndex cpu -SkipCudaToolkit
setup_windows.bat -ForceRecreateVenv
```
