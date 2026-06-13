import os
import sys
import argparse
import csv
import json
import re

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger


def _safe_run_name(value):
    value = str(value or "").strip().lower()
    value = value.replace("+", "plus")
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("._-")


def get_next_run_dir(base_dir="EXPERIMENT", suffix=None):
    os.makedirs(base_dir, exist_ok=True)
    suffix = _safe_run_name(suffix)
    while True:
        existing_runs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
        run_nums = []
        for run_name in existing_runs:
            try:
                run_nums.append(int(run_name.split('_')[1]))
            except (IndexError, ValueError):
                continue

        next_num = max(run_nums, default=0) + 1
        run_name = f'run_{next_num:03d}'
        if suffix:
            run_name = f'{run_name}_{suffix}'
        run_dir = os.path.join(base_dir, run_name)
        try:
            os.mkdir(run_dir)
            return run_dir
        except FileExistsError:
            continue


def load_classes(classes_file, num_classes, logger=None):
    """Load class names from yaml or txt, fallback to generic names."""
    if not os.path.exists(classes_file):
        return {i: f"Class {i}" for i in range(num_classes)}

    # Try YAML (custom_data.yaml format)
    if classes_file.endswith('.yaml') or classes_file.endswith('.yml'):
        import yaml

        with open(classes_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict) or 'names' not in data:
            raise KeyError(f"'names' key missing in {classes_file}")
        if isinstance(data['names'], dict):
            return {int(k): v for k, v in data['names'].items()}
        if isinstance(data['names'], list):
            return {i: name for i, name in enumerate(data['names'])}
        raise TypeError(f"'names' in {classes_file} must be a dict or list")

    # Fallback: plain text list
    with open(classes_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if not lines:
        if logger:
            logger.warning(f"{classes_file} is empty or unreadable. Using generic labels.")
        return {i: f"Class {i}" for i in range(num_classes)}
        
    return {i: name for i, name in enumerate(lines)}


def build_model(config):
    """Model factory: returns the right model class based on config."""
    model_name = config['model'].get('name', 'resnet50').lower()
    num_classes = config['data']['num_classes']
    pretrained = config['model']['pretrained']
    dropout_rate = config['model']['dropout_rate']
    
    if model_name == 'efficientnet_b2':
        from models.efficientnet_classifier import EfficientNetB2Classifier
        return EfficientNetB2Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    if model_name == 'resnet50':
        from models.resnet_classifier import ResNet50Classifier
        return ResNet50Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    if model_name == 'mobilenet_v2':
        from models.mobilenet_classifier import MobileNetV2Classifier
        return MobileNetV2Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )
    if model_name == 'densenet121':
        from models.densenet_classifier import DenseNet121Classifier
        return DenseNet121Classifier(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_rate=dropout_rate
        )

    valid_models = {'efficientnet_b2', 'resnet50', 'mobilenet_v2', 'densenet121'}
    raise ValueError(f"Unknown model '{model_name}'. Choose from {valid_models}")


def apply_config_overrides(config, overrides):
    """Apply CLI overrides like training.epochs=5 using YAML value parsing."""
    import yaml

    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override '{override}'. Use key.path=value.")

        key_path, raw_value = override.split('=', 1)
        keys = [key for key in key_path.split('.') if key]
        if not keys:
            raise ValueError(f"Invalid override key in '{override}'.")

        value = yaml.safe_load(raw_value)
        cursor = config
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value


def _json_safe(value):
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def save_run_summary(
    save_dir,
    config,
    complexity,
    train_losses,
    val_losses,
    val_accuracies,
    results,
    temperature=None
):
    classification = results.get('classification', {})
    calibration = results.get('calibration', {})
    uncertainty = results.get('uncertainty', {})
    xai = results.get('xai', {})

    summary = {
        'run_dir': save_dir,
        'model': {
            'name': config['model']['name'],
            'num_params': complexity['num_params'],
            'trainable_params': complexity['trainable_params'],
            'size_mb': complexity['size_mb']
        },
        'training': {
            'epochs_requested': config['training']['epochs'],
            'epochs_completed': len(train_losses),
            'best_val_accuracy': max(val_accuracies, default=0.0),
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'label_smoothing': config['training'].get('label_smoothing', 0.0)
        },
        'classification': classification,
        'calibration': {
            'temperature_scaling': config.get('calibration', {}).get('temperature_scaling', False),
            'temperature': temperature,
            'metrics': calibration
        },
        'uncertainty': {
            'method': config.get('uncertainty', {}).get('method', 'none'),
            'num_samples': config.get('uncertainty', {}).get('num_samples', 1),
            'metrics': uncertainty
        },
        'explainability': {
            'method': config.get('explainability', {}).get('method', 'none'),
            'variant': config.get('explainability', {}).get('variant', 'none'),
            'metrics': xai
        },
        'latency_ms': results.get('latency_ms', 0.0)
    }

    json_path = os.path.join(save_dir, 'metrics_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(summary), f, indent=2)

    row = {
        'run_dir': save_dir,
        'model_name': config['model']['name'],
        'num_params': complexity['num_params'],
        'num_params_m': complexity['num_params'] / 1_000_000,
        'trainable_params': complexity['trainable_params'],
        'size_mb': complexity['size_mb'],
        'epochs_completed': len(train_losses),
        'best_val_accuracy': max(val_accuracies, default=0.0),
        'label_smoothing': config['training'].get('label_smoothing', 0.0),
        'uq_method': config.get('uncertainty', {}).get('method', 'none'),
        'uq_samples': config.get('uncertainty', {}).get('num_samples', 1),
        'temperature_scaling': config.get('calibration', {}).get('temperature_scaling', False),
        'temperature': temperature,
        'xai_method': config.get('explainability', {}).get('method', 'none'),
        'xai_variant': config.get('explainability', {}).get('variant', 'none'),
        'accuracy': classification.get('accuracy'),
        'precision_macro': classification.get('precision'),
        'recall_macro': classification.get('recall'),
        'f1_macro': classification.get('f1'),
        'auc_ovr': classification.get('auc'),
        'ece': calibration.get('ece'),
        'mce': calibration.get('mce'),
        'nll': calibration.get('nll'),
        'brier': calibration.get('brier'),
        'latency_ms': results.get('latency_ms', 0.0),
        'predictive_entropy_mean': uncertainty.get('predictive_entropy_mean'),
        'predictive_entropy_std': uncertainty.get('predictive_entropy_std'),
        'probability_variance_mean': uncertainty.get('probability_variance_mean'),
        'probability_variance_std': uncertainty.get('probability_variance_std'),
        'xai_time_ms': xai.get('avg_time_ms'),
        'xai_fidelity': xai.get('fidelity'),
        'xai_stability': xai.get('stability')
    }

    csv_path = os.path.join(save_dir, 'metrics_summary.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(_json_safe(row))

    return {'json': json_path, 'csv': csv_path}


def select_device(config_device, logger):
    import torch

    if config_device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Config requested device '{config_device}', but CUDA is not available. "
                "Use --set experiment.device=cpu for a CPU test run."
            )
        try:
            device = torch.device(config_device)
            torch.cuda.get_device_properties(device)
            return device
        except (AssertionError, RuntimeError) as exc:
            raise ValueError(f"Invalid CUDA device '{config_device}'") from exc

    if config_device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        logger.warning("MPS unavailable; falling back to CPU.")
        return torch.device('cpu')

    return torch.device(config_device)


def main():
    parser = argparse.ArgumentParser(description="TSR Research Framework")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument(
        '--set',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Override config values, e.g. --set model.name=resnet50 --set calibration.temperature_scaling=false'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to a resumable checkpoint, e.g. EXPERIMENT/run_001/models/checkpoints/latest.pth'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default=None,
        help='Path to model weights for evaluation/XAI-only runs, e.g. EXPERIMENT/run_001/models/best_model.pth'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Skip training and evaluate/generate XAI from --weights.'
    )
    parser.add_argument(
        '--run-suffix',
        type=str,
        default=None,
        help='Optional suffix for the experiment folder name.'
    )
    args = parser.parse_args()

    import torch
    import torch.optim as optim
    from utils.seed import seed_everything
    from utils.visualization import plot_training_curves
    from data.data_loader import get_dataloaders
    from training.trainer import Trainer
    from training.evaluator import Evaluator
    from metrics.calibration_metrics import count_model_parameters
    import yaml

    with open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    apply_config_overrides(config, args.set)
    if args.resume:
        config.setdefault('training', {}).setdefault('checkpoint', {})['resume_from'] = args.resume
    if args.eval_only and not args.weights:
        raise ValueError("--eval-only requires --weights.")

    # Setup run directory
    save_dir = get_next_run_dir(suffix=args.run_suffix)
    if config.get('experiment', {}).get('save_config', False):
        with open(os.path.join(save_dir, 'config.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, sort_keys=False)

    # 1. Reproducibility
    experiment_cfg = config.get('experiment', {})
    training_cfg = config.get('training', {})
    seed_everything(
        experiment_cfg.get('seed', 42),
        deterministic=experiment_cfg.get('deterministic', True),
        cudnn_benchmark=experiment_cfg.get('cudnn_benchmark', False)
    )
    if training_cfg.get('allow_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Logger
    logger = setup_logger(save_dir)
    logger.info(f"Initialized experiment in {save_dir}")

    # Device
    device = select_device(config['experiment']['device'], logger)
    logger.info(f"Using device: {device}")

    # 2. Dataloaders
    logger.info("Preparing dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config, logger=logger)
    logger.info("Dataloaders ready.")
    num_classes = config['data']['num_classes']
    logger.info(f"Loading class names from {config['data']['classes_file']}")
    classes = load_classes(config['data']['classes_file'], num_classes, logger=logger)

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    if train_size == 0 or val_size == 0 or test_size == 0:
        logger.error(
            "One or more datasets are empty. Please check paths in config.yaml. "
            f"train={train_size}, val={val_size}, test={test_size}"
        )
        sys.exit(1)

    logger.info("\nDataset Summary")
    logger.info("Dataset: Vietnamese Traffic Signs")
    logger.info(f"Total samples : {train_size + val_size + test_size}")
    logger.info(f"Classes       : {num_classes}")
    logger.info(f"Training samples   : {train_size}")
    logger.info(f"Validation samples : {val_size}")
    logger.info(f"Testing samples    : {test_size}\n")

    # 3. Model
    logger.info(f"Building model: {config['model']['name']}")
    model = build_model(config).to(device)
    logger.info("Model ready.")

    complexity = count_model_parameters(model)
    logger.info(f"Model: {config['model']['name'].upper()}")
    logger.info(f"Model Parameters: {complexity['num_params']:,}")
    logger.info(f"Trainable Parameters: {complexity['trainable_params']:,}")
    logger.info(f"Model Size: {complexity['size_mb']:.2f} MB\n")

    # 4. Setup summary
    logger.info("Experimental Setup")
    logger.info("Framework : PyTorch")
    logger.info(f"Device    : {device.type.upper()}")
    logger.info(f"Batch size: {config['data']['batch_size']}")
    logger.info(f"Epochs    : {config['training']['epochs']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Optimizer : {config['training']['optimizer'].capitalize()}")
    logger.info(f"Scheduler : {config['training'].get('scheduler', 'none')}")
    logger.info(f"Mixed precision: {config['training'].get('mixed_precision', False)}")
    logger.info(f"Label smoothing: {config['training'].get('label_smoothing', 0.0)}")
    logger.info(f"Class weights: {config['training'].get('use_class_weights', False)}")
    logger.info(f"Grad clip : {config['training'].get('grad_clip', 'none')}")
    early_stopping_cfg = config['training'].get('early_stopping', {})
    logger.info(
        "Early stopping: "
        f"enabled={early_stopping_cfg.get('enabled', True)}, "
        f"monitor={early_stopping_cfg.get('monitor', 'val_accuracy')}, "
        f"patience={early_stopping_cfg.get('patience', config['training'].get('patience', 12))}, "
        f"min_delta={early_stopping_cfg.get('min_delta', 0.0)}"
    )
    logger.info(f"Uncertainty method: {config['uncertainty']['method']}")
    logger.info(f"Explanation method: {config['explainability']['method']}\n")

    train_losses, val_losses, val_accuracies = [], [], []
    if args.eval_only:
        logger.info(f"Evaluation-only mode: loading weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        model.eval()
    else:
        # 5. Optimizer
        lr = config['training']['learning_rate']
        wd = config['training']['weight_decay']
        if config['training']['optimizer'].lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

        # 6. Training
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            logger=logger,
            save_dir=save_dir,
            config=config
        )

        resume_from = config.get('training', {}).get('checkpoint', {}).get('resume_from')
        if resume_from:
            trainer.load_checkpoint(resume_from)

        train_losses, val_losses, val_accuracies = trainer.train(config['training']['epochs'])
        os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
        plot_training_curves(
            train_losses, val_losses, val_accuracies,
            os.path.join(save_dir, 'plots')
        )

        # Load best model
        best_path = os.path.join(save_dir, 'models', 'best_model.pth')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            model.eval()
            logger.info("Loaded best model checkpoint.")

    # 7. Temperature Scaling (post-hoc calibration)
    calibration_cfg = config.get('calibration', {})
    use_temp_scaling = calibration_cfg.get('temperature_scaling', False)
    min_calibration_acc = calibration_cfg.get('min_accuracy_for_temperature_scaling', 0.3)
    best_val_acc = max(val_accuracies, default=0.0)
    eval_model = model
    temperature_value = None
    if args.eval_only and use_temp_scaling:
        logger.info("Skipping Temperature Scaling in evaluation-only mode.")
    elif use_temp_scaling and val_size > 0 and best_val_acc >= min_calibration_acc:
        from models.temperature_scaling import TemperatureScaling
        ts = TemperatureScaling()
        eval_model, T = ts.calibrate_model(model, val_loader, device, logger)
        temperature_value = T
        logger.info(f"Applied Temperature Scaling (T={T:.4f}) for evaluation.")
        
        # Save temperature parameter
        temp_path = os.path.join(save_dir, 'models', 'temperature.pth')
        torch.save({'temperature': T}, temp_path)
        logger.info(f"Saved temperature parameter to {temp_path}")
    elif use_temp_scaling and val_size > 0:
        logger.warning(
            "Skipping Temperature Scaling because best validation accuracy "
            f"({best_val_acc:.4f}) is below threshold {min_calibration_acc:.4f}."
        )

    # 8. Evaluation
    if test_size > 0:
        evaluator = Evaluator(
            model=eval_model,
            dataloader=test_loader,
            device=device,
            logger=logger,
            save_dir=save_dir,
            classes=classes,
            config=config
        )

        results = evaluator.evaluate()
        logger.info(
            f"Final Test Accuracy: {results['classification']['accuracy']:.4f} | "
            f"F1-macro: {results['classification'].get('f1', 0.0):.4f}"
        )
        calibration = results.get('calibration', {})
        if calibration:
            logger.info(
                "Calibration | "
                f"ECE: {calibration['ece']:.4f} | "
                f"MCE: {calibration['mce']:.4f} | "
                f"NLL: {calibration['nll']:.4f} | "
                f"Brier: {calibration['brier']:.4f}"
            )
        uncertainty = results.get('uncertainty', {})
        if uncertainty:
            logger.info(
                "Uncertainty | "
                f"MC samples: {uncertainty['num_samples']} | "
                f"Entropy: {uncertainty['predictive_entropy_mean']:.4f} "
                f"+/- {uncertainty['predictive_entropy_std']:.4f} | "
                f"Variance: {uncertainty['probability_variance_mean']:.6f} "
                f"+/- {uncertainty['probability_variance_std']:.6f}"
            )
        if results['latency_ms'] > 0:
            logger.info(f"Inference Latency: {results['latency_ms']:.2f} ms/image")

        # 8.5 Uncertainty Quantification (Formal Stage)
        if config['uncertainty']['method'] == 'mc_dropout':
            from metrics.uncertainty import MCDropoutUQ
            uq_analyzer = MCDropoutUQ(
                model=model,  # use uncalibrated base model for pure epistemic UQ
                dataloader=test_loader,
                device=device,
                num_samples=config['uncertainty']['num_samples']
            )
            uq_metrics = uq_analyzer.compute_predictive_uncertainty()
            logger.info("--- Formal UQ Analysis ---")
            logger.info(f"Mean Predictive Entropy: {uq_metrics['mean_entropy']:.4f}")
            logger.info(f"Mean Predictive Variance: {uq_metrics['mean_variance']:.4f}")
            
            # Sanity check for zero-variance (indicating dropout might be inactive)
            if uq_metrics['mean_variance'] < 1e-6:
                logger.warning(
                    f"Extremely low predictive variance ({uq_metrics['mean_variance']:.2e}). "
                    "Verify that 'dropout_rate' in config is > 0 and that the model contains Dropout layers."
                )

        if config.get('explainability', {}).get('generate', True):
            # XAI — use base model (not calibration wrapper) for Grad-CAM
            model.eval()
            evaluator.model = model
            xai_samples = int(config.get('explainability', {}).get('num_samples', 5))
            xai_results = evaluator.generate_explanations(num_samples=xai_samples)
            if xai_results:
                results['xai'] = xai_results
        else:
            logger.info("Skipping XAI generation because explainability.generate=false.")

        if config.get('experiment', {}).get('save_summary', False):
            summary_paths = save_run_summary(
                save_dir=save_dir,
                config=config,
                complexity=complexity,
                train_losses=train_losses,
                val_losses=val_losses,
                val_accuracies=val_accuracies,
                results=results,
                temperature=temperature_value
            )
            logger.info(f"Saved metrics summary: {summary_paths['json']} and {summary_paths['csv']}")

    logger.info(f"Experiment {save_dir} completed successfully.")


if __name__ == "__main__":
    main()
