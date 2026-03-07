import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.seed import seed_everything
from utils.logger import setup_logger
from utils.visualization import plot_training_curves
from data.data_loader import get_dataloaders
from models.resnet_classifier import ResNet50Classifier
from training.trainer import Trainer
from training.evaluator import Evaluator
from metrics.calibration_metrics import count_model_parameters

def get_next_run_dir(base_dir="EXPERIMENT"):
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    if not existing_runs:
        return os.path.join(base_dir, 'run_001')
    run_nums = [int(d.split('_')[1]) for d in existing_runs]
    next_num = max(run_nums) + 1
    return os.path.join(base_dir, f'run_{next_num:03d}')

def load_classes(classes_file):
    if not os.path.exists(classes_file):
        return [f"Class {i}" for i in range(29)]
    with open(classes_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        if not lines:
            return [f"Class {i}" for i in range(29)]
        return lines

def main():
    parser = argparse.ArgumentParser(description="TSR Research Framework")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup specific run directory
    save_dir = get_next_run_dir()
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Reproducibility
    seed_everything(config['experiment']['seed'])
    
    # Setup logger
    logger = setup_logger(save_dir)
    logger.info(f"Initialized experiment in {save_dir}")

    # Set device
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Data processing
    train_loader, val_loader, test_loader = get_dataloaders(config)
    classes = load_classes(config['data']['classes_file'])
    
    # Check if dataloaders are empty
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    
    if train_size == 0 or val_size == 0 or test_size == 0:
        logger.warning("One or more datasets are empty. Please check paths in config.yaml.")
    
    # Output Dataset Summary
    logger.info("\nDataset Summary")
    logger.info("Dataset: Vietnamese Traffic Signs")
    logger.info(f"Total samples : {train_size + val_size + test_size}")
    logger.info(f"Classes       : {len(classes)}")
    logger.info(f"Training samples : {train_size}")
    logger.info(f"Validation samples : {val_size}")
    logger.info(f"Testing samples  : {test_size}\n")

    # 3. Model mapping
    model = ResNet50Classifier(
        num_classes=config['data']['num_classes'],
        pretrained=config['model']['pretrained'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)
    
    # Log complexity
    complexity = count_model_parameters(model)
    logger.info(f"Model Parameters: {complexity['num_params']:,}")
    logger.info(f"Trainable Parameters: {complexity['trainable_params']:,}")
    logger.info(f"Model Size: {complexity['size_mb']:.2f} MB\n")

    # 4. Experimental Setup Output
    logger.info("Experimental Setup")
    logger.info("Framework : PyTorch")
    logger.info(f"Device    : {device.type.upper()}")
    logger.info(f"Batch size: {config['data']['batch_size']}")
    logger.info(f"Epochs    : {config['training']['epochs']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Optimizer : {config['training']['optimizer'].capitalize()}")
    logger.info(f"Model: {config['model']['name'].capitalize()}")
    logger.info(f"Uncertainty method: {config['uncertainty']['method']}")
    logger.info(f"Explanation method: {config['explainability']['method']}\n")

    # Optimizer
    if config['training']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['training']['learning_rate'], momentum=0.9, weight_decay=config['training']['weight_decay'])

    # 5. Training
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        logger=logger,
        save_dir=save_dir,
        patience=config['training']['patience']
    )
    
    if train_size > 0:
        train_losses, val_losses, val_accuracies = trainer.train(config['training']['epochs'])
        os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
        plot_training_curves(train_losses, val_losses, val_accuracies, os.path.join(save_dir, 'plots'))

        # Load best model for evaluation
        best_path = os.path.join(save_dir, 'models', 'best_model.pth')
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device))
        
    # 6. Evaluation
    if test_size > 0:
        evaluator = Evaluator(
            model=model,
            dataloader=test_loader,
            device=device,
            logger=logger,
            save_dir=save_dir,
            classes=classes,
            config=config
        )
        
        results = evaluator.evaluate()
        logger.info(f"Final Test Accuracy: {results['classification']['accuracy']:.4f}")
        if results['latency_ms'] > 0:
            logger.info(f"Inference Latency: {results['latency_ms']:.2f} ms/image")
            
        # XAI
        evaluator.generate_explanations(num_samples=5)
        
    logger.info(f"Experiment {save_dir} completed successfully.")

if __name__ == "__main__":
    main()
