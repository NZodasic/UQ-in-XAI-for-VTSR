import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str):
    """
    Plots and saves the confusion matrix.
    """
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, cmap='Blues', fmt='g',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_curves(train_losses: list, val_losses: list, val_accuracies: list, save_dir: str):
    """
    Plots training and validation loss, and validation accuracy.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    if val_losses:
        plt.plot(epochs, val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if val_accuracies:
        plt.plot(epochs, val_accuracies, label='Val Accuracy', color='orange')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curve.png'))
    plt.close()

def save_xai_overlay(image_tensor: torch.Tensor, heatmap: np.ndarray, save_path: str):
    """
    Overlays a heatmap on an image tensor and saves it.
    image_tensor: [C, H, W] normalized
    heatmap: [H, W] normalized between 0 and 1
    """
    # Denormalize image for visualization
    # We will assume standard ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_calibration_curve(fraction_of_positives, mean_predicted_value, save_path: str):
    """
    Plots a calibration curve.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Fraction of Positives (Accuracy)")
    plt.title("Calibration Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
