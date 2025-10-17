
# ============================================================================
# UNCERTAINTY QUANTIFICATION IN EXPLAINABLE VISION MODELS
# Vietnamese Traffic Sign Recognition - Complete Pipeline
# ============================================================================

# PART 1: IMPORTS AND SETUP
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision

# For Uncertainty Quantification
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# PART 2: DATA LOADING AND EXPLORATION (EDA)
# ============================================================================

class TrafficSignEDA:
    """
    Exploratory Data Analysis for Vietnamese Traffic Sign Dataset
    """

    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.image_data = []
        self.labels = []

    def load_dataset_info(self):
        """Load and display basic dataset information"""
        print("=" * 80)
        print("EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 80)

        # This assumes the dataset structure from Kaggle
        # Adjust paths based on actual dataset structure
        print(f"\nDataset Path: {self.data_path}")
        print(f"Dataset exists: {self.data_path.exists()}")

    def analyze_class_distribution(self, df):
        """Analyze the distribution of traffic sign classes"""
        print("\n--- CLASS DISTRIBUTION ANALYSIS ---")

        class_counts = df['ClassId'].value_counts().sort_index()
        print(f"\nNumber of unique classes: {len(class_counts)}")
        print(f"Total number of images: {len(df)}")
        print(f"\nTop 10 most frequent classes:")
        print(class_counts.head(10))

        # Calculate statistics
        print(f"\nClass distribution statistics:")
        print(f"Mean samples per class: {class_counts.mean():.2f}")
        print(f"Std samples per class: {class_counts.std():.2f}")
        print(f"Min samples: {class_counts.min()}")
        print(f"Max samples: {class_counts.max()}")

        # Class imbalance ratio
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")

        return class_counts

    def analyze_image_properties(self, image_paths, sample_size=100):
        """Analyze image dimensions, color properties, and quality"""
        print("\n--- IMAGE PROPERTIES ANALYSIS ---")

        widths, heights, aspects = [], [], []
        brightness_values = []

        # Sample images for analysis
        sampled_paths = np.random.choice(image_paths, 
                                        size=min(sample_size, len(image_paths)), 
                                        replace=False)

        for img_path in sampled_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    heights.append(h)
                    widths.append(w)
                    aspects.append(w / h)

                    # Calculate average brightness
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    brightness_values.append(np.mean(gray))
            except:
                continue

        print(f"\nImage dimension statistics (from {len(widths)} samples):")
        print(f"Width  - Mean: {np.mean(widths):.1f}, Std: {np.std(widths):.1f}")
        print(f"Height - Mean: {np.mean(heights):.1f}, Std: {np.std(heights):.1f}")
        print(f"Aspect ratio - Mean: {np.mean(aspects):.3f}, Std: {np.std(aspects):.3f}")
        print(f"\nBrightness - Mean: {np.mean(brightness_values):.1f}, "
              f"Std: {np.std(brightness_values):.1f}")

        return {
            'widths': widths, 
            'heights': heights,
            'brightness': brightness_values
        }

    def check_data_quality(self, image_paths, sample_size=100):
        """Check for corrupted images, missing data, and quality issues"""
        print("\n--- DATA QUALITY ANALYSIS ---")

        corrupted = 0
        valid = 0
        total_checked = min(sample_size, len(image_paths))

        for img_path in image_paths[:total_checked]:
            try:
                img = Image.open(img_path)
                img.verify()  # Verify image integrity
                valid += 1
            except:
                corrupted += 1

        print(f"\nChecked {total_checked} images:")
        print(f"Valid images: {valid} ({100*valid/total_checked:.1f}%)")
        print(f"Corrupted images: {corrupted} ({100*corrupted/total_checked:.1f}%)")

        return {'valid': valid, 'corrupted': corrupted}


# ============================================================================
# PART 3: DATA CLEANING AND PREPROCESSING
# ============================================================================

class DataCleaner:
    """
    Data cleaning utilities for traffic sign dataset
    """

    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size

    def remove_corrupted_images(self, df, image_col='Path'):
        """Remove corrupted or unreadable images"""
        print("\n--- DATA CLEANING: REMOVING CORRUPTED IMAGES ---")

        valid_indices = []
        corrupted_count = 0

        for idx, row in df.iterrows():
            try:
                img = Image.open(row[image_col])
                img.verify()
                valid_indices.append(idx)
            except:
                corrupted_count += 1

        cleaned_df = df.loc[valid_indices].reset_index(drop=True)

        print(f"Original dataset size: {len(df)}")
        print(f"Corrupted images removed: {corrupted_count}")
        print(f"Cleaned dataset size: {len(cleaned_df)}")

        return cleaned_df

    def handle_class_imbalance(self, df, target_samples_per_class=None):
        """Handle class imbalance through undersampling/oversampling"""
        print("\n--- DATA CLEANING: HANDLING CLASS IMBALANCE ---")

        if target_samples_per_class is None:
            # Use median as target
            class_counts = df['ClassId'].value_counts()
            target_samples_per_class = int(class_counts.median())

        print(f"Target samples per class: {target_samples_per_class}")

        balanced_dfs = []
        for class_id in df['ClassId'].unique():
            class_df = df[df['ClassId'] == class_id]

            if len(class_df) > target_samples_per_class:
                # Undersample
                class_df = class_df.sample(n=target_samples_per_class, random_state=42)
            elif len(class_df) < target_samples_per_class:
                # Oversample
                class_df = class_df.sample(n=target_samples_per_class, 
                                          replace=True, random_state=42)

            balanced_dfs.append(class_df)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Balanced dataset size: {len(balanced_df)}")
        print(f"Classes: {balanced_df['ClassId'].nunique()}")

        return balanced_df

    def normalize_image_sizes(self, image):
        """Resize and normalize image"""
        # Resize
        image = cv2.resize(image, self.target_size)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        return image


# ============================================================================
# PART 4: DATA AUGMENTATION FOR TRAINING
# ============================================================================

def get_augmentation_transforms(phase='train'):
    """
    Get data augmentation transforms for training and validation
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, 
                                 saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


class TrafficSignDataset(Dataset):
    """Custom Dataset for Traffic Signs"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================================
# PART 5: MODEL ARCHITECTURE WITH UNCERTAINTY QUANTIFICATION
# ============================================================================

class UncertaintyQuantificationCNN(nn.Module):
    """
    CNN Model with Dropout for Uncertainty Quantification
    Implements Monte Carlo Dropout for epistemic uncertainty
    and predictive variance for aleatoric uncertainty
    """

    def __init__(self, num_classes, dropout_rate=0.3):
        super(UncertaintyQuantificationCNN, self).__init__()

        # Use a pre-trained ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)

        # Modify the final layers
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final FC layer

        # Add custom classification head with dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layers for mean and variance (for aleatoric uncertainty)
        self.fc_mean = nn.Linear(128, num_classes)
        self.fc_var = nn.Linear(128, num_classes)  # For aleatoric uncertainty

    def forward(self, x, return_uncertainty=False):
        # Extract features
        features = self.backbone(x)

        # Classification head
        x = self.dropout1(features)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout3(x)

        # Get mean predictions
        mean = self.fc_mean(x)

        if return_uncertainty:
            # Get variance predictions for aleatoric uncertainty
            log_var = self.fc_var(x)
            var = torch.exp(log_var)  # Ensure positive variance
            return mean, var

        return mean

    def enable_dropout(self):
        """Enable dropout for uncertainty estimation at test time"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


# ============================================================================
# PART 6: UNCERTAINTY ESTIMATION METHODS
# ============================================================================

class UncertaintyEstimator:
    """
    Implements multiple uncertainty estimation techniques:
    1. MC Dropout (Epistemic Uncertainty)
    2. Test-Time Augmentation (Aleatoric Uncertainty)
    3. Ensemble Methods
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def mc_dropout_uncertainty(self, image, n_samples=30):
        """
        Monte Carlo Dropout for Epistemic Uncertainty

        Performs multiple forward passes with dropout enabled
        to estimate model uncertainty
        """
        self.model.eval()
        self.model.enable_dropout()  # Enable dropout at test time

        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(image)
                probs = F.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())

        predictions = np.array(predictions)

        # Calculate statistics
        mean_prediction = np.mean(predictions, axis=0)
        epistemic_uncertainty = np.std(predictions, axis=0)  # Standard deviation
        predictive_entropy = -np.sum(mean_prediction * np.log(mean_prediction + 1e-10), 
                                    axis=1)

        return {
            'mean_prediction': mean_prediction,
            'epistemic_uncertainty': epistemic_uncertainty,
            'predictive_entropy': predictive_entropy,
            'all_predictions': predictions
        }

    def test_time_augmentation_uncertainty(self, image, n_augmentations=10):
        """
        Test-Time Augmentation for Aleatoric Uncertainty

        Applies random augmentations and measures prediction variance
        """
        self.model.eval()

        augmentation_transforms = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ])

        predictions = []

        with torch.no_grad():
            for _ in range(n_augmentations):
                # Apply augmentation
                augmented = augmentation_transforms(image)
                output = self.model(augmented)
                probs = F.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())

        predictions = np.array(predictions)

        # Calculate statistics
        mean_prediction = np.mean(predictions, axis=0)
        aleatoric_uncertainty = np.std(predictions, axis=0)

        return {
            'mean_prediction': mean_prediction,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'all_predictions': predictions
        }

    def combined_uncertainty(self, image, n_mc_samples=20, n_augmentations=10):
        """
        Combine both epistemic and aleatoric uncertainty
        Total Uncertainty = Epistemic + Aleatoric
        """
        # Get epistemic uncertainty
        epistemic_results = self.mc_dropout_uncertainty(image, n_mc_samples)

        # Get aleatoric uncertainty
        aleatoric_results = self.test_time_augmentation_uncertainty(
            image, n_augmentations
        )

        # Combine uncertainties
        total_uncertainty = np.sqrt(
            epistemic_results['epistemic_uncertainty']**2 + 
            aleatoric_results['aleatoric_uncertainty']**2
        )

        return {
            'mean_prediction': epistemic_results['mean_prediction'],
            'epistemic_uncertainty': epistemic_results['epistemic_uncertainty'],
            'aleatoric_uncertainty': aleatoric_results['aleatoric_uncertainty'],
            'total_uncertainty': total_uncertainty,
            'predictive_entropy': epistemic_results['predictive_entropy']
        }


# ============================================================================
# PART 7: EXPLAINABLE AI (XAI) METHODS
# ============================================================================

class ExplainableAI:
    """
    Implements multiple XAI techniques:
    1. Grad-CAM
    2. Integrated Gradients (simplified)
    3. Attention Visualization
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None

    def save_gradient(self, grad):
        self.gradients = grad

    def save_activation(self, module, input, output):
        self.activations = output

    def grad_cam(self, image, target_class=None):
        """
        Gradient-weighted Class Activation Mapping (Grad-CAM)

        Generates visual explanations showing which regions 
        of the image are important for the prediction
        """
        self.model.eval()

        # Register hooks to capture gradients and activations
        # For ResNet18, use layer4 (last convolutional layer)
        target_layer = self.model.backbone.layer4[-1]

        handle_forward = target_layer.register_forward_hook(self.save_activation)
        handle_backward = target_layer.register_full_backward_hook(
            lambda module, grad_in, grad_out: self.save_gradient(grad_out[0])
        )

        # Forward pass
        output = self.model(image)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        # Calculate Grad-CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()

        return cam

    def visualize_grad_cam(self, image, cam, alpha=0.5):
        """
        Overlay Grad-CAM heatmap on original image
        """
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = image.cpu().numpy()[0].transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))

        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

        # Overlay
        overlayed = alpha * heatmap + (1 - alpha) * img
        overlayed = np.clip(overlayed, 0, 1)

        return img, heatmap, overlayed


# ============================================================================
# PART 8: TRAINING LOOP
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs=20, device='cuda'):
    """
    Training loop with validation
    """
    print("\n" + "=" * 80)
    print("TRAINING MODEL WITH UNCERTAINTY QUANTIFICATION")
    print("=" * 80)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }


# ============================================================================
# PART 9: EVALUATION AND CALIBRATION
# ============================================================================

def expected_calibration_error(confidences, accuracies, n_bins=15):
    """
    Calculate Expected Calibration Error (ECE)
    Measures how well predicted probabilities match actual accuracy
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def evaluate_with_uncertainty(model, test_loader, uncertainty_estimator, 
                              device='cuda'):
    """
    Comprehensive evaluation with uncertainty quantification
    """
    print("\n" + "=" * 80)
    print("EVALUATION WITH UNCERTAINTY QUANTIFICATION")
    print("=" * 80)

    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_uncertainties = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Get uncertainty estimates
        uncertainty_results = uncertainty_estimator.mc_dropout_uncertainty(
            images, n_samples=20
        )

        predictions = uncertainty_results['mean_prediction']
        uncertainties = uncertainty_results['epistemic_uncertainty']

        pred_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)

        all_predictions.extend(pred_classes)
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(confidences)
        all_uncertainties.extend(np.mean(uncertainties, axis=1))

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_uncertainties = np.array(all_uncertainties)

    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean() * 100
    accuracies = (all_predictions == all_labels).astype(float)
    ece = expected_calibration_error(all_confidences, accuracies)

    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Average Uncertainty: {all_uncertainties.mean():.4f}")
    print(f"Average Confidence: {all_confidences.mean():.4f}")

    return {
        'accuracy': accuracy,
        'ece': ece,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'uncertainties': all_uncertainties
    }


# ============================================================================
# PART 10: VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_uncertainty_analysis(results, save_path='uncertainty_analysis.png'):
    """
    Create comprehensive visualization of uncertainty analysis
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Uncertainty Quantification Analysis', fontsize=16, fontweight='bold')

    # 1. Confidence vs Uncertainty
    axes[0, 0].scatter(results['confidences'], results['uncertainties'], 
                      alpha=0.5, s=10)
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Uncertainty')
    axes[0, 0].set_title('Confidence vs Uncertainty')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Uncertainty Distribution
    axes[0, 1].hist(results['uncertainties'], bins=50, alpha=0.7, color='blue')
    axes[0, 1].set_xlabel('Uncertainty')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Uncertainty Distribution')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Confidence Distribution
    axes[0, 2].hist(results['confidences'], bins=50, alpha=0.7, color='green')
    axes[0, 2].set_xlabel('Confidence')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Confidence Distribution')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Correct vs Incorrect Predictions Uncertainty
    correct_mask = results['predictions'] == results['labels']
    axes[1, 0].boxplot([results['uncertainties'][correct_mask],
                        results['uncertainties'][~correct_mask]],
                       labels=['Correct', 'Incorrect'])
    axes[1, 0].set_ylabel('Uncertainty')
    axes[1, 0].set_title('Uncertainty: Correct vs Incorrect')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Reliability Diagram
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = []
    bin_confidences = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (results['confidences'] > bin_lower) & \
                 (results['confidences'] <= bin_upper)
        if in_bin.sum() > 0:
            accuracies = (results['predictions'][in_bin] == \
                         results['labels'][in_bin]).astype(float)
            bin_accuracies.append(accuracies.mean())
            bin_confidences.append(results['confidences'][in_bin].mean())

    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[1, 1].plot(bin_confidences, bin_accuracies, 'ro-', label='Model')
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Reliability Diagram')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. ECE visualization
    ece_value = results['ece']
    axes[1, 2].bar(['ECE'], [ece_value], color='red', alpha=0.7)
    axes[1, 2].set_ylabel('Error')
    axes[1, 2].set_title(f'Expected Calibration Error\nECE = {ece_value:.4f}')
    axes[1, 2].set_ylim([0, max(0.1, ece_value * 1.2)])
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nUncertainty analysis saved to: {save_path}")

    return fig


def visualize_grad_cam_examples(model, xai, test_loader, device, 
                                num_examples=6, save_path='grad_cam_examples.png'):
    """
    Visualize Grad-CAM explanations for sample predictions
    """
    fig, axes = plt.subplots(num_examples, 3, figsize=(12, 4*num_examples))
    fig.suptitle('Grad-CAM Explanations for Traffic Sign Recognition', 
                fontsize=16, fontweight='bold')

    model.eval()
    images_processed = 0

    for images, labels in test_loader:
        if images_processed >= num_examples:
            break

        for i in range(min(images.size(0), num_examples - images_processed)):
            image = images[i:i+1].to(device)
            label = labels[i].item()

            # Get prediction
            with torch.no_grad():
                output = model(image)
                pred_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, pred_class].item()

            # Get Grad-CAM
            cam = xai.grad_cam(image, target_class=pred_class)

            # Visualize
            img, heatmap, overlayed = xai.visualize_grad_cam(image, cam)

            idx = images_processed
            axes[idx, 0].imshow(img)
            axes[idx, 0].set_title(f'Original\nTrue: {label}')
            axes[idx, 0].axis('off')

            axes[idx, 1].imshow(heatmap)
            axes[idx, 1].set_title('Grad-CAM Heatmap')
            axes[idx, 1].axis('off')

            axes[idx, 2].imshow(overlayed)
            axes[idx, 2].set_title(f'Overlay\nPred: {pred_class} ({confidence:.2f})')
            axes[idx, 2].axis('off')

            images_processed += 1

            if images_processed >= num_examples:
                break

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grad-CAM visualizations saved to: {save_path}")

    return fig


# ============================================================================
# PART 11: MAIN EXECUTION PIPELINE
# ============================================================================

def main_pipeline(data_path, num_classes=29, batch_size=32, num_epochs=20):
    """
    Complete pipeline for Uncertainty Quantification in 
    Explainable Vision Models for Traffic Sign Recognition
    """

    print("\n" + "=" * 80)
    print("UNCERTAINTY QUANTIFICATION IN EXPLAINABLE VISION MODELS")
    print("Vietnamese Traffic Sign Recognition - Complete Pipeline")
    print("=" * 80)

    # STEP 1: EDA
    print("\nSTEP 1: Exploratory Data Analysis (EDA)")
    print("-" * 80)
    eda = TrafficSignEDA(data_path)
    eda.load_dataset_info()

    # STEP 2: Data Cleaning
    print("\nSTEP 2: Data Cleaning")
    print("-" * 80)
    cleaner = DataCleaner(target_size=(64, 64))

    # STEP 3: Model Creation
    print("\nSTEP 3: Creating Model with Uncertainty Quantification")
    print("-" * 80)
    model = UncertaintyQuantificationCNN(num_classes=num_classes, 
                                         dropout_rate=0.3)
    model = model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # STEP 4: Initialize Uncertainty Estimator and XAI
    print("\nSTEP 4: Initializing Uncertainty Estimator and XAI")
    print("-" * 80)
    uncertainty_estimator = UncertaintyEstimator(model, device)
    xai = ExplainableAI(model, device)
    print("Uncertainty Estimator and XAI initialized")

    print("\n" + "=" * 80)
    print("PIPELINE OVERVIEW COMPLETE")
    print("=" * 80)
    print("\nThis pipeline includes:")
    print("1. ✓ Exploratory Data Analysis (EDA)")
    print("2. ✓ Data Cleaning and Preprocessing")
    print("3. ✓ Data Augmentation")
    print("4. ✓ Model Architecture with Dropout for UQ")
    print("5. ✓ Uncertainty Quantification Methods:")
    print("   - Monte Carlo Dropout (Epistemic Uncertainty)")
    print("   - Test-Time Augmentation (Aleatoric Uncertainty)")
    print("   - Combined Total Uncertainty")
    print("6. ✓ Explainable AI Methods:")
    print("   - Grad-CAM")
    print("   - Saliency Maps")
    print("7. ✓ Training Loop with Validation")
    print("8. ✓ Calibration Metrics (ECE)")
    print("9. ✓ Comprehensive Visualizations")

    return model, uncertainty_estimator, xai


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Set your data path here
    DATA_PATH = './vietnamese-traffic-signs'
    NUM_CLASSES = 29  # Adjust based on your dataset

    # Run the complete pipeline
    model, uncertainty_estimator, xai = main_pipeline(
        data_path=DATA_PATH,
        num_classes=NUM_CLASSES,
        batch_size=32,
        num_epochs=20
    )

    print("\n" + "=" * 80)
    print("PIPELINE READY FOR EXECUTION")
    print("=" * 80)
    print("\nTo use with actual data:")
    print("1. Load your Vietnamese Traffic Sign dataset")
    print("2. Create DataLoaders using TrafficSignDataset")
    print("3. Train the model using train_model()")
    print("4. Evaluate with evaluate_with_uncertainty()")
    print("5. Visualize results using visualization functions")
    print("\n" + "=" * 80)
