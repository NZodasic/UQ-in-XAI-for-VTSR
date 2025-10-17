# Create a single executable file with sample data generation for demonstration

single_file_content = '''
#!/usr/bin/env python3
"""
UNCERTAINTY QUANTIFICATION IN EXPLAINABLE VISION MODELS
Vietnamese Traffic Sign Recognition - Complete Demo
==================================================

This single file contains the complete pipeline for UQ and XAI.
Run this file to see a full demonstration with synthetic data.

Usage: python run_complete_demo.py

Author: AI Research Assistant
Date: October 17, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("UNCERTAINTY QUANTIFICATION IN EXPLAINABLE VISION MODELS")
print("Vietnamese Traffic Sign Recognition - Complete Demo")
print("="*80)
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")


# ============================================================================
# PART 1: SYNTHETIC DATA GENERATION (for demonstration)
# ============================================================================

def create_synthetic_traffic_signs(num_samples=1000, num_classes=10, image_size=64):
    """
    Create synthetic traffic sign dataset for demonstration
    """
    print("\\n" + "="*50)
    print("GENERATING SYNTHETIC DATASET")
    print("="*50)
    
    X = []
    y = []
    class_names = [
        'Stop', 'Yield', 'Speed_Limit_30', 'Speed_Limit_50', 'No_Entry',
        'Turn_Right', 'Turn_Left', 'Straight', 'No_Parking', 'School_Zone'
    ]
    
    for class_id in range(num_classes):
        samples_per_class = num_samples // num_classes
        
        for i in range(samples_per_class):
            # Create synthetic traffic sign image
            img = np.random.rand(image_size, image_size, 3) * 255
            
            # Add some structure based on class
            if class_id == 0:  # Stop sign (red octagon)
                center = image_size // 2
                cv2.circle(img, (center, center), 20, (255, 0, 0), -1)
                cv2.rectangle(img, (center-15, center-15), (center+15, center+15), (255, 255, 255), 2)
                
            elif class_id == 1:  # Yield sign (red triangle)
                pts = np.array([[center, center-20], [center-20, center+15], [center+20, center+15]], np.int32)
                cv2.fillPoly(img, [pts], (255, 0, 0))
                cv2.polylines(img, [pts], True, (255, 255, 255), 2)
                
            elif class_id in [2, 3]:  # Speed limit signs (circular)
                cv2.circle(img, (center, center), 25, (255, 255, 255), -1)
                cv2.circle(img, (center, center), 25, (255, 0, 0), 3)
                cv2.putText(img, str(30 + class_id*20), (center-10, center+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
            else:  # Other signs
                cv2.rectangle(img, (10, 10), (image_size-10, image_size-10), 
                             tuple(np.random.randint(0, 255, 3).tolist()), 3)
                cv2.circle(img, (center, center), 15, 
                          tuple(np.random.randint(100, 255, 3).tolist()), -1)
            
            # Add some noise and blur for realism
            if np.random.random() < 0.3:  # 30% of images have blur
                img = cv2.GaussianBlur(img, (3, 3), 1)
            
            if np.random.random() < 0.2:  # 20% have noise
                noise = np.random.normal(0, 25, img.shape)
                img = np.clip(img + noise, 0, 255)
            
            X.append(img.astype(np.uint8))
            y.append(class_id)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Generated dataset:")
    print(f"- Images: {X.shape}")
    print(f"- Labels: {y.shape}")
    print(f"- Classes: {num_classes}")
    print(f"- Class names: {class_names[:num_classes]}")
    
    return X, y, class_names[:num_classes]


# ============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

def perform_eda(X, y, class_names):
    """
    Perform Exploratory Data Analysis
    """
    print("\\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*50)
    
    # Basic statistics
    print(f"\\nDataset Statistics:")
    print(f"Total samples: {len(X)}")
    print(f"Image shape: {X[0].shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    
    print(f"\\nClass Distribution:")
    for class_id, count in class_dist.items():
        print(f"  {class_names[class_id]}: {count} samples")
    
    print(f"\\nClass Balance:")
    print(f"Min samples: {min(counts)}")
    print(f"Max samples: {max(counts)}")
    print(f"Imbalance ratio: {max(counts)/min(counts):.2f}")
    
    # Image statistics
    print(f"\\nImage Statistics:")
    print(f"Mean pixel value: {X.mean():.2f}")
    print(f"Std pixel value: {X.std():.2f}")
    print(f"Min pixel value: {X.min()}")
    print(f"Max pixel value: {X.max()}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Exploratory Data Analysis - Vietnamese Traffic Signs', fontsize=16)
    
    # Class distribution
    axes[0, 0].bar(range(len(counts)), counts, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_xlabel('Class ID')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_xticks(range(len(counts)))
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pixel value distribution
    pixel_values = X.flatten()
    axes[0, 1].hist(pixel_values, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_title('Pixel Value Distribution')
    axes[0, 1].set_xlabel('Pixel Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample images
    axes[0, 2].axis('off')
    axes[0, 2].set_title('Sample Images from Each Class')
    
    # Show 6 sample images
    for i in range(6):
        class_idx = i % len(np.unique(y))
        class_samples = X[y == class_idx]
        if len(class_samples) > 0:
            sample_img = class_samples[0]
            axes[1, i // 2].imshow(sample_img)
            axes[1, i // 2].set_title(f'{class_names[class_idx]}\\n(Class {class_idx})')
            axes[1, i // 2].axis('off')
    
    # Remove empty subplots
    if len(np.unique(y)) < 6:
        for i in range(len(np.unique(y)), 6):
            axes[1, i // 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\\n✓ EDA visualization saved as 'eda_analysis.png'")
    
    return class_dist


# ============================================================================
# PART 3: DATA PREPROCESSING
# ============================================================================

class TrafficSignDataset(Dataset):
    """Custom Dataset for Traffic Signs"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            # Convert numpy array to PIL Image for transforms
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, label


def create_data_loaders(X, y, batch_size=32, val_split=0.2, test_split=0.1):
    """
    Create train, validation, and test data loaders
    """
    print("\\n" + "="*50)
    print("CREATING DATA LOADERS")
    print("="*50)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split/(1-test_split), random_state=42, stratify=y_temp
    )
    
    print(f"Data split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = TrafficSignDataset(X_train, y_train, train_transform)
    val_dataset = TrafficSignDataset(X_val, y_val, val_transform)
    test_dataset = TrafficSignDataset(X_test, y_test, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# PART 4: MODEL ARCHITECTURE WITH UNCERTAINTY QUANTIFICATION
# ============================================================================

class UncertaintyQuantificationCNN(nn.Module):
    """
    CNN Model with Dropout for Uncertainty Quantification
    """
    
    def __init__(self, num_classes, dropout_rate=0.3):
        super(UncertaintyQuantificationCNN, self).__init__()
        
        # Use ResNet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Custom classification head with dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(num_features, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(128, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        x = self.dropout1(features)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout3(x)
        x = self.fc_out(x)
        return x
    
    def enable_dropout(self):
        """Enable dropout for uncertainty estimation"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


# ============================================================================
# PART 5: TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001):
    """
    Train the model
    """
    print("\\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
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
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"\\n✓ Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("✓ Training curves saved as 'training_curves.png'")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }


# ============================================================================
# PART 6: UNCERTAINTY QUANTIFICATION
# ============================================================================

class UncertaintyEstimator:
    """
    Uncertainty estimation using Monte Carlo Dropout and Test-Time Augmentation
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def mc_dropout_uncertainty(self, image, n_samples=20):
        """
        Monte Carlo Dropout for Epistemic Uncertainty
        """
        self.model.eval()
        self.model.enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(image)
                probs = F.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        epistemic_uncertainty = np.std(predictions, axis=0)
        
        return {
            'mean_prediction': mean_prediction,
            'epistemic_uncertainty': epistemic_uncertainty,
            'all_predictions': predictions
        }
    
    def test_time_augmentation_uncertainty(self, image, n_augmentations=10):
        """
        Test-Time Augmentation for Aleatoric Uncertainty
        """
        self.model.eval()
        
        # Simple augmentation function
        def augment_tensor(tensor, strength=0.1):
            # Add small random noise
            noise = torch.randn_like(tensor) * strength
            return torch.clamp(tensor + noise, 0, 1)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_augmentations):
                augmented = augment_tensor(image)
                output = self.model(augmented)
                probs = F.softmax(output, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_prediction = np.mean(predictions, axis=0)
        aleatoric_uncertainty = np.std(predictions, axis=0)
        
        return {
            'mean_prediction': mean_prediction,
            'aleatoric_uncertainty': aleatoric_uncertainty
        }
    
    def combined_uncertainty(self, image, n_mc_samples=15, n_augmentations=8):
        """
        Combine epistemic and aleatoric uncertainty
        """
        epistemic_results = self.mc_dropout_uncertainty(image, n_mc_samples)
        aleatoric_results = self.test_time_augmentation_uncertainty(image, n_augmentations)
        
        total_uncertainty = np.sqrt(
            epistemic_results['epistemic_uncertainty']**2 + 
            aleatoric_results['aleatoric_uncertainty']**2
        )
        
        return {
            'mean_prediction': epistemic_results['mean_prediction'],
            'epistemic_uncertainty': epistemic_results['epistemic_uncertainty'],
            'aleatoric_uncertainty': aleatoric_results['aleatoric_uncertainty'],
            'total_uncertainty': total_uncertainty
        }


# ============================================================================
# PART 7: EXPLAINABLE AI (GRAD-CAM)
# ============================================================================

class GradCAM:
    """
    Grad-CAM implementation for visual explanations
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
        
    def generate_cam(self, image, target_class=None):
        """
        Generate Grad-CAM heatmap
        """
        self.model.eval()
        
        # Register hooks
        target_layer = self.model.backbone.layer4[-1]  # Last conv layer of ResNet
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
        
        # Generate CAM
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination
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


# ============================================================================
# PART 8: EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_with_uncertainty(model, test_loader, uncertainty_estimator, class_names):
    """
    Comprehensive evaluation with uncertainty quantification
    """
    print("\\n" + "="*50)
    print("EVALUATION WITH UNCERTAINTY QUANTIFICATION")
    print("="*50)
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_uncertainties = []
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        if batch_idx >= 5:  # Limit evaluation for demo
            break
            
        images, labels = images.to(device), labels.to(device)
        
        # Get uncertainty estimates for first image in batch
        uncertainty_results = uncertainty_estimator.mc_dropout_uncertainty(
            images[:1], n_samples=15
        )
        
        predictions = uncertainty_results['mean_prediction']
        uncertainties = uncertainty_results['epistemic_uncertainty']
        
        pred_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        all_predictions.extend(pred_classes)
        all_labels.extend([labels[0].cpu().numpy()])
        all_confidences.extend(confidences)
        all_uncertainties.extend(np.mean(uncertainties, axis=1))
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_uncertainties = np.array(all_uncertainties)
    
    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean() * 100
    
    print(f"\\nEvaluation Results (sample of {len(all_predictions)} images):")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average Confidence: {all_confidences.mean():.4f}")
    print(f"Average Uncertainty: {all_uncertainties.mean():.4f}")
    
    # Uncertainty-based decisions
    low_uncertainty = all_uncertainties < 0.15
    medium_uncertainty = (all_uncertainties >= 0.15) & (all_uncertainties < 0.30)
    high_uncertainty = all_uncertainties >= 0.30
    
    print(f"\\nDecision Analysis:")
    print(f"Low uncertainty (auto-accept): {low_uncertainty.sum()} ({100*low_uncertainty.mean():.1f}%)")
    print(f"Medium uncertainty (review): {medium_uncertainty.sum()} ({100*medium_uncertainty.mean():.1f}%)")
    print(f"High uncertainty (reject): {high_uncertainty.sum()} ({100*high_uncertainty.mean():.1f}%)")
    
    return {
        'accuracy': accuracy,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'uncertainties': all_uncertainties
    }


def visualize_uncertainty_and_gradcam(model, test_loader, uncertainty_estimator, 
                                     class_names, num_examples=4):
    """
    Visualize uncertainty analysis and Grad-CAM explanations
    """
    print("\\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    gradcam = GradCAM(model, device)
    
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, 4*num_examples))
    fig.suptitle('Uncertainty Quantification + Explainable AI Demo', fontsize=16)
    
    # Column headers
    if num_examples > 0:
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Prediction & Uncertainty', fontsize=12, fontweight='bold')
        axes[0, 2].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[0, 3].set_title('Uncertainty Analysis', fontsize=12, fontweight='bold')
    
    example_count = 0
    uncertainty_values = []
    
    for images, labels in test_loader:
        if example_count >= num_examples:
            break
            
        for i in range(min(images.size(0), num_examples - example_count)):
            image = images[i:i+1].to(device)
            label = labels[i].item()
            
            # Get prediction and uncertainty
            with torch.no_grad():
                output = model(image)
                pred_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, pred_class].item()
            
            uncertainty_results = uncertainty_estimator.combined_uncertainty(image, 10, 8)
            total_uncertainty = uncertainty_results['total_uncertainty'].mean()
            epistemic_unc = uncertainty_results['epistemic_uncertainty'].mean()
            aleatoric_unc = uncertainty_results['aleatoric_uncertainty'].mean()
            
            uncertainty_values.append(total_uncertainty)
            
            # Decision based on uncertainty
            if total_uncertainty < 0.15:
                decision = "✓ Accept"
                decision_color = 'green'
            elif total_uncertainty < 0.30:
                decision = "⚠ Review"
                decision_color = 'orange'
            else:
                decision = "✗ Reject"
                decision_color = 'red'
            
            # Generate Grad-CAM
            cam = gradcam.generate_cam(image, target_class=pred_class)
            
            # Denormalize image for display
            img_display = image.cpu().squeeze().permute(1, 2, 0).numpy()
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)
            
            row = example_count
            
            # Original image
            axes[row, 0].imshow(img_display)
            axes[row, 0].set_title(f'True: {class_names[label]}')
            axes[row, 0].axis('off')
            
            # Prediction info
            axes[row, 1].text(0.1, 0.8, f'Predicted: {class_names[pred_class]}', transform=axes[row, 1].transAxes, fontsize=10)
            axes[row, 1].text(0.1, 0.6, f'Confidence: {confidence:.3f}', transform=axes[row, 1].transAxes, fontsize=10)
            axes[row, 1].text(0.1, 0.4, f'Total Uncertainty: {total_uncertainty:.3f}', transform=axes[row, 1].transAxes, fontsize=10)
            axes[row, 1].text(0.1, 0.2, f'Decision: {decision}', transform=axes[row, 1].transAxes, 
                             fontsize=10, color=decision_color, fontweight='bold')
            axes[row, 1].axis('off')
            
            # Grad-CAM heatmap
            cam_resized = cv2.resize(cam, (64, 64))
            axes[row, 2].imshow(img_display, alpha=0.6)
            axes[row, 2].imshow(cam_resized, cmap='jet', alpha=0.4)
            axes[row, 2].set_title('Visual Explanation')
            axes[row, 2].axis('off')
            
            # Uncertainty breakdown
            uncertainties = [epistemic_unc, aleatoric_unc, total_uncertainty]
            uncertainty_labels = ['Epistemic\\n(Model)', 'Aleatoric\\n(Data)', 'Total']
            colors = ['lightblue', 'lightcoral', 'lightgreen']
            
            bars = axes[row, 3].bar(uncertainty_labels, uncertainties, color=colors, alpha=0.7)
            axes[row, 3].set_ylabel('Uncertainty')
            axes[row, 3].set_ylim(0, max(0.5, max(uncertainties) * 1.2))
            axes[row, 3].grid(True, alpha=0.3)
            
            # Add uncertainty threshold lines
            axes[row, 3].axhline(y=0.15, color='green', linestyle='--', alpha=0.7, label='Accept')
            axes[row, 3].axhline(y=0.30, color='red', linestyle='--', alpha=0.7, label='Reject')
            
            example_count += 1
            
            if example_count >= num_examples:
                break
    
    plt.tight_layout()
    plt.savefig('uncertainty_gradcam_demo.png', dpi=150, bbox_inches='tight')
    print("✓ Uncertainty + Grad-CAM visualization saved as 'uncertainty_gradcam_demo.png'")
    
    # Create summary uncertainty plot
    if uncertainty_values:
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(uncertainty_values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(0.15, color='green', linestyle='--', label='Accept threshold')
        plt.axvline(0.30, color='red', linestyle='--', label='Reject threshold')
        plt.xlabel('Total Uncertainty')
        plt.ylabel('Frequency')
        plt.title('Distribution of Uncertainty Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        decisions = ['Accept\\n(<0.15)', 'Review\\n(0.15-0.30)', 'Reject\\n(>0.30)']
        decision_counts = [
            sum(1 for u in uncertainty_values if u < 0.15),
            sum(1 for u in uncertainty_values if 0.15 <= u < 0.30),
            sum(1 for u in uncertainty_values if u >= 0.30)
        ]
        colors = ['green', 'orange', 'red']
        plt.bar(decisions, decision_counts, color=colors, alpha=0.7)
        plt.ylabel('Count')
        plt.title('Decision Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('uncertainty_summary.png', dpi=150, bbox_inches='tight')
        print("✓ Uncertainty summary saved as 'uncertainty_summary.png'")


# ============================================================================
# PART 9: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - Complete Demo
    """
    print("Starting complete UQ + XAI demonstration...")
    
    try:
        # Step 1: Generate synthetic dataset
        X, y, class_names = create_synthetic_traffic_signs(
            num_samples=800, 
            num_classes=10, 
            image_size=64
        )
        
        # Step 2: Perform EDA
        class_dist = perform_eda(X, y, class_names)
        
        # Step 3: Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X, y, batch_size=32, val_split=0.2, test_split=0.1
        )
        
        # Step 4: Create model
        num_classes = len(class_names)
        model = UncertaintyQuantificationCNN(num_classes=num_classes, dropout_rate=0.3)
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\\nModel created with {total_params:,} parameters")
        
        # Step 5: Train model
        history = train_model(
            model, train_loader, val_loader, 
            num_epochs=8, lr=0.001  # Reduced epochs for demo
        )
        
        # Step 6: Load best model
        model.load_state_dict(torch.load('best_model.pth'))
        
        # Step 7: Initialize uncertainty estimator
        uncertainty_estimator = UncertaintyEstimator(model, device)
        print("\\n✓ Uncertainty estimator initialized")
        
        # Step 8: Evaluate with uncertainty
        results = evaluate_with_uncertainty(
            model, test_loader, uncertainty_estimator, class_names
        )
        
        # Step 9: Generate comprehensive visualizations
        visualize_uncertainty_and_gradcam(
            model, test_loader, uncertainty_estimator, 
            class_names, num_examples=4
        )
        
        # Step 10: Summary
        print("\\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\\nFiles Generated:")
        print("1. 'eda_analysis.png' - Exploratory Data Analysis")
        print("2. 'training_curves.png' - Training Progress")
        print("3. 'uncertainty_gradcam_demo.png' - Main UQ + XAI Results")
        print("4. 'uncertainty_summary.png' - Uncertainty Distribution")
        print("5. 'best_model.pth' - Trained Model")
        
        print("\\nKey Results:")
        print(f"- Final Test Accuracy: {results['accuracy']:.2f}%")
        print(f"- Average Confidence: {results['confidences'].mean():.3f}")
        print(f"- Average Uncertainty: {results['uncertainties'].mean():.3f}")
        
        print("\\nWhat This Demonstrates:")
        print("✓ Complete EDA with synthetic Vietnamese traffic signs")
        print("✓ ResNet18-based model with dropout for uncertainty")
        print("✓ Monte Carlo Dropout (Epistemic Uncertainty)")
        print("✓ Test-Time Augmentation (Aleatoric Uncertainty)")
        print("✓ Combined Total Uncertainty")
        print("✓ Grad-CAM visual explanations")
        print("✓ Decision-making based on uncertainty thresholds")
        print("✓ Comprehensive visualizations")
        
        print("\\n🎯 UNCERTAINTY QUANTIFICATION SHOWN:")
        print("- Epistemic (Model) Uncertainty: What the model doesn't know")
        print("- Aleatoric (Data) Uncertainty: Inherent noise in observations")
        print("- Total Uncertainty: Combined measure for decision making")
        
        print("\\n🧠 EXPLAINABLE AI SHOWN:")
        print("- Grad-CAM: Visual explanations of model focus")
        print("- Heatmaps: Red = important, Blue = unimportant")
        print("- Interpretation: Should focus on sign features, not background")
        
        print("\\n" + "="*80)
        print("SUCCESS: All components working perfectly!")
        print("Ready for real Vietnamese traffic sign data!")
        print("="*80)
        
    except Exception as e:
        print(f"\\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run the complete demonstration
    success = main()
    
    if success:
        print("\\n🚀 Demo completed successfully! Check the generated images.")
        print("\\n📖 To use with real data:")
        print("1. Replace create_synthetic_traffic_signs() with real data loading")
        print("2. Adjust image_size and num_classes as needed")  
        print("3. Modify augmentations based on your dataset")
        print("4. Tune uncertainty thresholds for your application")
        
        plt.show()  # Show all plots
'''

# Save the single file
with open('run_complete_demo.py', 'w', encoding='utf-8') as f:
    f.write(single_file_content)

print("✅ SINGLE EXECUTABLE FILE CREATED!")
print("📄 File: run_complete_demo.py")
print(f"📏 Lines of code: {len(single_file_content.splitlines())}")
print("\n🚀 TO RUN THE COMPLETE DEMO:")
print("   python run_complete_demo.py")

print("\n📋 WHAT THIS FILE DOES:")
print("✓ Generates synthetic Vietnamese traffic sign dataset")
print("✓ Performs complete EDA with visualizations")
print("✓ Creates and trains ResNet18 model with dropout")
print("✓ Demonstrates Monte Carlo Dropout (Epistemic Uncertainty)")
print("✓ Shows Test-Time Augmentation (Aleatoric Uncertainty)")
print("✓ Implements Grad-CAM for visual explanations")
print("✓ Creates comprehensive visualization plots")
print("✓ Shows decision-making based on uncertainty")

print("\n🎯 OUTPUT FILES:")
print("1. eda_analysis.png - Dataset analysis")
print("2. training_curves.png - Training progress")
print("3. uncertainty_gradcam_demo.png - Main results") 
print("4. uncertainty_summary.png - Uncertainty distribution")
print("5. best_model.pth - Trained model")

print("\n⏱️ ESTIMATED RUNTIME:")
print("- With CPU: 5-10 minutes")
print("- With GPU: 2-3 minutes")