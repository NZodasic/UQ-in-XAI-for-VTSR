
# QUICK START GUIDE
# Vietnamese Traffic Sign Recognition with UQ & XAI

## 🚀 STEP-BY-STEP IMPLEMENTATION

### Step 1: Setup and Data Loading (5 minutes)

```python
import torch
import pandas as pd
from pathlib import Path

# Load dataset
DATA_PATH = './vietnamese-traffic-signs'
train_df = pd.read_csv(f'{DATA_PATH}/Train.csv')
test_df = pd.read_csv(f'{DATA_PATH}/Test.csv')

# Basic info
print(f"Training samples: {len(train_df)}")
print(f"Number of classes: {train_df['ClassId'].nunique()}")
```

---

### Step 2: Exploratory Data Analysis (10 minutes)

```python
from vietnamese_traffic_sign_uq_xai_pipeline import TrafficSignEDA

# Initialize EDA
eda = TrafficSignEDA(DATA_PATH)
eda.load_dataset_info()

# Analyze distribution
class_counts = eda.analyze_class_distribution(train_df)

# Check image properties
image_paths = [Path(DATA_PATH) / p for p in train_df['Path']]
stats = eda.analyze_image_properties(image_paths, sample_size=100)

# Quality check
quality = eda.check_data_quality(image_paths, sample_size=100)
```

**Expected Output:**
```
Number of unique classes: 29
Total number of images: 3200
Class imbalance ratio: 3.45
Mean samples per class: 110.34
Valid images: 98 (98.0%)
```

---

### Step 3: Data Cleaning (10 minutes)

```python
from vietnamese_traffic_sign_uq_xai_pipeline import DataCleaner

# Initialize cleaner
cleaner = DataCleaner(target_size=(64, 64))

# Remove corrupted images
train_df_clean = cleaner.remove_corrupted_images(train_df, image_col='Path')

# Handle class imbalance (optional)
# Option A: Balance to median
train_df_balanced = cleaner.handle_class_imbalance(train_df_clean)

# Option B: Balance to specific number
# train_df_balanced = cleaner.handle_class_imbalance(
#     train_df_clean, 
#     target_samples_per_class=150
# )
```

---

### Step 4: Create Data Loaders (15 minutes)

```python
from torch.utils.data import DataLoader
from vietnamese_traffic_sign_uq_xai_pipeline import (
    TrafficSignDataset, 
    get_augmentation_transforms
)
from sklearn.model_selection import train_test_split

# Split data
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_df_balanced['Path'].values,
    train_df_balanced['ClassId'].values,
    test_size=0.2,
    random_state=42,
    stratify=train_df_balanced['ClassId']
)

# Create datasets
train_dataset = TrafficSignDataset(
    image_paths=[Path(DATA_PATH) / p for p in train_paths],
    labels=train_labels,
    transform=get_augmentation_transforms(phase='train')
)

val_dataset = TrafficSignDataset(
    image_paths=[Path(DATA_PATH) / p for p in val_paths],
    labels=val_labels,
    transform=get_augmentation_transforms(phase='val')
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
```

---

### Step 5: Create Model (5 minutes)

```python
from vietnamese_traffic_sign_uq_xai_pipeline import UncertaintyQuantificationCNN

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create model
num_classes = train_df['ClassId'].nunique()
model = UncertaintyQuantificationCNN(
    num_classes=num_classes,
    dropout_rate=0.3  # For uncertainty quantification
)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

---

### Step 6: Train Model (30-60 minutes depending on GPU)

```python
from vietnamese_traffic_sign_uq_xai_pipeline import train_model

# Setup training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=20,
    device=device
)

# Plot training curves
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_losses'], label='Train Loss')
plt.plot(history['val_losses'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history['train_accs'], label='Train Acc')
plt.plot(history['val_accs'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300)
```

---

### Step 7: Uncertainty Quantification (15 minutes)

```python
from vietnamese_traffic_sign_uq_xai_pipeline import UncertaintyEstimator

# Load best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Initialize uncertainty estimator
uncertainty_estimator = UncertaintyEstimator(model, device)

# Test on a single image
# Get first batch from test loader
test_loader = DataLoader(
    TrafficSignDataset(
        image_paths=[Path(DATA_PATH) / p for p in test_df['Path']],
        labels=test_df['ClassId'].values,
        transform=get_augmentation_transforms(phase='val')
    ),
    batch_size=1,
    shuffle=False
)

for images, labels in test_loader:
    images = images.to(device)

    # Get epistemic uncertainty (MC Dropout)
    epistemic_results = uncertainty_estimator.mc_dropout_uncertainty(
        images, n_samples=30
    )

    print("\n=== EPISTEMIC UNCERTAINTY (Model Uncertainty) ===")
    print(f"Mean prediction shape: {epistemic_results['mean_prediction'].shape}")
    print(f"Predicted class: {epistemic_results['mean_prediction'].argmax()}")
    print(f"Prediction confidence: {epistemic_results['mean_prediction'].max():.4f}")
    print(f"Epistemic uncertainty: {epistemic_results['epistemic_uncertainty'].mean():.4f}")
    print(f"Predictive entropy: {epistemic_results['predictive_entropy'][0]:.4f}")

    # Get aleatoric uncertainty (Test-Time Augmentation)
    aleatoric_results = uncertainty_estimator.test_time_augmentation_uncertainty(
        images, n_augmentations=10
    )

    print("\n=== ALEATORIC UNCERTAINTY (Data Uncertainty) ===")
    print(f"Aleatoric uncertainty: {aleatoric_results['aleatoric_uncertainty'].mean():.4f}")

    # Get combined uncertainty
    combined = uncertainty_estimator.combined_uncertainty(
        images, n_mc_samples=20, n_augmentations=10
    )

    print("\n=== TOTAL UNCERTAINTY ===")
    print(f"Total uncertainty: {combined['total_uncertainty'].mean():.4f}")

    # Decision making
    uncertainty_threshold_low = 0.1
    uncertainty_threshold_high = 0.2

    total_unc = combined['total_uncertainty'].mean()

    if total_unc < uncertainty_threshold_low:
        decision = "✓ ACCEPT - High confidence"
    elif total_unc < uncertainty_threshold_high:
        decision = "⚠ FLAG - Medium confidence, review recommended"
    else:
        decision = "✗ REJECT - Low confidence, human review required"

    print(f"\nDecision: {decision}")

    break  # Just test first image
```

---

### Step 8: Explainable AI (10 minutes)

```python
from vietnamese_traffic_sign_uq_xai_pipeline import ExplainableAI

# Initialize XAI
xai = ExplainableAI(model, device)

# Test on a single image
for images, labels in test_loader:
    images = images.to(device)

    # Get prediction
    with torch.no_grad():
        output = model(images)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()

    print(f"\n=== EXPLAINABLE AI (Grad-CAM) ===")
    print(f"True label: {labels[0].item()}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")

    # Generate Grad-CAM
    cam = xai.grad_cam(images, target_class=pred_class)

    # Visualize
    original, heatmap, overlay = xai.visualize_grad_cam(images, cam, alpha=0.5)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title(f'Original\nTrue: {labels[0].item()}')
    axes[0].axis('off')

    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title(f'Overlay\nPred: {pred_class} ({confidence:.2f})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('grad_cam_example.png', dpi=300)

    print("\nInterpretation:")
    print("- Red regions: Most important for prediction")
    print("- Check if model focuses on sign features (shape, color, symbols)")
    print("- If focusing on background, model may be using spurious correlations")

    break
```

---

### Step 9: Complete Evaluation (15 minutes)

```python
from vietnamese_traffic_sign_uq_xai_pipeline import (
    evaluate_with_uncertainty,
    visualize_uncertainty_analysis
)

# Comprehensive evaluation
results = evaluate_with_uncertainty(
    model=model,
    test_loader=test_loader,
    uncertainty_estimator=uncertainty_estimator,
    device=device
)

# Print results
print("\n=== EVALUATION RESULTS ===")
print(f"Test Accuracy: {results['accuracy']:.2f}%")
print(f"Expected Calibration Error (ECE): {results['ece']:.4f}")
print(f"Average Confidence: {results['confidences'].mean():.4f}")
print(f"Average Uncertainty: {results['uncertainties'].mean():.4f}")

# Visualize
fig = visualize_uncertainty_analysis(
    results=results,
    save_path='uncertainty_analysis.png'
)
```

---

### Step 10: Batch Prediction with Uncertainty (5 minutes)

```python
def predict_with_uncertainty(model, uncertainty_estimator, image_path, device):
    """
    Predict a single image with uncertainty quantification
    """
    import cv2
    from torchvision import transforms

    # Load and preprocess image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get uncertainty
    results = uncertainty_estimator.combined_uncertainty(
        image_tensor, n_mc_samples=20, n_augmentations=10
    )

    pred_class = results['mean_prediction'].argmax()
    confidence = results['mean_prediction'].max()
    total_uncertainty = results['total_uncertainty'].mean()

    return {
        'predicted_class': pred_class,
        'confidence': confidence,
        'epistemic_uncertainty': results['epistemic_uncertainty'].mean(),
        'aleatoric_uncertainty': results['aleatoric_uncertainty'].mean(),
        'total_uncertainty': total_uncertainty
    }

# Example usage
# result = predict_with_uncertainty(
#     model, uncertainty_estimator, 
#     'path/to/image.jpg', device
# )
```

---

## 📊 INTERPRETING RESULTS

### Uncertainty Levels

| Total Uncertainty | Interpretation | Action |
|------------------|----------------|--------|
| < 0.10 | Very Low | Accept automatically |
| 0.10 - 0.20 | Low | Accept with logging |
| 0.20 - 0.30 | Medium | Flag for review |
| 0.30 - 0.40 | High | Require human verification |
| > 0.40 | Very High | Reject, manual classification |

### Expected Calibration Error (ECE)

| ECE Value | Calibration Quality |
|-----------|---------------------|
| < 0.05 | Excellent |
| 0.05 - 0.10 | Good |
| 0.10 - 0.15 | Fair |
| > 0.15 | Poor (needs calibration) |

### Grad-CAM Interpretation

**Good XAI (Model is correct):**
- ✓ Focuses on sign shape (circle, triangle, octagon)
- ✓ Highlights sign color (red, blue, yellow)
- ✓ Emphasizes symbols/text on sign
- ✓ Ignores background

**Bad XAI (Model may be wrong):**
- ✗ Focuses on background
- ✗ Highlights irrelevant objects
- ✗ Ignores actual sign features
- ✗ May indicate spurious correlations

---

## 🎯 COMMON ISSUES AND SOLUTIONS

### Issue 1: High Training Loss
**Solution:**
- Reduce learning rate: `lr=0.0001`
- Add learning rate scheduler
- Check data augmentation (may be too aggressive)

### Issue 2: High Validation Loss (Overfitting)
**Solution:**
- Increase dropout rate: `dropout_rate=0.5`
- Add more data augmentation
- Reduce model complexity
- Use early stopping

### Issue 3: Poor Calibration (High ECE)
**Solution:**
- Apply temperature scaling
- Use label smoothing: `label_smoothing=0.1`
- Train longer
- Collect more diverse data

### Issue 4: Uncertainty Too High for All Predictions
**Solution:**
- Train longer (model not confident yet)
- Reduce dropout rate: `dropout_rate=0.2`
- Check if test data is too different from train data
- Verify model architecture

### Issue 5: Grad-CAM Focuses on Background
**Solution:**
- Check training data quality
- Verify labels are correct
- Add more training data
- Use attention mechanisms
- Crop images to focus on signs

---

## 💡 TIPS FOR BEST RESULTS

1. **Data Quality Matters**
   - Clean labels are critical
   - Diverse training data improves generalization
   - Balance classes for better performance

2. **Uncertainty Thresholds**
   - Tune thresholds based on application
   - Safety-critical: Use conservative thresholds
   - Consider cost of false positives/negatives

3. **Computational Cost**
   - MC Dropout: 20-30 samples is usually sufficient
   - Test-Time Augmentation: 10 samples works well
   - More samples = better estimates but slower

4. **Model Selection**
   - ResNet18: Good balance of speed and accuracy
   - ResNet50: Better accuracy, slower
   - MobileNet: Faster, slightly lower accuracy

5. **Monitoring**
   - Track both accuracy AND uncertainty
   - Log high-uncertainty predictions
   - Regularly review edge cases

---

## 📚 NEXT STEPS

1. **Improve Model**
   - Try different architectures (ResNet50, EfficientNet)
   - Ensemble multiple models
   - Use knowledge distillation

2. **Better Uncertainty**
   - Try Deep Ensembles
   - Implement Bayesian Neural Networks
   - Use evidential deep learning

3. **Advanced XAI**
   - Add LIME explanations
   - Implement SHAP values
   - Try attention visualization

4. **Production Deployment**
   - Optimize model (quantization, pruning)
   - Set up monitoring system
   - Create API endpoint
   - Build confidence dashboard

---

**Good luck with your Vietnamese Traffic Sign Recognition project! 🚀🚦**
