
# Create a comprehensive README and usage guide

readme_content = """
# Uncertainty Quantification in Explainable Vision Models
## Vietnamese Traffic Sign Recognition - Complete Pipeline

This repository contains a complete, end-to-end implementation for **Uncertainty Quantification (UQ)** and **Explainable AI (XAI)** applied to Vietnamese traffic sign recognition.

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Dataset Structure](#dataset-structure)
5. [Pipeline Components](#pipeline-components)
6. [Usage Guide](#usage-guide)
7. [Understanding Uncertainty Quantification](#understanding-uncertainty-quantification)
8. [Understanding Explainable AI](#understanding-explainable-ai)
9. [Results and Visualization](#results-and-visualization)
10. [References](#references)

---

## 🎯 Overview

This project implements state-of-the-art techniques for:

- **Uncertainty Quantification (UQ)**: Measuring how confident the model is in its predictions
  - **Epistemic Uncertainty**: Model uncertainty (knowledge uncertainty)
  - **Aleatoric Uncertainty**: Data uncertainty (inherent noise)
  
- **Explainable AI (XAI)**: Understanding why the model makes certain predictions
  - **Grad-CAM**: Visual explanations showing important image regions
  - **Saliency Maps**: Highlighting influential pixels
  
- **Complete ML Pipeline**: From data exploration to deployment-ready model

---

## ✨ Features

### 1. Exploratory Data Analysis (EDA)
- Class distribution analysis
- Image property statistics (size, brightness, aspect ratio)
- Data quality checks (corrupted images, missing data)
- Imbalance detection and quantification

### 2. Data Cleaning
- Corrupted image removal
- Class imbalance handling (oversampling/undersampling)
- Image normalization and standardization
- Quality assurance checks

### 3. Data Augmentation
- Random rotation and affine transformations
- Color jittering (brightness, contrast, saturation)
- Random cropping and flipping
- Normalization with ImageNet statistics

### 4. Model Architecture
- ResNet18 backbone (pre-trained on ImageNet)
- Custom classification head with dropout layers
- Dual output heads (mean + variance)
- Batch normalization for stability

### 5. Uncertainty Quantification Methods

#### A. Monte Carlo Dropout (Epistemic Uncertainty)
```
Epistemic uncertainty represents model uncertainty - what the model doesn't know.
- Enable dropout at test time
- Perform multiple forward passes (typically 20-30)
- Calculate mean and standard deviation of predictions
- Higher variance = Higher epistemic uncertainty
```

#### B. Test-Time Augmentation (Aleatoric Uncertainty)
```
Aleatoric uncertainty represents data uncertainty - inherent noise in observations.
- Apply random augmentations to input
- Perform multiple forward passes with different augmentations
- Calculate variance across predictions
- Higher variance = Higher aleatoric uncertainty
```

#### C. Combined Total Uncertainty
```
Total Uncertainty = √(Epistemic² + Aleatoric²)
- Provides comprehensive uncertainty estimate
- Helps identify when model predictions are unreliable
```

### 6. Explainable AI Methods

#### A. Grad-CAM (Gradient-weighted Class Activation Mapping)
```
Shows which regions of the image are important for the prediction:
1. Forward pass through the network
2. Backward pass to compute gradients
3. Weight activation maps by gradients
4. Generate heatmap showing important regions
5. Overlay heatmap on original image
```

#### B. Saliency Maps
```
Highlights pixels that strongly influence the prediction:
- Compute gradients of output with respect to input
- Visualize magnitude of gradients
- Shows pixel-level importance
```

### 7. Model Calibration
- Expected Calibration Error (ECE) metric
- Reliability diagrams
- Confidence vs accuracy analysis
- Calibration improvement techniques

---

## 🔧 Installation

### Requirements
```bash
# Python 3.8+
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install opencv-python pillow
pip install scikit-learn scipy
```

### Quick Start
```bash
# Clone or download the repository
git clone <your-repo-url>
cd vietnamese-traffic-sign-uq

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python vietnamese_traffic_sign_uq_xai_pipeline.py
```

---

## 📁 Dataset Structure

Expected structure for Vietnamese Traffic Signs dataset:

```
vietnamese-traffic-signs/
├── Train/
│   ├── class_0/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   ├── class_1/
│   └── ...
├── Test/
│   ├── class_0/
│   └── ...
├── Train.csv
└── Test.csv
```

CSV format:
```
Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId,Path
50,50,5,5,45,45,0,Train/class_0/image_001.jpg
```

---

## 🔬 Pipeline Components

### Component 1: Exploratory Data Analysis (EDA)

```python
# Initialize EDA
eda = TrafficSignEDA(data_path='./vietnamese-traffic-signs')
eda.load_dataset_info()

# Analyze class distribution
class_counts = eda.analyze_class_distribution(df)

# Analyze image properties
image_stats = eda.analyze_image_properties(image_paths, sample_size=100)

# Check data quality
quality_report = eda.check_data_quality(image_paths, sample_size=100)
```

**What it shows:**
- Number of classes and samples per class
- Class imbalance ratio
- Image dimensions (width, height, aspect ratio)
- Brightness statistics
- Data quality metrics

### Component 2: Data Cleaning

```python
# Initialize cleaner
cleaner = DataCleaner(target_size=(64, 64))

# Remove corrupted images
cleaned_df = cleaner.remove_corrupted_images(df, image_col='Path')

# Handle class imbalance
balanced_df = cleaner.handle_class_imbalance(cleaned_df, target_samples_per_class=200)

# Normalize images
normalized_image = cleaner.normalize_image_sizes(image)
```

**What it does:**
- Removes unreadable/corrupted images
- Balances class distribution
- Standardizes image sizes
- Normalizes pixel values to [0, 1]

### Component 3: Data Augmentation

```python
# Get augmentation transforms
train_transform = get_augmentation_transforms(phase='train')
val_transform = get_augmentation_transforms(phase='val')

# Create dataset
train_dataset = TrafficSignDataset(
    image_paths=train_paths,
    labels=train_labels,
    transform=train_transform
)
```

**Augmentation techniques:**
- Random rotation (±15 degrees)
- Random translation (±10%)
- Color jittering (brightness, contrast, saturation)
- Normalization with ImageNet statistics

### Component 4: Model Training

```python
# Create model
model = UncertaintyQuantificationCNN(
    num_classes=29,
    dropout_rate=0.3
)
model = model.to(device)

# Train model
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.CrossEntropyLoss(),
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    num_epochs=20,
    device=device
)
```

**Model architecture:**
- ResNet18 backbone (pre-trained)
- Custom classification head
- Multiple dropout layers for UQ
- Batch normalization layers

### Component 5: Uncertainty Estimation

```python
# Initialize uncertainty estimator
uncertainty_estimator = UncertaintyEstimator(model, device)

# Get epistemic uncertainty (MC Dropout)
epistemic_results = uncertainty_estimator.mc_dropout_uncertainty(
    image, n_samples=30
)

# Get aleatoric uncertainty (Test-Time Augmentation)
aleatoric_results = uncertainty_estimator.test_time_augmentation_uncertainty(
    image, n_augmentations=10
)

# Get combined uncertainty
combined_results = uncertainty_estimator.combined_uncertainty(
    image, n_mc_samples=20, n_augmentations=10
)
```

**Uncertainty outputs:**
- Mean prediction (class probabilities)
- Epistemic uncertainty (model uncertainty)
- Aleatoric uncertainty (data uncertainty)
- Total uncertainty (combined)
- Predictive entropy

### Component 6: Explainable AI

```python
# Initialize XAI
xai = ExplainableAI(model, device)

# Generate Grad-CAM explanation
cam = xai.grad_cam(image, target_class=predicted_class)

# Visualize Grad-CAM
original_img, heatmap, overlay = xai.visualize_grad_cam(image, cam, alpha=0.5)
```

**XAI outputs:**
- Grad-CAM heatmap (which regions are important)
- Overlay visualization (heatmap on original image)
- Class-specific explanations

### Component 7: Evaluation with Uncertainty

```python
# Evaluate model with uncertainty quantification
results = evaluate_with_uncertainty(
    model=model,
    test_loader=test_loader,
    uncertainty_estimator=uncertainty_estimator,
    device=device
)

# Results include:
# - Accuracy
# - Expected Calibration Error (ECE)
# - Predictions, labels, confidences, uncertainties
```

**Evaluation metrics:**
- Classification accuracy
- Expected Calibration Error (ECE)
- Confidence statistics
- Uncertainty statistics
- Reliability diagrams

### Component 8: Visualization

```python
# Visualize uncertainty analysis
fig1 = visualize_uncertainty_analysis(
    results=results,
    save_path='uncertainty_analysis.png'
)

# Visualize Grad-CAM examples
fig2 = visualize_grad_cam_examples(
    model=model,
    xai=xai,
    test_loader=test_loader,
    device=device,
    num_examples=6,
    save_path='grad_cam_examples.png'
)
```

**Visualization outputs:**
- Confidence vs uncertainty scatter plot
- Uncertainty distribution histogram
- Reliability diagram
- Grad-CAM overlays
- Correct vs incorrect prediction analysis

---

## 📖 Usage Guide

### Complete Example

```python
import torch
from torch.utils.data import DataLoader
import pandas as pd

# 1. Load dataset
data_path = './vietnamese-traffic-signs'
train_df = pd.read_csv(f'{data_path}/Train.csv')
test_df = pd.read_csv(f'{data_path}/Test.csv')

# 2. Perform EDA
eda = TrafficSignEDA(data_path)
class_counts = eda.analyze_class_distribution(train_df)

# 3. Clean data
cleaner = DataCleaner(target_size=(64, 64))
train_df_cleaned = cleaner.remove_corrupted_images(train_df)
train_df_balanced = cleaner.handle_class_imbalance(train_df_cleaned)

# 4. Create datasets and loaders
train_transform = get_augmentation_transforms(phase='train')
test_transform = get_augmentation_transforms(phase='test')

train_dataset = TrafficSignDataset(
    image_paths=train_df_balanced['Path'].values,
    labels=train_df_balanced['ClassId'].values,
    transform=train_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 5. Create and train model
num_classes = train_df['ClassId'].nunique()
model = UncertaintyQuantificationCNN(num_classes=num_classes)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=20,
    device=device
)

# 6. Evaluate with uncertainty
uncertainty_estimator = UncertaintyEstimator(model, device)
results = evaluate_with_uncertainty(
    model=model,
    test_loader=test_loader,
    uncertainty_estimator=uncertainty_estimator,
    device=device
)

# 7. Generate explanations
xai = ExplainableAI(model, device)
visualize_grad_cam_examples(
    model=model,
    xai=xai,
    test_loader=test_loader,
    device=device
)

# 8. Visualize results
visualize_uncertainty_analysis(results)
```

---

## 🔍 Understanding Uncertainty Quantification

### What is Uncertainty?

Uncertainty quantification helps answer: **"How confident should we be in the model's prediction?"**

### Types of Uncertainty

#### 1. Epistemic Uncertainty (Model Uncertainty)
- **Definition**: Uncertainty due to lack of knowledge in the model
- **Cause**: Insufficient training data, model limitations
- **Can be reduced**: Yes, with more training data
- **Example**: Model uncertain about a traffic sign type it rarely saw during training

**Mathematical Formula:**
```
Epistemic Uncertainty = Var[E[y|x, θ]]
where θ ~ p(θ|D) (distribution over model parameters)
```

**Implementation: Monte Carlo Dropout**
```python
# Perform N forward passes with dropout enabled
predictions = []
for i in range(N):
    pred = model_with_dropout(image)
    predictions.append(pred)

epistemic_uncertainty = np.std(predictions, axis=0)
```

#### 2. Aleatoric Uncertainty (Data Uncertainty)
- **Definition**: Uncertainty inherent in the data
- **Cause**: Sensor noise, occlusion, blur, poor lighting
- **Cannot be reduced**: Even with more data
- **Example**: Blurry traffic sign in fog - inherently uncertain

**Mathematical Formula:**
```
Aleatoric Uncertainty = E[Var[y|x, θ]]
where the variance is over the data distribution
```

**Implementation: Test-Time Augmentation**
```python
# Apply random augmentations and measure variance
predictions = []
for i in range(N):
    augmented_image = apply_augmentation(image)
    pred = model(augmented_image)
    predictions.append(pred)

aleatoric_uncertainty = np.std(predictions, axis=0)
```

#### 3. Total Uncertainty
```
Total Uncertainty = √(Epistemic² + Aleatoric²)
```

### When is Uncertainty High?

**High Epistemic Uncertainty:**
- Out-of-distribution samples
- Rare classes
- Edge cases not in training data

**High Aleatoric Uncertainty:**
- Low-quality images (blur, noise)
- Occlusions
- Poor lighting conditions
- Ambiguous signs

### Why is Uncertainty Important?

1. **Safety**: In autonomous driving, knowing when the model is uncertain is critical
2. **Reliability**: Helps identify when predictions should not be trusted
3. **Active Learning**: Focus data collection on uncertain samples
4. **Human-in-the-Loop**: Defer uncertain predictions to human experts
5. **Risk Assessment**: Quantify prediction reliability

---

## 🧠 Understanding Explainable AI

### What is Explainable AI?

XAI helps answer: **"Why did the model make this prediction?"**

### Grad-CAM (Gradient-weighted Class Activation Mapping)

**What it does:**
Shows which regions of the image are most important for the model's prediction.

**How it works:**
1. Forward pass: Get prediction
2. Backward pass: Compute gradients of target class w.r.t. last conv layer
3. Global average pooling: Get weights for each feature map
4. Weighted combination: Sum weighted feature maps
5. ReLU: Keep only positive contributions
6. Upsample: Resize to input image size
7. Overlay: Create heatmap visualization

**Mathematical Formula:**
```
L_Grad-CAM^c = ReLU(Σ_k α_k^c A^k)

where:
α_k^c = (1/Z) Σ_i Σ_j (∂y^c / ∂A_ij^k)  (global average pooling of gradients)
A^k = activation maps of layer k
y^c = score for class c
```

**Interpretation:**
- **Red regions**: Most important for prediction
- **Blue regions**: Least important for prediction
- **Yellow/Orange**: Moderately important

**Example:**
```
Input: Image of stop sign
Prediction: Stop sign (98% confidence)
Grad-CAM: Highlights the octagonal shape and red color
Interpretation: Model correctly focuses on shape and color
```

### Benefits of XAI

1. **Trust**: Understand model reasoning
2. **Debugging**: Identify when model focuses on wrong features
3. **Bias Detection**: Discover spurious correlations
4. **Regulatory Compliance**: Meet explainability requirements
5. **Model Improvement**: Identify weaknesses

---

## 📊 Results and Visualization

### Uncertainty Analysis Plots

1. **Confidence vs Uncertainty Scatter Plot**
   - Shows relationship between prediction confidence and uncertainty
   - Ideal: High confidence → Low uncertainty

2. **Uncertainty Distribution**
   - Histogram showing distribution of uncertainty values
   - Helps identify threshold for rejection

3. **Reliability Diagram**
   - Plots predicted confidence vs actual accuracy
   - Perfect calibration: diagonal line
   - Measures ECE (Expected Calibration Error)

4. **Correct vs Incorrect Predictions**
   - Box plot comparing uncertainty for correct and incorrect predictions
   - Incorrect predictions should have higher uncertainty

### Grad-CAM Visualizations

1. **Original Image**: Input traffic sign
2. **Heatmap**: Grad-CAM activation map
3. **Overlay**: Heatmap superimposed on original image

**Good Explanation:**
- Focuses on relevant sign features (shape, color, symbols)
- Ignores background

**Bad Explanation:**
- Focuses on background
- Ignores sign features
- May indicate model using wrong cues

---

## 📈 Expected Calibration Error (ECE)

### What is Calibration?

A model is **well-calibrated** if its predicted probabilities match actual frequencies.

Example:
- If model predicts 80% confidence for class A
- Then 80% of those predictions should be correct

### Expected Calibration Error (ECE)

**Formula:**
```
ECE = Σ_m (|B_m|/n) |acc(B_m) - conf(B_m)|

where:
B_m = predictions in bin m
acc(B_m) = accuracy in bin m
conf(B_m) = average confidence in bin m
n = total number of predictions
```

**Interpretation:**
- **ECE = 0**: Perfect calibration
- **ECE < 0.05**: Well-calibrated
- **ECE > 0.15**: Poorly calibrated

### How to Improve Calibration

1. **Temperature Scaling**: Scale logits before softmax
2. **Label Smoothing**: Smooth one-hot labels
3. **Focal Loss**: Down-weight easy examples
4. **Calibration Methods**: Platt scaling, isotonic regression

---

## 🎯 Practical Applications

### Use Cases

1. **Autonomous Vehicles**
   - High certainty: Proceed normally
   - High uncertainty: Request human intervention

2. **Medical Diagnosis**
   - High certainty: Automated diagnosis
   - High uncertainty: Consult specialist

3. **Quality Control**
   - High certainty: Automatic pass/fail
   - High uncertainty: Manual inspection

### Decision Thresholds

Example thresholds for traffic sign recognition:

```python
# Define uncertainty thresholds
LOW_UNCERTAINTY = 0.1
MEDIUM_UNCERTAINTY = 0.2
HIGH_UNCERTAINTY = 0.3

# Decision logic
if uncertainty < LOW_UNCERTAINTY:
    action = "Accept prediction"
elif uncertainty < MEDIUM_UNCERTAINTY:
    action = "Flag for review"
else:
    action = "Reject - require human review"
```

---

## 📚 References

### Uncertainty Quantification

1. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. ICML.

2. Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? NeurIPS.

3. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS.

4. Wang, G., et al. (2019). Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks. Neurocomputing.

### Explainable AI

5. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV.

6. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. KDD.

7. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.

### Calibration

8. Guo, C., et al. (2017). On calibration of modern neural networks. ICML.

9. Nixon, J., et al. (2019). Measuring calibration in deep learning. CVPR Workshops.

10. Conde, P., et al. (2023). Approaching test time augmentation in the context of uncertainty calibration for deep neural networks. arXiv.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

---

## 📝 License

This project is licensed under the MIT License.

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- Vietnamese Traffic Signs Dataset creators
- PyTorch community
- Research papers and authors cited above
- Open-source computer vision community

---

**Happy Training! 🚀**
"""

# Save README
with open('README_Vietnamese_Traffic_Sign_UQ_XAI.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("✓ Comprehensive README created successfully!")
print(f"✓ File saved as: README_Vietnamese_Traffic_Sign_UQ_XAI.md")
print(f"✓ Total lines: {len(readme_content.splitlines())}")
