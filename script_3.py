
import pandas as pd

# Create comprehensive summary tables

# Table 1: UQ Methods Comparison
uq_methods = pd.DataFrame({
    'Method': [
        'Monte Carlo Dropout',
        'Test-Time Augmentation',
        'Deep Ensembles',
        'Bayesian Neural Networks',
        'Evidential Deep Learning'
    ],
    'Uncertainty Type': [
        'Epistemic',
        'Aleatoric',
        'Both',
        'Both',
        'Both'
    ],
    'Computational Cost': [
        'Medium (20-30 forward passes)',
        'Medium (10-20 augmentations)',
        'High (train multiple models)',
        'Very High (approximate inference)',
        'Low (single forward pass)'
    ],
    'Ease of Implementation': [
        'Easy (add dropout at test time)',
        'Easy (apply augmentations)',
        'Medium (train multiple models)',
        'Hard (complex inference)',
        'Medium (custom loss function)'
    ],
    'Accuracy': [
        'Good',
        'Good',
        'Excellent',
        'Excellent',
        'Very Good'
    ],
    'Best Use Case': [
        'Model uncertainty, out-of-distribution',
        'Data uncertainty, noisy inputs',
        'Critical applications, highest accuracy',
        'Principled uncertainty, research',
        'Single-pass uncertainty, production'
    ]
})

# Table 2: XAI Methods Comparison
xai_methods = pd.DataFrame({
    'Method': [
        'Grad-CAM',
        'LIME',
        'SHAP',
        'Integrated Gradients',
        'Attention Visualization'
    ],
    'Explanation Type': [
        'Visual (heatmap)',
        'Feature importance',
        'Feature importance',
        'Visual (attribution)',
        'Visual (attention weights)'
    ],
    'Model Requirement': [
        'CNN with conv layers',
        'Any model (model-agnostic)',
        'Any model (model-agnostic)',
        'Differentiable model',
        'Model with attention'
    ],
    'Computational Cost': [
        'Low (single forward+backward)',
        'High (many perturbations)',
        'High (many perturbations)',
        'Medium (multiple forward passes)',
        'Very Low (extract weights)'
    ],
    'Interpretability': [
        'High (visual, intuitive)',
        'Medium (feature-based)',
        'Medium (numerical values)',
        'High (pixel attribution)',
        'High (attention scores)'
    ],
    'Best For': [
        'Image classification',
        'Tabular and image data',
        'Global and local explanations',
        'Detailed pixel importance',
        'Transformer-based models'
    ]
})

# Table 3: Pipeline Components
pipeline_components = pd.DataFrame({
    'Component': [
        'EDA',
        'Data Cleaning',
        'Data Augmentation',
        'Model Architecture',
        'Training',
        'Uncertainty Quantification',
        'Explainable AI',
        'Evaluation',
        'Calibration'
    ],
    'Purpose': [
        'Understand data distribution and quality',
        'Remove errors, balance classes',
        'Increase data diversity, prevent overfitting',
        'Feature extraction and classification',
        'Learn from data',
        'Estimate prediction reliability',
        'Understand model decisions',
        'Measure performance',
        'Align confidence with accuracy'
    ],
    'Key Outputs': [
        'Class distribution, image stats, quality metrics',
        'Cleaned dataset, balanced classes',
        'Augmented images, normalized inputs',
        'Trained model parameters',
        'Loss curves, accuracy curves',
        'Epistemic, aleatoric, total uncertainty',
        'Grad-CAM heatmaps, feature importance',
        'Accuracy, precision, recall, F1',
        'ECE, reliability diagrams'
    ],
    'Approximate Time': [
        '10 minutes',
        '10 minutes',
        'During training',
        '5 minutes setup',
        '30-60 minutes',
        '15 minutes',
        '10 minutes',
        '15 minutes',
        '5 minutes'
    ]
})

# Table 4: Uncertainty Decision Thresholds
uncertainty_thresholds = pd.DataFrame({
    'Uncertainty Level': [
        'Very Low',
        'Low',
        'Medium',
        'High',
        'Very High'
    ],
    'Threshold Range': [
        '< 0.10',
        '0.10 - 0.20',
        '0.20 - 0.30',
        '0.30 - 0.40',
        '> 0.40'
    ],
    'Confidence Level': [
        'Very High (> 0.90)',
        'High (0.75 - 0.90)',
        'Medium (0.60 - 0.75)',
        'Low (0.50 - 0.60)',
        'Very Low (< 0.50)'
    ],
    'Recommended Action': [
        '✓ Accept automatically',
        '✓ Accept with logging',
        '⚠ Flag for review',
        '⚠ Require verification',
        '✗ Reject - Human required'
    ],
    'Use Case Example': [
        'Clear, well-lit traffic signs',
        'Standard conditions',
        'Slightly occluded or angled signs',
        'Poor visibility, unusual angles',
        'Heavily occluded, damaged, or unfamiliar signs'
    ],
    'Risk Level': [
        'Very Low',
        'Low',
        'Medium',
        'High',
        'Very High'
    ]
})

# Table 5: Performance Metrics
performance_metrics = pd.DataFrame({
    'Metric': [
        'Accuracy',
        'Precision',
        'Recall',
        'F1-Score',
        'Expected Calibration Error (ECE)',
        'Maximum Calibration Error (MCE)',
        'Average Uncertainty',
        'Predictive Entropy'
    ],
    'What It Measures': [
        'Overall correctness',
        'How many predictions are correct when model predicts positive',
        'How many actual positives are found',
        'Harmonic mean of precision and recall',
        'Average calibration error across bins',
        'Maximum calibration error in any bin',
        'Average model uncertainty',
        'Average prediction uncertainty (information theory)'
    ],
    'Good Value': [
        '> 0.90 (90%)',
        '> 0.85',
        '> 0.85',
        '> 0.85',
        '< 0.05',
        '< 0.10',
        '< 0.20',
        '< 1.0'
    ],
    'Formula': [
        '(TP + TN) / Total',
        'TP / (TP + FP)',
        'TP / (TP + FN)',
        '2 × (Precision × Recall) / (Precision + Recall)',
        'Σ |accuracy - confidence| × bin_proportion',
        'max |accuracy - confidence|',
        'mean(uncertainty)',
        '-Σ p(c) log p(c)'
    ]
})

# Save all tables
with pd.ExcelWriter('UQ_XAI_Reference_Tables.xlsx', engine='openpyxl') as writer:
    uq_methods.to_excel(writer, sheet_name='UQ Methods', index=False)
    xai_methods.to_excel(writer, sheet_name='XAI Methods', index=False)
    pipeline_components.to_excel(writer, sheet_name='Pipeline Components', index=False)
    uncertainty_thresholds.to_excel(writer, sheet_name='Decision Thresholds', index=False)
    performance_metrics.to_excel(writer, sheet_name='Performance Metrics', index=False)

print("✓ Reference tables created successfully!")
print(f"✓ File saved as: UQ_XAI_Reference_Tables.xlsx")
print(f"✓ Contains 5 sheets with comprehensive information")

# Also save as CSV for easy viewing
uq_methods.to_csv('table_1_uq_methods.csv', index=False)
xai_methods.to_csv('table_2_xai_methods.csv', index=False)
pipeline_components.to_csv('table_3_pipeline_components.csv', index=False)
uncertainty_thresholds.to_csv('table_4_uncertainty_thresholds.csv', index=False)
performance_metrics.to_csv('table_5_performance_metrics.csv', index=False)

print("\n✓ Individual CSV files also created for each table")

# Display summary
print("\n" + "="*80)
print("TABLES OVERVIEW")
print("="*80)

print("\n1. UQ Methods Comparison:")
print(uq_methods[['Method', 'Uncertainty Type', 'Ease of Implementation']].to_string(index=False))

print("\n2. XAI Methods Comparison:")
print(xai_methods[['Method', 'Explanation Type', 'Interpretability']].to_string(index=False))

print("\n3. Uncertainty Decision Thresholds:")
print(uncertainty_thresholds[['Uncertainty Level', 'Threshold Range', 'Recommended Action']].to_string(index=False))

print("\n" + "="*80)
