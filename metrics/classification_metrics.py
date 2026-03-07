import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

def calculate_classification_metrics(targets, predictions, probabilities=None):
    """
    Computes standard classification metrics.
    targets: 1D array of true labels
    predictions: 1D array of predicted labels
    probabilities: 2D array of predicted probabilities [n_samples, n_classes]
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(targets, predictions)
    metrics['precision'] = precision_score(targets, predictions, average='macro', zero_division=0)
    metrics['recall'] = recall_score(targets, predictions, average='macro', zero_division=0)
    metrics['f1'] = f1_score(targets, predictions, average='macro', zero_division=0)
    
    # Make confusion matrix available
    metrics['cm'] = confusion_matrix(targets, predictions)
    
    metrics['auc'] = float('nan')
    if probabilities is not None:
        try:
            # use one-vs-rest for multi-class AUC
            num_classes = probabilities.shape[1]
            if len(np.unique(targets)) == num_classes:
                metrics['auc'] = roc_auc_score(targets, probabilities, multi_class='ovr')
        except ValueError:
            # Handles issues like missing classes in the test set
            pass
            
    return metrics
