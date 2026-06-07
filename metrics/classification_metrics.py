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
            present_classes = np.unique(targets)
            
            if len(present_classes) == num_classes:
                metrics['auc'] = roc_auc_score(targets, probabilities, multi_class='ovr')
            elif len(present_classes) > 1:
                # filter and re-normalize probabilities for present classes only
                probs_present = probabilities[:, present_classes]
                row_sums = probs_present.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1e-7
                probs_present = probs_present / row_sums
                
                if len(present_classes) == 2:
                    y_true_binary = (targets == present_classes[1]).astype(int)
                    metrics['auc'] = roc_auc_score(y_true_binary, probs_present[:, 1])
                else:
                    targets_mapped = np.array([np.where(present_classes == val)[0][0] for val in targets])
                    metrics['auc'] = roc_auc_score(targets_mapped, probs_present, multi_class='ovr')
        except ValueError:
            # Handles issues like missing classes in the test set
            pass
            
    return metrics
