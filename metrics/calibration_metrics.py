import numpy as np
from sklearn.metrics import log_loss

def calculate_ece(confidences, predictions, labels, num_bins=10):
    """
    Expected Calibration Error (ECE).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = predictions == labels
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def calculate_mce(confidences, predictions, labels, num_bins=10):
    """
    Maximum Calibration Error (MCE).
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = predictions == labels
    
    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
            
    return mce

def calculate_calibration_metrics(probabilities, labels):
    """
    Computes calibration metrics: ECE, MCE, Brier Score, NLL.
    """
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    
    ece = calculate_ece(confidences, predictions, labels)
    mce = calculate_mce(confidences, predictions, labels)
    
    # NLL is essentially Log Loss
    try:
        num_classes = probabilities.shape[1]
        nll = log_loss(labels, probabilities, labels=np.arange(num_classes))
    except ValueError:
        nll = float('nan') # In case of missing classes in a tiny batch
    
    # Brier score (multi-class version)
    # One-hot encode labels
    num_classes = probabilities.shape[1]
    labels_one_hot = np.eye(num_classes)[labels]
    brier = np.mean(np.sum((probabilities - labels_one_hot) ** 2, axis=1))
    
    return {
        'ece': ece,
        'mce': mce,
        'nll': nll,
        'brier': brier
    }

def count_model_parameters(model):
    """Return number of parameters and model size in MB"""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # rough approximation of size in MB assuming fp32 (4 bytes per parameter)
    size_mb = num_params * 4 / (1024 ** 2)
    
    return {
        'num_params': num_params,
        'trainable_params': num_trainable,
        'size_mb': size_mb
    }
