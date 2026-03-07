import torch
import torch.nn as nn

class MCDropoutWrapper:
    """Wrapper to estimate epistemic uncertainty using Monte Carlo Dropout."""
    def __init__(self, model: nn.Module, num_samples: int = 30):
        self.model = model
        self.num_samples = num_samples

    def predict(self, x: torch.Tensor):
        """
        Runs multiple forward passes with dropout enabled.
        Returns: predictions of shape [num_samples, batch_size, num_classes]
        """
        self.model.eval()
        
        # Explicitly enable dropout modules
        for m in self.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
                
        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                out = self.model(x)
                probs = torch.softmax(out, dim=1)
                predictions.append(probs.unsqueeze(0))
                
        # [num_samples, batch_size, num_classes]
        predictions = torch.cat(predictions, dim=0)
        return predictions

    def get_uncertainty_metrics(self, predictions: torch.Tensor):
        """
        Calculates expected probabilities, predictive entropy, and variation.
        predictions: [num_samples, batch_size, num_classes]
        """
        # Expected probability (mean over stochastic passes)
        expected_p = predictions.mean(dim=0) # [batch_size, num_classes]
        
        # Predictive entropy: -sum(p * log(p))
        entropy = -torch.sum(expected_p * torch.log(expected_p + 1e-12), dim=1) # [batch_size]
        
        # Variance of predicted probabilities
        variance = predictions.var(dim=0) # [batch_size, num_classes]
        
        return expected_p, entropy, variance
