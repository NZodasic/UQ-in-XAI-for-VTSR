import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class MCDropoutUQ:
    """Class to quantify predictive uncertainty using Monte Carlo Dropout."""
    def __init__(self, model, dataloader, device, num_samples=20):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_samples = num_samples

    def compute_predictive_uncertainty(self):
        """
        Runs stochastic forward passes on the dataset.
        Returns mean predictive entropy and variance.
        """
        module_states = {id(m): (m, m.training) for m in self.model.modules()}

        all_entropies = []
        all_variances = []
        
        try:
            self.model.eval()
            for m in self.model.modules():
                if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)):
                    m.eval()
                elif isinstance(m, nn.modules.dropout._DropoutNd):
                    m.train()

            with torch.no_grad():
                for inputs, _ in tqdm(self.dataloader, desc="UQ Analysis (MC Dropout)"):
                    inputs = inputs.to(self.device)
                    batch_preds = []

                    for _ in range(self.num_samples):
                        out = torch.softmax(self.model(inputs), dim=1)
                        batch_preds.append(out.unsqueeze(0))

                    # [num_samples, batch_size, num_classes]
                    batch_preds = torch.cat(batch_preds, dim=0)

                    # Expected probability
                    expected_p = batch_preds.mean(dim=0)

                    # Predictive Entropy
                    entropy = -torch.sum(expected_p * torch.log(expected_p + 1e-12), dim=1)
                    all_entropies.append(entropy.cpu().numpy())

                    # Predictive Variance (mean variance across classes)
                    variance = batch_preds.var(dim=0, unbiased=False).mean(dim=1)
                    all_variances.append(variance.cpu().numpy())
        finally:
            for module, was_training in module_states.values():
                module.train(was_training)
        
        entropies = np.concatenate(all_entropies)
        variances = np.concatenate(all_variances)
        
        return {
            'mean_entropy': float(np.mean(entropies)),
            'std_entropy': float(np.std(entropies)),
            'mean_variance': float(np.mean(variances)),
            'std_variance': float(np.std(variances))
        }
