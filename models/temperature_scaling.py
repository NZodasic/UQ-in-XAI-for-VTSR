import torch
import torch.nn as nn
from torch.optim import LBFGS


class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration via Temperature Scaling.
    Learns a single scalar T on validation set after training is done.
    Divides logits by T before softmax → well-calibrated probabilities.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.05)

    def calibrate(self, model, val_loader, device, logger=None):
        """Fit temperature on validation set using NLL loss."""
        model.eval()
        self.to(device)

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                all_logits.append(logits)
                all_labels.append(labels.to(device))

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=100)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        with torch.no_grad():
            self.temperature.clamp_(min=0.05)

        T = self.temperature.item()
        if logger:
            logger.info(f"Temperature Scaling: fitted T = {T:.4f}")
        return T

    def calibrate_model(self, model, val_loader, device, logger=None):
        """Returns a calibrated version of model (wrapped)."""
        T = self.calibrate(model, val_loader, device, logger)
        return CalibratedModel(model, self), T


class CalibratedModel(nn.Module):
    """Wraps base model with temperature scaling for evaluation."""

    def __init__(self, model, temperature_scaler):
        super().__init__()
        self.model = model
        self.temperature_scaler = temperature_scaler

    def forward(self, x):
        logits = self.model(x)
        return self.temperature_scaler(logits)

    def get_cam_layer(self):
        return self.model.get_cam_layer()
