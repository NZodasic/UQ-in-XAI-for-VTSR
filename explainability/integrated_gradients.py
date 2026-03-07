import torch
import numpy as np

class IntegratedGradients:
    def __init__(self, model, steps=50):
        self.model = model
        self.steps = steps

    def generate(self, input_tensor, target_class=None, baseline=None):
        self.model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
            
        # Get target class if not provided
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
                
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, self.steps).view(-1, 1, 1, 1).to(input_tensor.device)
        interpolated_inputs = baseline + alphas * (input_tensor - baseline)
        interpolated_inputs.requires_grad_()
        
        # Forward pass on all interpolated inputs
        outputs = self.model(interpolated_inputs)
        
        # Get scores for the target class
        scores = outputs[:, target_class]
        
        # Backward pass to get sum of gradients
        self.model.zero_grad()
        scores.sum().backward()
        
        # Calculate gradients
        gradients = interpolated_inputs.grad.cpu().data.numpy() # [steps, C, H, W]
        
        # Average gradients across steps
        avg_gradients = np.mean(gradients, axis=0) # [C, H, W]
        
        # Multiply by (input - baseline) as per axiom
        input_np = input_tensor.cpu().data.numpy()[0]
        baseline_np = baseline.cpu().data.numpy()[0]
        
        integrated_grad = (input_np - baseline_np) * avg_gradients
        
        # Absolute heatmap
        ig_heatmap = np.abs(integrated_grad)
        
        # Max over color channels
        ig_heatmap = np.max(ig_heatmap, axis=0) # [H, W]
        
        # Normalize
        ig_heatmap = ig_heatmap - np.min(ig_heatmap)
        ig_heatmap = ig_heatmap / (np.max(ig_heatmap) + 1e-8)
        
        return ig_heatmap
