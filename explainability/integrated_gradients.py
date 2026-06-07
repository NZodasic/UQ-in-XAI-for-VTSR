import torch
import numpy as np

class IntegratedGradients:
    def __init__(self, model, steps=50):
        self.model = model
        self.steps = steps

    def generate(self, input_tensor, target_class=None, baseline=None):
        module_states = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
            
        # Get target class if not provided
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        target_class = int(target_class)
                
        # Generate interpolated inputs
        batch_size = input_tensor.size(0)
        alphas = torch.linspace(0, 1, self.steps, device=input_tensor.device).view(-1, 1, 1, 1, 1)
        interpolated_inputs = baseline.unsqueeze(0) + alphas * (input_tensor.unsqueeze(0) - baseline.unsqueeze(0))
        interpolated_inputs = interpolated_inputs.flatten(0, 1)
        interpolated_inputs = interpolated_inputs.detach().requires_grad_(True)
        
        try:
            # Forward pass on all interpolated inputs
            outputs = self.model(interpolated_inputs)

            # Get scores for the target class
            scores = outputs[:, target_class]

            # Backward pass to get sum of gradients
            self.model.zero_grad()
            if interpolated_inputs.grad is not None:
                interpolated_inputs.grad.zero_()
            scores.sum().backward()

            # Calculate gradients
            gradients = interpolated_inputs.grad.detach().view(self.steps, batch_size, *input_tensor.shape[1:])
            gradients = gradients.cpu().numpy()
        finally:
            for module, was_training in module_states:
                module.train(was_training)
        
        # Average gradients across steps
        avg_gradients = np.mean(gradients, axis=0) # [B, C, H, W]
        
        # Multiply by (input - baseline) as per axiom
        input_np = input_tensor.detach().cpu().numpy()
        baseline_np = baseline.detach().cpu().numpy()
        
        integrated_grad = (input_np - baseline_np) * avg_gradients
        
        # Absolute heatmap
        ig_heatmap = np.abs(integrated_grad)
        
        # Max over color channels
        ig_heatmap = np.max(ig_heatmap, axis=1) # [B, H, W]
        
        # Normalize
        mins = ig_heatmap.min(axis=(1, 2), keepdims=True)
        maxs = ig_heatmap.max(axis=(1, 2), keepdims=True)
        ig_heatmap = (ig_heatmap - mins) / (maxs + 1e-8)
        
        return ig_heatmap[0] if batch_size == 1 else ig_heatmap
