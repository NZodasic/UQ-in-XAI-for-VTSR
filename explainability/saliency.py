import torch
import numpy as np

class SaliencyMap:
    def __init__(self, model):
        self.model = model

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        # Require gradient for input image
        input_tensor.requires_grad_()
        
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()
        
        # Absolute gradients
        saliency = torch.abs(input_tensor.grad.cpu().data[0]) # [C, H, W]
        saliency = saliency.numpy()
        
        # Max over color channels
        saliency = np.max(saliency, axis=0) # [H, W]
        
        # Normalize
        saliency = saliency - np.min(saliency)
        saliency = saliency / (np.max(saliency) + 1e-8)
        
        return saliency
