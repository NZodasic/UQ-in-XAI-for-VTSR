import torch
import torch.nn.functional as F
import numpy as np

try:
    import cv2
except ImportError:
    pass

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling on the gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # ReLU to keep only positive influences
        cam = np.maximum(cam, 0)
        
        # Resize to input dimensions
        input_h, input_w = input_tensor.shape[2:]
        try:
            cam = cv2.resize(cam, (input_w, input_h))
        except NameError:
            # Fallback if cv2 is not available
            from PIL import Image
            cam = np.array(Image.fromarray(cam).resize((input_w, input_h), Image.BILINEAR))
        
        # Normalize between 0 and 1
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam
