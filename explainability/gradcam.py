import torch
import numpy as np
import torch.nn as nn

try:
    import cv2
except ImportError:
    pass

def normalize_map(values):
    values = values - np.min(values)
    return values / (np.max(values) + 1e-8)

class GradCAM:
    """Basic Grad-CAM implementation (self-contained)."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self._forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self._backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self._forward_hook.remove()
        self._backward_hook.remove()

    def generate(self, input_tensor, target_class=None):
        # We don't force eval() here anymore, let the caller decide (important for MC Dropout)
        self.gradients = None
        self.activations = None
        
        # Ensure input requires grad
        input_tensor = input_tensor.detach().requires_grad_(True)

        with torch.enable_grad():
            output = self.model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()
            target_class = int(target_class)

            self.model.zero_grad()
            score = output[0, target_class]
            score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "GradCAM hooks did not fire. Check that target_layer is part of the forward pass."
            )

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = normalize_map(cam)

        input_h, input_w = input_tensor.shape[2:]
        try:
            cam = cv2.resize(cam, (input_w, input_h))
        except NameError:
            from PIL import Image
            cam = np.array(
                Image.fromarray(cam).resize(
                    (input_w, input_h),
                    Image.Resampling.BILINEAR
                )
            )

        return cam

class GradCAMLibraryWrapper:
    """Wrapper for pytorch-grad-cam library (GradCAM++, EigenCAM, etc)."""
    def __init__(self, model, target_layer, method='gradcam'):
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM, HiResCAM
        
        methods = {
            'gradcam': GradCAM,
            'gradcam++': GradCAMPlusPlus,
            'eigencam': EigenCAM,
            'hirescam': HiResCAM
        }
        
        if method.lower() not in methods:
            raise ValueError(f"Method {method} not supported. Choose from {list(methods.keys())}")
            
        self.cam = methods[method.lower()](model=model, target_layers=[target_layer])

    def generate(self, input_tensor, target_class=None):
        import torch
        # pytorch-grad-cam expects target_class as a list of targets
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        
        targets = None
        if target_class is not None:
            targets = [ClassifierOutputTarget(int(target_class))]
            
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]

    def remove_hooks(self):
        pass # Library handles hooks internally

class MCGradCAM:
    """Monte Carlo Grad-CAM for uncertainty quantification in explanations."""
    def __init__(self, model, target_layer, cam_method='gradcam', num_samples=10):
        self.model = model
        self.num_samples = num_samples
        
        # Try to use library wrapper if requested and available
        try:
            if cam_method.lower() != 'gradcam_basic':
                self.cam_gen = GradCAMLibraryWrapper(model, target_layer, method=cam_method)
            else:
                self.cam_gen = GradCAM(model, target_layer)
        except ImportError:
            self.cam_gen = GradCAM(model, target_layer)

    def generate(self, input_tensor, target_class=None):
        """
        Generates multiple CAMs with dropout enabled.
        Returns: (mean_cam, std_cam)
        """
        # Save original training state
        module_states = {id(m): (m, m.training) for m in self.model.modules()}
        
        cams = []
        try:
            self.model.eval()
            # Enable Dropout modules for MC
            for m in self.model.modules():
                if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)):
                    m.eval()
                elif isinstance(m, nn.modules.dropout._DropoutNd):
                    m.train()
            
            for _ in range(self.num_samples):
                cam = self.cam_gen.generate(input_tensor, target_class)
                cams.append(cam)
        finally:
            # Restore original state
            for module_id, (module, was_training) in module_states.items():
                module.train(was_training)
        
        cams = np.array(cams) # [num_samples, H, W]
        mean_cam = np.mean(cams, axis=0)
        std_cam = np.std(cams, axis=0)
        
        # Normalize mean_cam
        mean_cam = normalize_map(mean_cam)
        
        return mean_cam, std_cam

    def remove_hooks(self):
        if hasattr(self.cam_gen, 'remove_hooks'):
            self.cam_gen.remove_hooks()
