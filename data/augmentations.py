import torch
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import ImageFilter, Image

class AddGaussianNoise(object):
    """Simulate camera noise."""
    def __init__(self, mean=0., std=0.05, p=0.5):
        self.std = std
        self.mean = mean
        self.p = p
        
    def __call__(self, tensor):
        if random.random() < self.p:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        return tensor

class AddFog(object):
    """Simulate fog and reduced visibility."""
    def __init__(self, fog_intensity=0.3, p=0.5):
        self.fog_intensity = fog_intensity
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            img_np = np.array(img).astype(np.float32)
            fog = np.ones_like(img_np) * 255
            blended = img_np * (1 - self.fog_intensity) + fog * self.fog_intensity
            return Image.fromarray(blended.astype(np.uint8))
        return img
        
class AddMotionBlur(object):
    """Simulate motion blur caused by fast driving."""
    def __init__(self, radius=2, p=0.5):
        self.radius = radius
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.GaussianBlur(self.radius))
        return img

def get_train_transforms(image_size):
    """
    Simulates real-world TS recognition conditions:
    occlusion, weather, lighting changes, camera noise, motion blur.
    """
    return transforms.Compose([
        transforms.Resize((int(image_size[0] * 1.05), int(image_size[1] * 1.05))), # slight scale up
        transforms.RandomCrop(image_size), # Scaling and slight translation
        transforms.RandomRotation(15),     # Rotation
        AddFog(fog_intensity=0.3, p=0.2),  # Fog simulation
        AddMotionBlur(radius=2, p=0.2),    # Motion blur
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # lighting changes
        transforms.ToTensor(),
        AddGaussianNoise(mean=0.0, std=0.05, p=0.2), # Gaussian camera noise
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0) # occlusion patches
    ])

def get_val_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
