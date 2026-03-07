import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=29, pretrained=True, dropout_rate=0.5):
        super(ResNet50Classifier, self).__init__()
        
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.model = resnet50(weights=None)
            
        in_features = self.model.fc.in_features
        
        # We replace the final fully-connected layer
        # Include dropout to enable Monte Carlo Dropout for uncertainty
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
        
    def get_cam_layer(self):
        """Returns the final convolutional block for Grad-CAM"""
        # For ResNet50, it is the last BasicBlock/Bottleneck in layer4
        return self.model.layer4[-1]
