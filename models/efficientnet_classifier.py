import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class EfficientNetB2Classifier(nn.Module):
    def __init__(self, num_classes=29, pretrained=True, dropout_rate=0.4):
        super(EfficientNetB2Classifier, self).__init__()

        if pretrained:
            self.model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        else:
            self.model = efficientnet_b2(weights=None)

        in_features = self.model.classifier[1].in_features

        # Replace classifier with Dropout + Linear
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def get_cam_layer(self):
        """Returns the final pointwise convolution layer for Grad-CAM."""
        last_block = self.model.features[-1]
        conv_layers = [module for module in last_block.modules() if isinstance(module, nn.Conv2d)]
        if conv_layers:
            return conv_layers[-1]
        raise RuntimeError("Could not find Conv2d in EfficientNet last features block.")
