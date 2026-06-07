import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=29, pretrained=True, dropout_rate=0.4):
        super().__init__()

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = mobilenet_v2(weights=weights)

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def get_cam_layer(self):
        """Returns the final convolution layer for Grad-CAM."""
        for module in reversed(list(self.model.features.modules())):
            if isinstance(module, nn.Conv2d):
                return module
        raise RuntimeError("Could not find Conv2d in MobileNetV2 features.")
