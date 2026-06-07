import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121Classifier(nn.Module):
    def __init__(self, num_classes=29, pretrained=True, dropout_rate=0.4):
        super().__init__()

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = densenet121(weights=weights)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=False),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def get_cam_layer(self):
        """Returns the final dense block for Grad-CAM."""
        return self.model.features.denseblock4
