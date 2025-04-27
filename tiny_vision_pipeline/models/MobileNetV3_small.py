
import torch.nn as nn
import torchvision.models as models

class SmallDragonModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Load MobileNetV3Large without pretrained weights (random init)
        self.base_model = models.mobilenet_v3_small(weights = None)

        # Dynamically get the correct feature size
        feature_dim = self.base_model.classifier[0].in_features

        # Replace classifier to match your number of classes
        self.base_model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, num_classes)  # 960 is the default final feature size
        )

        # Make sure all parameters are trainable (should be by default)
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.base_model(x)
