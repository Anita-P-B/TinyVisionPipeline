
import torch.nn as nn
import torchvision.models as models

class DragonModel(nn.Module):
    def __init__(self, num_classes=10, model_name='mobilenet_v3_small', dropout_rate=0.3):
        super().__init__()
        if model_name == 'mobilenet_v3_small':
            self.base_model = models.mobilenet_v3_small(weights=None)
        elif model_name == 'mobilenet_v3_large':
            self.base_model = models.mobilenet_v3_large(weights=None)
        else:
            raise ValueError(f"Unknown model_name '{model_name}'")

        # Dynamically get the feature dimension
        feature_dim = self.base_model.classifier[0].in_features

        # Replace classifier with dropout
        self.base_model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)
