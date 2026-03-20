"""
DenseNet121 Model for Pneumonia Detection
==========================================
Uses transfer learning from ImageNet-pretrained DenseNet121.
The final classifier is replaced with a 2-class head for binary
classification (Normal vs Pneumonia).

Architecture:
  DenseNet121 features (frozen) → AdaptiveAvgPool → Dropout → Linear(1024, 2)
"""

import torch
import torch.nn as nn
from torchvision import models


class PneumoniaNet(nn.Module):
    """
    DenseNet121-based binary classifier for pneumonia detection.
    
    The feature extraction layers are frozen to leverage pretrained
    ImageNet representations. Only the final classifier head is trained,
    making training fast and effective even with limited medical data.
    """

    def __init__(self, num_classes=2, pretrained=True):
        super(PneumoniaNet, self).__init__()

        # Load pretrained DenseNet121 backbone
        self.densenet = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Freeze feature extraction layers for transfer learning
        for param in self.densenet.features.parameters():
            param.requires_grad = False

        # Replace the classifier head
        # DenseNet121 has 1024 features before the classifier
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """Forward pass through the network."""
        return self.densenet(x)

    def get_features_module(self):
        """
        Returns the feature extraction module (used by Grad-CAM).
        Specifically returns the last DenseBlock for activation mapping.
        """
        return self.densenet.features

    def get_last_conv_layer(self):
        """
        Returns the last convolutional layer name for Grad-CAM.
        In DenseNet121, this is 'features.denseblock4'.
        """
        return self.densenet.features.denseblock4
