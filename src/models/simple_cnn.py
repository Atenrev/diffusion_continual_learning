import torch.nn as nn

from typing import Tuple


class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network
    """

    def __init__(self, n_channels: int = 3, num_classes: int = 10):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(128, 128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d(1),
            nn.Dropout(p=0.25),
        )
        self.classifier = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x