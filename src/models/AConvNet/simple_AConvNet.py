"""Simplified AConvNet variants for quick experimentation."""

import torch.nn as nn


class Simple_AConvNet(nn.Module):
    """Flat re-implementation of AConvNet without the block abstraction.

    Mirrors the five-convolution architecture of :class:`AConvNet` using
    explicit layer attributes instead of ``Conv2DBlock``.  Intended for quick
    experimentation and debugging.

    Args:
        num_classes (int): Number of output classes (default 10).
    """

    def __init__(self, num_classes=10):
        super(Simple_AConvNet, self).__init__()

        # Conv1: 1 → 16 channels, 5x5, ReLU, MaxPool
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2: 16 → 32 channels, 5x5, ReLU, MaxPool
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv3: 32 → 64 channels, 6x6, ReLU, MaxPool
        self.conv3 = nn.Conv2d(32, 64, kernel_size=6, padding=0)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv4: 64 → 128 channels, 5x5, ReLU, Dropout
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=0)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # Conv5: 128 → num_classes channels, 3x3
        self.conv5 = nn.Conv2d(128, num_classes, kernel_size=3, padding=0)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.dropout(self.relu4(self.conv4(x)))
        x = self.conv5(x)
        x = self.gap(x)
        x = self.flatten(x)
        return x


class AConvNet_batchnorm(nn.Module):
    """AConvNet variant with BatchNorm inserted after every convolutional layer.

    Adds ``nn.BatchNorm2d`` after each conv to accelerate convergence.
    Architecture otherwise mirrors :class:`Simple_AConvNet`.

    Args:
        num_classes (int): Number of output classes (default 10).
        in_channels (int): Number of input channels (default 1).
    """

    def __init__(self, num_classes=10, in_channels=1):
        super(AConvNet_batchnorm, self).__init__()

        # OPTIMIZED: Reduced channels for faster computation and less memory
        # Conv1: in_channels → 16 channels, 5x5, ReLU, MaxPool
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm2d(16)  # Added BatchNorm for faster convergence
        self.relu1 = nn.ReLU(inplace=True)  # inplace=True saves memory
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2: 16 → 32 channels, 5x5, ReLU, MaxPool
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv3: 32 → 64 channels, 6x6, ReLU, MaxPool
        self.conv3 = nn.Conv2d(32, 64, kernel_size=6, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv4: 64 → 128 channels, 5x5, ReLU, Dropout
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

        # Conv5: 128 → num_classes channels, 3x3
        self.conv5 = nn.Conv2d(128, num_classes, kernel_size=3, padding=0)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout(self.relu4(self.bn4(self.conv4(x))))
        x = self.conv5(x)
        x = self.gap(x)
        x = self.flatten(x)
        return x
