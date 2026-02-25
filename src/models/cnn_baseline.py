import torch
import torch.nn as nn


class CNNSuperResolution(nn.Module):
    """
    Baseline CNN model for MRI Super-Resolution.

    Architecture:
    Input (1 channel)
        ↓
    Conv2D (64 filters)
        ↓
    ReLU
        ↓
    Conv2D (64 filters)
        ↓
    ReLU
        ↓
    Conv2D (1 filter)
        ↓
    Output (Reconstructed image)
    """

    def __init__(self):
        super(CNNSuperResolution, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)

        return x
