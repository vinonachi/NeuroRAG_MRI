import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNSuperResolution(nn.Module):
    def __init__(self):
        super(CNNSuperResolution, self).__init__()

        # Feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Reconstruction
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # Feature extraction
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Reconstruction
        x = self.relu(self.conv3(x))
        x = self.conv4(x)

        # Output normalized
        x = torch.sigmoid(x)

        return x
