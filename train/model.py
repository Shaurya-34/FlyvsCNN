import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionCNN(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2)   # 48x64 -> 24x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # -> 12x16
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)  # -> 6x8
        self.fc1 = nn.Linear(16 * 6 * 8, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))
