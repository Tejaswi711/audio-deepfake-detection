import torch
import torch.nn as nn
import torch.nn.functional as F


class RawNet2(nn.Module):
    def __init__(self, num_classes=2):
        super(RawNet2, self).__init__()

        # First convolution block
        self.conv1 = nn.Conv1d(1, 128, 251, stride=80, padding=125)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResBlock(128, 128) for _ in range(4)]
        )

        # GRU layers
        self.gru = nn.GRU(128, 128, num_layers=2, bidirectional=True)

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        x, _ = self.gru(x)
        w = self.attention(x)
        x = torch.sum(x * w, dim=1)
        x = self.fc(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x