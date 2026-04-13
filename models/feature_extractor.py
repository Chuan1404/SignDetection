import torch
import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

        feat = self.cnn(x).view(B*T, -1)
        feat = self.fc(feat)

        return feat.view(B, T, -1)  # [B, T, 512]