import torch
import torch.nn as nn

class MultiStreamFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Linear(dim * 3, dim)

    def forward(self, sign, finger, lip):
        x = torch.cat([sign, finger, lip], dim=-1)
        return self.fc(x)  # [B, T, 512]

