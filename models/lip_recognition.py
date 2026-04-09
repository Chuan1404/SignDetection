import torch
import torch.nn as nn
import torch.nn.functional as F


class LipreadingEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        return F.relu(self.fc(x))  # [batch, seq_len, hidden_dim]
