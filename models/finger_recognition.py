import torch
import torch.nn as nn

class FingerspellingEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.lstm(x)[0]  # [B, T, 512]
