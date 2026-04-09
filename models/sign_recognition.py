import torch
import torch.nn as nn


class SignEncoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()

        # Long Short-Term Memory
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)
        return out  # [batch, seq_len, hidden_dim*2]
