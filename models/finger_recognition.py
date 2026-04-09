import torch
import torch.nn as nn

class FingerspellingEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [batch, input_dim, seq_len]
        out = self.cnn(x)  # [batch, hidden_dim, seq_len]
        return out.transpose(1, 2)