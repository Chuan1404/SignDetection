import torch
import torch.nn as nn

class MultiStreamFusion(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

    def forward(self, streams):
        # streams: list of [batch, seq_len, dim]
        concat = torch.cat(streams, dim=-1)  # [batch, seq_len, sum_dim]
        fused, _ = self.attn(concat, concat, concat)
        return fused  # [batch, seq_len, hidden_dim]
