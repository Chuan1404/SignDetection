import torch
import torch.nn as nn


class LLMDecoder(nn.Module):
    def __init__(self, hidden_dim=512, vocab_size=10000):
        super().__init__()
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=6)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, enc_out, tgt_seq):
        # enc_out: [batch, seq_len, hidden_dim]
        # tgt_seq: [batch, tgt_len, hidden_dim] (embedded tokens)
        out = self.transformer(enc_out, tgt_seq)
        return self.fc_out(out)  # logits [batch, tgt_len, vocab_size]
