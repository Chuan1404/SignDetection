import torch
import torch.nn as nn


class LLMDecoder(nn.Module):
    def __init__(self, hidden_dim=512, vocab_size=5000):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                batch_first=True
            ),
            num_layers=4
        )

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, enc_out, tgt_seq):
        # enc_out: [B, T, D]
        # tgt_seq: [B, T]

        tgt_emb = self.token_emb(tgt_seq)

        out = self.decoder(tgt_emb, enc_out)
        return self.fc_out(out)  # [B, T, vocab]

