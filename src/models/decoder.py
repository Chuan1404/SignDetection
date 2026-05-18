import torch
from torch import nn

from models.positional_encoding import PositionalEncoding

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

class TextDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=4
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=PAD_IDX
        )

        self.position = PositionalEncoding(embed_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(
            layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(
            embed_dim,
            vocab_size
        )

    def generate_mask(self, size, device):

        mask = torch.triu(
            torch.ones(size, size),
            diagonal=1
        ).bool()

        return mask.to(device)

    def forward(self, tgt, memory):

        tgt = self.embedding(tgt)

        tgt = self.position(tgt)

        tgt_mask = self.generate_mask(
            tgt.size(1),
            tgt.device
        )

        out = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask
        )

        out = self.fc(out)

        return out