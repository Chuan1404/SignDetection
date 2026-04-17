import torch
import torch.nn as nn
import torch.nn.functional as F


class SignTranslator(nn.Module):
    def __init__(self, d1, d2, vocab_size):
        super().__init__()

        self.hidden = 256

        # Project features
        self.i3d_proj = nn.Linear(d1, self.hidden)
        self.mp_proj  = nn.Linear(d2, self.hidden)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=8,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)

        # Decoder
        self.embed = nn.Embedding(vocab_size, self.hidden)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden,
            nhead=8,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=4)

        self.fc_out = nn.Linear(self.hidden, vocab_size)

    def forward(self, i3d, mp, tgt):
        """
        i3d: (B, T1, 1024)
        mp : (B, T2, 99)
        tgt: (B, L)
        """

        x1 = self.i3d_proj(i3d)   # (B,T1,256)
        x2 = self.mp_proj(mp)     # (B,T2,256)

        T = min(x1.size(1), x2.size(1))

        x1 = x1[:, :T, :]
        x2 = x2[:, :T, :]

        x = x1 + x2               # (B,T,256)

        memory = self.encoder(x)

        tgt = self.embed(tgt)     # (B,L,256)

        out = self.decoder(tgt, memory)

        out = self.fc_out(out)

        return out