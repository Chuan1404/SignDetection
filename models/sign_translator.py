import torch
import torch.nn as nn
import torch.nn.functional as F

from models.positional_encoding import PositionalEncoding


class SignTranslator(nn.Module):
    def __init__(self, d1, d2, vocab_size):
        super().__init__()
        self.hidden = 256

        self.i3d_proj = nn.Linear(d1, self.hidden)
        self.mp_proj  = nn.Linear(d2, self.hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden, nhead=8, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=4)

        self.embed = nn.Embedding(vocab_size, self.hidden)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=4)

        self.fc_out = nn.Linear(self.hidden, vocab_size)

        # add positional encodings
        self.pos_enc_enc = PositionalEncoding(self.hidden, max_len=2000)
        self.pos_enc_dec = PositionalEncoding(self.hidden, max_len=2000)

    def forward(self, i3d, mp, tgt):
        x1 = self.i3d_proj(i3d)
        x2 = self.mp_proj(mp)
        T = min(x1.size(1), x2.size(1))
        x = x1[:, :T, :] + x2[:, :T, :]

        x = self.pos_enc_enc(x)   # encoder PE
        memory = self.encoder(x)

        tgt = self.embed(tgt)
        tgt = self.pos_enc_dec(tgt)  # decoder PE

        out = self.decoder(tgt, memory)
        return self.fc_out(out)