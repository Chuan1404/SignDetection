import torch
import torch.nn as nn

from models.positional_encoding import PositionalEncoding


class SignTranslator(nn.Module):
    def __init__(self, i3d_dim, mp_dim, vocab_size, hidden=256, nhead=8, num_layers=4):
        super().__init__()

        self.hidden = hidden

        # ======================
        # INPUT PROJECTIONS
        # ======================
        self.i3d_proj = nn.Linear(i3d_dim, hidden)
        self.mp_proj = nn.Linear(mp_dim, hidden)

        # fuse after projection (IMPORTANT FIX)
        self.fusion = nn.Linear(hidden * 2, hidden)

        # ======================
        # ENCODER
        # ======================
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=hidden * 4
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ======================
        # DECODER
        # ======================
        self.embed = nn.Embedding(vocab_size, hidden)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=hidden * 4
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden, vocab_size)

        # ======================
        # POSITIONAL ENCODING
        # ======================
        self.pos_enc = PositionalEncoding(hidden)

    # ======================
    # FORWARD PASS (TRAINING)
    # ======================
    def forward(self, i3d, mp, tgt):

        # ---- encode modalities
        i3d = self.i3d_proj(i3d)
        mp = self.mp_proj(mp)

        # ---- align time (IMPORTANT FIX: do NOT destroy by sum)
        T = min(i3d.size(1), mp.size(1))
        i3d = i3d[:, :T, :]
        mp = mp[:, :T, :]

        # ---- fusion (IMPORTANT FIX)
        x = torch.cat([i3d, mp], dim=-1)
        x = self.fusion(x)

        # ---- encoder
        x = self.pos_enc(x)
        memory = self.encoder(x)

        # ---- decoder
        tgt = self.embed(tgt)
        tgt = self.pos_enc(tgt)

        # causal mask (VERY IMPORTANT)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(tgt.device)

        out = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        return self.fc_out(out)