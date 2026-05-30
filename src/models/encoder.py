import torch
from torch import nn
from torchvision import models


# =========================================================
# FRAME ENCODER
# =========================================================

class FrameEncoder(nn.Module):
    """
    Video Encoder

    Input:
        (B, T, C, H, W)

    Output:
        (B, T', embed_dim)
    """

    def __init__(
        self,
        embed_dim=512,
        dropout=0.3
    ):
        super().__init__()

        # -------------------------------------------------
        # Backbone
        # -------------------------------------------------

        backbone = models.video.r3d_18(weights="DEFAULT")

        self.backbone = nn.Sequential(
            backbone.stem,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            # remove layer4 to preserve more temporal info
        )

        # output channels after layer3 = 256
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.projection = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        x:
            (B, T, C, H, W)
        """

        # ---------------------------------------------
        # convert format
        # ---------------------------------------------

        x = x.permute(0, 2, 1, 3, 4)
        # (B, C, T, H, W)

        # ---------------------------------------------
        # backbone
        # ---------------------------------------------

        x = self.backbone(x)

        # shape:
        # (B, 256, T', H', W')

        # ---------------------------------------------
        # preserve temporal dimension
        # ---------------------------------------------

        x = self.pool(x)

        # (B, 256, T', 1, 1)

        x = x.squeeze(-1).squeeze(-1)

        # (B, 256, T')

        x = x.permute(0, 2, 1)

        # (B, T', 256)

        x = self.projection(x)

        # (B, T', embed_dim)

        return x


# =========================================================
# POSITIONAL ENCODING
# =========================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len].to(x.device)

        return x


# =========================================================
# TEMPORAL ENCODER
# =========================================================

class TemporalEncoder(nn.Module):

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        num_layers=4,
        ff_dim=2048,
        dropout=0.3
    ):
        super().__init__()

        self.position = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x,
        padding_mask=None
    ):
        """
        x:
            (B, T, D)

        padding_mask:
            (B, T)

            True = PAD
            False = VALID
        """

        # ---------------------------------------------
        # positional encoding
        # ---------------------------------------------

        x = self.position(x)

        # ---------------------------------------------
        # transformer encoder
        # ---------------------------------------------

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        # ---------------------------------------------
        # final norm
        # ---------------------------------------------

        x = self.norm(x)

        return x


# =========================================================
# CREATE PADDING MASK
# =========================================================

def create_padding_mask(lengths, max_len=None):
    """
    lengths:
        tensor/list of sequence lengths

    returns:
        mask (B, T)

    True = PAD
    False = VALID
    """

    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths)

    if max_len is None:
        max_len = lengths.max()

    seq = torch.arange(max_len).unsqueeze(0)

    mask = seq >= lengths.unsqueeze(1)

    return mask


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    B = 2
    T = 64
    C = 3
    H = 224
    W = 224

    x = torch.randn(B, T, C, H, W)

    encoder = FrameEncoder()

    features = encoder(x)

    print("Frame features:", features.shape)

    lengths = torch.tensor([
        features.shape[1],
        features.shape[1] - 10
    ])

    mask = create_padding_mask(
        lengths,
        features.shape[1]
    )

    temporal = TemporalEncoder()

    out = temporal(
        features,
        padding_mask=mask
    )

    print("Temporal output:", out.shape)