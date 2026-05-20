import torch
from torch import nn
from torchvision import models

class FrameEncoder(nn.Module):

    def __init__(self, embed_dim=512):

        super().__init__()

        backbone = models.resnet18(pretrained=True)

        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.projection = nn.Linear(
            512,
            embed_dim
        )

    def forward(self, x):

        """
        x:
        (B,T,C,H,W)
        """

        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)

        features = self.backbone(x)

        features = features.reshape(B * T, -1)

        features = self.projection(features)

        features = features.reshape(B, T, -1)

        return features

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(max_len).unsqueeze(1)

        even_dims = torch.arange(0, d_model, 2)

        div_term = 1 / (10000 ** (even_dims / d_model))

        angle = position * div_term

        # Apply sin to even dimensions
        pe[:, 0::2] = torch.sin(angle)

        # Apply cos to odd dimensions
        pe[:, 1::2] = torch.cos(angle)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape:
        # (batch_size, seq_len, d_model)
        seq_len = x.size(1)

        # Add positional encoding
        x = x + self.pe[:, :seq_len]

        return x

class TemporalEncoder(nn.Module):

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        num_layers=4
    ):

        super().__init__()

        self.position = PositionalEncoding(embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers
        )

    def forward(self, x):

        x = self.position(x)

        x = self.encoder(x)

        return x