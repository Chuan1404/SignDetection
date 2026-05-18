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

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

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