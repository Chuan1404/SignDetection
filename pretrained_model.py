import torch
from torch import nn
from transformers import VideoMAEModel
from transformers import MT5ForConditionalGeneration

class SLTEncoder(nn.Module):

    def __init__(
        self,
        hidden_dim=768,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    ):

        super().__init__()

        self.video_encoder = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-base"
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim)
        )

    def forward(self, pixel_values):

        """
        pixel_values:
            [B, T, C, H, W]
        """

        outputs = self.video_encoder(
            pixel_values=pixel_values
        )

        # [B, Seq, D]
        features = outputs.last_hidden_state

        memory = self.temporal_transformer(features)

        return memory

class SLTDecoder(nn.Module):

    def __init__(
        self,
        encoder_dim=768,
        model_name="google/mt5-small"
    ):

        super().__init__()

        self.decoder = MT5ForConditionalGeneration.from_pretrained(
            model_name
        )

        t5_dim = self.decoder.config.d_model

        self.proj = nn.Linear(
            encoder_dim,
            t5_dim
        )

    def forward(
        self,
        memory,
        labels,
        attention_mask=None
    ):

        memory = self.proj(memory)

        outputs = self.decoder(
            inputs_embeds=memory,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs

import torch
from torch import nn


import torch
from torch import nn
from transformers import MT5ForConditionalGeneration


class SLTModel(nn.Module):
    def __init__(self, feature_dim=768, hidden_dim=512, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True,
            norm_first=True
        )

        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 3. decoder
        self.decoder = MT5ForConditionalGeneration.from_pretrained(
            "google/mt5-small"
        )

        self.cross_proj = nn.Linear(hidden_dim, self.decoder.config.d_model)

    def forward(self, pixel_values, labels=None, attention_mask=None):

        # pixel_values: (B, T, 768)
        x = self.input_proj(pixel_values)     # (B, T, H)

        memory = self.temporal_encoder(x)     # (B, T, H)

        memory = self.cross_proj(memory)      # (B, T, d_model)

        outputs = self.decoder(
            inputs_embeds=memory,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs