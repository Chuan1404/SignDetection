import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.positional_encoding import PositionalEncoding
from src.models.spatial_graph import SpatialGraphConv, NUM_NODES


class ClassificationOutput:

    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits


def masked_mean_pool(x, video_mask):
    mask = video_mask.unsqueeze(-1).float()          # (B, T, 1)
    summed = (x * mask).sum(dim=1)                    # (B, D)
    counts = mask.sum(dim=1).clamp(min=1.0)           # (B, 1) — avoid /0
    return summed / counts


# =============================================================================
# V1 — Baseline: Bi-LSTM + Projection -> Linear gloss classifier
# -----------------------------------------------------------------------------
# Pipeline: raw features -> BiLSTM (temporal) -> Linear projection
#           -> masked mean pool -> Linear classifier -> gloss logits

class SignLanguageTranslatorV1(nn.Module):

    def __init__(
        self,
        input_dim=186,
        hidden_dim=256,
        temporal_hidden=256,
        dropout=0.1,
        num_classes=2000
    ):
        super().__init__()

        self.temporal_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=temporal_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        temporal_out_dim = temporal_hidden * 2

        self.input_projection = nn.Sequential(
            nn.Linear(temporal_out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def get_optimizer_param_groups(self, lr):
        return [{"params": self.parameters(), "lr": lr, "name": "all"}]

    def encode(self, features, video_mask):
        """
        features  : (B, T, input_dim)  # fused pose + hand
        video_mask: (B, T)
        """
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()
        lengths = torch.clamp(video_mask.sum(dim=1).cpu(), min=1)

        packed = pack_padded_sequence(
            features, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.temporal_encoder(packed)
        x, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=features.size(1)
        )

        return self.input_projection(x)   # (B, T, hidden_dim)

    def forward(self, features, labels=None, video_mask=None):
        """
        labels: (B,) LongTensor of gloss class indices, or None at inference.
        """
        x = self.encode(features, video_mask)
        pooled = masked_mean_pool(x, video_mask.bool())
        logits = self.classifier(pooled)          # (B, num_classes)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ClassificationOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def predict(self, features, video_mask=None, top_k=1):
        """Convenience wrapper for inference — returns class indices,
        top-1 (shape (B,)) or top-k (shape (B, top_k))."""
        logits = self.forward(features, video_mask=video_mask).logits
        if top_k == 1:
            return logits.argmax(dim=-1)
        return logits.topk(k=top_k, dim=-1).indices


# =============================================================================
# V2 — Pure Transformer Encoder: Projection -> PositionalEncoding -> Transformer (6L) -> mT5
# -----------------------------------------------------------------------------
# Pipeline: raw features -> Linear Projection -> PositionalEncoding
#           -> Transformer Encoder (6 layers) -> mT5 decoder

class SignLanguageTranslatorV2(nn.Module):

    def __init__(
        self,
        input_dim=186,
        hidden_dim=256,
        num_encoder_layers=6,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.2,
        max_seq_len=5000,
        pretrained_model="google/mt5-small"
    ):
        super().__init__()

        self.mt5 = MT5ForConditionalGeneration.from_pretrained(pretrained_model)
        d_model = self.mt5.config.d_model  # 512 for mt5-small

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.encoder_norm = nn.LayerNorm(d_model)

    def get_optimizer_param_groups(self, lr_new_modules, lr_mt5):
        mt5_param_ids = {id(p) for p in self.mt5.parameters()}
        new_module_params = [p for p in self.parameters() if id(p) not in mt5_param_ids]
        mt5_params = list(self.mt5.parameters())

        return [
            {"params": new_module_params, "lr": lr_new_modules, "name": "new_modules"},
            {"params": mt5_params,        "lr": lr_mt5,         "name": "mt5_pretrained"},
        ]

    def set_mt5_trainable(self, embeddings: bool, encoder_layers_from: int):
        num_layers = len(self.mt5.encoder.block)
        encoder_layers_from = max(0, min(encoder_layers_from, num_layers))

        for p in self.mt5.shared.parameters():
            p.requires_grad = embeddings

        for i, block in enumerate(self.mt5.encoder.block):
            trainable = i >= encoder_layers_from
            for p in block.parameters():
                p.requires_grad = trainable

        for p in self.mt5.encoder.final_layer_norm.parameters():
            p.requires_grad = True

    def encode(self, features, video_mask):
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()

        x = self.input_projection(features)      # (B, T, d_model)
        x = self.pos_encoder(x)                   # (B, T, d_model)
        x = self.encoder(
            x,
            src_key_padding_mask=~video_mask      # True = ignore (padding)
        )                                          # (B, T, d_model)
        x = self.encoder_norm(x)                  # (B, T, d_model)

        return x

    def forward(
        self,
        features,
        text_ids=None,
        video_mask=None
    ):

        encoder_hidden_states = self.encode(features, video_mask)

        return self.mt5(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=encoder_hidden_states
            ),
            attention_mask=video_mask.long() if video_mask is not None else None,
            labels=text_ids
        )

    @torch.no_grad()
    def generate(
        self,
        features,
        video_mask=None,
        max_length=64,
        num_beams=4,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    ):

        encoder_hidden_states = self.encode(features, video_mask)

        return self.mt5.generate(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=encoder_hidden_states
            ),
            attention_mask=video_mask.long() if video_mask is not None else None,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )


# =============================================================================
# V3 — Pure Transformer Encoder -> Linear gloss classifier
# -----------------------------------------------------------------------------

class SignLanguageTranslatorV3(nn.Module):

    def __init__(
        self,
        input_dim=186,
        hidden_dim=256,
        num_encoder_layers=6,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.2,
        max_seq_len=5000,
        num_classes=2000
    ):
        super().__init__()

        d_model = hidden_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        self.encoder_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def get_optimizer_param_groups(self, lr):
        return [{"params": self.parameters(), "lr": lr, "name": "all"}]

    def encode(self, features, video_mask):
        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()

        x = self.input_projection(features)       # (B, T, d_model)
        x = self.pos_encoder(x)                    # (B, T, d_model)
        x = self.encoder(
            x,
            src_key_padding_mask=~video_mask       # True = ignore (padding)
        )                                           # (B, T, d_model)
        x = self.encoder_norm(x)                   # (B, T, d_model)

        return x

    def forward(self, features, labels=None, video_mask=None):
        x = self.encode(features, video_mask)
        pooled = masked_mean_pool(x, video_mask.bool())
        logits = self.classifier(pooled)           # (B, num_classes)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ClassificationOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def predict(self, features, video_mask=None, top_k=1):
        logits = self.forward(features, video_mask=video_mask).logits
        if top_k == 1:
            return logits.argmax(dim=-1)
        return logits.topk(k=top_k, dim=-1).indices


class _TemporalConv(nn.Module):
    """
    Temporal convolution dọc theo trục T — giống paper.
    kernel_size kt×1: slide qua T frames, không slide qua N nodes.
    """

    def __init__(self, channels, kernel_size=9, stride=1, dropout=0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, N, C) → (B, C, T, N)
        x = x.permute(0, 3, 1, 2)
        x = self.drop(self.conv(x))
        x = self.bn(x)
        x = x.permute(0, 2, 3, 1)  # (B, T, N, C)
        return x


class _STGCNBlock(nn.Module):
    """
    1 ST-GCN block = Spatial GCN (partitioned) + BN + ReLU
                   + Temporal Conv + BN
                   + Residual + ReLU

    Theo Figure 6 trong paper: mỗi block có spatial GCN → temporal GCN → BN.
    Residual match dimension bằng 1×1 Conv2d nếu channels thay đổi.
    """

    def __init__(self, in_ch, out_ch, dropout=0.0, kernel_size=9):
        super().__init__()

        # Spatial GCN — dùng partitioned adjacency từ spatial_graph.py
        self.spatial = SpatialGraphConv(
            in_channels=in_ch,
            out_channels=out_ch
        )
        self.spatial_bn = nn.BatchNorm1d(out_ch)
        self.spatial_relu = nn.ReLU()

        # Temporal Conv
        self.temporal = _TemporalConv(out_ch, kernel_size=kernel_size, dropout=dropout)
        self.temporal_relu = nn.ReLU()

        # Residual
        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        res = x

        # Spatial GCN
        x = self.spatial(x)  # (B, T, N, C_out)
        # BN trên channel — reshape để BatchNorm1d hoạt động
        x = self.spatial_bn(
            x.reshape(B * T * N, -1)
        ).reshape(B, T, N, -1)
        x = self.spatial_relu(x)

        # Temporal Conv
        x = self.temporal(x)  # (B, T, N, C_out)

        # Residual — cần (B, C, T, N) cho Conv2d
        if not isinstance(self.residual, nn.Identity):
            res = res.permute(0, 3, 1, 2)  # (B, C_in, T, N)
            res = self.residual(res)  # (B, C_out, T, N)
            res = res.permute(0, 2, 3, 1)  # (B, T, N, C_out)

        return self.temporal_relu(x + res)


class SignLanguageTranslatorV4(nn.Module):

    NUM_JOINTS = NUM_NODES  # 61

    def __init__(
            self,
            num_classes=2000,
            dropout=0.25,  # paper dùng dropout + DropGraph
            kernel_size=9  # temporal kernel — paper dùng kt=9
    ):
        super().__init__()

        # Input BN — normalize raw coordinates trước khi vào GCN
        # Paper: "normalize the keypoint coordinates to [-1,1]"
        # BN làm điều tương tự một cách learned
        self.input_bn = nn.BatchNorm1d(self.NUM_JOINTS * 3)

        # 10 ST-GCN blocks theo Figure 6
        # ch = [3, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        ch = [3, 32, 64, 64, 64, 128, 128, 256]
        self.blocks = nn.ModuleList([
            _STGCNBlock(ch[i], ch[i + 1], dropout=dropout, kernel_size=kernel_size)
            for i in range(7)
        ])

        # Global classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def get_optimizer_param_groups(self, lr):
        return [{"params": self.parameters(), "lr": lr, "name": "all"}]

    def encode(self, features, video_mask):
        """
        features  : (B, T, 185)
        video_mask: (B, T)
        Returns   : (B, 256)
        """
        B, T, _ = features.shape

        joint_coords = features[..., :self.NUM_JOINTS * 3]  # (B, T, 179)

        # Input BN trên joint-coordinate dim
        x = self.input_bn(
            joint_coords.reshape(B * T, self.NUM_JOINTS * 3)
        ).reshape(B, T, self.NUM_JOINTS, 3)  # (B, T, 61, 3)
        # x = joint_coords.reshape(B, T, self.NUM_JOINTS, 3)


        # 10 ST-GCN blocks
        for block in self.blocks:
            x = block(x)  # (B, T, 59, 256)

        # Masked mean pool qua T — bỏ padding frames
        mask = video_mask.bool().unsqueeze(-1).unsqueeze(-1).float()  # (B, T, 1, 1)
        x = (x * mask).sum(dim=1)  # (B, 61, 256)
        x = x / mask.sum(dim=1).clamp(min=1.0)

        # Mean pool qua N joints → (B, 256)
        x = x.mean(dim=1)

        return x

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")

        pooled = self.encode(features, video_mask)  # (B, 256)
        logits = self.classifier(pooled)  # (B, num_classes)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ClassificationOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def predict(self, features, video_mask=None, top_k=1):
        logits = self.forward(features, video_mask=video_mask).logits
        if top_k == 1:
            return logits.argmax(dim=-1)
        return logits.topk(k=top_k, dim=-1).indices