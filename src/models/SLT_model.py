import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.positional_encoding import PositionalEncoding

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