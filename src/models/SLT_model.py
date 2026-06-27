import torch
import torch.nn as nn

from transformers import MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.positional_encoding import PositionalEncoding


class SignLanguageTranslatorV1(nn.Module):

    def __init__(
        self,
        input_dim=186,
        hidden_dim=256,
        temporal_hidden=256,
        pretrained_model="google/mt5-small",
        freeze_mt5_encoder=False
    ):
        super().__init__()

        self.mt5 = MT5ForConditionalGeneration.from_pretrained(pretrained_model)
        d_model = self.mt5.config.d_model

        if freeze_mt5_encoder:
            for param in self.mt5.encoder.parameters():
                param.requires_grad = False

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
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )

    def encode(self, features, video_mask):
        """
        features  : (B, T, 186)  # fused pose + hand
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

        return self.input_projection(x)

    def forward(self, features, text_ids=None, video_mask=None):
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

class SignLanguageTranslatorV2(nn.Module):

    def __init__(
        self,
        input_dim=126,
        hidden_dim=256,
        temporal_hidden=256,
        num_encoder_layers=3,
        nhead=8,
        max_seq_len=5000,
        pretrained_model="google/mt5-small"
    ):
        super().__init__()

        self.mt5 = MT5ForConditionalGeneration.from_pretrained(
            pretrained_model
        )

        d_model = self.mt5.config.d_model

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
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )

        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=0.1
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.1,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

    def encode(self, hand_features, video_mask):
        """
        hand_features: (B, T, 126)
        video_mask   : (B, T)
        """

        if video_mask is None:
            raise ValueError("video_mask is required")

        video_mask = video_mask.bool()

        lengths = video_mask.sum(dim=1).cpu()
        lengths = torch.clamp(lengths, min=1)

        packed = pack_padded_sequence(
            hand_features,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.temporal_encoder(packed)

        x, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=hand_features.size(1)
        )

        x = self.input_projection(x)

        x = self.pos_encoder(x)

        x = self.encoder(
            x,
            src_key_padding_mask=~video_mask
        )

        return x

    def forward(
        self,
        hand_features,
        text_ids=None,
        video_mask=None
    ):

        encoder_hidden_states = self.encode(
            hand_features,
            video_mask
        )

        # week encoder
        # attension mask

        outputs = self.mt5(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=encoder_hidden_states
            ),
            labels=text_ids
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        hand_features,
        video_mask=None,
        max_length=64,
        num_beams=4,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    ):

        encoder_hidden_states = self.encode(
            hand_features,
            video_mask
        )

        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )

        return self.mt5.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

class SignLanguageTranslator(nn.Module):

    def __init__(
        self,
        input_dim=126,
        pretrained_model="google/mt5-small",
        dropout=0.1
    ):
        super().__init__()

        self.mt5 = MT5ForConditionalGeneration.from_pretrained(pretrained_model)
        d_model = self.mt5.config.d_model  # 512 for mt5-small

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

    def encode(self, hand_features, video_mask=None):

        x = self.input_projection(hand_features)  # (B, T, d_model)
        encoder_outputs = self.mt5.encoder(
            inputs_embeds=x,
            attention_mask=video_mask
        )

        return encoder_outputs.last_hidden_state

    def forward(
        self,
        hand_features,
        text_ids=None,
        video_mask=None
    ):

        x = self.input_projection(hand_features)

        outputs = self.mt5(
            inputs_embeds=x,
            attention_mask=video_mask,
            labels=text_ids
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        hand_features,
        video_mask=None,
        max_length=64,
        num_beams=4,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    ):

        x = self.input_projection(hand_features)

        return self.mt5.generate(
            inputs_embeds=x,
            attention_mask=video_mask,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )