import torch
import torch.nn as nn

from transformers import MT5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SignLanguageTranslator(nn.Module):

    def __init__(
        self,
        input_dim=126,
        hidden_dim=256,
        temporal_hidden=256,
        num_encoder_layers=3,
        nhead=8,
        pretrained_model="google/mt5-small"
    ):
        super().__init__()

        self.mt5 = MT5ForConditionalGeneration.from_pretrained(pretrained_model)
        d_model = self.mt5.config.d_model  # usually 512

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
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=0.1
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

    def encode(self, hand_features, attention_mask=None):
        """
        hand_features: (B, T, 126)
        attention_mask: (B, T) 1 = valid, 0 = padding
        """

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
        else:
            lengths = torch.full(
                (hand_features.size(0),),
                hand_features.size(1),
                dtype=torch.long,
                device=hand_features.device
            )

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

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        x = self.encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )

        return x

    def forward(
        self,
        hand_features,
        text_ids=None,
        attention_mask=None
    ):

        encoder_hidden_states = self.encode(hand_features, attention_mask)

        outputs = self.mt5(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=encoder_hidden_states
            ),
            attention_mask=attention_mask,
            labels=text_ids
        )

        return outputs

    # --------------------------------------------------
    # inference
    # --------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        hand_features,
        attention_mask=None,
        max_length=64,
        num_beams=4
    ):

        encoder_hidden_states = self.encode(hand_features, attention_mask)

        if attention_mask is None:
            attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                device=encoder_hidden_states.device,
                dtype=torch.long
            )

        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )

        return self.mt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )