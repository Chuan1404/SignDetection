import torch
import torch.nn as nn
from torchvision import models

from src.models.decoder import TextDecoder
from src.models.encoder import FrameEncoder, TemporalEncoder

class SLTModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=4
    ):

        super().__init__()

        self.fusion = nn.Linear(
            embed_dim * 2,
            embed_dim
        )

        self.encoder = TemporalEncoder(
            embed_dim,
            num_heads,
            num_layers
        )

        self.decoder = TextDecoder(
            vocab_size,
            embed_dim,
            num_heads,
            num_layers
        )


    def forward(
        self,
        left_feat,
        right_feat,
        tgt
    ):

        x = torch.cat(
            [left_feat, right_feat],
            dim=-1
        )

        x = self.fusion(x)

        memory = self.encoder(x)

        out = self.decoder(
            tgt,
            memory
        )

        return out

@torch.no_grad()
def generate(
    model,
    left_video,
    right_video,
    sos_token,
    eos_token,
    max_len=50,
    device="cuda"
):

    model.eval()

    generated = [sos_token]

    for _ in range(max_len):

        current_tokens = torch.tensor(
            generated,
            device=device
        ).unsqueeze(0)

        output = model(
            left_video,
            right_video,
            current_tokens
        )

        next_token = output[:, -1].argmax(-1).item()

        generated.append(next_token)

        if next_token == eos_token:
            break

    return generated

