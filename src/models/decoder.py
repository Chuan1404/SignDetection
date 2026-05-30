from torch import nn
import torch
import torch.nn.functional as F
import math

from models.positional_encoding import PositionalEncoding


PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


class TextDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    ):

        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # =====================================================
        # Token Embedding
        # =====================================================
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_IDX
        )

        # =====================================================
        # Positional Encoding
        # =====================================================
        self.position = PositionalEncoding(embed_dim)

        self.dropout = nn.Dropout(dropout)

        # =====================================================
        # Transformer Decoder Layer
        # =====================================================
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        # =====================================================
        # Multi-layer Decoder
        # =====================================================
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )

        # =====================================================
        # Vocabulary Projection
        # =====================================================
        self.fc = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False
        )

        # =====================================================
        # Weight Tying
        # =====================================================
        self.fc.weight = self.embedding.weight

        # =====================================================
        # Initialize Parameters
        # =====================================================
        self._init_weights()

    # =====================================================
    # Weight Initialization
    # =====================================================
    def _init_weights(self):

        for p in self.parameters():

            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # =====================================================
    # Generate Causal Mask
    # =====================================================
    def generate_causal_mask(
        self,
        size,
        device
    ):

        """
        Prevent decoder from seeing future tokens

        Shape:
            [T, T]
        """

        mask = torch.triu(
            torch.full(
                (size, size),
                float("-inf"),
                device=device
            ),
            diagonal=1
        )

        return mask

    # =====================================================
    # Forward (Training)
    # =====================================================
    def forward(
        self,
        tgt_ids,
        memory,
        memory_padding_mask=None
    ):

        """
        Args
        ----
        tgt_ids:
            [B, T]

        memory:
            [B, S, D]

        memory_padding_mask:
            [B, S]

        Returns
        -------
        logits:
            [B, T, vocab_size]
        """

        # -------------------------------------------------
        # Padding Mask
        # -------------------------------------------------
        tgt_padding_mask = (
            tgt_ids == PAD_IDX
        )

        # -------------------------------------------------
        # Embedding
        # -------------------------------------------------
        tgt = self.embedding(tgt_ids)

        tgt = tgt * math.sqrt(self.embed_dim)

        tgt = self.position(tgt)

        tgt = self.dropout(tgt)

        # -------------------------------------------------
        # Causal Mask
        # -------------------------------------------------
        tgt_mask = self.generate_causal_mask(
            size=tgt.size(1),
            device=tgt.device
        )

        # -------------------------------------------------
        # Transformer Decoder
        # -------------------------------------------------
        out = self.decoder(

            tgt=tgt,

            memory=memory,

            tgt_mask=tgt_mask,

            tgt_key_padding_mask=tgt_padding_mask,

            memory_key_padding_mask=memory_padding_mask
        )

        # -------------------------------------------------
        # Vocabulary Projection
        # -------------------------------------------------
        logits = self.fc(out)

        return logits

    # =====================================================
    # Greedy Search Decoding
    # =====================================================
    @torch.no_grad()
    def greedy_decode(
        self,
        memory,
        memory_padding_mask=None,
        max_len=50,
        device="cuda"
    ):

        self.eval()

        # -----------------------------------------
        # Start with SOS
        # -----------------------------------------
        generated = torch.tensor(
            [[SOS_IDX]],
            device=device
        )

        for _ in range(max_len):

            logits = self.forward(
                tgt_ids=generated,
                memory=memory,
                memory_padding_mask=memory_padding_mask
            )

            # Last token prediction
            next_token_logits = logits[:, -1, :]

            # Greedy choice
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            # Append token
            generated = torch.cat(
                [generated, next_token],
                dim=1
            )

            # Stop if EOS
            if next_token.item() == EOS_IDX:
                break

        return generated.squeeze(0).tolist()

    # =====================================================
    # Beam Search Decoding
    # =====================================================
    @torch.no_grad()
    def beam_search_decode(
        self,
        memory,
        memory_padding_mask=None,
        beam_size=5,
        max_len=50,
        alpha=0.6,
        device="cuda"
    ):

        """
        Beam Search with Length Penalty

        Args
        ----
        memory:
            [1, S, D]

        beam_size:
            Number of beams

        alpha:
            Length penalty strength
        """

        self.eval()

        # =================================================
        # Initial Beam
        # =================================================
        beams = [
            (
                torch.tensor([[SOS_IDX]], device=device),
                0.0
            )
        ]

        completed = []

        # =================================================
        # Decoding Loop
        # =================================================
        for _ in range(max_len):

            candidates = []

            for seq, score in beams:

                # -----------------------------------------
                # Stop expanding EOS
                # -----------------------------------------
                if seq[0, -1].item() == EOS_IDX:

                    completed.append((seq, score))
                    continue

                # -----------------------------------------
                # Forward
                # -----------------------------------------
                logits = self.forward(
                    tgt_ids=seq,
                    memory=memory,
                    memory_padding_mask=memory_padding_mask
                )

                # Last token
                logits = logits[:, -1, :]

                # Log probs
                log_probs = F.log_softmax(
                    logits,
                    dim=-1
                )

                # Top-k
                topk_log_probs, topk_ids = torch.topk(
                    log_probs,
                    beam_size,
                    dim=-1
                )

                # -----------------------------------------
                # Expand Beams
                # -----------------------------------------
                for k in range(beam_size):

                    next_token = topk_ids[0, k].view(1, 1)

                    next_score = (
                        score +
                        topk_log_probs[0, k].item()
                    )

                    next_seq = torch.cat(
                        [seq, next_token],
                        dim=1
                    )

                    candidates.append(
                        (next_seq, next_score)
                    )

            # ---------------------------------------------
            # No Candidates
            # ---------------------------------------------
            if len(candidates) == 0:
                break

            # =============================================
            # Length Penalty
            # =============================================
            def normalized_score(item):

                seq, score = item

                length = seq.size(1)

                return score / (length ** alpha)

            # ---------------------------------------------
            # Keep Top Beams
            # ---------------------------------------------
            candidates = sorted(
                candidates,
                key=normalized_score,
                reverse=True
            )

            beams = candidates[:beam_size]

        # =================================================
        # Final Selection
        # =================================================
        completed.extend(beams)

        completed = sorted(
            completed,
            key=lambda x: (
                x[1] / (x[0].size(1) ** alpha)
            ),
            reverse=True
        )

        best_seq = completed[0][0]

        return best_seq.squeeze(0).tolist()