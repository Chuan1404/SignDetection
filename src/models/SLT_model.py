import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional_encoding import PositionalEncoding

from config import _N_POSE, _N_HAND_FLAGS, _N_LIPS


class ClassificationOutput:

    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits


def masked_mean_pool(x, video_mask):
    mask = video_mask.unsqueeze(-1).float()          # (B, T, 1)
    summed = (x * mask).sum(dim=1)                    # (B, D)
    counts = mask.sum(dim=1).clamp(min=1.0)           # (B, 1) — avoid /0
    return summed / counts


class SignLanguageTranslatorV1(nn.Module):

    def __init__(
        self,
        input_dim=137,
        hidden_dim=512,
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

from src.models.spatial_graph import (
    _build_two_hand_adjacency, NUM_FULL_BODY_NODES,
    _STGCNBlock, build_full_body_adjacency,
    _LEFT_WRIST, _RIGHT_WRIST, _N_HAND,
    _STBiG_GCNBlock
)

class SignLanguageTranslatorV2(nn.Module):

    NUM_NODES = NUM_FULL_BODY_NODES

    def __init__(
        self,
        channels=(32, 64, 64, 128),
        kernel_size=9,
        dropout=0.2,
        num_classes=2000,
    ):
        super().__init__()

        self.register_buffer("body_adjacency", build_full_body_adjacency())

        ch = [3, *channels]
        self.blocks = nn.ModuleList([
            _STGCNBlock(
                ch[i], ch[i + 1],
                adjacency=self.body_adjacency,
                dropout=dropout,
                kernel_size=kernel_size
            )
            for i in range(len(ch) - 1)
        ])
        out_dim = channels[-1]

        self.classifier = nn.Linear(out_dim, num_classes)

        self._left_hand_slice = slice(_LEFT_WRIST, _LEFT_WRIST + _N_HAND)
        self._right_hand_slice = slice(_RIGHT_WRIST, _RIGHT_WRIST + _N_HAND)

        # Fixed node-group weights: pose=0.1 | lips=0.3 | hand=0.6
        # Mỗi node trong nhóm nhận weight_nhóm / số_node_nhóm
        # → tổng toàn bộ node = 0.1 + 0.3 + 0.3 + 0.3 = 1.0
        _LIPS_START = _N_POSE
        node_weights = torch.zeros(self.NUM_NODES)
        node_weights[0:_N_POSE]                           = 0.1 / _N_POSE   # pose
        node_weights[_LIPS_START:_LIPS_START + _N_LIPS]   = 0.3 / _N_LIPS   # lips
        node_weights[_LEFT_WRIST:_LEFT_WRIST + _N_HAND]   = 0.3 / _N_HAND   # tay trái
        node_weights[_RIGHT_WRIST:_RIGHT_WRIST + _N_HAND] = 0.3 / _N_HAND   # tay phải
        self.register_buffer("node_weights", node_weights)

    def encode(self, features, video_mask):
        video_mask = video_mask.bool()  # an toàn dù gọi trực tiếp hay qua forward()
        B, T, _ = features.shape

        coords = features[..., :self.NUM_NODES * 3].reshape(B, T, self.NUM_NODES, 3)
        left_present = features[..., -2]  # (B, T)
        right_present = features[..., -1]  # (B, T)

        node_mask = torch.ones(B, T, self.NUM_NODES, device=features.device, dtype=coords.dtype)
        node_mask[:, :, self._left_hand_slice] = left_present.unsqueeze(-1)
        node_mask[:, :, self._right_hand_slice] = right_present.unsqueeze(-1)

        x = coords * node_mask.unsqueeze(-1)

        for block in self.blocks:
            x = block(x)  # (B, T, N, C)
            
        # Masked mean pool theo chiều T
        combined_mask = video_mask.unsqueeze(-1).float() * node_mask  # (B, T, N)
        combined_mask = combined_mask.unsqueeze(-1)                   # (B, T, N, 1)
        x = (x * combined_mask).sum(dim=1) / combined_mask.sum(dim=1).clamp(min=1.0)  # (B, N, C)

        # Fixed weighted sum theo chiều N
        w = self.node_weights.view(1, self.NUM_NODES, 1)  # (1, N, 1)
        x = (x * w).sum(dim=1)                           # (B, C)

        return x

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")

        pooled = self.encode(features, video_mask)
        logits = self.classifier(pooled)

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

class SignLanguageTranslatorV3(nn.Module):

    NUM_NODES = NUM_FULL_BODY_NODES

    def __init__(
        self,
        input_dim=257,
        hidden_dim=256,
        num_encoder_layers=6,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.2,
        max_seq_len=5000,
        num_classes=2000,
        channels=(32, 64, 64, 128),
        kernel_size=9,
        aux_loss_weight_a=0.3,
        aux_loss_weight_b=0.3,
    ):
        super().__init__()

        d_model = hidden_dim

        self.hand_feature_start = _N_POSE * 3 + _N_LIPS * 3
        self.hand_feature_end   = input_dim - _N_HAND_FLAGS

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
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

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.register_buffer("hand_adjacency", _build_two_hand_adjacency())

        ch = [3, *channels]
        self.blocks = nn.ModuleList([
            _STGCNBlock(
                ch[i], ch[i + 1],
                adjacency=self.hand_adjacency,
                dropout=dropout,
                kernel_size=kernel_size
            )
            for i in range(len(ch) - 1)
        ])
        hand_out_dim = channels[-1]

        # ---------------- Fusion + classifiers ----------------
        self.aux_loss_weight_a = aux_loss_weight_a
        self.aux_loss_weight_b = aux_loss_weight_b

        self.fusion = nn.Sequential(
            nn.Linear(d_model + hand_out_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        self.classifier = nn.Linear(d_model, num_classes)

        self.stream_a_aux_classifier = nn.Linear(d_model, num_classes)
        self.stream_b_aux_classifier = nn.Linear(hand_out_dim, num_classes)

    def get_optimizer_param_groups(self, lr):
        return [{"params": self.parameters(), "lr": lr, "name": "all"}]

    # -------- Stream A: transformer encoder (giữ nguyên logic của V1.encode) --------
    def encode_stream_a(self, features, video_mask):
        x = self.input_projection(features)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        x = self.encoder_norm(x)                       # (B, T, d_model)
        return x

    # -------- Stream B: ST-GCN cho 2 bàn tay --------
    def encode_stream_b(self, features, hand_mask):

        B, T, _ = features.shape

        x = features.reshape(B, T, self.NUM_HAND_NODES, 3)

        for block in self.hand_blocks:
            x = block(x)                                # (B, T, N, C)

        # Pool chỉ trên frames có tay hiện diện
        mask = hand_mask.unsqueeze(-1).unsqueeze(-1).float()    # (B, T, 1, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)                  # (B, 1, 1)
        x = (x * mask).sum(dim=1) / denom                      # (B, N, C)
        x = x.mean(dim=1)                                       # (B, C)

        return x

    def forward(self, features, hand_normalize_features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")
        video_mask = video_mask.bool()

        seq_a = self.encode_stream_a(features, video_mask)
        pooled_a = masked_mean_pool(seq_a, video_mask)             # (B, d_model)

        # hand_features = features[:, :, self.hand_feature_start:self.hand_feature_end]
        pooled_b = self.encode_stream_b(hand_normalize_features, video_mask)

        # Fusion
        fused = self.fusion(torch.cat([pooled_a, pooled_b], dim=-1))
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

            # Deep supervision riêng cho từng nhánh, mỗi nhánh 1 trọng số độc lập —
            # cho phép tăng/giảm hoặc tắt hẳn (weight=0) aux loss của 1 nhánh mà
            # không ảnh hưởng nhánh còn lại (vd: nhánh tay nhiễu hơn -> hạ weight_b).
            if self.aux_loss_weight_a > 0:
                aux_logits_a = self.stream_a_aux_classifier(pooled_a)
                loss = loss + self.aux_loss_weight_a * F.cross_entropy(aux_logits_a, labels)

            if self.aux_loss_weight_b > 0:
                aux_logits_b = self.stream_b_aux_classifier(pooled_b)
                loss = loss + self.aux_loss_weight_b * F.cross_entropy(aux_logits_b, labels)

        return ClassificationOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def predict(self, features, hand_normalize_features, video_mask=None, top_k=1):
        logits = self.forward(
            features, hand_normalize_features, video_mask=video_mask
        ).logits
        if top_k == 1:
            return logits.argmax(dim=-1)
        return logits.topk(k=top_k, dim=-1).indices

class SignLanguageTranslatorV4(nn.Module):
    NUM_NODES = NUM_FULL_BODY_NODES

    def __init__(
        self,
        channels=(64, 64, 64, 64, 128, 128, 128, 128, 256, 256),
        kernel_size=9,
        dropout=0.2,
        num_classes=2000,
    ):
        super().__init__()

        self.register_buffer("body_adjacency", build_full_body_adjacency())

        # V4 has 10 blocks. The input to the first block is 3 coords (since we multiply by mask beforehand).
        ch = [3] + list(channels)
        
        self.blocks = nn.ModuleList()
        for i in range(len(ch) - 1):
            # Downsample and use EMA at the 5th block (idx 4) and 8th block (idx 7)
            stride = 2 if i in [4, 7] else 1
            use_ema = True if i in [4, 7] else False
            
            self.blocks.append(
                _STBiG_GCNBlock(
                    in_ch=ch[i],
                    out_ch=ch[i + 1],
                    adjacency=self.body_adjacency,
                    dropout=dropout,
                    kernel_size=kernel_size,
                    stride=stride,
                    use_ema=use_ema
                )
            )

        out_dim = channels[-1]

        self.classifier = nn.Linear(out_dim, num_classes)

        self._left_hand_slice = slice(_LEFT_WRIST, _LEFT_WRIST + _N_HAND)
        self._right_hand_slice = slice(_RIGHT_WRIST, _RIGHT_WRIST + _N_HAND)

        # Fixed node-group weights: pose=0.1 | lips=0.3 | hand=0.6
        _LIPS_START = _N_POSE
        node_weights = torch.zeros(self.NUM_NODES)
        node_weights[0:_N_POSE]                           = 0.1 / _N_POSE   # pose
        node_weights[_LIPS_START:_LIPS_START + _N_LIPS]   = 0.2 / _N_LIPS   # lips
        node_weights[_LEFT_WRIST:_LEFT_WRIST + _N_HAND]   = 0.35 / _N_HAND   # tay trái
        node_weights[_RIGHT_WRIST:_RIGHT_WRIST + _N_HAND] = 0.35 / _N_HAND   # tay phải
        self.register_buffer("node_weights", node_weights)

    def encode(self, features, video_mask):
        video_mask = video_mask.bool()
        B, T, _ = features.shape

        coords = features[..., :self.NUM_NODES * 3].reshape(B, T, self.NUM_NODES, 3)
        left_present = features[..., -2]  # (B, T)
        right_present = features[..., -1]  # (B, T)

        node_mask = torch.ones(B, T, self.NUM_NODES, device=features.device, dtype=coords.dtype)
        node_mask[:, :, self._left_hand_slice] = left_present.unsqueeze(-1)
        node_mask[:, :, self._right_hand_slice] = right_present.unsqueeze(-1)

        x = coords * node_mask.unsqueeze(-1)

        for i, block in enumerate(self.blocks):
            x = block(x)  # (B, T_out, N, C)
            if i in [4, 7]:
                video_mask = video_mask[:, ::2]
                node_mask = node_mask[:, ::2, :]

        # Masked mean pool theo chiều T
        combined_mask = video_mask.unsqueeze(-1).float() * node_mask  # (B, T_out, N)
        combined_mask = combined_mask.unsqueeze(-1)                   # (B, T_out, N, 1)
        x = (x * combined_mask).sum(dim=1) / combined_mask.sum(dim=1).clamp(min=1.0)  # (B, N, C)

        # Fixed weighted sum theo chiều N
        w = self.node_weights.view(1, self.NUM_NODES, 1)  # (1, N, 1)
        x = (x * w).sum(dim=1)                           # (B, C)

        return x

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")

        pooled = self.encode(features, video_mask)
        logits = self.classifier(pooled)

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