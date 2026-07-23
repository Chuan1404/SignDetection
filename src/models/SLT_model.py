import torch
import torch.nn as nn

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
        input_dim=143,
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

    def __init__(self, in_ch, out_ch, dropout=0.0, kernel_size=9):
        super().__init__()

        self.spatial = SpatialGraphConv(
            in_channels=in_ch,
            out_channels=out_ch
        )
        self.spatial_bn = nn.BatchNorm1d(out_ch)
        self.spatial_relu = nn.ReLU()

        # Temporal Conv
        self.temporal = _TemporalConv(out_ch, kernel_size=kernel_size, dropout=dropout)
        self.temporal_relu = nn.ReLU()

        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        B, T, N, C = x.shape
        res = x

        x = self.spatial(x)  # (B, T, N, C_out)
        # BN trên channel — reshape để BatchNorm1d hoạt động
        x = self.spatial_bn(
            x.reshape(B * T * N, -1)
        ).reshape(B, T, N, -1)

        x = self.spatial_relu(x)

        # Temporal Conv
        x = self.temporal(x)  # (B, T, N, C_out)
        # Residual — cần (B, C, T, N) cho Conv2d
        print(not isinstance(self.residual, nn.Identity))
        if not isinstance(self.residual, nn.Identity):
            res = res.permute(0, 3, 1, 2)  # (B, C_in, T, N)
            res = self.residual(res)  # (B, C_out, T, N)
            res = res.permute(0, 2, 3, 1)  # (B, T, N, C_out)

        return self.temporal_relu(x + res)


class SignLanguageTranslatorV4(nn.Module):

    NUM_JOINTS = NUM_NODES

    def __init__(
            self,
            num_classes=2000,
            dropout=0.25,  # paper dùng dropout + DropGraph
            kernel_size=9  # temporal kernel — paper dùng kt=9
    ):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(self.NUM_JOINTS * 3)

        # 10 ST-GCN blocks theo Figure 6
        ch = [3, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        # ch = [3, 32, 64, 64, 64, 128, 128, 256]
        self.blocks = nn.ModuleList([
            _STGCNBlock(ch[i], ch[i + 1], dropout=dropout, kernel_size=kernel_size)
            for i in range(10)
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

        x = self.input_bn(
            joint_coords.reshape(B * T, self.NUM_JOINTS * 3)
        ).reshape(B, T, self.NUM_JOINTS, 3)  # (B, T, 59, 3)


        # 10 ST-GCN blocks
        for block in self.blocks:
            x = block(x)
            print(f"x shape: {x.shape}")

        # Masked mean pool qua T — bỏ padding frames
        mask = video_mask.bool().unsqueeze(-1).unsqueeze(-1).float()  # (B, T, 1, 1)
        x = (x * mask).sum(dim=1)  # (B, 59, 256)
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

"""
V5 — STBiG-GCN (Spatio-Temporal Bidirectional Gated GCN)
Dựa trên: Yin, Wang, Zhou, Wang, Sun, Deng —
"Spatio-Temporal Bidirectional Gated Graph Convolutional Network for
Skeleton Action Recognition in Dynamic Complex Environments"
IEEE Internet of Things Journal, Vol. 12, No. 21, 1 Nov 2025.

Ý tưởng chính của paper, áp dụng vào bài toán SLT (thay ST-GCN gốc của V4):

1) BiG-TCN thay cho temporal conv 1 chiều thường:
   - Forward branch : conv nhân quả trên chuỗi gốc (chỉ nhìn quá khứ).
   - Backward branch: đảo ngược trục T, conv nhân quả, rồi đảo lại
     (mô phỏng luồng thông tin "từ tương lai về hiện tại").
   - Gate g(t) = sigmoid(Wg·x(t)+bg) trộn động 2 luồng:
       out = forward*g + backward*(1-g)
     thay vì cộng cứng như Bi-TCN thường -> paper cho thấy gating > cộng cứng.

2) EMA (Efficient Multi-scale Attention, Ouyang et al. 2023) — thay cho
   SE/CBAM ở phần skip connection. Ở đây feature map dạng (B, C, T, N)
   được coi tương đương ảnh (C, H, W) với T~H (thời gian), N~W (khớp
   xương): chia nhóm kênh, pool riêng theo T và theo N, nhánh conv1x1
   (ngữ nghĩa toàn cục) + nhánh conv3x3 (chi tiết cục bộ), rồi
   "cross-space learning" (matmul + softmax) để tạo trọng số attention
   cuối cùng.

3) Skip connections + EMA: lấy đặc trưng ngay sau khối GCN5 (128 kênh)
   và GCN8 (256 kênh) — đúng 2 vị trí downsample thời gian (stride=2)
   theo Fig.7(b) của paper — cho qua EMA rồi cộng residual (sau khi
   chiếu kênh + adaptive-pool khớp kích thước) vào đặc trưng cuối cùng
   (256 kênh) trước khi pooling & phân loại.

Cấu trúc 10 khối vẫn giữ nguyên số kênh như paper (Fig.7b) và như V4:
    ch = [3, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
    stride = 2 ở khối thứ 5 (L5, idx=4) và khối thứ 8 (L8, idx=7).

Module này độc lập, import từ `spatial_graph.py` sẵn có (đồ thị 59 node,
3 phân vùng root/centripetal/centrifugal) — không đổi phần đồ thị không
gian, chỉ thay phần thời gian (TCN -> BiG-TCN) và thêm nhánh skip+EMA,
đúng tinh thần "giữ nguyên spatial module, cải tiến temporal module +
attention" mà paper đề xuất.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.spatial_graph import SpatialGraphConv, NUM_NODES
from src.models.SLT_model import ClassificationOutput, masked_mean_pool  # noqa: F401 (giữ tương thích API)


# =============================================================================
# 1) BiG-TCN — Bidirectional Gated Temporal Convolution
# =============================================================================
class BiGTCN(nn.Module):
    """
    Thay cho temporal conv 1 chiều (nhân quả một phía) trong ST-GCN gốc.

    Input / Output: (B, C, T, N)  — giống layout dùng trong _TemporalConv
    của V4, để có thể tái sử dụng hạ tầng permute hiện có.
    """

    def __init__(self, channels, kernel_size=9, stride=1, dropout=0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self._pad = kernel_size - 1  # pad "về phía quá khứ" để giữ tính nhân quả

        # --- Forward TCN: nhìn quá khứ (t-k+1 .. t) ---
        self.forward_conv = nn.Conv2d(
            channels, channels, kernel_size=(kernel_size, 1),
            stride=(stride, 1), padding=0, bias=False
        )
        self.forward_bn = nn.BatchNorm2d(channels)
        self.forward_drop = nn.Dropout(dropout)

        # --- Backward TCN: chạy trên chuỗi đảo ngược -> "nhìn tương lai" ---
        self.backward_conv = nn.Conv2d(
            channels, channels, kernel_size=(kernel_size, 1),
            stride=(stride, 1), padding=0, bias=False
        )
        self.backward_bn = nn.BatchNorm2d(channels)
        self.backward_drop = nn.Dropout(dropout)

        # --- Gate: g(t) = sigmoid(Wg . x(t) + bg), công thức (8) trong paper ---
        self.gate_conv = nn.Conv2d(
            channels, channels, kernel_size=(kernel_size, 1),
            stride=(stride, 1), padding=0, bias=True
        )
        self.gate_bn = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        """x: (B, C, T, N) -> (B, C, T', N), T' = T nếu stride=1 else ~T/stride"""

        # Forward path (công thức 3)
        x_fwd_in = F.pad(x, (0, 0, self._pad, 0))          # pad trái theo T
        fwd = self.forward_conv(x_fwd_in)
        fwd = self.relu(self.forward_drop(self.forward_bn(fwd)))

        # Backward path (công thức 4-7): đảo T -> conv nhân quả -> đảo lại
        x_rev = torch.flip(x, dims=[2])
        x_bwd_in = F.pad(x_rev, (0, 0, self._pad, 0))
        bwd = self.backward_conv(x_bwd_in)
        bwd = self.relu(self.backward_drop(self.backward_bn(bwd)))
        bwd = torch.flip(bwd, dims=[2])                     # khôi phục đúng chiều thời gian

        # Gating mechanism (công thức 8-12)
        x_gate_in = F.pad(x, (0, 0, self._pad, 0))
        gate = torch.sigmoid(self.gate_bn(self.gate_conv(x_gate_in)))

        out = fwd * gate + bwd * (1.0 - gate)
        return out


# =============================================================================
# 2) EMA — Efficient Multi-scale Attention (Ouyang et al., 2023)
#    Áp dụng cho skip connections theo Fig.5/Fig.6 của paper STBiG-GCN.
# =============================================================================
class EMA(nn.Module):
    """
    Feature map (B, C, T, N) được coi như ảnh (C, H, W) với H<->T
    (thời gian), W<->N (khớp xương).

    Lưu ý: đây là bản triển khai rút gọn, giữ đúng tinh thần của EMA
    (chia nhóm kênh, 2 nhánh pooling theo từng trục, nhánh 1x1 (ngữ
    nghĩa toàn cục) + nhánh 3x3 (chi tiết cục bộ), fuse chéo bằng
    softmax + matmul, tái trọng số bằng sigmoid) — không phải bản port
    1-1 từng phép toán trong ảnh gốc, vì input ở đây là skeleton chứ
    không phải ảnh RGB.
    """

    def __init__(self, channels, groups=16):
        super().__init__()
        assert channels % groups == 0, "channels phải chia hết cho groups"
        self.groups = groups
        gc = channels // groups

        self.pool_T = nn.AdaptiveAvgPool2d((None, 1))   # pool theo N, giữ T
        self.pool_N = nn.AdaptiveAvgPool2d((1, None))   # pool theo T, giữ N

        self.conv1x1 = nn.Conv2d(gc, gc, kernel_size=1)
        self.conv3x3 = nn.Conv2d(gc, gc, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(gc, gc)

    def forward(self, x):
        B, C, T, N = x.shape
        G = self.groups
        gc = C // G

        x_g = x.reshape(B * G, gc, T, N)

        # --- nhánh 1x1: attention theo 2 trục T & N (giống toạ độ X/Y trong Fig.5) ---
        x_t = self.pool_T(x_g)                              # (BG, gc, T, 1)
        x_n = self.pool_N(x_g).permute(0, 1, 3, 2)           # (BG, gc, N, 1)
        tn = self.conv1x1(torch.cat([x_t, x_n], dim=2))      # (BG, gc, T+N, 1)
        t_att, n_att = torch.split(tn, [T, N], dim=2)

        x1 = x_g * t_att.sigmoid() * n_att.permute(0, 1, 3, 2).sigmoid()
        x1 = self.group_norm(x1)

        # --- nhánh 3x3: đặc trưng không gian-thời gian cục bộ ---
        x2 = self.conv3x3(x_g)

        # --- cross-space learning: fuse 2 nhánh bằng softmax + matmul ---
        x1_flat = x1.reshape(B * G, gc, -1)
        x2_flat = x2.reshape(B * G, gc, -1)

        att1 = torch.softmax(x1.mean(dim=1, keepdim=True).reshape(B * G, 1, -1), dim=-1)
        att2 = torch.softmax(x2.mean(dim=1, keepdim=True).reshape(B * G, 1, -1), dim=-1)

        out1 = torch.matmul(x2_flat, att1.transpose(1, 2))   # (BG, gc, 1)
        out2 = torch.matmul(x1_flat, att2.transpose(1, 2))   # (BG, gc, 1)

        weight = torch.sigmoid(out1 + out2).reshape(B * G, gc, 1, 1)

        out = (x_g * weight).reshape(B, C, T, N)
        return out


# =============================================================================
# 3) Skip block — chiếu kênh (nếu cần) + adaptive-pool khớp kích thước
# =============================================================================
class SkipBlock(nn.Module):
    """Dùng cho skip1 (128->256) và skip2 (256->256) theo Fig.6."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x, target_size):
        """x: (B, C_in, T, N) -> (B, C_out, T', N') khớp target_size = (T', N')"""
        x = self.proj(x)
        x = F.adaptive_avg_pool2d(x, target_size)
        return x


# =============================================================================
# 4) Khối cơ bản: Spatial GCN (dùng lại SpatialGraphConv) + BiG-TCN + residual
# =============================================================================
class _STBiGGCNBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout=0.0, kernel_size=9, stride=1):
        super().__init__()

        self.spatial = SpatialGraphConv(in_channels=in_ch, out_channels=out_ch)
        self.spatial_bn = nn.BatchNorm1d(out_ch)
        self.spatial_relu = nn.ReLU()

        self.temporal = BiGTCN(out_ch, kernel_size=kernel_size, stride=stride, dropout=dropout)

        if in_ch != out_ch or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.residual = nn.Identity()

        self.out_relu = nn.ReLU()

    def forward(self, x):
        """x: (B, T, N, C_in) -> (B, T', N, C_out)"""
        B, T, N, C = x.shape
        res = x.permute(0, 3, 1, 2)  # (B, C_in, T, N), dùng cho residual

        # Spatial GCN (giữ nguyên như V4)
        x = self.spatial(x)  # (B, T, N, C_out)
        x = self.spatial_bn(x.reshape(B * T * N, -1)).reshape(B, T, N, -1)
        x = self.spatial_relu(x)

        # BiG-TCN thay cho temporal conv thường
        x = x.permute(0, 3, 1, 2)     # (B, C_out, T, N)
        x = self.temporal(x)          # (B, C_out, T', N)

        if not isinstance(self.residual, nn.Identity):
            res = self.residual(res)   # (B, C_out, T', N)

        # phòng trường hợp T lệch do padding/stride giữa 2 nhánh
        Tout = x.shape[2]
        if res.shape[2] != Tout:
            res = F.adaptive_avg_pool2d(res, (Tout, res.shape[3]))

        out = self.out_relu(x + res)     # (B, C_out, T', N)
        return out.permute(0, 2, 3, 1)    # -> (B, T', N, C_out)


# =============================================================================
# 5) SignLanguageTranslatorV5 — STBiG-GCN đầy đủ
# =============================================================================
class SignLanguageTranslatorV5(nn.Module):
    NUM_JOINTS = NUM_NODES

    def __init__(self, num_classes=2000, dropout=0.25, kernel_size=9, ema_groups=16):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(self.NUM_JOINTS * 3)

        ch = [3, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        strides = [1, 1, 1, 1, 2, 1, 1, 2, 1, 1]

        self.blocks = nn.ModuleList([
            _STBiGGCNBlock(ch[i], ch[i + 1], dropout=dropout,
                            kernel_size=kernel_size, stride=strides[i])
            for i in range(10)
        ])

        # EMA + skip lấy sau khối L5 (idx=4, 128 kênh) và L8 (idx=7, 256 kênh)
        self.ema5 = EMA(128, groups=ema_groups)
        self.ema8 = EMA(256, groups=ema_groups)
        self.skip1 = SkipBlock(128, 256)
        self.skip2 = SkipBlock(256, 256)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def get_optimizer_param_groups(self, lr):
        return [{"params": self.parameters(), "lr": lr, "name": "all"}]

    def encode(self, features, video_mask):
        B, T, _ = features.shape

        joint_coords = features[..., :self.NUM_JOINTS * 3]

        x = self.input_bn(
            joint_coords.reshape(B * T, self.NUM_JOINTS * 3)
        ).reshape(B, T, self.NUM_JOINTS, 3)

        skip1_feat = None
        skip2_feat = None

        for i, block in enumerate(self.blocks):
            x = block(x)  # (B, T_i, N, C_i)
            if i == 4:     # sau GCN5
                skip1_feat = self.ema5(x.permute(0, 3, 1, 2))
            elif i == 7:   # sau GCN8
                skip2_feat = self.ema8(x.permute(0, 3, 1, 2))

        main_feat = x.permute(0, 3, 1, 2)          # (B, 256, T_final, N)
        target_size = (main_feat.shape[2], main_feat.shape[3])

        s1 = self.skip1(skip1_feat, target_size)   # (B, 256, T_final, N)
        s2 = self.skip2(skip2_feat, target_size)   # (B, 256, T_final, N)

        fused = main_feat + s1 + s2
        fused = fused.permute(0, 2, 3, 1)          # (B, T_final, N, 256)

        # Downsample mask cho khớp T_final (2 lần stride=2 -> ~T/4)
        mask = video_mask.bool()
        Tf = fused.shape[1]
        if mask.shape[1] != Tf:
            mask_f = mask.float().unsqueeze(1)               # (B, 1, T)
            mask_f = F.adaptive_max_pool1d(mask_f, Tf)        # (B, 1, Tf)
            mask = mask_f.squeeze(1).bool()

        mask4 = mask.unsqueeze(-1).unsqueeze(-1).float()      # (B, Tf, 1, 1)
        pooled = (fused * mask4).sum(dim=1) / mask4.sum(dim=1).clamp(min=1.0)  # (B, N, 256)
        pooled = pooled.mean(dim=1)                            # (B, 256)

        return pooled

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")

        pooled = self.encode(features, video_mask)  # (B, 256)
        logits = self.classifier(pooled)              # (B, num_classes)

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