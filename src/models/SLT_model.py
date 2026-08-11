import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.positional_encoding import PositionalEncoding

from config import _N_POSE

from src.models.spatial_graph import (
    NUM_FULL_BODY_NODES,
    _STGCNBlock, build_full_body_adjacency,
    _LEFT_WRIST, _RIGHT_WRIST, _N_HAND,
    _STBiG_GCNBlock,
    build_hands_only_adjacency,
    _CausalTemporalConv1d, _FinslerEnergyGate, sinkhorn,
)


class ClassificationOutput:

    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits


def masked_mean_pool(x, video_mask):
    mask = video_mask.unsqueeze(-1).float()  # (B, T, 1)
    summed = (x * mask).sum(dim=1)  # (B, D)
    counts = mask.sum(dim=1).clamp(min=1.0)  # (B, 1) — avoid /0
    return summed / counts


class SignLanguageTranslatorV1(nn.Module):
    def __init__(
            self,
            input_dim=90,
            hidden_dim=128,
            num_encoder_layers=6,
            nhead=8,
            dim_feedforward=256,
            dropout=0,
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

        x = self.input_projection(features)  # (B, T, d_model)
        x = self.pos_encoder(x)  # (B, T, d_model)
        x = self.encoder(
            x,
            src_key_padding_mask=~video_mask  # True = ignore (padding)
        )  # (B, T, d_model)
        x = self.encoder_norm(x)  # (B, T, d_model)

        return x

    def forward(self, features, labels=None, video_mask=None):
        x = self.encode(features, video_mask)
        pooled = masked_mean_pool(x, video_mask.bool())
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


class SignLanguageTranslatorV2(nn.Module):
    NUM_NODES = NUM_FULL_BODY_NODES

    def __init__(
            self,
            channels=(64, 64, 128, 128),
            kernel_size=9,
            dropout=0,
            num_classes=2000,
    ):
        super().__init__()

        self.register_buffer("body_adjacency", build_full_body_adjacency())

        ch = [2, *channels]
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

        node_weights = torch.zeros(self.NUM_NODES)
        node_weights[0:_N_POSE] = 0.1 / _N_POSE  # pose
        node_weights[_LEFT_WRIST:_LEFT_WRIST + _N_HAND] = 0.45 / _N_HAND  # tay trái
        node_weights[_RIGHT_WRIST:_RIGHT_WRIST + _N_HAND] = 0.45 / _N_HAND  # tay phải
        self.register_buffer("node_weights", node_weights)

    def encode(self, features, video_mask):
        video_mask = video_mask.bool()
        B, T, _ = features.shape

        x = features.reshape(B, T, -1, 2)

        for block in self.blocks:
            x = block(x)  # (B, T, N, C)

        # Masked mean pool theo chiều T
        mask = video_mask.unsqueeze(-1).float()
        mask = mask.unsqueeze(-1)  # (B, T, N, 1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (B, N, C)

        # Fixed weighted sum theo chiều N
        w = self.node_weights.view(1, self.NUM_NODES, 1)  # (1, N, 1)
        x = (x * w).sum(dim=1)  # (B, C)

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
            input_dim=90,
            hidden_dim=128,
            num_encoder_layers=3,
            nhead=8,
            dim_feedforward=256,
            dropout=0,
            max_seq_len=5000,
            num_classes=2000,
            channels=(64, 64, 128, 128),
            kernel_size=9,
            aux_loss_weight_a=0.3,
            aux_loss_weight_b=0.3,
    ):
        super().__init__()

        d_model = hidden_dim

        self.hand_feature_start = _N_POSE * 2

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
        self.register_buffer("body_adjacency", build_full_body_adjacency())

        ch = [2, *channels]
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

        # ---------------- Fusion + classifiers ----------------
        self.aux_loss_weight_a = aux_loss_weight_a
        self.aux_loss_weight_b = aux_loss_weight_b

        self.fusion = nn.Sequential(
            nn.Linear(d_model + out_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        self.classifier = nn.Linear(d_model, num_classes)

        self.stream_a_aux_classifier = nn.Linear(d_model, num_classes)
        self.stream_b_aux_classifier = nn.Linear(out_dim, num_classes)

    def get_optimizer_param_groups(self, lr):
        return [{"params": self.parameters(), "lr": lr, "name": "all"}]

    def encode_stream_a(self, features, video_mask):
        x = self.input_projection(features)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=~video_mask)
        x = self.encoder_norm(x)  # (B, T, d_model)
        return x

    def encode_stream_b(self, features, hand_mask):

        B, T, _ = features.shape

        x = features.reshape(B, T, self.NUM_NODES, 2)

        for block in self.blocks:
            x = block(x)  # (B, T, N, C)

        # Pool chỉ trên frames có tay hiện diện
        mask = hand_mask.unsqueeze(-1).unsqueeze(-1).float()  # (B, T, 1, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)  # (B, 1, 1)
        x = (x * mask).sum(dim=1) / denom  # (B, N, C)
        x = x.mean(dim=1)  # (B, C)

        return x

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")
        video_mask = video_mask.bool()

        seq_a = self.encode_stream_a(features, video_mask)
        pooled_a = masked_mean_pool(seq_a, video_mask)  # (B, d_model)

        pooled_b = self.encode_stream_b(features, video_mask)

        fused = self.fusion(torch.cat([pooled_a, pooled_b], dim=-1))
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

            if self.aux_loss_weight_a > 0:
                aux_logits_a = self.stream_a_aux_classifier(pooled_a)
                loss = loss + self.aux_loss_weight_a * F.cross_entropy(aux_logits_a, labels)

            if self.aux_loss_weight_b > 0:
                aux_logits_b = self.stream_b_aux_classifier(pooled_b)
                loss = loss + self.aux_loss_weight_b * F.cross_entropy(aux_logits_b, labels)

        return ClassificationOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def predict(self, features, video_mask=None, top_k=1):
        logits = self.forward(
            features, video_mask=video_mask
        ).logits
        if top_k == 1:
            return logits.argmax(dim=-1)
        return logits.topk(k=top_k, dim=-1).indices


class SignLanguageTranslatorV4(nn.Module):
    NUM_NODES = NUM_FULL_BODY_NODES

    def __init__(
            self,
            channels=(64, 64, 64, 64, 128, 128, 128, 128),
            kernel_size=9,
            dropout=0.2,
            num_classes=2000,
    ):
        super().__init__()

        self.register_buffer("body_adjacency", build_full_body_adjacency())

        # V4 has 10 blocks. The input to the first block is 3 coords (since we multiply by mask beforehand).
        ch = [2] + list(channels)

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
        node_weights[0:_N_POSE] = 0.2 / _N_POSE  # pose
        node_weights[_LEFT_WRIST:_LEFT_WRIST + _N_HAND] = 0.4 / _N_HAND  # tay trái
        node_weights[_RIGHT_WRIST:_RIGHT_WRIST + _N_HAND] = 0.4 / _N_HAND  # tay phải
        self.register_buffer("node_weights", node_weights)

    def encode(self, features, video_mask):
        video_mask = video_mask.bool()
        B, T, _ = features.shape

        x = features.reshape(B, T, self.NUM_NODES, -1)

        for i, block in enumerate(self.blocks):
            x = block(x)  # (B, T_out, N, C)
            if i in [4, 7]:
                video_mask = video_mask[:, ::2]

        # Masked mean pool theo chiều T
        combined_mask = video_mask.unsqueeze(-1).float()
        combined_mask = combined_mask.unsqueeze(-1)
        x = (x * combined_mask).sum(dim=1) / combined_mask.sum(dim=1).clamp(min=1.0)

        # Fixed weighted sum theo chiều N
        w = self.node_weights.view(1, self.NUM_NODES, 1)
        x = (x * w).sum(dim=1)

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


class SignLanguageTranslatorV5(nn.Module):
    """
    DSLNet-style dual-reference, dual-stream architecture, following
    "Skeleton-based Sign Language Recognition using a Dual-Stream
    Spatio-Temporal Dynamic Graph Convolutional Network" (arXiv:2509.08661).

    Pipeline:
      1. Dual-reference frame normalization (Eqs. 1-2): the raw skeleton is
         decomposed into (a) a wrist-centric "shape" representation of the
         two hands, translation-invariant per hand, and (b) a facial-centric
         "trajectory" representation of both wrists, invariant to the
         signer's pose/scale.
      2. Dual-stream feature extraction:
           - TSSN (morphology stream): a stack of topology-aware STGC blocks
             over the two hand graphs, with multi-scale fusion, a BiLSTM,
             and attention pooling into a single feature F_s.
           - FTDE (trajectory stream): a causal ST-Conv + BiLSTM encoder
             modulated by a learnable, physics-informed ("Finsler") energy
             weighting (Eq. 3), producing a per-timestep sequence F_t.
      3. Cross-attention enhancement between the two streams, followed by a
         geometry-driven Optimal Transport (Geo-OT) fusion (Sec. 2.3) that
         softly aligns the trajectory sequence onto the morphological
         feature before concatenation and classification.
      4. An auxiliary geometric consistency loss L_geo (Eq. 5) encourages
         cosine alignment between the two streams' projected features,
         added to the standard cross-entropy loss (Eq. 4).

    Expects the same flattened (x, y) skeleton input as V2/V4
    (features: (B, T, NUM_NODES * 2)), where node layout is
    [face/pose nodes | left hand (21) | right hand (21)].
    """

    NUM_NODES = NUM_FULL_BODY_NODES

    def __init__(
            self,
            d_model=128,
            num_classes=2000,
            # TSSN (morphology / shape stream over both hands)
            shape_channels=(64, 64, 128, 128),
            shape_kernel_size=9,
            shape_lstm_layers=1,
            shape_nhead=4,
            # FTDE (trajectory stream over both wrists, facial-anchored frame)
            traj_hidden=128,
            traj_kernel_size=9,
            traj_lstm_layers=1,
            # Geo-OT fusion
            ot_iters=20,
            ot_epsilon=0.1,
            geo_loss_weight=0.3,
            dropout=0.1,
            eps=1e-6,
    ):
        super().__init__()
        self.eps = eps
        self.geo_loss_weight = geo_loss_weight

        self._left_hand_slice = slice(_LEFT_WRIST, _LEFT_WRIST + _N_HAND)
        self._right_hand_slice = slice(_RIGHT_WRIST, _RIGHT_WRIST + _N_HAND)

        # ---------------- Stream A: TSSN (morphology / shape) ----------------
        self.register_buffer("hands_adjacency", build_hands_only_adjacency())

        ch = [2, *shape_channels]
        self.shape_blocks = nn.ModuleList([
            _STGCNBlock(
                ch[i], ch[i + 1],
                adjacency=self.hands_adjacency,
                dropout=dropout,
                kernel_size=shape_kernel_size,
            )
            for i in range(len(ch) - 1)
        ])
        # Multi-scale fusion: project each block's node-pooled output into a
        # shared d_model space, then sum ("Features from multiple STGC
        # blocks are aggregated to form a rich, multi-scale representation").
        self.multiscale_proj = nn.ModuleList([
            nn.Linear(c, d_model) for c in shape_channels
        ])

        self.shape_lstm = nn.LSTM(
            input_size=d_model, hidden_size=d_model // 2,
            num_layers=shape_lstm_layers, batch_first=True, bidirectional=True,
        )
        self.shape_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.shape_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=shape_nhead, batch_first=True, dropout=dropout
        )
        self.shape_norm = nn.LayerNorm(d_model)

        # ---------------- Stream B: FTDE (trajectory) ----------------
        traj_in_dim = 4  # (left wrist x,y) + (right wrist x,y), facial-anchored frame
        self.traj_energy_gate = _FinslerEnergyGate(in_channels=traj_in_dim, hidden_dim=32, eps=eps)
        self.traj_conv = _CausalTemporalConv1d(
            traj_in_dim, traj_hidden, kernel_size=traj_kernel_size, dropout=dropout
        )
        self.traj_lstm = nn.LSTM(
            input_size=traj_hidden, hidden_size=d_model // 2,
            num_layers=traj_lstm_layers, batch_first=True, bidirectional=True,
        )
        self.traj_norm = nn.LayerNorm(d_model)

        # ---------------- Cross-attention enhancement ----------------
        self.cross_attn_s2t = nn.MultiheadAttention(
            d_model, num_heads=shape_nhead, batch_first=True, dropout=dropout
        )
        self.cross_attn_t2s = nn.MultiheadAttention(
            d_model, num_heads=shape_nhead, batch_first=True, dropout=dropout
        )

        # ---------------- Geo-OT fusion ----------------
        self.ot_iters = ot_iters
        self.ot_epsilon = ot_epsilon
        self.ot_cost_proj_s = nn.Linear(d_model, d_model)
        self.ot_cost_proj_t = nn.Linear(d_model, d_model)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.classifier = nn.Linear(d_model, num_classes)

        # Geometric consistency projection heads (Eq. 5)
        self.geo_head_s = nn.Linear(d_model, d_model)
        self.geo_head_t = nn.Linear(d_model, d_model)

    def get_optimizer_param_groups(self, lr):
        return [{"params": self.parameters(), "lr": lr, "name": "all"}]

    # -------- Dual-reference frame normalization (Eqs. 1-2) --------
    def _dual_reference_normalize(self, coords):
        """
        coords: (B, T, NUM_NODES, 2) raw (already [-1, 1]-normalized) skeleton.
        Returns:
            shape_feats: (B, T, 2*_N_HAND, 2) — each hand re-centered on its
                         own wrist, translation-invariant (Eq. 1).
            traj_feats:  (B, T, 4) — both wrists' positions in a facial
                         centroid/scale-anchored frame (Eq. 2).
        """
        face = coords[:, :, : self._left_hand_slice.start, :]  # (B, T, _N_POSE, 2)
        left_hand = coords[:, :, self._left_hand_slice, :]  # (B, T, _N_HAND, 2)
        right_hand = coords[:, :, self._right_hand_slice, :]  # (B, T, _N_HAND, 2)

        # --- Wrist Morphological Frame (Eq. 1) ---
        left_wrist = left_hand[:, :, 0:1, :]
        right_wrist = right_hand[:, :, 0:1, :]
        left_shape = left_hand - left_wrist
        right_shape = right_hand - right_wrist
        shape_feats = torch.cat([left_shape, right_shape], dim=2)  # (B, T, 2*_N_HAND, 2)

        # --- Facial Semantic Frame (Eq. 2) ---
        face_centroid = face.mean(dim=2)  # c_f(t): (B, T, 2)
        face_scale = (face - face_centroid.unsqueeze(2)).norm(dim=-1).mean(dim=2, keepdim=True)  # s_f(t)
        face_scale = face_scale.clamp(min=self.eps)

        left_wrist_global = left_hand[:, :, 0, :]
        right_wrist_global = right_hand[:, :, 0, :]
        left_traj = (left_wrist_global - face_centroid) / (face_scale + self.eps)
        right_traj = (right_wrist_global - face_centroid) / (face_scale + self.eps)
        traj_feats = torch.cat([left_traj, right_traj], dim=-1)  # (B, T, 4)

        return shape_feats, traj_feats

    # -------- Stream A: TSSN backbone (STGC blocks + multi-scale fusion + BiLSTM) --------
    def _shape_backbone(self, shape_feats):
        x = shape_feats
        multi_scale = []
        for block in self.shape_blocks:
            x = block(x)  # (B, T, N, C_i)
            multi_scale.append(x.mean(dim=2))  # node-pool -> (B, T, C_i)

        fused = 0.0
        for feat, proj in zip(multi_scale, self.multiscale_proj):
            fused = fused + proj(feat)  # (B, T, d_model), summed across scales

        seq_s, _ = self.shape_lstm(fused)  # (B, T, d_model)
        return seq_s

    def encode_stream_a(self, shape_feats, video_mask):
        """Returns the pooled, global morphological feature F_s: (B, d_model)."""
        video_mask = video_mask.bool()
        seq_s = self._shape_backbone(shape_feats)
        F_s, _ = self.shape_attn(
            self.shape_query.expand(shape_feats.shape[0], -1, -1), seq_s, seq_s,
            key_padding_mask=~video_mask,
        )
        return self.shape_norm(F_s.squeeze(1))

    # -------- Stream B: FTDE (causal ST-Conv + BiLSTM, Finsler energy-gated) --------
    def encode_stream_b(self, traj_feats, video_mask):
        """Returns the trajectory feature sequence F_t: (B, T, d_model)."""
        video_mask = video_mask.bool()
        conv_feats = self.traj_conv(traj_feats)  # (B, T, traj_hidden)
        lstm_out, _ = self.traj_lstm(conv_feats)  # (B, T, d_model)
        energy = self.traj_energy_gate(traj_feats, video_mask=video_mask)  # (B, T, 1), Eq. 3
        F_t = lstm_out * energy
        return self.traj_norm(F_t)

    def forward(self, features, labels=None, video_mask=None):
        if video_mask is None:
            raise ValueError("video_mask is required")
        video_mask = video_mask.bool()
        B, T, _ = features.shape
        coords = features.reshape(B, T, self.NUM_NODES, 2)

        shape_feats, traj_feats = self._dual_reference_normalize(coords)
        key_padding_mask = ~video_mask

        # ---- Stream A: TSSN ----
        seq_s = self._shape_backbone(shape_feats)
        F_s, _ = self.shape_attn(
            self.shape_query.expand(B, -1, -1), seq_s, seq_s,
            key_padding_mask=key_padding_mask,
        )
        F_s = self.shape_norm(F_s.squeeze(1))  # (B, d_model)

        # ---- Stream B: FTDE ----
        F_t = self.encode_stream_b(traj_feats, video_mask)  # (B, T, d_model)

        # ---- Cross-attention enhancement (Sec. 2.3) ----
        F_s_query = F_s.unsqueeze(1)  # (B, 1, d_model)
        F_s_attn, _ = self.cross_attn_s2t(
            F_s_query, F_t, F_t, key_padding_mask=key_padding_mask
        )
        F_s_attn = F_s_attn.squeeze(1)  # (B, d_model)
        F_t_attn, _ = self.cross_attn_t2s(F_t, F_s_query, F_s_query)  # (B, T, d_model)

        # ---- Geo-OT fusion: align F_t_attn onto F_s_attn via Sinkhorn OT ----
        proj_s = self.ot_cost_proj_s(F_s_attn)  # (B, d_model)
        proj_t = self.ot_cost_proj_t(F_t_attn)  # (B, T, d_model)
        cost = 1.0 - F.cosine_similarity(proj_s.unsqueeze(1), proj_t, dim=-1)  # (B, T)
        cost = cost.unsqueeze(1)  # (B, 1, T)

        row_mask = torch.ones(B, 1, device=features.device)
        col_mask = video_mask.float()
        gamma = sinkhorn(
            cost, num_iters=self.ot_iters, epsilon=self.ot_epsilon,
            row_mask=row_mask, col_mask=col_mask,
        )  # (B, 1, T)
        F_aligned_t = torch.einsum("bmt,btd->bmd", gamma, F_t_attn).squeeze(1)  # (B, d_model)

        fused = self.fusion(torch.cat([F_s_attn, F_aligned_t], dim=-1))
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

            if self.geo_loss_weight > 0:
                g_s = F.normalize(self.geo_head_s(F_s_attn), dim=-1)
                g_t = F.normalize(self.geo_head_t(F_aligned_t), dim=-1)
                geo_loss = 1.0 - (g_s * g_t).sum(dim=-1).mean()  # Eq. 5
                loss = loss + self.geo_loss_weight * geo_loss

        return ClassificationOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def predict(self, features, video_mask=None, top_k=1):
        logits = self.forward(features, video_mask=video_mask).logits
        if top_k == 1:
            return logits.argmax(dim=-1)
        return logits.topk(k=top_k, dim=-1).indices