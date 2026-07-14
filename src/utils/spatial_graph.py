"""
Skeleton graph structure for the fused 61-point skeleton produced by
fusion_component.py, and a partitioned spatial graph convolution layer
(ST-GCN style: root / centripetal / centrifugal partitions) to replace or
augment the plain nn.Linear currently used in input_projection.

=== Node layout (must match fusion_component.py's output order exactly) ===

_fuse_single_frame() returns:
    np.concatenate([pose_coords, left_hand, right_hand], axis=0)

pose_coords has already been trimmed by _REMOVE_POSE_IDX, so the ORIGINAL
MediaPipe Pose indices that remain, in their original relative order, are:
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,23,24
(33 landmarks minus the 14 removed = 19 remain)

NOTE — this graph now assumes fusion_component.py's _REMOVE_POSE_IDX has
been fixed to ALSO drop original MediaPipe index 32 (right_foot_index), so
the pose sub-skeleton no longer carries an orphaned, asymmetric foot node
(the old node 19). fusion_component.py must be updated to match (drop 32
from its kept-index list) or this graph's indices will silently misalign
with the real feature tensor.

New index -> original MediaPipe Pose landmark:
  0 nose             5 right_eye         10 mouth_right      15 left_wrist*
  1 left_eye_inner   6 right_eye_outer   11 left_shoulder     16 right_wrist*
  2 left_eye         7 left_ear          12 right_shoulder    17 left_hip
  3 left_eye_outer   8 right_ear         13 left_elbow        18 right_hip
  4 right_eye_inner  9 mouth_left        14 right_elbow

  * left_wrist(15)/right_wrist(16) hold the HAND landmarker's wrist
    coordinate, not the pose landmarker's — see _merge_and_trim_pose().
    Same physical joint, better source; treat as "wrist" for graph purposes.

Then, in global index terms (0-based over all 61 nodes):
  0..18   = pose (above)
  19..39  = left hand  (MediaPipe Hand topology, 0=wrist..20=pinky tip)
  40..60  = right hand (same topology, offset by 21)

MediaPipe Hand topology (applies to both hand blocks):
  0 wrist
  1-4   thumb   (cmc, mcp, ip, tip)
  5-8   index   (mcp, pip, dip, tip)
  9-12  middle  (mcp, pip, dip, tip)
  13-16 ring    (mcp, pip, dip, tip)
  17-20 pinky   (mcp, pip, dip, tip)
"""

import torch
import torch.nn as nn

NUM_NODES = 61

# --- Pose sub-skeleton (indices 0-18) ---------------------------------------
# Loose face chain (minor for SLR, kept so these nodes aren't fully isolated)
_POSE_FACE_EDGES = [
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7),
    (4, 5), (5, 6), (6, 8), (9, 0), (10, 0),
]
# Head -> shoulders (approximate neck link; MediaPipe has no neck landmark)
_POSE_NECK_EDGES = [(0, 11), (0, 12)]
# Torso + arms
_POSE_BODY_EDGES = [
    (11, 12),          # shoulder to shoulder
    (11, 13), (13, 15),  # left shoulder -> elbow -> wrist
    (12, 14), (14, 16),  # right shoulder -> elbow -> wrist
    (11, 17), (12, 18),  # shoulders -> matching hips
    (17, 18),           # hip to hip
]

# --- Hand sub-skeleton (applies to both hands via HAND_EDGES + offset) -----
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),             # across knuckles, adds stability
]

LEFT_HAND_OFFSET = 19
RIGHT_HAND_OFFSET = 40

# --- Bridge edges: connect each pose wrist node to its hand's wrist node ---
# Without these two edges the hand sub-skeletons would be fully disconnected
# from the pose sub-skeleton — no path exists between e.g. the elbow and the
# fingers, so a GCN could never let finger shape influence arm-level context
# or vice versa.
_BRIDGE_EDGES = [
    (15, LEFT_HAND_OFFSET + 0),    # pose left_wrist  <-> left hand wrist
    (16, RIGHT_HAND_OFFSET + 0),   # pose right_wrist <-> right hand wrist
]

EDGES = (
    _POSE_FACE_EDGES
    + _POSE_NECK_EDGES
    + _POSE_BODY_EDGES
    + [(a + LEFT_HAND_OFFSET, b + LEFT_HAND_OFFSET) for a, b in HAND_EDGES]
    + [(a + RIGHT_HAND_OFFSET, b + RIGHT_HAND_OFFSET) for a, b in HAND_EDGES]
    + _BRIDGE_EDGES
)

# Root node for computing centripetal/centrifugal direction: hip midpoint is
# the closest thing to a "center of gravity" available in this node set.
# BFS distance from this node decides, for every edge, which endpoint is
# "closer to the torso" (centripetal target) and which is "farther"
# (centrifugal target) — same convention as the original ST-GCN paper.
CENTER_NODE = 17  # left_hip; symmetric enough for this purpose


def _bfs_distances(num_nodes, edges, source):
    adj = [[] for _ in range(num_nodes)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    dist = [-1] * num_nodes
    dist[source] = 0
    queue = [source]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for nxt in adj[node]:
            if dist[nxt] == -1:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)

    # Any node unreachable from source (shouldn't happen with the edge list
    # above, but guard anyway) is treated as maximally far.
    max_dist = max(d for d in dist if d >= 0)
    return [d if d >= 0 else max_dist + 1 for d in dist]


def build_partitioned_adjacency(num_nodes=NUM_NODES, edges=EDGES, center_node=CENTER_NODE):
    """
    Builds the 3 partition adjacency matrices used by ST-GCN-style spatial
    graph convolution:
        A[0] = root       (self-loops only, identity)
        A[1] = centripetal (edges pointing from a farther node to a nearer one)
        A[2] = centrifugal (edges pointing from a nearer node to a farther one)

    Each is symmetrically-normalized (D^-1/2 A D^-1/2) per Kipf & Welling /
    ST-GCN convention, so a node with many neighbors doesn't dominate a node
    with few.

    Returns: torch.FloatTensor of shape (3, num_nodes, num_nodes)
    """
    dist = _bfs_distances(num_nodes, edges, center_node)

    A_root = torch.eye(num_nodes)
    A_centri_in = torch.zeros(num_nodes, num_nodes)   # nearer -> farther (centrifugal)
    A_centri_out = torch.zeros(num_nodes, num_nodes)  # farther -> nearer (centripetal)

    for a, b in edges:
        if dist[a] == dist[b]:
            # Same hop-distance from center (e.g. left_hip-right_hip): no
            # natural centripetal/centrifugal direction, so ST-GCN convention
            # puts these in the centripetal partition by default.
            A_centri_out[a, b] = 1
            A_centri_out[b, a] = 1
        elif dist[a] < dist[b]:
            A_centri_out[b, a] = 1   # b (farther) receives from a (nearer)
            A_centri_in[a, b] = 1    # a (nearer) receives from b (farther)
        else:
            A_centri_out[a, b] = 1
            A_centri_in[b, a] = 1

    def _normalize(A):
        deg = A.sum(dim=1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5)
        D = torch.diag(d_inv_sqrt)
        return D @ A @ D

    return torch.stack([
        A_root,                       # partition 0: root/self
        _normalize(A_centri_out),     # partition 1: centripetal (toward center)
        _normalize(A_centri_in),      # partition 2: centrifugal (away from center)
    ])


class SpatialGraphConv(nn.Module):
    """
    ST-GCN-style spatial graph convolution over the 61-node skeleton.

    Input : (B, T, N, C_in)  — N=61 nodes, C_in usually 3 (x, y, z)
    Output: (B, T, N, C_out)

    Each partition gets its own learned Linear (W_root, W_centripetal,
    W_centrifugal); a node's new feature is the sum of what its neighbors
    in each partition contribute, weighted by the fixed (non-learned)
    normalized adjacency and the learned per-partition weight.

    Meant to run per-frame (apply the same spatial conv at every timestep,
    weights shared across time) — plug its output into input_projection's
    flatten-then-Linear step, or use it to REPLACE input_projection's first
    Linear, before the sequence goes into the existing temporal
    PositionalEncoding + TransformerEncoder.
    """

    def __init__(self, in_channels=3, out_channels=64,
                 num_nodes=NUM_NODES, edges=EDGES, center_node=CENTER_NODE):
        super().__init__()
        A = build_partitioned_adjacency(num_nodes, edges, center_node)
        self.register_buffer("A", A)  # (3, N, N), not trained — fixed graph structure
        self.W = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(A.shape[0])
        ])

    def forward(self, x):
        # x: (B, T, N, C_in)
        out = 0
        for k, linear in enumerate(self.W):
            projected = linear(x)                     # (B, T, N, C_out)
            # A[k]: (N, N) — mixes node features according to partition k's graph
            out = out + torch.einsum("nm,btmc->btnc", self.A[k], projected)
        return out                                     # (B, T, N, C_out)