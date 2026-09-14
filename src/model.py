"""
Spatio-Temporal Graph Convolutional Network (ST-GCN).

Reference:
    Yan et al., "Spatial Temporal Graph Convolutional Networks for
    Skeleton-Based Action Recognition", AAAI 2018.
    https://arxiv.org/abs/1801.07455

Architecture (Table 1 of the paper):
    data_bn → 9 ST-GCN blocks → global avg-pool → dropout → FC

    Block  in→out    t-stride   Notes
    ─────  ────────  ────────   ─────────────────────────────
      1     3 → 64      1       no residual (channels differ, λ=0)
      2    64 → 64      1
      3    64 → 64      1
      4    64 → 64      1
      5    64 →128      2       temporal downsampling T→T/2
      6   128 →128      1
      7   128 →128      1
      8   128 →256      2       temporal downsampling T→T/4
      9   256 →256      1
    ─────────────────────────────────────────────────────────
    → GlobalAvgPool(T, V) → Dropout(0.5) → FC(256, n_classes)

Input tensor  : (N, C, T, V) = (batch, 3, 64, 53)
Output tensor : (N, n_classes) logits

Each ST-GCN block contains:
    1. Spatial graph convolution (SpatialGraphConv)
       — K independent weight matrices, one per adjacency partition
       — learnable edge-importance mask M (Yan §3.4), init to all-ones
    2. Batch normalisation + ReLU
    3. Temporal Conv2d  kernel=(9,1), padding=(4,0), stride=(t_stride,1)
    4. Batch normalisation + Dropout
    5. Residual connection (identity or 1×1 conv projection)
    6. Final ReLU
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ── Spatial graph convolution ─────────────────────────────────────────────────

class SpatialGraphConv(nn.Module):
    """
    Graph convolution over the joint (spatial) dimension.

    For K partitions the operation is (Yan et al. Eq. 4):

        out[v] = Σ_k  Σ_{u}  A_eff[k, u, v] · W_k · x[u]

    where A_eff[k] = A[k] * M[k]  (element-wise, learnable mask M).

    Implementation: apply K independent 1×1 convolutions first (via a single
    grouped Conv2d), then aggregate over neighbour joints using einsum.
    """

    def __init__(self, in_channels: int, out_channels: int, A: torch.Tensor, adaptive: bool = False):
        """
        Parameters
        ----------
        in_channels  : C_in
        out_channels : C_out
        A            : (K, V, V) normalised adjacency; stored as a buffer
                       (not updated by the optimiser)
        adaptive     : if True, add a learnable additive offset B (init zero)
                       on top of A*M — 2s-AGCN style data-driven edges that
                       capture non-physical joint correlations (Shi et al. 2019)
        """
        super().__init__()
        K = A.shape[0]
        self.K = K
        self.register_buffer("A", A)          # fixed adjacency

        # Learnable edge importance mask (Yan §3.4).
        # Multiplied element-wise with A at forward time; init to 1 so the
        # initial pass is equivalent to standard GCN.
        self.M = nn.Parameter(torch.ones(K, A.shape[1], A.shape[2]))

        # Adaptive adjacency offset (2s-AGCN style): fully learnable, init zero
        # so the model starts identical to the fixed-topology variant.
        self.B = nn.Parameter(torch.zeros(K, A.shape[1], A.shape[2])) if adaptive else None

        # One Conv2d produces K*C_out channels — K independent transformations.
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (N, C_in, T, V)
        out : (N, C_out, T, V)
        """
        N, C, T, V = x.size()

        # Step 1 — transform: (N, C_in, T, V) → (N, K·C_out, T, V)
        z = self.conv(x)
        C_out = z.size(1) // self.K
        z = z.view(N, self.K, C_out, T, V)      # (N, K, C_out, T, V)

        # Step 2 — aggregate: apply masked adjacency along the joint axis.
        # A_eff[k, src, dst] = how much src contributes to dst in partition k.
        # out[n, c, t, dst] = Σ_k Σ_src  z[n,k,c,t,src] · A_eff[k,src,dst]
        A_eff = self.A * self.M                  # (K, V, V)
        if self.B is not None:
            A_eff = A_eff + self.B                # adaptive offset (2s-AGCN style)
        out = torch.einsum("nkctv, kvw -> nctw", z, A_eff)

        return out.contiguous()                  # (N, C_out, T, V)


# ── Single ST-GCN block ───────────────────────────────────────────────────────

class STGCNBlock(nn.Module):
    """
    One building block of ST-GCN:
        GCN → BN → ReLU → temporal-Conv → BN → (+residual) → ReLU

    The temporal convolution uses kernel size 9 with padding 4 so that
    the time dimension is only affected by the stride parameter.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
        adaptive: bool = False,
    ):
        super().__init__()

        self.gcn = SpatialGraphConv(in_channels, out_channels, A, adaptive=adaptive)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(9, 1),
                padding=(4, 0),
                stride=(stride, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            # First block: no residual path (Yan et al. implementation)
            self.residual = lambda _x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            # Projection shortcut to match dimensions
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C_in, T, V) → (N, C_out, T', V)"""
        res = self.residual(x)
        x   = self.gcn(x)
        x   = self.tcn(x) + res
        return self.relu(x)


# ── Full ST-GCN network ───────────────────────────────────────────────────────

class STGCN(nn.Module):
    """
    9-block ST-GCN for skeleton-based action / sign-language recognition.

    Parameters
    ----------
    n_classes   : number of output classes (262 for INCLUDE)
    A           : (K, V, V) normalised adjacency from src.graph.build_adjacency()
    in_channels : coordinate dimensions per joint (3 for x, y, z)
    dropout     : dropout probability before the final FC layer
    adaptive    : if True, every block adds a learnable additive adjacency
                  offset on top of A (2s-AGCN style) — used for the
                  topology ablation's "adaptive" condition (RQ3)
    """

    def __init__(
        self,
        n_classes: int,
        A: np.ndarray,
        in_channels: int = 3,
        dropout: float = 0.5,
        adaptive: bool = False,
    ):
        super().__init__()

        A_t = torch.from_numpy(A.astype(np.float32))   # (K, V, V)
        V   = A.shape[-1]

        # Data batch normalisation: normalise each (joint, coord) feature
        # independently across batch and time (Yan et al. §4.1).
        # Input is reshaped to (N, V*C, T) for BN1d.
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        # 9 ST-GCN blocks — (out_channels, temporal_stride)
        _cfg = [
            (64,  1),   # block 1
            (64,  1),   # block 2
            (64,  1),   # block 3
            (64,  1),   # block 4
            (128, 2),   # block 5 — T ÷ 2
            (128, 1),   # block 6
            (128, 1),   # block 7
            (256, 2),   # block 8 — T ÷ 4
            (256, 1),   # block 9
        ]

        blocks = []
        c_in = in_channels
        for i, (c_out, stride) in enumerate(_cfg):
            blocks.append(
                STGCNBlock(
                    c_in, c_out, A_t,
                    stride=stride,
                    dropout=0.0,
                    residual=(i > 0),   # no residual on block 1
                    adaptive=adaptive,
                )
            )
            c_in = c_out

        self.blocks = nn.ModuleList(blocks)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (N, C, T, V)
        returns: (N, n_classes) logits
        """
        N, C, T, V = x.size()

        # Data BN: reshape to (N, V*C, T), normalise, reshape back.
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)   # (N, V*C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()    # (N, C, T, V)

        for block in self.blocks:
            x = block(x)                  # (N, 256, T/4, V) after last block

        # Global average pool over time and joints → (N, 256)
        x = x.mean(dim=[2, 3])
        return self.fc(self.drop(x))      # (N, n_classes)
