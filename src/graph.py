"""
Spatio-temporal graph construction for ISL 53-joint skeleton.

Implements the spatial-configuration partitioning strategy from:
    Yan et al., "Spatial Temporal Graph Convolutional Networks for
    Skeleton-Based Action Recognition", AAAI 2018.

Three partitions (K=3):
    0 — self / equidistant  : d(neighbour, root) == d(joint, root)
    1 — centripetal         : d(neighbour, root) <  d(joint, root)
    2 — centrifugal         : d(neighbour, root) >  d(joint, root)

Joint index layout (53 joints):
    0–20  : Left hand  (0 = wrist, 1–4 thumb, 5–8 index, 9–12 middle,
                        13–16 ring, 17–20 pinky)
    21–41 : Right hand (same pattern, offset +21)
    42–52 : Upper body (42 nose, 43 l-ear, 44 r-ear, 45 l-shoulder,
                        46 r-shoulder, 47 l-elbow, 48 r-elbow,
                        49 l-wrist(pose), 50 r-wrist(pose),
                        51 l-hip, 52 r-hip)
"""

from __future__ import annotations
import numpy as np
import networkx as nx

# ── Skeleton edges (undirected) ───────────────────────────────────────────────
# Exactly the edges from notebook 03_graph_topology.
_EDGES: list[tuple[int, int]] = [
    # Left hand
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),          # MCP knuckle arch
    # Right hand
    (21, 22), (22, 23), (23, 24), (24, 25),
    (21, 26), (26, 27), (27, 28), (28, 29),
    (21, 30), (30, 31), (31, 32), (32, 33),
    (21, 34), (34, 35), (35, 36), (36, 37),
    (21, 38), (38, 39), (39, 40), (40, 41),
    (26, 30), (30, 34), (34, 38),
    # Upper body
    (42, 43), (42, 44),
    (43, 45), (44, 46),
    (45, 46), (45, 51), (46, 52), (51, 52),
    (45, 47), (47, 49),
    (46, 48), (48, 50),
    # Inter-hand chain: hand wrists connect to pose wrists
    (0, 49), (21, 50),
]

# Gravity centre for partitioning (Yan et al. §3.3).
# We use left hip (51) as the single joint closest to the anatomical
# centre of mass in our 53-joint upper-body skeleton.
_ROOT: int = 51


def _normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """
    Apply symmetric normalisation D^{-1/2} A D^{-1/2} to each partition.

    Zero-degree nodes (isolated joints) are left at zero to avoid division
    by zero — they carry no information in the graph conv anyway.
    """
    out = np.zeros_like(A)
    for k in range(A.shape[0]):
        row_sum = A[k].sum(axis=1)                          # (V,)
        d_inv_sqrt = np.zeros_like(row_sum)
        mask = row_sum > 0
        d_inv_sqrt[mask] = row_sum[mask] ** -0.5            # safe: skip zero-degree nodes
        # D^{-1/2} A D^{-1/2} via broadcasting avoids constructing the full diagonal matrix
        out[k] = d_inv_sqrt[:, None] * A[k] * d_inv_sqrt[None, :]
    return out


def build_adjacency(
    n_joints: int = 53,
    strategy: str = "spatial",
    root: int = _ROOT,
) -> np.ndarray:
    """
    Build the normalised adjacency tensor A of shape (K, V, V).

    Parameters
    ----------
    n_joints : number of skeleton joints V (default 53)
    strategy : 'spatial'  — 3-partition spatial-configuration (Yan 2018 §3.3)
               'uniform'  — 1-partition, all neighbours treated equally
    root     : index of the gravity-centre joint for spatial partitioning

    Returns
    -------
    A : np.ndarray, shape (K, V, V), dtype float32, D^{-1/2}AD^{-1/2} normalised
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_joints))
    G.add_edges_from(_EDGES)

    # Shortest-path distance from each joint to the root.
    d_root = nx.single_source_shortest_path_length(G, root)

    if strategy == "spatial":
        K = 3
        A = np.zeros((K, n_joints, n_joints), dtype=np.float32)

        # Self-loops → partition 0 (a joint is always in its own equidistant set)
        for v in range(n_joints):
            A[0, v, v] = 1.0

        # For each undirected edge (i, j) add two directed entries.
        # Convention: A[k, src, dst] = 1 means src contributes to dst in partition k.
        # The partition that src belongs to is determined by comparing dist(src, root)
        # with dist(dst, root):
        #   dist(src) <  dist(dst) → src is centripetal  for dst → partition 1
        #   dist(src) >  dist(dst) → src is centrifugal  for dst → partition 2
        #   dist(src) == dist(dst) → equidistant          for dst → partition 0
        for i, j in _EDGES:
            di, dj = d_root[i], d_root[j]

            # i contributes to j
            if di < dj:
                A[1, i, j] = 1.0   # i centripetal for j
            elif di > dj:
                A[2, i, j] = 1.0   # i centrifugal for j
            else:
                A[0, i, j] = 1.0   # equidistant

            # j contributes to i
            if dj < di:
                A[1, j, i] = 1.0   # j centripetal for i
            elif dj > di:
                A[2, j, i] = 1.0   # j centrifugal for i
            else:
                A[0, j, i] = 1.0   # equidistant

    elif strategy == "uniform":
        A = np.zeros((1, n_joints, n_joints), dtype=np.float32)
        for v in range(n_joints):
            A[0, v, v] = 1.0
        for i, j in _EDGES:
            A[0, i, j] = 1.0
            A[0, j, i] = 1.0

    else:
        raise ValueError(f"Unknown graph strategy: {strategy!r}. "
                         f"Choose 'spatial' or 'uniform'.")

    return _normalize_adjacency(A)
