# Annotation: 2s-AGCN — Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition

**Authors:** Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu
**Venue:** CVPR 2019
**Paper File:** lit_rev/paper2.pdf
**Code:** Available (cited in roadmap)

---

## 1. Core Contribution
Replaces ST-GCN's **fixed** adjacency matrix with an **adaptive** one learned end-to-end. Introduces a **two-stream** approach combining joint positions and bone vectors. Key insight: the optimal graph topology differs across samples and layers — it should be data-driven.

## 2. Method Summary

### Adaptive Graph Decomposition
The adjacency matrix is decomposed into three components:
```
A_total = A_k + B_k + C_k
```
- **A_k** — fixed physical structure (same as ST-GCN; normalized adjacency)
- **B_k** — global learned residual; shared across all samples, updated by backpropagation
- **C_k** — data-dependent residual; computed per sample via: `C_k = softmax(θ(f_in)^T φ(f_in))` (dot-product attention between joint embeddings)

### Two-Stream Architecture
- **Stream 1 (Joint stream):** Joint coordinates x_i as input
- **Stream 2 (Bone stream):** Bone vectors `b_{ij} = x_j - x_i` where j is the source joint (distal end) and i is the target joint. Captures directional force/velocity information.
- **Fusion:** Softmax scores from both streams are summed at inference time.

### GCN Forward Pass
`f_out = Σ_k W_k f_in (A_k + B_k + C_k)`

## 3. Key Results
| Dataset | Protocol | Top-1 | Top-5 |
|---------|----------|-------|-------|
| NTU-RGBD 60 | X-Sub | 88.5% | 95.1% |
| NTU-RGBD 60 | X-View | 95.1% | — |
| NTU-RGBD 120 | X-Sub | 82.9% | — |
| NTU-RGBD 120 | X-Set | 84.9% | — |
| Kinetics | — | 36.1% | 58.7% |

Outperforms ST-GCN by ~7% on NTU-RGBD 60 X-Sub.

## 4. Strengths for ISL Project
- **Adaptive adjacency is critical for ISL:** ISL compound signs involve non-physical joint relationships (e.g., simultaneous hand configurations encode compound meanings). C_k learns these per-sample.
- **Bone stream captures sign dynamics:** ISL sign phonology is fundamentally about hand movement direction and velocity — bone vectors directly encode this.
- **Roadmap Milestone 2:** 2s-AGCN is the second stage of the model progression.
- Proven improvement: +7% over ST-GCN on action recognition benchmark.

## 5. Limitations / Gaps for ISL
- **Still no hand-topology-specific graph:** Both streams operate on the same full-body skeleton graph. ISL signs are 90%+ hand-dominant.
- **Bone stream direction choice:** The convention of which joint is "source" vs. "target" affects the directional encoding — needs careful choice for the 53-joint ISL graph.
- **Two-stream fusion is fixed (score summation):** Learned adaptive fusion (as in HA-GCN) may work better.
- **C_k computation is O(N²):** For 53 joints this is 2809 pairwise comparisons per frame — manageable but worth monitoring memory.

## 6. Relevance to ISL ST-GCN Roadmap
- **Milestone 2 (Week 3-4):** Direct upgrade from baseline ST-GCN.
- **Ablation A3:** Tests adaptive (A_k + B_k + C_k) vs. fixed (A_k only) adjacency — quantifies the benefit of learning joint correlations specific to ISL sign structure.
- **Ablation A2:** Whether using bone vectors (bone stream) improves over joint-only stream for ISL.
- Directly adopted in HA-GCN (paper 8), where the adjacency formula extends to include SH and PH hand graphs.

## 7. Citation
Shi, L., Zhang, Y., Cheng, J., & Lu, H. (2019). Two-stream adaptive graph convolutional networks for skeleton-based action recognition. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 12026–12035).
