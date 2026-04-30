# Annotation: ST-GCN — Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition

**Authors:** Sijie Yan, Yuanjun Xiong, Dahua Lin
**Venue:** AAAI 2018
**Paper File:** lit_rev/paper1.pdf (referenced from roadmap; foundational work)
**Code:** https://github.com/yysijie/st-gcn

---

## 1. Core Contribution
ST-GCN is the first end-to-end deep learning model that treats the human skeleton as a **spatial graph** and learns joint representations via graph convolution combined with temporal convolution. Eliminates hand-crafted feature engineering from prior skeleton-based action recognition.

## 2. Method Summary
- **Graph construction:** Joints = nodes; bones = edges. Graph G = (V, E). Intra-body natural connections form spatial edges; inter-frame connections of the same joint form temporal edges.
- **Spatial partitioning strategy:** Three subsets — root (centroid), centripetal (closer to centroid), centrifugal (farther from centroid). Each partition gets a separate learnable weight matrix.
- **GCN layer:** `f_out = Σ_k W_k f_in A_k ⊙ M_k` where A_k is the fixed partition adjacency matrix and M_k is a learnable edge importance mask.
- **Temporal convolution:** 1D convolution (kernel Γ) along the time axis after each spatial GCN.
- **Architecture:** 9 ST-GCN blocks (64→64→64→128→128→128→256→256→256 channels), global average pooling, softmax.
- **Input:** 2D or 3D joint coordinate sequences (C × T × N tensors: channels × time frames × joints).

## 3. Key Results
| Dataset | Protocol | Top-1 Acc |
|---------|----------|-----------|
| Kinetics (400 classes) | — | 30.7% |
| NTU-RGBD 60 | X-Sub | 81.5% |
| NTU-RGBD 60 | X-View | 88.3% |

## 4. Strengths for ISL Project
- **Direct architectural foundation** for the roadmap's Milestone 1 baseline.
- The spatial partitioning strategy is directly applicable to sign language — centripetal nodes (fingertips moving toward palm) vs. centrifugal (extending outward) maps naturally to ISL phonology.
- Fixed adjacency matrix preserves anatomical structure — good starting point.
- Open-source code; ASLLVD-Skeleton (paper 7) demonstrates straightforward adaptation to SLR.

## 5. Limitations / Gaps for ISL
- **Fixed adjacency matrix:** Cannot learn non-physical joint relationships (e.g., left index → right wrist in compound ISL signs). Addressed by 2s-AGCN (paper 2).
- **No hand topology awareness:** Treats all body joints equally; hands are critical for sign language. Addressed by HA-GCN (paper 8).
- **No sign-language-specific evaluation:** Original work on action recognition datasets, not SLR.
- Temporal convolution is fixed-kernel; cannot model multi-scale temporal dynamics. Addressed by MS-G3D (paper 3).

## 6. Relevance to ISL ST-GCN Roadmap
- **Milestone 1 (Week 1-2):** This paper's architecture is the literal starting point. The roadmap trains ST-GCN on INCLUDE as the baseline.
- **Expected baseline performance on INCLUDE:** ~85-90% top-1 (SD), based on HA-GCN ablation: without SH and PH graphs, baseline ST-GCN achieves **99.02%** on INCLUDE (surprisingly high, likely due to small dataset size with only 263 classes and 4287 videos).
- **Ablation A3 (adjacency type):** Tests fixed vs. adaptive adjacency — directly compares ST-GCN formulation against 2s-AGCN style.

## 7. Citation
Yan, S., Xiong, Y., & Lin, D. (2018). Spatial temporal graph convolutional networks for skeleton-based action recognition. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1).
