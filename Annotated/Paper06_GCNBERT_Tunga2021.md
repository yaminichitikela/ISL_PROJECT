# Paper 6 — GCN-BERT: Sign Language Recognition Using Graph Convolutional Networks with BERT
**Tunga et al., WACV 2021**

---

## What is this paper about?
This paper applies GCN + BERT to American Sign Language recognition (WLASL dataset). It's interesting for your project because it shows one way to combine:
- **GCN** for spatial structure (per-frame: what are the hand shapes?)
- **BERT** for temporal modeling (across frames: what is the sequence of shapes?)

This is a hybrid architecture — spatial by graph, temporal by transformer.

---

## The Key Idea
Traditional ST-GCN uses temporal convolution (1D conv) for time modeling. This paper replaces that with **BERT**, which uses self-attention to model long-range temporal dependencies.

### Architecture:
```
Per frame → GCN (fully-connected spatial graph) → Joint feature vector
Sequence of feature vectors → BERT encoder → Temporal context
BERT output + GCN output (skip connection) → Classification
```

**Why BERT?**
- BERT's self-attention lets every frame attend to every other frame — not just nearby frames
- For compound ISL signs that unfold over 30+ frames, this long-range modeling is important
- 1D temporal convolution in ST-GCN only looks at neighboring frames (fixed kernel size)

**Why a fully-connected spatial graph?**
- Instead of using anatomical connections, every joint is connected to every other joint per frame
- The adjacency is learned — lets the model discover which joint pairs matter for each sign
- This is similar in spirit to the C_k component in 2s-AGCN

### Skip Connection
The GCN's spatial output is added to the BERT's temporal output before classification. This preserves per-frame hand shape information that BERT might smooth over.

---

## Key Results
| Dataset | Method | Top-1 Acc |
|---------|--------|-----------|
| WLASL-100 | Pose-TGCN (baseline) | 55.29% |
| WLASL-100 | GCN-BERT (this paper) | **60.15%** |

+5% improvement over the previous graph-based baseline on WLASL-100.

---

## What's Important for YOUR Project
- **Motivation for hybrid architectures:** If you extend beyond Milestone 3, GCN-BERT shows one path to combine graph spatial modeling with transformer temporal modeling
- **Fully-connected learned adjacency:** The idea that spatial graph topology should be fully learned (not just anatomically fixed) is related to the C_k component your 2s-AGCN milestone uses
- **Skip connections in GCN:** Good regularization technique — preserves frame-level detail alongside temporal context. Worth including in your architecture.
- **WLASL vs INCLUDE:** This paper uses WLASL (American SL, 100+ classes). Your project uses INCLUDE (ISL, 263 classes). Results are not directly comparable, but the methods transfer.

---

## Limitation for Your Project
- BERT adds significant compute overhead — for a 263-class problem with 4,287 training videos, a simpler temporal model (TCN or LSTM) may generalize better
- WLASL-100 achieves only 60.15% — the architecture has room for improvement
- This paper doesn't use hand-specific graph topology (key gap that HA-GCN fills)

---

## One-line summary
> GCN-BERT combines per-frame graph convolution (spatial) with BERT self-attention (temporal) for SLR — relevant if you explore transformer-based temporal modeling as a future extension.
