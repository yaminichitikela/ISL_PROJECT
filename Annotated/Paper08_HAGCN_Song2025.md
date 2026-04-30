# Paper 8 — HA-GCN: Hand-Aware Graph Convolution Network for Sign Language Recognition
**Song et al., Journal of Information and Intelligence, 2025**

---

## What is this paper about?
**This is the most important competitor paper for your project.** HA-GCN is the current state-of-the-art on your INCLUDE dataset (99.63% accuracy). It was published in 2025 and tests directly on INCLUDE — meaning you MUST compare your results against it.

The core insight: **existing GCN methods treat all body joints equally, but hands are by far the most important body part for sign language.** HA-GCN explicitly builds hand topology into the GCN.

---

## The Key Idea (in plain terms)
HA-GCN adds **two hand-specific sub-graphs** on top of the regular body graph:

**Graph 1 — Structured Hand (SH):** Physical hand connections (finger joints connected as they actually are anatomically). Fixed, like anatomical edges in ST-GCN but only for hands.

**Graph 2 — Parameterized Hand (PH):** Fully learnable hand connections. The model can learn that "index fingertip and thumb tip often co-move in this sign" even though they're not physically adjacent.

Combined adjacency formula:
```
A_total = A (body) + B (learned global) + SH × α + PH × β
```
where α and β are learned coefficients that balance how much to weight the hand-specific graphs.

---

## Additional Innovation: Adaptive DropGraph
- During training, randomly drop some spatial nodes and temporal frames
- Prevents overfitting (INCLUDE is small — only 4,287 videos)
- Separate spatial DropGraph and temporal DropGraph, controlled by gating mechanism

---

## Multi-Stream Architecture
HA-GCN runs 4 parallel streams, each processed by 10 HA-GCN blocks:
1. **Joint stream** — raw joint coordinates
2. **Bone stream** — bone direction vectors (like 2s-AGCN)
3. **Joint motion stream** — frame-to-frame joint movement
4. **Bone motion stream** — frame-to-frame bone movement

Final prediction = adaptive fusion of all 4 streams.

---

## Results on INCLUDE (your dataset!)
| Method | Top-1 Accuracy |
|--------|---------------|
| ST-GCN (baseline, no SH/PH) | 99.02% |
| HA-GCN single stream (joints) | **99.39%** |
| HA-GCN multi-stream | **99.63%** |

Also tested on AUTSL (Turkish SL): 95.88% single, 96.82% multi-stream.

---

## Critical Takeaway for YOUR PROJECT
- ST-GCN alone already achieves 99.02% on INCLUDE — this dataset seems "easy" for GCN methods
- The original INCLUDE paper's skeleton baseline (XGBoost: 63.1%) was very primitive — modern GCNs easily surpass it
- Your project's target should be to:
  1. Reproduce the ST-GCN → 2s-AGCN progression on INCLUDE
  2. Propose your own dual-hand-graph design (Milestone 3)
  3. **Most importantly: run signer-independent evaluation** — HA-GCN only does signer-dependent evaluation. Your SI results are genuinely new.
- The high INCLUDE numbers (99%+) suggest signer-dependent evaluation may be too easy — **signer-independent is where the real challenge lies**

---

## What You Can Take from This Paper
- **Architecture inspiration:** Your Milestone 3 model should include hand-specific sub-graphs (SH and PH design), adapted for your 53-joint ISL topology
- **Adaptive DropGraph:** Consider implementing this for regularization on the small INCLUDE dataset
- **Multi-stream fusion:** Your ablations should test 1-stream vs. 2-stream vs. 4-stream
- **This is your main baseline to beat** — particularly on signer-independent evaluation where HA-GCN has no published result

---

## One-line summary
> HA-GCN is the current state-of-the-art on INCLUDE (99.63%) — it adds hand-specific sub-graphs and multi-stream fusion to ST-GCN — your project must compare against it and go beyond it via signer-independent evaluation.
