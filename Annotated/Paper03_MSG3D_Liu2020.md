# Paper 3 — MS-G3D: Disentangled Multi-Scale Graph Convolution
**Liu et al., CVPR 2020**

---

## What is this paper about?
This paper fixes two subtle but important problems that exist in ST-GCN and 2s-AGCN:
1. When you compute "2-hop neighbors" by squaring the adjacency matrix (A²), you accidentally count some paths more than others — biasing the model toward joints closer to the center of mass.
2. Spatial (which joints) and temporal (when) convolutions happen separately — but in sign language, WHERE your hand is at time t is inseparable from WHERE it was at time t-1.

---

## The Key Idea (in plain terms)

**Problem 1 fix — Disentangled Multi-Scale (MS-GCN):**
- Instead of using A¹ + A² + A³ (which double-counts paths), define separate matrices for exactly 1-hop, exactly 2-hop, exactly 3-hop neighbors
- Each "scale" gets its own independent convolution
- Think of it like: "what do my direct neighbors tell me?" vs. "what do joints 2 bones away tell me?" — answered separately, not mixed together

**Problem 2 fix — G3D (Unified Spatial-Temporal Operator):**
- Instead of: spatial GCN → temporal conv → repeat
- Build a graph where nodes are (joint, time) pairs
- A joint at time t is connected to its spatial neighbors at time t AND to itself at t-1 and t+1
- One convolution captures both spatial structure and temporal dynamics simultaneously

---

## Key Results
| Dataset | Top-1 |
|---------|-------|
| NTU-RGBD 60 X-Sub | **91.5%** |
| NTU-RGBD 60 X-View | **96.2%** |
| NTU-RGBD 120 X-Sub | **86.9%** |

State of the art at time of publication. +3% over 2s-AGCN.

---

## What's Important for YOUR Project
- The G3D unified operator is very relevant to ISL where a sign's meaning comes from a **combination of hand shape (spatial) AND movement trajectory (temporal)** — you can't separate these
- Multi-scale temporal aggregation helps capture ISL signs at different speeds:
  - Fast fingerspelling (needs small temporal window)
  - Slow compound signs (needs large temporal window)
  - MS-G3D handles both simultaneously
- Your **Ablation A1** (testing temporal windows T=32, 48, 64, 96) is directly motivated by this paper — understanding which temporal scale matters most for ISL

---

## What it Can't Do
- More computationally expensive than ST-GCN or 2s-AGCN
- INCLUDE dataset is small (4,287 videos) — complex models may overfit
- Still no special treatment for hand joints

---

## One-line summary
> MS-G3D fixes biased multi-hop aggregation and unifies spatial + temporal graph convolution — motivates why you test multiple temporal window sizes in your ablations.
