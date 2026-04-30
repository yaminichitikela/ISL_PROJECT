# Paper 1 — ST-GCN: Spatial Temporal Graph Convolutional Networks
**Yan et al., AAAI 2018**

---

## What is this paper about?
This is the **foundational paper** for the entire ISL project. It introduces ST-GCN — the idea of treating a human skeleton as a graph (joints = nodes, bones = edges) and running graph convolution over it to recognize actions.

Before this paper, skeleton-based recognition used hand-crafted features or just flattened joint coordinates into a vector — ignoring that joints are connected to each other.

---

## The Key Idea (in plain terms)
- A skeleton is naturally a **graph** — joints are connected by bones
- Graph convolution learns features from each joint AND its neighbors simultaneously
- Time is modeled by stacking frames and running a 1D convolution along the time axis
- The result: a single end-to-end network that takes raw joint coordinates and predicts the action

---

## How does it work?
1. **Input:** A sequence of body joint coordinates (x, y) or (x, y, z) over time → shape: [Channels × Time × Joints]
2. **Graph:** Fixed edges = anatomical bone connections
3. **Spatial GCN:** For each joint, aggregate features from neighbors, weighted by learned importance masks
4. **Temporal Conv:** After spatial GCN, apply 1D convolution across time frames
5. **Repeat** through 9 blocks, then global average pooling → classification

---

## Key Results
- NTU-RGBD (large action recognition dataset): **81.5%** accuracy
- Big leap over previous skeleton methods

---

## What's Important for YOUR Project
- **This is your Milestone 1 baseline.** You train this exact model on INCLUDE first.
- The INCLUDE skeleton baseline before ST-GCN was only 63.1% (XGBoost). ST-GCN should push this much higher.
- HA-GCN (paper 8) shows that ST-GCN alone achieves **~99% on INCLUDE** — so the dataset may be easy for GCN methods.
- All subsequent papers in this reading list build on top of ST-GCN.

---

## What it Can't Do (why we need later papers)
- Fixed graph — can't learn non-physical joint relationships
- Treats all joints equally — hands (most important for signs) get no special attention
- Fixed temporal scale — can't capture both fast fingerspelling and slow compound signs simultaneously

---

## One-line summary
> ST-GCN turns a skeleton into a graph and learns to recognize actions from joint coordinate sequences — this is the architecture your project builds on.
