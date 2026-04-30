# Paper 2 — 2s-AGCN: Two-Stream Adaptive Graph Convolutional Networks
**Shi et al., CVPR 2019**

---

## What is this paper about?
This paper asks: **what if the skeleton graph doesn't have to be fixed?** ST-GCN hardcodes which joints are connected. But for signs, joints that aren't physically connected (e.g., left index finger and right elbow) might co-move in meaningful ways. 2s-AGCN learns the graph structure from data.

It also introduces a second input stream: **bone vectors** (direction from one joint to the next) instead of just joint positions.

---

## The Key Idea (in plain terms)
Two upgrades over ST-GCN:

**1. Adaptive Adjacency Matrix**
Instead of a fixed A (anatomy), use:
```
A_total = A (fixed anatomy) + B (global learned adjustment) + C (per-sample learned)
```
- B is learned during training — shared across all samples
- C is computed on-the-fly per video using attention (each joint looks at every other joint)
- This means the model can learn "for this specific sign, these two non-adjacent joints always move together"

**2. Two-Stream Fusion**
- Stream 1: Joint coordinates (where each joint IS)
- Stream 2: Bone vectors (direction from parent joint to child joint — encodes movement direction)
- Final prediction = sum of scores from both streams

---

## Key Results
| Dataset | Top-1 |
|---------|-------|
| NTU-RGBD 60 X-Sub | **88.5%** |
| NTU-RGBD 60 X-View | **95.1%** |

+7% over ST-GCN on the same dataset.

---

## What's Important for YOUR Project
- **This is your Milestone 2 model.** You upgrade from ST-GCN to 2s-AGCN.
- For ISL, adaptive adjacency is especially useful because:
  - ISL compound signs involve non-physical joint relationships
  - The model can learn "these two joints co-move for the sign BROTHER" without needing to hardcode it
- Bone vectors (Stream 2) capture hand movement direction — critical for ISL where HOW you move matters as much as WHERE your joints are

---

## What it Can't Do
- Still uses the same graph for the whole body — hands don't get special treatment
- Bone stream adds compute cost
- Two streams are just summed — not learned fusion

---

## One-line summary
> 2s-AGCN upgrades ST-GCN by making the graph learnable (adaptive adjacency) and adding bone direction features as a second stream — your Milestone 2 model.
