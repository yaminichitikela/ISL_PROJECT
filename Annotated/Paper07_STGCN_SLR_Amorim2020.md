# Paper 7 — ST-GCN Applied to Sign Language Recognition (ASLLVD-Skeleton)
**Amorim, Macedo, Zanchettin — 2020**

---

## What is this paper about?
This paper shows **how to adapt ST-GCN specifically for sign language** (as opposed to general action recognition). It applies ST-GCN to the American Sign Language Lexicon Video Dataset (ASLLVD) and creates a public skeleton version of that dataset called **ASLLVD-Skeleton**.

Think of this as a "how-to guide" for using ST-GCN on sign language — very practically useful for your project setup.

---

## What They Did (Step by Step)
Their preprocessing pipeline for sign language:

1. **Acquire samples** — take ASLLVD videos
2. **Segment samples** — clip videos to only the signing portion
3. **Estimate skeletons** — run OpenPose to get 130 body keypoints
4. **Filter keypoints** — keep only 27 relevant upper-body joints (10 per hand + 7 body)
5. **Split dataset** — train/test division
6. **Normalize and serialize** — normalize coordinates, save as Python files for fast loading

### Keypoint Selection (27 joints):
- 5 joints per hand × 2 hands = 10 hand joints
- Plus: nose, shoulders, elbows, wrists (7 body joints)
- Total: 27 joints

This is similar to your 53-joint approach but uses fewer hand joints (5 vs. 21 per hand).

---

## Key Results
| Dataset | Signs | Top-1 (Top-1) | Top-5 |
|---------|-------|--------------|-------|
| ASLLVD (20 signs subset) | 20 | **61.04%** | **86.36%** |
| Full ASLLVD | 2,745 | 37.15% (approx.) | — |

---

## Why Performance is Moderate
- ASLLVD is extremely challenging: 2,745 signs, very few samples per sign
- 27 hand keypoints are insufficient — misses detailed finger joint positions
- OpenPose (their extractor) is less accurate than MediaPipe for hand details

---

## What's Important for YOUR Project

### 1. Pipeline Template
Their 6-step preprocessing pipeline is almost exactly what your project needs — just swap OpenPose for MediaPipe Holistic and use 53 joints instead of 27.

### 2. Keypoint Selection Justification
They explicitly validate that keeping ONLY upper-body joints works for sign language — lower body is irrelevant. This validates your 53-joint choice.

### 3. ST-GCN Adaptation Tricks
- They modified the ST-GCN graph to support custom topologies (not just the fixed NTU-RGBD graph)
- Key code change: added "custom_layout" option and "graph_args" parameter — lets you define any skeleton topology
- This is what you'll need when defining your 53-joint ISL graph

### 4. Data Normalization
They normalize all joint coordinates to [-1, 1] range relative to the video center. This is important for your Ablation A5 (normalization strategies).

---

## Comparison with Your Project
| Aspect | This Paper | Your Project |
|--------|-----------|-------------|
| Language | American SL | Indian SL |
| Dataset | ASLLVD | INCLUDE |
| Joints | 27 | 53 |
| Pose estimator | OpenPose | MediaPipe Holistic |
| GCN model | ST-GCN only | ST-GCN → 2s-AGCN → dual-graph |
| Novel eval | No | Yes (signer-independent) |

---

## One-line summary
> This paper shows exactly how to adapt ST-GCN for sign language — custom skeleton topology, keypoint filtering, normalization — giving you a practical template for your INCLUDE pipeline.
