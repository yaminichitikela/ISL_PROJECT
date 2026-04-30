# Paper 10 — GCAR: Sign Language Recognition Using Graph Convolution with Attention and Residual Connection
**Miah et al., IEEE Access, 2024**

---

## What is this paper about?
This paper proposes GCAR — a two-stream GCN architecture that adds **channel attention** and **residual connections** to graph convolution for sign language recognition. It's tested on four large-scale datasets (WLASL, MSL, ASLLVD, PSL) but NOT on INCLUDE.

It's a good reference for architectural techniques you can incorporate into your ISL project.

---

## The Architecture (GCAR System)

### Two Streams:
**Stream 1 — Joint Skeleton Stream:**
- Raw joint coordinates as input
- Sep-TCN (Separable Temporal CNN) for initial feature extraction
- Multiple GCA (Graph Convolution with Attention) modules
- Residual connections throughout

**Stream 2 — Joint Motion Stream:**
- Frame-to-frame joint movement as input (joint position difference between consecutive frames)
- Same Sep-TCN + GCA + residual connection structure
- Mirrors Stream 1 exactly

**Fusion:** Concatenate the feature vectors from both streams → classification layer

### GCA Module (key innovation):
- Standard graph convolution
- Channel attention module: learns which feature channels are most informative for distinguishing signs
- The attention dynamically weights features for non-connected skeleton points that may move together during specific signs

### Sep-TCN (Separable Temporal CNN):
- A depthwise separable version of temporal convolution
- Reduces parameters significantly (fewer than standard TCN)
- Helps prevent overfitting on smaller datasets

---

## Key Results
| Dataset | Accuracy |
|---------|---------|
| WLASL | **90.31%** |
| MSL (Mexican SL) | **94.10%** |
| ASLLVD | **99.75%** |
| PSL (Pakistani SL) | **34.41%** |

Only **0.69 million parameters** — very lightweight compared to HA-GCN's multi-stream setup.

---

## Why PSL is Only 34.41%?
PSL (Pakistani SL) is the largest and most diverse dataset they test on — more signers, more lexical variation, more challenging. This shows that large-vocabulary generalization remains an open problem.

---

## What's Important for YOUR Project

### 1. Channel Attention is Useful for Small Datasets
INCLUDE has only 4,287 videos. Channel attention helps the model focus on the most discriminative feature dimensions — reduces overfitting risk compared to treating all channels equally.

### 2. Residual Connections = Stable Training
With 263 sign classes and limited data, deep networks can be hard to train. Residual (skip) connections (like ResNet) allow gradients to flow through more easily and prevent vanishing gradient problems.

### 3. Sep-TCN = Parameter Efficiency
Separable temporal convolution dramatically reduces parameter count. For INCLUDE's small dataset, fewer parameters = less overfitting risk. Consider using Sep-TCN in your temporal modeling layer.

### 4. Motion Stream is Valuable
The joint motion stream (frame differences) consistently improves accuracy. For ISL where the TRAJECTORY of a sign matters (not just the static pose), motion features are important. Your ablation A2 could test joint-only vs. joint+motion streams.

### 5. Lightweight Model Advantage
0.69M parameters means this model could run in real-time on edge devices. If your ISL project targets deployment, parameter efficiency is a design consideration.

---

## What's Missing / Gaps
- Not tested on INCLUDE — can't directly compare numbers
- PSL results (34.41%) suggest the architecture struggles with larger, more diverse vocabularies
- No signer-independent evaluation

---

## One-line summary
> GCAR adds channel attention and residual connections to a two-stream (joint + motion) GCN for SLR, achieving 90-99% on 3 of 4 datasets with only 0.69M parameters — useful techniques to incorporate in your ISL model.
