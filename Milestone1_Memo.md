# Milestone 1 — Scoping Memo
**Project:** Skeleton-Based Indian Sign Language Recognition Using ST-GCN  
**Author:** Yamini Chitikela, Vizuara Research Team  
**Date:** April 2026  

---

## 1. Research Questions

**RQ1.**
Can an ST-GCN trained on MediaPipe skeleton keypoints achieve competitive or superior accuracy compared to the published INCLUDE baselines (CNN/LSTM on RGB frames)?

**RQ2.**
How does signer-independent evaluation (leave-N-signers-out) degrade performance relative to signer-dependent (random-split) evaluation, and which sign categories are most affected?

**RQ3.**
Does explicitly modelling two-hand graph topology (dual-graph or merged-graph) improve recognition of two-handed signs compared to a naïve single-graph approach?

---

## 2. Graph Topology Rationale

### 2.1 Joint Selection — 53 Joints
MediaPipe Holistic produces 75 keypoints per frame: 33 body-pose + 21 left-hand + 21 right-hand landmarks.

This project uses a **53-joint subset**:

| Group | Joints | Count |
|-------|--------|-------|
| Left hand | All finger joints (MCP, PIP, DIP, TIP × 5 fingers + wrist) | 21 |
| Right hand | All finger joints (MCP, PIP, DIP, TIP × 5 fingers + wrist) | 21 |
| Upper body | Nose, left/right shoulder, left/right elbow, left/right wrist + 4 more | 11 |
| **Total** | | **53** |

**Why not all 75?** The lower body (hips, knees, ankles, feet) is irrelevant for ISL — all signs are performed in the upper body signing space.

### 2.2 Spatial Edges
Spatial edges follow the **natural bone structure** — joints are connected as they are anatomically (e.g., wrist → MCP → PIP → DIP → TIP along each finger).

### 2.3 Inter-Hand Edge
ISL uses predominantly two-handed signs. To allow the model to reason about coordination between both hands, a chain of edges connects the two hands through the shoulder:

```
Left wrist → Left elbow → Left shoulder → Right shoulder → Right elbow → Right wrist
```

This is the only path connecting the left and right hand sub-graphs.

### 2.4 Single-Hand Signs
For signs that use only one hand, the **inactive hand's keypoints are zeroed out** during preprocessing. This masking strategy will be investigated in Ablation A2 (graph topology ablation).

### 2.5 Why Skeleton-Based ST-GCN for ISL?
- RGB-based models (CNN+LSTM) are sensitive to background, clothing, and lighting
- Skeleton keypoints are signer-appearance-invariant by construction — a natural fit for signer-independent generalisation
- The graph structure explicitly encodes hand/body topology, unlike flat feature vectors from CNNs

---

## 3. Evaluation Plan

### 3.1 Metrics
| Metric | Why |
|--------|-----|
| Top-1 Accuracy | Primary classification metric |
| Top-5 Accuracy | Secondary; useful for near-miss analysis |
| Macro F1-Score | Critical — 263 classes are imbalanced (~16 videos/class average) |
| Per-category accuracy | Across 15 INCLUDE word categories; single-hand vs. two-hand breakdown |

### 3.2 Evaluation Protocols

**Protocol 1 — Signer-Dependent (SD):**
- Official INCLUDE train/test split
- Carve 15% validation from training set
- Used for: comparison with all prior published work

**Protocol 2 — Signer-Independent (SI) — Primary Novel Contribution:**
- Leave-N-signers-out cross-validation
- 5 folds; 3 signers held out per fold (15 signers ÷ 5 folds)
- Train on 12 signers, test on 3 held-out signers per fold
- Report mean ± std across all 5 folds
- SI is the harder, more realistic protocol — no published results on INCLUDE before this project

### 3.3 Baselines to Compare Against

| Baseline | Modality | Source |
|----------|----------|--------|
| I3D | RGB video | INCLUDE paper (Sridhar et al. 2020) |
| CNN+LSTM (MobileNetV2+BiLSTM) | RGB video | INCLUDE paper — 85.6% |
| CNN+LSTM | Pose-overlay video | INCLUDE paper — 82.2% |
| XGBoost | Skeleton only | INCLUDE paper — 63.1% |
| LSTM on flattened keypoints | Skeleton | To be implemented |
| 1D-CNN on keypoint sequences | Skeleton | To be implemented |
| ST-GCN (HA-GCN paper baseline) | Skeleton | Song et al. 2025 — 99.02% SD |
| HA-GCN multi-stream | Skeleton | Song et al. 2025 — 99.63% SD (no SI reported) |

---

## 4. Deliverable #1 Acceptance Check

- [x] `literature_matrix.csv` — 12 papers, all columns filled
- [x] This memo — all 3 RQs stated, graph topology documented, evaluation plan defined
- [ ] `notebooks/00_mediapipe_smoke_test.ipynb` — MediaPipe extracts 53 keypoints from 5 sample INCLUDE videos without errors; skeleton overlay visualised

---

*Next step: complete the MediaPipe smoke test notebook, then proceed to Milestone 2 (Weeks 3–4).*
