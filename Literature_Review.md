# Literature Review: ISL Sign Language Recognition Using ST-GCN
**Project: Skeleton-Based ISL Recognition on the INCLUDE Dataset**
**Prepared by: Yamini Chitikela, Vizuara Research Team**

---

## Overview — How to Read This Document

This review covers all 12 papers in the reading list. It is organized to build your understanding progressively:
1. **What the field is doing** (Section 1 — field overview)
2. **The dataset you're using** (Section 2 — INCLUDE)
3. **The model family you're using** (Section 3 — GCN progression)
4. **Tools and infrastructure** (Section 4 — pose estimation)
5. **Direct competitors** (Section 5 — models tested on INCLUDE)
6. **What your project contributes** (Section 6 — novelty analysis)
7. **Results comparison table** (Section 7 — numbers at a glance)

For deep notes on each individual paper → see the `/Annotated/` folder.

---

## Section 1 — The Field at a Glance

Sign language recognition (SLR) is the task of automatically identifying sign language gestures from video. It matters because:
- Over 5 million deaf and hard-of-hearing people use Indian Sign Language (ISL) in India alone
- Only a small number of certified interpreters exist — technology can bridge this gap
- Current commercial solutions rely on RGB video, requiring powerful hardware and raising privacy concerns

**Two main approaches:**

| Approach | Input | Pros | Cons |
|----------|-------|------|------|
| RGB-based | Raw video pixels | High accuracy | Privacy risk, lighting-sensitive, computationally heavy |
| Skeleton-based | Joint coordinates | Lightweight, private, lighting-robust | Lower accuracy historically |

**The skeleton gap is closing.** The PRISMA review (Paper 12) of 19 recent papers shows that skeleton-based GCN methods now reach 93-99% accuracy on curated isolated recognition datasets. The original INCLUDE skeleton baseline (63.1% — XGBoost, 2020) severely underrepresented what modern GCN methods can do.

**Dominant architecture in 2024-2025: ST-GCN and variants.** 11 of 19 papers in the systematic review use spatial-temporal graph convolutional networks. This validates your project's architectural choice.

---

## Section 2 — The INCLUDE Dataset (Paper 4)

**What it is:** The first large-scale, publicly available Indian Sign Language dataset. Collected at a school for the deaf in Chennai, India.

**Key numbers:**
- 263 sign classes (15 categories: Animals, Numbers, Family, etc.)
- 4,287 videos from 15 experienced signers
- 1920×1080 resolution, 25 fps, ~2.5 seconds average duration
- Only ~16 videos per class on average

**Evaluation protocols:**
- **Signer-Dependent (SD):** 80/20 random split — official INCLUDE protocol
- **Signer-Independent (SI):** Leave-signers-out — NOT published for INCLUDE before your project

**Published baselines on INCLUDE:**

| Method | Modality | Top-1 (263 classes) |
|--------|----------|-------------------|
| XGBoost | Skeleton only | 63.1% |
| MobileNetV2 + BiLSTM | Pose video | 82.2% |
| MobileNetV2 + BiLSTM | RGB video | **85.6%** |
| ST-GCN (HA-GCN baseline) | Skeleton | 99.02% |
| HA-GCN (multi-stream) | Skeleton | **99.63%** |

**Key insight:** The 63.1% skeleton baseline in the original INCLUDE paper used XGBoost — a weak classifier on raw joint angles. Modern GCNs (HA-GCN) achieve 99.63% on the same dataset. The gap is not a fundamental limitation of skeleton data — it was a limitation of the method.

**ISL linguistic properties that affect design:**
- Mostly two-handed signs → you need both hands in your graph
- Compound morphology (Male + Sibling = Brother) → temporal modeling must span full sign duration
- Upper signing space encodes social hierarchy → normalizing relative to torso important
- Non-manual markers (facial grammar) → skeleton-only approach misses this

---

## Section 3 — The GCN Model Progression (Papers 1, 2, 3, 6, 8, 10)

Your project follows a deliberate progression through GCN architectures. Here's how they connect:

### 3.1 ST-GCN (Paper 1) — The Foundation
**Yan et al., AAAI 2018**

The original idea: model a skeleton as a graph. Joints = nodes, bones = edges. Apply graph convolution spatially (which joints are connected?) and temporal convolution along time.

**What it introduced:**
- Spatial configuration partitioning (root / centripetal / centrifugal)
- Learnable edge importance masks M_k
- End-to-end training from raw coordinates

**Limitation for signs:** Fixed adjacency matrix — can't learn that non-adjacent joints co-move in specific signs.

---

### 3.2 2s-AGCN (Paper 2) — Adaptive Graph
**Shi et al., CVPR 2019**

Upgrades the fixed graph to a learnable one. Adjacency = fixed anatomy + global learned adjustment + per-sample computed attention.

Adds a second stream: bone vectors (joint direction) alongside joint coordinates.

**Why this matters for ISL:** ISL compound signs involve non-physical joint co-movements. 2s-AGCN's data-dependent C_k component can learn these without hardcoding them.

**Result:** +7% over ST-GCN on NTU-RGBD benchmark.

---

### 3.3 MS-G3D (Paper 3) — Multi-Scale + Unified Space-Time
**Liu et al., CVPR 2020**

Two fixes:
1. Disentangled k-adjacency (exact k-hop, not up to k-hop) — removes biased counting
2. G3D operator — processes spatial and temporal dimensions simultaneously via cross-spacetime skip connections

**Why this matters for ISL:** Signs unfold at multiple speeds (fast fingerspelling vs. slow compound signs). Multi-scale aggregation captures both. Motivates your Ablation A1 (testing T = 32, 48, 64, 96 frame windows).

**Result:** +3% over 2s-AGCN on NTU-RGBD.

---

### 3.4 GCN-BERT (Paper 6) — Hybrid Spatial-Temporal
**Tunga et al., WACV 2021**

Replaces temporal convolution with BERT self-attention. Per-frame GCN for spatial features, BERT transformer for long-range temporal dependencies. Skip connection preserves frame-level detail.

**Why this matters:** Shows that transformer-based temporal modeling is a viable alternative to convolution — relevant if you want to add transformer components in a future extension.

**Result:** 60.15% on WLASL-100 (+5% over previous GCN baseline).

---

### 3.5 GCAR (Paper 10) — Lightweight with Attention
**Miah et al., IEEE Access 2024**

Two-stream (joint + joint motion) with channel attention modules and residual connections. Very lightweight (0.69M parameters) but achieves 90-99% on 3 of 4 datasets.

**Why this matters for ISL:** INCLUDE is small (4,287 videos). Lightweight models with attention overfit less than large models. Channel attention helps focus on discriminative feature dimensions.

---

### 3.6 HA-GCN (Paper 8) — Current SOTA on INCLUDE
**Song et al., Journal of Information and Intelligence, 2025**

The most important paper for your project. HA-GCN adds hand-specific sub-graphs to the standard body graph:
- **Structured Hand (SH):** Physical hand anatomical connections
- **Parameterized Hand (PH):** Fully learned hand connections

Also adds:
- Adaptive DropGraph (regularization)
- 4-stream fusion (joint, bone, joint motion, bone motion)

**Results on INCLUDE:**
- Single stream: **99.39%**
- Multi-stream: **99.63%**

This is your ceiling — your project must compare against these numbers and argue where you add value (signer-independent evaluation, ISL-specific graph design).

---

### GCN Progression Summary

```
ST-GCN (2018)          →    fixed graph, anatomical edges
    ↓ +7%
2s-AGCN (2019)         →    adaptive graph, bone stream added
    ↓ +3%
MS-G3D (2020)          →    multi-scale aggregation, unified space-time
    ↓ hand-aware
HA-GCN (2025)          →    hand-specific sub-graphs, 4-stream, SOTA
    ↑
YOUR PROJECT           →    ISL-specific 53-joint graph, signer-independent eval
```

---

## Section 4 — Pose Estimation Infrastructure (Papers 5, 7, 11)

### 4.1 MediaPipe Holistic / BlazePose (Paper 5)
**Google Research, 2020**

Your pose extraction backbone. Gives 75 keypoints per frame:
- 33 BlazePose body keypoints
- 21 left hand keypoints
- 21 right hand keypoints

Real-time (30+ fps on mobile), free, works from regular RGB camera.

**Your project uses 53 joints:** Left hand (21) + Right hand (21) + Upper body (11: nose, shoulders, elbows, wrists + 2 more). Lower body excluded — irrelevant for ISL.

### 4.2 ST-GCN for SLR Pipeline (Paper 7)
**Amorim et al., 2020**

Shows how to adapt ST-GCN for sign language step by step. Their 6-step pipeline (acquire → segment → estimate skeletons → filter keypoints → split → normalize/serialize) is the template for your preprocessing. They use 27 joints (5 per hand + 7 body). You use 53 (21 per hand + 11 body) — more detail for better accuracy.

### 4.3 MediaPipe Hand ROI Fix (Paper 11)
**Moryossef, 2024**

Fixes MediaPipe's tendency to misplace the hand bounding box when hands are rotated. Proposes an MLP replacement for the heuristic ROI calculation (+6% IoU). INCLUDE is a controlled dataset so this bug may not severely affect you, but check your extracted keypoints for quality.

---

## Section 5 — Direct Competitors (models tested on INCLUDE)

Only one paper in your reading list tests on INCLUDE with a skeleton-based GCN:

### HA-GCN (Paper 8) — Song et al. 2025
| Protocol | Top-1 Accuracy |
|----------|---------------|
| Signer-Dependent (SD) | 99.63% |
| Signer-Independent (SI) | **Not reported** |

**Your project's competitive positioning:**
- On SD evaluation: you're unlikely to beat 99.63% with ST-GCN alone; aim for competitive performance with less compute
- On SI evaluation: HA-GCN has NO published results — this is your open territory

---

## Section 6 — What YOUR Project Contributes (Novelty Analysis)

Based on the full literature review, here is where your project's contributions are genuine:

### Novel Contribution 1 (STRONGEST): Signer-Independent Evaluation on INCLUDE
No paper in the literature (including HA-GCN) has published signer-independent cross-validation results on the INCLUDE dataset. Your 5-fold SI evaluation (3 signers held out per fold from 15 total) is genuinely new and addresses a real gap identified in Paper 12's systematic review.

**Why it matters:** Signer-dependent evaluation tests memorization of signers. SI tests true generalization — far more relevant for real-world deployment.

### Novel Contribution 2 (SOLID): ISL-Specific 53-Joint Graph Design
The 53-joint topology (21L + 21R + 11 upper body) with explicit inter-hand edges through the shoulder chain is designed specifically for ISL's predominantly two-handed signing. HA-GCN uses a generic 27-joint graph from INCLUDE's default keypoint set. A richer, purpose-designed ISL graph is a meaningful technical contribution.

### Novel Contribution 3 (SUPPORTING): Systematic Ablation on INCLUDE
No prior work systematically ablates temporal window size (A1), graph topology (A2), adjacency type (A3), augmentation (A4), and normalization (A5) on INCLUDE. This ablation study establishes reproducible design choices for future ISL research.

### What is NOT Novel
- Applying ST-GCN to sign language — done (Papers 1, 7, 8)
- Using MediaPipe for pose extraction — standard
- Adaptive adjacency matrix — 2s-AGCN (2019)
- Multi-stream GCN — widespread
- Hand-aware sub-graphs — HA-GCN (2025)

---

## Section 7 — Results at a Glance

### Skeleton-Based SLR Results on INCLUDE

| Method | Year | Architecture | Top-1 SD | Top-1 SI |
|--------|------|-------------|---------|---------|
| XGBoost (INCLUDE paper) | 2020 | Hand-crafted + ML | 63.1% | — |
| MobileNetV2+BiLSTM (pose video) | 2020 | CNN+RNN | 82.2% | — |
| **HA-GCN (single stream)** | **2025** | **Hand-aware GCN** | **99.39%** | **—** |
| **HA-GCN (multi-stream)** | **2025** | **Hand-aware GCN** | **99.63%** | **—** |
| **YOUR PROJECT (target)** | **2025** | **ISL-GCN** | **~97-99%** | **TBD (FIRST)** |

### GCN Architecture Performance Across Benchmarks

| Method | NTU-60 X-Sub | NTU-60 X-View | AUTSL | INCLUDE SD |
|--------|-------------|--------------|-------|-----------|
| ST-GCN | 81.5% | 88.3% | 91.99% | 99.02% |
| 2s-AGCN | 88.5% | 95.1% | — | — |
| MS-G3D | 91.5% | 96.2% | — | — |
| HA-GCN | — | — | 96.82% | **99.63%** |

*Note: INCLUDE numbers are very high because the dataset is signer-dependent and relatively small — 15 signers, 4,287 videos. Signer-independent numbers will be significantly lower.*

---

## Section 8 — Gaps and Open Questions

From the full literature review, the following open questions directly affect your project:

1. **How much does SI accuracy drop vs. SD on INCLUDE?** — Nobody knows. HA-GCN reports 99.63% SD. A significant drop in SI would validate the importance of your contribution.

2. **Which sign categories are hardest?** — No per-category analysis exists for INCLUDE. Numbers vs. Animals vs. Greetings — do they require different temporal scales?

3. **How many frames are needed?** — Your Ablation A1 (T = 32, 48, 64, 96) will be the first systematic answer to this for ISL.

4. **Does MediaPipe quality limit the ceiling?** — If keypoint extraction is noisy (especially for fast compound signs), even a perfect GCN will underperform. Paper 11's fix could help.

5. **Is 15 signers enough for generalization?** — With only 15 signers, even 5-fold SI leaves only 3 held-out signers. The variance will be high. This is a limitation to acknowledge.

---

## Reading Order Recommendation

If you're new to this area, read in this order to build understanding:

1. **Paper 4 (INCLUDE)** — understand your dataset first
2. **Paper 5 (MediaPipe)** — understand how you get skeleton data
3. **Paper 1 (ST-GCN)** — understand the base architecture
4. **Paper 2 (2s-AGCN)** — understand the adaptive upgrade
5. **Paper 8 (HA-GCN)** — understand the current SOTA on your dataset
6. **Paper 12 (PRISMA review)** — see the full field picture
7. **Papers 3, 6, 7, 10** — supporting methods and techniques
8. **Papers 9, 11** — adjacent topics and tools

---

*Last updated: April 2026*
*All paper details → see /Annotated/ folder for individual paper notes*
