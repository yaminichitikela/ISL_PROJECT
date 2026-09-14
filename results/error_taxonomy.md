# Error Taxonomy — ISL ST-GCN (INCLUDE)

**Checkpoint:** `single_T32_full_torso.pt`  
**Test Top-1:** 0.6655  **Top-5:** 0.9102564102564102  **Macro-F1:** 0.6560  **Total failures analysed:** 287

Five error categories are defined based on structural heuristics applied to each misclassified test sample. Categories are assigned in priority order (keypoint dropout → fine-grained → two-hand → temporal → style).

---
## Keypoint Dropout  (15 failures, 5.2% of errors)

**Root cause:** MediaPipe fails to track one or both hands in ≥50% of frames (mean hand joint validity < 0.5). The ST-GCN receives near-zero coordinates for invisible joints; the spatial graph convolution propagates this noise through the hand subgraph, corrupting the sign's spatial pattern.

**Representative failures:**

| # | Sample | Ground truth | Predicted | Confidence | Notes |
|---|--------|-------------|-----------|------------|-------|
| 1 | idx=3 | 54. Cell phone | 48. Green | 0.404 | Mean hand joint visibility < 50 % hv=0.453 |
| 2 | idx=83 | 66. Brother | 70. Grandmother | 0.882 | Mean hand joint visibility < 50 % hv=0.461 |
| 3 | idx=169 | 87. hot | 69. Grandfather | 0.153 | Mean hand joint visibility < 50 % hv=0.43 |
| 4 | idx=170 | 61. Father | 77. Boy | 0.150 | Mean hand joint visibility < 50 % hv=0.375 |
| 5 | idx=253 | 80. Minute | 75. Yesterday | 0.358 | Mean hand joint visibility < 50 % hv=0.469 |

_...and 10 more (showing first 5)._

---
## Single vs Two-Hand Confusion  (0 failures, 0.0% of errors)

**Root cause:** The ground-truth sign uses two hands while the predicted sign uses one (or vice versa). Both hands share the same graph topology; when one hand is consistently dominant the model may not learn the bilateral symmetry cue robustly enough.

_No failures assigned to this category in the current evaluation run._

---
## Temporal Aliasing / Static Sign  (26 failures, 9.1% of errors)

**Root cause:** The skeleton shows very low inter-frame motion (bottom 10th percentile of mean velocity). This often indicates a static sign (e.g. hand shape held for the full clip), over-segmented gesture, or looped frames during resampling to T=32. ST-GCN's temporal convolutions find little discriminative signal across frames.

**Representative failures:**

| # | Sample | Ground truth | Predicted | Confidence | Notes |
|---|--------|-------------|-----------|------------|-------|
| 1 | idx=121 | 47. Red | 57. Colour | 0.607 | Very low inter-frame motion — possible temporal aliasing / static sign vel=0.11807 hv=1.0 |
| 2 | idx=125 | 94. good | 95. bad | 0.449 | Very low inter-frame motion — possible temporal aliasing / static sign vel=0.11441 hv=1.0 |
| 3 | idx=142 | 34. Pen | 19. Ball | 0.438 | Very low inter-frame motion — possible temporal aliasing / static sign vel=0.11241 hv=0.961 |
| 4 | idx=195 | 3. Fish | 87. Doctor | 0.463 | Very low inter-frame motion — possible temporal aliasing / static sign vel=0.11493 hv=0.844 |
| 5 | idx=278 | 78. Girl | 92. old | 0.243 | Very low inter-frame motion — possible temporal aliasing / static sign vel=0.11563 hv=1.0 |

_...and 21 more (showing first 5)._

---
## Signer Style Variation  (244 failures, 85.0% of errors)

**Root cause:** The sign is well-detected (good hand visibility, normal motion) but the signer's execution deviates from the training distribution: e.g., different hand orientation, signing speed, or path trajectory. With ~9 samples/class INCLUDE cannot cover full signer variability.

**Representative failures:**

| # | Sample | Ground truth | Predicted | Confidence | Notes |
|---|--------|-------------|-----------|------------|-------|
| 1 | idx=1 | 78. Year | 34. Ground | 0.875 | No structural cue found — attributed to signer execution variability hv=0.961 |
| 2 | idx=8 | 35. Bank | 56. Pleased | 0.235 | No structural cue found — attributed to signer execution variability hv=0.812 |
| 3 | idx=18 | 78. Girl | 96. wet | 0.148 | No structural cue found — attributed to signer execution variability hv=0.938 |
| 4 | idx=20 | 67. Monday | 68. Tuesday | 0.664 | No structural cue found — attributed to signer execution variability hv=1.0 |
| 5 | idx=21 | 84. small little | 13. Bicycle | 0.497 | No structural cue found — attributed to signer execution variability hv=1.0 |

_...and 239 more (showing first 5)._

---
## Fine-Grained Hand Shape Confusion  (2 failures, 0.7% of errors)

**Root cause:** Ground-truth and predicted sign belong to the same INCLUDE semantic category (same numeric prefix) but differ in a subtle hand-shape detail — e.g., bent vs. extended fingers, wrist rotation. The spatial GCN with 53 joints may not resolve these fine-grained configurations reliably at 32 frames.

**Representative failures:**

| # | Sample | Ground truth | Predicted | Confidence | Notes |
|---|--------|-------------|-----------|------------|-------|
| 1 | idx=141 | 53. Fan | 53. Orange | 0.489 | Same category '53' — fine-grained shape confusion hv=1.0 |
| 2 | idx=150 | 53. Fan | 53. Orange | 0.474 | Same category '53' — fine-grained shape confusion hv=1.0 |

---
## Summary

| Error Type | Count | % of errors |
|------------|------:|-------------|
| Keypoint Dropout | 15 | 5.2% |
| Single vs Two-Hand Confusion | 0 | 0.0% |
| Temporal Aliasing / Static Sign | 26 | 9.1% |
| Signer Style Variation | 244 | 85.0% |
| Fine-Grained Hand Shape Confusion | 2 | 0.7% |

### Implications for future work

1. **Keypoint dropout**: Use confidence-weighted adjacency or a    learned validity mask channel (already scaffolded in `dataset.py`).
2. **Fine-grained hand shape**: A higher-resolution hand sub-graph    (dedicated 21-joint branch per hand) may resolve subtle    hand-shape differences — evaluated in ablation A2c (late fusion).
3. **Temporal aliasing**: Static signs may benefit from an explicit    'no-motion' detection branch or longer temporal windows with    stride-1 convolutions.
4. **Signer style variation**: Data augmentation (spatial jitter,    rotation) and larger training sets per class are the primary    remedies.
5. **Two-hand confusion**: Explicit handedness channel or    asymmetric graph partition (left vs right subgraph with    different weights) could help.

_Generated by `src/evaluate.py`._