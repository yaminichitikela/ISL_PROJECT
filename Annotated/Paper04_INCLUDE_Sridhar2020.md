# Paper 4 — INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition
**Sridhar et al., ACM Multimedia 2020**

---

## What is this paper about?
This paper **introduces the dataset you use for your entire project**. Before INCLUDE, there was no large, publicly available ISL dataset. Researchers were using tiny private collections that couldn't be compared across studies. This paper fixes that.

---

## Dataset at a Glance
| Property | Value |
|----------|-------|
| Language | Indian Sign Language (ISL) |
| Total sign classes | 263 |
| Total videos | 4,287 |
| Signers | 15 experienced signers |
| Recording location | School for the deaf, Chennai |
| Video resolution | 1920 × 1080 pixels |
| Frame rate | 25 fps |
| Average video length | ~2.5 seconds |

### Sign categories (15 total):
Adjectives, Animals, Body Parts, Clothes, Colors, Days/Months/Years, Environment, Family, Food, Greetings, Health/Medical, Numbers, Places, Transport, Vegetables

---

## Key Baselines the Paper Reports
| Method | Input Type | Accuracy (263 classes) |
|--------|-----------|----------------------|
| MobileNetV2 + BiLSTM | RGB video | **85.6%** |
| MobileNetV2 + BiLSTM | Pose-overlay video | 82.2% |
| XGBoost | Skeleton coordinates only | 63.1% |

**The skeleton baseline (63.1%) is much lower than the RGB baseline (85.6%).** This is the central challenge your project addresses — can GCN-based skeleton methods close this gap?

---

## Important Properties of ISL (from the paper)
- **Mostly two-handed:** ISL signs typically use both hands simultaneously
- **Upper signing space:** Signs performed near the forehead encode social hierarchy
- **Compound signs:** One sign can be composed of two concepts (Male + Sibling = Brother)
- **Non-manual markers:** Facial expressions and head tilt add grammatical meaning
- **Regional variation:** The dataset is Chennai-specific; ISL has dialects across India

---

## What's NOT in the Paper (gaps your project fills)
1. **No signer-independent evaluation** — the paper only does random 80/20 train/test split. Your project is the FIRST to report signer-independent (leave-signers-out) evaluation on INCLUDE.
2. **No advanced GCN baseline** — only XGBoost for skeleton. Your ST-GCN pipeline is a new contribution.
3. **No per-category analysis** — doesn't break down which sign categories are hardest.

---

## How to Evaluate on INCLUDE
- **Signer-Dependent (SD):** Random stratified 80/20 split across all signers — this is the official split, use for comparison with prior work
- **Signer-Independent (SI) [YOUR NOVEL CONTRIBUTION]:** 5-fold cross-validation, hold out 3 signers per fold — never done before on INCLUDE

---

## What's Important for YOUR Project
- This is your **primary dataset** — everything you do is tested here
- The 63.1% skeleton baseline is your floor — your ST-GCN should significantly exceed this
- The gap between 63.1% (skeleton) and 85.6% (RGB) is what you're trying to close
- HA-GCN (paper 8) already achieves 99.39% on INCLUDE — you need to know this number
- With only ~16 videos per class and 15 signers, **overfitting is a real risk** — augmentation (Ablation A4) and normalization (Ablation A5) are critical
- Report **Macro F1** alongside Top-1 accuracy because class sizes are unequal

---

## One-line summary
> INCLUDE is the ISL dataset your project runs on — 263 signs, 4,287 videos, 15 signers — and your project is the first to do signer-independent evaluation on it.
