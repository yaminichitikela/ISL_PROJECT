# Paper 12 — PRISMA Systematic Review: Recent Advances in Skeleton-Based Sign Language Recognition
**Sembayev & Akbarov, Vestnik KazNPU, 2025**

---

## What is this paper about?
This is a **systematic literature review** (PRISMA protocol) of 19 skeleton-based SLR papers published between January 2015 and December 2024. It's the most comprehensive survey in your reading list.

Reading this paper is like reading a summary of the entire field — it tells you where things stand RIGHT NOW, what works, what doesn't, and what the open problems are.

---

## How the Review Was Conducted
- Searched Scopus database → 207 papers found
- Removed duplicates, non-skeleton papers → 19 papers remained for final analysis
- Each paper analyzed for: dataset, pose source, architecture, streams, metrics, results

---

## The Field's Trajectory (5-axis taxonomy)

The review organizes the field across 5 dimensions:

| Dimension | What varies | Current trend |
|-----------|-------------|---------------|
| Task | Isolated SLR vs. Continuous SLT | Both active, ISLR dominant |
| Input encodings | Joint positions, bones, motion, velocity | Multi-stream is now standard |
| Model families | ST-GCN, CNN+RNN, Transformers, Hybrids | ST-GCN variants dominate |
| Evaluation protocols | Signer-dependent vs. independent | SD still most common |
| Dataset anchors | AUTSL, WLASL, CSL core benchmarks | INCLUDE, BosphorusSign22k growing |

---

## Key Findings from the Review

### What Works (validated across papers)
- **Multi-stream GCN** (joint + bone + motion) consistently beats single-stream: 3-5% improvement
- **ST-GCN + variants** are the dominant architecture — 11 of 19 papers use ST-GCN or extensions
- **MediaPipe / OpenPose / MMPose** are standard pose extractors — OpenPose for older papers, MediaPipe for newer
- **Top-1 accuracy on curated isolated datasets** typically reaches 93-98% range
- **AUTSL, WLASL, CSL** are the core benchmarks; INCLUDE is growing

### What Doesn't Work Well
- **Large-vocabulary WLASL** subsets depress accuracy to 63% on SD and 24% on SI (WLASL-2000)
- **Continuous SLR** streaming reports WER = 11-19% offline — still far from deployment-ready
- **Skeleton-only** approaches still lag behind RGB on some challenging datasets

### Open Problems (directly relevant to your project)
1. **Signer-independent evaluation** is rare — most papers only report signer-dependent results
2. **Robustness to pose estimation errors** — what happens when MediaPipe gives bad keypoints?
3. **Generalization across datasets** — models trained on AUTSL don't always transfer to INCLUDE
4. **3D body-face-hand integration** — combining face expressions with skeleton for grammatical markers
5. **Lightweight models for mobile deployment** — most SOTA models are too heavy for phones

---

## Summary Table of Best Results (from the review's Table 3)

| Paper | Dataset | Architecture | Best Top-1 |
|-------|---------|-------------|------------|
| HA-GCN [Song 2025] | AUTSL, INCLUDE | Multi-stream GCN + hand graphs | 96.00% AUTSL, 99.63% INCLUDE |
| BosphorusSign22k | BosphorusSign22k | ST-GCN + Multi-Cue LSTM | 92.58% |
| CSL-500 paper | CSL-500 | Attention-enhanced Multi-Scale GCN | 98.08% |
| WLASL-100 paper | WLASL-100/300 | Graph Conv + General DNN | 90.31% |

---

## What's Important for YOUR Project

### 1. Your Work is Novel
The review confirms: **signer-independent evaluation on INCLUDE has not been published**. This is your primary novel contribution — validated by an independent systematic review.

### 2. Multi-stream is Essential
All top-performing papers use multi-stream. Your roadmap's Milestone 3 (dual-stream with joint + bone) is aligned with the field's best practices.

### 3. The Evaluation Gap
The review explicitly notes that Top-1 on small, signer-dependent INCLUDE may be "too easy" — the field needs more challenging evaluations. Your SI evaluation addresses exactly this.

### 4. Modality Comparison Table (Table 1 in paper)
The paper provides a clear comparison of all modalities:
- RGB: high dimensionality, computationally expensive, privacy concerns
- Depth: needs special hardware, struggles with reflective surfaces
- Sensor (wearables): intrusive, uncomfortable
- **Skeleton: computationally efficient, privacy-preserving, lighting-robust** ← your choice is justified

### 5. Dataset Comparison (Table 2)
Very useful for your related work section — shows INCLUDE alongside AUTSL, WLASL, CSL etc. with direct statistics comparison.

---

## How to Use This in Your Paper
- Cite this review in your Related Work section to establish the field's state
- Use their taxonomy (Table 1) to position your work
- Use their findings to argue why SI evaluation is needed
- Use their modality comparison to justify skeleton-based approach

---

## One-line summary
> A PRISMA systematic review of 19 skeleton-based SLR papers confirms: multi-stream GCN dominates, signer-independent evaluation is rarely published, and INCLUDE's SI evaluation is an open research gap — directly validating your project's novel contribution.
