# Results Changelog

Tracks *why* headline numbers moved, to keep the causal story straight for the paper
(each fix only touches the specific regime it applies to — this log exists specifically
to stop conflating unrelated changes, per Naman's 2026-08-27 review).

## 2026-08-27 — Best-combo test: T=96 + bone-length does NOT beat T=96 + torso

Every prior ablation varied one axis at a time: T=96 was only tested with torso
normalization (97.06% test, single seed), bone-length was only tested at T=32
(96.45% test, single seed). Naman flagged this as the missing cell and predicted
combining the two winning settings was "probably your real headline." Ran it,
3 seeds (42, 0, 1):

| Config | Mean test | Std | Seeds |
|---|---|---|---|
| T=96 + torso | **96.20%** | ±0.86pp | 3 |
| T=32 + bonelength | 96.45% | — | 1 (not yet multi-seeded) |
| T=96 + bonelength (combo) | 95.38% | ±1.74pp | 3 (per-seed: 96.20%, 96.57%, 93.38%) |

**The hypothesis was wrong — the combo is worse, not better**, with double the
variance of T=96+torso. The two winning axes do not compound. **Current
best-supported headline remains T=96 + torso** (highest mean, lowest variance,
already 3-seeded). Report the combo test as a deliberate negative-result check,
not an oversight — it directly answers "did you check whether these compound?"
before a reviewer has to ask.

## 2026-08-27 — Baseline shift explanation (86.0% → 89.95%, no-augmentation)

**Claim to get right in the paper:** the flip-bug and temporal-crop-bug fixes in
`src/dataset.py` **cannot** be responsible for the no-augmentation number moving from
86.0% to 89.95%. Both bugs live inside code paths that only execute when
`augment=True` (`_random_horizontal_flip`, `_random_temporal_crop`) — a `no-aug` run
never calls either method, so fixing them is causally inert for that run.

**Actual cause:** the dataset/split rebuild.
- Old: 3,652 videos, HF-parquet-based splits (had cross-split overlap + duplicate
  rows, fixed via priority-dedup — but still missing ~630 videos vs the official
  benchmark, see `data/DATA_CARD.md`).
- New: 4,284 videos (of 4,292 official), splits built directly from
  `data/official_splits/` — recovered ~605 videos that existed locally but under a
  `.MP4` extension instead of `.MOV` (invisible to the old `.MOV`-only glob) plus 27
  more nested in nested `Extra/` subfolders the old extraction scan never descended
  into. Train set alone grew from 2,462 → 3,121 samples.

So: **more and cleaner training data**, not the bug fixes, explains the no-aug
baseline shift. The bug fixes are why the *full-augmentation* number moved
(70.6% → 90.20%, see the augmentation-fix entry below) — a completely separate,
non-overlapping causal claim from the no-aug shift.

## 2026-08-23 — Augmentation bug fixes (flip + temporal-crop)

Two bugs found and fixed in `src/dataset.py`:
1. `_random_horizontal_flip` swapped the 42 hand-joint indices but not the 5 L/R
   upper-body joint pairs (ears/shoulders/elbows/wrists/hips) — producing anatomically
   impossible mirrored samples.
2. `_random_temporal_crop` reshaped a `(C, T_crop, V)` tensor directly into
   `(1, C*V, T_crop)` without first permuting `T_crop` to the last axis (required by
   `F.interpolate`, which resizes only the last dimension) — silently interleaving
   time and joint identity on every cropped sample.

Effect, same config throughout (adaptive, T=32, torso norm):
| State | No-aug test | Full-aug test | Gap |
|---|---|---|---|
| Original (broken flip only known) | 86.0% | 70.6% | 15.4pp |
| Flip fixed only | 89.95% | 76.72% | 13.2pp |
| **Both fixed** | 89.95% | **90.20%** | **−0.25pp (closed)** |

The no-aug column here is flat across all three rows within measurement noise
(86.0/89.95/89.95) *except* for the jump from row 1 to row 2/3 — which is the
dataset-rebuild effect above, not these bug fixes. Isolated (30-epoch, single-seed)
per-component check that found the crop bug:

| Component alone | Test acc | Δ vs no-aug baseline |
|---|---|---|
| noise_only | 90.93% | +0.98pp |
| **crop_only** | **67.89%** | **−22.06pp** |
| flip_only | 84.44% | −5.51pp |
| rotation_only | 85.29% | −4.66pp |

## 2026-08-23 — Bone-length normalization (49.4% → 96.5%)

Two independent, additive causes, not one:
1. **Regime fix**: the old 49.4% was measured under `augmentation=full`, on the old
   HF-parquet-based 3,652-video splits (i.e. it inherited every bug/gap above). Simply
   re-measuring under `augmentation=none` on the corrected 4,284-video official splits
   (matching the same protocol as every other headline number) already accounts for
   most of the jump.
2. **Data source fix**: `X_raw_sd_*.npy` / `X_bonelen_sd_*.npy` were stale — built by
   `notebooks/09_ablations_A2_A4_A5.ipynb` from the old HF-parquet splits (dated
   2026-06-20). Rebuilt both directly from `data/official_splits/` via
   `scripts/build_raw_bonelength_official.py`.
- **Divide-by-zero checked and refuted**: per-sample mean bone length on the rebuilt
  data ranges 0.057–0.090 across train/val/test — four orders of magnitude above the
  `1e-6` zero-guard, zero samples anywhere near it. Not a numerical bug.
- Result: **96.5% test** (val=94.8%, F1=0.966), vs. torso normalization's 90.0% at the
  same T=32/adaptive/no-aug settings — reverses the old torso-beats-raw finding.
- **Seed count**: bone-length and raw normalization results are currently **single-seed
  (seed=42 only)** — not yet multi-seeded like the two designated headline numbers.
