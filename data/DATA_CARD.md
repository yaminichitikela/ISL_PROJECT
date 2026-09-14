# Data Card — INCLUDE Dataset

## Dataset Identity

| Field | Value |
|---|---|
| Name | INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition |
| Version | v1.0 (HuggingFace snapshot, accessed March 2026) |
| Paper | Sridhar et al., "INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition", ACM Multimedia 2020 |
| Source | HuggingFace: `ai4bharat/include-dataset`; original: AI4Bharat GitHub |
| Licence | **Creative Commons Attribution 4.0 International (CC-BY 4.0)** |
| DOI | https://doi.org/10.1145/3394171.3413528 |

## Licence & Restrictions

- **Permitted**: academic research, non-commercial use, redistribution with attribution, creation of derivative works (e.g. extracted keypoints) provided the original licence is preserved.
- **Required**: cite Sridhar et al. (2020) in any publication that uses this dataset or derivatives.
- **Prohibited**: commercial deployment without explicit permission from AI4Bharat; removal of attribution.
- The extracted keypoint arrays in `data/raw_keypoints/` are a derivative work of INCLUDE and inherit the CC-BY 4.0 licence.

## Content Summary

| Field | Value |
|---|---|
| Sign language | Indian Sign Language (ISL) — Chennai regional variant |
| Sign classes | 263 word-level signs across 15 semantic categories |
| Signers | 15 (identities anonymised) |
| Total videos | ~4,287 (approx. 15 per sign × 263 signs + re-takes) |
| Video format | `.MOV`, recorded with standard webcams |
| Resolution | Variable (typically 1280×720) |
| Frame rate | ~30 fps |
| Duration | 0.5–4 s per sign (median ~2 s at 30 fps) |

## Official Splits

Ground truth is `AI4Bharat/INCLUDE` GitHub repo, `train_test_paths/{include_train,include_val,include_test}.txt` (saved locally in `data/official_splits/`) — 4,292 unique videos, zero cross-split overlap. The HuggingFace parquet mirror is a **separate, corrupted re-packaging** of this same benchmark (duplicate rows and train/val/test path overlap); do not use it as the source of truth for splits.

| Split | Videos (official path lists) |
|---|---|
| Train | 3,127 |
| Val | 348 |
| Test | 817 |

**Local corpus reconciliation (2026-08-21):** the original Milestone 1 Zenodo download left `include_videos/` at 3,679 `.MOV` files (613 short of the 4,292 official total). Root cause: some source videos are physically MP4-encoded but the official path lists always name them with a `.MOV` suffix — 605 of the "missing" videos were sitting in `include_videos/` all along as `.MP4` files, invisible to any `.MOV`-only glob (including `notebooks/01_extract_keypoints.ipynb`'s scan and this data card's original count). These were re-fetched from the Zenodo record (`zenodo.org/records/4010759`) and copied in under their official `.MOV` name; the original redundant `.MP4` copies were deleted after a byte-size check confirmed they were identical. Local corpus is now **4,284 / 4,292 (99.8%)**.

The remaining 8 videos (`Days_and_Time/Second (Number)/MVI_{4647,4648,4649,4650,5503,5504,5505,5506}.MOV`) do not exist anywhere in the Zenodo archive under any sign folder — the actual folder is `81. Second`, not `Second (Number)`, and none of its video IDs match. This is the known corrupted-path issue below; these 8 rows are excluded from training/evaluation and are believed unrecoverable from this source.

Note: one label (`"Second (Number)"`) appears in the parquet with incorrect video paths — those rows are skipped during preprocessing (see `notebooks/02_preprocess.ipynb`).

## Derivative Artefacts in This Repository

| File / Folder | Description | Licence |
|---|---|---|
| `data/raw_keypoints/*.npy` | Per-video MediaPipe keypoints, shape (T, 53, 3) | CC-BY 4.0 (derivative) |
| `data/processed/X_sd_*.npy` | Resampled + normalised keypoint arrays, shape (N, 64, 53, 3) | CC-BY 4.0 (derivative) |
| `data/processed/y_sd_*.npy` | Integer class labels | CC-BY 4.0 (derivative) |
| `data/processed/si_splits.pkl` | 5-fold signer-independent split indices | CC-BY 4.0 (derivative) |
| `data/processed/label_encoder.pkl` | sklearn LabelEncoder for 262 sign classes | CC-BY 4.0 (derivative) |

## Known Limitations

1. **Regional variant**: INCLUDE covers ISL as used in Chennai, Tamil Nadu. ISL varies across India; results may not generalise to other regional variants.
2. **Controlled recording**: Videos were recorded in a lab setting (consistent background, lighting). Real-world performance may differ.
3. **Signer IDs not provided**: The HuggingFace parquet does not include explicit signer identifiers. Signer IDs in `data/signer_ids.csv` are inferred by sorting video filenames within each sign class — documented as a study limitation.
4. **Missing label**: `"Second (Number)"` exists in the parquet but has corrupted video-path assignments; all such rows are excluded from training/evaluation.
5. **Word-level only**: INCLUDE contains isolated word signs, not continuous sign language sentences.

## Citation

```bibtex
@inproceedings{sridhar2020include,
  title     = {INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition},
  author    = {Sridhar, Advaith and Ganesan, Rohith Gandhi and Kumar, Pratyush
               and Khapra, Mitesh},
  booktitle = {Proceedings of the 28th ACM International Conference on
               Multimedia (ACM MM)},
  year      = {2020},
  doi       = {10.1145/3394171.3413528}
}
```
