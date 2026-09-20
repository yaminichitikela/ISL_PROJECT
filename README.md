# ISL_PROJECT

Skeleton-based Indian Sign Language (ISL) recognition on the [INCLUDE](https://doi.org/10.1145/3394171.3413528) dataset using an ST-GCN variant, with a graph-topology ablation as the main study and a benchmark-provenance audit as a supporting finding.

Accompanying paper: *Graph Topology Ablation and Evaluation Protocols for Skeleton-Based Indian Sign Language Recognition* (`IEEE-conference-template-062824/`, not tracked in this repo).

## Headline results

Official INCLUDE split rebuilt from AI4Bharat's path lists (3,121/347/816 train/val/test, 262 classes; see `data/DATA_CARD.md`), no training-time augmentation.

| Config | Test acc. | Seeds |
|---|---|---|
| 1D-CNN baseline | 90.9% | 1 |
| ST-GCN, single-partition graph (K=1) | 81.8% ± 2.2pp | 3 |
| ST-GCN, fixed spatial partitions (K=3) | 90.7% ± 1.7pp | 3 |
| ST-GCN, adaptive adjacency | 93.4% ± 3.1pp | 3 |
| ST-GCN, adaptive, T=96, torso norm. (best) | **96.2% ± 0.9pp** | 3 |

A capacity-matched control (widening the single-partition model to ~2.42M parameters to match the spatial model) shows the spatial advantage is not explained by parameter count alone, under either torso or raw coordinate normalization. Full numbers, per-seed breakdowns, and every caveat are in the paper.

## Repository layout

- `src/` — core library: `graph.py` (53-joint skeleton + adjacency construction), `model.py` (ST-GCN), `dataset.py` (loading, augmentation, resampling), `train.py` (CLI training entry point), `evaluate.py`.
- `scripts/` — one-off experiment runners, each independently reproducible:
  - `run_topology_equal_budget.py` — single / spatial / adaptive topology ablation.
  - `run_temporal_window_equal_budget.py` — T ∈ {32,48,64,96} sweep.
  - `run_multiseed_headline.py` — ≥2-seed reruns of the two headline configs.
  - `run_best_combo_multiseed.py` — T=96 + bone-length combination check.
  - `run_capacity_matched_single.py` / `run_capacity_matched_raw.py` — capacity-matched single-vs-spatial control, torso and raw coordinates.
  - `run_two_hand_subset_eval.py` — single-hand vs. two-hand subset accuracy.
  - `build_raw_bonelength_official.py` — builds raw/bone-length-normalized arrays from the official split.
  - `run_inflation.py`, `isolate_augmentation_bug.py` — historical leakage and augmentation-bug diagnostics.
- `notebooks/` — the original pipeline, in order: `01_extract_keypoints` → `02_preprocess` → `03_graph_topology` → `04_baselines` → `05_eda` → `06_stgcn_verify` → `07_stgcn_train` → `08_leakage_topology_equalbudget` → `09_ablations_A2_A4_A5` → `10_si_evaluation`.
- `data/` — `DATA_CARD.md` (dataset provenance, licence, known issues), `official_splits/` (AI4Bharat path lists), `processed/` (preprocessed arrays; large files gitignored). Raw videos and extracted keypoints are not committed.
- `results/` — `metrics_all.csv` (every run, keyed by experiment/condition/T), `CHANGELOG.md` (causal attribution for headline-number changes across the project's history), plus ablation-specific CSVs.
- `checkpoints/` — trained model weights (gitignored; regenerate via the scripts above).

## Reproducing a result

Each script in `scripts/` is self-contained and appends its results to `results/metrics_all.csv`. For a single custom run, use the CLI directly, e.g.:

```bash
python src/train.py --topology adaptive --T 96 --augmentation none --normalisation torso
```

`--topology` accepts `single`, `dual` (fixed spatial partitions — an internal name, not a two-hand-specific graph; see the paper's Method section), `adaptive`, `hands42`, or `latefusion`.

## Known limitations

INCLUDE releases no public signer identities, so no verified signer-independent evaluation is possible from this dataset alone; a recording-session/filename-order proxy is reported as an indicative supplementary probe only, never as signer-independent. Several ablations are single-seed and/or used test-accuracy-informed configuration selection; both are disclosed explicitly in the paper.
