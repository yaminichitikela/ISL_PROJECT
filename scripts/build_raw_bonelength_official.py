"""
Rebuild X_raw_sd_*.npy and X_bonelen_sd_*.npy directly from the official
AI4Bharat path lists (data/official_splits/), replacing the stale versions
that notebooks/09_ablations_A2_A4_A5.ipynb built from the old, corrupted
HF-parquet splits (dated 2026-06-20, from the 3,652-video corpus).

X_raw_sd_*  : gap-fill (fix_missing_hand_v2) + resample to T=64, NO torso
              normalization — logic copied from notebooks/02_preprocess.ipynb
              cells 4-5, minus the normalize_skeleton() step.
X_bonelen_sd_* : X_raw divided by each sample's own mean inter-joint bone
              length (60 edges, averaged over all T frames) — logic copied
              from notebooks/09_ablations_A2_A4_A5.ipynb cell 6.

Used to investigate Naman's bone-length-normalization concern (49.4% old
result "looks like divide-by-zero on zeroed missing-hand joints") on the
correct, official-path-based data before re-running the ablation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/yamini/Desktop/projects/ISL PROJECT")
sys.path.insert(0, str(ROOT / "src"))

from graph import _EDGES

DATA_DIR     = ROOT / "data"
RAW_KP_DIR   = DATA_DIR / "raw_keypoints"
PROC_DIR     = DATA_DIR / "processed"
OFFICIAL_DIR = DATA_DIR / "official_splits"
T_TARGET = 64


def resample_sequence(kps, t_target=T_TARGET):
    T_src = kps.shape[0]
    if T_src == t_target:
        return kps.astype(np.float32)
    src_idx = np.arange(T_src, dtype=np.float32)
    tgt_idx = np.linspace(0, T_src - 1, t_target, dtype=np.float32)
    out = np.zeros((t_target, 53, 3), dtype=np.float32)
    for j in range(53):
        for c in range(3):
            out[:, j, c] = np.interp(tgt_idx, src_idx, kps[:, j, c])
    return out


def fix_missing_hand_v2(kps):
    T = kps.shape[0]
    filled = kps.copy().astype(np.float32)
    for j in range(53):
        col = kps[:, j, :]
        detected = ~np.all(col == 0, axis=1)
        if not detected.any():
            continue
        valid_frames = np.where(detected)[0]
        for c in range(3):
            filled[:, j, c] = np.interp(
                np.arange(T, dtype=np.float64),
                valid_frames.astype(np.float64),
                col[valid_frames, c],
            )
    return filled


def load_raw(npy_name):
    path = RAW_KP_DIR / npy_name
    if not path.exists():
        return None
    kps = np.load(str(path))
    if kps.shape[0] < 2:
        return None
    kps = fix_missing_hand_v2(kps)
    kps = resample_sequence(kps, T_TARGET)
    return kps


def bonelength_normalize(X):
    edges = np.array(_EDGES, dtype=np.int64)
    bone_vecs = X[:, :, edges[:, 0], :] - X[:, :, edges[:, 1], :]
    bone_lens = np.linalg.norm(bone_vecs, axis=-1)
    mean_bl = bone_lens.mean(axis=(1, 2))
    n_near_zero = int((mean_bl < 1e-6).sum())
    n_small     = int((mean_bl < 0.01).sum())
    print(f"    mean_bl stats: min={mean_bl.min():.5f}  p1={np.percentile(mean_bl,1):.5f}  "
          f"median={np.median(mean_bl):.5f}  max={mean_bl.max():.5f}")
    print(f"    samples with mean_bl < 1e-6 (zero-guard triggers): {n_near_zero}")
    print(f"    samples with mean_bl < 0.01 (scale-blowup risk):   {n_small}")
    mean_bl_safe = np.where(mean_bl < 1e-6, 1.0, mean_bl)
    return (X / mean_bl_safe[:, None, None, None]).astype(np.float32), mean_bl


def main():
    meta_df = pd.read_csv(DATA_DIR / "keypoints_metadata.csv")
    path_to_npy = dict(zip(meta_df["rel_path"], meta_df["out_name"]))
    print(f"path_to_npy: {len(path_to_npy)} entries")

    for split in ("train", "val", "test"):
        paths = (OFFICIAL_DIR / f"include_{split}.txt").read_text().splitlines()
        X_list, skipped = [], 0
        for i, rel_path in enumerate(paths):
            if i % 500 == 0:
                print(f"  [{split}] {i}/{len(paths)}", end="\r")
            npy_name = path_to_npy.get(rel_path)
            if npy_name is None:
                skipped += 1
                continue
            kps = load_raw(npy_name)
            if kps is None:
                skipped += 1
                continue
            X_list.append(kps)
        X = np.stack(X_list).astype(np.float32)
        print(f"\n  [{split}] built {len(X_list)} samples, {skipped} skipped (matches y_sd_{split}.npy row count check below)")

        y = np.load(PROC_DIR / f"y_sd_{split}.npy")
        assert len(X) == len(y), f"{split}: X has {len(X)} samples but y_sd_{split}.npy has {len(y)} — order/count mismatch!"

        np.save(PROC_DIR / f"X_raw_sd_{split}.npy", X)
        print(f"  Saved X_raw_sd_{split}.npy {X.shape}")

        X_bl, mean_bl = bonelength_normalize(X)
        np.save(PROC_DIR / f"X_bonelen_sd_{split}.npy", X_bl)
        print(f"  Saved X_bonelen_sd_{split}.npy {X_bl.shape}")


if __name__ == "__main__":
    main()
