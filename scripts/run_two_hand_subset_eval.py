"""
Two-hand-sign subset accuracy for RQ3 (Naman's review item 4).

"Overall accuracy does not test the [topology] claim" — a richer graph
topology (dual/adaptive) should matter most on signs that need coordinated
two-hand motion, and matter least on single-hand signs. This script:

1. Classifies each of the 262 sign classes as "single-hand" or "two-hand"
   using the same velocity-ratio heuristic as the original (pre-fix)
   notebooks/09_ablations_A2_A4_A5.ipynb cell 29: for each class, compare
   mean left-hand vs right-hand frame-to-frame motion on the test set; a
   class where the quieter hand moves less than 50% as much as the active
   hand is "single-hand".
2. Loads the three corrected, augmentation-off checkpoints from
   scripts/run_topology_equal_budget.py's 2026-08-23 re-run
   (single_graph_noaug_fixed_T32.pt, dual_graph_noaug_fixed_T32.pt,
   adaptive_noaug_fixed_T32.pt) and evaluates each on the official-path-based
   test split (data/processed/X_sd_test.npy, 816 samples).
3. Reports overall / single-hand-subset / two-hand-subset accuracy per
   topology, replacing the old results/ablation_A2_topology_subset.csv
   (which was computed on the pre-fix, full-augmentation checkpoints on the
   old 3,652-video splits).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/Users/yamini/Desktop/projects/ISL PROJECT")
sys.path.insert(0, str(ROOT / "src"))

from graph    import build_adjacency
from model    import STGCN
from dataset  import SkeletonDataset
from torch.utils.data import DataLoader

PROC_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"
N_CLASSES = 262
T = 32

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Load test split ────────────────────────────────────────────────────────────
X_te = np.load(PROC_DIR / "X_sd_test.npy")   # (N, 64, 53, 3) — torso-normalised, T=64 stored
y_te = np.load(PROC_DIR / "y_sd_test.npy")
print(f"Test set: {X_te.shape}, {len(np.unique(y_te))} classes present")

# ── Classify each class as single-hand or two-hand (velocity-based) ──────────
l_hand_vel = np.abs(np.diff(X_te[:, :, :21,  :], axis=1)).mean(axis=(1, 2, 3))   # (N,)
r_hand_vel = np.abs(np.diff(X_te[:, :, 21:42,:], axis=1)).mean(axis=(1, 2, 3))   # (N,)

hand_type = {}
for cls in range(N_CLASSES):
    mask = y_te == cls
    if mask.sum() == 0:
        continue
    l_mean = l_hand_vel[mask].mean()
    r_mean = r_hand_vel[mask].mean()
    ratio  = min(l_mean, r_mean) / (max(l_mean, r_mean) + 1e-9)
    hand_type[cls] = "two_hand" if ratio > 0.50 else "single_hand"

two_hand_classes    = {c for c, t in hand_type.items() if t == "two_hand"}
single_hand_classes = {c for c, t in hand_type.items() if t == "single_hand"}
print(f"Two-hand classes  : {len(two_hand_classes)}")
print(f"Single-hand classes: {len(single_hand_classes)}")

single_mask = np.isin(y_te, list(single_hand_classes))
two_mask    = np.isin(y_te, list(two_hand_classes))
print(f"Test samples — single: {single_mask.sum()}, two: {two_mask.sum()}")


def run_inference(model, X, T, device, batch_size=64):
    ds = SkeletonDataset(X, np.zeros(len(X), dtype=np.int64), resample_T=T, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ── Evaluate each corrected topology checkpoint ───────────────────────────────
topology_ckpts = {
    "single (53j fixed-uniform)": ("single_graph_noaug_fixed_T32.pt", "uniform", False),
    "dual (53j fixed-spatial)":   ("dual_graph_noaug_fixed_T32.pt",   "spatial", False),
    "adaptive (53j learnable)":   ("adaptive_noaug_fixed_T32.pt",     "spatial", True),
}

rows = []
for label, (ckpt_name, strategy, adaptive) in topology_ckpts.items():
    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"  SKIP {ckpt_name} — not found")
        continue

    A = build_adjacency(n_joints=53, strategy=strategy)
    model = STGCN(n_classes=N_CLASSES, A=A, adaptive=adaptive).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    y_pred = run_inference(model, X_te, T, DEVICE)

    overall = (y_pred == y_te).mean()
    s_acc = (y_pred[single_mask] == y_te[single_mask]).mean() if single_mask.sum() > 0 else float("nan")
    t_acc = (y_pred[two_mask]    == y_te[two_mask]).mean()    if two_mask.sum()    > 0 else float("nan")

    rows.append(dict(
        topology=label,
        overall=round(overall, 4),
        single_hand=round(s_acc, 4),
        two_hand=round(t_acc, 4),
        single_n=int(single_mask.sum()),
        two_n=int(two_mask.sum()),
    ))
    print(f"  {label}: overall={overall:.4f}  single={s_acc:.4f}  two_hand={t_acc:.4f}")

df = pd.DataFrame(rows)
print("\n=== Two-hand-sign subset accuracy (corrected, aug-off, official splits) ===")
print(df.to_string(index=False))

out_path = ROOT / "results" / "ablation_A2_topology_subset_fixed.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print("(old results/ablation_A2_topology_subset.csv left in place as the documented-invalid pre-fix record)")
