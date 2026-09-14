"""
The winning combination nobody has actually run yet (Naman review, 2026-08-27):
adaptive topology + T=96 + bone-length normalisation + augmentation=none, 3 seeds.

Every prior "best" number was one axis at a time:
  - T=96, torso:      97.06% test (seed 42), best across the temporal-window sweep
  - T=32, bonelength: 96.45% test (seed 42), best across the normalisation sweep
Nobody has combined the two winning axes. Naman: "Run that one cell. It is
probably your real headline."

Reuses train.py's run_training() directly (same code path as every other
train.py-based ablation) so this is apples-to-apples with everything else,
just called 3x with different seeds and CSV rows written with an explicit
seed suffix so seeds don't clobber each other (train.py's own _update_csv
upserts by (experiment,condition,T) with no seed component).
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/yamini/Desktop/projects/ISL PROJECT")
sys.path.insert(0, str(ROOT / "src"))

from train import run_training

CSV_PATH = ROOT / "results" / "metrics_all.csv"


def upsert_csv(experiment, condition, T, val_acc, test_acc, macro_f1):
    df = pd.read_csv(CSV_PATH)
    mask = (df["experiment"] == experiment) & (df["condition"] == condition) & (df["T"] == T)
    row = dict(experiment=experiment, condition=condition, T=T,
               val_acc=round(val_acc, 4), test_acc=round(test_acc, 4), macro_f1=round(macro_f1, 4))
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)


results = []
for seed in (42, 0, 1):
    print(f"\n{'#'*70}\n  BEST-COMBO seed={seed}: adaptive, T=96, bonelength, aug=none\n{'#'*70}")
    r = run_training(
        topology="adaptive", T=96, augmentation="none", normalisation="bonelength",
        epochs=80, patience=10, seed=seed, verbose=True,
    )
    # run_training already wrote checkpoints/adaptive_T96_none_bonelength.pt and an
    # unsuffixed CSV row each time (overwriting itself) -- capture per-seed results
    # here and additionally write seed-suffixed rows so all 3 survive side by side.
    upsert_csv("best_combo", f"adaptive_T96_none_bonelength_seed{seed}", 96,
               r["val_acc"], r["test_acc"], r["macro_f1"])
    results.append(dict(seed=seed, **r))
    print(f"  seed={seed}: val={r['val_acc']:.4f} test={r['test_acc']:.4f} F1={r['macro_f1']:.4f}")

print("\n" + "=" * 70)
print("BEST COMBO: adaptive, T=96, bonelength, augmentation=none -- 3 seeds")
print("=" * 70)
import numpy as np
val_vals  = np.array([r["val_acc"] for r in results])
test_vals = np.array([r["test_acc"] for r in results])
f1_vals   = np.array([r["macro_f1"] for r in results])
for r in results:
    print(f"  seed={r['seed']}: val={r['val_acc']:.4f} test={r['test_acc']:.4f} F1={r['macro_f1']:.4f}")
print(f"\n  MEAN test = {test_vals.mean():.4f}  STD = {test_vals.std(ddof=1):.4f}")
print(f"  MEAN F1   = {f1_vals.mean():.4f}  STD = {f1_vals.std(ddof=1):.4f}")

upsert_csv("best_combo", "adaptive_T96_none_bonelength_mean3seed", 96,
           val_vals.mean(), test_vals.mean(), f1_vals.mean())
print("\nWrote per-seed + mean rows to results/metrics_all.csv under experiment='best_combo'")
