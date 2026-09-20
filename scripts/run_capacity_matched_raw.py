"""
Capacity-matched single-partition vs. fixed-spatial, RAW coordinates (2026-09-18).

Follow-up to scripts/run_capacity_matched_single.py, which showed that
widening the single-partition model to match the fixed-spatial model's
parameter count (2,420,451 vs. 2,419,527) does not close the topology gap
under TORSO normalization (79.00+/-1.70% vs 90.69+/-1.70% test). Since the
normalization ablation (Table tab:normalization) found raw coordinates
change results substantially relative to torso, this script re-runs the
same capacity-matched comparison under RAW (unnormalized) coordinates to
check whether the topology effect (not just capacity) is specific to torso
normalization.

Six runs: {single-widened-matched, spatial} x seeds {42, 0, 1}.
Widths are fixed (same channels as the torso run); only the input
coordinate normalization changes (data/processed/X_raw_sd_*.npy, T=64,
resampled to T=32 here, same mechanism as every other T-resampled run in
this project). No augmentation. Same training schedule and equal budget
(SGD+Nesterov lr=0.01, cosine annealing, weight_decay=1e-4, batch=32, max
80 epochs, patience=10, validation-accuracy-only checkpoint selection) as
scripts/run_topology_equal_budget.py and run_capacity_matched_single.py.

New rows appended to metrics_all.csv: experiment='topology',
condition in {'single-widened-matched-raw'[_seed0/_seed1],
'spatial-matched-raw'[_seed0/_seed1]}, T=32.
"""
import os, sys, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

ROOT = Path("/Users/yamini/Desktop/projects/ISL PROJECT")
sys.path.insert(0, str(ROOT / "src"))

from graph   import build_adjacency
from model   import STGCN
from dataset import SkeletonDataset

N_EPOCHS = 80
PATIENCE = 10
T        = 32
SEEDS    = [42, 0, 1]
PROC_DIR = ROOT / "data" / "processed"

CONDITIONS = {
    "single-widened-matched-raw": dict(strategy="uniform", adaptive=False, channels=(70, 136, 280)),
    "spatial-matched-raw":        dict(strategy="spatial", adaptive=False, channels=(64, 128, 256)),
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}  |  T={T}  |  max_epochs={N_EPOCHS}  |  patience={PATIENCE}  |  normalization=raw")


def load_raw_loaders(batch_size=32, num_workers=0):
    X_train = np.load(PROC_DIR / "X_raw_sd_train.npy")
    y_train = np.load(PROC_DIR / "y_sd_train.npy")
    X_val   = np.load(PROC_DIR / "X_raw_sd_val.npy")
    y_val   = np.load(PROC_DIR / "y_sd_val.npy")
    X_test  = np.load(PROC_DIR / "X_raw_sd_test.npy")
    y_test  = np.load(PROC_DIR / "y_sd_test.npy")
    assert len(X_train) == len(y_train) == 3121
    assert len(X_val)   == len(y_val)   == 347
    assert len(X_test)  == len(y_test)  == 816

    train_ds = SkeletonDataset(X_train, y_train, augment=False, resample_T=T)
    val_ds   = SkeletonDataset(X_val,   y_val,   augment=False, resample_T=T)
    test_ds  = SkeletonDataset(X_test,  y_test,  augment=False, resample_T=T)

    kw = dict(num_workers=num_workers, pin_memory=False)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
            DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw))


def run_one(cond_name, strategy, adaptive, channels, seed):
    label = cond_name if seed == 42 else f"{cond_name}_seed{seed}"
    print(f"\n{'='*60}\n  [{label}]  seed={seed}  strategy={strategy}  channels={channels}  T={T}\n{'='*60}")

    set_seed(seed)
    train_loader, val_loader, test_loader = load_raw_loaders()

    A = build_adjacency(n_joints=53, strategy=strategy)
    set_seed(seed)
    model = STGCN(n_classes=262, A=A, adaptive=adaptive, channels=channels).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                                 nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0
    t0           = time.time()

    for ep in range(1, N_EPOCHS + 1):
        model.train(); correct = total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.detach().argmax(1) == y).sum().item()
            total   += len(y)
        tr_acc = correct / total

        model.eval(); vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                vc += (model(x).argmax(1) == y).sum().item()
                vt += len(y)
        vl_acc = vc / vt
        scheduler.step()

        elapsed = (time.time() - t0) / 60
        print(f"  [{label}] ep {ep:3d}/{N_EPOCHS}  tr={tr_acc:.4f}  vl={vl_acc:.4f}  ({elapsed:.1f}m)", flush=True)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  [{label}] Early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    model.eval()
    correct = total = 0
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += len(y)
            y_true_all.extend(y.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())
    test_acc = correct / total
    f1       = f1_score(y_true_all, y_pred_all, average='macro', zero_division=0)
    total_min = (time.time() - t0) / 60

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{label.replace('-', '_')}_T{T}.pt"
    torch.save({"state_dict": best_state, "condition": label, "T": T,
                "adaptive": adaptive, "channels": channels, "n_params": n_params,
                "seed": seed, "normalization": "raw"}, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")
    print(f"\n  [{label}] FINAL  seed={seed}  val={best_val_acc:.4f}  test={test_acc:.4f}  macro-F1={f1:.4f}  params={n_params:,}  ({total_min:.1f}m)")

    return dict(label=label, cond_name=cond_name, seed=seed, val_acc=best_val_acc,
                test_acc=test_acc, macro_f1=f1, n_params=n_params)


all_results = []
for cond_name, cfg in CONDITIONS.items():
    for seed in SEEDS:
        r = run_one(cond_name, cfg["strategy"], cfg["adaptive"], cfg["channels"], seed)
        all_results.append(r)

print("\n" + "="*70)
print(f"CAPACITY-MATCHED SINGLE vs SPATIAL, RAW COORDINATES (T={T})")
print("="*70)

summary_rows = []
for cond_name in CONDITIONS:
    rows = [r for r in all_results if r["cond_name"] == cond_name]
    test_vals = np.array([r["test_acc"] for r in rows])
    val_vals  = np.array([r["val_acc"]  for r in rows])
    f1_vals   = np.array([r["macro_f1"] for r in rows])
    print(f"\n  {cond_name} (params={rows[0]['n_params']:,}):")
    for r in rows:
        print(f"    seed={r['seed']:<3} val={r['val_acc']:.4f}  test={r['test_acc']:.4f}  F1={r['macro_f1']:.4f}")
    print(f"    MEAN  val={val_vals.mean():.4f}  test={test_vals.mean():.4f}  F1={f1_vals.mean():.4f}")
    print(f"    STD   val={val_vals.std(ddof=1):.4f}  test={test_vals.std(ddof=1):.4f}  F1={f1_vals.std(ddof=1):.4f}")
    summary_rows.append(dict(experiment='topology', condition=f'{cond_name}_mean3seed', T=T,
                              val_acc=round(val_vals.mean(), 4),
                              test_acc=round(test_vals.mean(), 4),
                              macro_f1=round(f1_vals.mean(), 4)))
print("="*70)

# ── Update metrics_all.csv ────────────────────────────────────────────────────
csv_path = ROOT / "results" / "metrics_all.csv"
df = pd.read_csv(csv_path)

new_rows = pd.DataFrame([
    dict(experiment='topology', condition=r['label'], T=T,
         val_acc=round(r['val_acc'], 4),
         test_acc=round(r['test_acc'], 4),
         macro_f1=round(r['macro_f1'], 4))
    for r in all_results
] + summary_rows)

df = df[~((df['experiment']=='topology') & (df['T']==T) &
          (df['condition'].str.contains('-raw')))].copy()
df = pd.concat([df, new_rows], ignore_index=True)
df = df.sort_values(['experiment', 'T', 'condition']).reset_index(drop=True)
df.to_csv(csv_path, index=False)
print(f"\nUpdated: {csv_path}")
