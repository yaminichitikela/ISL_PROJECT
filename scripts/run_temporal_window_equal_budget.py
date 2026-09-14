"""
Temporal-window ablation — adaptive topology, T in {32, 48, 64, 96}, augmentation OFF.

CORRECTED RE-RUN (2026-08-22): the previous temporal-window numbers in
metrics_all.csv (experiment='temporal', condition='adaptive-T{32,48,64,96}')
were trained on the old 3,652-video HF-parquet splits, under augment_train=True
with the broken horizontal-flip augmentation (src/dataset.py swapped hand
joints but not the L/R body-joint pairs). This script re-runs the ablation on
the new official-path-based splits (data/processed/, rebuilt from
data/official_splits/), with the fixed flip, and augmentation OFF — matching
the equal-budget topology ablation (scripts/run_topology_equal_budget.py).

T=32 is NOT re-trained here: it is identical in every setting (adaptive
topology, T=32, augmentation off, equal budget) to the "adaptive-noaug-fixed"
row already produced by run_topology_equal_budget.py
(val=0.9078, test=0.9191, F1=0.9206) — re-running it would just reproduce the
same numbers under the same fixed seed. T=48/64/96 are trained fresh here.

Unlike the topology script's first version, this one saves model checkpoints
(checkpoints/adaptive_T{T}_none_torso_fixed.pt) from the start.

New rows appended to metrics_all.csv (condition suffixed "-noaug-fixed"):
  temporal | adaptive-T48-noaug-fixed | T=48 | ...
  temporal | adaptive-T64-noaug-fixed | T=64 | ...
  temporal | adaptive-T96-noaug-fixed | T=96 | ...
"""
import os, sys, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = Path("/Users/yamini/Desktop/projects/ISL PROJECT")
sys.path.insert(0, str(ROOT / "src"))

from graph   import build_adjacency
from model   import STGCN
from dataset import load_sd_loaders

SEED     = 42
N_EPOCHS = 80
PATIENCE = 10
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)

set_seed()

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}  |  max_epochs={N_EPOCHS}  |  patience={PATIENCE}")

A_adaptive = build_adjacency(n_joints=53, strategy="spatial")


def run_experiment(T):
    label = f"adaptive-T{T}-noaug-fixed"
    print(f"\n{'='*60}")
    print(f"  [{label}]  T={T}  max_ep={N_EPOCHS}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = load_sd_loaders(
        ROOT / "data" / "processed",
        batch_size=32, num_workers=0, augment_train=False, resample_T=T
    )

    set_seed()
    model = STGCN(n_classes=262, A=A_adaptive, adaptive=True).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9,
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

    ckpt_path = CKPT_DIR / f"adaptive_T{T}_none_torso_fixed.pt"
    torch.save({"state_dict": best_state, "topology": "adaptive", "T": T}, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    print(f"\n  [{label}] FINAL  val={best_val_acc:.4f}  test={test_acc:.4f}  macro-F1={f1:.4f}  ({total_min:.1f}m)")
    return dict(label=label, T=T, val_acc=best_val_acc, test_acc=test_acc, macro_f1=f1)


results = []
for T in (48, 64, 96):
    r = run_experiment(T)
    results.append(r)

print("\n" + "="*70)
print(f"TEMPORAL-WINDOW ABLATION AT EQUAL BUDGET, AUGMENTATION OFF")
print("="*70)
print(f"{'Condition':<26} {'T':>4}  {'Val Acc':>9}  {'Test Acc':>9}  {'Macro-F1':>9}")
print(f"{'-'*65}")
print(f"  {'adaptive-noaug-fixed (T32, reused)':<24} {32:>4}  {0.9078:>9.4f}  {0.9191:>9.4f}  {0.9206:>9.4f}")
for r in results:
    print(f"  {r['label']:<24} {r['T']:>4}  {r['val_acc']:>9.4f}  {r['test_acc']:>9.4f}  {r['macro_f1']:>9.4f}")
print("="*70)

csv_path = ROOT / "results" / "metrics_all.csv"
df = pd.read_csv(csv_path)

new_rows = pd.DataFrame([
    dict(experiment='temporal', condition=r['label'], T=r['T'],
         val_acc=round(r['val_acc'], 4),
         test_acc=round(r['test_acc'], 4),
         macro_f1=round(r['macro_f1'], 4))
    for r in results
])

df = df[~((df['experiment'] == 'temporal') & (df['condition'].str.endswith('-noaug-fixed')))].copy()
df = pd.concat([df, new_rows], ignore_index=True)
df = df.sort_values(['experiment', 'T', 'condition']).reset_index(drop=True)
df.to_csv(csv_path, index=False)
print(f"\nUpdated: {csv_path}")
print(df[df['experiment'].isin(['topology', 'temporal'])].to_string(index=False))
