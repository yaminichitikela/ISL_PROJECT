"""
Equal-budget topology ablation — T=32, max 80 epochs, patience=10, augmentation OFF.

CORRECTED RE-RUN (2026-08-22): the previous version of this script (and the
adaptive-T32 result it read from metrics_all.csv) trained under
augment_train=True with a broken horizontal-flip augmentation (src/dataset.py
swapped hand joints but not the L/R body-joint pairs, producing anatomically
impossible training samples). That regime gave the confounded ranking
Dual 75.4 > Adaptive 70.6 > Single 66.6 (topology,dual-graph/topology,single-graph
+ ablation,adaptive_full_torso rows in metrics_all.csv, all on the old
3,652-video HF-parquet splits). This version:
  - runs on the new official-path-based splits (data/processed/, rebuilt from
    data/official_splits/, Train=3121/Val=347/Test=816)
  - uses the fixed flip augmentation (src/dataset.py now swaps all 5 L/R body
    pairs alongside the hands)
  - trains all THREE topologies (single/dual/adaptive) fresh, augmentation OFF,
    matching the "86% model" regime (adaptive_none_torso in the old ablation
    table) rather than reading a stale augmented adaptive row from the CSV.

New rows appended to metrics_all.csv (condition suffixed "-noaug-fixed" to
avoid ambiguity with the old, invalid full-augmentation rows):
  topology | single-graph-noaug-fixed   | T=32 | ...
  topology | dual-graph-noaug-fixed     | T=32 | ...
  topology | adaptive-noaug-fixed       | T=32 | ...
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
T        = 32

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

print(f"Device: {DEVICE}  |  T={T}  |  max_epochs={N_EPOCHS}  |  patience={PATIENCE}")

# ── Data ──────────────────────────────────────────────────────────────────────
# augment_train=False: this ablation isolates topology, so augmentation must be
# off (matches the "86% model" no-augmentation regime) — see module docstring.
train_loader, val_loader, test_loader = load_sd_loaders(
    ROOT / "data" / "processed",
    batch_size=32, num_workers=0, augment_train=False, resample_T=T
)
xb, _ = next(iter(train_loader))
print(f"Input shape: {tuple(xb.shape)}")

# ── Adjacency matrices ────────────────────────────────────────────────────────
A_single = build_adjacency(n_joints=53, strategy="uniform")     # K=1
A_dual   = build_adjacency(n_joints=53, strategy="spatial")     # K=3

print(f"A_single: {A_single.shape}  A_dual: {A_dual.shape}")

# ── Training loop ─────────────────────────────────────────────────────────────
def run_experiment(label, A, adaptive=False):
    print(f"\n{'='*60}")
    print(f"  [{label}]  adaptive={adaptive}  T={T}  max_ep={N_EPOCHS}")
    print(f"{'='*60}")

    set_seed()
    model = STGCN(n_classes=262, A=A, adaptive=adaptive).to(DEVICE)
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

    # Test evaluation
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
    torch.save({"state_dict": best_state, "topology": label, "T": T, "adaptive": adaptive}, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    print(f"\n  [{label}] FINAL  val={best_val_acc:.4f}  test={test_acc:.4f}  macro-F1={f1:.4f}  ({total_min:.1f}m)")
    return dict(label=label, val_acc=best_val_acc, test_acc=test_acc, macro_f1=f1)

# ── Run all three topologies fresh, augmentation OFF ─────────────────────────
A_adaptive = build_adjacency(n_joints=53, strategy="spatial")   # same base graph as dual; learnable B added inside STGCN when adaptive=True
results = []
for label, A, adaptive in [
    ("single-graph-noaug-fixed",   A_single,   False),
    ("dual-graph-noaug-fixed",     A_dual,     False),
    ("adaptive-noaug-fixed",       A_adaptive, True),
]:
    r = run_experiment(label, A, adaptive)
    results.append(r)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"TOPOLOGY ABLATION AT EQUAL BUDGET, AUGMENTATION OFF (T={T}, max {N_EPOCHS}ep, patience={PATIENCE})")
print("="*70)
print(f"{'Condition':<28} {'Val Acc':>9}  {'Test Acc':>9}  {'Macro-F1':>9}")
print(f"{'-'*65}")
for r in results:
    print(f"  {r['label']:<26} {r['val_acc']:>9.4f}  {r['test_acc']:>9.4f}  {r['macro_f1']:>9.4f}")
print("="*70)

# ── Update metrics_all.csv ────────────────────────────────────────────────────
csv_path = ROOT / "results" / "metrics_all.csv"
df = pd.read_csv(csv_path)

new_rows = pd.DataFrame([
    dict(experiment='topology', condition=r['label'], T=T,
         val_acc=round(r['val_acc'], 4),
         test_acc=round(r['test_acc'], 4),
         macro_f1=round(r['macro_f1'], 4))
    for r in results
])

# Only removes rows from a PRIOR run of this same corrected script (condition
# ends in "-noaug-fixed"); the old, invalid full-augmentation topology rows
# (condition in {single-graph, dual-graph}, T=32) are left in place as a
# documented-invalid record, not silently deleted.
df = df[~((df['experiment']=='topology') & (df['T']==T) & (df['condition'].str.endswith('-noaug-fixed')))].copy()
df = pd.concat([df, new_rows], ignore_index=True)
df = df.sort_values(['experiment', 'T', 'condition']).reset_index(drop=True)
df.to_csv(csv_path, index=False)
print(f"\nUpdated: {csv_path}")
print(df[df['experiment'].isin(['topology','temporal'])].to_string(index=False))
