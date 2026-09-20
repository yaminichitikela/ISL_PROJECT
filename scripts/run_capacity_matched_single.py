"""
Capacity-matched single-partition control (2026-09-17).

The topology ablation compares single (K=1, uniform, 2,063,173 params) vs.
fixed spatial (K=3, 2,419,527 params) vs. adaptive (K=3+B, 2,495,370 params)
at equal training budget but UNEQUAL parameter count. This script asks
whether widening the single-partition model to approximately match the
fixed-spatial model's parameter count closes the gap — i.e., whether part
of the topology-richness effect is actually a model-capacity effect.

Channel widths (70, 136, 280) were found by searching nearby integer widths
around the original (64, 128, 256) pattern for the closest match to
2,419,527 params under K=1 (uniform) adjacency; the result, 2,420,451
params, is within 0.04% of the fixed-spatial model.

Same protocol as scripts/run_topology_equal_budget.py: T=32, torso
normalization (data/processed/X_sd_*.npy), no augmentation, SGD+Nesterov
lr=0.01, cosine annealing, weight_decay=1e-4, batch=32, max 80 epochs,
patience=10, early stopping and checkpoint selection on validation
accuracy only. Run across seeds 42, 0, 1.

New rows appended to metrics_all.csv: experiment='topology',
condition='single-widened-matched'[_seed0/_seed1], T=32.
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

N_EPOCHS = 80
PATIENCE = 10
T        = 32
CHANNELS = (70, 136, 280)
SEEDS    = [42, 0, 1]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}  |  T={T}  |  channels={CHANNELS}  |  max_epochs={N_EPOCHS}  |  patience={PATIENCE}")

A_single = build_adjacency(n_joints=53, strategy="uniform")   # K=1

probe = STGCN(n_classes=262, A=A_single, adaptive=False, channels=CHANNELS)
N_PARAMS = sum(p.numel() for p in probe.parameters() if p.requires_grad)
print(f"Widened single-partition params: {N_PARAMS:,}  (fixed-spatial reference: 2,419,527)")
del probe


def run_seed(seed):
    label = "single-widened-matched" if seed == 42 else f"single-widened-matched_seed{seed}"
    print(f"\n{'='*60}\n  [{label}]  seed={seed}  channels={CHANNELS}  T={T}\n{'='*60}")

    set_seed(seed)
    train_loader, val_loader, test_loader = load_sd_loaders(
        ROOT / "data" / "processed",
        batch_size=32, num_workers=0, augment_train=False, resample_T=T
    )

    set_seed(seed)
    model = STGCN(n_classes=262, A=A_single, adaptive=False, channels=CHANNELS).to(DEVICE)
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
    torch.save({"state_dict": best_state, "topology": label, "T": T,
                "adaptive": False, "channels": CHANNELS, "n_params": n_params,
                "seed": seed}, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")
    print(f"\n  [{label}] FINAL  seed={seed}  val={best_val_acc:.4f}  test={test_acc:.4f}  macro-F1={f1:.4f}  params={n_params:,}  ({total_min:.1f}m)")

    return dict(label=label, seed=seed, val_acc=best_val_acc, test_acc=test_acc,
                macro_f1=f1, n_params=n_params)


results = [run_seed(s) for s in SEEDS]

test_vals = np.array([r["test_acc"] for r in results])
val_vals  = np.array([r["val_acc"]  for r in results])
f1_vals   = np.array([r["macro_f1"] for r in results])

print("\n" + "="*70)
print(f"CAPACITY-MATCHED SINGLE-PARTITION CONTROL (T={T}, channels={CHANNELS}, params={N_PARAMS:,})")
print("="*70)
for r in results:
    print(f"  seed={r['seed']:<3} val={r['val_acc']:.4f}  test={r['test_acc']:.4f}  F1={r['macro_f1']:.4f}")
print(f"\n  MEAN  val={val_vals.mean():.4f}  test={test_vals.mean():.4f}  F1={f1_vals.mean():.4f}")
print(f"  STD   val={val_vals.std(ddof=1):.4f}  test={test_vals.std(ddof=1):.4f}  F1={f1_vals.std(ddof=1):.4f}")
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
] + [
    dict(experiment='topology', condition='single-widened-matched_mean3seed', T=T,
         val_acc=round(val_vals.mean(), 4),
         test_acc=round(test_vals.mean(), 4),
         macro_f1=round(f1_vals.mean(), 4))
])

df = df[~((df['experiment']=='topology') & (df['T']==T) &
          (df['condition'].str.startswith('single-widened-matched')))].copy()
df = pd.concat([df, new_rows], ignore_index=True)
df = df.sort_values(['experiment', 'T', 'condition']).reset_index(drop=True)
df.to_csv(csv_path, index=False)
print(f"\nUpdated: {csv_path}")
print(f"Exact param count: {N_PARAMS:,} (vs. fixed-spatial 2,419,527, diff {N_PARAMS-2419527:+,})")
