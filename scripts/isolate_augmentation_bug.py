"""
Isolate which augmentation component(s) still cause the ~13pp accuracy drop
after the flip-bug fix (full aug: 76.72% test vs no-aug: 89.95% test, T=32,
adaptive, torso, official splits).

Since SkeletonDataset applies noise -> temporal-crop -> flip -> rotation as a
single fixed sequence whenever augment=True (regardless of noise_std/
crop_ratio values), isolating one component means building loaders directly
rather than going through train.py's fixed "none/spatial/temporal/full"
presets. Uses a shorter budget (30 epochs, patience=6) for faster turnaround
across 4 diagnostic runs; this is for isolating the cause, not a paper number.
"""
import os, sys, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path("/Users/yamini/Desktop/projects/ISL PROJECT")
sys.path.insert(0, str(ROOT / "src"))

from graph    import build_adjacency
from model    import STGCN
from dataset  import SkeletonDataset

PROC_DIR = ROOT / "data" / "processed"
N_EPOCHS = 30
PATIENCE = 6
T = 32

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


class IsolatedAugDataset(SkeletonDataset):
    """Same as SkeletonDataset, but lets each of the 4 augmentation steps be
    toggled independently, instead of all-or-nothing via `augment`."""
    def __init__(self, X, y, use_noise=False, use_crop=False, use_flip=False, use_rotation=False, **kw):
        super().__init__(X, y, augment=False, **kw)
        self.use_noise    = use_noise
        self.use_crop     = use_crop
        self.use_flip     = use_flip
        self.use_rotation = use_rotation

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.use_noise:
            x = self._add_joint_noise(x)
        if self.use_crop:
            x = self._random_temporal_crop(x)
        if self.use_flip:
            x = self._random_horizontal_flip(x)
        if self.use_rotation:
            x = self._random_rotation(x)
        return x, self.y[idx]


def load_isolated(split, **kw):
    X = np.load(PROC_DIR / f"X_sd_{split}.npy")
    y = np.load(PROC_DIR / f"y_sd_{split}.npy")
    return IsolatedAugDataset(X, y, resample_T=T, **kw)


def run_config(label, **aug_flags):
    print(f"\n{'='*60}\n  [{label}]  flags={aug_flags}\n{'='*60}")
    train_ds = load_isolated("train", **aug_flags)
    val_ds   = load_isolated("val")   # no aug on val/test ever
    test_ds  = load_isolated("test")

    kwl = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  **kwl)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, **kwl)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, **kwl)

    set_seed(42)
    A = build_adjacency(n_joints=53, strategy="spatial")
    model = STGCN(n_classes=262, A=A, adaptive=True).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state, no_imp = 0.0, None, 0
    t0 = time.time()
    for ep in range(1, N_EPOCHS + 1):
        model.train(); correct = total = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x); loss = criterion(out, y)
            loss.backward(); optimizer.step()
            correct += (out.detach().argmax(1) == y).sum().item(); total += len(y)
        tr_acc = correct / total

        model.eval(); vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                vc += (model(x).argmax(1) == y).sum().item(); vt += len(y)
        vl_acc = vc / vt
        scheduler.step()
        print(f"  [{label}] ep {ep:2d}/{N_EPOCHS}  tr={tr_acc:.4f}  vl={vl_acc:.4f}  ({(time.time()-t0)/60:.1f}m)", flush=True)

        if vl_acc > best_val:
            best_val, best_state, no_imp = vl_acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"  [{label}] Early stop at epoch {ep}"); break

    model.load_state_dict(best_state); model.eval()
    correct = total = 0; yt, yp = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(1).cpu()
            correct += (preds == y).sum().item(); total += len(y)
            yt.extend(y.numpy()); yp.extend(preds.numpy())
    test_acc = correct / total
    f1 = f1_score(yt, yp, average="macro", zero_division=0)
    print(f"\n  [{label}] FINAL  val={best_val:.4f}  test={test_acc:.4f}  F1={f1:.4f}  ({(time.time()-t0)/60:.1f}m)")
    return dict(label=label, val_acc=best_val, test_acc=test_acc, macro_f1=f1)


configs = [
    ("none (baseline)",     dict()),
    ("noise_only",          dict(use_noise=True)),
    ("crop_only",           dict(use_crop=True)),
    ("flip_only",           dict(use_flip=True)),
    ("rotation_only",       dict(use_rotation=True)),
    ("full (all 4)",        dict(use_noise=True, use_crop=True, use_flip=True, use_rotation=True)),
]

results = []
for label, flags in configs:
    r = run_config(label, **flags)
    results.append(r)

print("\n" + "=" * 70)
print(f"AUGMENTATION COMPONENT ISOLATION (30 epochs, patience=6, T=32, adaptive, torso)")
print("=" * 70)
for r in results:
    print(f"  {r['label']:<20} val={r['val_acc']:.4f}  test={r['test_acc']:.4f}  F1={r['macro_f1']:.4f}")
