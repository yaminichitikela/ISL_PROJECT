"""
Inflation experiment — apples-to-apples leakage quantification.

Same CNN1D as NB04 (Adam lr=1e-3, batch=32, epochs=40, patience=8, clip=1.0).
Only the training/test split varies:
  - Clean   : our deduplicated SD split (data/processed/, 0 cross-split overlap)
  - Leaked  : raw HuggingFace parquet (30 % of test videos also in train)

CLEAN_BASELINE (0.865) is the locked NB04 SEED=42 result.

Outputs written to results/metrics_all.csv:
  leakage | 1D-CNN-clean           | NB04 reference (hardcoded)
  leakage | 1D-CNN-leaked          | trained on HF train, eval on HF test
  leakage | 1D-CNN-leaked-on-clean | same model, eval on clean SD test
"""
import os, sys, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

ROOT     = Path("/Users/yamini/Desktop/projects/ISL PROJECT")
PROC_DIR = ROOT / "data" / "processed"
KP_DIR   = ROOT / "data" / "raw_keypoints"
PAR_DIR  = ROOT / "include_dataset" / "data"
RES_DIR  = ROOT / "results"

SEED           = 42
CLEAN_BASELINE = 0.865   # locked NB04 SEED=42 result
L_SH, R_SH    = 45, 46  # left/right shoulder joint indices
FEAT_DIM       = 53 * 3  # 159 flattened features per frame
N_CLASSES      = 262
T_FIXED        = 64

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)

set_seed()

DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")

# ── Label encoder ─────────────────────────────────────────────────────────────
import pickle
with open(PROC_DIR / "label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
valid_labels = set(le.classes_)

# ── Preprocessing helpers (identical to NB02) ─────────────────────────────────
def fix_gaps(kps):
    filled = kps.copy().astype(np.float32)
    T = kps.shape[0]
    for j in range(53):
        col = kps[:, j, :]
        det = ~np.all(col == 0, axis=1)
        if not det.any():
            continue
        vf = np.where(det)[0]
        for c in range(3):
            filled[:, j, c] = np.interp(np.arange(T, dtype=np.float64),
                                         vf.astype(np.float64), col[vf, c])
    return filled

def resample(kps, t_tgt=64):
    T = kps.shape[0]
    if T == t_tgt:
        return kps.astype(np.float32)
    src = np.arange(T, dtype=np.float32)
    tgt = np.linspace(0, T - 1, t_tgt, dtype=np.float32)
    out = np.zeros((t_tgt, 53, 3), dtype=np.float32)
    for j in range(53):
        for c in range(3):
            out[:, j, c] = np.interp(tgt, src, kps[:, j, c])
    return out

def normalize(kps):
    result = kps.copy()
    T = kps.shape[0]
    ls_all, rs_all = kps[:, L_SH, :], kps[:, R_SH, :]
    ok = ~np.all(ls_all == 0, axis=1)
    if ok.any():
        lsm, rsm = ls_all[ok].mean(0), rs_all[ok].mean(0)
        wm = np.linalg.norm(lsm[:2] - rsm[:2]) + 1e-6
    else:
        lsm = rsm = np.zeros(3); wm = 1.0
    for t in range(T):
        ls, rs = kps[t, L_SH, :], kps[t, R_SH, :]
        if np.all(ls == 0) and np.all(rs == 0):
            center, width = (lsm + rsm) / 2, wm
        else:
            center = (ls + rs) / 2
            width  = np.linalg.norm(ls[:2] - rs[:2]) + 1e-6
        result[t] = (kps[t] - center) / width
    return result

def vp_to_npy(vp):
    return vp.replace('.MOV', '').replace('.mp4', '').replace('.MP4', '').replace('/', '__') + '.npy'

def build_split(df_pq, name):
    rows = df_pq[df_pq['label'].isin(valid_labels)].reset_index(drop=True)
    Xl, yl, skip = [], [], 0
    t0 = time.time()
    for i, row in rows.iterrows():
        p = KP_DIR / vp_to_npy(row['video_path'])
        if not p.exists():
            skip += 1; continue
        kps = np.load(str(p))
        if kps.shape[0] < 2:
            skip += 1; continue
        kps = fix_gaps(kps)
        kps = resample(kps, T_FIXED)
        kps = normalize(kps)
        Xl.append(kps)
        yl.append(le.transform([row['label']])[0])
        if (i + 1) % 500 == 0:
            print(f"  [{name}] {i+1}/{len(rows)}  elapsed {time.time()-t0:.0f}s", flush=True)
    X = np.stack(Xl).astype(np.float32).reshape(len(Xl), T_FIXED, FEAT_DIM)
    y = np.array(yl, dtype=np.int64)
    print(f"  [{name}] done: {len(X):,} samples, {skip} skipped ({time.time()-t0:.0f}s)")
    return X, y

# ── Model (identical to NB04 CNN1D) ──────────────────────────────────────────
class CNN1D(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, n_classes=N_CLASSES, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feat_dim, 64,  kernel_size=3, padding=1), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,  128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(256, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.fc(self.drop(self.net(x).squeeze(-1)))

class ArrayDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def train_cnn(X_tr, y_tr, X_va, y_va, label, epochs=40, patience=8, lr=1e-3, batch=32):
    set_seed()
    model  = CNN1D().to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    crit   = nn.CrossEntropyLoss()
    tr_dl  = DataLoader(ArrayDS(X_tr, y_tr), batch_size=batch, shuffle=True,  num_workers=0)
    va_dl  = DataLoader(ArrayDS(X_va, y_va), batch_size=batch, shuffle=False, num_workers=0)
    best_vl, best_state, no_imp = float('inf'), None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); rl = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rl += loss.item() * len(yb)
        tr_loss = rl / len(tr_dl.dataset)
        model.eval(); vl = 0.0; preds, labs = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                vl += crit(out, yb).item() * len(yb)
                preds.extend(out.argmax(1).cpu().numpy())
                labs.extend(yb.cpu().numpy())
        vl /= len(va_dl.dataset)
        sched.step(vl)
        va_acc = accuracy_score(labs, preds)
        print(f"  [{label}] ep {ep:2d}/{epochs}  tr={tr_loss:.4f}  vl={vl:.4f}  "
              f"val_acc={va_acc:.4f}  ({(time.time()-t0)/60:.1f}m)", flush=True)
        if vl < best_vl:
            best_vl, best_state, no_imp = vl, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  [{label}] Early stop ep {ep}"); break
    model.load_state_dict(best_state)
    return model

def eval_cnn(model, X, y):
    dl = DataLoader(ArrayDS(X, y), batch_size=64, shuffle=False, num_workers=0)
    model.eval(); preds, labs = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
            labs.extend(yb.numpy())
    acc = accuracy_score(labs, preds)
    f1  = f1_score(labs, preds, average='macro', labels=list(range(N_CLASSES)), zero_division=0)
    return acc, f1

# ── Step 1: Analytical overlap ────────────────────────────────────────────────
print("Loading parquet files ...")
df_tr = pq.read_table(PAR_DIR / "train-00000-of-00001.parquet").to_pandas()
df_va = pq.read_table(PAR_DIR / "val-00000-of-00001.parquet").to_pandas()
df_te = pq.read_table(PAR_DIR / "test-00000-of-00001.parquet").to_pandas()

train_ids = set(df_tr['video_path'].astype(str))
test_ids  = set(df_te['video_path'].astype(str))
overlap   = train_ids & test_ids
leak_pct  = 100 * len(overlap) / max(len(test_ids), 1)
print(f"Analytical: train ∩ test = {len(overlap)} unique videos ({leak_pct:.1f}% of unique test)")

# ── Step 2: Build leaked splits ───────────────────────────────────────────────
print("\nBuilding leaked splits from raw parquet ...")
X_lk_tr, y_lk_tr = build_split(df_tr, "leaked-train")
X_lk_va, y_lk_va = build_split(df_va, "leaked-val")
X_lk_te, y_lk_te = build_split(df_te, "leaked-test")
print(f"\nLeaked: train={len(y_lk_tr):,}  val={len(y_lk_va):,}  test={len(y_lk_te):,}")

# Load clean SD test (already processed)
X_cl_te = np.load(PROC_DIR / "X_sd_test.npy").reshape(-1, T_FIXED, FEAT_DIM)
y_cl_te = np.load(PROC_DIR / "y_sd_test.npy")

# ── Step 3: Train CNN1D on leaked split ───────────────────────────────────────
print("\n" + "="*66)
print("Training CNN1D on LEAKED train split ...")
print("  (same Adam lr=1e-3, batch=32, epochs=40, patience=8 as NB04)")
print("="*66)
model_lk = train_cnn(X_lk_tr, y_lk_tr, X_lk_va, y_lk_va, "leaked")

lk_acc,       lk_f1       = eval_cnn(model_lk, X_lk_te, y_lk_te)
lk_on_cl_acc, lk_on_cl_f1 = eval_cnn(model_lk, X_cl_te, y_cl_te)

inflation_lk = lk_acc       - CLEAN_BASELINE
inflation_cl = lk_on_cl_acc - CLEAN_BASELINE

# ── Step 4: Report ────────────────────────────────────────────────────────────
print("\n" + "="*66)
print("LEAKAGE QUANTIFICATION — apples-to-apples (same CNN1D as NB04)")
print("="*66)
print(f"  Analytical:  {leak_pct:.1f}% of unique test videos present in raw train\n")
print(f"  {'Condition':<30} {'Train':>7} {'Test':>6} {'Acc':>8} {'Macro-F1':>10}")
print(f"  {'-'*64}")
print(f"  {'Clean SD (NB04 locked baseline)':<30} {'2,462':>7} {'858':>6} {CLEAN_BASELINE:>8.4f} {'—':>10}")
print(f"  {'Leaked HF (train on HF, test on HF)':<30} {len(y_lk_tr):>7,} {len(y_lk_te):>6,} {lk_acc:>8.4f} {lk_f1:>10.4f}")
print(f"  {'Leaked model → clean SD test':<30} {'':>7} {'858':>6} {lk_on_cl_acc:>8.4f} {lk_on_cl_f1:>10.4f}")
print()
print(f"  Inflation (leaked HF test − clean):       {inflation_lk:+.4f}  ({inflation_lk*100:+.1f} pp)")
print(f"  Train contamination only (on clean test): {inflation_cl:+.4f}  ({inflation_cl*100:+.1f} pp)")
print(f"  Test-contamination-only effect:           {(lk_acc-lk_on_cl_acc)*100:+.1f} pp")
print("="*66)

# ── Step 5: Update metrics_all.csv ───────────────────────────────────────────
csv_path = RES_DIR / "metrics_all.csv"
df_all = pd.read_csv(csv_path)
df_all = df_all[df_all['experiment'] != 'leakage'].copy()
new_rows = pd.DataFrame([
    dict(experiment='leakage', condition='1D-CNN-clean',
         T=64, val_acc=None, test_acc=CLEAN_BASELINE, macro_f1=None),
    dict(experiment='leakage', condition='1D-CNN-leaked',
         T=64, val_acc=None, test_acc=round(lk_acc, 4),       macro_f1=round(lk_f1, 4)),
    dict(experiment='leakage', condition='1D-CNN-leaked-on-clean',
         T=64, val_acc=None, test_acc=round(lk_on_cl_acc, 4), macro_f1=round(lk_on_cl_f1, 4)),
])
df_all = pd.concat([new_rows, df_all], ignore_index=True)
df_all.to_csv(csv_path, index=False)
print(f"\nUpdated: {csv_path}")
print(df_all[df_all['experiment'] == 'leakage'].to_string(index=False))
