"""
src/train.py — Configurable ST-GCN training for ISL INCLUDE (Milestone 3).

CLI usage:
    python src/train.py --topology adaptive --T 32 --augmentation full --normalisation torso
    python src/train.py --topology hands42  --T 32 --augmentation full --normalisation torso
    python src/train.py --topology latefusion --T 32 --augmentation full --normalisation torso

Saves:
    checkpoints/{topology}_T{T}_{augmentation}_{normalisation}.pt
    Appends one row to results/metrics_all.csv

Normalisation file convention (build in NB09 for raw/bonelength):
    data/processed/X_sd_{split}.npy          — torso-centred (default)
    data/processed/X_raw_sd_{split}.npy      — no normalisation
    data/processed/X_bonelen_sd_{split}.npy  — bone-length normalised
"""
from __future__ import annotations
import argparse, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import SkeletonDataset, load_sd_loaders
from graph   import build_adjacency, _EDGES
from model   import STGCN, STGCNBlock

PROC_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"
RES_CSV  = ROOT / "results" / "metrics_all.csv"
N_CLASSES = 262
SEED      = 42


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


def get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Hands-only (42-joint) adjacency ──────────────────────────────────────────

def build_hands42_adjacency(strategy: str = "spatial") -> np.ndarray:
    """
    Adjacency for 42-joint hand-only graph (joints 0-41).
    Upper-body joints 42-52 are dropped; L_wrist(0)→R_wrist(21) is added as
    the inter-hand bridge replacing the shoulder chain.
    """
    import networkx as nx

    hand_edges = [(u, v) for u, v in _EDGES if u < 42 and v < 42]
    hand_edges.append((0, 21))   # direct L_wrist → R_wrist inter-hand bridge
    n = 42

    if strategy == "uniform":
        A = np.zeros((1, n, n), dtype=np.float32)
        for u, v in hand_edges:
            A[0, u, v] = A[0, v, u] = 1.0
        A[0] += np.eye(n, dtype=np.float32)
        d = A[0].sum(1, keepdims=True); d[d == 0] = 1
        A[0] /= d
        return A

    # Spatial-configuration (K=3)
    G = nx.Graph(); G.add_nodes_from(range(n))
    for u, v in hand_edges:
        G.add_edge(u, v)
    root = 0   # L_wrist as root
    dist = dict(nx.shortest_path_length(G, source=root))
    A3 = np.zeros((3, n, n), dtype=np.float32)
    for u in range(n):
        for v in range(n):
            if not (u == v or G.has_edge(u, v)):
                continue
            du, dv = dist.get(u, 99), dist.get(v, 99)
            if u == v or du == dv:
                A3[0, u, v] = 1.0   # self / equidistant
            elif dv < du:
                A3[1, u, v] = 1.0   # centripetal
            else:
                A3[2, u, v] = 1.0   # centrifugal
    for k in range(3):
        d = A3[k].sum(1, keepdims=True); d[d == 0] = 1
        A3[k] /= d
    return A3


def build_hand21_adjacency(strategy: str = "spatial") -> np.ndarray:
    """
    Adjacency for a single 21-joint hand (used in late-fusion branches).
    Works for either hand — both have identical topology, joints re-indexed 0-20.
    """
    import networkx as nx

    # Left-hand edges (joints 0-20); right-hand edges are identical re-indexed
    single_edges = [(u, v) for u, v in _EDGES if u < 21 and v < 21]
    n = 21

    if strategy == "uniform":
        A = np.zeros((1, n, n), dtype=np.float32)
        for u, v in single_edges:
            A[0, u, v] = A[0, v, u] = 1.0
        A[0] += np.eye(n, dtype=np.float32)
        d = A[0].sum(1, keepdims=True); d[d == 0] = 1
        A[0] /= d
        return A

    G = nx.Graph(); G.add_nodes_from(range(n))
    for u, v in single_edges:
        G.add_edge(u, v)
    root = 0
    dist = dict(nx.shortest_path_length(G, source=root))
    A3 = np.zeros((3, n, n), dtype=np.float32)
    for u in range(n):
        for v in range(n):
            if not (u == v or G.has_edge(u, v)):
                continue
            du, dv = dist.get(u, 99), dist.get(v, 99)
            if u == v or du == dv:
                A3[0, u, v] = 1.0
            elif dv < du:
                A3[1, u, v] = 1.0
            else:
                A3[2, u, v] = 1.0
    for k in range(3):
        d = A3[k].sum(1, keepdims=True); d[d == 0] = 1
        A3[k] /= d
    return A3


# ── Late-fusion model ─────────────────────────────────────────────────────────

class _STGCNBody(nn.Module):
    """STGCN backbone without the final FC — outputs 256-dim global-pool features."""

    def __init__(self, n_joints: int, A: np.ndarray, adaptive: bool = False):
        super().__init__()
        A_t = torch.from_numpy(A.astype(np.float32))
        self.data_bn = nn.BatchNorm1d(3 * n_joints)
        cfg = [(64,1),(64,1),(64,1),(64,1),(128,2),(128,1),(128,1),(256,2),(256,1)]
        blocks, c_in = [], 3
        for i, (c_out, stride) in enumerate(cfg):
            blocks.append(STGCNBlock(
                c_in, c_out, A_t,
                stride=stride, dropout=0.0,
                residual=(i > 0), adaptive=adaptive,
            ))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.drop   = nn.Dropout(0.5)
        self.n_joints = n_joints

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            x = block(x)
        return self.drop(x.mean(dim=[2, 3]))   # (N, 256)


class LateFusionSTGCN(nn.Module):
    """
    Two independent STGCN bodies (one per hand), late-fused by concatenation.

    Input : (N, 3, T, 42) — first 21 joints = L-hand, next 21 = R-hand
    Output: (N, n_classes) logits
    """

    def __init__(self, n_classes: int, A_hand: np.ndarray, adaptive: bool = False):
        super().__init__()
        n = A_hand.shape[-1]   # 21
        self.left  = _STGCNBody(n, A_hand, adaptive=adaptive)
        self.right = _STGCNBody(n, A_hand, adaptive=adaptive)
        self.fc    = nn.Linear(512, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xl = x[:, :, :, :21]    # (N, 3, T, 21)
        xr = x[:, :, :, 21:42]  # (N, 3, T, 21)
        return self.fc(torch.cat([self.left(xl), self.right(xr)], dim=1))


# ── Data loading ──────────────────────────────────────────────────────────────

_AUG_PARAMS = {
    "none":     dict(augment=False, noise_std=0.01, crop_ratio=0.1),
    "spatial":  dict(augment=True,  noise_std=0.01, crop_ratio=0.0),
    "temporal": dict(augment=True,  noise_std=0.0,  crop_ratio=0.1),
    "full":     dict(augment=True,  noise_std=0.01, crop_ratio=0.1),
}

_NORM_PREFIX = {
    "torso":     "X_sd",
    "raw":       "X_raw_sd",
    "bonelength":"X_bonelen_sd",
}


def load_splits(
    normalisation: str = "torso",
    T: int = 32,
    augmentation: str = "full",
    batch_size: int = 32,
    joint_slice: slice | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    joint_slice : if set, subsets the joint dimension — used for hands42 ([:42])
                  and late-fusion ([:42] too — the model handles the split internally).
    """
    prefix = _NORM_PREFIX[normalisation]
    aug    = _AUG_PARAMS[augmentation]

    def load(split: str) -> tuple[np.ndarray, np.ndarray]:
        X = np.load(PROC_DIR / f"{prefix}_{split}.npy")   # (N, T_stored, 53, 3)
        y = np.load(PROC_DIR / f"y_sd_{split}.npy")
        if joint_slice is not None:
            X = X[:, :, joint_slice, :]
        return X, y

    X_tr, y_tr = load("train")
    X_va, y_va = load("val")
    X_te, y_te = load("test")

    train_ds = SkeletonDataset(X_tr, y_tr, resample_T=T,
                               **aug)
    val_ds   = SkeletonDataset(X_va, y_va, resample_T=T,
                               augment=False)
    test_ds  = SkeletonDataset(X_te, y_te, resample_T=T,
                               augment=False)

    kw = dict(num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
    )


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(topology: str, device: torch.device) -> nn.Module:
    if topology == "single":
        A = build_adjacency(n_joints=53, strategy="uniform")
        return STGCN(n_classes=N_CLASSES, A=A, adaptive=False).to(device)
    if topology == "dual":
        A = build_adjacency(n_joints=53, strategy="spatial")
        return STGCN(n_classes=N_CLASSES, A=A, adaptive=False).to(device)
    if topology == "adaptive":
        A = build_adjacency(n_joints=53, strategy="spatial")
        return STGCN(n_classes=N_CLASSES, A=A, adaptive=True).to(device)
    if topology == "hands42":
        A = build_hands42_adjacency(strategy="spatial")
        return STGCN(n_classes=N_CLASSES, A=A, adaptive=False).to(device)
    if topology == "latefusion":
        A = build_hand21_adjacency(strategy="spatial")
        return LateFusionSTGCN(n_classes=N_CLASSES, A_hand=A).to(device)
    raise ValueError(f"Unknown topology: {topology!r}")


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(
    topology:      str = "adaptive",
    T:             int = 32,
    augmentation:  str = "full",
    normalisation: str = "torso",
    epochs:        int = 80,
    patience:      int = 10,
    lr:            float = 0.01,
    batch_size:    int = 32,
    seed:          int = SEED,
    verbose:       bool = True,
) -> dict:
    """
    Full training run. Returns result dict with val_acc, test_acc, macro_f1.
    Saves best checkpoint to checkpoints/.
    """
    set_seed(seed)
    device = get_device()
    CKPT_DIR.mkdir(exist_ok=True)

    joint_slice = slice(0, 42) if topology in ("hands42", "latefusion") else None
    train_loader, val_loader, test_loader = load_splits(
        normalisation=normalisation, T=T,
        augmentation=augmentation, batch_size=batch_size,
        joint_slice=joint_slice,
    )

    model     = build_model(topology, device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    if verbose:
        tag = f"{topology}_T{T}_{augmentation}_{normalisation}"
        print(f"\n{'='*60}")
        print(f"  {tag}  |  params={n_params:,}  |  device={device}")
        print(f"{'='*60}")

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0
    t0           = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.detach().argmax(1) == y).sum().item()
            total   += len(y)
        tr_acc = correct / total

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vc += (model(x).argmax(1) == y).sum().item()
                vt += len(y)
        vl_acc = vc / vt
        scheduler.step()

        elapsed = (time.time() - t0) / 60
        if verbose:
            print(f"  ep {ep:3d}/{epochs}  tr={tr_acc:.4f}  vl={vl_acc:.4f}  "
                  f"({elapsed:.1f}m)", flush=True)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop at epoch {ep}")
                break

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    correct = total = 0
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += len(y)
            y_true_all.extend(y.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())

    test_acc = correct / total
    macro_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    total_min = (time.time() - t0) / 60

    if verbose:
        print(f"\n  RESULT  val={best_val_acc:.4f}  test={test_acc:.4f}  "
              f"F1={macro_f1:.4f}  ({total_min:.1f}m)")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    tag       = f"{topology}_T{T}_{augmentation}_{normalisation}"
    ckpt_path = CKPT_DIR / f"{tag}.pt"
    torch.save({
        "topology":      topology,
        "T":             T,
        "augmentation":  augmentation,
        "normalisation": normalisation,
        "val_acc":       best_val_acc,
        "test_acc":      test_acc,
        "macro_f1":      macro_f1,
        "state_dict":    best_state,
    }, ckpt_path)
    if verbose:
        print(f"  Checkpoint saved: {ckpt_path}")

    # ── Append to metrics_all.csv ─────────────────────────────────────────────
    _update_csv(topology, augmentation, normalisation, T,
                best_val_acc, test_acc, macro_f1)

    return dict(
        topology=topology, T=T, augmentation=augmentation,
        normalisation=normalisation,
        val_acc=best_val_acc, test_acc=test_acc, macro_f1=macro_f1,
    )


def _update_csv(topology, augmentation, normalisation, T,
                val_acc, test_acc, macro_f1) -> None:
    """Upsert one row in metrics_all.csv keyed by (experiment, condition, T)."""
    condition  = f"{topology}_{augmentation}_{normalisation}"
    experiment = "ablation"

    df = pd.read_csv(RES_CSV) if RES_CSV.exists() else pd.DataFrame(
        columns=["experiment","condition","T","val_acc","test_acc","macro_f1"])

    mask = (df["experiment"] == experiment) & \
           (df["condition"]  == condition)  & \
           (df["T"]          == T)
    row = dict(experiment=experiment, condition=condition, T=T,
               val_acc=round(val_acc, 4), test_acc=round(test_acc, 4),
               macro_f1=round(macro_f1, 4))

    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(RES_CSV, index=False)


# ── CLI entry point ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ST-GCN for ISL INCLUDE")
    p.add_argument("--topology",      choices=["single","dual","adaptive","hands42","latefusion"],
                   default="adaptive")
    p.add_argument("--T",             type=int,   default=32)
    p.add_argument("--augmentation",  choices=["none","spatial","temporal","full"],
                   default="full")
    p.add_argument("--normalisation", choices=["torso","raw","bonelength"],
                   default="torso")
    p.add_argument("--epochs",        type=int,   default=80)
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--lr",            type=float, default=0.01)
    p.add_argument("--batch",         type=int,   default=32)
    p.add_argument("--seed",          type=int,   default=SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(
        topology=args.topology,
        T=args.T,
        augmentation=args.augmentation,
        normalisation=args.normalisation,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch,
        seed=args.seed,
    )
    print(f"\nDone: {result}")


if __name__ == "__main__":
    main()
