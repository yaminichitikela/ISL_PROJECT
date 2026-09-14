"""
Multi-seed reruns for the two headline numbers (Naman review item 7):
  1. Best ST-GCN: adaptive topology, T=96, torso normalisation, augmentation=none
     (SEED=42 result: val=0.9568, test=0.9706, F1=0.9695)
  2. Final topology ranking: single/dual/adaptive, T=32, torso, augmentation=none
     (SEED=42 results: single 79.9%, dual 89.6%, adaptive 90.0% test)

Runs each config with 2 additional seeds (0, 1) on top of the existing SEED=42
result, so all three headline claims can be reported as mean +/- std over 3
seeds rather than a single (possibly MPS-noisy, per the T=64 fluke and the
topology-checkpoint-rerun drift already observed this session) run.

Saves checkpoints as checkpoints/{tag}_seed{N}.pt and appends CSV rows with
an explicit "_seed{N}" suffix, upserted by EXACT (experiment,condition,T)
match (not a broad suffix-delete) to avoid the CSV-clobbering bug hit earlier
this session with the T=64 diagnostic rerun.
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
CKPT_DIR = ROOT / "checkpoints"
CSV_PATH = ROOT / "results" / "metrics_all.csv"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}  |  max_epochs={N_EPOCHS}  |  patience={PATIENCE}")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)


def upsert_csv(experiment, condition, T, val_acc, test_acc, macro_f1):
    df = pd.read_csv(CSV_PATH) if CSV_PATH.exists() else pd.DataFrame(
        columns=["experiment", "condition", "T", "val_acc", "test_acc", "macro_f1"])
    mask = (df["experiment"] == experiment) & (df["condition"] == condition) & (df["T"] == T)
    row = dict(experiment=experiment, condition=condition, T=T,
               val_acc=round(val_acc, 4), test_acc=round(test_acc, 4), macro_f1=round(macro_f1, 4))
    if mask.any():
        for k, v in row.items():
            df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)


def run_experiment(tag, condition, experiment, T, strategy, adaptive, seed):
    label = f"{tag}_seed{seed}"
    print(f"\n{'='*70}")
    print(f"  [{label}]  T={T}  strategy={strategy}  adaptive={adaptive}  seed={seed}")
    print(f"{'='*70}")

    train_loader, val_loader, test_loader = load_sd_loaders(
        ROOT / "data" / "processed",
        batch_size=32, num_workers=0, augment_train=False, resample_T=T
    )

    set_seed(seed)
    A = build_adjacency(n_joints=53, strategy=strategy)
    model = STGCN(n_classes=262, A=A, adaptive=adaptive).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {n_params:,}")

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

    ckpt_path = CKPT_DIR / f"{tag}_seed{seed}.pt"
    torch.save({"state_dict": best_state, "tag": tag, "T": T, "seed": seed,
                "adaptive": adaptive, "strategy": strategy}, ckpt_path)
    print(f"  Checkpoint saved: {ckpt_path}")

    upsert_csv(experiment, f"{condition}_seed{seed}", T, best_val_acc, test_acc, f1)
    print(f"  CSV updated: {experiment} / {condition}_seed{seed} / T={T}")

    print(f"\n  [{label}] FINAL  val={best_val_acc:.4f}  test={test_acc:.4f}  macro-F1={f1:.4f}  ({total_min:.1f}m)")
    return dict(label=label, val_acc=best_val_acc, test_acc=test_acc, macro_f1=f1)


# ── Queue: 2 seeds x (1 best-ST-GCN config + 3 topology configs) = 8 runs ────
QUEUE = []
for seed in (0, 1):
    QUEUE.append(dict(tag="adaptive_T96_none_torso", condition="adaptive-T96-noaug-fixed",
                       experiment="temporal", T=96, strategy="spatial", adaptive=True, seed=seed))
for seed in (0, 1):
    QUEUE.append(dict(tag="single_graph_noaug_fixed_T32", condition="single-graph-noaug-fixed",
                       experiment="topology", T=32, strategy="uniform", adaptive=False, seed=seed))
    QUEUE.append(dict(tag="dual_graph_noaug_fixed_T32", condition="dual-graph-noaug-fixed",
                       experiment="topology", T=32, strategy="spatial", adaptive=False, seed=seed))
    QUEUE.append(dict(tag="adaptive_noaug_fixed_T32", condition="adaptive-noaug-fixed",
                       experiment="topology", T=32, strategy="spatial", adaptive=True, seed=seed))

print(f"Queue: {len(QUEUE)} runs")
for q in QUEUE:
    print(f"  {q['condition']}_seed{q['seed']}  (T={q['T']})")

results = []
for i, cfg in enumerate(QUEUE, 1):
    print(f"\n\n########## RUN {i}/{len(QUEUE)} ##########")
    r = run_experiment(cfg["tag"], cfg["condition"], cfg["experiment"],
                        cfg["T"], cfg["strategy"], cfg["adaptive"], cfg["seed"])
    results.append({**cfg, **r})

print("\n" + "=" * 80)
print("ALL MULTI-SEED RUNS COMPLETE")
print("=" * 80)
for r in results:
    print(f"  {r['condition']}_seed{r['seed']} (T={r['T']}): val={r['val_acc']:.4f} test={r['test_acc']:.4f} F1={r['macro_f1']:.4f}")
