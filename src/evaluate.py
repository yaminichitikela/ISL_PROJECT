"""
src/evaluate.py — Load a saved checkpoint and run full evaluation.

Outputs:
    • Prints overall acc / macro-F1 / per-class summary to stdout
    • Writes / overwrites results/error_taxonomy.md
    • Optionally updates results/metrics_all.csv (--update-csv flag)

Usage:
    python src/evaluate.py --checkpoint checkpoints/adaptive_T32_full_torso.pt
    python src/evaluate.py --checkpoint checkpoints/adaptive_T32_full_torso.pt --update-csv
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import SkeletonDataset
from graph   import build_adjacency
from model   import STGCN
from train   import (build_hands42_adjacency, build_hand21_adjacency,
                     LateFusionSTGCN, N_CLASSES, SEED, get_device, set_seed,
                     _NORM_PREFIX)

PROC_DIR = ROOT / "data" / "processed"
CKPT_DIR = ROOT / "checkpoints"
RES_DIR  = ROOT / "results"
RES_CSV  = RES_DIR / "metrics_all.csv"
TAX_MD   = RES_DIR / "error_taxonomy.md"


# ── Model reconstruction ──────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt: dict, device: torch.device) -> torch.nn.Module:
    topology = ckpt["topology"]
    if topology == "single":
        A     = build_adjacency(n_joints=53, strategy="uniform")
        model = STGCN(n_classes=N_CLASSES, A=A, adaptive=False)
    elif topology == "dual":
        A     = build_adjacency(n_joints=53, strategy="spatial")
        model = STGCN(n_classes=N_CLASSES, A=A, adaptive=False)
    elif topology == "adaptive":
        A     = build_adjacency(n_joints=53, strategy="spatial")
        model = STGCN(n_classes=N_CLASSES, A=A, adaptive=True)
    elif topology == "hands42":
        A     = build_hands42_adjacency(strategy="spatial")
        model = STGCN(n_classes=N_CLASSES, A=A, adaptive=False)
    elif topology == "latefusion":
        A     = build_hand21_adjacency(strategy="spatial")
        model = LateFusionSTGCN(n_classes=N_CLASSES, A_hand=A)
    else:
        raise ValueError(f"Unknown topology in checkpoint: {topology!r}")

    model.load_state_dict(ckpt["state_dict"])
    return model.to(device)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_arrays(normalisation: str, T: int,
                     topology: str) -> tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
    """
    Returns X_test (N,T,V,3), y_test (N,), M_test (N,T,V), X_test_raw (N,T,V,3).
    X_test_raw is always the torso-normalised version (used for heuristics only).
    M_test comes from data/processed/M_sd_test.npy regardless of normalisation.
    """
    prefix = _NORM_PREFIX[normalisation]
    X = np.load(PROC_DIR / f"{prefix}_test.npy")
    y = np.load(PROC_DIR / "y_sd_test.npy")
    M = np.load(PROC_DIR / "M_sd_test.npy")
    X_raw = np.load(PROC_DIR / "X_sd_test.npy")

    joint_slice = slice(0, 42) if topology in ("hands42", "latefusion") else None
    if joint_slice is not None:
        X     = X[:, :, joint_slice, :]
        X_raw = X_raw[:, :, joint_slice, :]
        M     = M[:, :, joint_slice]

    return X, y, M, X_raw


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module,
    X: np.ndarray,
    T: int,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (y_pred, y_prob_max, y_logits) where:
        y_pred     : (N,) int  — predicted class indices
        y_prob_max : (N,) float — max softmax probability (confidence)
        y_logits   : (N, C) float — raw logits
    """
    ds = SkeletonDataset(X, np.zeros(len(X), dtype=np.int64),
                         resample_T=T, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
    model.eval()
    preds_all, logits_all = [], []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x.to(device))
            preds_all.append(logits.argmax(1).cpu().numpy())
            logits_all.append(logits.cpu().numpy())

    y_pred    = np.concatenate(preds_all)
    y_logits  = np.concatenate(logits_all)
    y_prob    = F.softmax(torch.from_numpy(y_logits), dim=1).numpy().max(1)
    return y_pred, y_prob, y_logits


# ── Error taxonomy heuristics ─────────────────────────────────────────────────

_ERROR_TYPES = [
    "keypoint_dropout",
    "single_vs_two_hand_confusion",
    "temporal_aliasing",
    "signer_style_variation",
    "fine_grained_hand_shape",
]


def _label_name(le, idx: int) -> str:
    return str(le.classes_[idx]) if le is not None else str(idx)


def classify_failures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    M:      np.ndarray,      # (N, T, V) validity mask
    X_raw:  np.ndarray,      # (N, T, V, 3) — used for motion stats
    le,
) -> dict[str, list[dict]]:
    """
    Assign each misclassified sample to one primary error type using heuristics.

    Rules (evaluated in priority order):
      1. keypoint_dropout       : mean hand validity < 0.5
      2. fine_grained_hand_shape: same semantic category (prefix before ".")
      3. single_vs_two_hand_confusion: one class name contains "two" / "both"
                                        and the other does not (or vice versa)
      4. temporal_aliasing      : sequence mean-velocity in lowest 10th percentile
                                  (very static/slow — likely over-segmented or looped)
      5. signer_style_variation : default bucket (high confidence but wrong)

    Returns dict mapping each error type to list of example dicts.
    """
    errors: dict[str, list[dict]] = {et: [] for et in _ERROR_TYPES}
    wrong = np.where(y_true != y_pred)[0]

    # Pre-compute motion energy per sample (mean frame-to-frame velocity)
    if X_raw is not None and X_raw.ndim == 4:
        velocities = np.linalg.norm(
            np.diff(X_raw, axis=1), axis=-1).mean(axis=(1, 2))  # (N,)
        vel_10th   = np.percentile(velocities, 10)
    else:
        velocities = None
        vel_10th   = None

    hand_joints = slice(0, min(42, M.shape[2]))

    for idx in wrong:
        gt_name   = _label_name(le, y_true[idx])
        pred_name = _label_name(le, y_pred[idx])
        conf      = float(y_prob[idx])

        # hand validity: mean over all hand joints and all frames
        hand_val  = float(M[idx, :, hand_joints].mean())

        # --- heuristic 1: keypoint dropout ---
        if hand_val < 0.50:
            errors["keypoint_dropout"].append(dict(
                sample_idx=int(idx), gt=gt_name, pred=pred_name,
                confidence=round(conf, 3), hand_validity=round(hand_val, 3),
                note="Mean hand joint visibility < 50 %",
            ))
            continue

        # --- heuristic 2: fine-grained hand shape (same category prefix) ---
        gt_cat   = gt_name.split(".")[0].strip()
        pred_cat = pred_name.split(".")[0].strip()
        if gt_cat == pred_cat and gt_cat != "":
            errors["fine_grained_hand_shape"].append(dict(
                sample_idx=int(idx), gt=gt_name, pred=pred_name,
                confidence=round(conf, 3), hand_validity=round(hand_val, 3),
                note=f"Same category '{gt_cat}' — fine-grained shape confusion",
            ))
            continue

        # --- heuristic 3: single vs two-hand confusion ---
        two_kw = {"two", "both", "double"}
        gt_two   = any(kw in gt_name.lower()   for kw in two_kw)
        pred_two = any(kw in pred_name.lower() for kw in two_kw)
        if gt_two != pred_two:
            errors["single_vs_two_hand_confusion"].append(dict(
                sample_idx=int(idx), gt=gt_name, pred=pred_name,
                confidence=round(conf, 3), hand_validity=round(hand_val, 3),
                note="One sign uses two hands, confusable with one-hand variant",
            ))
            continue

        # --- heuristic 4: temporal aliasing (very low motion) ---
        if velocities is not None and velocities[idx] <= vel_10th:
            errors["temporal_aliasing"].append(dict(
                sample_idx=int(idx), gt=gt_name, pred=pred_name,
                confidence=round(conf, 3), hand_validity=round(hand_val, 3),
                mean_velocity=round(float(velocities[idx]), 5),
                note="Very low inter-frame motion — possible temporal aliasing / static sign",
            ))
            continue

        # --- default: signer style variation ---
        errors["signer_style_variation"].append(dict(
            sample_idx=int(idx), gt=gt_name, pred=pred_name,
            confidence=round(conf, 3), hand_validity=round(hand_val, 3),
            note="No structural cue found — attributed to signer execution variability",
        ))

    return errors


# ── Taxonomy markdown writer ──────────────────────────────────────────────────

_DESCRIPTIONS = {
    "keypoint_dropout": (
        "Keypoint Dropout",
        "MediaPipe fails to track one or both hands in ≥50% of frames "
        "(mean hand joint validity < 0.5). The ST-GCN receives near-zero "
        "coordinates for invisible joints; the spatial graph convolution "
        "propagates this noise through the hand subgraph, corrupting the "
        "sign's spatial pattern."
    ),
    "single_vs_two_hand_confusion": (
        "Single vs Two-Hand Confusion",
        "The ground-truth sign uses two hands while the predicted sign uses "
        "one (or vice versa). Both hands share the same graph topology; "
        "when one hand is consistently dominant the model may not learn "
        "the bilateral symmetry cue robustly enough."
    ),
    "temporal_aliasing": (
        "Temporal Aliasing / Static Sign",
        "The skeleton shows very low inter-frame motion (bottom 10th "
        "percentile of mean velocity). This often indicates a static sign "
        "(e.g. hand shape held for the full clip), over-segmented gesture, "
        "or looped frames during resampling to T=32. ST-GCN's temporal "
        "convolutions find little discriminative signal across frames."
    ),
    "signer_style_variation": (
        "Signer Style Variation",
        "The sign is well-detected (good hand visibility, normal motion) "
        "but the signer's execution deviates from the training distribution: "
        "e.g., different hand orientation, signing speed, or path trajectory. "
        "With ~9 samples/class INCLUDE cannot cover full signer variability."
    ),
    "fine_grained_hand_shape": (
        "Fine-Grained Hand Shape Confusion",
        "Ground-truth and predicted sign belong to the same INCLUDE semantic "
        "category (same numeric prefix) but differ in a subtle hand-shape "
        "detail — e.g., bent vs. extended fingers, wrist rotation. The "
        "spatial GCN with 53 joints may not resolve these fine-grained "
        "configurations reliably at 32 frames."
    ),
}


def write_taxonomy_md(
    errors:  dict[str, list[dict]],
    metrics: dict,
    ckpt_path: str,
) -> None:
    total_wrong = sum(len(v) for v in errors.values())
    lines = [
        "# Error Taxonomy — ISL ST-GCN (INCLUDE)",
        "",
        f"**Checkpoint:** `{Path(ckpt_path).name}`  ",
        f"**Test Top-1:** {metrics['test_acc']:.4f}  "
        f"**Top-5:** {metrics.get('top5_acc', 'n/a')}  "
        f"**Macro-F1:** {metrics['macro_f1']:.4f}  "
        f"**Total failures analysed:** {total_wrong}",
        "",
        "Five error categories are defined based on structural heuristics applied "
        "to each misclassified test sample. Categories are assigned in priority "
        "order (keypoint dropout → fine-grained → two-hand → temporal → style).",
        "",
    ]

    for etype in _ERROR_TYPES:
        title, desc = _DESCRIPTIONS[etype]
        examples = errors[etype]
        n = len(examples)
        lines += [
            f"---",
            f"## {title}  ({n} failures, "
            f"{100*n/max(total_wrong,1):.1f}% of errors)",
            "",
            f"**Root cause:** {desc}",
            "",
        ]

        if n == 0:
            lines += [
                "_No failures assigned to this category in the current evaluation run._",
                "",
            ]
            continue

        # Show up to 5 examples
        lines += ["**Representative failures:**", ""]
        lines += ["| # | Sample | Ground truth | Predicted | Confidence | Notes |"]
        lines += ["|---|--------|-------------|-----------|------------|-------|"]
        for i, ex in enumerate(examples[:5]):
            extra = ""
            if "mean_velocity" in ex:
                extra = f" vel={ex['mean_velocity']}"
            if "hand_validity" in ex:
                extra += f" hv={ex['hand_validity']}"
            lines.append(
                f"| {i+1} | idx={ex['sample_idx']} "
                f"| {ex['gt']} | {ex['pred']} "
                f"| {ex['confidence']:.3f} "
                f"| {ex.get('note','')}{extra} |"
            )
        if n > 5:
            lines.append(f"\n_...and {n-5} more (showing first 5)._")
        lines.append("")

    lines += [
        "---",
        "## Summary",
        "",
        "| Error Type | Count | % of errors |",
        "|------------|------:|-------------|",
    ]
    for et in _ERROR_TYPES:
        n = len(errors[et])
        title = _DESCRIPTIONS[et][0]
        lines.append(f"| {title} | {n} | {100*n/max(total_wrong,1):.1f}% |")

    lines += [
        "",
        "### Implications for future work",
        "",
        "1. **Keypoint dropout**: Use confidence-weighted adjacency or a "
        "   learned validity mask channel (already scaffolded in `dataset.py`).",
        "2. **Fine-grained hand shape**: A higher-resolution hand sub-graph "
        "   (dedicated 21-joint branch per hand) may resolve subtle "
        "   hand-shape differences — evaluated in ablation A2c (late fusion).",
        "3. **Temporal aliasing**: Static signs may benefit from an explicit "
        "   'no-motion' detection branch or longer temporal windows with "
        "   stride-1 convolutions.",
        "4. **Signer style variation**: Data augmentation (spatial jitter, "
        "   rotation) and larger training sets per class are the primary "
        "   remedies.",
        "5. **Two-hand confusion**: Explicit handedness channel or "
        "   asymmetric graph partition (left vs right subgraph with "
        "   different weights) could help.",
        "",
        "_Generated by `src/evaluate.py`._",
    ]

    TAX_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {TAX_MD}")


# ── Main evaluation pipeline ──────────────────────────────────────────────────

def evaluate(
    ckpt_path: str | Path,
    update_csv: bool = False,
    batch_size: int = 64,
) -> dict:
    set_seed(SEED)
    device   = get_device()
    ckpt     = torch.load(ckpt_path, map_location="cpu")
    topology = ckpt["topology"]
    T        = ckpt["T"]
    norm     = ckpt.get("normalisation", "torso")

    print(f"\nEvaluating: {Path(ckpt_path).name}")
    print(f"  topology={topology}  T={T}  normalisation={norm}")

    model = load_model_from_checkpoint(ckpt, device)
    X, y_true, M, X_raw = load_test_arrays(norm, T, topology)

    y_pred, y_prob, y_logits = run_inference(model, X, T, device, batch_size)

    test_acc  = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Top-5 accuracy
    top5_preds = torch.from_numpy(y_logits).topk(5, dim=1).indices.numpy()
    top5_acc   = sum(y_true[i] in top5_preds[i] for i in range(len(y_true))) / len(y_true)

    print(f"\n  Test Top-1 acc: {test_acc:.4f}")
    print(f"  Test Top-5 acc: {top5_acc:.4f}")
    print(f"  Macro-F1      : {macro_f1:.4f}")
    print(f"  (stored val)  : {ckpt.get('val_acc', 'n/a')}")

    # Load label encoder for readable class names
    le = None
    le_path = PROC_DIR / "label_encoder.pkl"
    if le_path.exists():
        with open(le_path, "rb") as f:
            le = pickle.load(f)

    # Worst-performing classes
    class_data = []
    for k, v in per_class.items():
        if k not in ("accuracy", "macro avg", "weighted avg"):
            idx = int(k)
            name = _label_name(le, idx)
            class_data.append(dict(idx=idx, name=name,
                                   f1=v["f1-score"], support=v["support"]))
    class_df = pd.DataFrame(class_data).sort_values("f1")
    print(f"\n  10 worst classes (by F1):")
    print(class_df.head(10).to_string(index=False))

    print(f"\n  10 best classes (by F1):")
    print(class_df.tail(10).sort_values("f1", ascending=False).to_string(index=False))

    # Error taxonomy
    print("\n  Classifying failures ...")
    errors  = classify_failures(y_true, y_pred, y_prob, M, X_raw, le)
    for et in _ERROR_TYPES:
        print(f"    {et}: {len(errors[et])}")

    metrics = dict(test_acc=test_acc, top5_acc=top5_acc, macro_f1=macro_f1)
    write_taxonomy_md(errors, metrics, ckpt_path)

    # Optionally update CSV
    if update_csv and RES_CSV.exists():
        df   = pd.read_csv(RES_CSV)
        cond = f"{topology}_{ckpt.get('augmentation','full')}_{norm}"
        mask = (df["experiment"] == "ablation") & \
               (df["condition"] == cond) & \
               (df["T"] == T)
        if mask.any():
            df.loc[mask, "test_acc"]  = round(test_acc, 4)
            df.loc[mask, "macro_f1"]  = round(macro_f1, 4)
            df.to_csv(RES_CSV, index=False)
            print(f"  Updated metrics_all.csv row: {cond}")

    return dict(**metrics, errors={et: len(v) for et, v in errors.items()})


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ISL ST-GCN checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .pt file in checkpoints/")
    p.add_argument("--update-csv", action="store_true",
                   help="Update metrics_all.csv with recomputed numbers")
    p.add_argument("--batch",      type=int, default=64)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    result = evaluate(args.checkpoint, update_csv=args.update_csv,
                      batch_size=args.batch)
    print(f"\nDone: {result}")


if __name__ == "__main__":
    main()
