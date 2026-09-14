"""
PyTorch Dataset for ISL skeleton sequences.

ST-GCN expects input shape (N, C, T, V):
    N — batch size
    C — coordinate channels (3: x,y,z  or  4: x,y,z,mask)
    T — time frames (64 after resampling)
    V — joints (53)

The .npy files on disk are stored as (N, T, V, C); this class handles
the transposition and optional data augmentation.

Augmentation follows Yan et al. (2018) Appendix:
    - Gaussian joint noise  : σ = 0.01
    - Random temporal crop  : drop up to 10 % of frames, then resample to T

Optional validity mask (M):
    Pass M of shape (N, T, V) float32.  It will be appended as a 4th
    channel → final shape (N, 4, T, V).  The model receives in_channels=4.
    When M is None the output is (N, 3, T, V) as before.
"""

from __future__ import annotations
import math
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class SkeletonDataset(Dataset):
    """
    Wraps pre-processed (N, T, V, C) numpy arrays for ST-GCN training.

    Parameters
    ----------
    X         : (N, T, V, 3) float32 array — skeleton coords
    y         : (N,) int64 label array
    M         : (N, T, V) float32 array — validity mask (optional)
                When provided, appended as 4th channel → (N, 4, T, V)
    augment   : enable data augmentation (use only for training split)
    noise_std : standard deviation of per-joint Gaussian noise
    crop_ratio: maximum fraction of frames to drop in temporal crop
    resample_T: if set, resample the time axis from its stored length to
                this many frames (linear interpolation) once at load time —
                used for the temporal-window ablation (T = 32/48/64/96)
                without needing separately preprocessed .npy files
    """

    # Left-hand joints 0-20, right-hand joints 21-41 (within the 53-joint layout)
    _L_HAND = list(range(0, 21))
    _R_HAND = list(range(21, 42))
    # Upper-body L/R joint pairs (see src/graph.py layout): l/r-ear, l/r-shoulder,
    # l/r-elbow, l/r-wrist(pose), l/r-hip. Nose (42) is midline and stays put.
    _L_BODY = [43, 45, 47, 49, 51]
    _R_BODY = [44, 46, 48, 50, 52]

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        M: Optional[np.ndarray] = None,
        augment: bool = False,
        noise_std: float = 0.01,
        crop_ratio: float = 0.1,
        flip_prob: float = 0.5,
        resample_T: Optional[int] = None,
    ):
        if M is not None:
            # Concatenate mask as 4th channel: (N,T,V,3) + (N,T,V,1) → (N,T,V,4)
            X_with_mask = np.concatenate(
                [X, M[..., np.newaxis].astype(np.float32)], axis=-1
            )
        else:
            X_with_mask = X

        # (N, T, V, C) → (N, C, T, V)
        self.X = torch.from_numpy(
            X_with_mask.transpose(0, 3, 1, 2).astype(np.float32)
        )

        if resample_T is not None and resample_T != self.X.shape[2]:
            N, C, T, V = self.X.shape
            x_flat = self.X.permute(0, 1, 3, 2).reshape(N, C * V, T)   # (N, C*V, T)
            x_flat = F.interpolate(x_flat, size=resample_T, mode="linear", align_corners=False)
            self.X = x_flat.reshape(N, C, V, resample_T).permute(0, 1, 3, 2).contiguous()

        self.y = torch.from_numpy(y.astype(np.int64))
        self.augment    = augment
        self.noise_std  = noise_std
        self.crop_ratio = crop_ratio
        self.flip_prob  = flip_prob

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()   # (C, T, V)

        if self.augment:
            x = self._add_joint_noise(x)
            x = self._random_temporal_crop(x)
            x = self._random_horizontal_flip(x)
            x = self._random_rotation(x)

        return x, self.y[idx]

    # ── augmentation helpers ──────────────────────────────────────────────────

    def _add_joint_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Additive zero-mean Gaussian noise on all joint coordinates."""
        return x + torch.randn_like(x) * self.noise_std

    def _random_horizontal_flip(self, x: torch.Tensor) -> torch.Tensor:
        """
        With probability flip_prob, swap left-hand and right-hand joint columns
        (and the L/R upper-body joint pairs) and mirror the x-coordinate
        (index 0 of the C dimension).

        Swapping L↔R joints simulates a mirrored signer.  The x-coordinate
        is negated so the spatial layout remains consistent after the swap.
        Every L/R pair in the 53-joint layout must be swapped together — hands
        AND ears/shoulders/elbows/wrists/hips — otherwise the mirrored hand
        ends up wired to the un-mirrored arm via the inter-hand chain edges
        (0-49, 21-50), producing an anatomically impossible skeleton.
        Only applied when the input has V=53 (full skeleton).  For hands-only or
        late-fusion inputs (V=42 or V=21) the flip is skipped to avoid breaking
        the joint ordering assumed by those models.
        """
        C, T, V = x.shape
        if V != 53:
            return x
        if torch.rand(1).item() > self.flip_prob:
            return x
        x = x.clone()
        # Swap joint columns L↔R hand and L↔R upper-body pairs
        l_idx = self._L_HAND + self._L_BODY
        r_idx = self._R_HAND + self._R_BODY
        x[:, :, l_idx], x[:, :, r_idx] = (
            x[:, :, r_idx].clone(),
            x[:, :, l_idx].clone(),
        )
        # Mirror x-coordinate (channel 0) so left side becomes right
        x[0] = -x[0]
        return x

    def _random_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate all joints in the x-y plane by a random angle in [-15°, +15°].
        Simulates the signer being slightly off-centre relative to the camera.
        The z-channel is left unchanged (depth is not reliable from a webcam).
        """
        angle = (torch.rand(1).item() - 0.5) * 2 * (15.0 * math.pi / 180.0)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        x = x.clone()
        x_ch = x[0].clone()   # (T, V) — x-coordinate
        y_ch = x[1].clone()   # (T, V) — y-coordinate
        x[0] = cos_a * x_ch - sin_a * y_ch
        x[1] = sin_a * x_ch + cos_a * y_ch
        return x

    def _random_temporal_crop(self, x: torch.Tensor) -> torch.Tensor:
        """
        Drop a random number of frames (up to crop_ratio * T), starting from
        a random offset, then resample back to the original length via linear
        interpolation.  Preserves the (C, T, V) shape.
        """
        C, T, V = x.shape
        max_drop = int(T * self.crop_ratio)
        n_drop = torch.randint(0, max_drop + 1, (1,)).item()

        if max_drop == 0 or n_drop == 0:
            return x

        start  = torch.randint(0, n_drop + 1, (1,)).item()
        x_crop = x[:, start : start + T - n_drop, :]    # (C, T_crop, V)

        T_crop = x_crop.shape[1]
        if T_crop < 2:
            return x

        # F.interpolate works on (N, Channels, Length) and resizes the LAST
        # dimension. x_crop is (C, T_crop, V) with T_crop in the middle, so it
        # must be permuted to put T_crop last BEFORE collapsing C*V — reshaping
        # directly (as this used to do) silently interleaves the T and V axes,
        # scrambling which joint's data ends up at which time step. Mirrors the
        # resample done in __init__ (self.X.permute(0, 1, 3, 2).reshape(...)).
        x_resized = F.interpolate(
            x_crop.permute(0, 2, 1).reshape(1, C * V, T_crop),
            size=T,
            mode="linear",
            align_corners=False,
        ).reshape(C, V, T).permute(0, 2, 1)

        return x_resized


# ── Factory functions ─────────────────────────────────────────────────────────

def load_sd_loaders(
    proc_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    augment_train: bool = True,
    use_mask: bool = False,
    resample_T: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the Signer-Dependent (SD) splits from data/processed/ and return
    (train_loader, val_loader, test_loader).

    Parameters
    ----------
    use_mask   : if True, load M_sd_*.npy and pass to SkeletonDataset as 4th channel.
                 Model must be built with in_channels=4.
    resample_T : if set, resample every split to this many frames (temporal-window ablation).
    """
    proc_dir = Path(proc_dir)

    X_train = np.load(proc_dir / "X_sd_train.npy")
    y_train = np.load(proc_dir / "y_sd_train.npy")
    X_val   = np.load(proc_dir / "X_sd_val.npy")
    y_val   = np.load(proc_dir / "y_sd_val.npy")
    X_test  = np.load(proc_dir / "X_sd_test.npy")
    y_test  = np.load(proc_dir / "y_sd_test.npy")

    M_train = M_val = M_test = None
    if use_mask:
        M_train = np.load(proc_dir / "M_sd_train.npy")
        M_val   = np.load(proc_dir / "M_sd_val.npy")
        M_test  = np.load(proc_dir / "M_sd_test.npy")

    train_ds = SkeletonDataset(X_train, y_train, M=M_train, augment=augment_train, resample_T=resample_T)
    val_ds   = SkeletonDataset(X_val,   y_val,   M=M_val,   augment=False,         resample_T=resample_T)
    test_ds  = SkeletonDataset(X_test,  y_test,  M=M_test,  augment=False,         resample_T=resample_T)

    kw = dict(num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader


def load_si_fold(
    proc_dir: str | Path,
    fold: int,
    batch_size: int = 32,
    num_workers: int = 0,
    augment_train: bool = True,
    use_mask: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Load one fold of the Signer-Independent (SI) 5-fold splits.
    Returns (train_loader, test_loader) for the specified fold.

    Parameters
    ----------
    use_mask : if True, load M_train/M_test from the split dict and pass
               as 4th channel.  Model must be built with in_channels=4.
    """
    proc_dir = Path(proc_dir)
    with open(proc_dir / "si_splits.pkl", "rb") as f:
        si_splits = pickle.load(f)

    split = si_splits[fold]

    M_train = split.get("M_train") if use_mask else None
    M_test  = split.get("M_test")  if use_mask else None

    train_ds = SkeletonDataset(split["X_train"], split["y_train"], M=M_train, augment=augment_train)
    test_ds  = SkeletonDataset(split["X_test"],  split["y_test"],  M=M_test,  augment=False)

    kw = dict(num_workers=num_workers, pin_memory=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw)

    return train_loader, test_loader
