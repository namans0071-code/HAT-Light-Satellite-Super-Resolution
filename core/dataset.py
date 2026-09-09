"""
PyTorch Dataset and Processing Utilities for 4-Channel Sentinel-2 Super-Resolution.
Handles uint16 (0-65535) .npy patches with full-dynamic-range normalization.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


# Sentinel-2 standard BOA reflectance scale (1.0 reflectance = 10,000 DN)
REFLECTANCE_SCALE = 10000.0
# Backward compatibility alias
REFLECTANCE_MAX = REFLECTANCE_SCALE


def normalize_patch(arr: np.ndarray) -> np.ndarray:
    """Normalize uint16 reflectance to float32 using Sentinel-2 scale (10000 DN = 1.0)."""
    arr_f32 = arr.astype(np.float32)
    return np.clip(arr_f32 / REFLECTANCE_SCALE, 0.0, 2.0)


def denormalize_patch(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized float tensor back to uint16 numpy array."""
    arr = torch.clamp(tensor * REFLECTANCE_SCALE, 0.0, 65535.0).cpu().numpy().astype(np.uint16)
    return arr


def stretch_channel(band: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    """Normalize 2D band to 0-255 uint8 using percentile stretch for visual display."""
    p_low, p_high = np.percentile(band, (lower_pct, upper_pct))
    if p_high <= p_low:
        p_high = p_low + 1.0
    stretched = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


def make_rgb_composite(patch: np.ndarray) -> np.ndarray:
    """
    True Color RGB composite.
    Sentinel-2: Ch0 = Red (B04), Ch1 = Green (B03), Ch2 = Blue (B02).
    Returns (H, W, 3) uint8 image.
    """
    if patch.ndim == 4:
        patch = patch[0]
    r = stretch_channel(patch[0])
    g = stretch_channel(patch[1])
    b = stretch_channel(patch[2])
    return np.stack([r, g, b], axis=-1)


def make_cir_composite(patch: np.ndarray) -> np.ndarray:
    """
    Color Infrared (CIR) False Color composite.
    Sentinel-2: Ch3 = NIR (B08), Ch0 = Red (B04), Ch1 = Green (B03).
    Returns (H, W, 3) uint8 image.
    """
    if patch.ndim == 4:
        patch = patch[0]
    r = stretch_channel(patch[3])  # NIR
    g = stretch_channel(patch[0])  # Red
    b = stretch_channel(patch[1])  # Green
    return np.stack([r, g, b], axis=-1)


class SatelliteSRDataset(Dataset):
    """
    Dataset for paired Satellite Super-Resolution.
    Loads HR and corresponding LR patches (pre-computed or on-the-fly downsampled).
    """
    def __init__(
        self,
        split_dir: str or Path,
        scale: int = 4,
        is_train: bool = True,
        augment: bool = True,
        max_samples: Optional[int] = None
    ):
        super().__init__()
        self.split_dir = Path(split_dir)
        self.scale = scale
        self.is_train = is_train
        self.augment = augment and is_train

        hr_dir = self.split_dir / "HR"
        lr_dir = self.split_dir / "LR"

        if hr_dir.exists():
            self.hr_files = sorted(list(hr_dir.glob("*.npy")))
        elif self.split_dir.exists():
            self.hr_files = sorted(list(self.split_dir.glob("*.npy")))
        else:
            self.hr_files = []

        if max_samples and max_samples < len(self.hr_files):
            self.hr_files = self.hr_files[:max_samples]

        self.lr_dir = lr_dir if (lr_dir.exists() and scale == 4) else None

    def __len__(self) -> int:
        return len(self.hr_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_path = self.hr_files[idx]
        hr_arr = np.load(hr_path)  # (4, H, W) uint16
        hr_norm = normalize_patch(hr_arr)
        hr_tensor = torch.from_numpy(hr_norm)  # (4, H, W) float32

        if self.lr_dir:
            lr_path = self.lr_dir / hr_path.name
            if lr_path.exists():
                lr_arr = np.load(lr_path)
                lr_norm = normalize_patch(lr_arr)
                lr_tensor = torch.from_numpy(lr_norm)
            else:
                lr_tensor = self._downsample_lr(hr_tensor)
        else:
            lr_tensor = self._downsample_lr(hr_tensor)

        if self.augment:
            if random.random() > 0.5:
                hr_tensor = torch.flip(hr_tensor, dims=[2])
                lr_tensor = torch.flip(lr_tensor, dims=[2])
            if random.random() > 0.5:
                hr_tensor = torch.flip(hr_tensor, dims=[1])
                lr_tensor = torch.flip(lr_tensor, dims=[1])
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                hr_tensor = torch.rot90(hr_tensor, k=rot_k, dims=[1, 2])
                lr_tensor = torch.rot90(lr_tensor, k=rot_k, dims=[1, 2])

        return lr_tensor.contiguous(), hr_tensor.contiguous()

    def _downsample_lr(self, hr_tensor: torch.Tensor) -> torch.Tensor:
        """Bicubic antialiased downsampling of HR to generate paired LR."""
        c, h, w = hr_tensor.shape
        lr_h, lr_w = h // self.scale, w // self.scale
        hr_b = hr_tensor.unsqueeze(0)
        with torch.no_grad():
            lr_b = F.interpolate(
                hr_b,
                size=(lr_h, lr_w),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )
            lr_b = torch.clamp(lr_b, 0.0, 2.0)
        return lr_b.squeeze(0)
