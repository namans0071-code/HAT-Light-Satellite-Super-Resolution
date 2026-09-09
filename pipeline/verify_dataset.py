"""
Dataset Verification and Quality Inspection Script
Validates patch counts, shapes, dtypes, radiometric ranges,
and generates visual RGB + NIR false-color sample composites.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def stretch_channel(band: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    """Normalize a uint16 band to 0-255 using percentile stretch."""
    p_low, p_high = np.percentile(band, (lower_pct, upper_pct))
    if p_high <= p_low:
        p_high = p_low + 1.0
    stretched = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


def make_rgb_composite(patch: np.ndarray) -> Image.Image:
    """Create True Color RGB (Ch0=R, Ch1=G, Ch2=B)."""
    r = stretch_channel(patch[0])
    g = stretch_channel(patch[1])
    b = stretch_channel(patch[2])
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray(rgb)


def make_nir_composite(patch: np.ndarray) -> Image.Image:
    """Create False Color Infrared CIR (R=NIR/Ch3, G=Red/Ch0, B=Green/Ch1)."""
    r = stretch_channel(patch[3])  # NIR
    g = stretch_channel(patch[0])  # Red
    b = stretch_channel(patch[1])  # Green
    cir = np.stack([r, g, b], axis=-1)
    return Image.fromarray(cir)


def verify_dataset(dataset_dir: Path, preview_path: Optional[Path] = None, num_samples: int = 16):
    dataset_dir = Path(dataset_dir)
    files = sorted(list(dataset_dir.glob("*.npy")))
    total_files = len(files)

    print(f"\n==========================================")
    print(f"      Dataset Verification Report")
    print(f"==========================================")
    print(f"Directory: {dataset_dir}")
    print(f"Total .npy patch files found: {total_files}")

    if total_files == 0:
        print("ERROR: No .npy patch files found.")
        return False

    shapes_valid = True
    dtype_valid = True
    nan_inf_found = False
    all_zero_found = False

    sample_patches = []
    sample_names = []

    # Category counter
    categories = {}

    for i, f in enumerate(files):
        try:
            arr = np.load(f)
        except Exception as e:
            print(f"Corrupted file: {f.name} ({e})")
            continue

        cat = f.name.split("_")[1] if len(f.name.split("_")) > 2 else "other"
        categories[cat] = categories.get(cat, 0) + 1

        if arr.shape != (4, 128, 128):
            shapes_valid = False
            print(f"Invalid shape: {f.name} has shape {arr.shape}")

        if arr.dtype != np.uint16:
            dtype_valid = False

        if np.isnan(arr).any() or np.isinf(arr).any():
            nan_inf_found = True

        if (arr == 0).all():
            all_zero_found = True

        if len(sample_patches) < num_samples and i % max(1, (total_files // num_samples)) == 0:
            sample_patches.append(arr)
            sample_names.append(f.stem)

    print(f"\n-- Shape Check: {'PASSED (all (4, 128, 128))' if shapes_valid else 'FAILED'}")
    print(f"-- Dtype Check: {'PASSED (all uint16)' if dtype_valid else 'FAILED'}")
    print(f"-- NaN/Inf Check: {'PASSED (none found)' if not nan_inf_found else 'FAILED'}")
    print(f"-- Non-Empty Check: {'PASSED (no all-zero patches)' if not all_zero_found else 'FAILED'}")

    print(f"\nCategory Distribution:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count} patches")

    # Generate preview grid
    if preview_path and len(sample_patches) > 0:
        n = min(len(sample_patches), 8)
        fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.5))
        if n == 1:
            axes = np.array([[axes[0]], [axes[1]]])

        for col in range(n):
            p = sample_patches[col]
            rgb = make_rgb_composite(p)
            nir = make_nir_composite(p)

            axes[0, col].imshow(rgb)
            axes[0, col].set_title(f"RGB #{col+1}", fontsize=9)
            axes[0, col].axis("off")

            axes[1, col].imshow(nir)
            axes[1, col].set_title(f"NIR CIR #{col+1}", fontsize=9)
            axes[1, col].axis("off")

        plt.suptitle("Sample 4-Channel Patches: Row 1 = True Color (RGB), Row 2 = False Color (NIR-R-G)", fontsize=11)
        plt.tight_layout()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(preview_path, dpi=150)
        plt.close()
        print(f"\nSample visual composite saved to: {preview_path}")

    return shapes_valid and dtype_valid and not nan_inf_found and not all_zero_found


def verify_splits(data_dir: Path):
    data_dir = Path(data_dir)
    print(f"\n==========================================")
    print(f"      Train / Val / FinalTest Split Check")
    print(f"==========================================")
    print(f"Data Directory: {data_dir}")

    splits = ["Train", "Val", "FinalTest"]
    all_ok = True

    for sp in splits:
        sp_path = data_dir / sp
        hr_path = sp_path / "HR"
        lr_path = sp_path / "LR"
        preview_file = sp_path / "sample_preview.png"

        hr_files = sorted(list(hr_path.glob("*.npy")))
        lr_files = sorted(list(lr_path.glob("*.npy")))

        print(f"\n[{sp} Set]")
        print(f"  HR count: {len(hr_files)} files")
        print(f"  LR count: {len(lr_files)} files")
        print(f"  Preview exists: {preview_file.exists()}")

        if len(hr_files) != len(lr_files):
            print(f"  ERROR: Count mismatch between HR ({len(hr_files)}) and LR ({len(lr_files)})")
            all_ok = False

        # Check 1-to-1 match
        hr_names = set(f.name for f in hr_files)
        lr_names = set(f.name for f in lr_files)
        if hr_names != lr_names:
            print(f"  ERROR: Filename set mismatch between HR and LR!")
            all_ok = False
        else:
            print(f"  Filenames match: 1-to-1 exact pairing verified")

        # Spot check shapes and dtypes
        if hr_files:
            hr_sample = np.load(hr_files[0])
            lr_sample = np.load(lr_files[0])
            print(f"  Sample HR shape: {hr_sample.shape} ({hr_sample.dtype})")
            print(f"  Sample LR shape: {lr_sample.shape} ({lr_sample.dtype})")

            if hr_sample.shape != (4, 128, 128):
                print(f"  ERROR: Invalid HR shape {hr_sample.shape}")
                all_ok = False
            if lr_sample.shape != (4, 32, 32):
                print(f"  ERROR: Invalid LR shape {lr_sample.shape}")
                all_ok = False

    comp_preview = data_dir / "hr_vs_lr_comparison.png"
    print(f"\nHR vs LR Comparison Preview exists: {comp_preview.exists()}")
    print(f"\n-- Split Verification Result: {'PASSED ALL CHECKS' if all_ok else 'FAILED'}")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--dir", default=str(script_dir / "data" / "4_Channel_Data"))
    parser.add_argument("--preview", default=str(script_dir / "sample_patches_preview.png"))
    parser.add_argument("--check-splits", action="store_true", help="Verify Train/Val/FinalTest data directories")
    parser.add_argument("--data-dir", default=str(script_dir / "data"))
    args = parser.parse_args()

    if args.check_splits:
        verify_splits(Path(args.data_dir))
    else:
        verify_dataset(Path(args.dir), Path(args.preview))

