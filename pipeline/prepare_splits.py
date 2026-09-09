"""
Dataset Splits Preparation & Realistic Sensor PSF Degradation Generator
Divides 8,000 4-channel patches into stratified Train (80%), Val (10%), and FinalTest (10%) sets.
Applies a realistic optical sensor Point Spread Function (PSF) Gaussian blur kernel (sigma in [0.6, 1.4])
before 4x bicubic downsampling to match real-world 10m sensors (like Sentinel-2).
Generates sample previews for each folder and an HR vs LR comparison preview.
"""

import os
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def stretch_channel(band: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    """Normalize band to 0-255 uint8 using percentile stretch."""
    p_low, p_high = np.percentile(band, (lower_pct, upper_pct))
    if p_high <= p_low:
        p_high = p_low + 1.0
    stretched = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
    return (stretched * 255).astype(np.uint8)


def make_rgb(patch: np.ndarray) -> np.ndarray:
    """True Color RGB composite (Ch0=R, Ch1=G, Ch2=B). Returns (H, W, 3) uint8."""
    r = stretch_channel(patch[0])
    g = stretch_channel(patch[1])
    b = stretch_channel(patch[2])
    return np.stack([r, g, b], axis=-1)


def make_cir(patch: np.ndarray) -> np.ndarray:
    """Color Infrared (CIR) False Color (Ch3=NIR, Ch0=R, Ch1=G). Returns (H, W, 3) uint8."""
    r = stretch_channel(patch[3])  # NIR
    g = stretch_channel(patch[0])  # Red
    b = stretch_channel(patch[1])  # Green
    return np.stack([r, g, b], axis=-1)


def make_gaussian_psf_kernel(
    kernel_size: int = 7,
    sigma: float = 1.0,
    channels: int = 4,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Creates a 2D Gaussian Point Spread Function (PSF) kernel
    formatted for depthwise conv2d: (channels, 1, kernel_size, kernel_size).
    """
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
    g = g / g.sum()
    kernel2d = torch.outer(g, g)
    return kernel2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)


def create_split_preview(hr_dir: Path, output_file: Path, split_name: str, num_samples: int = 8):
    """Generate a sample preview image for a split showing RGB and NIR rows."""
    files = sorted(list(hr_dir.glob("*.npy")))
    if not files:
        return

    step = max(1, len(files) // num_samples)
    samples = [files[i] for i in range(0, min(len(files), step * num_samples), step)][:num_samples]

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.8))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, f in enumerate(samples):
        arr = np.load(f)
        rgb = make_rgb(arr)
        cir = make_cir(arr)

        cat = f.stem.split("_")[1] if len(f.stem.split("_")) > 2 else ""
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"{cat}\nRGB (128x128)", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(cir)
        axes[1, col].set_title(f"NIR CIR", fontsize=8)
        axes[1, col].axis("off")

    plt.suptitle(f"{split_name} Set Sample Patches (Row 1: True Color RGB, Row 2: False Color CIR)", fontsize=11)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"  Saved preview: {output_file.name}")


def create_hr_lr_comparison(hr_dir: Path, lr_dir: Path, output_file: Path, num_samples: int = 6):
    """Generate a side-by-side comparison between HR (128x128) and PSF-degraded LR (32x32)."""
    hr_files = sorted(list(hr_dir.glob("*.npy")))
    if not hr_files:
        return

    cats = ["military", "airport", "city", "mountain", "vegetation", "water"]
    chosen_files = []
    for cat in cats:
        matching = [f for f in hr_files if f"_{cat}_" in f.name]
        if matching:
            chosen_files.append(matching[len(matching) // 2])
        if len(chosen_files) >= num_samples:
            break

    if len(chosen_files) < num_samples:
        for f in hr_files:
            if f not in chosen_files:
                chosen_files.append(f)
            if len(chosen_files) >= num_samples:
                break

    n = len(chosen_files)
    fig, axes = plt.subplots(n, 4, figsize=(10, 2.5 * n))
    if n == 1:
        axes = np.array([axes])

    for row, f in enumerate(chosen_files):
        hr_arr = np.load(f)
        lr_file = lr_dir / f.name
        lr_arr = np.load(lr_file)

        cat = f.stem.split("_")[1] if len(f.stem.split("_")) > 2 else ""

        hr_rgb = make_rgb(hr_arr)
        lr_rgb = make_rgb(lr_arr)
        hr_cir = make_cir(hr_arr)
        lr_cir = make_cir(lr_arr)

        # Real HR RGB
        axes[row, 0].imshow(hr_rgb)
        axes[row, 0].set_title(f"{cat} - Real HR (128x128)\nTrue Color RGB", fontsize=8)
        axes[row, 0].axis("off")

        # Degraded LR RGB (PSF blur + 4x downsample)
        axes[row, 1].imshow(lr_rgb, interpolation="nearest")
        axes[row, 1].set_title(f"PSF Degraded LR (32x32)\nTrue Color RGB", fontsize=8)
        axes[row, 1].axis("off")

        # Real HR CIR
        axes[row, 2].imshow(hr_cir)
        axes[row, 2].set_title(f"Real HR (128x128)\nFalse Color NIR", fontsize=8)
        axes[row, 2].axis("off")

        # Degraded LR CIR (PSF blur + 4x downsample)
        axes[row, 3].imshow(lr_cir, interpolation="nearest")
        axes[row, 3].set_title(f"PSF Degraded LR (32x32)\nFalse Color NIR", fontsize=8)
        axes[row, 3].axis("off")

    plt.suptitle("Super-Resolution Pairs: Real HR (128x128) vs Sensor PSF Blurred & 4x Downsampled LR (32x32)", fontsize=11)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved HR vs LR comparison preview: {output_file}")


def process_split(
    file_list: List[Path],
    split_name: str,
    target_split_dir: Path,
    scale: int = 4,
    sigma_min: float = 0.6,
    sigma_max: float = 1.4,
    kernel_size: int = 7,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Copy HR files and generate corresponding LR files with sensor Point Spread Function (PSF)
    Gaussian blur (sigma in [sigma_min, sigma_max]) followed by bicubic antialiased downsampling.
    """
    hr_out = target_split_dir / "HR"
    lr_out = target_split_dir / "LR"
    hr_out.mkdir(parents=True, exist_ok=True)
    lr_out.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {split_name} set ({len(file_list)} files, device={device})...")

    pad = kernel_size // 2
    batch_size = 64
    rng = random.Random(seed)

    for i in tqdm(range(0, len(file_list), batch_size), desc=f"Generating {split_name} LR"):
        batch_files = file_list[i : i + batch_size]
        arrays = []
        for f in batch_files:
            arr = np.load(f)  # (4, 128, 128) uint16
            arrays.append(arr)
            # Ensure HR file exists in HR folder
            dest_hr = hr_out / f.name
            if not dest_hr.exists():
                shutil.copy2(f, dest_hr)

        # Batch tensor
        batch_np = np.stack(arrays, axis=0).astype(np.float32)  # (B, 4, 128, 128)
        tensor = torch.from_numpy(batch_np).to(device)

        # Sample a realistic sensor PSF sigma in [sigma_min, sigma_max] for this batch
        batch_sigma = rng.uniform(sigma_min, sigma_max)
        psf_kernel = make_gaussian_psf_kernel(
            kernel_size=kernel_size,
            sigma=batch_sigma,
            channels=4,
            device=device,
        )

        _, _, H, W = tensor.shape
        lr_h, lr_w = H // scale, W // scale

        with torch.no_grad():
            # 1. Optical Sensor Point Spread Function (PSF) blur with reflection padding
            padded = F.pad(tensor, (pad, pad, pad, pad), mode="reflect")
            blurred = F.conv2d(padded, psf_kernel, groups=4)

            # 2. Sensor sampling / downsampling via antialiased bicubic interpolation
            lr_tensor = F.interpolate(
                blurred,
                size=(lr_h, lr_w),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            lr_tensor = torch.clamp(lr_tensor, 0, 65535)

        lr_batch_np = lr_tensor.cpu().numpy().astype(np.uint16)

        # Save LR files
        for j, f in enumerate(batch_files):
            dest_lr = lr_out / f.name
            np.save(dest_lr, lr_batch_np[j])

    # Generate sample preview for this split
    preview_file = target_split_dir / "sample_preview.png"
    create_split_preview(hr_out, preview_file, split_name)


def main():
    parser = argparse.ArgumentParser(description="Prepare Train/Val/FinalTest splits and generate PSF-degraded LR patches.")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--source-dir", default=str(script_dir / "data" / "4_Channel_Data"))
    parser.add_argument("--output-dir", default=str(script_dir / "data"))
    parser.add_argument("--scale", type=int, default=4, help="Downsampling scale factor (default: 4)")
    parser.add_argument("--sigma-min", type=float, default=0.6, help="Min Gaussian PSF sigma")
    parser.add_argument("--sigma-max", type=float, default=1.4, help="Max Gaussian PSF sigma")
    parser.add_argument("--kernel-size", type=int, default=7, help="Gaussian PSF kernel size")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    all_files = sorted(list(source_dir.glob("*.npy")))
    total_files = len(all_files)
    print(f"Total source patches found: {total_files}")
    if total_files == 0:
        print("ERROR: No patches found in source directory.")
        return

    # Group files by category for stratified splitting
    categories: Dict[str, List[Path]] = {}
    for f in all_files:
        cat = f.stem.split("_")[1] if len(f.stem.split("_")) > 2 else "other"
        categories.setdefault(cat, []).append(f)

    print("\nSource Category Breakdown:")
    for cat, files in categories.items():
        print(f"  - {cat}: {len(files)} files")

    # Stratified split
    rng = random.Random(args.seed)
    train_files: List[Path] = []
    val_files: List[Path] = []
    test_files: List[Path] = []

    for cat, files in categories.items():
        shuffled = files.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(round(n * args.train_ratio))
        n_val = int(round(n * args.val_ratio))
        train_cat = shuffled[:n_train]
        val_cat = shuffled[n_train : n_train + n_val]
        test_cat = shuffled[n_train + n_val :]

        train_files.extend(train_cat)
        val_files.extend(val_cat)
        test_files.extend(test_cat)

    print(f"\nStratified Split Allocation:")
    print(f"  Train: {len(train_files)} files ({len(train_files)/total_files*100:.1f}%)")
    print(f"  Val:   {len(val_files)} files ({len(val_files)/total_files*100:.1f}%)")
    print(f"  Test:  {len(test_files)} files ({len(test_files)/total_files*100:.1f}%)")

    # Process each split: copy HR and generate LR with PSF degradation
    splits = [
        ("Train", train_files, output_dir / "Train"),
        ("Val", val_files, output_dir / "Val"),
        ("FinalTest", test_files, output_dir / "FinalTest"),
    ]

    for split_name, flist, split_path in splits:
        process_split(
            file_list=flist,
            split_name=split_name,
            target_split_dir=split_path,
            scale=args.scale,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            kernel_size=args.kernel_size,
            seed=args.seed,
        )

    # Generate the comprehensive HR vs LR comparison preview
    comparison_file = output_dir / "hr_vs_lr_comparison.png"
    train_hr = output_dir / "Train" / "HR"
    train_lr = output_dir / "Train" / "LR"
    create_hr_lr_comparison(train_hr, train_lr, comparison_file)

    # Save split summary metadata
    summary = {
        "total_samples": total_files,
        "scale_factor": args.scale,
        "degradation": f"sensor_psf_gaussian_blur(sigma={args.sigma_min}-{args.sigma_max}, k={args.kernel_size}) + bicubic_antialias_4x",
        "random_seed": args.seed,
        "psf_parameters": {
            "kernel_size": args.kernel_size,
            "sigma_range": [args.sigma_min, args.sigma_max],
            "padding": "reflect",
        },
        "splits": {
            "Train": {"count": len(train_files), "hr_shape": [4, 128, 128], "lr_shape": [4, 128 // args.scale, 128 // args.scale]},
            "Val": {"count": len(val_files), "hr_shape": [4, 128, 128], "lr_shape": [4, 128 // args.scale, 128 // args.scale]},
            "FinalTest": {"count": len(test_files), "hr_shape": [4, 128, 128], "lr_shape": [4, 128 // args.scale, 128 // args.scale]},
        },
    }
    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSplit summary saved to: {summary_path}")
    print("=== Training Data Preparation Completed Successfully with Sensor PSF Degradation ===")


if __name__ == "__main__":
    main()
