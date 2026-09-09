"""
Sentinel-2 4-Channel Patch Extractor
Extracts 128x128 patches (R, G, B, IR) from Sentinel-2 L2A 10m JP2 bands
with strict cloud, no-data, and variance validation.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class PatchExtractor:
    def __init__(
        self,
        patch_size: int = 128,
        min_std: float = 25.0,
        max_cloud_reflectance: int = 3800,
        max_no_data_val: int = 0,
    ):
        """
        Args:
            patch_size: Dimension of square patches (128x128).
            min_std: Minimum average standard deviation across bands to reject flat/blank patches.
            max_cloud_reflectance: Reflectance threshold (scaled by 10,000) to reject cloud saturation.
            max_no_data_val: Value representing no-data pixels (Sentinel-2 L2A uses 0).
        """
        self.patch_size = patch_size
        self.min_std = min_std
        self.max_cloud_reflectance = max_cloud_reflectance
        self.max_no_data_val = max_no_data_val

    def load_scene_bands(self, band_paths: Dict[str, Path]) -> np.ndarray:
        """
        Load the 4 bands (Red=B04, Green=B03, Blue=B02, NIR=B08) and stack into (4, H, W) uint16 array.
        Order: [Red, Green, Blue, NIR]
        """
        required = ["B04", "B03", "B02", "B08"]
        for b in required:
            if b not in band_paths or not Path(band_paths[b]).exists():
                raise FileNotFoundError(f"Missing required band file for {b}: {band_paths.get(b)}")

        # Red (B04), Green (B03), Blue (B02), NIR (B08)
        loaded = []
        expected_shape = None

        for b in required:
            img = Image.open(band_paths[b])
            arr = np.array(img, dtype=np.uint16)
            if expected_shape is None:
                expected_shape = arr.shape
            elif arr.shape != expected_shape:
                raise ValueError(f"Shape mismatch: {b} has shape {arr.shape}, expected {expected_shape}")
            loaded.append(arr)

        # Stack to (4, H, W)
        scene = np.stack(loaded, axis=0)
        return scene

    def is_patch_valid(self, patch: np.ndarray) -> bool:
        """
        Validate patch quality:
        - Must have shape (4, patch_size, patch_size)
        - No missing/zero pixels (borders or nodata)
        - No high-reflectance cloud saturation
        - Sufficient variance / texture (not completely featureless)
        """
        if patch.shape != (4, self.patch_size, self.patch_size):
            return False

        # 1. No-data check: Reject if any pixel is 0 or NaN
        if (patch <= self.max_no_data_val).any():
            return False

        # 2. Cloud / saturation check:
        # In Sentinel-2 L2A, thick clouds typically have reflectance > 3800 across R, G, B
        # If > 10% of visible pixels exceed threshold, reject
        vis_bands = patch[0:3]  # R, G, B
        cloud_mask = (vis_bands > self.max_cloud_reflectance).all(axis=0)
        if np.mean(cloud_mask) > 0.05:  # more than 5% cloud pixels
            return False

        # Also check if mean visible brightness is unnaturally high (cloud top)
        if np.mean(vis_bands) > 3200:
            return False

        # 3. Variance check: Reject flat / featureless regions (e.g. pure open water or uniform haze)
        std_per_channel = np.std(patch, axis=(1, 2))
        if np.mean(std_per_channel) < self.min_std:
            return False

        return True

    def extract_patches_from_scene(
        self,
        scene: np.ndarray,
        target_count: int,
        category: str,
        output_dir: Path,
        start_index: int = 0,
        max_candidates: int = 20000,
        seed: int = 42,
    ) -> Tuple[int, List[Path]]:
        """
        Randomly and uniformly sample patches across the scene until target_count valid patches are found.
        Saves each patch as .npy array with shape (4, 128, 128) in output_dir.
        Returns: (number of saved patches, list of saved file paths).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _, H, W = scene.shape
        # Use half-stride (step=64) to allow rich spatial sampling across 100km scene
        step = self.patch_size // 2

        # Generate candidate grid coordinates
        y_coords = list(range(0, H - self.patch_size, step))
        x_coords = list(range(0, W - self.patch_size, step))

        candidates = [(y, x) for y in y_coords for x in x_coords]
        rng = random.Random(seed)
        rng.shuffle(candidates)

        saved_files: List[Path] = []
        current_idx = start_index

        for y, x in candidates:
            if len(saved_files) >= target_count:
                break

            patch = scene[:, y : y + self.patch_size, x : x + self.patch_size]
            if self.is_patch_valid(patch):
                filename = f"patch_{category}_{current_idx:06d}.npy"
                file_path = output_dir / filename
                np.save(file_path, patch)
                saved_files.append(file_path)
                current_idx += 1

        return len(saved_files), saved_files
