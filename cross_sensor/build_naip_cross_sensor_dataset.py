import os
import sys
from pathlib import Path
import math
import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import torch
import torch.nn.functional as F

def make_gaussian_psf(kernel_size=7, sigma=1.0, channels=4):
    """Gaussian PSF kernel simulating Sentinel-2 optical MTF."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
    g = g / g.sum()
    k2d = torch.outer(g, g)
    return k2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)

def get_naip_sas_token():
    token_url = 'https://planetarycomputer.microsoft.com/api/sas/v1/token/naip'
    resp = requests.get(token_url, timeout=15)
    resp.raise_for_status()
    return resp.json().get('token', '')

def extract_naip_cross_sensor_dataset():
    root_dir = Path(__file__).resolve().parent
    hr_dir = root_dir / "data" / "HR"
    lr_dir = root_dir / "data" / "LR"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    aois = {
        'airport_runway': {
            'bbox': [-118.425, 33.935, -118.390, 33.955],
            'desc': 'Los Angeles International Airport (LAX) - Runways, taxiways, centerlines'
        },
        'urban_grid': {
            'bbox': [-118.280, 34.030, -118.230, 34.070],
            'desc': 'Downtown Los Angeles - Dense orthogonal street grid and building blocks'
        },
        'military_airbase': {
            'bbox': [-117.165, 32.865, -117.115, 32.895],
            'desc': 'MCAS Miramar Military Airbase - Hangars, revetments, taxiways, aprons'
        },
        'agriculture_canopy': {
            'bbox': [-120.680, 36.480, -120.600, 36.540],
            'desc': 'California Central Valley - Crop parcels, orchards, irrigation lineaments'
        },
        'harbor_coastal': {
            'bbox': [-118.250, 33.725, -118.200, 33.765],
            'desc': 'Port of Los Angeles / Long Beach - Wharves, piers, docks, container terminals'
        }
    }

    print("Requesting SAS token from Planetary Computer...")
    sas = get_naip_sas_token()
    search_url = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'

    psf_kernel = make_gaussian_psf(kernel_size=7, sigma=1.0, channels=4)
    # Radiometric calibration factor: transforms NAIP 8-bit counts to Sentinel-2 BOA reflectance DNs [0, 10000]
    # Channels: 0=Red (B04), 1=Green (B03), 2=Blue (B02), 3=NIR (B08)
    radiometric_gains = np.array([2784.0 / 125.0, 3065.0 / 125.0, 2626.0 / 115.0, 8350.0 / 128.0], dtype=np.float32)[:, None, None]

    target_patches_per_cat = 20
    manifest = {}

    for cat_name, info in aois.items():
        print(f"\n==========================================")
        print(f"Processing Category: {cat_name.upper()}")
        print(f"Description: {info['desc']}")
        print(f"==========================================")

        payload = {'collections': ['naip'], 'bbox': info['bbox'], 'limit': 1}
        r = requests.post(search_url, json=payload, timeout=20).json()
        feats = r.get('features', [])
        if not feats:
            raise RuntimeError(f"No NAIP feature found for {cat_name}")

        feat = feats[0]
        image_href = feat['assets']['image']['href'] + '?' + sas
        print(f"Acquiring COG: {feat['id']}")

        with rasterio.open(image_href) as src:
            w_native = src.width
            h_native = src.height
            res_x, res_y = src.res
            print(f"Native scene resolution: {res_x}m x {res_y}m, Dimension: {w_native}x{h_native}")

            # 128 pixels at 2.5m = 320 meters
            native_win_w = int(round(128 * 2.5 / res_x))
            native_win_h = int(round(128 * 2.5 / res_y))

            margin_x = int(w_native * 0.1)
            margin_y = int(h_native * 0.1)
            step_x = int(native_win_w * 1.5)
            step_y = int(native_win_h * 1.5)

            candidates = []
            for y in range(margin_y, h_native - margin_y - native_win_h, step_y):
                for x in range(margin_x, w_native - margin_x - native_win_w, step_x):
                    candidates.append((x, y))

            rng = np.random.RandomState(42)
            rng.shuffle(candidates)

            collected = 0
            for x, y in candidates:
                if collected >= target_patches_per_cat:
                    break

                win = Window(x, y, native_win_w, native_win_h)
                # Resample 0.6m -> 2.5m using area averaging (Resampling.average)
                raw_data = src.read(window=win, out_shape=(4, 128, 128), resampling=Resampling.average)

                # Quality gate: spatial standard deviation across raw 8-bit values
                spatial_std = raw_data.astype(np.float32).std(axis=(1, 2))
                mean_std = float(spatial_std.mean())

                if mean_std < 14.0:
                    continue

                val_range = raw_data.max() - raw_data.min()
                if val_range < 50:
                    continue

                # Radiometric calibration to BOA reflectance scale [0, 10000]
                hr_dn = np.clip(raw_data.astype(np.float32) * radiometric_gains, 0, 65535).astype(np.uint16)

                # Wald Protocol LR Generation:
                # 1. Convert to normalized float32
                hr_tensor = torch.from_numpy(hr_dn.astype(np.float32) / 10000.0).unsqueeze(0)

                # 2. MTF/PSF Gaussian blur (7x7, sigma=1.0) with reflection padding
                pad = 7 // 2
                padded = F.pad(hr_tensor, (pad, pad, pad, pad), mode='reflect')
                blurred = F.conv2d(padded, psf_kernel, groups=4)

                # 3. Antialiased decimation by factor of 4 (128x128 -> 32x32)
                lr_tensor = F.interpolate(blurred, size=(32, 32), mode='bicubic', align_corners=False, antialias=True)
                lr_tensor = torch.clamp(lr_tensor, 0.0, 2.0)

                # 4. Save LR in uint16 DN format
                lr_dn = torch.clamp(torch.round(lr_tensor * 10000.0), 0, 65535).squeeze(0).numpy().astype(np.uint16)

                patch_name = f"patch_{cat_name}_{collected+1:03d}.npy"
                np.save(hr_dir / patch_name, hr_dn)
                np.save(lr_dir / patch_name, lr_dn)

                manifest[patch_name] = {
                    'category': cat_name,
                    'scene_id': feat['id'],
                    'native_window': [x, y, native_win_w, native_win_h],
                    'hr_res_m': 2.5,
                    'lr_res_m': 10.0,
                    'spatial_std': mean_std
                }
                collected += 1
                print(f"[{cat_name}] Patch {collected:02d}/{target_patches_per_cat}: saved {patch_name} (std={mean_std:.1f})")

            print(f"Category {cat_name}: successfully extracted {collected} authentic 2.5m patches.")

    manifest_path = root_dir / "naip_dataset_manifest.json"
    import json
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nCompleted! Total authentic 2.5m cross-sensor patches generated: {len(manifest)}")
    print(f"Manifest saved to: {manifest_path}")

if __name__ == "__main__":
    extract_naip_cross_sensor_dataset()
