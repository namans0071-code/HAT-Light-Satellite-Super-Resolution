import os
import sys
from pathlib import Path
import math
import numpy as np
import requests
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import transform
import torch
import torch.nn.functional as F

def make_gaussian_psf(kernel_size=7, sigma=1.0, channels=4):
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

def main():
    root_dir = Path(__file__).resolve().parent
    hr_dir = root_dir / "data" / "HR"
    lr_dir = root_dir / "data" / "LR"
    hr_dir.mkdir(parents=True, exist_ok=True)
    lr_dir.mkdir(parents=True, exist_ok=True)

    scenes = {
        'airport_runway': {
            'point': [-118.4085, 33.9425], # LAX Runway complex
            'desc': 'Los Angeles International Airport (LAX) - Runways, taxiways, markings'
        },
        'urban_grid': {
            'point': [-118.2550, 34.0480], # Downtown Los Angeles
            'desc': 'Downtown Los Angeles - Dense orthogonal street networks & buildings'
        },
        'military_airbase': {
            'point': [-117.1425, 32.8685], # MCAS Miramar Airbase
            'desc': 'MCAS Miramar Military Airbase - Aircraft revetments, aprons, hangars'
        },
        'agriculture_canopy': {
            'point': [-119.8500, 36.6500], # Central Valley Agricultural Belt (Fresno)
            'desc': 'San Joaquin Valley - High-NDVI crop parcels, orchards, irrigation lineaments'
        },
        'harbor_coastal': {
            'point': [-118.2600, 33.7250], # Port of Los Angeles Pier 400
            'desc': 'Port of Los Angeles - Wharves, container terminals, docks, berths'
        }
    }

    print("Requesting SAS token from Planetary Computer...")
    sas = get_naip_sas_token()
    search_url = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'

    psf_kernel = make_gaussian_psf(kernel_size=7, sigma=1.0, channels=4)
    radiometric_gains = np.array([2784.0 / 125.0, 3065.0 / 125.0, 2626.0 / 115.0, 8350.0 / 128.0], dtype=np.float32)[:, None, None]

    target_patches_per_cat = 20
    manifest = {}

    for cat_name, info in scenes.items():
        print(f"\n==========================================")
        print(f"Acquiring Biome: {cat_name.upper()}")
        print(f"Coordinates: {info['point']} ({info['desc']})")
        print(f"==========================================")

        payload = {'collections': ['naip'], 'intersects': {'type': 'Point', 'coordinates': info['point']}, 'limit': 1}
        r = requests.post(search_url, json=payload, timeout=20).json()
        feats = r.get('features', [])
        if not feats:
            raise RuntimeError(f"No NAIP feature found for {cat_name}")

        feat = feats[0]
        image_href = feat['assets']['image']['href'] + '?' + sas
        print(f"Acquiring STAC COG: {feat['id']} ({feat['properties']['datetime'][:10]})")

        with rasterio.open(image_href) as src:
            w_native = src.width
            h_native = src.height
            res_x, res_y = src.res
            print(f"Scene Resolution: {res_x}m x {res_y}m, Dimensions: {w_native}x{h_native}")

            # Project coordinates from EPSG:4326 to src.crs
            xs, ys = transform('EPSG:4326', src.crs, [info['point'][0]], [info['point'][1]])
            cy_target, cx_target = src.index(xs[0], ys[0])
            print(f"Target Feature Pixel Center: ({cx_target}, {cy_target})")

            # 128 pixels at 2.5m = 320 meters
            native_win_w = int(round(128 * 2.5 / res_x))
            native_win_h = int(round(128 * 2.5 / res_y))

            # Sample dense region around target
            radius = 3200
            x_min = max(0, cx_target - radius)
            x_max = min(w_native - native_win_w, cx_target + radius)
            y_min = max(0, cy_target - radius)
            y_max = min(h_native - native_win_h, cy_target + radius)

            step = int(native_win_w * 0.9)
            candidates = []
            for y in range(y_min, y_max, step):
                for x in range(x_min, x_max, step):
                    dist = math.hypot(x - cx_target, y - cy_target)
                    candidates.append((dist, x, y))

            # Sort by distance to target point to prioritize center feature
            candidates.sort(key=lambda item: item[0])

            collected = 0
            for _, x, y in candidates:
                if collected >= target_patches_per_cat:
                    break

                win = Window(x, y, native_win_w, native_win_h)
                raw_data = src.read(window=win, out_shape=(4, 128, 128), resampling=Resampling.average)

                spatial_std = raw_data.astype(np.float32).std(axis=(1, 2))
                mean_std = float(spatial_std.mean())

                if mean_std < 12.0:
                    continue

                val_range = raw_data.max() - raw_data.min()
                if val_range < 45:
                    continue

                # Radiometric calibration to BOA reflectance scale [0, 10000]
                hr_dn = np.clip(raw_data.astype(np.float32) * radiometric_gains, 0, 65535).astype(np.uint16)

                # Wald Protocol LR Generation:
                hr_tensor = torch.from_numpy(hr_dn.astype(np.float32) / 10000.0).unsqueeze(0)
                pad = 7 // 2
                padded = F.pad(hr_tensor, (pad, pad, pad, pad), mode='reflect')
                blurred = F.conv2d(padded, psf_kernel, groups=4)
                lr_tensor = F.interpolate(blurred, size=(32, 32), mode='bicubic', align_corners=False, antialias=True)
                lr_tensor = torch.clamp(lr_tensor, 0.0, 2.0)
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

if __name__ == "__main__":
    main()
