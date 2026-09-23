import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import math
import time
import json
import numpy as np

from core.hat_light import HATLightSR
from core.dataset import SatelliteSRDataset
from torch.utils.data import DataLoader
from core.losses import calculate_band_metrics, calculate_psnr


def compute_sam(pred: torch.Tensor, gt: torch.Tensor) -> float:
    dot = torch.sum(pred * gt, dim=1)
    norm_p = torch.sqrt(torch.sum(pred ** 2, dim=1) + 1e-8)
    norm_g = torch.sqrt(torch.sum(gt ** 2, dim=1) + 1e-8)
    cos_theta = torch.clamp(dot / (norm_p * norm_g), -1.0, 1.0)
    angle_rad = torch.acos(cos_theta)
    return torch.rad2deg(angle_rad).mean().item()


def compute_ergas(pred: torch.Tensor, gt: torch.Tensor, scale: float = 4.0) -> float:
    B, C, H, W = pred.shape
    sum_terms = 0.0
    for c in range(C):
        mse = torch.mean((pred[:, c] - gt[:, c]) ** 2).item()
        mean_gt = torch.mean(gt[:, c]).item() + 1e-6
        sum_terms += mse / (mean_gt ** 2)
    return (100.0 / scale) * math.sqrt(sum_terms / C)


def calc_band_psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.5):
    diff = pred - gt
    psnr_all = 10.0 * math.log10((max_val ** 2) / max(1e-10, torch.mean(diff ** 2).item()))
    psnr_b04 = 10.0 * math.log10((max_val ** 2) / max(1e-10, torch.mean(diff[:, 0] ** 2).item()))
    psnr_b03 = 10.0 * math.log10((max_val ** 2) / max(1e-10, torch.mean(diff[:, 1] ** 2).item()))
    psnr_b02 = 10.0 * math.log10((max_val ** 2) / max(1e-10, torch.mean(diff[:, 2] ** 2).item()))
    psnr_b08 = 10.0 * math.log10((max_val ** 2) / max(1e-10, torch.mean(diff[:, 3] ** 2).item()))
    return psnr_all, psnr_b04, psnr_b03, psnr_b02, psnr_b08


def evaluate():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    test_dir = REPO_ROOT / 'test_dataset'
    ds = SatelliteSRDataset(test_dir, scale=4, is_train=False, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    print(f"Loaded {len(ds)} test patches from {test_dir}.")

    model = HATLightSR().to(device)
    weights_path = REPO_ROOT / 'weights' / 'best_model.pth'
    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    bic_metrics = {'psnr': [], 'ssim': [], 'mae': [], 'sam': [], 'ergas': [], 'b04': [], 'b03': [], 'b02': [], 'b08': []}
    hat_metrics = {'psnr': [], 'ssim': [], 'mae': [], 'sam': [], 'ergas': [], 'b04': [], 'b03': [], 'b02': [], 'b08': []}

    start_time = time.time()
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            # Bicubic
            bic = torch.clamp(torch.nn.functional.interpolate(lr, size=(hr.shape[2], hr.shape[3]), mode='bicubic', align_corners=False), 0.0, 2.0)
            # HAT-Light
            sr = model(lr)

            # Bicubic metrics
            b_m = calculate_band_metrics(bic, hr)
            bp_all, bp_04, bp_03, bp_02, bp_08 = calc_band_psnr(bic, hr)
            bic_metrics['psnr'].append(bp_all)
            bic_metrics['ssim'].append(b_m['ssim'])
            bic_metrics['mae'].append(torch.mean(torch.abs(bic - hr)).item())
            bic_metrics['sam'].append(compute_sam(bic, hr))
            bic_metrics['ergas'].append(compute_ergas(bic, hr))
            bic_metrics['b04'].append(bp_04)
            bic_metrics['b03'].append(bp_03)
            bic_metrics['b02'].append(bp_02)
            bic_metrics['b08'].append(bp_08)

            # HAT metrics
            h_m = calculate_band_metrics(sr, hr)
            hp_all, hp_04, hp_03, hp_02, hp_08 = calc_band_psnr(sr, hr)
            hat_metrics['psnr'].append(hp_all)
            hat_metrics['ssim'].append(h_m['ssim'])
            hat_metrics['mae'].append(torch.mean(torch.abs(sr - hr)).item())
            hat_metrics['sam'].append(compute_sam(sr, hr))
            hat_metrics['ergas'].append(compute_ergas(sr, hr))
            hat_metrics['b04'].append(hp_04)
            hat_metrics['b03'].append(hp_03)
            hat_metrics['b02'].append(hp_02)
            hat_metrics['b08'].append(hp_08)

    eval_time = time.time() - start_time
    print(f"Evaluation finished in {eval_time:.2f}s ({len(ds)/eval_time:.2f} patches/s)\n")

    results = {
        "num_patches": len(ds),
        "Bicubic": {
            "PSNR_overall": float(np.mean(bic_metrics['psnr'])),
            "PSNR_Red_B04": float(np.mean(bic_metrics['b04'])),
            "PSNR_Green_B03": float(np.mean(bic_metrics['b03'])),
            "PSNR_Blue_B02": float(np.mean(bic_metrics['b02'])),
            "PSNR_NIR_B08": float(np.mean(bic_metrics['b08'])),
            "SSIM": float(np.mean(bic_metrics['ssim'])),
            "SAM_deg": float(np.mean(bic_metrics['sam'])),
            "ERGAS": float(np.mean(bic_metrics['ergas'])),
            "MAE": float(np.mean(bic_metrics['mae']))
        },
        "HAT-Light": {
            "PSNR_overall": float(np.mean(hat_metrics['psnr'])),
            "PSNR_Red_B04": float(np.mean(hat_metrics['b04'])),
            "PSNR_Green_B03": float(np.mean(hat_metrics['b03'])),
            "PSNR_Blue_B02": float(np.mean(hat_metrics['b02'])),
            "PSNR_NIR_B08": float(np.mean(hat_metrics['b08'])),
            "SSIM": float(np.mean(hat_metrics['ssim'])),
            "SAM_deg": float(np.mean(hat_metrics['sam'])),
            "ERGAS": float(np.mean(hat_metrics['ergas'])),
            "MAE": float(np.mean(hat_metrics['mae']))
        }
    }

    print("=== FINAL CURATED TEST BENCHMARK RESULTS (600 PATCHES) ===")
    for model_name, m in results.items():
        if isinstance(m, dict):
            print(f"[{model_name}]")
            print(f"  PSNR Overall : {m['PSNR_overall']:.2f} dB (Red: {m['PSNR_Red_B04']:.2f}, Green: {m['PSNR_Green_B03']:.2f}, Blue: {m['PSNR_Blue_B02']:.2f}, NIR: {m['PSNR_NIR_B08']:.2f})")
            print(f"  SSIM         : {m['SSIM']:.4f}")
            print(f"  SAM          : {m['SAM_deg']:.2f} deg")
            print(f"  ERGAS        : {m['ERGAS']:.2f}")
            print(f"  MAE          : {m['MAE']:.4f}\n")

    gain_psnr = results['HAT-Light']['PSNR_overall'] - results['Bicubic']['PSNR_overall']
    gain_ssim = results['HAT-Light']['SSIM'] - results['Bicubic']['SSIM']
    print(f"NET GAIN OVER BICUBIC: +{gain_psnr:.2f} dB PSNR | +{gain_ssim:.4f} SSIM")

    out_file = Path(__file__).resolve().parent / "curated_test_evaluation.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Saved results to {out_file}")


if __name__ == '__main__':
    evaluate()
