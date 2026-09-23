import sys
from pathlib import Path

CROSS_SENSOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = CROSS_SENSOR_DIR.parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCHMARKS_DIR))

import json
import numpy as np
import torch
import math
from torch.utils.data import DataLoader
from core.hat_light import HATLightSR
from baselines.edsr import EDSR
from baselines.rcan import RCAN
from baselines.swinir_light import SwinIRLight
from core.dataset import SatelliteSRDataset
from core.losses import calculate_band_metrics

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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running Wald Protocol Cross-Sensor Evaluation on: {device}")

    # Load 100 genuine 2.5m patches
    data_dir = CROSS_SENSOR_DIR / "data"
    weights_dir = REPO_ROOT / "weights"
    if not data_dir.exists():
        print(f"Warning: {data_dir} does not exist. Run build_naip_cross_sensor_dataset.py first.")
        return

    ds = SatelliteSRDataset(data_dir, scale=4, is_train=False, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    print(f"Loaded {len(ds)} patches from USGS NAIP 2.5m reference dataset.\n")

    models = {}
    models["Bicubic Baseline"] = {"model": None, "params": 0}

    # 2. EDSR (Optional Baseline)
    edsr_path = weights_dir / "edsr_best.pth"
    if edsr_path.exists():
        edsr = EDSR(n_feats=64, n_resblocks=16).to(device)
        edsr.load_state_dict(torch.load(str(edsr_path), map_location=device, weights_only=False)["model_state"])
        edsr.eval()
        models["EDSR"] = {"model": edsr, "params": sum(p.numel() for p in edsr.parameters())}
    else:
        print("Note: 'weights/edsr_best.pth' not found (optional baseline). Skipping EDSR.")

    # 3. RCAN (Optional Baseline)
    rcan_path = weights_dir / "rcan_best.pth"
    if rcan_path.exists():
        rcan = RCAN(n_feats=64, n_resgroups=4, n_rcab=4).to(device)
        rcan.load_state_dict(torch.load(str(rcan_path), map_location=device, weights_only=False)["model_state"])
        rcan.eval()
        models["RCAN"] = {"model": rcan, "params": sum(p.numel() for p in rcan.parameters())}
    else:
        print("Note: 'weights/rcan_best.pth' not found (optional baseline). Skipping RCAN.")

    # 4. SwinIR-Light (Optional Baseline)
    swin_path = weights_dir / "swinir_light_best.pth"
    if swin_path.exists():
        swin = SwinIRLight(embed_dim=60, num_rstb=4, depth_per_rstb=4).to(device)
        swin.load_state_dict(torch.load(str(swin_path), map_location=device, weights_only=False)["model_state"])
        swin.eval()
        models["SwinIR-Light"] = {"model": swin, "params": sum(p.numel() for p in swin.parameters())}
    else:
        print("Note: 'weights/swinir_light_best.pth' not found (optional baseline). Skipping SwinIR-Light.")

    # 5. HAT-Light (Proposed - Primary Model)
    hat_path = weights_dir / "best_model.pth"
    if hat_path.exists():
        hat = HATLightSR().to(device)
        hat.load_state_dict(torch.load(str(hat_path), map_location=device, weights_only=False)["model_state"])
        hat.eval()
        models["HAT-Light (Ours)"] = {"model": hat, "params": sum(p.numel() for p in hat.parameters())}
    else:
        print("Error: Primary model checkpoint 'weights/best_model.pth' not found!")

    categories = ['airport_runway', 'urban_grid', 'military_airbase', 'agriculture_canopy', 'harbor_coastal']
    results = {}

    for name, m_info in models.items():
        m = m_info["model"]
        metrics = {
            "psnr": [], "ssim": [], "mae": [], "sam": [], "ergas": [],
            "b04": [], "b03": [], "b02": [], "b08": []
        }
        cat_psnr = {c: [] for c in categories}
        cat_ssim = {c: [] for c in categories}

        with torch.no_grad():
            for lr, hr in loader:
                lr, hr = lr.to(device), hr.to(device)
                if m is None:
                    sr = torch.clamp(torch.nn.functional.interpolate(lr, size=(hr.shape[2], hr.shape[3]), mode="bicubic", align_corners=False), 0.0, 2.0)
                else:
                    sr = m(lr)

                p_all, p_04, p_03, p_02, p_08 = calc_band_psnr(sr, hr)
                m_ssim = calculate_band_metrics(sr, hr)["ssim"]
                mae = torch.mean(torch.abs(sr - hr)).item()
                sam = compute_sam(sr, hr)
                ergas = compute_ergas(sr, hr)

                metrics["psnr"].append(p_all)
                metrics["ssim"].append(m_ssim)
                metrics["mae"].append(mae)
                metrics["sam"].append(sam)
                metrics["ergas"].append(ergas)
                metrics["b04"].append(p_04)
                metrics["b03"].append(p_03)
                metrics["b02"].append(p_02)
                metrics["b08"].append(p_08)

        results[name] = {
            "psnr_overall": round(float(np.mean(metrics["psnr"])), 2),
            "psnr_red": round(float(np.mean(metrics["b04"])), 2),
            "psnr_green": round(float(np.mean(metrics["b03"])), 2),
            "psnr_blue": round(float(np.mean(metrics["b02"])), 2),
            "psnr_nir": round(float(np.mean(metrics["b08"])), 2),
            "ssim": round(float(np.mean(metrics["ssim"])), 4),
            "sam_deg": round(float(np.mean(metrics["sam"])), 2),
            "ergas": round(float(np.mean(metrics["ergas"])), 2),
            "mae": round(float(np.mean(metrics["mae"])), 4)
        }
        print(f"[{name:18s}] PSNR: {results[name]['psnr_overall']} dB | SSIM: {results[name]['ssim']} | SAM: {results[name]['sam_deg']} deg | ERGAS: {results[name]['ergas']}")

    # Save master JSON
    out_json = CROSS_SENSOR_DIR / "cross_sensor_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved cross_sensor_results.json to {out_json}")

    # Generate Markdown Table
    md_lines = [
        "# Zero-Shot Cross-Sensor Generalization Benchmark (USGS NAIP 2.5m Ground Truth)\n",
        "| Architecture | PSNR Overall (dB) | Red (B04) | Green (B03) | Blue (B02) | NIR (B08) | SSIM | SAM (deg) | ERGAS | Net Gain vs Bicubic |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    bic_psnr = results["Bicubic Baseline"]["psnr_overall"]
    for name, r in results.items():
        is_ours = "Ours" in name
        p_str = f"**{r['psnr_overall']:.2f}**" if is_ours else f"{r['psnr_overall']:.2f}"
        s_str = f"**{r['ssim']:.4f}**" if is_ours else f"{r['ssim']:.4f}"
        sam_str = f"**{r['sam_deg']:.2f}**" if is_ours else f"{r['sam_deg']:.2f}"
        ergas_str = f"**{r['ergas']:.2f}**" if is_ours else f"{r['ergas']:.2f}"
        gain = r["psnr_overall"] - bic_psnr
        gain_str = f"**+{gain:.2f} dB**" if is_ours else f"+{gain:.2f} dB" if gain > 0 else "0.00 dB"
        md_lines.append(f"| **{name}** | {p_str} | {r['psnr_red']} | {r['psnr_green']} | {r['psnr_blue']} | {r['psnr_nir']} | {s_str} | {sam_str} | {ergas_str} | {gain_str} |")

    out_md = CROSS_SENSOR_DIR / "table_cross_sensor.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved Markdown table to {out_md}")


if __name__ == "__main__":
    main()
