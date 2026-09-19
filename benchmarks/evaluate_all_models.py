import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import json
import math
import torch
import numpy as np
from torch.utils.data import DataLoader

from baselines.edsr import EDSR
from baselines.rcan import RCAN
from baselines.swinir_light import SwinIRLight
from core.hat_light import HATLightSR
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
    print(f"Executing master comparative benchmark on: {device}")

    ds = SatelliteSRDataset("test_dataset", scale=4, is_train=False, augment=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    print(f"Evaluating across all {len(ds)} held-out curated test patches.\n")

    # Instantiate and load all models
    models = {}

    # 1. Bicubic (functional)
    models["Bicubic"] = {"model": None, "params": 0}

    # 2. EDSR
    edsr = EDSR(n_feats=64, n_resblocks=16).to(device)
    ckpt_e = torch.load("weights/edsr_best.pth", map_location=device, weights_only=False)
    edsr.load_state_dict(ckpt_e["model_state"])
    edsr.eval()
    models["EDSR"] = {"model": edsr, "params": sum(p.numel() for p in edsr.parameters())}

    # 3. RCAN
    rcan = RCAN(n_feats=64, n_resgroups=4, n_rcab=4).to(device)
    ckpt_r = torch.load("weights/rcan_best.pth", map_location=device, weights_only=False)
    rcan.load_state_dict(ckpt_r["model_state"])
    rcan.eval()
    models["RCAN"] = {"model": rcan, "params": sum(p.numel() for p in rcan.parameters())}

    # 4. SwinIR-Light
    swin = SwinIRLight(embed_dim=60, num_rstb=4, depth_per_rstb=4).to(device)
    ckpt_s = torch.load("weights/swinir_light_best.pth", map_location=device, weights_only=False)
    swin.load_state_dict(ckpt_s["model_state"])
    swin.eval()
    models["SwinIR-Light"] = {"model": swin, "params": sum(p.numel() for p in swin.parameters())}

    # 5. HAT-Light (Proposed)
    hat = HATLightSR().to(device)
    ckpt_h = torch.load("weights/best_model.pth", map_location=device, weights_only=False)
    hat.load_state_dict(ckpt_h["model_state"])
    hat.eval()
    models["HAT-Light (Ours)"] = {"model": hat, "params": sum(p.numel() for p in hat.parameters())}

    benchmark_results = {}

    for name, m_info in models.items():
        m = m_info["model"]
        param_count = m_info["params"]
        print(f"Evaluating {name} ({param_count / 1e6:.2f}M parameters)...")

        metrics = {
            "psnr": [], "ssim": [], "mae": [], "sam": [], "ergas": [],
            "b04": [], "b03": [], "b02": [], "b08": []
        }

        # Warmup GPU
        dummy = torch.randn(1, 4, 32, 32, device=device)
        with torch.no_grad():
            if m is not None:
                _ = m(dummy)
            torch.cuda.synchronize()

        start_time = time.time()
        with torch.no_grad():
            for lr, hr in loader:
                lr, hr = lr.to(device), hr.to(device)
                if m is None:
                    # Bicubic
                    sr = torch.clamp(
                        torch.nn.functional.interpolate(lr, size=(hr.shape[2], hr.shape[3]), mode="bicubic", align_corners=False),
                        0.0, 2.0
                    )
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

        torch.cuda.synchronize()
        total_time = time.time() - start_time
        fps = len(ds) / total_time
        latency_ms = (total_time / len(ds)) * 1000.0

        benchmark_results[name] = {
            "params_m": round(param_count / 1e6, 2),
            "psnr_overall": round(float(np.mean(metrics["psnr"])), 2),
            "psnr_red": round(float(np.mean(metrics["b04"])), 2),
            "psnr_green": round(float(np.mean(metrics["b03"])), 2),
            "psnr_blue": round(float(np.mean(metrics["b02"])), 2),
            "psnr_nir": round(float(np.mean(metrics["b08"])), 2),
            "ssim": round(float(np.mean(metrics["ssim"])), 4),
            "sam_deg": round(float(np.mean(metrics["sam"])), 2),
            "ergas": round(float(np.mean(metrics["ergas"])), 2),
            "mae": round(float(np.mean(metrics["mae"])), 4),
            "latency_ms": round(latency_ms, 2),
            "fps": round(fps, 1)
        }
        print(f"  -> PSNR: {benchmark_results[name]['psnr_overall']} dB | SSIM: {benchmark_results[name]['ssim']} | SAM: {benchmark_results[name]['sam_deg']} deg | Latency: {latency_ms:.1f}ms")

    # Save comparative JSON
    json_path = Path("benchmarks/comparative_results.json")
    json_path.write_text(json.dumps(benchmark_results, indent=2))
    print(f"\nSaved master results to {json_path}")

    # Generate Markdown Table
    md_lines = [
        "# Quantitative Benchmark Comparison on Held-Out Test Split (600 Curated Patches)\n",
        "| Architecture | Params (M) | PSNR Overall (dB) | Red (B04) | Green (B03) | Blue (B02) | NIR (B08) | SSIM | SAM (deg) | ERGAS | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for name, r in benchmark_results.items():
        is_ours = "Ours" in name
        p_str = f"**{r['psnr_overall']:.2f}**" if is_ours else f"{r['psnr_overall']:.2f}"
        s_str = f"**{r['ssim']:.4f}**" if is_ours else f"{r['ssim']:.4f}"
        sam_str = f"**{r['sam_deg']:.2f}**" if is_ours else f"{r['sam_deg']:.2f}"
        ergas_str = f"**{r['ergas']:.2f}**" if is_ours else f"{r['ergas']:.2f}"
        row = f"| **{name}** | {r['params_m']} | {p_str} | {r['psnr_red']} | {r['psnr_green']} | {r['psnr_blue']} | {r['psnr_nir']} | {s_str} | {sam_str} | {ergas_str} | {r['latency_ms']} |"
        md_lines.append(row)

    md_path = Path("benchmarks/table_comparative_results.md")
    md_path.write_text("\n".join(md_lines))
    print(f"Saved Markdown table to {md_path}")

    # Generate IEEEtran LaTeX Table
    tex_lines = [
        "% IEEEtran Double-Column Comparative Benchmark Table",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Quantitative benchmark results across the curated 600-patch Sentinel-2 test split ($10\\text{m} \\to 2.5\\text{m}$, Scale $s=4$). Best results in \\textbf{bold}.}",
        "\\label{tab:comparative_results}",
        "\\begin{tabular}{l c ccccc ccc c}",
        "\\hline",
        "\\textbf{Method} & \\textbf{Params} & \\multicolumn{5}{c}{\\textbf{PSNR (dB)} $\\uparrow$} & \\textbf{SSIM} $\\uparrow$ & \\textbf{SAM} ($^\\circ$) $\\downarrow$ & \\textbf{ERGAS} $\\downarrow$ & \\textbf{Latency} \\\\",
        "& (M) & All & Red (B4) & Green (B3) & Blue (B2) & NIR (B8) & & & & (ms) \\\\",
        "\\hline"
    ]
    for name, r in benchmark_results.items():
        is_ours = "Ours" in name
        b = lambda s: f"\\textbf{{{s}}}" if is_ours else str(s)
        tex_lines.append(
            f"{name} & {r['params_m']} & {b(r['psnr_overall'])} & {b(r['psnr_red'])} & {b(r['psnr_green'])} & {b(r['psnr_blue'])} & {b(r['psnr_nir'])} & {b(r['ssim'])} & {b(r['sam_deg'])} & {b(r['ergas'])} & {r['latency_ms']} \\\\"
        )
    tex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    tex_path = Path("benchmarks/table_comparative_results.tex")
    tex_path.write_text("\n".join(tex_lines))
    print(f"Saved LaTeX table to {tex_path}")


if __name__ == "__main__":
    main()
