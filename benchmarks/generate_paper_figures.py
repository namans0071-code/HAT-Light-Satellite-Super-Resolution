import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from baselines.edsr import EDSR
from baselines.rcan import RCAN
from baselines.swinir_light import SwinIRLight
from core.hat_light import HATLightSR
from core.dataset import normalize_patch, stretch_channel, make_rgb_composite, make_cir_composite


def load_all_models(device):
    models = {}
    
    # EDSR
    edsr = EDSR(n_feats=64, n_resblocks=16).to(device)
    ckpt_e = torch.load("weights/edsr_best.pth", map_location=device, weights_only=False)
    edsr.load_state_dict(ckpt_e["model_state"])
    edsr.eval()
    models["EDSR"] = edsr
    
    # RCAN
    rcan = RCAN(n_feats=64, n_resgroups=4, n_rcab=4).to(device)
    ckpt_r = torch.load("weights/rcan_best.pth", map_location=device, weights_only=False)
    rcan.load_state_dict(ckpt_r["model_state"])
    rcan.eval()
    models["RCAN"] = rcan
    
    # SwinIR-Light
    swin = SwinIRLight(embed_dim=60, num_rstb=4, depth_per_rstb=4).to(device)
    ckpt_s = torch.load("weights/swinir_light_best.pth", map_location=device, weights_only=False)
    swin.load_state_dict(ckpt_s["model_state"])
    swin.eval()
    models["SwinIR-Light"] = swin
    
    # HAT-Light
    hat = HATLightSR().to(device)
    ckpt_h = torch.load("weights/best_model.pth", map_location=device, weights_only=False)
    hat.load_state_dict(ckpt_h["model_state"])
    hat.eval()
    models["HAT-Light"] = hat
    
    return models


def generate_figure1_comparisons(models, device):
    print("Generating Figure 1: Multi-Biome Visual Comparison...")
    hr_dir = Path("test_dataset/HR")
    lr_dir = Path("test_dataset/LR")
    
    # Select 4 representative patches from different biomes
    categories = ["airport", "city", "military", "vegetation"]
    selected_files = []
    for cat in categories:
        matches = sorted(list(hr_dir.glob(f"*{cat}*.npy")))
        if matches:
            selected_files.append(matches[len(matches) // 3]) # picked representative sample
            
    fig, axes = plt.subplots(len(selected_files), 7, figsize=(18, 2.6 * len(selected_files)), dpi=300)
    col_titles = ["(a) Low-Res (10m)", "(b) Bicubic", "(c) EDSR", "(d) RCAN", "(e) SwinIR-Light", "(f) HAT-Light (Ours)", "(g) Ground Truth (2.5m)"]
    
    with torch.no_grad():
        for row_idx, f_hr in enumerate(selected_files):
            arr_hr = np.load(f_hr)
            t_hr = torch.from_numpy(normalize_patch(arr_hr)).unsqueeze(0).to(device)
            
            f_lr = lr_dir / f_hr.name
            if f_lr.exists():
                arr_lr = np.load(f_lr)
                t_lr = torch.from_numpy(normalize_patch(arr_lr)).unsqueeze(0).to(device)
            else:
                t_lr = torch.nn.functional.interpolate(t_hr, size=(32, 32), mode="bicubic", align_corners=False, antialias=True)
                
            t_bic = torch.clamp(torch.nn.functional.interpolate(t_lr, size=(128, 128), mode="bicubic", align_corners=False), 0.0, 2.0)
            t_edsr = models["EDSR"](t_lr)
            t_rcan = models["RCAN"](t_lr)
            t_swin = models["SwinIR-Light"](t_lr)
            t_hat = models["HAT-Light"](t_lr)
            
            cat_name = f_hr.stem.split("_")[1].capitalize()
            
            # Convert to RGB images
            # Zoom crop into center 64x64 for crystal clear visual inspection of details
            def get_crop(tensor, is_lr=False):
                arr = tensor.squeeze(0).cpu().numpy()
                rgb = make_rgb_composite(arr)
                if is_lr:
                    return rgb[8:24, 8:24]
                else:
                    return rgb[32:96, 32:96]
                    
            crops = [
                get_crop(t_lr, is_lr=True),
                get_crop(t_bic),
                get_crop(t_edsr),
                get_crop(t_rcan),
                get_crop(t_swin),
                get_crop(t_hat),
                get_crop(t_hr)
            ]
            
            for col_idx in range(7):
                ax = axes[row_idx, col_idx]
                ax.imshow(crops[col_idx])
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(col_titles[col_idx], fontsize=11, fontweight="bold", pad=8)
                if col_idx == 0:
                    ax.text(-0.15, 0.5, f"{cat_name}\nBiome", transform=ax.transAxes, fontsize=11,
                            fontweight="bold", va="center", ha="right")
                            
    plt.tight_layout()
    out_path = Path("paper/figures/fig1_visual_comparison.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  -> Saved {out_path}")


def generate_figure2_transects(models, device):
    print("Generating Figure 2: Lineament 1D Transect Edge Profiles...")
    hr_dir = Path("test_dataset/HR")
    lr_dir = Path("test_dataset/LR")
    
    # Pick a high-contrast runway or road patch
    airport_patches = sorted(list(hr_dir.glob("*airport*.npy")))
    target_patch = airport_patches[len(airport_patches) // 2]
    
    arr_hr = np.load(target_patch)
    t_hr = torch.from_numpy(normalize_patch(arr_hr)).unsqueeze(0).to(device)
    f_lr = lr_dir / target_patch.name
    t_lr = torch.from_numpy(normalize_patch(np.load(f_lr))).unsqueeze(0).to(device)
    
    with torch.no_grad():
        t_bic = torch.clamp(torch.nn.functional.interpolate(t_lr, size=(128, 128), mode="bicubic", align_corners=False), 0.0, 2.0)
        t_edsr = models["EDSR"](t_lr)
        t_swin = models["SwinIR-Light"](t_lr)
        t_hat = models["HAT-Light"](t_lr)
        
    # Extract horizontal transect across high-contrast feature (row 64, red band)
    row = 64
    x_coords = np.arange(30, 95)
    
    prof_gt = t_hr[0, 0, row, 30:95].cpu().numpy()
    prof_bic = t_bic[0, 0, row, 30:95].cpu().numpy()
    prof_edsr = t_edsr[0, 0, row, 30:95].cpu().numpy()
    prof_swin = t_swin[0, 0, row, 30:95].cpu().numpy()
    prof_hat = t_hat[0, 0, row, 30:95].cpu().numpy()
    
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300, gridspec_kw={"width_ratios": [1, 1.4]})
    
    # Left: Patch crop with transect line
    crop_rgb = make_rgb_composite(t_hr[0].cpu().numpy())
    ax_img.imshow(crop_rgb)
    ax_img.axhline(y=row, xmin=30/128, xmax=95/128, color="yellow", linestyle="--", linewidth=2.5, label="Transect A-B")
    ax_img.scatter([30, 94], [row, row], color="red", s=40, zorder=5)
    ax_img.text(26, row - 4, "A", color="yellow", fontsize=12, fontweight="bold")
    ax_img.text(96, row - 4, "B", color="yellow", fontsize=12, fontweight="bold")
    ax_img.set_title("Airport Scene Transect (A -> B)", fontsize=11, fontweight="bold")
    ax_img.axis("off")
    ax_img.legend(loc="lower right")
    
    # Right: 1D Profile Plot
    ax_plot.plot(x_coords, prof_gt, label="Ground Truth (HR 2.5m)", color="black", linewidth=2.2)
    ax_plot.plot(x_coords, prof_hat, label="HAT-Light (Ours)", color="#007ACC", linewidth=2.0)
    ax_plot.plot(x_coords, prof_swin, label="SwinIR-Light", color="#FF9500", linewidth=1.5, linestyle="-.")
    ax_plot.plot(x_coords, prof_edsr, label="EDSR", color="#34C759", linewidth=1.5, linestyle=":")
    ax_plot.plot(x_coords, prof_bic, label="Bicubic Baseline", color="#FF3B30", linewidth=1.5, linestyle="--")
    
    ax_plot.set_xlabel("Spatial Coordinate along Transect (Pixels)", fontsize=11)
    ax_plot.set_ylabel("Normalized Surface Reflectance (Band 4 Red)", fontsize=11)
    ax_plot.set_title("Edge Transition Fidelity & Sharpness Profile", fontsize=11, fontweight="bold")
    ax_plot.grid(True, linestyle="--", alpha=0.5)
    ax_plot.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    out_path = Path("paper/figures/fig2_transect_edge_profile.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  -> Saved {out_path}")


def generate_figure3_ndvi(models, device):
    print("Generating Figure 3: Biophysical NDVI Scatter Correlation...")
    hr_dir = Path("test_dataset/HR")
    lr_dir = Path("test_dataset/LR")
    veg_patches = sorted(list(hr_dir.glob("*vegetation*.npy")))[:12]
    
    gt_ndvis, bic_ndvis, hat_ndvis = [], [], []
    
    with torch.no_grad():
        for p in veg_patches:
            arr_hr = np.load(p)
            t_hr = torch.from_numpy(normalize_patch(arr_hr)).unsqueeze(0).to(device)
            f_lr = lr_dir / p.name
            t_lr = torch.from_numpy(normalize_patch(np.load(f_lr))).unsqueeze(0).to(device)
            
            t_bic = torch.clamp(torch.nn.functional.interpolate(t_lr, size=(128, 128), mode="bicubic", align_corners=False), 0.0, 2.0)
            t_hat = models["HAT-Light"](t_lr)
            
            # NDVI = (NIR - Red) / (NIR + Red) = (Ch3 - Ch0) / (Ch3 + Ch0 + 1e-6)
            def calc_ndvi(t):
                nir = t[:, 3]
                red = t[:, 0]
                ndvi = (nir - red) / (nir + red + 1e-6)
                return ndvi.flatten().cpu().numpy()
                
            gt_ndvis.append(calc_ndvi(t_hr))
            bic_ndvis.append(calc_ndvi(t_bic))
            hat_ndvis.append(calc_ndvi(t_hat))
            
    gt_all = np.concatenate(gt_ndvis)
    bic_all = np.concatenate(bic_ndvis)
    hat_all = np.concatenate(hat_ndvis)
    
    # Subsample 4,000 points for clear visualization
    idx = np.random.choice(len(gt_all), 4000, replace=False)
    gt_sub = gt_all[idx]
    bic_sub = bic_all[idx]
    hat_sub = hat_all[idx]
    
    # Compute R2
    def get_r2(true, pred):
        corr = np.corrcoef(true, pred)[0, 1]
        return corr ** 2
        
    r2_bic = get_r2(gt_sub, bic_sub)
    r2_hat = get_r2(gt_sub, hat_sub)
    mae_bic = np.mean(np.abs(gt_sub - bic_sub))
    mae_hat = np.mean(np.abs(gt_sub - hat_sub))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)
    
    # Bicubic plot
    ax1.scatter(gt_sub, bic_sub, alpha=0.25, s=6, color="#FF3B30", label="Pixel Samples")
    ax1.plot([-0.2, 0.9], [-0.2, 0.9], "k--", linewidth=1.5, label="1:1 Ideal Identity")
    ax1.set_xlabel("Ground Truth NDVI (Native 10m L2A)", fontsize=11)
    ax1.set_ylabel("Reconstructed NDVI (Bicubic)", fontsize=11)
    ax1.set_title(f"Bicubic Baseline ($R^2 = {r2_bic:.3f}$, MAE = {mae_bic:.4f})", fontsize=11, fontweight="bold")
    ax1.set_xlim([-0.2, 0.9])
    ax1.set_ylim([-0.2, 0.9])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")
    
    # HAT-Light plot
    ax2.scatter(gt_sub, hat_sub, alpha=0.25, s=6, color="#007ACC", label="Pixel Samples")
    ax2.plot([-0.2, 0.9], [-0.2, 0.9], "k--", linewidth=1.5, label="1:1 Ideal Identity")
    ax2.set_xlabel("Ground Truth NDVI (Native 10m L2A)", fontsize=11)
    ax2.set_ylabel("Reconstructed NDVI (HAT-Light)", fontsize=11)
    ax2.set_title(f"HAT-Light Proposed ($R^2 = {r2_hat:.3f}$, MAE = {mae_hat:.4f})", fontsize=11, fontweight="bold")
    ax2.set_xlim([-0.2, 0.9])
    ax2.set_ylim([-0.2, 0.9])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    out_path = Path("paper/figures/fig3_ndvi_biophysical_fidelity.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  -> Saved {out_path} (HAT-Light R2: {r2_hat:.3f})")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Generating publication figures using device: {device}")
    models = load_all_models(device)
    
    generate_figure1_comparisons(models, device)
    generate_figure2_transects(models, device)
    generate_figure3_ndvi(models, device)
    print("\nAll publication figures generated successfully in paper/figures/!")


if __name__ == "__main__":
    main()
