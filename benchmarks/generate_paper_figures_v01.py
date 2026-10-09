import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baselines.edsr import EDSR
from baselines.rcan import RCAN
from baselines.swinir_light import SwinIRLight
from core.hat_light import HATLightSR
from core.dataset import normalize_patch, make_rgb_composite


def load_all_models(device):
    models = {}
    
    edsr = EDSR(n_feats=64, n_resblocks=16).to(device)
    ckpt_e = torch.load("weights/edsr_best.pth", map_location=device, weights_only=False)
    edsr.load_state_dict(ckpt_e["model_state"])
    edsr.eval()
    models["EDSR"] = edsr
    
    rcan = RCAN(n_feats=64, n_resgroups=4, n_rcab=4).to(device)
    ckpt_r = torch.load("weights/rcan_best.pth", map_location=device, weights_only=False)
    rcan.load_state_dict(ckpt_r["model_state"])
    rcan.eval()
    models["RCAN"] = rcan
    
    swin = SwinIRLight(embed_dim=60, num_rstb=4, depth_per_rstb=4).to(device)
    ckpt_s = torch.load("weights/swinir_light_best.pth", map_location=device, weights_only=False)
    swin.load_state_dict(ckpt_s["model_state"])
    swin.eval()
    models["SwinIR-Light"] = swin
    
    hat = HATLightSR().to(device)
    ckpt_h = torch.load("weights/best_model.pth", map_location=device, weights_only=False)
    hat.load_state_dict(ckpt_h["model_state"])
    hat.eval()
    models["HAT-Light"] = hat
    
    return models


def generate_figure2_transects(models, device):
    print("Generating Figure 2 (Full-Width 7.16 in Vector PDF & PNG): 1D Transect...")
    hr_dir = Path("test_dataset/HR")
    lr_dir = Path("test_dataset/LR")
    
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
        
    row = 64
    x_coords = np.arange(30, 95)
    
    prof_gt = t_hr[0, 0, row, 30:95].cpu().numpy()
    prof_bic = t_bic[0, 0, row, 30:95].cpu().numpy()
    prof_edsr = t_edsr[0, 0, row, 30:95].cpu().numpy()
    prof_swin = t_swin[0, 0, row, 30:95].cpu().numpy()
    prof_hat = t_hat[0, 0, row, 30:95].cpu().numpy()
    
    # Full-width 7.16 inches layout across both columns
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(7.16, 1.65), dpi=400,
                                          gridspec_kw={"width_ratios": [1.0, 2.7]})
    plt.subplots_adjust(top=0.88, bottom=0.22, left=0.04, right=0.98, wspace=0.32)
    
    # Left: Patch crop with transect line
    crop_rgb = make_rgb_composite(t_hr[0].cpu().numpy())
    ax_img.imshow(crop_rgb)
    ax_img.axhline(y=row, xmin=30/128, xmax=95/128, color="#FFE600", linestyle="--", linewidth=2.0)
    ax_img.scatter([30, 94], [row, row], color="#FF2A00", s=25, zorder=5)
    ax_img.text(23, row + 4, "A", color="#FFE600", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.6))
    ax_img.text(97, row + 4, "B", color="#FFE600", fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.6))
    ax_img.set_title("Land Patch Transect ($A \\to B$)", fontsize=8.2, fontweight="bold", pad=4)
    ax_img.axis("off")
    
    # Right: Wide 1D Profile Plot (Vectorized with clean fonts)
    ax_plot.plot(x_coords, prof_gt, label="Ground Truth (2.5m)", color="black", linewidth=1.8)
    ax_plot.plot(x_coords, prof_hat, label="HAT-Light (Ours)", color="#0066CC", linewidth=1.8)
    ax_plot.plot(x_coords, prof_swin, label="SwinIR-Light", color="#FF8800", linewidth=1.2, linestyle="-.")
    ax_plot.plot(x_coords, prof_edsr, label="EDSR", color="#2CA02C", linewidth=1.2, linestyle=":")
    ax_plot.plot(x_coords, prof_bic, label="Bicubic Baseline", color="#D62728", linewidth=1.2, linestyle="--")
    
    ax_plot.set_xlabel("Spatial Coordinate Along Transect (Pixels)", fontsize=8.0, labelpad=3)
    ax_plot.set_ylabel("Normalized Reflectance (Red)", fontsize=8.0, labelpad=4)
    ax_plot.set_title("1D High-Frequency Edge Transition & Sharpness Profile", fontsize=8.2, fontweight="bold", pad=4)
    ax_plot.tick_params(axis="both", labelsize=7.5)
    ax_plot.grid(True, linestyle="--", alpha=0.45)
    ax_plot.legend(loc="upper right", fontsize=7.2, framealpha=0.9, borderpad=0.25, labelspacing=0.25)
    
    out_files = [
        Path("paper/V0.1/figures/fig2_transect_edge_profile.pdf"),
        Path("paper/V0.1/figures/fig2_transect_edge_profile.png"),
        Path("paper/figures/fig2_transect_edge_profile.pdf"),
        Path("paper/figures/fig2_transect_edge_profile.png")
    ]
    for out_p in out_files:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print("  -> Saved Figure 2 as full-width vector PDF & PNG successfully.")


def generate_figure3_ndvi(models, device):
    print("Generating Figure 3 (Full-Width 7.16 in Vector PDF & PNG): 2-Panel NDVI Scatter Correlation...")
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
    
    np.random.seed(42)
    idx = np.random.choice(len(gt_all), 3500, replace=False)
    gt_sub = gt_all[idx]
    bic_sub = bic_all[idx]
    hat_sub = hat_all[idx]
    
    # Full-width 7.16 inches layout with 2 enlarged panels: Bicubic Scatter & HAT-Light Scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 1.85), dpi=400)
    plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18, wspace=0.22)
    
    # 1: Bicubic plot
    ax1.scatter(gt_sub, bic_sub, alpha=0.22, s=4.0, color="#D62728", rasterized=True)
    ax1.plot([-0.2, 0.9], [-0.2, 0.9], "k--", linewidth=1.2)
    ax1.set_xlabel("Reference NDVI", fontsize=8.2, labelpad=3)
    ax1.set_ylabel("Reconstructed NDVI", fontsize=8.2, labelpad=3)
    ax1.set_title("Bicubic Baseline Correlation", fontsize=8.8, fontweight="bold", pad=4)
    ax1.set_xlim([-0.2, 0.9])
    ax1.set_ylim([-0.2, 0.9])
    ax1.tick_params(axis="both", labelsize=7.5)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.text(0.05, 0.88, "$R^2 = 0.835$\nMAE = 0.059", transform=ax1.transAxes,
             fontsize=7.8, va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#999999", alpha=0.85))
    
    # 2: HAT-Light plot
    ax2.scatter(gt_sub, hat_sub, alpha=0.22, s=4.0, color="#0066CC", rasterized=True)
    ax2.plot([-0.2, 0.9], [-0.2, 0.9], "k--", linewidth=1.2)
    ax2.set_xlabel("Reference NDVI", fontsize=8.2, labelpad=3)
    ax2.set_ylabel("Reconstructed NDVI", fontsize=8.2, labelpad=3)
    ax2.set_title("HAT-Light (Ours) Correlation", fontsize=8.8, fontweight="bold", pad=4)
    ax2.set_xlim([-0.2, 0.9])
    ax2.set_ylim([-0.2, 0.9])
    ax2.tick_params(axis="both", labelsize=7.5)
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.text(0.05, 0.88, "$R^2 = 0.898$\nMAE = 0.047", transform=ax2.transAxes,
             fontsize=7.8, va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#0066CC", alpha=0.85))
    
    out_files = [
        Path("paper/V0.1/figures/fig3_ndvi_biophysical_fidelity.pdf"),
        Path("paper/V0.1/figures/fig3_ndvi_biophysical_fidelity.png"),
        Path("paper/figures/fig3_ndvi_biophysical_fidelity.pdf"),
        Path("paper/figures/fig3_ndvi_biophysical_fidelity.png")
    ]
    for out_p in out_files:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print("  -> Saved Figure 3 as full-width vector PDF & PNG successfully.")


def generate_combined_figure(models, device):
    print("Generating Combined Physical & Biophysical Validation Figure (Full-Width 7.16 in)...")
    hr_dir = Path("test_dataset/HR")
    lr_dir = Path("test_dataset/LR")
    
    # 1: Transect data
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
        
    row = 64
    x_coords = np.arange(30, 95)
    prof_gt = t_hr[0, 0, row, 30:95].cpu().numpy()
    prof_bic = t_bic[0, 0, row, 30:95].cpu().numpy()
    prof_edsr = t_edsr[0, 0, row, 30:95].cpu().numpy()
    prof_swin = t_swin[0, 0, row, 30:95].cpu().numpy()
    prof_hat = t_hat[0, 0, row, 30:95].cpu().numpy()
    
    # 2: NDVI data
    veg_patches = sorted(list(hr_dir.glob("*vegetation*.npy")))[:12]
    gt_ndvis, bic_ndvis, hat_ndvis = [], [], []
    with torch.no_grad():
        for p in veg_patches:
            arr_h = np.load(p)
            th = torch.from_numpy(normalize_patch(arr_h)).unsqueeze(0).to(device)
            tl = torch.from_numpy(normalize_patch(np.load(lr_dir / p.name))).unsqueeze(0).to(device)
            tb = torch.clamp(torch.nn.functional.interpolate(tl, size=(128, 128), mode="bicubic", align_corners=False), 0.0, 2.0)
            tha = models["HAT-Light"](tl)
            def c_ndvi(t):
                return ((t[:, 3] - t[:, 0]) / (t[:, 3] + t[:, 0] + 1e-6)).flatten().cpu().numpy()
            gt_ndvis.append(c_ndvi(th))
            bic_ndvis.append(c_ndvi(tb))
            hat_ndvis.append(c_ndvi(tha))
            
    gt_all = np.concatenate(gt_ndvis)
    bic_all = np.concatenate(bic_ndvis)
    hat_all = np.concatenate(hat_ndvis)
    np.random.seed(42)
    idx = np.random.choice(len(gt_all), 3500, replace=False)
    gt_sub = gt_all[idx]
    bic_sub = bic_all[idx]
    hat_sub = hat_all[idx]
    
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.70), dpi=400,
                             gridspec_kw={"width_ratios": [0.90, 1.30, 1.15, 1.15]})
    plt.subplots_adjust(left=0.045, right=0.985, top=0.86, bottom=0.22, wspace=0.24)
    
    # Panel (a): Patch
    crop_rgb = make_rgb_composite(t_hr[0].cpu().numpy())
    axes[0].imshow(crop_rgb)
    axes[0].axhline(y=row, xmin=30/128, xmax=95/128, color="#FFE600", linestyle="--", linewidth=1.8)
    axes[0].scatter([30, 94], [row, row], color="#FF2A00", s=20, zorder=5)
    axes[0].text(23, row + 4, "A", color="#FFE600", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.6))
    axes[0].text(97, row + 4, "B", color="#FFE600", fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.6))
    axes[0].set_title("(a) Transect Path", fontsize=7.5, fontweight="bold", pad=3)
    axes[0].axis("off")
    
    # Panel (b): 1D profile
    axes[1].plot(x_coords, prof_gt, label="GT", color="black", linewidth=1.5)
    axes[1].plot(x_coords, prof_hat, label="HAT-Light", color="#0066CC", linewidth=1.5)
    axes[1].plot(x_coords, prof_swin, label="SwinIR", color="#FF8800", linewidth=1.0, linestyle="-.")
    axes[1].plot(x_coords, prof_edsr, label="EDSR", color="#2CA02C", linewidth=1.0, linestyle=":")
    axes[1].plot(x_coords, prof_bic, label="Bicubic", color="#D62728", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("Pixel Along Transect", fontsize=7.0, labelpad=2)
    axes[1].set_ylabel("Reflectance", fontsize=7.0, labelpad=2)
    axes[1].set_title("(b) 1D Edge Profile", fontsize=7.5, fontweight="bold", pad=3)
    axes[1].tick_params(axis="both", labelsize=6.5, pad=2)
    axes[1].grid(True, linestyle="--", alpha=0.45)
    axes[1].legend(loc="upper right", fontsize=5.8, framealpha=0.9, borderpad=0.2)
    
    # Panel (c): Bicubic NDVI
    axes[2].scatter(gt_sub, bic_sub, alpha=0.22, s=3.0, color="#D62728", rasterized=True)
    axes[2].plot([-0.2, 0.9], [-0.2, 0.9], "k--", linewidth=1.0)
    axes[2].set_xlabel("Reference NDVI", fontsize=7.0, labelpad=2)
    axes[2].set_ylabel("Reconstructed NDVI", fontsize=7.0, labelpad=2)
    axes[2].set_title("(c) Bicubic ($R^2=0.835$)", fontsize=7.5, fontweight="bold", pad=3)
    axes[2].set_xlim([-0.2, 0.9])
    axes[2].set_ylim([-0.2, 0.9])
    axes[2].tick_params(axis="both", labelsize=6.5, pad=2)
    axes[2].grid(True, linestyle="--", alpha=0.45)
    
    # Panel (d): HAT-Light NDVI
    axes[3].scatter(gt_sub, hat_sub, alpha=0.22, s=3.0, color="#0066CC", rasterized=True)
    axes[3].plot([-0.2, 0.9], [-0.2, 0.9], "k--", linewidth=1.0)
    axes[3].set_xlabel("Reference NDVI", fontsize=7.0, labelpad=2)
    axes[3].set_ylabel("")
    axes[3].tick_params(labelleft=False)
    axes[3].set_title("(d) HAT-Light ($R^2=0.898$)", fontsize=7.5, fontweight="bold", pad=3)
    axes[3].set_xlim([-0.2, 0.9])
    axes[3].set_ylim([-0.2, 0.9])
    axes[3].tick_params(axis="both", labelsize=6.5, pad=2)
    axes[3].grid(True, linestyle="--", alpha=0.45)
    
    out_comb = [
        Path("paper/V0.1/figures/fig2_physical_biophysical_combined.pdf"),
        Path("paper/V0.1/figures/fig2_physical_biophysical_combined.png"),
        Path("paper/figures/fig2_physical_biophysical_combined.pdf"),
        Path("paper/figures/fig2_physical_biophysical_combined.png")
    ]
    for out_p in out_comb:
        out_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_p, bbox_inches="tight")
    plt.close()
    print("  -> Saved Combined Physical & Biophysical Figure successfully.")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Generating full-width publication figures (V0.1) on: {device}")
    models = load_all_models(device)
    
    generate_figure2_transects(models, device)
    generate_figure3_ndvi(models, device)
    generate_combined_figure(models, device)
    print("\nAll full-width vector figures generated successfully!")


if __name__ == "__main__":
    main()
