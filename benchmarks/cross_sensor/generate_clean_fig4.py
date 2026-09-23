import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCHMARKS_DIR))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from core.hat_light import HATLightSR
from baselines.edsr import EDSR
from baselines.rcan import RCAN
from baselines.swinir_light import SwinIRLight
from core.losses import calculate_band_metrics

def normalize_rgb(patch: np.ndarray, p_low=None, p_high=None):
    """Normalized RGB using fixed reference percentiles."""
    r, g, b = patch[0].astype(np.float32), patch[1].astype(np.float32), patch[2].astype(np.float32)
    rgb = np.stack([r, g, b], axis=-1)
    if p_low is None or p_high is None:
        p_low = np.percentile(rgb, 2.0)
        p_high = np.percentile(rgb, 98.0)
    if p_high <= p_low:
        p_high = p_low + 1.0
    stretched = np.clip((rgb - p_low) / (p_high - p_low), 0.0, 1.0)
    return stretched, p_low, p_high

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Rendering publication Fig 4 on: {device}")

    # Load models
    hat = HATLightSR().to(device)
    hat.load_state_dict(torch.load("weights/best_model.pth", map_location=device, weights_only=False)["model_state"])
    hat.eval()

    edsr = EDSR(n_feats=64, n_resblocks=16).to(device)
    edsr.load_state_dict(torch.load("weights/edsr_best.pth", map_location=device, weights_only=False)["model_state"])
    edsr.eval()

    rcan = RCAN(n_feats=64, n_resgroups=4, n_rcab=4).to(device)
    rcan.load_state_dict(torch.load("weights/rcan_best.pth", map_location=device, weights_only=False)["model_state"])
    rcan.eval()

    swin = SwinIRLight(embed_dim=60, num_rstb=4, depth_per_rstb=4).to(device)
    swin.load_state_dict(torch.load("weights/swinir_light_best.pth", map_location=device, weights_only=False)["model_state"])
    swin.eval()

    cases = [
        {
            "cat": "Airport Terminal & Runways",
            "file": "patch_airport_runway_001.npy",
            "crop": (15, 20, 42, 42),
            "desc": "LAX Terminal Apron, Jetways & Aircraft Wings"
        },
        {
            "cat": "Urban Street Grid",
            "file": "patch_urban_grid_001.npy",
            "crop": (35, 35, 42, 42),
            "desc": "Downtown LA High-Density Blocks & Shadows"
        },
        {
            "cat": "Military Airbase",
            "file": "patch_military_airbase_001.npy",
            "crop": (25, 40, 42, 42),
            "desc": "MCAS Miramar Runway Asphalt & Blast Pad Lineaments"
        },
        {
            "cat": "Agricultural Canopy",
            "file": "patch_agriculture_canopy_001.npy",
            "crop": (18, 18, 42, 42),
            "desc": "San Joaquin Valley Crop Parcels & Tree Orchards"
        },
        {
            "cat": "Coastal Harbor",
            "file": "patch_harbor_coastal_001.npy",
            "crop": (25, 25, 42, 42),
            "desc": "Port of LA Pier 400 Intermodal Container Yard"
        }
    ]

    col_names = [
        "Degraded LR (10m)\n[Wald PSF Decimation]",
        "Bicubic\n[Baseline]",
        "EDSR\n[Lim et al.]",
        "RCAN\n[Zhang et al.]",
        "SwinIR-Light\n[Liang et al.]",
        "HAT-Light (Ours)\n[Continuous FiLM]",
        "Ground Truth (2.5m)\n[Native USGS NAIP]"
    ]

    fig, axes = plt.subplots(len(cases), 7, figsize=(18, 13.5), dpi=300)
    plt.subplots_adjust(wspace=0.03, hspace=0.18, left=0.05, right=0.98, top=0.94, bottom=0.02)

    for row_idx, case in enumerate(cases):
        hr_path = Path("cross_sensor/data/HR") / case["file"]
        lr_path = Path("cross_sensor/data/LR") / case["file"]

        hr_np = np.load(hr_path).astype(np.float32) / 10000.0
        lr_np = np.load(lr_path).astype(np.float32) / 10000.0

        hr_t = torch.from_numpy(hr_np).unsqueeze(0).to(device)
        lr_t = torch.from_numpy(lr_np).unsqueeze(0).to(device)

        with torch.no_grad():
            bic_t = torch.clamp(F.interpolate(lr_t, size=(128, 128), mode="bicubic", align_corners=False), 0, 2)
            edsr_t = edsr(lr_t)
            rcan_t = rcan(lr_t)
            swin_t = swin(lr_t)
            hat_t = hat(lr_t)

        # Ground truth reference stretch
        gt_rgb, p_low, p_high = normalize_rgb(hr_np)

        preds = {
            "LR": F.interpolate(lr_t, size=(128, 128), mode="nearest")[0].cpu().numpy(),
            "Bicubic": bic_t[0].cpu().numpy(),
            "EDSR": edsr_t[0].cpu().numpy(),
            "RCAN": rcan_t[0].cpu().numpy(),
            "SwinIR-Light": swin_t[0].cpu().numpy(),
            "HAT-Light (Ours)": hat_t[0].cpu().numpy(),
            "Ground Truth": hr_np
        }

        def get_metrics(pred_t):
            p = 10.0 * np.log10(1.5**2 / max(1e-10, torch.mean((pred_t - hr_t)**2).item()))
            s = calculate_band_metrics(pred_t, hr_t)["ssim"]
            return p, s

        m_bic = get_metrics(bic_t)
        m_edsr = get_metrics(edsr_t)
        m_rcan = get_metrics(rcan_t)
        m_swin = get_metrics(swin_t)
        m_hat = get_metrics(hat_t)

        metrics_map = {
            "LR": "10m GSD",
            "Bicubic": f"{m_bic[0]:.2f} dB / {m_bic[1]:.4f}",
            "EDSR": f"{m_edsr[0]:.2f} dB / {m_edsr[1]:.4f}",
            "RCAN": f"{m_rcan[0]:.2f} dB / {m_rcan[1]:.4f}",
            "SwinIR-Light": f"{m_swin[0]:.2f} dB / {m_swin[1]:.4f}",
            "HAT-Light (Ours)": f"{m_hat[0]:.2f} dB / {m_hat[1]:.4f}",
            "Ground Truth": "Reference (2.5m)"
        }

        cy, cx, ch, cw = case["crop"]

        for col_idx, (m_key, col_title) in enumerate(zip(preds.keys(), col_names)):
            ax = axes[row_idx, col_idx]
            img_np = preds[m_key]
            rgb, _, _ = normalize_rgb(img_np, p_low, p_high)

            ax.imshow(rgb)
            ax.set_xticks([])
            ax.set_yticks([])

            # Highlight bounding box for zoom inset
            rect = patches.Rectangle((cx, cy), cw, ch, linewidth=1.5, edgecolor="yellow", facecolor="none", linestyle="--")
            ax.add_patch(rect)

            # Inset zoom in bottom-right corner
            crop_patch = rgb[cy:cy+ch, cx:cx+cw]
            ax_ins = inset_axes(ax, width="40%", height="40%", loc="lower right", borderpad=0.2)
            ax_ins.imshow(crop_patch)
            ax_ins.set_xticks([])
            ax_ins.set_yticks([])
            for spine in ax_ins.spines.values():
                spine.set_edgecolor("yellow")
                spine.set_linewidth(1.5)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=10, fontweight="bold", pad=8)

            score_text = metrics_map[m_key]
            is_hat = "HAT" in m_key
            ax.text(
                0.5, -0.07, score_text,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=8.5,
                fontweight="bold" if is_hat else "normal",
                color="#006600" if is_hat else "black"
            )

        axes[row_idx, 0].set_ylabel(
            f"{case['cat']}\n({case['desc'][:22]}...)",
            fontsize=9.5,
            fontweight="bold",
            labelpad=10
        )

    out_fig = REPO_ROOT / "assets" / "fig4_cross_sensor_eval.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated clean 5-category publication Figure 4: {out_fig}")

if __name__ == "__main__":
    main()
