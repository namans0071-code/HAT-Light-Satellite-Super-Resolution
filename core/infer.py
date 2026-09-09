"""
Inference and Demo Engine for Lightweight Hybrid Attention Transformer (HAT-Light).
Supports:
- Evaluation on test patches with quantitative metrics (PSNR, SSIM, MAE)
- Continuous scale factor conditioning (2x, 2.5x, 3x, 4x, 8x)
- Seamless Sliding-Window Tiling Inference with 2D Hann window blending (zero seams)
- 0-Shot Mode: Direct upscaling of native 10m Sentinel-2 patches
- True Color (RGB: B04, B03, B02) and Color Infrared (CIR: B08, B04, B03) composite rendering
- Base64 encoding for instantaneous real-time Web GUI slider and side-by-side rendering
"""

import io
import os
import sys
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from core.hat_light import HATLightSR, check_cuda_device
from core.dataset import normalize_patch, denormalize_patch, make_rgb_composite, make_cir_composite
from core.losses import calculate_band_metrics

try:
    import rasterio
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def array_to_base64_png(img_uint8: np.ndarray) -> str:
    """Convert (H, W, 3) uint8 numpy array to base64 data URI string."""
    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG", optimize=True)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def make_residual_heatmap(tensor1: torch.Tensor, tensor2: torch.Tensor, boost: float = 6.0) -> np.ndarray:
    """Compute high-frequency spatial difference heatmap between two tensors."""
    import matplotlib.cm as cm
    diff = torch.abs(tensor1 - tensor2).mean(dim=1)[0].cpu().numpy()
    diff_norm = np.clip(diff * boost, 0.0, 1.0)
    heatmap = (cm.turbo(diff_norm)[:, :, :3] * 255.0).astype(np.uint8)
    return heatmap


def get_2d_blend_window(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Generate 2D Hann blending window for seamless sliding-window stitching."""
    if h <= 2 or w <= 2:
        return torch.ones((1, 1, h, w), device=device, dtype=torch.float32)
    win_h = torch.hann_window(h, periodic=False, device=device, dtype=torch.float32).unsqueeze(1)
    win_w = torch.hann_window(w, periodic=False, device=device, dtype=torch.float32).unsqueeze(0)
    win_2d = torch.clamp(win_h * win_w, min=1e-3)
    return win_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)


def tiled_inference(
    model: torch.nn.Module,
    lr_tensor: torch.Tensor,
    scale: float = 4.0,
    tile_size: int = 64,
    overlap: int = 16,
    device: Optional[torch.device] = None,
    progress_callback: Optional[Any] = None
) -> torch.Tensor:
    """
    Seamless Sliding-Window Tiling Inference for large satellite scenes and images.
    Uses 2D Hann window blending to eliminate all boundary seams and edge discontinuities.

    Args:
        model: Super-resolution model (HATLightSR)
        lr_tensor: (B, C, H, W) normalized input tensor
        scale: continuous super-resolution scale factor
        tile_size: size of input tile (e.g. 64 or 128)
        overlap: pixel overlap between adjacent tiles (e.g. 16 or 32)
        device: target torch device
        progress_callback: optional callback(current_tile, total_tiles)
    Returns:
        sr_tensor: (B, C, H_out, W_out) seamlessly stitched output
    """
    if device is None:
        device = lr_tensor.device

    B, C, H, W = lr_tensor.shape
    H_out = int(round(H * scale))
    W_out = int(round(W * scale))

    # If image is smaller than or equal to tile size, run direct forward pass
    if H <= tile_size and W <= tile_size:
        with torch.no_grad():
            res = model(lr_tensor, scale=scale)
            if progress_callback:
                progress_callback(1, 1)
            return res

    accum_device = device
    # For exceptionally massive images, accumulate on CPU to prevent CUDA OOM
    if H_out * W_out > 5000 * 5000:
        accum_device = torch.device("cpu")

    stride = max(1, tile_size - overlap)
    output = torch.zeros((B, C, H_out, W_out), device=accum_device, dtype=torch.float32)
    weight_map = torch.zeros((1, 1, H_out, W_out), device=accum_device, dtype=torch.float32)

    # Calculate grid coordinates
    y_steps = list(range(0, max(1, H - tile_size + 1), stride))
    if y_steps[-1] + tile_size < H:
        y_steps.append(H - tile_size)
    x_steps = list(range(0, max(1, W - tile_size + 1), stride))
    if x_steps[-1] + tile_size < W:
        x_steps.append(W - tile_size)

    total_tiles = len(y_steps) * len(x_steps)
    tile_idx = 0

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                lr_tile = lr_tensor[:, :, y : min(y + tile_size, H), x : min(x + tile_size, W)].to(device)
                sr_tile = model(lr_tile, scale=scale)

                out_y = int(round(y * scale))
                out_x = int(round(x * scale))
                tile_out_h = sr_tile.shape[2]
                tile_out_w = sr_tile.shape[3]

                blend_w = get_2d_blend_window(tile_out_h, tile_out_w, device)

                tile_weighted = (sr_tile * blend_w).to(accum_device)
                blend_w_accum = blend_w.to(accum_device)

                output[:, :, out_y : out_y + tile_out_h, out_x : out_x + tile_out_w] += tile_weighted
                weight_map[:, :, out_y : out_y + tile_out_h, out_x : out_x + tile_out_w] += blend_w_accum

                tile_idx += 1
                if progress_callback:
                    progress_callback(tile_idx, total_tiles)

    output = output / torch.clamp(weight_map, min=1e-5)
    return torch.clamp(output, 0.0, 2.0).to(device)


def read_image_or_geotiff(input_path: Union[str, Path]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Reads an image from disk supporting JPEG, PNG, and GeoTIFF (.tif, .tiff).
    Returns:
        tensor: (1, 4, H, W) normalized float32 tensor
        meta: dictionary containing original metadata, channels, georeferencing, etc.
    """
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ext = p.suffix.lower()
    is_geotiff = ext in [".tif", ".tiff", ".geotiff"]

    if is_geotiff:
        if HAS_RASTERIO:
            with rasterio.open(str(p)) as src:
                data = src.read()  # (C, H, W)
                profile = src.profile.copy()
                crs = src.crs
                transform = src.transform
                nodata = src.nodata
                orig_dtype = data.dtype
                orig_count = src.count
                h, w = src.height, src.width
        else:
            import tifffile
            data = tifffile.imread(str(p))
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            elif data.ndim == 3 and data.shape[-1] in [1, 3, 4]:
                data = np.transpose(data, (2, 0, 1))
            orig_dtype = data.dtype
            orig_count = data.shape[0]
            h, w = data.shape[1], data.shape[2]
            profile = {"driver": "GTiff", "count": orig_count, "height": h, "width": w, "dtype": str(orig_dtype)}
            crs = None
            transform = None
            nodata = None

        # Format channel count to 4 channels for HAT-Light
        if orig_count == 1:
            data_4ch = np.repeat(data, 4, axis=0)
        elif orig_count == 3:
            # R, G, B -> synthesize NIR as (R + G) / 2
            r, g, b = data[0], data[1], data[2]
            nir = (r.astype(np.float32) * 0.5 + g.astype(np.float32) * 0.5).astype(orig_dtype)
            data_4ch = np.stack([r, g, b, nir], axis=0)
        elif orig_count == 4:
            data_4ch = data[:4]
        else:
            data_4ch = data[:4]

        # Normalization
        if orig_dtype == np.uint8:
            norm_data = (data_4ch.astype(np.float32) / 255.0).clip(0.0, 2.0)
        elif orig_dtype == np.uint16:
            max_val = 10000.0 if data_4ch.max() > 255 else 255.0
            norm_data = (data_4ch.astype(np.float32) / max_val).clip(0.0, 2.0)
        else:
            max_v = max(1.0, float(data_4ch.max()))
            norm_data = (data_4ch.astype(np.float32) / max_v).clip(0.0, 2.0)

        meta = {
            "format": "GEOTIFF",
            "extension": ext,
            "orig_shape": [int(h), int(w)],
            "orig_count": orig_count,
            "orig_dtype": str(orig_dtype),
            "orig_mode": f"{orig_count}Band",
            "alpha_channel": None,
            "is_geotiff": True,
            "profile": profile,
            "crs": crs,
            "transform": transform,
            "nodata": nodata,
        }
    else:
        pil_img = Image.open(str(p))
        orig_format = (pil_img.format or ext.replace(".", "")).upper()
        orig_mode = pil_img.mode
        arr = np.array(pil_img)
        orig_dtype = arr.dtype
        alpha_channel = None

        if orig_mode == "RGBA" or (arr.ndim == 3 and arr.shape[2] == 4):
            orig_count = 4
            h, w, _ = arr.shape
            r, g, b, alpha_channel = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
            nir = (r.astype(np.float32) * 0.5 + g.astype(np.float32) * 0.5).astype(orig_dtype)
            data_4ch = np.stack([r, g, b, nir], axis=0)
        elif orig_mode == "RGB" or (arr.ndim == 3 and arr.shape[2] == 3):
            orig_count = 3
            h, w, _ = arr.shape
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            nir = (r.astype(np.float32) * 0.5 + g.astype(np.float32) * 0.5).astype(orig_dtype)
            data_4ch = np.stack([r, g, b, nir], axis=0)
        elif orig_mode in ["L", "1"] or arr.ndim == 2:
            orig_count = 1
            h, w = arr.shape
            data_4ch = np.stack([arr, arr, arr, arr], axis=0)
        else:
            arr_rgb = np.array(pil_img.convert("RGB"))
            orig_count = 3
            h, w, _ = arr_rgb.shape
            r, g, b = arr_rgb[:, :, 0], arr_rgb[:, :, 1], arr_rgb[:, :, 2]
            nir = (r.astype(np.float32) * 0.5 + g.astype(np.float32) * 0.5).astype(arr_rgb.dtype)
            data_4ch = np.stack([r, g, b, nir], axis=0)

        if orig_dtype == np.uint8 or orig_dtype == bool:
            norm_data = (data_4ch.astype(np.float32) / 255.0).clip(0.0, 2.0)
        elif orig_dtype == np.uint16:
            max_val = 10000.0 if data_4ch.max() > 255 else 255.0
            norm_data = (data_4ch.astype(np.float32) / max_val).clip(0.0, 2.0)
        else:
            norm_data = (data_4ch.astype(np.float32) / 255.0).clip(0.0, 2.0)

        meta = {
            "format": orig_format,
            "extension": ext,
            "orig_shape": [int(h), int(w)],
            "orig_count": orig_count,
            "orig_dtype": str(orig_dtype),
            "orig_mode": orig_mode,
            "alpha_channel": alpha_channel,
            "is_geotiff": False,
            "profile": None,
            "crs": None,
            "transform": None,
            "nodata": None,
        }

    tensor = torch.from_numpy(norm_data).unsqueeze(0).float()
    return tensor, meta


def write_image_or_geotiff(
    output_path: Union[str, Path],
    sr_tensor: torch.Tensor,
    meta: Dict[str, Any],
    scale: float
) -> str:
    """
    Writes super-resolved tensor to output_path matching original format
    and preserving geospatial CRS/Affine transformation if GeoTIFF.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    out_h = sr_tensor.shape[2]
    out_w = sr_tensor.shape[3]
    orig_dtype_str = meta.get("orig_dtype", "uint8")
    is_geotiff = meta.get("is_geotiff", False) or out_p.suffix.lower() in [".tif", ".tiff", ".geotiff"]

    # Denormalize
    if "uint16" in orig_dtype_str:
        max_val = 10000.0
        sr_arr = (torch.clamp(sr_tensor[0], 0.0, 6.5535).cpu().numpy() * max_val).round().astype(np.uint16)
    else:
        sr_arr = (torch.clamp(sr_tensor[0], 0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)

    if is_geotiff and HAS_RASTERIO:
        profile = meta.get("profile") or {}
        new_profile = profile.copy()
        orig_count = meta.get("orig_count", 3)
        export_count = min(orig_count, sr_arr.shape[0])

        new_profile.update({
            "driver": "GTiff",
            "height": out_h,
            "width": out_w,
            "count": export_count,
            "dtype": "uint16" if "uint16" in orig_dtype_str else "uint8"
        })

        if meta.get("crs"):
            new_profile["crs"] = meta["crs"]
        if meta.get("nodata") is not None:
            new_profile["nodata"] = meta["nodata"]

        transform = meta.get("transform")
        if transform:
            new_transform = Affine(
                transform.a / scale,
                transform.b,
                transform.c,
                transform.d,
                transform.e / scale,
                transform.f
            )
            new_profile["transform"] = new_transform

        with rasterio.open(str(out_p), "w", **new_profile) as dst:
            for b in range(export_count):
                dst.write(sr_arr[b], b + 1)

    elif is_geotiff and not HAS_RASTERIO:
        import tifffile
        orig_count = meta.get("orig_count", 3)
        export_count = min(orig_count, sr_arr.shape[0])
        tif_data = sr_arr[:export_count]
        tifffile.imwrite(str(out_p), tif_data)

    else:
        orig_mode = meta.get("orig_mode", "RGB")
        alpha_channel = meta.get("alpha_channel")
        orig_count = meta.get("orig_count", 3)

        if orig_mode == "RGBA" and alpha_channel is not None:
            alpha_t = torch.from_numpy(alpha_channel).unsqueeze(0).unsqueeze(0).float()
            alpha_up = F.interpolate(alpha_t, size=(out_h, out_w), mode="bicubic", align_corners=False)
            alpha_up_arr = torch.clamp(alpha_up[0, 0], 0.0, 255.0).byte().cpu().numpy()
            rgb = np.transpose(sr_arr[:3], (1, 2, 0))
            rgba = np.dstack([rgb, alpha_up_arr])
            out_img = Image.fromarray(rgba, mode="RGBA")
        elif orig_count == 1:
            out_img = Image.fromarray(sr_arr[0], mode="L")
        else:
            rgb = np.transpose(sr_arr[:3], (1, 2, 0))
            out_img = Image.fromarray(rgb, mode="RGB")

        ext = out_p.suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            if out_img.mode != "RGB":
                out_img = out_img.convert("RGB")
            out_img.save(str(out_p), format="JPEG", quality=95)
        elif ext == ".png":
            out_img.save(str(out_p), format="PNG", optimize=True)
        else:
            out_img.save(str(out_p))

    return str(out_p.resolve())


class SRInferEngine:
    """
    Inference and Demo Engine for HAT-Light (Lightweight Hybrid Attention Transformer).
    """
    def __init__(self):
        self.device = check_cuda_device()
        self.current_model: Optional[HATLightSR] = None
        self.current_model_path: Optional[str] = None
        self.current_scale: float = 4.0

    def load_model(self, checkpoint_path: Optional[str], scale: float = 4.0) -> bool:
        """Loads model weights onto the CUDA GPU, or initializes clean model if path doesn't exist."""
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"[InferEngine Warning] Checkpoint '{checkpoint_path}' not found on disk. Initializing clean HAT-Light weights.")
            self.current_model = HATLightSR(
                in_channels=4,
                out_channels=4,
                scale=scale,
                embed_dim=60,
                num_rhag=4,
                num_hab_per_rhag=4,
                num_heads=6,
                window_size=8
            ).to(self.device)
            self.current_model.eval()
            self.current_model_path = None
            self.current_scale = scale
            return False

        print(f"[InferEngine] Loading checkpoint: {checkpoint_path} on {self.device}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        model_scale = float(ckpt.get("scale", scale)) if isinstance(ckpt, dict) else float(scale)
        self.current_scale = model_scale

        state_dict = ckpt["model_state"] if (isinstance(ckpt, dict) and "model_state" in ckpt) else ckpt
        embed_dim = 60
        num_rhag = 4
        num_heads = 6
        if isinstance(state_dict, dict):
            if "input_layer.0.weight" in state_dict:
                embed_dim = state_dict["input_layer.0.weight"].shape[0]
            rhag_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith("rhags.") and k.split('.')[1].isdigit()]
            if rhag_indices:
                num_rhag = max(rhag_indices) + 1
            if embed_dim % 8 == 0:
                num_heads = 8
            elif embed_dim % 6 == 0:
                num_heads = 6

        model = HATLightSR(
            in_channels=4,
            out_channels=4,
            scale=model_scale,
            embed_dim=embed_dim,
            num_rhag=num_rhag,
            num_hab_per_rhag=4,
            num_heads=num_heads,
            window_size=8
        ).to(self.device)

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        self.current_model = model
        self.current_model_path = checkpoint_path
        return True

    def run_patch_inference(
        self,
        patch_path: str,
        checkpoint_path: Optional[str] = None,
        scale: float = 4.0,
        use_tiling: bool = False,
        tile_size: int = 64,
        overlap: int = 16
    ) -> Dict[str, Any]:
        """
        Runs standard super-resolution inference on a test patch.
        Simulates LR (downsampling) if LR file is not directly present,
        then runs the model and computes metrics against ground truth HR.
        """
        scale = float(scale)
        if checkpoint_path and (self.current_model_path != checkpoint_path or self.current_model is None):
            self.load_model(checkpoint_path, scale=scale)

        if self.current_model is None:
            print("[InferEngine] Initializing default HATLightSR instance.")
            self.current_model = HATLightSR(in_channels=4, out_channels=4, scale=scale).to(self.device)
            self.current_model.eval()
            self.current_scale = scale

        patch_file = Path(patch_path)
        if not patch_file.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_path}")

        # Load HR array
        hr_arr = np.load(patch_file)  # (4, H, W) uint16
        hr_norm = normalize_patch(hr_arr)
        hr_tensor = torch.from_numpy(hr_norm).unsqueeze(0).to(self.device)  # (1, 4, H, W)

        # Generate LR (e.g. 128 -> 32)
        _, _, h, w = hr_tensor.shape
        lr_h, lr_w = int(round(h / scale)), int(round(w / scale))
        with torch.no_grad():
            lr_tensor = F.interpolate(
                hr_tensor,
                size=(lr_h, lr_w),
                mode="bicubic",
                align_corners=False,
                antialias=True
            )
            lr_tensor = torch.clamp(lr_tensor, 0.0, 2.0)

            # Measure inference latency
            t0 = time.time()
            if use_tiling:
                sr_tensor = tiled_inference(
                    self.current_model,
                    lr_tensor,
                    scale=scale,
                    tile_size=tile_size,
                    overlap=overlap,
                    device=self.device
                )
            else:
                sr_tensor = self.current_model(lr_tensor, scale=scale)
            torch.cuda.synchronize()
            latency_ms = (time.time() - t0) * 1000.0

            # Bicubic baseline upsampling for comparison slider
            lr_bicubic = torch.clamp(
                F.interpolate(lr_tensor, size=(h, w), mode="bicubic", align_corners=False),
                0.0,
                2.0
            )

        # Compute evaluation metrics
        metrics = calculate_band_metrics(sr_tensor, hr_tensor)
        bicubic_metrics = calculate_band_metrics(lr_bicubic, hr_tensor)

        # Convert tensors to uint16 numpy arrays
        lr_bicubic_arr = denormalize_patch(lr_bicubic[0])
        sr_arr = denormalize_patch(sr_tensor[0])

        # Generate RGB composites
        lr_rgb = make_rgb_composite(lr_bicubic_arr)
        sr_rgb = make_rgb_composite(sr_arr)
        hr_rgb = make_rgb_composite(hr_arr)

        # Generate CIR composites
        lr_cir = make_cir_composite(lr_bicubic_arr)
        sr_cir = make_cir_composite(sr_arr)
        hr_cir = make_cir_composite(hr_arr)

        # Generate high-frequency residual difference heatmaps
        diff_heatmap = make_residual_heatmap(sr_tensor, lr_bicubic, boost=8.0)
        error_heatmap = make_residual_heatmap(sr_tensor, hr_tensor, boost=8.0)

        param_count = sum(p.numel() for p in self.current_model.parameters())
        arch_name = "HAT-Sat-Pro (Hybrid Attention Transformer)" if param_count > 3_000_000 else "HAT-Light (Hybrid Attention Transformer)"

        return {
            "patch_name": patch_file.name,
            "architecture": arch_name,
            "model_parameters": f"{param_count / 1e6:.2f}M",
            "scale": scale,
            "use_tiling": use_tiling,
            "tile_size": tile_size,
            "overlap": overlap,
            "latency_ms": round(latency_ms, 2),
            "lr_shape": [int(lr_h), int(lr_w)],
            "hr_shape": [int(h), int(w)],
            "metrics": {
                "psnr": round(metrics["psnr"], 2),
                "ssim": round(metrics["ssim"], 4),
                "mae": round(metrics["mae"], 4),
                "psnr_rgb": round(metrics["psnr_rgb"], 2),
                "ssim_rgb": round(metrics["ssim_rgb"], 4),
                "psnr_nir": round(metrics["psnr_nir"], 2),
                "ssim_nir": round(metrics["ssim_nir"], 4),
                "bicubic_psnr": round(bicubic_metrics["psnr"], 2),
                "bicubic_ssim": round(bicubic_metrics["ssim"], 4),
                "psnr_gain": round(metrics["psnr"] - bicubic_metrics["psnr"], 2),
            },
            "images": {
                "lr_rgb": array_to_base64_png(lr_rgb),
                "sr_rgb": array_to_base64_png(sr_rgb),
                "hr_rgb": array_to_base64_png(hr_rgb),
                "lr_cir": array_to_base64_png(lr_cir),
                "sr_cir": array_to_base64_png(sr_cir),
                "hr_cir": array_to_base64_png(hr_cir),
                "diff_heatmap": array_to_base64_png(diff_heatmap),
                "error_heatmap": array_to_base64_png(error_heatmap),
            }
        }

    def run_zero_shot_inference(
        self,
        patch_path: str,
        checkpoint_path: Optional[str] = None,
        scale: float = 4.0,
        use_tiling: bool = False,
        tile_size: int = 64,
        overlap: int = 16
    ) -> Dict[str, Any]:
        """
        0-Shot Super-Resolution Mode:
        Takes a native 10m Sentinel-2 patch directly as input and super-resolves it dynamically.
        """
        scale = float(scale)
        if checkpoint_path and (self.current_model_path != checkpoint_path or self.current_model is None):
            self.load_model(checkpoint_path, scale=scale)

        if self.current_model is None:
            self.current_model = HATLightSR(in_channels=4, out_channels=4, scale=scale).to(self.device)
            self.current_model.eval()
            self.current_scale = scale

        patch_file = Path(patch_path)
        arr_10m = np.load(patch_file)  # (4, H, W) uint16
        norm_10m = normalize_patch(arr_10m)
        tensor_10m = torch.from_numpy(norm_10m).unsqueeze(0).to(self.device)

        _, _, h, w = tensor_10m.shape
        out_h, out_w = int(round(h * scale)), int(round(w * scale))

        with torch.no_grad():
            t0 = time.time()
            if use_tiling:
                sr_tensor = tiled_inference(
                    self.current_model,
                    tensor_10m,
                    scale=scale,
                    tile_size=tile_size,
                    overlap=overlap,
                    device=self.device
                )
            else:
                sr_tensor = self.current_model(tensor_10m, scale=scale)
            torch.cuda.synchronize()
            latency_ms = (time.time() - t0) * 1000.0

            # Interpolate native 10m to match output dimension for slider comparison
            input_upscaled = torch.clamp(
                F.interpolate(tensor_10m, size=(out_h, out_w), mode="bicubic", align_corners=False),
                0.0,
                2.0
            )

        input_arr = denormalize_patch(input_upscaled[0])
        sr_arr = denormalize_patch(sr_tensor[0])

        input_rgb = make_rgb_composite(input_arr)
        sr_rgb = make_rgb_composite(sr_arr)
        input_cir = make_cir_composite(input_arr)
        sr_cir = make_cir_composite(sr_arr)

        simulated_gsd = round(10.0 / scale, 2)
        diff_heatmap = make_residual_heatmap(sr_tensor, input_upscaled, boost=8.0)

        param_count = sum(p.numel() for p in self.current_model.parameters())
        arch_name = "HAT-Sat-Pro (Hybrid Attention Transformer)" if param_count > 3_000_000 else "HAT-Light (Hybrid Attention Transformer)"

        return {
            "patch_name": patch_file.name,
            "mode": "0-shot",
            "architecture": arch_name,
            "model_parameters": f"{param_count / 1e6:.2f}M",
            "scale": scale,
            "use_tiling": use_tiling,
            "tile_size": tile_size,
            "overlap": overlap,
            "input_resolution": "10m GSD (Native Sentinel-2)",
            "output_resolution": f"~{simulated_gsd}m GSD (0-Shot Super-Resolved)",
            "input_shape": [int(h), int(w)],
            "output_shape": [int(out_h), int(out_w)],
            "latency_ms": round(latency_ms, 2),
            "images": {
                "before_rgb": array_to_base64_png(input_rgb),
                "after_rgb": array_to_base64_png(sr_rgb),
                "before_cir": array_to_base64_png(input_cir),
                "after_cir": array_to_base64_png(sr_cir),
                "diff_heatmap": array_to_base64_png(diff_heatmap),
            }
        }

    def run_custom_file_inference(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        scale: float = 4.0,
        use_tiling: bool = True,
        tile_size: int = 128,
        overlap: int = 32
    ) -> Dict[str, Any]:
        """
        Runs full super-resolution inference on any JPEG, PNG, or GeoTIFF file.
        Exports the result to output_path in the identical data format.
        """
        scale = float(scale)
        if checkpoint_path and (self.current_model_path != checkpoint_path or self.current_model is None):
            self.load_model(checkpoint_path, scale=scale)

        if self.current_model is None:
            self.current_model = HATLightSR(in_channels=4, out_channels=4, scale=scale).to(self.device)
            self.current_model.eval()
            self.current_scale = scale

        # Read input image/geotiff
        tensor_lr, meta = read_image_or_geotiff(input_path)
        tensor_lr = tensor_lr.to(self.device)
        _, _, h, w = tensor_lr.shape
        out_h, out_w = int(round(h * scale)), int(round(w * scale))

        # Determine output file path
        p_in = Path(input_path)
        if not output_path or not output_path.strip():
            output_path = str(p_in.parent / f"{p_in.stem}_SR_{scale}x{p_in.suffix}")
        else:
            p_out = Path(output_path.strip())
            if p_out.is_dir() or output_path.endswith("/") or output_path.endswith("\\"):
                output_path = str(p_out / f"{p_in.stem}_SR_{scale}x{p_in.suffix}")
            elif not p_out.suffix:
                output_path = f"{output_path}{p_in.suffix}"

        # Execute inference
        t0 = time.time()
        with torch.no_grad():
            if use_tiling:
                sr_tensor = tiled_inference(
                    self.current_model,
                    tensor_lr,
                    scale=scale,
                    tile_size=tile_size,
                    overlap=overlap,
                    device=self.device
                )
            else:
                sr_tensor = self.current_model(tensor_lr, scale=scale)
        torch.cuda.synchronize()
        latency_ms = (time.time() - t0) * 1000.0

        # Export result to disk in the same data format
        saved_file = write_image_or_geotiff(output_path, sr_tensor, meta, scale=scale)
        file_size_mb = round(Path(saved_file).stat().st_size / (1024 * 1024), 2)

        # Generate lightweight base64 previews for GUI
        max_preview_dim = 1600
        preview_scale = min(1.0, max_preview_dim / max(out_h, out_w))

        with torch.no_grad():
            # Bicubic upsampling of input for side-by-side / slider comparison
            input_upscaled = torch.clamp(
                F.interpolate(tensor_lr, size=(out_h, out_w), mode="bicubic", align_corners=False),
                0.0,
                2.0
            )

            if preview_scale < 1.0:
                prev_h, prev_w = int(round(out_h * preview_scale)), int(round(out_w * preview_scale))
                p_sr = F.interpolate(sr_tensor, size=(prev_h, prev_w), mode="bilinear", align_corners=False)
                p_lr = F.interpolate(input_upscaled, size=(prev_h, prev_w), mode="bilinear", align_corners=False)
            else:
                p_sr = sr_tensor
                p_lr = input_upscaled

        # Convert to displayable arrays
        if "uint16" in meta.get("orig_dtype", ""):
            lr_disp = denormalize_patch(p_lr[0])
            sr_disp = denormalize_patch(p_sr[0])
            lr_rgb = make_rgb_composite(lr_disp)
            sr_rgb = make_rgb_composite(sr_disp)
            has_cir = meta.get("orig_count", 0) >= 4
            lr_cir = make_cir_composite(lr_disp) if has_cir else None
            sr_cir = make_cir_composite(sr_disp) if has_cir else None
        else:
            # 8-bit RGB
            lr_arr_8 = torch.clamp(p_lr[0, :3] * 255.0, 0, 255).byte().permute(1, 2, 0).cpu().numpy()
            sr_arr_8 = torch.clamp(p_sr[0, :3] * 255.0, 0, 255).byte().permute(1, 2, 0).cpu().numpy()
            lr_rgb = lr_arr_8
            sr_rgb = sr_arr_8
            lr_cir = None
            sr_cir = None

        diff_heatmap = make_residual_heatmap(p_sr, p_lr, boost=8.0)

        param_count = sum(p.numel() for p in self.current_model.parameters())
        arch_name = "HAT-Sat-Pro (Hybrid Attention Transformer)" if param_count > 3_000_000 else "HAT-Light (Hybrid Attention Transformer)"

        images_dict = {
            "before_rgb": array_to_base64_png(lr_rgb),
            "after_rgb": array_to_base64_png(sr_rgb),
            "diff_heatmap": array_to_base64_png(diff_heatmap),
        }
        if lr_cir is not None and sr_cir is not None:
            images_dict["before_cir"] = array_to_base64_png(lr_cir)
            images_dict["after_cir"] = array_to_base64_png(sr_cir)

        crs_desc = str(meta["crs"]) if meta.get("crs") else "Standard Non-Georeferenced"

        return {
            "success": True,
            "filename": p_in.name,
            "input_path": str(p_in.resolve()).replace("\\", "/"),
            "output_path": str(Path(saved_file).resolve()).replace("\\", "/"),
            "format": meta["format"],
            "extension": meta["extension"],
            "is_geotiff": meta["is_geotiff"],
            "crs": crs_desc,
            "architecture": arch_name,
            "model_parameters": f"{param_count / 1e6:.2f}M",
            "scale": scale,
            "use_tiling": use_tiling,
            "tile_size": tile_size,
            "overlap": overlap,
            "input_shape": [int(h), int(w)],
            "output_shape": [int(out_h), int(out_w)],
            "latency_ms": round(latency_ms, 2),
            "file_size_mb": file_size_mb,
            "images": images_dict
        }


# Global inference singleton
infer_engine = SRInferEngine()
