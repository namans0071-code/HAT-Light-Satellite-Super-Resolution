"""
Multi-Format Satellite & Geospatial Image I/O Module.
Supports:
- GeoTIFF (.tif, .tiff) with CRS & Affine geotransform preservation and scaling.
- NumPy arrays (.npy, .npz) for 4-channel raw uint16 satellite reflectance patches.
- Standard images (PNG, JPEG, WebP, BMP, JP2).
- Input ingestion from either file paths or raw bytes in-memory.
"""

import io
import os
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import torch
from PIL import Image

try:
    import rasterio
    from rasterio.transform import Affine
    from rasterio.io import MemoryFile
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def stretch_channel(band: np.ndarray, lower_pct: float = 2.0, upper_pct: float = 98.0) -> np.ndarray:
    """Normalize 2D band to 0-255 uint8 using percentile contrast stretch."""
    if band.ndim > 2:
        band = band.squeeze()
    p_low, p_high = np.percentile(band, (lower_pct, upper_pct))
    if p_high <= p_low:
        p_high = p_low + 1.0
    stretched = np.clip((band - p_low) / (p_high - p_low), 0.0, 1.0)
    return (stretched * 255.0).astype(np.uint8)


def array_to_base64_png(img_uint8: np.ndarray) -> str:
    """Convert (H, W, 3) uint8 numpy array to base64 data URI string."""
    pil_img = Image.fromarray(img_uint8)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG", optimize=True)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def make_rgb_composite(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    True Color RGB composite.
    Sentinel-2: Ch0 = Red (B04), Ch1 = Green (B03), Ch2 = Blue (B02).
    Returns (H, W, 3) uint8 array.
    """
    if isinstance(data, torch.Tensor):
        data = data.squeeze(0).detach().cpu().numpy()
    if data.ndim == 4:
        data = data[0]

    r = stretch_channel(data[0])
    g = stretch_channel(data[1])
    b = stretch_channel(data[2])
    return np.stack([r, g, b], axis=-1)


def make_cir_composite(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Color Infrared (CIR / NIR) False Color composite.
    Sentinel-2: Ch3 = NIR (B08), Ch0 = Red (B04), Ch1 = Green (B03).
    Returns (H, W, 3) uint8 array.
    """
    if isinstance(data, torch.Tensor):
        data = data.squeeze(0).detach().cpu().numpy()
    if data.ndim == 4:
        data = data[0]

    nir = stretch_channel(data[3]) if data.shape[0] >= 4 else stretch_channel(data[0])
    r = stretch_channel(data[0])
    g = stretch_channel(data[1])
    return np.stack([nir, r, g], axis=-1)


def make_difference_heatmap(
    lr_upscaled: Union[np.ndarray, torch.Tensor],
    sr_output: Union[np.ndarray, torch.Tensor],
    boost: float = 8.0
) -> np.ndarray:
    """
    Generate high-frequency residual difference heatmap between bicubic baseline and SR.
    Uses vivid Turbo colormap mapping for crisp visualization.
    """
    if isinstance(lr_upscaled, torch.Tensor):
        lr_upscaled = lr_upscaled.squeeze(0).detach().cpu().numpy()
    if isinstance(sr_output, torch.Tensor):
        sr_output = sr_output.squeeze(0).detach().cpu().numpy()

    diff = np.mean(np.abs(sr_output.astype(np.float32) - lr_upscaled.astype(np.float32)), axis=0)
    p99 = np.percentile(diff, 99.0)
    norm_diff = np.clip((diff / max(1e-5, p99)) * (boost / 8.0), 0.0, 1.0)

    h, w = norm_diff.shape
    heatmap = np.zeros((h, w, 3), dtype=np.uint8)
    heatmap[:, :, 0] = np.clip((norm_diff * 2.0 - 0.5) * 255.0, 0, 255).astype(np.uint8)
    heatmap[:, :, 1] = np.clip(np.sin(norm_diff * np.pi) * 255.0, 0, 255).astype(np.uint8)
    heatmap[:, :, 2] = np.clip((1.0 - norm_diff * 2.0) * 255.0, 0, 255).astype(np.uint8)
    return heatmap


def read_multi_format_file(
    file_input: Union[str, Path, bytes],
    filename: Optional[str] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Loads any geospatial or standard image file from either disk path or in-memory bytes.
    Standardizes channel count to 4 (Red, Green, Blue, NIR) and normalizes to float32 [0.0, 2.0].

    Returns:
        tensor: (1, 4, H, W) normalized float32 tensor
        meta: dictionary preserving original format, bit-depth, shape, CRS, and geotransform
    """
    is_bytes = isinstance(file_input, (bytes, bytearray))
    if not is_bytes:
        p = Path(file_input)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        file_name = p.name
        ext = p.suffix.lower()
        source_path = str(p.resolve())
    else:
        file_name = filename or "uploaded_image.png"
        ext = Path(file_name).suffix.lower()
        source_path = "<memory>"

    meta: Dict[str, Any] = {
        "file_name": file_name,
        "source_path": source_path,
        "extension": ext,
        "format": ext.replace(".", "").upper(),
        "is_geotiff": False,
        "profile": None,
        "crs": None,
        "transform": None,
        "orig_shape": None,
        "orig_channels": 4,
        "orig_dtype": "uint16"
    }

    # 1. NumPy Array (.npy, .npz)
    if ext in [".npy", ".npz"]:
        if is_bytes:
            bio = io.BytesIO(file_input)
            if ext == ".npz":
                data_dict = np.load(bio)
                arr = data_dict[list(data_dict.keys())[0]]
            else:
                arr = np.load(bio)
        else:
            if ext == ".npz":
                data_dict = np.load(str(p))
                arr = data_dict[list(data_dict.keys())[0]]
            else:
                arr = np.load(str(p))

        meta["orig_dtype"] = str(arr.dtype)
        if arr.ndim == 2:
            arr = np.stack([arr] * 4, axis=0)
        elif arr.ndim == 3:
            if arr.shape[0] in [1, 3, 4]:
                pass
            elif arr.shape[-1] in [1, 3, 4]:
                arr = np.transpose(arr, (2, 0, 1))

        c, h, w = arr.shape
        meta["orig_shape"] = [int(h), int(w)]
        meta["orig_channels"] = c

        if c == 1:
            arr = np.repeat(arr, 4, axis=0)
        elif c == 3:
            nir = ((arr[0].astype(np.float32) + arr[1].astype(np.float32)) / 2.0).astype(arr.dtype)
            arr = np.concatenate([arr, nir[np.newaxis, ...]], axis=0)
        elif c > 4:
            arr = arr[:4]

        scale_divisor = 10000.0 if ("uint16" in meta["orig_dtype"] or arr.max() > 255) else 255.0
        arr_norm = np.clip(arr[:4].astype(np.float32) / scale_divisor, 0.0, 2.0)
        tensor = torch.from_numpy(arr_norm).unsqueeze(0).float()
        return tensor, meta

    # 2. GeoTIFF (.tif, .tiff, .geotiff)
    elif ext in [".tif", ".tiff", ".geotiff"]:
        meta["is_geotiff"] = True
        meta["format"] = "GEOTIFF"

        if HAS_RASTERIO:
            if is_bytes:
                with MemoryFile(file_input) as memfile:
                    with memfile.open() as src:
                        data = src.read()
                        meta["profile"] = src.profile.copy()
                        meta["crs"] = str(src.crs) if src.crs else None
                        meta["transform"] = list(src.transform) if src.transform else None
                        meta["orig_dtype"] = str(data.dtype)
                        meta["orig_shape"] = [int(src.height), int(src.width)]
                        meta["orig_channels"] = int(src.count)
            else:
                with rasterio.open(str(p)) as src:
                    data = src.read()
                    meta["profile"] = src.profile.copy()
                    meta["crs"] = str(src.crs) if src.crs else None
                    meta["transform"] = list(src.transform) if src.transform else None
                    meta["orig_dtype"] = str(data.dtype)
                    meta["orig_shape"] = [int(src.height), int(src.width)]
                    meta["orig_channels"] = int(src.count)
        elif HAS_TIFFFILE:
            bio = io.BytesIO(file_input) if is_bytes else str(p)
            data = tifffile.imread(bio)
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            elif data.ndim == 3 and data.shape[-1] in [1, 3, 4]:
                data = np.transpose(data, (2, 0, 1))
            meta["orig_dtype"] = str(data.dtype)
            meta["orig_shape"] = [int(data.shape[1]), int(data.shape[2])]
            meta["orig_channels"] = int(data.shape[0])
        else:
            bio = io.BytesIO(file_input) if is_bytes else str(p)
            img = Image.open(bio)
            data = np.array(img)
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            elif data.ndim == 3:
                data = np.transpose(data, (2, 0, 1))
            meta["orig_dtype"] = str(data.dtype)
            meta["orig_shape"] = [int(data.shape[1]), int(data.shape[2])]
            meta["orig_channels"] = int(data.shape[0])

        c, h, w = data.shape
        if c == 1:
            data = np.concatenate([data] * 4, axis=0)
        elif c == 3:
            nir = ((data[0].astype(np.float32) + data[1].astype(np.float32)) / 2.0).astype(data.dtype)
            data = np.concatenate([data, nir[np.newaxis, ...]], axis=0)
        elif c > 4:
            data = data[:4]

        scale_divisor = 10000.0 if ("uint16" in meta["orig_dtype"] or data.max() > 255) else 255.0
        data_norm = np.clip(data[:4].astype(np.float32) / scale_divisor, 0.0, 2.0)
        tensor = torch.from_numpy(data_norm).unsqueeze(0).float()
        return tensor, meta

    # 3. Standard Images (PNG, JPEG, WebP, BMP, JP2)
    else:
        meta["format"] = ext.replace(".", "").upper()
        meta["orig_dtype"] = "uint8"
        bio = io.BytesIO(file_input) if is_bytes else str(p)
        img = Image.open(bio).convert("RGB")
        arr = np.array(img)
        h, w, c = arr.shape
        meta["orig_shape"] = [int(h), int(w)]
        meta["orig_channels"] = c

        arr_trans = np.transpose(arr, (2, 0, 1))
        nir = ((arr_trans[0].astype(np.float32) + arr_trans[1].astype(np.float32)) / 2.0).astype(np.uint8)
        arr_4ch = np.concatenate([arr_trans, nir[np.newaxis, ...]], axis=0)

        arr_norm = np.clip(arr_4ch.astype(np.float32) / 255.0, 0.0, 2.0)
        tensor = torch.from_numpy(arr_norm).unsqueeze(0).float()
        return tensor, meta


def export_multi_format_file(
    output_path: Union[str, Path],
    sr_data: Union[torch.Tensor, np.ndarray],
    meta: Optional[Dict[str, Any]] = None,
    scale: float = 4.0
) -> str:
    """
    Exports the super-resolved tensor or numpy array in the exact same format as the input file:
    - GeoTIFF: writes GeoTIFF with scaled Affine transform and preserved CRS.
    - NumPy (.npy): writes 4-channel uint16 array (0-65535 DN).
    - PNG / JPEG: writes standard image file.
    """
    if meta is None:
        meta = {}

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ext = meta.get("extension", out_p.suffix.lower())
    if not ext:
        ext = ".tif"
        out_p = out_p.with_suffix(".tif")

    if isinstance(sr_data, np.ndarray):
        if sr_data.ndim == 3:
            if "uint16" in str(sr_data.dtype):
                t = torch.from_numpy(sr_data.astype(np.float32) / 10000.0).unsqueeze(0)
            elif "uint8" in str(sr_data.dtype):
                t = torch.from_numpy(sr_data.astype(np.float32) / 255.0).unsqueeze(0)
            else:
                t = torch.from_numpy(sr_data.astype(np.float32)).unsqueeze(0)
        else:
            t = torch.from_numpy(sr_data.astype(np.float32))
    else:
        t = sr_data

    sr_t = torch.clamp(t.squeeze(0), 0.0, 2.0).detach().cpu()
    c, out_h, out_w = sr_t.shape
    orig_dtype_str = meta.get("orig_dtype", "uint16")

    if "uint16" in orig_dtype_str or ext in [".npy", ".npz", ".tif", ".tiff"]:
        sr_arr = (sr_t.numpy() * 10000.0).clip(0, 65535).astype(np.uint16)
    else:
        sr_arr = (sr_t.numpy() * 255.0).clip(0, 255).astype(np.uint8)

    if ext in [".npy", ".npz"]:
        orig_c = meta.get("orig_channels", 4)
        save_arr = sr_arr[:orig_c]
        np.save(str(out_p), save_arr)
        return str(out_p.resolve())

    elif meta.get("is_geotiff") or ext in [".tif", ".tiff", ".geotiff"]:
        orig_count = min(meta.get("orig_channels", 4), sr_arr.shape[0])
        export_data = sr_arr[:orig_count]

        if HAS_RASTERIO and meta.get("profile"):
            profile = meta["profile"].copy()
            if "transform" in profile and profile["transform"] is not None:
                orig_aff = profile["transform"]
                new_aff = Affine(
                    orig_aff.a / scale, orig_aff.b, orig_aff.c,
                    orig_aff.d, orig_aff.e / scale, orig_aff.f
                )
            else:
                new_aff = None

            profile.update({
                "driver": "GTiff",
                "height": out_h,
                "width": out_w,
                "count": orig_count,
                "dtype": "uint16" if "uint16" in orig_dtype_str else "uint8"
            })
            if new_aff:
                profile["transform"] = new_aff

            with rasterio.open(str(out_p), "w", **profile) as dst:
                for b in range(orig_count):
                    dst.write(export_data[b], b + 1)
        elif HAS_TIFFFILE:
            tifffile.imwrite(str(out_p), export_data)
        else:
            rgb_arr = make_rgb_composite(sr_arr)
            Image.fromarray(rgb_arr).save(str(out_p))
        return str(out_p.resolve())

    else:
        rgb_arr = make_rgb_composite(sr_arr)
        img = Image.fromarray(rgb_arr)
        if ext in [".jpg", ".jpeg"]:
            img.save(str(out_p), format="JPEG", quality=95)
        elif ext == ".png":
            img.save(str(out_p), format="PNG", optimize=True)
        else:
            img.save(str(out_p))
        return str(out_p.resolve())


# Aliases
load_multiformat_image = read_multi_format_file
save_multiformat_image = export_multi_format_file
