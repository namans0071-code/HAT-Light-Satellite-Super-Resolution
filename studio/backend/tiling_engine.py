"""
CPU-Optimized Seamless 2D Hann Window Blending Tiling Engine.
Performs sliding window tiling on arbitrary-size satellite scenes with zero boundary seams.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np


def get_2d_hann_window(h: int, w: int, device: torch.device) -> torch.Tensor:
    win_h = torch.hann_window(h, periodic=False, device=device, dtype=torch.float32).unsqueeze(1)
    win_w = torch.hann_window(w, periodic=False, device=device, dtype=torch.float32).unsqueeze(0)
    win_2d = torch.clamp(win_h * win_w, min=1e-3)
    return win_2d.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)


def cpu_tiled_inference(
    model: torch.nn.Module,
    lr_tensor: torch.Tensor,
    scale: float = 4.0,
    tile_size: int = 128,
    overlap: int = 32,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Seamless Sliding-Window Tiling Inference on CPU.
    """
    model.eval()
    B, C, H, W = lr_tensor.shape
    H_out = int(round(H * scale))
    W_out = int(round(W * scale))

    # If already small, direct forward pass
    if H <= tile_size and W <= tile_size:
        with torch.no_grad():
            return model(lr_tensor.to(device), scale=scale)

    stride = max(1, tile_size - overlap)
    output = torch.zeros((B, C, H_out, W_out), device=device, dtype=torch.float32)
    weight_map = torch.zeros((1, 1, H_out, W_out), device=device, dtype=torch.float32)

    y_steps = list(range(0, max(1, H - tile_size + 1), stride))
    if y_steps[-1] + tile_size < H:
        y_steps.append(H - tile_size)
    x_steps = list(range(0, max(1, W - tile_size + 1), stride))
    if x_steps[-1] + tile_size < W:
        x_steps.append(W - tile_size)

    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:
                lr_tile = lr_tensor[:, :, y : min(y + tile_size, H), x : min(x + tile_size, W)].to(device)
                sr_tile = model(lr_tile, scale=scale)

                out_y = int(round(y * scale))
                out_x = int(round(x * scale))
                tile_h, tile_w = sr_tile.shape[2], sr_tile.shape[3]

                blend_w = get_2d_hann_window(tile_h, tile_w, device)
                output[:, :, out_y : out_y + tile_h, out_x : out_x + tile_w] += sr_tile * blend_w
                weight_map[:, :, out_y : out_y + tile_h, out_x : out_x + tile_w] += blend_w

    output = output / torch.clamp(weight_map, min=1e-5)
    return output
