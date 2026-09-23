"""
EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution
Adapted for 4-Channel (RGB + NIR) Sentinel-2 Satellite Super-Resolution.
Reference: Lim et al., CVPRW 2017.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block without Batch Normalization, with residual scaling."""
    def __init__(self, n_feats: int, kernel_size: int = 3, res_scale: float = 0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class Upsampler(nn.Sequential):
    """4x Upsampler via sub-pixel convolution (PixelShuffle)."""
    def __init__(self, scale: int, n_feats: int):
        m = []
        if (scale & (scale - 1)) == 0:  # scale is power of 2
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1, bias=True))
                m.append(nn.PixelShuffle(2))
                m.append(nn.ReLU(inplace=True))
        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1, bias=True))
            m.append(nn.PixelShuffle(3))
            m.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError(f"Scale {scale} is not supported.")
        super().__init__(*m)


class EDSR(nn.Module):
    """
    EDSR architecture for 4-channel Multi-Spectral Remote Sensing Super-Resolution.
    Features:
    - 4-channel input (Red, Green, Blue, NIR) and 4-channel output.
    - Global bicubic residual connection.
    - Zero-residual initialized output head for guaranteed lower-bound baseline.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        scale: int = 4,
        n_feats: int = 64,
        n_resblocks: int = 16,
        res_scale: float = 0.1
    ):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1. Shallow Feature Extraction
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1, bias=True)

        # 2. Deep Residual Backbone
        body = [ResBlock(n_feats, kernel_size=3, res_scale=res_scale) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True))
        self.body = nn.Sequential(*body)

        # 3. Upsampling Head
        self.upsample = Upsampler(scale, n_feats)

        # 4. Final Reconstruction Layer
        self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1, bias=True)

        # Zero-init tail to start from Bicubic baseline
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor, scale: float = 4.0) -> torch.Tensor:
        b, c, h, w = x.shape
        target_h, target_w = int(round(h * self.scale)), int(round(w * self.scale))

        # Global Bicubic reference base
        bicubic_base = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)

        # Forward pass through EDSR
        feat_shallow = self.head(x)
        res = self.body(feat_shallow)
        feat_deep = feat_shallow + res
        feat_up = self.upsample(feat_deep)
        res_out = self.tail(feat_up)

        out = bicubic_base + res_out
        return torch.clamp(out, 0.0, 2.0)
