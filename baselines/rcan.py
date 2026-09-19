"""
RCAN: Residual Channel Attention Networks
Adapted for 4-Channel (RGB + NIR) Sentinel-2 Satellite Super-Resolution.
Reference: Zhang et al., ECCV 2018.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CALayer(nn.Module):
    """Channel Attention (CA) Layer."""
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB)."""
    def __init__(self, n_feats: int, kernel_size: int = 3, reduction: int = 16, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=True),
            CALayer(n_feats, reduction)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x) * self.res_scale


class ResidualGroup(nn.Module):
    """Residual Group (RG) containing multiple RCAB blocks and a short skip connection."""
    def __init__(self, n_feats: int, n_rcab: int = 4, reduction: int = 16):
        super().__init__()
        modules = [RCAB(n_feats, reduction=reduction) for _ in range(n_rcab)]
        modules.append(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True))
        self.body = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class RCAN(nn.Module):
    """
    Residual Channel Attention Network (RCAN) for 4-channel Remote Sensing Super-Resolution.
    Features:
    - 4-channel input (RGB + NIR) and 4-channel output.
    - Long skip connection and channel attention across multi-spectral bands.
    - Global bicubic residual connection.
    - Zero-residual initialized output head.
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        scale: int = 4,
        n_feats: int = 64,
        n_resgroups: int = 4,
        n_rcab: int = 4,
        reduction: int = 16
    ):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1. Shallow Feature Extraction
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1, bias=True)

        # 2. Deep Residual Groups with Long Skip Connection (LSC)
        modules_body = [ResidualGroup(n_feats, n_rcab=n_rcab, reduction=reduction) for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True))
        self.body = nn.Sequential(*modules_body)

        # 3. Upsampling Head (PixelShuffle 4x)
        self.upsample = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats * 4, 3, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        # 4. Final Reconstruction Head
        self.tail = nn.Conv2d(n_feats, out_channels, 3, padding=1, bias=True)

        # Zero-init tail for lower-bound guarantee
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor, scale: float = 4.0) -> torch.Tensor:
        b, c, h, w = x.shape
        target_h, target_w = int(round(h * self.scale)), int(round(w * self.scale))

        bicubic_base = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False)

        feat_shallow = self.head(x)
        res = self.body(feat_shallow)
        feat_deep = feat_shallow + res
        feat_up = self.upsample(feat_deep)
        res_out = self.tail(feat_up)

        out = bicubic_base + res_out
        return torch.clamp(out, 0.0, 2.0)
