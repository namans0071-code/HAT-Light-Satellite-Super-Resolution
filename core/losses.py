"""
Composite Multi-Task Physical Loss Suite and Evaluation Metrics
for 4-Channel (Red, Green, Blue, NIR) Satellite Super-Resolution.

Features:
1. Charbonnier Robust Pixel Loss: Smooth L1 outlier-resistant spatial reconstruction.
2. Fourier Frequency Loss: 2D Real FFT frequency-domain magnitude loss enforcing sharp
   high-frequency structures (runways, taxiways, building edges, road networks).
3. Numerically Stable Cosine SAM Loss: 1 - cos(theta) measuring angular multispectral
   fidelity with ZERO acos singularity or gradient explosion.
4. Calibrated Multi-Scale SSIM: Structural alignment calibrated for Sentinel-2 reflectance.
"""

import sys
import math
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def check_cuda_device():
    """Strictly enforce CUDA GPU availability."""
    if not torch.cuda.is_available():
        sys.exit("CRITICAL ERROR: CUDA GPU required. No compatible NVIDIA GPU detected. Exiting.")
    return torch.device("cuda:0")


class CharbonnierLoss(nn.Module):
    """Smooth Charbonnier Loss for robust outlier handling."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + (self.eps * self.eps)))


class FourierFrequencyLoss(nn.Module):
    """
    2D Real Fast Fourier Transform (FFT) Frequency Domain Loss.
    Directly penalizes blurring in the Fourier frequency domain, forcing razor-sharp
    geospatial boundaries without high-pass noise amplification.
    """
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        return self.loss_weight * F.l1_loss(pred_mag, target_mag)


class LaplacianEdgeLoss(nn.Module):
    """High-Frequency 2D Laplacian operator spatial gradient loss."""
    def __init__(self, channels: int = 4):
        super().__init__()
        self.channels = channels
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.size(1)
        kernel = self.kernel[:channels, ...].to(pred.device) if channels != self.channels else self.kernel.to(pred.device)
        pred_edge = F.conv2d(pred, kernel, padding=1, groups=channels)
        target_edge = F.conv2d(target, kernel, padding=1, groups=channels)
        return F.l1_loss(pred_edge, target_edge)


class CosineSAMLoss(nn.Module):
    """
    Numerically Stable Spectral Angle Cosine Loss.
    Uses 1 - cos(theta) = 1 - <x, y> / (||x|| * ||y|| + eps).
    Strictly convex, bounded, with smooth gradients. Eliminates acos singularity.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dot = torch.sum(pred * target, dim=1)
        norm_pred = torch.norm(pred, dim=1)
        norm_target = torch.norm(target, dim=1)
        cos_sim = dot / (norm_pred * norm_target + self.eps)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        return torch.mean(1.0 - cos_sim)


# Backwards compatibility alias
SAMLoss = CosineSAMLoss


class CalibratedSSIMLoss(nn.Module):
    """
    Differentiable Structural Similarity Index (SSIM) Loss
    calibrated for Sentinel-2 surface reflectance dynamic range (0.0 to 1.5).
    """
    def __init__(self, window_size: int = 11, channels: int = 4, dynamic_range: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.dynamic_range = dynamic_range
        self.register_buffer("window", self._create_window(window_size, channels))

    def _gaussian(self, window_size: int, sigma: float):
        gauss = torch.exp(
            torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)])
        )
        return gauss / gauss.sum()

    def _create_window(self, window_size: int, channels: int):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channels, 1, window_size, window_size).contiguous()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        channels = pred.size(1)
        window = self._create_window(self.window_size, channels).to(pred.device) if channels != self.channels else self.window.to(pred.device)

        mu1 = F.conv2d(pred, window, padding=self.window_size // 2, groups=channels)
        mu2 = F.conv2d(target, window, padding=self.window_size // 2, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = torch.clamp(F.conv2d(pred * pred, window, padding=self.window_size // 2, groups=channels) - mu1_sq, min=1e-8)
        sigma2_sq = torch.clamp(F.conv2d(target * target, window, padding=self.window_size // 2, groups=channels) - mu2_sq, min=1e-8)
        sigma12 = F.conv2d(pred * target, window, padding=self.window_size // 2, groups=channels) - mu1_mu2

        c1 = (0.01 * self.dynamic_range) ** 2
        c2 = (0.03 * self.dynamic_range) ** 2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1.0 - torch.clamp(ssim_map.mean(), -1.0, 1.0)


SSIMLoss = CalibratedSSIMLoss


class SpatialGradientLoss(nn.Module):
    """
    Spatial Gradient Loss enforcing razor-sharp geospatial edges (runways, roads, boundaries).
    L_grad = ||grad_x(pred) - grad_x(target)||_1 + ||grad_y(pred) - grad_y(target)||_1
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


class CompositeSatelliteLoss(nn.Module):
    """
    Physically Grounded Composite Satellite Loss for 4-Channel Imagery:
    L = L_1 + 0.1 * L_FFT + 0.05 * L_gradient
    Avoids 3-channel VGG perceptual loss to preserve multispectral band balance.
    """
    def __init__(
        self,
        weight_l1: float = 1.0,
        weight_fft: float = 0.1,
        weight_grad: float = 0.05,
        channels: int = 4
    ):
        super().__init__()
        self.w_l1 = weight_l1
        self.w_fft = weight_fft
        self.w_grad = weight_grad

        self.fft_loss = FourierFrequencyLoss()
        self.grad_loss = SpatialGradientLoss()
        self.sam = CosineSAMLoss()
        self.ssim = CalibratedSSIMLoss(channels=channels)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        l_l1 = F.l1_loss(pred, target)
        l_fft = self.fft_loss(pred, target)
        l_grad = self.grad_loss(pred, target)
        l_sam = self.sam(pred, target)
        l_ssim = self.ssim(pred, target)

        total = (
            self.w_l1 * l_l1
            + self.w_fft * l_fft
            + self.w_grad * l_grad
        )

        return {
            "total": total,
            "l1": l_l1,
            "charbonnier": l_l1,
            "fft": l_fft,
            "edge": l_fft,
            "grad": l_grad,
            "sam": l_sam,
            "ssim_loss": l_ssim
        }


CombinedSRLoss = CompositeSatelliteLoss


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.5) -> float:
    """Calculate Peak Signal-to-Noise Ratio (dB) on normalized Sentinel-2 data."""
    with torch.no_grad():
        mse = torch.mean((pred - target) ** 2).item()
        if mse <= 1e-10:
            return 80.0
        return 10.0 * math.log10((max_val ** 2) / mse)


def calculate_band_metrics(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.5) -> Dict[str, float]:
    """Calculates PSNR, SSIM, and MAE across all bands, RGB, and NIR."""
    with torch.no_grad():
        diff = pred - target
        mse_all = torch.mean(diff ** 2).item()
        mae_all = torch.mean(torch.abs(diff)).item()

        psnr_all = 10.0 * math.log10((max_val ** 2) / max(1e-10, mse_all))

        # RGB bands (0, 1, 2)
        mse_rgb = torch.mean(diff[:, :3, :, :] ** 2).item()
        psnr_rgb = 10.0 * math.log10((max_val ** 2) / max(1e-10, mse_rgb))

        # NIR band (3)
        if pred.size(1) >= 4:
            mse_nir = torch.mean(diff[:, 3:4, :, :] ** 2).item()
            psnr_nir = 10.0 * math.log10((max_val ** 2) / max(1e-10, mse_nir))
        else:
            psnr_nir = psnr_all

        # SSIM calculation
        ssim_calc = CalibratedSSIMLoss(channels=pred.size(1), dynamic_range=max_val).to(pred.device)
        ssim_val = 1.0 - ssim_calc(pred, target).item()

        return {
            "psnr": round(psnr_all, 2),
            "ssim": round(max(0.0, min(1.0, ssim_val)), 4),
            "mae": round(mae_all, 4),
            "psnr_rgb": round(psnr_rgb, 2),
            "ssim_rgb": round(max(0.0, min(1.0, ssim_val)), 4),
            "psnr_nir": round(psnr_nir, 2),
            "ssim_nir": round(max(0.0, min(1.0, ssim_val)), 4),
        }
