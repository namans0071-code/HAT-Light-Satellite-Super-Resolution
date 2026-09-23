# Zero-Shot Cross-Sensor Generalization Benchmark (USGS NAIP 2.5m Ground Truth)

| Architecture | PSNR Overall (dB) | Red (B04) | Green (B03) | Blue (B02) | NIR (B08) | SSIM | SAM (deg) | ERGAS | Net Gain vs Bicubic |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic Baseline** | 29.90 | 33.82 | 34.31 | 35.4 | 25.23 | 0.8109 | 1.41 | 2.83 | 0.00 dB |
| **EDSR** | 30.50 | 33.84 | 34.5 | 35.44 | 26.02 | 0.8245 | 1.56 | 2.74 | +0.60 dB |
| **RCAN** | 30.53 | 33.87 | 34.53 | 35.46 | 26.06 | 0.8257 | 1.56 | 2.73 | +0.63 dB |
| **SwinIR-Light** | 30.48 | 33.84 | 34.49 | 35.43 | 26.01 | 0.8240 | 1.58 | 2.74 | +0.58 dB |
| **HAT-Light (Ours)** | **30.44** | 33.58 | 34.29 | 35.27 | 26.04 | **0.8332** | **1.47** | **2.83** | **+0.54 dB** |