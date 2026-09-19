# Systematic Component Ablation Study Results

| Configuration / Variant | Objective / Module | PSNR (dB) | SSIM | SAM (deg) | ERGAS | Physical Effect / Justification |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Proposed (HAT-Light)** | L1 + FFT + Grad + SAM + DW-FFN | **33.48** | **0.9048** | **1.37** | **1.45** | Full synergistic system |
| **w/o Cosine SAM Loss** | L1 + FFT + Grad | 33.26 | 0.9006 | 1.56 | 1.56 | Increases multi-spectral chromatic distortion |
| **w/o 2D FFT Spectral Loss** | L1 + Grad + SAM | 33.12 | 0.8970 | 1.45 | 1.61 | Loss of high-frequency periodic harmonics |
| **w/o Spatial Gradient Loss** | L1 + FFT + SAM | 33.19 | 0.8987 | 1.42 | 1.57 | Blurrier runway and wharf lineaments |
| **Pixel L1 Loss Only** | Charbonnier L1 only | 32.83 | 0.8903 | 1.63 | 1.73 | Lacks high-frequency and spectral constraints |
| **w/o Depthwise Conv FFN (Standard MLP)** | Full Loss + Linear FFN | 32.97 | 0.8936 | 1.51 | 1.67 | Removes localized translational inductive bias |
| **w/o Continuous FiLM Scale Head** | Full Loss + Discrete Head | 33.30 | 0.9013 | 1.43 | 1.53 | Loses arbitrary-scale magnification capability |