# 🔬 Architectural & Loss Component Ablation Study

This directory contains the ablation study scripts, numerical outputs, and LaTeX table generator for Table II of our research paper:

> **Paper Reference:** Section IV-B: *Component Ablation Study & Radiometric Constraints*

---

## 📊 Component Ablation Matrix (Table II)

Evaluated across the 600-patch Sentinel-2 test split under systematic isolation of individual architectural modules and physical loss functions:

| Configuration | PSNR (dB) $\uparrow$ | SSIM $\uparrow$ | SAM ($^\circ$) $\downarrow$ | ERGAS $\downarrow$ | $\Delta$ PSNR (dB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Proposed HAT-Light** | **33.48** | **0.9048** | **1.37** | **1.89** | **0.00** |
| w/o Cosine SAM Loss ($\mathcal{L}_{\text{SAM}}$) | 33.31 | 0.9012 | 1.56 | 1.94 | -0.17 |
| w/o 2D rFFT Spectral Loss ($\mathcal{L}_{\text{FFT}}$) | 33.12 | 0.8985 | 1.49 | 1.98 | -0.36 |
| w/o Spatial Gradient Loss ($\mathcal{L}_{\text{grad}}$) | 33.19 | 0.8994 | 1.46 | 1.96 | -0.29 |
| w/o DW-FFN (Standard Linear FFN) | 32.97 | 0.8936 | 1.52 | 2.01 | -0.51 |

---

## 💡 Scientific Takeaways

1. **Depthwise Convolutional FFN ($\Delta = -0.51\text{ dB}$):** The largest drop occurs when replacing the $3 \times 3$ Depthwise Convolution with a standard Linear FFN. This confirms that pure self-attention lacks positional inductive biases needed to resolve sub-pixel satellite lineaments (such as airport runways and road boundaries).
2. **2D Real FFT Spectral Loss ($\Delta = -0.36\text{ dB}$):** Enforcing frequency domain consistency prevents high-frequency attenuation, ensuring fine periodic textures (e.g. agricultural crop rows) remain sharp.
3. **Spatial Gradient Loss ($\Delta = -0.29\text{ dB}$):** Constrains directional Sobel derivatives, minimizing structural edge blurring.
4. **Cosine SAM Loss ($\Delta \text{SAM} = +0.19^\circ$):** Removing $\mathcal{L}_{\text{SAM}}$ degrades spectral angle fidelity from $1.37^\circ$ to $1.56^\circ$, validating its critical role in multi-spectral band alignment.

---

## 🚀 Running the Ablation Suite

```bash
python ablations/run_ablation.py
```

Outputs will be saved directly to `ablation_results.json`, `table_ablation.md`, and `table_ablation.tex`.
