# 📊 Comparative Benchmarks & Evaluation Suite

This directory contains the comparative evaluation harness, metric logs, and figure generation scripts for Table I and Figures 1–3 of our research paper:

> **Paper:** *Continuous Multi-Spectral Satellite Super-Resolution via Hybrid Attention Transformers and Radiometric Physical Constraints*  
> **Model Checkpoint:** [`../weights/best_model.pth`](../weights/best_model.pth)  
> **Test Corpus:** [`../test_dataset/`](../test_dataset/) (600 curated 4-band Sentinel-2 Level-2A granules across 6 biomes)

---

## 🏆 Comparative Benchmark Matrix (Table I)

Evaluated across all 600 held-out Sentinel-2 test patches (4 channels: B04 Red, B03 Green, B02 Blue, B08 NIR) at $4\times$ super-resolution ($10\text{m} \to 2.5\text{m}$ GSD):

| Architecture | Parameters | Latency (ms) | FPS | PSNR Overall (dB) $\uparrow$ | Red (B04) | Green (B03) | Blue (B02) | NIR (B08) | SSIM $\uparrow$ | SAM ($^\circ$) $\downarrow$ | ERGAS $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bicubic Baseline** | -- | 0.4 ms | >2000 | 32.23 | 37.10 | 35.81 | 35.53 | 26.54 | 0.8673 | 1.68 | 2.18 |
| **EDSR** (Lim et al.) | 1.52M | 5.2 ms | 192.3 | 32.75 | 38.33 | 36.87 | 36.63 | 27.27 | 0.8798 | 1.59 | 2.05 |
| **RCAN** (Zhang et al.) | 1.68M | 14.8 ms | 67.5 | 32.77 | 38.38 | 36.90 | 36.64 | 27.29 | 0.8801 | 1.59 | 2.05 |
| **SwinIR-Light** (Liang et al.) | 0.92M | 10.5 ms | 95.2 | 32.73 | 38.30 | 36.85 | 36.59 | 27.25 | 0.8795 | 1.60 | 2.06 |
| **HAT-Light (Ours)** | 5.09M | **12.7 ms** | **78.7** | **33.48** | **39.62** | **37.76** | **37.40** | **28.22** | **0.9048** | **1.37** | **1.89** |

- **PSNR Gain:** HAT-Light achieves **+1.25 dB** over Bicubic and **+0.71 dB** over RCAN.
- **Spectral Fidelity:** Lowest Spectral Angle Mapper (**1.37$^\circ$**), preserving multi-spectral band ratios critical for downstream remote sensing indices (NDVI, NDWI).
- **Inference Speed:** Runs at **78.7 FPS** (12.7 ms per patch) on a consumer laptop GPU (RTX 3050, 6GB VRAM).

---

## 🚀 Reproducing Benchmark Numbers

### 1. Run Full Evaluation Harness
To compute PSNR, SSIM, SAM, and ERGAS across all 600 test patches:
```bash
python benchmarks/evaluate_all_models.py
```
*Note: Evaluates Bicubic and HAT-Light automatically using `weights/best_model.pth`. If you wish to re-evaluate the baseline models (EDSR, RCAN, SwinIR), place their `.pth` checkpoints in `weights/` or retrain them using `baselines/train_baselines.py`.*

### 2. Generate Publication Figures (Figures 1–3)
To render high-DPI paper figures directly into `paper/figures/`:
```bash
python benchmarks/generate_paper_figures.py
```
This generates:
- `fig1_visual_comparison.png`: Multi-biome qualitative spatial grids (True Color & CIR).
- `fig2_transect_edge_profile.png`: 1D pixel reflectance transect across high-contrast airport runway lineaments.
- `fig3_ndvi_biophysical_fidelity.png`: Pixel-by-pixel NDVI correlation scatter plot ($R^2 = 0.898$).

---

## 📁 Directory Files

- `evaluate_all_models.py`: Automated multi-model evaluation harness.
- `generate_paper_figures.py`: Generates publication-ready figures for LaTeX.
- `comparative_results.json`: JSON output containing per-band metrics for all architectures.
- `table_comparative_results.md`: Standalone markdown table for documentation.
- `table_comparative_results.tex`: LaTeX table directly importable into manuscripts.
