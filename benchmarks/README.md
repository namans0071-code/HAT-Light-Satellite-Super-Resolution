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

### 1. Run Curated Test Evaluation (HAT-Light vs Bicubic)
To compute PSNR, SSIM, SAM, ERGAS, and per-band metrics on the 600 curated test patches:
```bash
python benchmarks/evaluate_curated_test.py
```

### 2. Run Master Comparative Benchmark (All Models)
To evaluate all models (Bicubic, EDSR, RCAN, SwinIR-Light, HAT-Light):
```bash
python benchmarks/evaluate_all_models.py
```
*Note: Evaluates Bicubic and HAT-Light automatically using `weights/best_model.pth`. If baseline checkpoints are placed in `weights/`, they are benchmarked in parallel.*

### 3. Run Component & Loss Ablations
To isolate each loss function and architectural component:
```bash
python benchmarks/ablations/run_ablation.py
```

### 4. Run Zero-Shot Cross-Sensor Validation (USGS NAIP)
To evaluate out-of-distribution transfer under the Wald protocol:
```bash
python benchmarks/cross_sensor/evaluate_cross_sensor.py
```

### 5. Generate Benchmark Figures
To render publication-quality figures directly into `assets/`:
```bash
python benchmarks/generate_paper_figures.py
```
This generates:
- `assets/fig1_visual_comparison.png`: Multi-biome qualitative spatial grids (True Color & CIR).
- `assets/fig2_transect_edge_profile.png`: 1D pixel reflectance transect across runway lineaments.
- `assets/fig3_ndvi_biophysical_fidelity.png`: Pixel-by-pixel NDVI correlation scatter plot ($R^2 = 0.898$).

---

## 📁 Benchmarks Directory Structure

- `evaluate_curated_test.py`: Standalone 600-patch test evaluation script.
- `evaluate_all_models.py`: Automated multi-model comparative evaluation harness.
- `generate_paper_figures.py`: Publication figure renderer.
- `baselines/`: Implementations of EDSR, RCAN, SwinIR-Light, and baseline training script.
- `ablations/`: Systematic ablation study runner, numerical results, and summary table.
- `cross_sensor/`: Wald protocol cross-sensor evaluation on USGS NAIP 2.5m data.
- `curated_test_evaluation.json`: Numerical results on the 600 test patches.
- `comparative_results.json`: Per-band comparative metrics across all architectures.
- `table_comparative_results.md`: Markdown summary table.
