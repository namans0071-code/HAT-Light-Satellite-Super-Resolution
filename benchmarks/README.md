# Benchmark Evaluation Suite

This directory contains evaluation scripts to evaluate HAT-Light and compare its performance against standard super-resolution baselines (Bicubic interpolation, RCAN, and SwinIR).

---

## Evaluation Scripts

### 1. Evaluate HAT-Light Checkpoint
Evaluates the trained HAT-Light model (weights/best_model.pth) across the 600 curated test patches in test_dataset/HR/.

`ash
python benchmarks/evaluate_curated_test.py
`

Metrics computed:
- Overall PSNR (Peak Signal-to-Noise Ratio)
- Per-band PSNR (B04 Red, B03 Green, B02 Blue, B08 NIR)
- SSIM (Structural Similarity Index Measure)
- SAM (Spectral Angle Mapper in degrees)
- ERGAS (Relative Dimensionless Global Error in Synthesis)

### 2. Comparative Evaluation Across All Models
Evaluates and benchmarks Bicubic, RCAN, SwinIR, and HAT-Light on identical test sets:

`ash
python benchmarks/evaluate_all_models.py
`

### 3. Ablation Studies
Located in benchmarks/ablations/:
`ash
python benchmarks/ablations/run_ablation.py
`
Evaluates the impact of removing individual loss components (SAM, FFT, Gradient) or architectural modules.

### 4. Cross-Sensor Generalization
Located in benchmarks/cross_sensor/:
`ash
python benchmarks/cross_sensor/evaluate_cross_sensor.py
`
Evaluates zero-shot generalization of the Sentinel-2-trained HAT-Light model on USGS NAIP airborne multi-spectral data under the Wald protocol.

---

## Results Summary

Evaluated on 600 curated 4-band Sentinel-2 test patches at 4x super-resolution:

| Model | Parameters | Latency (ms) | FPS | PSNR (dB) | SSIM | SAM (deg) | ERGAS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Bicubic | - | 0.8 | 1250 | 28.61 | 0.7912 | 2.54 | 3.42 |
| RCAN (4-band) | 15.6M | 42.1 | 23.8 | 32.23 | 0.8874 | 1.62 | 2.18 |
| SwinIR (4-band) | 11.9M | 36.4 | 27.5 | 32.74 | 0.8951 | 1.51 | 2.04 |
| **HAT-Light** | **4.2M** | **12.7** | **78.7** | **33.48** | **0.9048** | **1.37** | **1.89** |
