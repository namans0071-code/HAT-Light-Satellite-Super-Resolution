# Pre-trained Checkpoint: HAT-Light 4x Super-Resolution

This directory stores the trained model checkpoint for the HAT-Light 4-channel satellite super-resolution model.

## Checkpoint Details

- **Filename**: `best_model.pth`
- **File Size**: ~58.4 MB (58,468,018 bytes)
- **Epoch**: 82 of 100 (Optimal checkpoint selection based on validation PSNR)
- **Input Channels**: 4 (B04 Red, B03 Green, B02 Blue, B08 Near-Infrared)
- **Output Channels**: 4 (Super-resolved multi-spectral reflectance)
- **Upscaling Factor**: 4x (10m Native -> 2.5m Super-Resolved)
- **Architecture Hyperparameters**:
  - `embed_dim`: 96
  - `num_rhag`: 6 (Residual Hybrid Attention Groups)
  - `num_hab_per_rhag`: 4 (Hybrid Attention Blocks per RHAG)
  - `num_heads`: 8
  - `window_size`: 8

## Benchmark Performance on 600 Curated Test Patches

| Metric | Bicubic Baseline | HAT-Light | Delta |
| :--- | :---: | :---: | :---: |
| **Overall PSNR** | 32.23 dB | **33.48 dB** | **+1.25 dB** |
| **SSIM** | 0.8673 | **0.9048** | **+0.0375** |
| **SAM ($^\circ$)** | 1.68$^\circ$ | **1.37$^\circ$** | **-0.31$^\circ$** |
| **ERGAS** | 2.18 | **1.89** | **-0.29** |

## Comparative Baseline Weights Note

To maintain repository focus and adhere to Git LFS bandwidth best practices, this directory contains exclusively the official **HAT-Light** model checkpoint (`best_model.pth`). 

Baseline architectures (EDSR, RCAN, SwinIR-Light) evaluated in Table I of our research paper can be retrained directly via `benchmarks/baselines/train_baselines.py` or downloaded as release assets from the [GitHub Releases page](https://github.com/namans0071-code/HAT-Light-Satellite-Super-Resolution/releases). If downloaded, place `edsr_best.pth`, `rcan_best.pth`, and `swinir_light_best.pth` in this directory to evaluate them with `python benchmarks/evaluate_all_models.py`.

## Git LFS Note

If managing this repository with Git, this file is tracked via Git LFS (`.gitattributes` is preconfigured for `*.pth`).

