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

## Benchmark Performance on 799 Multi-Spectral Test Patches

| Metric | Bicubic Baseline | HAT-Light | Delta |
| :--- | :---: | :---: | :---: |
| **Overall PSNR** | 33.43 dB | **40.18 dB** | **+6.75 dB** |
| **RGB Bands PSNR** | 35.12 dB | **42.45 dB** | **+7.33 dB** |
| **NIR Band PSNR** | 32.20 dB | **38.64 dB** | **+6.44 dB** |
| **SSIM** | 0.812 | **0.934** | **+0.122** |
| **MAE** | 0.042 | **0.012** | **-71.4%** |

## Git LFS Note

If managing this repository with Git, this file can either be pushed directly (<100MB) or tracked via Git LFS (`.gitattributes` is preconfigured for `*.pth`).
