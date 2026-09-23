# HAT-Light: 4-Channel Satellite Imagery Super-Resolution

PyTorch implementation of **HAT-Light** (Hybrid Attention Transformer for Satellite Imagery), a lightweight super-resolution model for 4-channel Sentinel-2 optical imagery (Red, Green, Blue, Near-Infrared). HAT-Light performs 4x spatial super-resolution, enhancing 10m Ground Sampling Distance (GSD) imagery to 2.5m per pixel while preserving spectral and physical radiometric characteristics.

This repository includes data ingestion and preparation scripts, model training and inference pipelines, benchmark evaluation scripts against standard SR baselines, and an interactive web studio.

---

## Key Features

- **4-Band Multi-Spectral Processing:** Direct support for B04 (Red), B03 (Green), B02 (Blue), and B08 (NIR) multi-spectral bands in native 16-bit surface reflectance format.
- **Hybrid Attention Transformer:** Combines Window Multi-Head Self-Attention (W-MSA), Shifted-Window Attention (SW-MSA), and Depthwise Convolutional FFNs to capture broad spatial context with low computational overhead.
- **Physical Loss Formulation:** Trained with a composite objective combining Charbonnier loss, 2D FFT spectral alignment, spatial gradient losses, and Spectral Angle Mapper (SAM) to preserve vegetation indices (NDVI) and radiometric accuracy.
- **Interactive Web Studio:** Full-stack interface (React + FastAPI) for interactive visual comparisons (RGB true color, False-Color Infrared, NDVI) and on-the-fly super-resolution.
- **Reproducible Benchmarks:** Complete suite to evaluate and benchmark against Bicubic interpolation, RCAN, and SwinIR across 600 curated test patches.

---

## Project Structure

`	ext
HAT-Light-Satellite-Super-Resolution/
├── app.py                      # Launcher for the Web Studio (FastAPI backend + React frontend)
├── run_app.bat                 # One-click Windows batch launcher
├── requirements.txt            # Python dependencies
├── train.py                    # Top-level model training CLI script
│
├── core/                       # Core model architecture and runtime
│   ├── dataset.py              # PyTorch Dataset loader for 4-channel satellite imagery
│   ├── infer.py                # Standalone inference script for .npy / multi-band patches
│   ├── losses.py               # Composite physical loss functions (Charbonnier, FFT, Grad, SAM)
│   ├── metrics.py              # Geospatial and image quality metrics (PSNR, SSIM, SAM, ERGAS)
│   └── models/                 # Model definitions
│       ├── hat_light.py        # HAT-Light model architecture (RHAG, W-MSA, FiLM)
│       ├── rcan.py             # RCAN baseline architecture
│       └── swinir.py           # SwinIR baseline architecture
│
├── studio/                     # Interactive Web Studio application
│   ├── backend/                # FastAPI backend serving inference APIs
│   │   └── server.py           # API endpoints for inference, metrics, and patch browsing
│   └── frontend/               # React 18 + TypeScript + Vite web interface
│       ├── src/                # UI components (SplitSlider, SpectralPlot, MetricsBadge)
│       └── package.json        # Frontend dependencies
│
├── pipeline/                   # Dataset acquisition and preprocessing
│   ├── cdse_client.py          # Copernicus Data Space Ecosystem (CDSE) client
│   ├── patch_extractor.py      # Extract and filter 128x128 4-band patches from Sentinel-2 tiles
│   └── README.md               # Data collection instructions
│
├── test_dataset/               # Curated evaluation dataset
│   ├── HR/                     # 600 curated high-resolution (128x128x4) uint16 patches
│   └── README.md               # Dataset details and spectral band specifications
│
├── benchmarks/                 # Benchmark reproduction scripts and results
│   ├── evaluate_curated_test.py # Direct evaluation of HAT-Light on curated test patches
│   ├── evaluate_all_models.py  # Comparative evaluation (Bicubic, RCAN, SwinIR, HAT-Light)
│   ├── table_comparative_results.md # Benchmark results table
│   ├── ablations/              # Ablation studies (loss functions, attention components)
│   └── cross_sensor/           # Zero-shot cross-sensor evaluation (NAIP dataset)
│
├── weights/                    # Pre-trained model weights
│   ├── best_model.pth          # HAT-Light trained checkpoint
│   └── README.md               # Checkpoint details and hyperparameter config
│
└── assets/                     # Visual figures, previews, and architectural diagrams
`

---

## Quick Start

### 1. Installation

Clone the repository and install the Python dependencies:

`ash
git clone https://github.com/namans0071-code/HAT-Light-Satellite-Super-Resolution.git
cd HAT-Light-Satellite-Super-Resolution
pip install -r requirements.txt
`

For the web frontend:
`ash
cd studio/frontend
npm install
cd ../..
`

### 2. Launch Interactive Web Studio

Start both the backend server and frontend interface with a single command:

`ash
# Using Python:
python app.py

# Or on Windows using batch script:
run_app.bat
`

Once launched, navigate to http://localhost:5173 in your browser. The studio lets you:
- Browse and select from the 600 curated Sentinel-2 test patches.
- Switch between visualization modes: True Color (RGB), False Color Infrared (CIR: NIR-Red-Green), and Normalized Difference Vegetation Index (NDVI).
- Inspect high-resolution reconstructions side-by-side using interactive split-sliders.
- View real-time quantitative quality metrics (PSNR, SSIM, SAM, ERGAS).

---

## Running Inference

You can run super-resolution inference directly from the command line using core/infer.py:

`ash
python core/infer.py \
    --input test_dataset/HR/patch_0001.npy \
    --weights weights/best_model.pth \
    --output output_sr.npy \
    --scale 4
`

### Python API Example

`python
import torch
import numpy as np
from core.models.hat_light import HATLight

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HATLight(in_channels=4, out_channels=4, embed_dim=96, num_rhag=6).to(device)

# Load weights
checkpoint = torch.load('weights/best_model.pth', map_location=device)
model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
model.eval()

# Load low-resolution 4-channel input (shape: H x W x 4, uint16 or float32)
lr_patch = np.load('test_dataset/HR/patch_0001.npy').astype(np.float32) / 10000.0
lr_tensor = torch.from_numpy(lr_patch).permute(2, 0, 1).unsqueeze(0).to(device)

# Run 4x super-resolution
with torch.no_grad():
    sr_tensor = model(lr_tensor, scale=4.0)

# Convert back to numpy (shape: 4H x 4W x 4)
sr_patch = (sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 10000.0).clip(0, 10000).astype(np.uint16)
`

---

## Model Training

To train the HAT-Light model from scratch or resume training:

`ash
python train.py \
    --data_dir /path/to/dataset \
    --epochs 100 \
    --batch_size 16 \
    --lr 2e-4 \
    --scale 4 \
    --save_dir weights/
`

Training options:
- --data_dir: Directory containing training image patches.
- --scale: Upscaling factor (default: 4).
- --epochs: Number of training epochs (default: 100).
- --batch_size: Batch size (default: 16).
- --lr: Initial learning rate for Adam optimizer (default: 2e-4).
- --loss: Loss configuration: composite (Charbonnier + FFT + Grad + SAM) or l1.

---

## Benchmarks & Evaluation

All benchmark experiments can be run directly on the curated 600-patch test dataset:

### 1. Evaluate HAT-Light Checkpoint
`ash
python benchmarks/evaluate_curated_test.py
`
This evaluates weights/best_model.pth across all 600 curated test patches and reports overall PSNR, SSIM, SAM, and ERGAS, as well as per-band metrics (Red, Green, Blue, NIR).

### 2. Compare Against Baseline Models
`ash
python benchmarks/evaluate_all_models.py
`
This runs comparative evaluations against Bicubic interpolation, RCAN, and SwinIR.

### Summary Benchmark Results

Results evaluated on the 600 held-out Sentinel-2 test patches (4 channels: B04, B03, B02, B08) at 4x super-resolution:

| Method | Parameters | Latency (ms) | FPS | PSNR (dB) | SSIM | SAM (deg) | ERGAS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Bicubic Interpolation | - | 0.8 | 1250 | 28.61 | 0.7912 | 2.54 | 3.42 |
| RCAN (4-band) | 15.6M | 42.1 | 23.8 | 32.23 | 0.8874 | 1.62 | 2.18 |
| SwinIR (4-band) | 11.9M | 36.4 | 27.5 | 32.74 | 0.8951 | 1.51 | 2.04 |
| **HAT-Light (Ours)** | **4.2M** | **12.7** | **78.7** | **33.48** | **0.9048** | **1.37** | **1.89** |

---

## Dataset Acquisition Pipeline

To download and generate your own training dataset from Copernicus Sentinel-2 tiles:

1. Setup CDSE credentials in pipeline/cdse_client.py.
2. Query and download cloud-free Level-2A granules:
   `ash
   python pipeline/cdse_client.py --bbox <min_lon> <min_lat> <max_lon> <max_lat> --max_cloud 5
   `
3. Extract 4-channel patches:
   `ash
   python pipeline/patch_extractor.py --input_dir /path/to/granules --output_dir /path/to/patches --patch_size 128
   `

See [pipeline/README.md](pipeline/README.md) for detailed configuration options.

---

## Pre-Trained Weights

The model weights are available in weights/best_model.pth (~58 MB):
- **Input/Output Channels:** 4 (Red, Green, Blue, NIR)
- **Parameters:** ~4.2M
- **Training Resolution:** 128x128 HR patches
- **Scale:** 4x

See [weights/README.md](weights/README.md) for model dictionary structure and loading instructions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
