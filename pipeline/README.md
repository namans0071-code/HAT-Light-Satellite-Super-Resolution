# Data Ingestion and Preprocessing Pipeline

Tools for downloading Sentinel-2 Level-2A multi-spectral tiles from the Copernicus Data Space Ecosystem (CDSE) and extracting 4-band patches for training and evaluation.

---

## Pipeline Overview

1. `cdse_client.py`: Authenticates with the CDSE OData API, queries granules within a geographic bounding box and date range with cloud cover filtering, and downloads SAFE archives or band GeoTIFFs.
2. `patch_extractor.py`: Reads 10m bands (B02, B03, B04, B08), aligns and crops non-overlapping 128x128 patches, filters out cloudy/nodata tiles, and exports them as `.npy` files.

---

## Usage

### 1. Configure CDSE Credentials
Set your Copernicus Data Space Ecosystem credentials via environment variables:
```bash
export CDSE_CLIENT_ID="your_client_id"
export CDSE_CLIENT_SECRET="your_client_secret"
```

### 2. Search and Download Granules
```bash
python pipeline/cdse_client.py \
    --bbox 72.5 22.5 73.5 23.5 \
    --start_date 2024-01-01 \
    --end_date 2024-06-30 \
    --max_cloud 5 \
    --output_dir ./data/granules
```

### 3. Extract Multi-Spectral Patches
```bash
python pipeline/patch_extractor.py \
    --input_dir ./data/granules \
    --output_dir ./data/patches \
    --patch_size 128 \
    --cloud_threshold 0.05
```
