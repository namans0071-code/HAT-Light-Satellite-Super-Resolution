# 🧪 Curated Sentinel-2 Multi-Spectral Test Dataset

This directory contains the standardized, curated test split utilized for the comparative benchmarks (Table I) and ablation studies (Table II) in our research paper.

---

## 📊 Dataset Specifications

- **Total Test Patches:** 600 curated granules
- **Storage Format:** NumPy arrays (`.npy`), 16-bit unsigned integers (`uint16`)
- **Spectral Bands (4 Channels):**
  - **Channel 0:** Band 4 (Red, $\lambda_c = 665\text{ nm}$)
  - **Channel 1:** Band 3 (Green, $\lambda_c = 560\text{ nm}$)
  - **Channel 2:** Band 2 (Blue, $\lambda_c = 490\text{ nm}$)
  - **Channel 3:** Band 8 (Near-Infrared / NIR, $\lambda_c = 842\text{ nm}$)
- **Spatial Resolution & Dimensions:**
  - `HR/`: High-resolution native 10m surface reflectance ($128 \times 128 \times 4$)
  - Low-resolution inputs ($32 \times 32 \times 4$) are generated dynamically on-the-fly by the PyTorch loader via antialiased bicubic downsampling, eliminating dataset redundancy.
- **Scale Factor:** $4\times$ spatial magnification ($10\text{m} \to 2.5\text{m}$ operational target)

---

## 🛡️ Curated Quality Standards

Every patch in this dataset has undergone rigorous quality filtering:
1. **Zero No-Data / Fill Values:** Verified $100\%$ valid reflectance values (no border zeros or detector dropouts).
2. **Strict Cloud Saturation Elimination:** All saturated cloud-top anomalies (reflectance $> 3,800\text{ DN}$) have been purged.
3. **High Structural Entropy:** Filtered to eliminate flat, featureless water or barren desert tiles, maximizing geographic diversity across six biomes:
   - Aviation infrastructure (runways, taxiways, hangars)
   - Dense urban street grids and high-density buildings
   - Military airfields and logistics centers
   - Intensive agricultural parcels and crop canopies
   - Coastal ports, waterways, and marine terminals
   - High-relief alpine terrain and mountain ridgelines

---

## 🔬 Loading via PyTorch

```python
from core.dataset import SatelliteSRDataset
from torch.utils.data import DataLoader

# Load the test dataset
test_ds = SatelliteSRDataset("test_dataset", scale=4, is_train=False, augment=False)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

for lr, hr in test_loader:
    # lr: [B, 4, 32, 32] float32 in [0, 2.0]
    # hr: [B, 4, 128, 128] float32 in [0, 2.0]
    pass
```
