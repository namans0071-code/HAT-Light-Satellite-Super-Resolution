# Curated Test Dataset

This directory contains 600 curated Sentinel-2 Level-2A multi-spectral image patches used for evaluating model performance.

---

## Dataset Specifications

- **Total Patches:** 600
- **Location:** `test_dataset/HR/` (`patch_0001.npy` through `patch_0600.npy`)
- **Format:** NumPy `.npy` array
- **Data Type:** `uint16` (surface reflectance scaled by 10,000)
- **Spatial Resolution:** 128 x 128 pixels (at native 10m GSD)
- **Channels (4 Bands):**
  - Channel 0: B04 (Red, 665 nm)
  - Channel 1: B03 (Green, 560 nm)
  - Channel 2: B02 (Blue, 490 nm)
  - Channel 3: B08 (Near-Infrared / NIR, 842 nm)

---

## Loading Patches in Python

```python
import numpy as np

# Load a patch (shape: 128 x 128 x 4)
patch = np.load('test_dataset/HR/patch_0001.npy')

# Normalize to [0, 1] float32 for model input
patch_float = patch.astype(np.float32) / 10000.0

# Extract specific bands
red = patch_float[:, :, 0]
green = patch_float[:, :, 1]
blue = patch_float[:, :, 2]
nir = patch_float[:, :, 3]

# Compute NDVI
ndvi = (nir - red) / (nir + red + 1e-8)
```

Low-resolution inputs are generated on the fly during training and evaluation using antialiased bicubic downsampling (factor 4x) to simulate satellite sensor degradation.
