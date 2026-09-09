# Sentinel-2 4-Channel Data Acquisition & Degradation Pipeline

An automated, reproducible data engineering pipeline that interfaces with the **Copernicus Data Space Ecosystem (CDSE) OData API**, retrieves cloud-free Sentinel-2 Level-2A surface reflectance granules, extracts 16-bit 4-band patches (128x128), applies physical sensor Point Spread Function (PSF) simulation, and partitions the dataset into Train, Validation, and Test splits.

---

## 📊 Dataset Volume & Split Breakdown

The pipeline generates an uncompressed corpus of **8,000 paired multi-spectral patches** ($128 \times 128 \times 4$ in 16-bit unsigned integer format), totaling **$\approx 1.05\text{ GB}$ of raw physical Bottom-Of-Atmosphere (BOA) reflectance**:

| Split Name | Percentage | Patch Count | Spatial Dimensions | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Train Set** | 80% | 6,400 patches | $128 \times 128$ (HR), $32 \times 32$ (LR) | Model optimization with random dihedral augmentations |
| **Validation Set** | 10% | 800 patches | $128 \times 128$ (HR), $32 \times 32$ (LR) | Dynamic checkpointing and early stopping |
| **FinalTest Set** | 10% | 799 patches | $128 \times 128$ (HR), $32 \times 32$ (LR) | Isolated, zero-leakage benchmark evaluation |

---

## 🌍 Targeted Global Areas of Interest (16 AOIs)

To ensure geographic diversity and terrain generalization, granules were queried and downloaded across **16 globally distributed Areas of Interest**:

| Category | Target Name | Lat / Lon | Feature Description |
| :--- | :--- | :---: | :--- |
| **Military Base** | Norfolk Naval Station | $36.945^\circ\text{N}, -76.326^\circ\text{W}$ | Aircraft carriers, naval dry docks, coastal infrastructure |
| **Military Base** | Ramstein Air Base | $49.437^\circ\text{N}, 7.600^\circ\text{E}$ | Hardened shelters, military runways, forested perimeter |
| **Military Base** | Nellis AFB Nevada | $36.236^\circ\text{N}, -115.034^\circ\text{W}$ | Desert flightlines, test range complexes, high-albedo soil |
| **Military Base** | Hindon Air Force Station | $28.706^\circ\text{N}, 77.360^\circ\text{E}$ | Transport runways, military hangars, perimeter zones |
| **Airport** | Frankfurt Airport | $50.037^\circ\text{N}, 8.562^\circ\text{E}$ | Intersecting dual runways, terminals, road interchanges |
| **Airport** | Dubai Al Maktoum | $24.896^\circ\text{N}, 55.161^\circ\text{E}$ | Massive desert logistics hub, parallel taxiways |
| **Airport** | Tokyo Haneda | $35.549^\circ\text{N}, 139.779^\circ\text{E}$ | Coastal reclaimed runways, Tokyo Bay shipping channels |
| **Mountain Outpost** | Siachen Ladakh Himalayas| $34.800^\circ\text{N}, 77.000^\circ\text{E}$ | Glacial moraines, high-relief ridges, extreme shadow/snow |
| **Mountain Outpost** | Mont Blanc Alps | $45.832^\circ\text{N}, 6.865^\circ\text{E}$ | Alpine peaks, snow fields, glacial valleys |
| **Dense Urban** | New York City Manhattan | $40.712^\circ\text{N}, -74.006^\circ\text{W}$ | Rectangular street grid, high-rise building perimeters |
| **Dense Urban** | Paris Metropolitan | $48.856^\circ\text{N}, 2.352^\circ\text{E}$ | Radial boulevards, river corridors, historic architecture |
| **Dense Urban** | New Delhi NCR | $28.613^\circ\text{N}, 77.209^\circ\text{E}$ | High-density urban sprawl, arterial ring roads |
| **Vegetation Canopy**| Amazon Basin Manaus | $-3.119^\circ\text{S}, -60.021^\circ\text{W}$ | Dense tropical rainforest canopy, river meanders |
| **Vegetation Canopy**| Iowa Agricultural Farmland| $42.030^\circ\text{N}, -93.581^\circ\text{W}$ | Orthogonal crop parcels, rural drainage grids |
| **Strategic Maritime**| Suez Canal | $30.585^\circ\text{N}, 32.560^\circ\text{E}$ | Narrow shipping waterway, desert coast contours |
| **Strategic Maritime**| San Francisco Bay | $37.819^\circ\text{N}, -122.478^\circ\text{W}$ | Coastal bluffs, suspension bridges, port terminals |

---

## 🔬 Extracted Spectral Bands

| Index | Band | Description | Native Resolution | Central Wavelength | Bandwidth |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **B04** | Red | 10m | 665 nm | 31 nm |
| **1** | **B03** | Green | 10m | 560 nm | 36 nm |
| **2** | **B02** | Blue | 10m | 490 nm | 65 nm |
| **3** | **B08** | Near-Infrared (NIR) | 10m | 842 nm | 115 nm |

---

## 🛡️ Triple-Gate Quality Filtration Pipeline

1. **No-Data Rejection**: Rejects any candidate patch containing nodata or border pixels ($\text{pixel} \le 0$).
2. **Cloud Top Saturation Gate**: In Sentinel-2 L2A surface reflectance, thick clouds exceed reflectance values of $3,800$ (representing $> 38\%$ surface albedo). Patches containing dense cloud clusters are automatically rejected.
3. **Texture Variance Gate**: Computes average standard deviation $\sigma_{\text{avg}} = \frac{1}{4} \sum_{c=1}^4 \text{std}(X_c)$. Patches with $\sigma_{\text{avg}} < 25.0$ (uniform open water or flat desert) are rejected to maximize structural entropy.

---

## 🛰️ Sensor Optical Point Spread Function (PSF) Degradation

Rather than using naive bicubic downsampling, the pipeline simulates realistic optical aperture diffraction:

$$\mathbf{X}_{\text{LR}} = \Big(\mathbf{X}_{\text{HR}} \circledast \mathbf{k}_{\text{PSF}}\Big) \downarrow_{s=4}$$

$$\mathbf{k}_{\text{PSF}}(u, v) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{u^2 + v^2}{2\sigma^2}\right), \quad \sigma \sim \mathcal{U}(0.6, 1.4)$$

The $7 \times 7$ anisotropic Gaussian PSF kernel is convolved depthwise across each of the 4 spectral bands before $4\times$ decimation ($128 \times 128 \to 32 \times 32$), generating real-world sensor degradation dynamics.

---

## 📁 Pipeline Module Architecture

```
pipeline/
├── cdse_client.py         # CDSE OData API client (OAuth2 authentication, streaming band download)
├── patch_extractor.py     # 4-band patch slicing with cloud, nodata, and texture variance filters
├── build_dataset.py       # Multi-scene downloader & patch accumulator across diverse AOIs
├── prepare_splits.py      # Sensor PSF Gaussian blur + 4x downsampling (HR/LR)
├── verify_dataset.py      # Statistical verification and visual RGB/CIR preview generator
├── .env.example           # Configuration template for CDSE credentials
└── README.md
```

---

## 🚀 Quickstart

### 1. Configure Copernicus Credentials
1. Register for a free Copernicus Data Space account at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/).
2. Copy `.env.example` to `.env`:
   ```bash
   cp pipeline/.env.example pipeline/.env
   ```
3. Enter your CDSE credentials.

### 2. Survey AOIs (Dry Run)
```bash
python pipeline/build_dataset.py --dry-run
```

### 3. Ingest & Extract Patches
```bash
python pipeline/build_dataset.py --target-count 8000 --max-cloud 2.0
```

### 4. Generate Sensor PSF Degradations & Splits
```bash
python pipeline/prepare_splits.py --scale 4 --sigma-min 0.6 --sigma-max 1.4
```

### 5. Verify Dataset Quality & Visual Previews
```bash
python pipeline/verify_dataset.py --dir ./data/4_Channel_Data --preview ./sample_patches_preview.png
```
