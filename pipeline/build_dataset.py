"""
Sentinel-2 4-Channel Dataset Builder
Downloads cloud-free scenes from CDSE OData API across diverse global locations,
extracts 4-channel (R, G, B, IR) 128x128 patches, and saves them as .npy arrays.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent))
from cdse_client import CDSEClient
from patch_extractor import PatchExtractor


# Curated list of diverse geographic targets
TARGET_AOIS = [
    {
        "category": "military_base",
        "name": "Norfolk Naval Station",
        "lat": 36.945,
        "lon": -76.326,
        "description": "Aircraft carriers, naval dry docks, coastal infrastructure",
    },
    {
        "category": "military_base",
        "name": "Ramstein Air Base",
        "lat": 49.437,
        "lon": 7.600,
        "description": "NATO military runways, hangars, forested base perimeters",
    },
    {
        "category": "military_base",
        "name": "Nellis AFB Nevada",
        "lat": 36.236,
        "lon": -115.034,
        "description": "Desert military air station, test ranges, barren terrain",
    },
    {
        "category": "military_base",
        "name": "Hindon Air Force Station",
        "lat": 28.706,
        "lon": 77.360,
        "description": "Major military airfield, hangars, perimeter installations",
    },
    {
        "category": "airport",
        "name": "Frankfurt Airport",
        "lat": 50.037,
        "lon": 8.562,
        "description": "Major European airport, intersecting runways, terminals, roads",
    },
    {
        "category": "airport",
        "name": "Dubai Al Maktoum",
        "lat": 24.896,
        "lon": 55.161,
        "description": "Massive desert airport hub, taxiways, modern transit corridors",
    },
    {
        "category": "airport",
        "name": "Tokyo Haneda",
        "lat": 35.549,
        "lon": 139.779,
        "description": "Coastal runways, Tokyo Bay, shipping channels, port infrastructure",
    },
    {
        "category": "mountain_outpost",
        "name": "Siachen Ladakh Himalayas",
        "lat": 34.800,
        "lon": 77.000,
        "description": "High-altitude Himalayan ridges, glaciers, outpost topography",
    },
    {
        "category": "mountain_outpost",
        "name": "Mont Blanc Alps",
        "lat": 45.832,
        "lon": 6.865,
        "description": "Alpine peaks, snow fields, glacial valleys, mountain passes",
    },
    {
        "category": "city_roads",
        "name": "New York City",
        "lat": 40.712,
        "lon": -74.006,
        "description": "Dense urban grid, bridges, highways, waterfronts",
    },
    {
        "category": "city_roads",
        "name": "Paris",
        "lat": 48.856,
        "lon": 2.352,
        "description": "European urban radial roads, boulevards, river corridors",
    },
    {
        "category": "city_roads",
        "name": "New Delhi NCR",
        "lat": 28.613,
        "lon": 77.209,
        "description": "Dense Asian urban sprawl, expressways, arterial ring roads",
    },
    {
        "category": "vegetation",
        "name": "Amazon Basin Manaus",
        "lat": -3.119,
        "lon": -60.021,
        "description": "Dense rainforest canopy, river meanders, undisturbed vegetation",
    },
    {
        "category": "vegetation",
        "name": "Iowa Farmlands",
        "lat": 42.030,
        "lon": -93.581,
        "description": "Geometric agricultural crop fields, rural road grids",
    },
    {
        "category": "water_ports",
        "name": "Suez Canal",
        "lat": 30.585,
        "lon": 32.560,
        "description": "Strategic maritime waterway, desert coasts, ship passages",
    },
    {
        "category": "water_ports",
        "name": "San Francisco Bay",
        "lat": 37.819,
        "lon": -122.478,
        "description": "Bay water, coastal bluffs, bridges, port facilities",
    },
]


def load_metadata(meta_file: Path) -> Dict:
    if meta_file.exists():
        try:
            with open(meta_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "total_patches": 0,
        "target_count": 8000,
        "patch_shape": [4, 128, 128],
        "channels": ["Red (B04)", "Green (B03)", "Blue (B02)", "NIR (B08)"],
        "dtype": "uint16",
        "processed_products": [],
        "category_counts": {},
    }


def save_metadata(meta_file: Path, data: Dict):
    with open(meta_file, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build 8,000 patch 4-channel dataset from CDSE.")
    parser.add_argument("--target-count", type=int, default=8000, help="Total number of patches to generate.")
    parser.add_argument("--patches-per-tile", type=int, default=500, help="Patches to extract per satellite scene.")
    parser.add_argument("--max-cloud", type=float, default=2.0, help="Maximum scene cloud cover percentage.")
    parser.add_argument("--dry-run", action="store_true", help="Only query products without downloading.")
    args = parser.parse_args()

    # Load environment (.env in script dir or parent dir)
    script_dir = Path(__file__).resolve().parent
    env_file = script_dir / ".env"
    if not env_file.exists():
        env_file = script_dir.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    default_output = str(script_dir / "data" / "4_Channel_Data")
    output_dir = Path(os.environ.get("DATASET_OUTPUT_DIR", default_output))
    temp_dir = script_dir / "temp_tiles"
    meta_file = output_dir / "metadata.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(meta_file)
    current_count = metadata.get("total_patches", 0)

    # Check actual existing .npy files in output_dir
    existing_files = list(output_dir.glob("*.npy"))
    if len(existing_files) > current_count:
        current_count = len(existing_files)
        metadata["total_patches"] = current_count

    print("=== Dataset Builder Initialized ===", flush=True)
    print(f"Target count: {args.target_count}", flush=True)
    print(f"Current count: {current_count}", flush=True)
    print(f"Remaining: {max(0, args.target_count - current_count)}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    client = CDSEClient()
    extractor = PatchExtractor(patch_size=128)

    if args.dry_run:
        print("\n--- Running Dry-Run Survey Across All 16 Target AOIs ---", flush=True)
        for i, aoi in enumerate(TARGET_AOIS, 1):
            cat = aoi["category"]
            name = aoi["name"]
            lat = aoi["lat"]
            lon = aoi["lon"]
            print(f"[{i}/{len(TARGET_AOIS)}] Testing AOI: {name} ({cat}) at ({lat}, {lon})...", end=" ", flush=True)
            try:
                prods = client.search_products(lat=lat, lon=lon, max_cloud=args.max_cloud, limit=2)
                if not prods and args.max_cloud < 5.0:
                    prods = client.search_products(lat=lat, lon=lon, max_cloud=5.0, limit=2)
                if prods:
                    print(f"FOUND ({len(prods)} products) -> {prods[0]['name']}", flush=True)
                else:
                    print("NO PRODUCTS FOUND", flush=True)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
        print("--- Dry-Run Survey Complete ---", flush=True)
        return

    if current_count >= args.target_count:
        print("Target already achieved! Run verify_dataset.py to inspect.", flush=True)
        return

    aoi_index = 0
    aois = TARGET_AOIS.copy()

    while current_count < args.target_count:
        aoi = aois[aoi_index % len(aois)]
        aoi_index += 1
        cat = aoi["category"]
        name = aoi["name"]
        lat = aoi["lat"]
        lon = aoi["lon"]

        print(f"\n[{aoi_index}] AOI: {name} ({cat}) at ({lat}, {lon})", flush=True)
        
        # Try search with progressive cloud threshold if necessary
        prods = []
        for max_c in [args.max_cloud, 5.0, 10.0]:
            try:
                prods = client.search_products(lat=lat, lon=lon, max_cloud=max_c, limit=5)
                if prods:
                    break
            except Exception as e:
                print(f"  Query error at max_cloud={max_c}: {e}", flush=True)

        candidate_prod = None
        for p in prods:
            if p["id"] not in metadata.get("processed_products", []):
                candidate_prod = p
                break

        if not candidate_prod:
            print(f"  No new products found for {name}, moving to next AOI...", flush=True)
            continue

        pid = candidate_prod["id"]
        pname = candidate_prod["name"]
        print(f"  Selected Product: {pname} (ID: {pid})", flush=True)

        # Download and extract bands
        temp_zip = temp_dir / f"{pid}.zip"
        bands_dir = temp_dir / f"{pid}_bands"

        try:
            band_paths = client.download_and_extract_bands(
                product_id=pid,
                extract_dir=bands_dir,
                temp_zip_path=temp_zip,
                bands=["B02", "B03", "B04", "B08"],
            )

            if len(band_paths) < 4:
                print("  Missing required bands, skipping scene...", flush=True)
                continue

            # Load scene into memory
            print("  Loading 10m bands into memory (Red, Green, Blue, NIR)...", flush=True)
            scene = extractor.load_scene_bands(band_paths)
            print(f"  Scene shape loaded: {scene.shape} ({scene.dtype})", flush=True)

            # Determine patches to extract
            needed = args.target_count - current_count
            to_extract = min(args.patches_per_tile, needed)

            print(f"  Extracting up to {to_extract} valid 128x128 patches...", flush=True)
            extracted_num, saved_paths = extractor.extract_patches_from_scene(
                scene=scene,
                target_count=to_extract,
                category=cat,
                output_dir=output_dir,
                start_index=current_count,
            )

            current_count += extracted_num
            metadata["total_patches"] = current_count
            metadata["processed_products"].append(pid)
            metadata["category_counts"][cat] = metadata["category_counts"].get(cat, 0) + extracted_num
            save_metadata(meta_file, metadata)

            print(f"  Extracted {extracted_num} patches. Progress: {current_count}/{args.target_count}", flush=True)

        except Exception as e:
            print(f"  Error processing product {pid}: {e}", flush=True)
            import traceback
            traceback.print_exc()

        finally:
            # Clean up intermediate band files immediately
            if bands_dir.exists():
                print(f"  Cleaning up band files: {bands_dir.name}", flush=True)
                shutil.rmtree(bands_dir, ignore_errors=True)
            if temp_zip.exists():
                temp_zip.unlink(missing_ok=True)

    print(f"\n=== Dataset Builder Complete: {current_count} patches generated ===", flush=True)


if __name__ == "__main__":
    main()
