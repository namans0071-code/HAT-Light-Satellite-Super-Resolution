"""
CDSE OData API Client for Sentinel-2 L2A Imagery
Handles authentication, spatial queries, and streaming downloads.
"""

import os
import time
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import argparse

# ==============================================================================
# IN-CODE CREDENTIALS CONFIGURATION (OPTIONAL)
# You can fill these in if you want to hardcode credentials for private use.
# By default, leave them empty and pass via CLI flags or environment variables.
# ==============================================================================
IN_CODE_CDSE_USERNAME: str = ""
IN_CODE_CDSE_PASSWORD: str = ""


class CDSEClient:
    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    ODATA_CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    ODATA_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
    CLIENT_ID = "cdse-public"

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        self.username = (
            username
            or (IN_CODE_CDSE_USERNAME.strip() if IN_CODE_CDSE_USERNAME.strip() else None)
            or os.environ.get("CDSE_USERNAME")
        )
        self.password = (
            password
            or (IN_CODE_CDSE_PASSWORD.strip() if IN_CODE_CDSE_PASSWORD.strip() else None)
            or os.environ.get("CDSE_PASSWORD")
        )

        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0

    def has_credentials(self) -> bool:
        return bool(self.username and self.password)

    def get_token(self, force_refresh: bool = False) -> str:
        """Obtain or refresh Copernicus Keycloak OAuth2 bearer token."""
        if not self.has_credentials():
            raise ValueError(
                "Missing CDSE credentials! Pass via CLI (--username / -u, --password / -p), "
                "fill IN_CODE_CDSE_USERNAME in cdse_client.py, or set CDSE_USERNAME in .env"
            )

        now = time.time()
        if not force_refresh and self._token and now < (self._token_expires_at - 60):
            return self._token

        data = {
            "client_id": self.CLIENT_ID,
            "username": self.username,
            "password": self.password,
            "grant_type": "password",
        }
        resp = requests.post(self.TOKEN_URL, data=data, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        self._token = payload["access_token"]
        expires_in = payload.get("expires_in", 1800)
        self._token_expires_at = now + expires_in
        return self._token

    def get_auth_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.get_token()}"}

    def search_products(
        self,
        lat: float,
        lon: float,
        max_cloud: float = 2.0,
        limit: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        """Search for Sentinel-2 Level-2A products intersecting coordinates with cloud filter."""
        filters = [
            "Collection/Name eq 'SENTINEL-2'",
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')",
            f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value lt {max_cloud:.2f})",
            f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon:.6f} {lat:.6f})')",
        ]

        if start_date:
            filters.append(f"ContentDate/Start gt {start_date}")
        if end_date:
            filters.append(f"ContentDate/Start lt {end_date}")

        params = {
            "$filter": " and ".join(filters),
            "$top": limit,
            "$orderby": "ContentDate/Start desc",
        }

        for attempt in range(1, 4):
            try:
                resp = requests.get(self.ODATA_CATALOG_URL, params=params, timeout=60)
                resp.raise_for_status()
                results = resp.json().get("value", [])
                break
            except Exception as e:
                if attempt == 3:
                    raise
                time.sleep(3 * attempt)

        # Parse cloud cover from attributes if available
        parsed = []
        for item in results:
            cloud_val = None
            for attr in item.get("Attributes", []):
                if attr.get("Name") == "cloudCover":
                    cloud_val = attr.get("Value")
                    break
            parsed.append({
                "id": item["Id"],
                "name": item["Name"],
                "content_length": item.get("ContentLength", 0),
                "date": item.get("ContentDate", {}).get("Start"),
                "cloud_cover": cloud_val,
                "s3_path": item.get("S3Path"),
            })
        return parsed

    def download_product(self, product_id: str, dest_path: Path, max_retries: int = 3) -> Path:
        """Stream download full product zip archive with progress bar and retry."""
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dest_path.with_suffix(dest_path.suffix + ".part")

        url = f"{self.ODATA_DOWNLOAD_URL}({product_id})/$value"

        for attempt in range(1, max_retries + 1):
            try:
                headers = self.get_auth_headers()
                downloaded = 0
                if part_path.exists():
                    downloaded = part_path.stat().st_size
                    headers["Range"] = f"bytes={downloaded}-"

                resp = requests.get(url, headers=headers, stream=True, timeout=60)
                if resp.status_code == 416:  # Range not satisfiable (already finished)
                    if part_path.exists():
                        part_path.rename(dest_path)
                    return dest_path

                resp.raise_for_status()

                total_size = int(resp.headers.get("content-length", 0)) + downloaded
                mode = "ab" if downloaded > 0 and resp.status_code == 206 else "wb"
                if mode == "wb":
                    downloaded = 0

                with open(part_path, mode) as f, tqdm(
                    desc=dest_path.name,
                    total=total_size,
                    initial=downloaded,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

                if part_path.exists():
                    part_path.rename(dest_path)
                return dest_path

            except Exception as e:
                print(f"[Attempt {attempt}/{max_retries}] Download failed: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(5)
        return dest_path

    def download_and_extract_bands(
        self,
        product_id: str,
        extract_dir: Path,
        temp_zip_path: Path,
        bands: List[str] = ["B02", "B03", "B04", "B08"],
    ) -> Dict[str, Path]:
        """
        Download product zip, extract only the required 10m bands,
        then delete the zip file immediately to conserve disk space.
        """
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading product {product_id}...")
        self.download_product(product_id, temp_zip_path)

        band_paths: Dict[str, Path] = {}
        try:
            print("Extracting 10m JP2 band files from archive...")
            with zipfile.ZipFile(temp_zip_path, "r") as zf:
                all_files = zf.namelist()
                for band in bands:
                    # In Sentinel-2 L2A, 10m bands are named e.g. T..._B04_10m.jp2
                    matching = [
                        f for f in all_files
                        if f.endswith(".jp2")
                        and f"_{band}_10m" in f
                        and "R10m" in f
                    ]
                    if not matching:
                        # Fallback for alternative naming
                        matching = [
                            f for f in all_files
                            if f.endswith(".jp2") and f"_{band}" in f and ("10m" in f or "R10m" in f)
                        ]

                    if matching:
                        target_entry = matching[0]
                        filename = Path(target_entry).name
                        out_file = extract_dir / filename
                        with zf.open(target_entry) as src, open(out_file, "wb") as dst:
                            dst.write(src.read())
                        band_paths[band] = out_file
                        print(f"  Extracted {band} -> {filename}")
                    else:
                        print(f"  WARNING: Band {band} not found in zip archive!")
        finally:
            if temp_zip_path.exists():
                print(f"Cleaning up temporary archive: {temp_zip_path.name}")
                try:
                    temp_zip_path.unlink()
                except Exception as e:
                    print(f"Failed to delete {temp_zip_path}: {e}")

        return band_paths


if __name__ == "__main__":
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    parser = argparse.ArgumentParser(description="Copernicus Data Space Ecosystem (CDSE) Ingestion Client")
    parser.add_argument("-u", "--username", type=str, default=None, help="CDSE username / email")
    parser.add_argument("-p", "--password", type=str, default=None, help="CDSE account password")
    parser.add_argument("--lat", type=float, default=50.037, help="Latitude (Default: Frankfurt Airport)")
    parser.add_argument("--lon", type=float, default=8.562, help="Longitude (Default: Frankfurt Airport)")
    parser.add_argument("--max-cloud", type=float, default=2.0, help="Maximum cloud cover percentage")
    parser.add_argument("--limit", type=int, default=2, help="Number of products to query")
    parser.add_argument("--check-token", action="store_true", help="Authenticate and verify token")
    args = parser.parse_args()

    client = CDSEClient(username=args.username, password=args.password)

    if args.check_token or client.has_credentials():
        try:
            token = client.get_token()
            print("Authentication successful! Token valid, length:", len(token))
        except Exception as e:
            print("Authentication check failed:", e)

    print(f"Searching cloud-free products for Lat: {args.lat}, Lon: {args.lon} (Max cloud: {args.max_cloud}%)...")
    results = client.search_products(lat=args.lat, lon=args.lon, max_cloud=args.max_cloud, limit=args.limit)
    print(f"Found {len(results)} products:")
    for r in results:
        print(f"  - {r['id']} | {r['name']} | Cloud: {r.get('cloud_cover')}%")
