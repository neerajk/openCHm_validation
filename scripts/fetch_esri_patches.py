#!/usr/bin/env python3
"""Download 512×512 RGB patches from ESRI World Imagery tiles.

Filename convention: esri_512_z{zoom}_{tile_x}_{tile_y}.png
The tile coords are unique — no extra identifier needed.
"""

from __future__ import annotations

import argparse
import io
import math
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
)


def latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to slippy-map tile indices at the given zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile


def fetch_single_tile(session: requests.Session, zoom: int, x: int, y: int) -> Image.Image | None:
    """Download one 256×256 tile. Returns None on failure (non-fatal — blank tile used instead)."""
    url = ESRI_TILE_URL.format(zoom=zoom, x=x, y=y)
    try:
        r = session.get(url, headers={"User-Agent": "Mozilla/5.0 (CHMv2 Validation)"}, timeout=20)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except requests.RequestException as exc:
        tqdm.write(f"  WARNING: tile z={zoom} x={x} y={y}: {exc}")
        return None
    except Exception as exc:
        tqdm.write(f"  WARNING: bad image z={zoom} x={x} y={y}: {exc}")
        return None


def stitch_512_patch(
    session: requests.Session,
    zoom: int,
    start_x: int,
    start_y: int,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Fetch a 2×2 block of 256px tiles and stitch into one 512×512 PNG.

    Output: esri_512_z{zoom}_{start_x}_{start_y}.png
    Skips the network fetch if the file already exists (pass overwrite=True to force).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = output_dir / f"esri_512_z{zoom}_{start_x}_{start_y}.png"
    if filename.exists() and not overwrite:
        return filename  # already on disk — skip fetch

    patch = Image.new("RGB", (512, 512))
    # 2×2 grid: (paste_x, paste_y, tile_dx, tile_dy)
    for px, py, dx, dy in [(0, 0, 0, 0), (256, 0, 1, 0), (0, 256, 0, 1), (256, 256, 1, 1)]:
        tile = fetch_single_tile(session, zoom, start_x + dx, start_y + dy)
        if tile is not None:
            patch.paste(tile, (px, py))

    patch.save(filename, format="PNG")
    return filename


def fetch_area_by_bbox(bbox: list[float], zoom: int, output_dir: str | Path) -> None:
    """Fetch all 512px patches covering a bounding box [min_lon, min_lat, max_lon, max_lat]."""
    min_lon, min_lat, max_lon, max_lat = bbox

    # Tile range covering the bbox (note: y increases southward)
    min_x, min_y = latlon_to_tile_xy(max_lat, min_lon, zoom)
    max_x, max_y = latlon_to_tile_xy(min_lat, max_lon, zoom)

    # Build 2×2-step grid covering the bbox
    patch_coords = [(x, y) for x in range(min_x, max_x + 1, 2) for y in range(min_y, max_y + 1, 2)]
    print(f"Bbox fetch: {len(patch_coords)} patches (tile grid x={min_x}..{max_x}, y={min_y}..{max_y})")

    with requests.Session() as session:
        for x, y in tqdm(patch_coords, desc="Stitching patches", unit="patch"):
            stitch_512_patch(session, zoom, x, y, output_dir)

    print(f"Done. Saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 512×512 ESRI World Imagery patches")
    parser.add_argument("--lat", type=float, help="Latitude (for single-patch mode)")
    parser.add_argument("--lon", type=float, help="Longitude (for single-patch mode)")
    parser.add_argument(
        "--bbox",
        type=float, nargs=4,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box for a grid of patches",
    )
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--out_dir", type=str, default="data/input/esri_patches")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.bbox:
        fetch_area_by_bbox(args.bbox, args.zoom, args.out_dir)
        return

    if args.lat is None or args.lon is None:
        raise SystemExit("Provide either --bbox or both --lat and --lon")

    start_x, start_y = latlon_to_tile_xy(args.lat, args.lon, args.zoom)
    print(f"Fetching patch at lat={args.lat}, lon={args.lon} (z={args.zoom}, tile={start_x},{start_y})")
    with requests.Session() as session:
        path = stitch_512_patch(session, args.zoom, start_x, start_y, args.out_dir, overwrite=args.overwrite)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
