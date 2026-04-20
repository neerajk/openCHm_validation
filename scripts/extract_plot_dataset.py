#!/usr/bin/env python3
"""Extract per-plot pixel datasets from CHM GeoTIFFs for statistical analysis.

For each field plot, this script:
1. Reads the CHM GeoTIFF corresponding to the plot's tile
2. Computes the bounding box around the plot center (12.5m x 12.5m square)
3. Extracts all pixels within the bounding box
4. Saves to a structured dataset directory
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from tqdm import tqdm

from validation_common import (
    load_field_plots, 
    load_manifest,
    slippy_pixel_xy,
    meters_per_pixel_webmercator
)

def parse_tile_key(tile_key: str) -> tuple[int, int, int]:
    """Extract zoom, x, and y from a tile key like 'z18_187906_107749'."""
    match = re.match(r"z(\d+)_(\d+)_(\d+)", tile_key)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    raise ValueError(f"Invalid tile_key format: {tile_key}")


def compute_bbox_pixels(
    lon: float,
    lat: float,
    tile_key: str,
    plot_size_m: float = 12.5,
) -> tuple[int, int, int, int]:
    """Compute pixel-space bounding box using strict slippy-map web mercator logic.
    
    Returns (col_min, row_min, col_max, row_max) inclusive bounds.
    """
    zoom, tile_x, tile_y = parse_tile_key(tile_key)
    
    # 1. Get global web mercator pixel coordinates for this lat/lon
    global_px, global_py = slippy_pixel_xy(lat, lon, zoom)
    
    # 2. Convert to local coordinates within the 512x512 stitched patch
    #    The patch starts exactly at tile_x, tile_y (each tile is 256px wide)
    local_col = global_px - (tile_x * 256.0)
    local_row = global_py - (tile_y * 256.0)
    
    # 3. Convert field plot size from meters to pixels at this latitude
    m_per_px = meters_per_pixel_webmercator(lat, zoom)
    half_px = (plot_size_m / m_per_px) / 2.0
    
    col_min = int(local_col - half_px)
    col_max = int(local_col + half_px)
    row_min = int(local_row - half_px)
    row_max = int(local_row + half_px)

    return col_min, row_min, col_max, row_max


def extract_chm_bbox(
    chm_path: Path,
    lon: float,
    lat: float,
    tile_key: str,
    plot_size_m: float = 12.5,
) -> tuple[np.ndarray, dict] | None:
    """Extract CHM pixels within a bounding box around the plot center."""
    if not chm_path.exists():
        return None

    with rasterio.open(chm_path) as src:
        col_min, row_min, col_max, row_max = compute_bbox_pixels(
            lon, lat, tile_key, plot_size_m
        )

        # Ensure bounds are within the 512x512 raster
        c_min = max(0, col_min)
        r_min = max(0, row_min)
        c_max = min(col_max, src.width)
        r_max = min(row_max, src.height)

        if c_min >= c_max or r_min >= r_max:
            return None

        window = rasterio.windows.Window(c_min, r_min, c_max - c_min, r_max - r_min)
        chm_data = src.read(1, window=window)

        nodata = src.nodata
        if nodata is not None:
            valid_mask = np.isfinite(chm_data) & (chm_data != nodata)
        else:
            valid_mask = np.isfinite(chm_data)

        valid_pixels = chm_data[valid_mask]

        if len(valid_pixels) == 0:
            return None

        metadata = {
            "bbox_pixels": {
                "col_min": int(c_min),
                "row_min": int(r_min),
                "col_max": int(c_max),
                "row_max": int(r_max),
            },
            "n_pixels_total": int(window.width * window.height),
            "n_pixels_valid": int(len(valid_pixels)),
            "coverage_ratio": float(len(valid_pixels) / (window.width * window.height)),
            "height_stats": {
                "mean": float(np.mean(valid_pixels)),
                "std": float(np.std(valid_pixels)),
                "min": float(np.min(valid_pixels)),
                "max": float(np.max(valid_pixels)),
                "median": float(np.median(valid_pixels)),
            },
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform)[:6],
        }

        return chm_data, metadata


def create_bbox_overlay(
    chm_path: Path,
    lon: float,
    lat: float,
    tile_key: str,
    plot_size_m: float = 12.5,
) -> Image.Image | None:
    """Create a visualisation showing the bbox on the CHM."""
    if not chm_path.exists():
        return None

    with rasterio.open(chm_path) as src:
        chm_data = src.read(1)
        nodata = src.nodata

        valid = np.isfinite(chm_data)
        if nodata is not None:
            valid &= chm_data != nodata

        if not valid.any():
            return None

        chm_valid = chm_data[valid]
        chm_min, chm_max = np.percentile(chm_valid, [2, 98])

        normalized = (chm_data - chm_min) / (chm_max - chm_min + 1e-8)
        normalized = np.clip(normalized, 0, 1)

        import matplotlib.pyplot as plt
        cmap = plt.cm.viridis
        rgb = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
        rgb[~valid] = 0

        img = Image.fromarray(rgb)

        col_min, row_min, col_max, row_max = compute_bbox_pixels(
            lon, lat, tile_key, plot_size_m
        )

        draw = ImageDraw.Draw(img)
        draw.rectangle([col_min, row_min, col_max, row_max], outline="red", width=2)

        center_col = (col_min + col_max) / 2
        center_row = (row_min + row_max) / 2
        r = 3
        draw.ellipse([center_col - r, center_row - r, center_col + r, center_row + r], fill="yellow")

        return img


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-plot pixel datasets from CHM GeoTIFFs"
    )
    parser.add_argument("--plots_geojson", default="data/input/field_plots.geojson")
    parser.add_argument("--manifest", default="data/input/plot_manifest.json")
    parser.add_argument("--chm_dir", default="data/output/esri_results")
    parser.add_argument("--output_dir", default="scripts/statistical_analysis/data")
    parser.add_argument("--plot_size_m", type=float, default=12.5)
    parser.add_argument("--min_coverage", type=float, default=0.5)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10000)
    args = parser.parse_args()

    plots = load_field_plots(args.plots_geojson)
    if not plots:
        raise SystemExit(f"No plots found in {args.plots_geojson}")

    manifest = load_manifest(args.manifest)
    selected = [p for p in plots if args.start <= p.sr <= args.end]
    print(f"Processing {len(selected)} plots (sr {args.start}-{args.end})")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    chm_root = Path(args.chm_dir)

    results_summary = []

    for plot in tqdm(selected, desc="Extracting plot datasets", unit="plot"):
        sr = plot.sr
        manifest_entry = manifest.get(sr, {})
        tile_key = manifest_entry.get("tile_key")

        if not tile_key:
            tqdm.write(f"  sr={sr}: not in manifest")
            continue

        chm_path = chm_root / f"esri_512_{tile_key}_CHM.tif"

        if not chm_path.exists():
            tqdm.write(f"  sr={sr}: CHM not found ({chm_path.name})")
            continue

        # Pass tile_key instead of relying on the raster's transform
        result = extract_chm_bbox(chm_path, plot.lon, plot.lat, tile_key, args.plot_size_m)

        if result is None:
            tqdm.write(f"  sr={sr}: extraction failed (Out of bounds or all NoData)")
            continue

        chm_pixels, metadata = result

        if metadata["coverage_ratio"] < args.min_coverage:
            tqdm.write(f"  sr={sr}: low coverage ({metadata['coverage_ratio']:.1%})")
            continue

        plot_out_dir = out_root / f"plot_{sr:03d}"
        plot_out_dir.mkdir(parents=True, exist_ok=True)

        np.save(plot_out_dir / "pixels.npy", chm_pixels)

        plot_metadata = {
            "sr": sr,
            "lat": plot.lat,
            "lon": plot.lon,
            "h_avg_field": plot.h_avg,
            "h_pred_mean": metadata["height_stats"]["mean"],
            "h_pred_std": metadata["height_stats"]["std"],
            "h_pred_min": metadata["height_stats"]["min"],
            "h_pred_max": metadata["height_stats"]["max"],
            "h_pred_median": metadata["height_stats"]["median"],
            "bbox": metadata["bbox_pixels"],
            "n_pixels": metadata["n_pixels_valid"],
            "coverage_ratio": metadata["coverage_ratio"],
            "tile_key": tile_key,
            "chm_file": chm_path.name,
            "crs": metadata["crs"],
        }

        with open(plot_out_dir / "metadata.json", "w") as f:
            json.dump(plot_metadata, f, indent=2)

        overlay = create_bbox_overlay(chm_path, plot.lon, plot.lat, tile_key, args.plot_size_m)
        if overlay is not None:
            overlay.save(plot_out_dir / "bbox_overlay.png")

        results_summary.append(plot_metadata)

    if results_summary:
        summary_path = out_root / "dataset_summary.csv"
        fieldnames = [
            "sr", "lat", "lon", "h_avg_field", "h_pred_mean", "h_pred_std",
            "h_pred_min", "h_pred_max", "h_pred_median", "n_pixels", "coverage_ratio", "tile_key"
        ]

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in results_summary:
                writer.writerow(row)

        print(f"\nDataset summary: {summary_path}")
        print(f"Total plots extracted: {len(results_summary)}")
        print(f"Output directory: {out_root}/")

if __name__ == "__main__":
    main()