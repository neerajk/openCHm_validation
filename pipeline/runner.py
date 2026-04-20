"""
pipeline/runner.py
==================
Pipeline orchestration:
- STAC full-scene mode
- ESRI patch mode
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine, from_bounds
from tqdm import tqdm

from .inference import run_patch_inference
from .model import load_model_and_processor
from .tiling import extract_patches, load_rgb_image, mosaic_patches
from .visualise import _embedding_pca_rgb, mosaic_visual, per_patch_visual, save_geotiff

_TILE_RE = re.compile(r"_z(?P<zoom>\d+)_(?P<x>\d+)_(?P<y>\d+)$")


def _parse_zoom_xy_from_stem(stem: str) -> tuple[int, int, int] | None:
    """Parse z/x/y from names like ..._z18_187906_107749."""
    match = _TILE_RE.search(stem)
    if not match:
        return None
    return int(match.group("zoom")), int(match.group("x")), int(match.group("y"))


def _esri_patch_transform(stem: str, width: int, height: int) -> tuple[Affine, str | None]:
    """Create EPSG:3857 transform from ESRI tile metadata in filename."""
    parsed = _parse_zoom_xy_from_stem(stem)
    if parsed is None:
        return Affine.identity(), None

    zoom, tile_x, tile_y = parsed
    tile_size = 256.0
    earth_radius = 6378137.0
    origin_shift = math.pi * earth_radius
    resolution = (2.0 * math.pi * earth_radius) / (tile_size * (2**zoom))

    tiles_w = width / tile_size
    tiles_h = height / tile_size

    minx = tile_x * tile_size * resolution - origin_shift
    maxx = (tile_x + tiles_w) * tile_size * resolution - origin_shift
    maxy = origin_shift - tile_y * tile_size * resolution
    miny = origin_shift - (tile_y + tiles_h) * tile_size * resolution

    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    return transform, "EPSG:3857"


class StacInferencePipeline:
    """Standard pipeline: full-scene GeoTIFF load, tile, infer, mosaic."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self) -> None:
        cfg = self.cfg

        out_dir = Path(cfg["output"]["output_dir"])
        patches_dir = out_dir / "patches"
        out_dir.mkdir(parents=True, exist_ok=True)
        patches_dir.mkdir(parents=True, exist_ok=True)

        model, processor, device = load_model_and_processor(cfg)

        rgb, geo_profile = load_rgb_image(cfg)
        original_shape = rgb.shape[:2]

        patch_size = cfg["tiling"]["patch_size"]
        overlap = cfg["tiling"]["overlap"]
        blend_mode = cfg["tiling"].get("blend_mode", "linear")

        patches, rgb_padded = extract_patches(rgb, patch_size, overlap)
        padded_shape = rgb_padded.shape[:2]

        predictions, embeddings = run_patch_inference(patches, model, processor, device, cfg)

        if cfg["output"].get("save_patch_visuals", True):
            print("[runner] Saving per-patch visualisations...")
            for patch, pred, emb in zip(patches, predictions, embeddings):
                per_patch_visual(patch, pred, emb, patches_dir, cfg)
            print(f"[runner] Patch visuals -> {patches_dir}/")

        print("[runner] Mosaicking patches...")
        mosaic = mosaic_patches(
            patches,
            predictions,
            padded_shape,
            original_shape,
            overlap,
            blend_mode,
        )
        print(f"[runner] Mosaic shape: {mosaic.shape}  min={mosaic.min():.2f}m  max={mosaic.max():.2f}m")

        if cfg["output"].get("save_mosaic_tif", True):
            mosaic_tif = out_dir / "canopy_height_mosaic.tif"
            save_geotiff(mosaic, geo_profile, mosaic_tif, band_name="canopy_height_m")

        if cfg["output"].get("save_mosaic_visual", True):
            print("[runner] Generating full-scene visualisation...")
            mosaic_visual(
                rgb_full=rgb,
                mosaic=mosaic,
                patches=patches,
                predictions=predictions,
                embeddings=embeddings,
                out_dir=out_dir,
                cfg=cfg,
            )

        print(f"\n{'=' * 60}")
        print("  STAC Pipeline complete! Outputs:")
        print(f"  GeoTIFF  : {out_dir / 'canopy_height_mosaic.tif'}")
        print(f"  Mosaic   : {out_dir / 'mosaic_visualisation.png'}")
        print(f"  Patches  : {patches_dir}/")
        print(f"{'=' * 60}\n")


class EsriPatchInferencePipeline:
    """Patch pipeline: infer CHM from native ESRI PNGs."""

    def __init__(self, cfg: dict, input_dir: str):
        self.cfg = cfg
        self.input_path = Path(input_dir)

    def _resolve_out_dir(self) -> Path:
        output_cfg = self.cfg["output"]
        root = Path(output_cfg["output_dir"])

        append_subdir = output_cfg.get("esri_append_subdir", True)
        subdir = output_cfg.get("esri_output_subdir", "esri_results")
        out_dir = root / subdir if append_subdir else root
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def run(self) -> None:
        cfg = self.cfg
        out_dir = self._resolve_out_dir()

        print("[runner] Loading model...")
        model, processor, device = load_model_and_processor(cfg)

        if self.input_path.is_file():
            png_files = [self.input_path] if self.input_path.suffix.lower() == ".png" else []
        else:
            png_files = sorted(self.input_path.glob("*.png"))
        if not png_files:
            print(f"[runner] No .png files found in {self.input_path}")
            return

        print(f"[runner] Found {len(png_files)} PNG patches in {self.input_path}. Running direct inference...")

        class NativePatch:
            def __init__(self, arr: np.ndarray):
                self.array = arr

        for png_path in tqdm(png_files, desc="Processing PNGs"):
            img = Image.open(png_path).convert("RGB")
            patch_rgb = np.array(img)

            predictions, embeddings = run_patch_inference([NativePatch(patch_rgb)], model, processor, device, cfg)
            height_pred = predictions[0]
            emb_pred = embeddings[0]

            base_name = png_path.stem
            chm_out = out_dir / f"{base_name}_CHM.tif"

            transform, crs = _esri_patch_transform(base_name, width=height_pred.shape[1], height=height_pred.shape[0])

            arr = np.asarray(height_pred, dtype=np.float32)
            arr[~np.isfinite(arr)] = -9999.0

            profile = {
                "driver": "GTiff",
                "height": arr.shape[0],
                "width": arr.shape[1],
                "count": 1,
                "dtype": rasterio.float32,
                "transform": transform,
                "compress": "lzw",
                "nodata": -9999.0,
            }
            if crs is not None:
                profile["crs"] = crs

            with rasterio.open(chm_out, "w", **profile) as dst:
                dst.write(arr, 1)
                dst.set_band_description(1, "canopy_height_m")

            if emb_pred is not None:
                emb_rgb = _embedding_pca_rgb(
                    emb=emb_pred,
                    spatial_shape=height_pred.shape,
                    cmap_name=cfg["output"].get("embedding_colormap", "turbo"),
                )
                emb_out = out_dir / f"{base_name}_EMB.png"
                Image.fromarray(emb_rgb).save(emb_out)

        print(f"\n{'=' * 60}")
        print("  ESRI Pipeline complete! Outputs saved to:")
        print(f"  {out_dir}/")
        print(f"{'=' * 60}\n")
