"""
CHMv2 Canopy Height Inference Pipeline
=======================================
DINOv3 + DPT head for per-pixel canopy height estimation (metres).

Modes
-----
  stac      Full-scene GeoTIFF → tile → infer → mosaic
  esri      Directory of ESRI PNG patches → infer → CHM TIFs
  validate  End-to-end: manifest → infer all tiles (model loaded once)
              → compare heights → render panels + dashboard

Usage
-----
  python run_inference.py --mode stac
  python run_inference.py --mode esri --esri_dir data/input/esri_patches
  python run_inference.py --mode validate
  python run_inference.py --mode validate --overwrite  # re-run inference even if CHM exists
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from pipeline.runner import EsriPatchInferencePipeline, StacInferencePipeline


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="CHMv2 Canopy Height Pipeline")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--mode",    choices=["stac", "esri", "validate"], default="stac")
    parser.add_argument("--esri_dir", default="data/input/esri_patches",
                        help="PNG directory (esri mode)")
    parser.add_argument("--output_dir", default=None,
                        help="Override output_dir from config")

    # validate-mode arguments
    parser.add_argument("--manifest",      default="data/input/plot_manifest.json")
    parser.add_argument("--plots_geojson", default="data/input/field_plots.geojson")
    parser.add_argument("--footprint_shape", choices=["square", "circle"], default="square")
    parser.add_argument("--plot_size_m",     type=float, default=12.5)
    parser.add_argument("--buffer_m",        type=float, default=12.5)
    parser.add_argument("--rotation_deg",    type=float, default=0.0)
    parser.add_argument("--min_coverage_ratio", type=float, default=0.97)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run inference even if CHM already exists on disk")

    args = parser.parse_args()
    cfg  = load_config(args.config)

    if args.output_dir:
        cfg.setdefault("output", {})
        cfg["output"]["output_dir"] = args.output_dir
        if args.mode in ("esri", "validate"):
            cfg["output"]["esri_append_subdir"] = False

    print(f"\n{'='*60}")
    print("  CHMv2 Canopy Height Pipeline  (DINOv3 + DPT Head)")
    print(f"  Mode   : {args.mode.upper()}")
    print(f"  Device : {cfg['model']['device']}")
    print(f"{'='*60}\n")

    # ── stac mode ─────────────────────────────────────────────────────────────
    if args.mode == "stac":
        print(f"  Input  : {cfg['input']['image_path']}")
        print(f"  Output : {cfg['output']['output_dir']}\n")
        StacInferencePipeline(cfg).run()
        return

    # ── esri mode ─────────────────────────────────────────────────────────────
    if args.mode == "esri":
        print(f"  Input  : {args.esri_dir}/*.png")
        print(f"  Output : {cfg['output']['output_dir']}\n")
        EsriPatchInferencePipeline(cfg, args.esri_dir).run()
        return

    # ── validate mode ─────────────────────────────────────────────────────────
    # Add scripts/ to sys.path so validation helpers are importable from repo root
    _scripts = Path(__file__).parent / "scripts"
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))

    import rasterio
    from PIL import Image

    from compare_heights import build_results_dataframe, render_summary_plot
    from pipeline.inference import run_patch_inference
    from pipeline.model import load_model_and_processor
    from pipeline.runner import _esri_patch_transform
    from pipeline.visualise import _embedding_pca_rgb
    from validation_common import (
        BENCHMARK_R,
        BENCHMARK_RMSE,
        compute_metrics,
        load_manifest,
    )
    from visualize_plots import make_dashboard, make_tile_panel

    # 1. Load manifest ─────────────────────────────────────────────────────────
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(
            f"Manifest not found: {manifest_path}\n"
            "Run:  python scripts/fetch_all_plots.py  first."
        )

    with manifest_path.open() as f:
        manifest_list = json.load(f)
    manifest = {int(e["sr"]): e for e in manifest_list}

    # Group manifest entries by tile_key (multiple plots may share one tile)
    tile_to_entries: dict[str, list[dict]] = {}
    for entry in manifest_list:
        tk = entry.get("tile_key")
        if tk:
            tile_to_entries.setdefault(tk, []).append(entry)

    unique_tiles = list(tile_to_entries.keys())
    print(f"Manifest : {len(manifest)} plots across {len(unique_tiles)} unique tiles")

    out_dir = Path(cfg["output"]["output_dir"])
    # CHM TIFs land in esri_results/ alongside EMB PNGs
    chm_dir = out_dir / cfg["output"].get("esri_output_subdir", "esri_results")
    chm_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Load model once ───────────────────────────────────────────────────────
    # Model is shared across all tiles — avoids reloading weights per tile
    print("\n[1/3] Loading model...")
    model, processor, device = load_model_and_processor(cfg)

    class _Patch:
        """Minimal wrapper so run_patch_inference gets the .array attribute it expects."""
        def __init__(self, arr: np.ndarray):
            self.array = arr

    # 3. Inference — one pass over all unique tiles ────────────────────────────
    # Suppress the per-batch inner progress bar from run_patch_inference so only
    # the outer tile-level bar is visible.
    cfg["logging"]["progress_bar"] = False
    cfg["logging"]["verbose"] = False

    print("\n[2/3] Running inference on tiles...")
    n_ok = n_skip = n_fail = 0

    for tile_key in tqdm(unique_tiles, desc="Inference", unit="tile"):
        entries      = tile_to_entries[tile_key]
        tile_png_str = entries[0].get("tile_png")

        if not tile_png_str:
            tqdm.write(f"  SKIP {tile_key}: no tile_png in manifest")
            n_skip += 1
            continue

        tile_png = Path(tile_png_str)
        if not tile_png.exists():
            tqdm.write(f"  SKIP {tile_key}: tile not on disk — {tile_png}")
            n_skip += 1
            continue

        chm_out = chm_dir / f"{tile_png.stem}_CHM.tif"
        if chm_out.exists() and not args.overwrite:
            n_skip += 1
            n_ok   += 1
            continue  # CHM already produced — skip silently (shown in final counts)

        try:
            # Load tile PNG and run CHMv2 inference
            img   = Image.open(tile_png).convert("RGB")
            patch = _Patch(np.array(img))
            predictions, embeddings = run_patch_inference(
                [patch], model, processor, device, cfg
            )
            height_pred = predictions[0]
            emb_pred    = embeddings[0]

            # Write geo-referenced CHM GeoTIFF (EPSG:3857 transform from tile coords)
            arr = np.asarray(height_pred, dtype=np.float32)
            arr[~np.isfinite(arr)] = -9999.0
            geo_transform, crs_str = _esri_patch_transform(
                tile_png.stem, width=arr.shape[1], height=arr.shape[0]
            )
            profile = {
                "driver": "GTiff",
                "height": arr.shape[0], "width": arr.shape[1],
                "count": 1, "dtype": rasterio.float32,
                "transform": geo_transform,
                "compress": "lzw", "nodata": -9999.0,
            }
            if crs_str:
                profile["crs"] = crs_str

            with rasterio.open(chm_out, "w", **profile) as dst:
                dst.write(arr, 1)
                dst.set_band_description(1, "canopy_height_m")

            # Save embedding PCA visualisation next to the CHM
            if emb_pred is not None:
                emb_rgb = _embedding_pca_rgb(
                    emb=emb_pred,
                    spatial_shape=height_pred.shape,
                    cmap_name=cfg["output"].get("embedding_colormap", "turbo"),
                )
                Image.fromarray(emb_rgb).save(chm_dir / f"{tile_png.stem}_EMB.png")

            n_ok += 1

        except Exception as exc:
            tqdm.write(f"  FAILED {tile_key}: {exc}")
            n_fail += 1

    print(f"Inference — ok={n_ok}  skipped={n_skip}  failed={n_fail}")

    # 4a. Compare heights → CSV + metrics JSON ─────────────────────────────────
    print("\n[3a/3] Computing validation metrics...")

    df = build_results_dataframe(
        plots_geojson=args.plots_geojson,
        chm_dir=chm_dir,
        manifest=manifest,          # enables tile-based CHM path resolution
        footprint_shape=args.footprint_shape,
        plot_size_m=args.plot_size_m,
        buffer_m=args.buffer_m,
        n_vertices=72,
        rotation_deg=args.rotation_deg,
        min_coverage_ratio=args.min_coverage_ratio,
    )

    # Attach tile_key / tile_png to df rows for panel grouping below
    df["tile_key"] = df["sr"].map(lambda sr: manifest.get(int(sr), {}).get("tile_key"))
    df["tile_png"] = df["sr"].map(lambda sr: manifest.get(int(sr), {}).get("tile_png"))

    csv_path = out_dir / "validation_results.csv"
    df.to_csv(csv_path, index=False)

    df_ok   = df[df["status"] == "ok"]
    metrics = compute_metrics(df_ok["h_avg"].to_numpy(float), df_ok["h_pred"].to_numpy(float))

    metrics_payload = {
        "chmv2": metrics,
        "benchmark_polinsar_tsi": {"rmse": BENCHMARK_RMSE, "r": BENCHMARK_R},
        "counts": {
            "total":        len(df),
            "ok":           len(df_ok),
            "low_coverage": int((df["status"] == "low_coverage").sum()),
            "missing_chm":  int((df["status"] == "missing_chm").sum()),
        },
    }
    metrics_path = out_dir / "validation_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"  CHMv2  RMSE={metrics['rmse']:.2f} m  r={metrics['r']:.3f}  n={metrics['n']}")
    print(f"  Bench  RMSE={BENCHMARK_RMSE:.2f} m  r={BENCHMARK_R:.2f}")

    render_summary_plot(
        df, metrics,
        out_dir / "validation_scatter.png",
        footprint_label=f"square {args.plot_size_m} m",
    )

    # 4b. Render tile panels + summary dashboard ───────────────────────────────
    print("\n[3b/3] Rendering validation panels...")
    panels_dir = out_dir / "validation_panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    # Only render tiles that have at least one ok plot
    tiles_to_render = {
        tk: tdf
        for tk, tdf in df[df["tile_key"].notna()].groupby("tile_key")
        if (tdf["status"] == "ok").any()
    }

    for tile_key, tile_df in tqdm(tiles_to_render.items(), desc="Panels", unit="tile"):
        tile_png_str = tile_df["tile_png"].iloc[0]
        tile_png     = Path(tile_png_str) if tile_png_str else None
        chm_tif      = (chm_dir / f"{tile_png.stem}_CHM.tif") if tile_png else None

        make_tile_panel(
            tile_rows=tile_df,
            tile_png=tile_png,
            chm_tif=chm_tif if (chm_tif and chm_tif.exists()) else None,
            out_path=panels_dir / f"tile_{tile_key}_panel.png",
            footprint_shape=args.footprint_shape,
            plot_size_m=args.plot_size_m,
            buffer_m=args.buffer_m,
            rotation_deg=args.rotation_deg,
        )

    dashboard_path = out_dir / "validation_summary_dashboard.png"
    make_dashboard(df=df, out_path=dashboard_path)

    # Final summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Validate complete!")
    print(f"  CSV       : {csv_path}")
    print(f"  Metrics   : {metrics_path}")
    print(f"  Scatter   : {out_dir / 'validation_scatter.png'}")
    print(f"  Panels    : {panels_dir}/  ({len(tiles_to_render)} tiles rendered)")
    print(f"  Dashboard : {dashboard_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
