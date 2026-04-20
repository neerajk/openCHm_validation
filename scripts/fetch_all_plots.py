#!/usr/bin/env python3
"""Fetch ESRI World Imagery tiles for all field plots — tile-deduplicated.

Multiple plots that fall on the same 512×512 tile share one PNG, fetched only once.
Writes a manifest mapping each plot to its tile_key and tile_png path, so downstream
scripts (run_inference.py, compare_heights.py) can resolve CHMs without glob patterns.

Usage:
    python scripts/fetch_all_plots.py --plots_geojson data/input/field_plots.geojson
    python scripts/fetch_all_plots.py --start 1 --end 50 --dry_run
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

from validation_common import latlon_to_tile_xy, load_field_plots

ESRI_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
)


# ── low-level tile fetch ──────────────────────────────────────────────────────

def _fetch_single_tile(session: requests.Session, zoom: int, x: int, y: int) -> Image.Image | None:
    """Download one 256×256 ESRI tile. Returns None on failure (non-fatal)."""
    url = ESRI_TILE_URL.format(zoom=zoom, x=x, y=y)
    try:
        r = session.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (CHMv2 Validation)"},
            timeout=20,
        )
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as exc:
        tqdm.write(f"    WARNING z={zoom} x={x} y={y}: {exc}")
        return None


def fetch_tile_512(
    session: requests.Session,
    zoom: int,
    tile_x: int,
    tile_y: int,
    out_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Stitch a 2×2 block of ESRI tiles into one 512×512 PNG.

    Output filename encodes tile coords: esri_512_z{zoom}_{tile_x}_{tile_y}.png
    Skips network fetch if the file already exists (idempotent re-runs).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"esri_512_z{zoom}_{tile_x}_{tile_y}.png"

    if out_path.exists() and not overwrite:
        return out_path  # already on disk

    patch = Image.new("RGB", (512, 512))
    # Paste 4 tiles at their pixel offsets within the 512×512 canvas
    for px, py, dx, dy in [(0, 0, 0, 0), (256, 0, 1, 0), (0, 256, 0, 1), (256, 256, 1, 1)]:
        tile = _fetch_single_tile(session, zoom, tile_x + dx, tile_y + dy)
        if tile:
            patch.paste(tile, (px, py))

    patch.save(out_path, format="PNG")
    return out_path


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ESRI tiles for all field plots (tile-deduplicated, manifest-writing)"
    )
    parser.add_argument("--plots_geojson", default="data/input/field_plots.geojson")
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--out_dir", default="data/input/esri_patches")
    parser.add_argument("--manifest", default="data/input/plot_manifest.json")
    parser.add_argument("--start", type=int, default=1, help="First sr to include")
    parser.add_argument("--end", type=int, default=10_000, help="Last sr to include")
    parser.add_argument("--first_n", type=int, default=None, help="Cap at N plots after start/end filter")
    parser.add_argument("--overwrite", action="store_true", help="Re-fetch even if tile PNG already exists")
    parser.add_argument("--dry_run", action="store_true", help="Print tile plan without fetching")
    args = parser.parse_args()

    # ── load + filter plots ──────────────────────────────────────────────────
    plots = load_field_plots(args.plots_geojson)
    if not plots:
        raise SystemExit(f"No plots found in {args.plots_geojson}")

    selected = [p for p in plots if args.start <= p.sr <= args.end]
    if args.first_n:
        selected = selected[: args.first_n]

    # ── group by tile coords ─────────────────────────────────────────────────
    # Plots that share a (tile_x, tile_y) at this zoom level share one PNG.
    # At zoom 18 (~0.6 m/px) a tile covers ~153×153 m, so nearby plots often collide.
    tile_to_plots: dict[tuple[int, int], list] = {}
    plot_to_tile: dict[int, tuple[int, int]] = {}
    for p in selected:
        tx, ty = latlon_to_tile_xy(p.lat, p.lon, args.zoom)
        tile_to_plots.setdefault((tx, ty), []).append(p)
        plot_to_tile[p.sr] = (tx, ty)

    unique_tiles = list(tile_to_plots.keys())
    print(f"Selected {len(selected)} plots → {len(unique_tiles)} unique tiles "
          f"({len(selected) - len(unique_tiles)} plots share a tile with another)")

    # ── dry run: just print the plan ─────────────────────────────────────────
    if args.dry_run:
        for (tx, ty), tile_plots in tile_to_plots.items():
            srs = [p.sr for p in tile_plots]
            tag = " ← shared" if len(srs) > 1 else ""
            print(f"  z{args.zoom}_{tx}_{ty}: sr={srs}{tag}")
        return

    # ── fetch unique tiles ───────────────────────────────────────────────────
    base_out = Path(args.out_dir)
    ok_count = fail_count = 0

    with requests.Session() as session:
        for tx, ty in tqdm(unique_tiles, desc="Fetching tiles", unit="tile"):
            srs = [p.sr for p in tile_to_plots[(tx, ty)]]
            expected = base_out / f"esri_512_z{args.zoom}_{tx}_{ty}.png"

            if expected.exists() and not args.overwrite:
                tqdm.write(f"  skip z{args.zoom}_{tx}_{ty}  sr={srs}")
                ok_count += 1
                continue

            try:
                fetch_tile_512(session, args.zoom, tx, ty, base_out, overwrite=args.overwrite)
                ok_count += 1
            except Exception as exc:
                tqdm.write(f"  FAILED z{args.zoom}_{tx}_{ty}: {exc}")
                fail_count += 1

    # ── write manifest ───────────────────────────────────────────────────────
    # Every plot (not just selected) gets an entry; unselected plots have tile_key=None.
    # Downstream scripts use tile_key to resolve the CHM path without glob matching.
    manifest_entries = []
    for p in plots:
        if p.sr in plot_to_tile:
            tx, ty = plot_to_tile[p.sr]
            tile_key = f"z{args.zoom}_{tx}_{ty}"
            tile_png = str(base_out / f"esri_512_{tile_key}.png")
        else:
            tile_key = tile_png = None  # outside selected range

        manifest_entries.append({
            "sr": p.sr,
            "lat": p.lat,
            "lon": p.lon,
            "h_avg": p.h_avg,
            "tile_key": tile_key,
            "tile_png": tile_png,
        })

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    print(f"\nDone — {ok_count} tiles fetched/cached, {fail_count} failed")
    print(f"Manifest: {manifest_path}  ({len(manifest_entries)} plots, {len(unique_tiles)} tiles)")


if __name__ == "__main__":
    main()
