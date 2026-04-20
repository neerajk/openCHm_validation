#!/usr/bin/env python3
"""Shared helpers for the field-plot validation pipeline."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

BENCHMARK_RMSE = 2.28
BENCHMARK_R = 0.62
DEFAULT_BUFFER_M = 12.5
DEFAULT_PLOT_SIZE_M = 12.5
WGS84 = "EPSG:4326"

_TILE_PATTERN = re.compile(
    r"_z(?P<z>\d+)_(?P<x>\d+)_(?P<y>\d+)(?:_[A-Z]+)?\.(?:png|tif)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class FieldPlot:
    sr: int
    lat: float
    lon: float
    h_avg: float


def load_field_plots(geojson_path: str | Path) -> list[FieldPlot]:
    """Load plot points from GeoJSON with properties: sr, h_avg."""
    path = Path(geojson_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    plots: list[FieldPlot] = []
    for idx, feature in enumerate(features, start=1):
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or [None, None]
        lon, lat = coords[0], coords[1]
        if lon is None or lat is None:
            continue

        sr = int(props.get("sr", idx))
        h_avg = float(props.get("h_avg", props.get("H_avg", np.nan)))
        if not np.isfinite(h_avg):
            continue

        plots.append(FieldPlot(sr=sr, lat=float(lat), lon=float(lon), h_avg=h_avg))

    plots.sort(key=lambda p: p.sr)
    return plots


def plot_dir_name(sr: int) -> str:
    return f"plot_{sr:03d}"


def find_esri_png(patch_root: str | Path, sr: int) -> Path | None:
    patch_root = Path(patch_root)
    if patch_root.is_file() and patch_root.suffix.lower() == ".png":
        return patch_root

    plot_dir = patch_root / plot_dir_name(sr)
    if not plot_dir.exists():
        flat_hits = sorted(patch_root.glob(f"{plot_dir_name(sr)}_*.png"))
        return flat_hits[0] if flat_hits else None

    pngs = sorted(plot_dir.glob("*.png"))
    if pngs:
        return pngs[0]

    flat_hits = sorted(patch_root.glob(f"{plot_dir_name(sr)}_*.png"))
    return flat_hits[0] if flat_hits else None


def find_chm_tif(chm_root: str | Path, sr: int) -> Path | None:
    chm_root = Path(chm_root)
    plot_dir = chm_root / plot_dir_name(sr)

    direct = [
        plot_dir / f"{plot_dir_name(sr)}_CHM.tif",
        chm_root / f"{plot_dir_name(sr)}_CHM.tif",
    ]
    for candidate in direct:
        if candidate.exists():
            return candidate

    flat_hits = sorted(chm_root.glob(f"{plot_dir_name(sr)}_*_CHM.tif"))
    if flat_hits:
        return flat_hits[0]

    if plot_dir.exists():
        hits = sorted(plot_dir.glob("*_CHM.tif"))
        if hits:
            return hits[0]
    return None


def geodesic_buffer_polygon(lon: float, lat: float, radius_m: float, n_vertices: int = 72) -> dict:
    """Create a geodesic circle polygon (GeoJSON dict) around a lon/lat point."""
    azimuths = np.linspace(0.0, 360.0, n_vertices, endpoint=False)

    try:
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        lons = np.full(n_vertices, lon, dtype=float)
        lats = np.full(n_vertices, lat, dtype=float)
        dists = np.full(n_vertices, radius_m, dtype=float)
        out_lon, out_lat, _ = geod.fwd(lons, lats, azimuths, dists)
        ring = [[float(x), float(y)] for x, y in zip(out_lon, out_lat)]
    except Exception:
        # Fallback approximation when pyproj is not available.
        r_lat = radius_m / 111_320.0
        r_lon = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-8))
        ring = []
        for az in azimuths:
            rad = math.radians(float(az))
            ring.append([float(lon + r_lon * math.cos(rad)), float(lat + r_lat * math.sin(rad))])

    ring.append(ring[0])

    return {"type": "Polygon", "coordinates": [ring]}


def _offset_lonlat(lon: float, lat: float, east_m: float, north_m: float) -> tuple[float, float]:
    """Move a lon/lat point by local east/north meter offsets."""
    distance_m = math.hypot(east_m, north_m)
    if distance_m == 0:
        return float(lon), float(lat)

    # Bearing is clockwise from north.
    bearing_deg = math.degrees(math.atan2(east_m, north_m))

    try:
        from pyproj import Geod

        geod = Geod(ellps="WGS84")
        out_lon, out_lat, _ = geod.fwd(lon, lat, bearing_deg, distance_m)
        return float(out_lon), float(out_lat)
    except Exception:
        d_lat = north_m / 111_320.0
        d_lon = east_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-8))
        return float(lon + d_lon), float(lat + d_lat)


def plot_square_polygon(
    lon: float,
    lat: float,
    side_m: float = DEFAULT_PLOT_SIZE_M,
    rotation_deg: float = 0.0,
) -> dict:
    """
    Create a square footprint polygon centered at (lon, lat).

    side_m is edge length in metres (e.g., 12.5 m x 12.5 m).
    rotation_deg rotates the square clockwise from North-up.
    """
    half = side_m / 2.0
    corners_local = [
        (-half, +half),  # NW
        (+half, +half),  # NE
        (+half, -half),  # SE
        (-half, -half),  # SW
    ]

    theta = math.radians(rotation_deg)
    c, s = math.cos(theta), math.sin(theta)

    ring = []
    for east, north in corners_local:
        # Clockwise rotation in local EN frame.
        e_rot = east * c + north * s
        n_rot = -east * s + north * c
        out_lon, out_lat = _offset_lonlat(lon, lat, east_m=e_rot, north_m=n_rot)
        ring.append([out_lon, out_lat])
    ring.append(ring[0])
    return {"type": "Polygon", "coordinates": [ring]}


def build_plot_footprint(
    lon: float,
    lat: float,
    shape: str = "square",
    plot_size_m: float = DEFAULT_PLOT_SIZE_M,
    buffer_m: float = DEFAULT_BUFFER_M,
    n_vertices: int = 72,
    rotation_deg: float = 0.0,
) -> dict:
    """Build plot footprint geometry from shape parameters."""
    shape = shape.lower()
    if shape == "square":
        return plot_square_polygon(lon=lon, lat=lat, side_m=plot_size_m, rotation_deg=rotation_deg)
    if shape == "circle":
        return geodesic_buffer_polygon(lon=lon, lat=lat, radius_m=buffer_m, n_vertices=n_vertices)
    raise ValueError(f"Unsupported footprint shape: {shape}")


def write_buffer_geojson(
    plots: Iterable[FieldPlot],
    out_path: str | Path,
    radius_m: float = DEFAULT_BUFFER_M,
    n_vertices: int = 72,
) -> Path:
    """Write buffered plot polygons to GeoJSON."""
    features = []
    for plot in plots:
        geom = geodesic_buffer_polygon(plot.lon, plot.lat, radius_m=radius_m, n_vertices=n_vertices)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "sr": plot.sr,
                    "h_avg": plot.h_avg,
                    "lat": plot.lat,
                    "lon": plot.lon,
                    "buffer_m": radius_m,
                },
                "geometry": geom,
            }
        )

    collection = {"type": "FeatureCollection", "features": features}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)
    return out_path


def write_footprint_geojson(
    plots: Iterable[FieldPlot],
    out_path: str | Path,
    shape: str = "square",
    plot_size_m: float = DEFAULT_PLOT_SIZE_M,
    buffer_m: float = DEFAULT_BUFFER_M,
    n_vertices: int = 72,
    rotation_deg: float = 0.0,
) -> Path:
    """Write plot footprint polygons (square or circle) to GeoJSON."""
    features = []
    for plot in plots:
        geom = build_plot_footprint(
            lon=plot.lon,
            lat=plot.lat,
            shape=shape,
            plot_size_m=plot_size_m,
            buffer_m=buffer_m,
            n_vertices=n_vertices,
            rotation_deg=rotation_deg,
        )
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "sr": plot.sr,
                    "h_avg": plot.h_avg,
                    "lat": plot.lat,
                    "lon": plot.lon,
                    "footprint_shape": shape,
                    "plot_size_m": plot_size_m,
                    "buffer_m": buffer_m,
                    "rotation_deg": rotation_deg,
                },
                "geometry": geom,
            }
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
    return out_path


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from scipy import stats

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask].astype(float)
    yp = y_pred[mask].astype(float)

    n = int(len(yt))
    if n == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "r": np.nan,
            "r2": np.nan,
        }

    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))
    bias = float(np.mean(yp - yt))

    if n >= 2:
        r, _ = stats.pearsonr(yt, yp)
        r = float(r)
    else:
        r = np.nan

    return {
        "n": n,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "r": r,
        "r2": float(r * r) if np.isfinite(r) else np.nan,
    }


def parse_tile_metadata(path: str | Path) -> tuple[int, int, int] | None:
    """Parse zoom/x/y from names like ..._z18_187906_107749.png or *_CHM.tif."""
    name = Path(path).name
    match = _TILE_PATTERN.search(name)
    if not match:
        return None
    return int(match.group("z")), int(match.group("x")), int(match.group("y"))


def latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert lat/lon to slippy-map tile indices (integer) at the given zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile


def slippy_pixel_xy(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Global slippy-map pixel coordinates for EPSG:3857 web tiles."""
    lat_rad = math.radians(lat)
    n = 2.0**zoom
    xtile = (lon + 180.0) / 360.0 * n
    ytile = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) / 2.0 * n
    return xtile * 256.0, ytile * 256.0


def meters_per_pixel_webmercator(lat: float, zoom: int) -> float:
    return 156543.03392804097 * math.cos(math.radians(lat)) / (2**zoom)


def load_manifest(manifest_path: str | Path) -> dict[int, dict]:
    """Load plot manifest and return a dict keyed by sr (int)."""
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        entries = json.load(f)
    return {int(e["sr"]): e for e in entries}


def find_chm_tif_by_tile(chm_root: str | Path, tile_key: str) -> Path | None:
    """Resolve a CHM GeoTIFF from a tile key (e.g. 'z18_187906_107749').

    Tile-based CHMs are named: esri_512_{tile_key}_CHM.tif
    """
    candidate = Path(chm_root) / f"esri_512_{tile_key}_CHM.tif"
    return candidate if candidate.exists() else None
