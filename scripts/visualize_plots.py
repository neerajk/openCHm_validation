#!/usr/bin/env python3
"""Render per-tile validation panels and a benchmark summary dashboard.

Each panel covers one ESRI 512×512 tile and overlays ALL field plots that fall
on that tile — each plot gets a distinct colour and sr label. Panels are only
generated for tiles that have at least one 'ok' status plot.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import Normalize
from rasterio.transform import rowcol
from rasterio.warp import transform, transform_geom

from validation_common import (
    BENCHMARK_R,
    BENCHMARK_RMSE,
    DEFAULT_BUFFER_M,
    DEFAULT_PLOT_SIZE_M,
    WGS84,
    build_plot_footprint,
    compute_metrics,
    find_chm_tif,
    find_chm_tif_by_tile,
    find_esri_png,
    load_field_plots,
    load_manifest,
    meters_per_pixel_webmercator,
    parse_tile_metadata,
    plot_dir_name,
    slippy_pixel_xy,
)

# ── colour palette ────────────────────────────────────────────────────────────
BG = "#0f1117"
PANEL_BG = "#1a1d27"
ACCENT = "#4fc3f7"
GREEN = "#06d6a0"
YELLOW = "#ffd166"
RED = "#ff6b6b"

# Distinct colours cycled across plots that share a tile
PLOT_COLORS = [ACCENT, GREEN, YELLOW, RED, "#c0a0ff", "#ff9f40", "#fa8231", "#a8dadc"]


def _tile_lonlat_extent(tile_key: str) -> tuple[float, float, float, float] | None:
    """Return (lon_min, lon_max, lat_min, lat_max) for a 2×2 512-px stitched tile."""
    m = re.match(r"z(\d+)_(\d+)_(\d+)", str(tile_key))
    if not m:
        return None
    z, tx, ty = int(m.group(1)), int(m.group(2)), int(m.group(3))
    n = 2.0 ** z
    lon_min = tx / n * 360.0 - 180.0
    lon_max = (tx + 2) / n * 360.0 - 180.0
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty / n))))
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (ty + 2) / n))))
    return lon_min, lon_max, lat_min, lat_max


# ── axis helpers ──────────────────────────────────────────────────────────────

def style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="white", labelsize=8)
    if title:
        ax.set_title(title, color="white", fontsize=10, pad=6, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color="#bbb", fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, color="#bbb", fontsize=8)


def draw_circle(ax, x: float, y: float, radius: float, color: str, label: str | None = None) -> None:
    circle = plt.Circle((x, y), radius, fill=False, edgecolor=color, linewidth=1.0, linestyle="--")
    ax.add_patch(circle)
    ax.plot(x, y, marker="+", color=color, markersize=10, markeredgewidth=2)
    if label:
        ax.text(x, y - radius - 8, label, ha="center", va="bottom", color=color, fontsize=7,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.5, "edgecolor": "none"})


def draw_polygon(ax, points_xy: list[tuple[float, float]], color: str, label: str | None = None) -> None:
    poly = mpatches.Polygon(points_xy, closed=True, fill=False, edgecolor=color, linewidth=1.0, linestyle="--")
    ax.add_patch(poly)
    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    ax.plot(cx, cy, marker="+", color=color, markersize=10, markeredgewidth=2)
    if label:
        ax.text(cx, min(ys) - 8, label, ha="center", va="bottom", color=color, fontsize=7,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.5, "edgecolor": "none"})


# ── geometry helpers (unchanged from original) ────────────────────────────────

def patch_overlay_geometry(
    png_path: Path,
    lat: float, lon: float,
    footprint_shape: str,
    plot_size_m: float, buffer_m: float, rotation_deg: float,
) -> tuple[float, float, float, list[tuple[float, float]] | None]:
    meta = parse_tile_metadata(png_path)
    if meta is None:
        if footprint_shape == "square":
            half = max(8.0, plot_size_m / 0.6) / 2.0
            poly = [(256 - half, 256 - half), (256 + half, 256 - half),
                    (256 + half, 256 + half), (256 - half, 256 + half)]
            return 256.0, 256.0, 0.0, poly
        return 256.0, 256.0, max(4.0, buffer_m / 0.6), None

    zoom, tile_x, tile_y = meta
    global_x, global_y = slippy_pixel_xy(lat=lat, lon=lon, zoom=zoom)
    x_px = global_x - tile_x * 256.0
    y_px = global_y - tile_y * 256.0
    mpp = meters_per_pixel_webmercator(lat=lat, zoom=zoom)

    if footprint_shape == "square":
        geom = build_plot_footprint(lon=lon, lat=lat, shape="square",
                                    plot_size_m=plot_size_m, buffer_m=buffer_m, rotation_deg=rotation_deg)
        ring = geom["coordinates"][0][:-1]
        pts = [(float(slippy_pixel_xy(lat=lv, lon=lv_lon, zoom=zoom)[0] - tile_x * 256.0),
                float(slippy_pixel_xy(lat=lv, lon=lv_lon, zoom=zoom)[1] - tile_y * 256.0))
               for lv_lon, lv in ring]
        return float(x_px), float(y_px), 0.0, pts

    r_px = max(3.0, buffer_m / max(mpp, 1e-6))
    return float(x_px), float(y_px), float(r_px), None


def chm_overlay_geometry(
    src: rasterio.io.DatasetReader,
    lat: float, lon: float,
    footprint_shape: str,
    plot_size_m: float, buffer_m: float, rotation_deg: float,
) -> tuple[int, int, float, list[tuple[float, float]] | None]:
    if src.crs is not None:
        xs, ys = transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = rowcol(src.transform, xs[0], ys[0])
    else:
        row, col = src.height // 2, src.width // 2

    row = int(np.clip(row, 0, src.height - 1))
    col = int(np.clip(col, 0, src.width - 1))

    px_m = (abs(src.transform.a) if src.crs is not None and src.crs.is_projected
            else abs(src.transform.a) * 111_320.0 * np.cos(np.radians(lat)))

    if footprint_shape == "square":
        geom_wgs = build_plot_footprint(lon=lon, lat=lat, shape="square",
                                        plot_size_m=plot_size_m, buffer_m=buffer_m, rotation_deg=rotation_deg)
        if src.crs is not None:
            geom_src = transform_geom(WGS84, src.crs, geom_wgs)
            ring = geom_src["coordinates"][0][:-1]
            xs_r = [pt[0] for pt in ring]
            ys_r = [pt[1] for pt in ring]
            rows_r, cols_r = rowcol(src.transform, xs_r, ys_r)
            pts = [(float(c), float(r)) for r, c in zip(rows_r, cols_r)]
        else:
            half = max(2.0, (plot_size_m / max(px_m, 1e-6)) / 2.0)
            pts = [(col - half, row - half), (col + half, row - half),
                   (col + half, row + half), (col - half, row + half)]
        return row, col, 0.0, pts

    r_px = float(max(1.0, buffer_m / max(px_m, 1e-6)))
    return row, col, r_px, None


# ── tile panel ────────────────────────────────────────────────────────────────

def make_tile_panel(
    tile_rows: pd.DataFrame,     # all plots sharing this tile (ok + non-ok)
    tile_png: Path | None,
    chm_tif: Path | None,
    out_path: Path,
    footprint_shape: str = "square",
    plot_size_m: float = DEFAULT_PLOT_SIZE_M,
    buffer_m: float = DEFAULT_BUFFER_M,
    rotation_deg: float = 0.0,
) -> None:
    """Render one validation panel for a shared tile, overlaying all co-located plots.

    Layout: [ESRI patch] [CHM] [per-plot stats table]
    Each plot gets a distinct colour; sr number is printed at the bbox centre.
    """
    n = len(tile_rows)
    colors = [PLOT_COLORS[i % len(PLOT_COLORS)] for i in range(n)]

    fig = plt.figure(figsize=(16, 6), facecolor=BG)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.6, 1.6, 1.0], wspace=0.28)
    ax_rgb  = fig.add_subplot(gs[0])
    ax_chm  = fig.add_subplot(gs[1])
    ax_stat = fig.add_subplot(gs[2])

    # ── ESRI panel ────────────────────────────────────────────────────────────
    style_ax(ax_rgb, title="ESRI Patch")
    if tile_png and tile_png.exists():
        ax_rgb.imshow(plt.imread(str(tile_png)))
        for (_, row), color in zip(tile_rows.iterrows(), colors):
            sr = int(row.sr)
            lat, lon = float(row.lat), float(row.lon)
            x_px, y_px, r_px, poly_pts = patch_overlay_geometry(
                tile_png, lat, lon, footprint_shape, plot_size_m, buffer_m, rotation_deg)
            if footprint_shape == "square" and poly_pts:
                draw_polygon(ax_rgb, poly_pts, color)
            else:
                draw_circle(ax_rgb, x_px, y_px, r_px, color)
            # Sr label at the bbox centre — small, low opacity background
            ax_rgb.text(x_px, y_px, str(sr), color=color, fontsize=7,
                        ha="center", va="center", fontweight="bold",
                        bbox={"boxstyle": "round,pad=0.1", "facecolor": "black",
                              "alpha": 0.45, "edgecolor": "none"})
    else:
        ax_rgb.text(0.5, 0.5, "No ESRI patch", transform=ax_rgb.transAxes,
                    ha="center", va="center", color="#888")
    ax_rgb.set_xticks([]); ax_rgb.set_yticks([])

    # ── CHM panel ─────────────────────────────────────────────────────────────
    style_ax(ax_chm, title="CHMv2 Canopy Height")
    if chm_tif and chm_tif.exists():
        with rasterio.open(chm_tif) as src:
            chm = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                chm = np.where(chm == nodata, np.nan, chm)
            vmin, vmax = (np.nanpercentile(chm, [2, 98]) if np.any(np.isfinite(chm)) else (0.0, 30.0))
            im = ax_chm.imshow(chm, cmap="YlGn", vmin=vmin, vmax=vmax)
            for (_, row), color in zip(tile_rows.iterrows(), colors):
                sr = int(row.sr)
                lat, lon = float(row.lat), float(row.lon)
                row_px, col_px, r_px, poly_pts = chm_overlay_geometry(
                    src, lat, lon, footprint_shape, plot_size_m, buffer_m, rotation_deg)
                if footprint_shape == "square" and poly_pts:
                    draw_polygon(ax_chm, poly_pts, color)
                else:
                    draw_circle(ax_chm, col_px, row_px, r_px, color)
                ax_chm.text(col_px, row_px, str(sr), color=color, fontsize=7,
                            ha="center", va="center", fontweight="bold",
                            bbox={"boxstyle": "round,pad=0.1", "facecolor": "black",
                                  "alpha": 0.45, "edgecolor": "none"})
            cbar = plt.colorbar(im, ax=ax_chm, fraction=0.046, pad=0.03)
            cbar.set_label("m", color="white")
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    else:
        ax_chm.text(0.5, 0.5, "No CHM", transform=ax_chm.transAxes,
                    ha="center", va="center", color="#888")
    ax_chm.set_xticks([]); ax_chm.set_yticks([])

    # ── stats table ───────────────────────────────────────────────────────────
    style_ax(ax_stat, title=f"Plot Stats  (n={n})")
    ax_stat.set_xticks([]); ax_stat.set_yticks([])

    line_h = min(0.12, 0.80 / max(n, 1))
    y = 0.95

    # Column header
    ax_stat.text(0.04, y, f"{'SR':>3}  {'Field':>6}  {'Pred':>6}  {'Δ':>6}  Status",
                 transform=ax_stat.transAxes, color="#888", fontsize=7.5, va="top", family="monospace")
    y -= line_h * 0.6
    ax_stat.plot([0.03, 0.97], [y, y], color="#444", linewidth=0.5, transform=ax_stat.transAxes)
    y -= line_h * 0.5

    for (_, row), color in zip(tile_rows.iterrows(), colors):
        sr      = int(row.sr)
        h_f     = float(row.h_avg)
        h_p     = float(row.h_pred) if pd.notna(row.h_pred) else float("nan")
        delta   = h_p - h_f if np.isfinite(h_p) else float("nan")
        status  = str(row.get("status", "?"))

        pred_s  = f"{h_p:>6.1f}" if np.isfinite(h_p) else "   N/A"
        delta_s = f"{delta:>+6.1f}" if np.isfinite(delta) else "     —"
        dc      = GREEN if np.isfinite(delta) and abs(delta) <= BENCHMARK_RMSE else (
                  RED if np.isfinite(delta) else "#888")

        # Main row in the plot's colour
        ax_stat.text(0.04, y, f"{sr:>3}  {h_f:>6.1f}  {pred_s}  ",
                     transform=ax_stat.transAxes, color=color, fontsize=7.5, va="top", family="monospace")
        # Delta in red/green based on benchmark
        ax_stat.text(0.66, y, delta_s,
                     transform=ax_stat.transAxes, color=dc, fontsize=7.5, va="top", family="monospace")
        # Status tag
        ax_stat.text(0.87, y, f" {status}",
                     transform=ax_stat.transAxes, color="#777", fontsize=6.5, va="top", family="monospace")
        y -= line_h

    # Per-tile summary metrics if ≥2 ok plots
    ok_rows = tile_rows[tile_rows["status"] == "ok"]
    if len(ok_rows) >= 2:
        m = compute_metrics(ok_rows["h_avg"].to_numpy(float), ok_rows["h_pred"].to_numpy(float))
        y -= line_h * 0.4
        ax_stat.plot([0.03, 0.97], [y, y], color="#444", linewidth=0.5, transform=ax_stat.transAxes)
        y -= line_h * 0.6
        ax_stat.text(0.04, y, f"RMSE={m['rmse']:.2f}m  r={m['r']:.2f}  n={m['n']}",
                     transform=ax_stat.transAxes, color="#aaa", fontsize=7.5, va="top", family="monospace")

    tile_stem = tile_png.stem if tile_png else "unknown_tile"
    fig.suptitle(f"Tile: {tile_stem}", color="white", fontsize=11, fontweight="bold", y=0.99)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ── summary dashboard (unchanged logic) ───────────────────────────────────────

def make_dashboard(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_ok = df[df["status"] == "ok"].copy()
    metrics = compute_metrics(df_ok["h_avg"].to_numpy(float), df_ok["h_pred"].to_numpy(float))

    fig = plt.figure(figsize=(24, 10), facecolor=BG)
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1.25, 1.25, 1.0, 1.5], wspace=0.28, hspace=0.35)

    # ── map: plot locations coloured by H_avg ─────────────────────────────────
    ax_map = fig.add_subplot(gs[:, 0])
    style_ax(ax_map, title="Field Plot Locations (colour = H_avg)", xlabel="Longitude", ylabel="Latitude")
    if "tile_key" in df.columns and "tile_png" in df.columns:
        for _, tr in df.drop_duplicates(subset=["tile_key"]).iterrows():
            ext = _tile_lonlat_extent(str(tr.get("tile_key", "")))
            tp = Path(str(tr.get("tile_png", "")))
            if ext and tp.exists():
                try:
                    img = plt.imread(str(tp))
                    ax_map.imshow(img, extent=[ext[0], ext[1], ext[2], ext[3]],
                                  aspect="auto", zorder=0, alpha=0.55)
                except Exception:
                    pass
    sc = ax_map.scatter(df["lon"], df["lat"], c=df["h_avg"], cmap="YlOrRd", s=45,
                        edgecolors="white", linewidths=0.35, zorder=2)
    cbar0 = plt.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.03)
    cbar0.set_label("H_avg (m)", color="white")
    cbar0.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar0.ax.yaxis.get_ticklabels(), color="white")

    # ── residual map ──────────────────────────────────────────────────────────
    ax_res = fig.add_subplot(gs[0, 1])
    style_ax(ax_res, title="Residual Map (Pred − Field)", xlabel="Longitude", ylabel="Latitude")
    if len(df_ok) > 0:
        residual = df_ok["h_pred"] - df_ok["h_avg"]
        vmax = max(float(np.nanmax(np.abs(residual))), 0.1)
        sc_r = ax_res.scatter(df_ok["lon"], df_ok["lat"], c=residual, cmap="RdYlGn_r",
                              norm=Normalize(vmin=-vmax, vmax=vmax), s=45,
                              edgecolors="white", linewidths=0.35)
        cbar1 = plt.colorbar(sc_r, ax=ax_res, fraction=0.046, pad=0.03)
        cbar1.set_label("Residual (m)", color="white")
        cbar1.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar1.ax.yaxis.get_ticklabels(), color="white")

    # ── benchmark bar chart ───────────────────────────────────────────────────
    ax_cmp = fig.add_subplot(gs[0, 2])
    style_ax(ax_cmp, title="Model vs PolInSAR Benchmark")
    names = ["RMSE (m)", "Pearson r"]
    chm_vals = [metrics["rmse"], metrics["r"]]
    bench_vals = [BENCHMARK_RMSE, BENCHMARK_R]
    x = np.arange(len(names)); w = 0.35
    b1 = ax_cmp.bar(x - w / 2, chm_vals, w, color="#4fc3f7", edgecolor="white", linewidth=0.5, label="CHMv2")
    b2 = ax_cmp.bar(x + w / 2, bench_vals, w, color="#ffd166", edgecolor="white", linewidth=0.5, label="PolInSAR")
    ax_cmp.set_xticks(x); ax_cmp.set_xticklabels(names, color="white")
    ax_cmp.legend(fontsize=8, labelcolor="white", facecolor=PANEL_BG, edgecolor="#555")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax_cmp.text(bar.get_x() + bar.get_width() / 2, h + 0.03,
                            f"{h:.2f}", ha="center", va="bottom", color="white", fontsize=8)

    # Verdict: for RMSE lower is better; for r higher is better
    beats_rmse = np.isfinite(metrics["rmse"]) and metrics["rmse"] < BENCHMARK_RMSE
    verdict = "BEATS benchmark" if beats_rmse else "Does not beat benchmark"
    verdict_color = GREEN if beats_rmse else RED
    ax_cmp.text(0.5, 0.05, verdict, transform=ax_cmp.transAxes, ha="center",
                color=verdict_color, fontsize=10, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "#2d3147",
                      "edgecolor": verdict_color, "alpha": 0.9})

    # ── scatter: CHMv2 vs field ───────────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[1, 1])
    style_ax(ax_sc, title="CHMv2 vs Field H_avg", xlabel="Field H_avg (m)", ylabel="CHMv2 Predicted (m)")
    if len(df_ok) > 0:
        y_true = df_ok["h_avg"].to_numpy(float)
        y_pred = df_ok["h_pred"].to_numpy(float)
        colors = np.where(y_true > 25, RED, np.where(y_true > 22, YELLOW, GREEN))
        ax_sc.scatter(y_true, y_pred, c=colors, s=45, edgecolors="white", linewidths=0.35)
        lo = min(float(y_true.min()), float(y_pred.min())) - 1
        hi = max(float(y_true.max()), float(y_pred.max())) + 1
        ax_sc.plot([lo, hi], [lo, hi], "--", color="#aaaaaa", linewidth=1)
        if len(y_true) >= 2:
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            xx = np.linspace(lo, hi, 100)
            ax_sc.plot(xx, slope * xx + intercept, "-", color=ACCENT, linewidth=1.4)
        ax_sc.set_xlim(lo, hi); ax_sc.set_ylim(lo, hi)

    # ── height distribution ───────────────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[1, 2])
    style_ax(ax_dist, title="Height Distribution", ylabel="Height (m)")
    bins = np.array([14, 16, 18, 20, 22, 24, 26, 28, 30])
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]
    xh = np.arange(len(labels)); wb = 0.42
    field_means = [df.loc[(df["h_avg"] >= bins[i]) & (df["h_avg"] < bins[i+1]), "h_avg"].mean()
                   for i in range(len(bins) - 1)]
    pred_means  = [df.loc[(df["h_avg"] >= bins[i]) & (df["h_avg"] < bins[i+1]) & df["h_pred"].notna(),
                          "h_pred"].mean()
                   for i in range(len(bins) - 1)]
    ax_dist.bar(xh - wb / 2, field_means, wb, color=YELLOW, edgecolor="white", linewidth=0.4, label="Field mean")
    ax_dist.bar(xh + wb / 2, pred_means,  wb, color=ACCENT,  edgecolor="white", linewidth=0.4, label="CHMv2 mean")
    ax_dist.set_xticks(xh)
    ax_dist.set_xticklabels(labels, rotation=30, ha="right", color="white", fontsize=7)
    ax_dist.legend(fontsize=8, labelcolor="white", facecolor=PANEL_BG, edgecolor="#555")

    # ── ESRI patches mosaic with footprint overlays ───────────────────────────
    ax_esri = fig.add_subplot(gs[:, 3])
    style_ax(ax_esri, title="ESRI Tiles + Plot Footprints", xlabel="Longitude", ylabel="Latitude")
    if "tile_key" in df.columns and "tile_png" in df.columns:
        for _, tr in df.drop_duplicates(subset=["tile_key"]).iterrows():
            ext = _tile_lonlat_extent(str(tr.get("tile_key", "")))
            tp = Path(str(tr.get("tile_png", "")))
            if ext and tp.exists():
                try:
                    img = plt.imread(str(tp))
                    ax_esri.imshow(img, extent=[ext[0], ext[1], ext[2], ext[3]],
                                   aspect="auto", zorder=0)
                except Exception:
                    pass
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            lat, lon = float(row.lat), float(row.lon)
            color = PLOT_COLORS[i % len(PLOT_COLORS)]
            geom = build_plot_footprint(lon=lon, lat=lat, shape="square",
                                        plot_size_m=DEFAULT_PLOT_SIZE_M,
                                        buffer_m=DEFAULT_BUFFER_M,
                                        rotation_deg=0.0)
            ring = geom["coordinates"][0][:-1]
            poly_pts = [(float(pt[0]), float(pt[1])) for pt in ring]
            poly = mpatches.Polygon(poly_pts, closed=True, fill=False,
                                    edgecolor=color, linewidth=1.0, linestyle="--", zorder=2)
            ax_esri.add_patch(poly)
            ax_esri.text(lon, lat, str(int(row.sr)), color=color, fontsize=5,
                         ha="center", va="center", zorder=3,
                         bbox={"boxstyle": "round,pad=0.1", "facecolor": "black",
                               "alpha": 0.4, "edgecolor": "none"})
        except Exception:
            pass
    ax_esri.autoscale()
    ax_esri.set_aspect("auto")

    fig.suptitle(
        "Barkot Range Validation Dashboard — CHMv2 vs Field Plots\n"
        f"n={metrics['n']} | CHMv2 RMSE={metrics['rmse']:.2f} m, r={metrics['r']:.3f} | "
        f"PolInSAR RMSE={BENCHMARK_RMSE:.2f} m, r={BENCHMARK_R:.2f}",
        color="white", fontsize=12, fontweight="bold", y=0.98,
    )

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Render tile-based validation panels and dashboard")
    parser.add_argument("--plots_geojson", default="data/input/field_plots.geojson")
    parser.add_argument("--csv",           default="data/output/validation_results.csv")
    parser.add_argument("--manifest",      default="data/input/plot_manifest.json",
                        help="Plot manifest from fetch_all_plots.py (for tile grouping)")
    parser.add_argument("--chm_dir",       default="data/output/esri_results")
    parser.add_argument("--out_dir",       default="data/output/validation_panels")
    parser.add_argument("--dashboard",     default="data/output/validation_summary_dashboard.png")
    parser.add_argument("--tile_key",      default=None, help="Render a single tile only")
    parser.add_argument("--summary_only",  action="store_true")
    parser.add_argument("--footprint_shape", choices=["square", "circle"], default="square")
    parser.add_argument("--plot_size_m",   type=float, default=DEFAULT_PLOT_SIZE_M)
    parser.add_argument("--buffer_m",      type=float, default=DEFAULT_BUFFER_M)
    parser.add_argument("--rotation_deg",  type=float, default=0.0)
    args = parser.parse_args()

    # ── load results CSV ─────────────────────────────────────────────────────
    csv_path = Path(args.csv)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        plots = load_field_plots(args.plots_geojson)
        if not plots:
            raise SystemExit(f"No plots found in {args.plots_geojson}")
        df = pd.DataFrame([{"sr": p.sr, "lat": p.lat, "lon": p.lon, "h_avg": p.h_avg,
                             "h_pred": float("nan"), "status": "missing_csv"} for p in plots])

    required = {"sr", "lat", "lon", "h_avg", "h_pred", "status"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    # ── attach tile_key column from manifest ─────────────────────────────────
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        df["tile_key"] = df["sr"].map(lambda sr: manifest.get(int(sr), {}).get("tile_key"))
        df["tile_png"] = df["sr"].map(lambda sr: manifest.get(int(sr), {}).get("tile_png"))
    else:
        # No manifest: fall back to per-plot find_esri_png (legacy)
        df["tile_key"] = df["sr"].astype(str)
        df["tile_png"] = df["sr"].map(lambda sr: str(find_esri_png("data/input/esri_patches", int(sr)) or ""))

    out_dir  = Path(args.out_dir)
    chm_dir  = Path(args.chm_dir)

    # ── render tile panels ───────────────────────────────────────────────────
    if not args.summary_only:
        # Group by tile_key; only render tiles with ≥1 ok plot
        tile_groups = df[df["tile_key"].notna()].groupby("tile_key")
        to_render = {
            tk: tdf for tk, tdf in tile_groups
            if (tdf["status"] == "ok").any()
        }
        if args.tile_key:
            to_render = {k: v for k, v in to_render.items() if k == args.tile_key}

        from tqdm import tqdm
        for tile_key, tile_df in tqdm(to_render.items(), desc="Rendering panels", unit="tile"):
            tile_png_str = tile_df["tile_png"].iloc[0]
            tile_png = Path(tile_png_str) if tile_png_str else None

            # Resolve CHM: tile-based name preferred, sr-based as fallback
            chm_tif = find_chm_tif_by_tile(chm_dir, tile_key)
            if chm_tif is None and len(tile_df) == 1:
                chm_tif = find_chm_tif(chm_dir, int(tile_df["sr"].iloc[0]))

            out_path = out_dir / f"tile_{tile_key}_panel.png"
            make_tile_panel(
                tile_rows=tile_df,
                tile_png=tile_png,
                chm_tif=chm_tif,
                out_path=out_path,
                footprint_shape=args.footprint_shape,
                plot_size_m=args.plot_size_m,
                buffer_m=args.buffer_m,
                rotation_deg=args.rotation_deg,
            )

    make_dashboard(df=df, out_path=Path(args.dashboard))
    print(f"Dashboard: {args.dashboard}")


if __name__ == "__main__":
    main()
