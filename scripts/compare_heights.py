#!/usr/bin/env python3
"""Compare CHMv2 predictions against field plot heights using clipped plot footprints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Patch
from rasterio.mask import mask as raster_mask
from rasterio.warp import transform_geom

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
    load_field_plots,
    load_manifest,
    write_footprint_geojson,
)


def extract_polygon_stats(
    src: rasterio.io.DatasetReader,
    geometry_wgs84: dict,
    expected_pixels: float | None = None,
) -> tuple[float, int, int, float]:
    """
    Return (mean_height, valid_pixel_count, inside_pixel_count, coverage_ratio).

    coverage_ratio is valid_pixel_count / expected_pixels when expected_pixels is given,
    otherwise valid_pixel_count / inside_pixel_count.
    """
    if src.crs is None:
        return float("nan"), 0, 0, 0.0

    geometry_src = transform_geom(src_crs=WGS84, dst_crs=src.crs, geom=geometry_wgs84)
    masked, _ = raster_mask(src, [geometry_src], crop=True, filled=False)
    band = masked[0]

    if np.ma.isMaskedArray(band):
        inside_values = band.compressed().astype(np.float32)
    else:
        inside_values = band.astype(np.float32).reshape(-1)

    inside_pixel_count = int(inside_values.size)

    nodata = src.nodata
    values = inside_values
    if nodata is not None:
        values = values[values != nodata]

    values = values[np.isfinite(values)]
    valid_pixel_count = int(values.size)

    if expected_pixels is not None and expected_pixels > 0:
        coverage_ratio = float(valid_pixel_count / expected_pixels)
    elif inside_pixel_count > 0:
        coverage_ratio = float(valid_pixel_count / inside_pixel_count)
    else:
        coverage_ratio = 0.0

    if valid_pixel_count == 0:
        return float("nan"), 0, inside_pixel_count, coverage_ratio

    return float(values.mean()), valid_pixel_count, inside_pixel_count, coverage_ratio


def expected_pixels_for_plot(
    src: rasterio.io.DatasetReader,
    footprint_shape: str,
    plot_size_m: float,
    buffer_m: float,
) -> float | None:
    """Estimate expected pixel count for the footprint from raster pixel area."""
    if src.crs is None or not src.crs.is_projected:
        return None

    pixel_area = abs(src.transform.a * src.transform.e - src.transform.b * src.transform.d)
    if pixel_area <= 0:
        return None

    if footprint_shape == "square":
        footprint_area = plot_size_m * plot_size_m
    else:
        footprint_area = np.pi * (buffer_m**2)

    return float(footprint_area / pixel_area)


def build_results_dataframe(
    plots_geojson: str | Path,
    chm_dir: str | Path,
    footprint_shape: str,
    plot_size_m: float,
    buffer_m: float,
    n_vertices: int,
    rotation_deg: float,
    min_coverage_ratio: float,
    manifest: dict | None = None,  # dict[sr -> {tile_key, tile_png, ...}] from load_manifest()
) -> pd.DataFrame:
    plots = load_field_plots(plots_geojson)
    chm_dir = Path(chm_dir)

    rows: list[dict] = []
    for plot in plots:
        # Tile-based lookup (new): use tile_key from manifest to find the shared CHM.
        # Fallback to sr-based glob lookup for backward compatibility.
        if manifest and plot.sr in manifest:
            tile_key = manifest[plot.sr].get("tile_key")
            chm_tif = find_chm_tif_by_tile(chm_dir, tile_key) if tile_key else find_chm_tif(chm_dir, plot.sr)
        else:
            chm_tif = find_chm_tif(chm_dir, plot.sr)
        polygon = build_plot_footprint(
            lon=plot.lon,
            lat=plot.lat,
            shape=footprint_shape,
            plot_size_m=plot_size_m,
            buffer_m=buffer_m,
            n_vertices=n_vertices,
            rotation_deg=rotation_deg,
        )

        if chm_tif is None:
            rows.append(
                {
                    "sr": plot.sr,
                    "lat": plot.lat,
                    "lon": plot.lon,
                    "h_avg": plot.h_avg,
                    "h_pred": np.nan,
                    "delta": np.nan,
                    "abs_error": np.nan,
                    "status": "missing_chm",
                    "n_pixels": 0,
                    "inside_pixels": 0,
                    "expected_pixels": np.nan,
                    "coverage_ratio": 0.0,
                    "chm_path": None,
                }
            )
            continue

        with rasterio.open(chm_tif) as src:
            expected_pixels = expected_pixels_for_plot(
                src=src,
                footprint_shape=footprint_shape,
                plot_size_m=plot_size_m,
                buffer_m=buffer_m,
            )
            h_pred_raw, n_pixels, inside_pixels, coverage_ratio = extract_polygon_stats(
                src,
                polygon,
                expected_pixels=expected_pixels,
            )

            if src.crs is None:
                status = "no_crs"
                h_pred = np.nan
            elif not np.isfinite(h_pred_raw) or n_pixels == 0:
                status = "nodata"
                h_pred = np.nan
            elif coverage_ratio < min_coverage_ratio:
                status = "low_coverage"
                h_pred = np.nan
            elif h_pred_raw <= 0.0:
                # Model returned zero — likely outside the trained domain or a fetch artifact
                status = "zero_pred"
                h_pred = np.nan
            else:
                status = "ok"
                h_pred = h_pred_raw

        delta = float(h_pred - plot.h_avg) if np.isfinite(h_pred) else np.nan
        rows.append(
            {
                "sr": plot.sr,
                "lat": plot.lat,
                "lon": plot.lon,
                "h_avg": plot.h_avg,
                "h_pred": float(h_pred) if np.isfinite(h_pred) else np.nan,
                "delta": delta,
                "abs_error": abs(delta) if np.isfinite(delta) else np.nan,
                "status": status,
                "n_pixels": n_pixels,
                "inside_pixels": inside_pixels,
                "expected_pixels": expected_pixels if expected_pixels is not None else np.nan,
                "coverage_ratio": coverage_ratio,
                "h_pred_raw": float(h_pred_raw) if np.isfinite(h_pred_raw) else np.nan,
                "chm_path": str(chm_tif),
            }
        )

    return pd.DataFrame(rows).sort_values("sr").reset_index(drop=True)


def render_summary_plot(df: pd.DataFrame, metrics: dict, out_path: str | Path, footprint_label: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")

    for ax in axes:
        ax.set_facecolor("#1a1d27")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.tick_params(colors="white")

    df_ok = df[df["status"] == "ok"].copy()

    ax = axes[0]
    if len(df_ok) > 0:
        y_true = df_ok["h_avg"].to_numpy(float)
        y_pred = df_ok["h_pred"].to_numpy(float)

        colors = np.where(y_true > 25, "#ff6b6b", np.where(y_true > 22, "#ffd166", "#06d6a0"))
        ax.scatter(y_true, y_pred, c=colors, s=58, alpha=0.9, edgecolors="white", linewidths=0.45)

        lo = min(float(y_true.min()), float(y_pred.min())) - 1.0
        hi = max(float(y_true.max()), float(y_pred.max())) + 1.0
        lims = [lo, hi]
        ax.plot(lims, lims, "--", color="#aaaaaa", linewidth=1.2, label="1:1")

        if len(y_true) >= 2:
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            x_line = np.linspace(lo, hi, 100)
            ax.plot(x_line, slope * x_line + intercept, "-", color="#4fc3f7", linewidth=1.6)

        ax.set_xlim(lims)
        ax.set_ylim(lims)

    ax.set_title("CHMv2 vs Field H_avg", color="white", fontweight="bold")
    ax.set_xlabel("Field H_avg (m)", color="white")
    ax.set_ylabel("CHMv2 Predicted (m)", color="white")

    text = (
        f"n = {metrics['n']}\n"
        f"RMSE = {metrics['rmse']:.2f} m\n"
        f"MAE = {metrics['mae']:.2f} m\n"
        f"r = {metrics['r']:.3f}\n"
        f"bias = {metrics['bias']:+.2f} m"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        color="white",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#2d3147", "edgecolor": "#555", "alpha": 0.9},
    )

    legend_items = [
        Patch(facecolor="#06d6a0", label="H <= 22 m"),
        Patch(facecolor="#ffd166", label="22 < H <= 25 m"),
        Patch(facecolor="#ff6b6b", label="H > 25 m"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=8, labelcolor="white", facecolor="#1a1d27", edgecolor="#555")

    ax2 = axes[1]
    names = ["RMSE (m)", "Pearson r", "|Bias| (m)"]
    model_vals = [metrics["rmse"], metrics["r"], abs(metrics["bias"]) if np.isfinite(metrics["bias"]) else np.nan]
    bench_vals = [BENCHMARK_RMSE, BENCHMARK_R, np.nan]

    x = np.arange(len(names))
    width = 0.36
    bars_model = ax2.bar(x - width / 2, model_vals, width, label="CHMv2 (optical DL)", color="#4fc3f7", edgecolor="white", linewidth=0.5)
    bars_bench = ax2.bar(x + width / 2, bench_vals, width, label="PolInSAR TSI (Khati 2014)", color="#ffd166", edgecolor="white", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, color="white")
    ax2.set_title("Model vs PolInSAR Benchmark", color="white", fontweight="bold")
    ax2.set_ylabel("Value", color="white")
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#1a1d27", edgecolor="#555")

    for bars in (bars_model, bars_bench):
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.2f}", ha="center", va="bottom", fontsize=8, color="white")

    verdict = "Beats RMSE benchmark" if metrics["rmse"] < BENCHMARK_RMSE else "Does not beat benchmark"
    verdict_color = "#06d6a0" if metrics["rmse"] < BENCHMARK_RMSE else "#ff6b6b"
    ax2.text(
        0.5,
        0.04,
        verdict,
        transform=ax2.transAxes,
        ha="center",
        color=verdict_color,
        fontsize=10,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#2d3147", "edgecolor": verdict_color, "alpha": 0.9},
    )

    fig.suptitle(
        f"Barkot Range Validation: CHMv2 vs Field Plots ({footprint_label})\n"
        "Paper benchmark: PolInSAR TSI RMSE=2.28 m, r=0.62",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CHMv2 CHM with field plots")
    parser.add_argument("--plots_geojson", default="data/input/field_plots.geojson")
    parser.add_argument("--chm_dir", default="data/output/esri_results")
    parser.add_argument("--manifest", default=None, help="Path to plot_manifest.json for tile-based CHM lookup")
    parser.add_argument("--out_csv", default="data/output/validation_results.csv")
    parser.add_argument("--out_metrics", default="data/output/validation_metrics.json")
    parser.add_argument("--out_plot", default="data/output/validation_scatter.png")
    parser.add_argument("--out_footprints_geojson", default="data/output/field_plot_footprints.geojson")
    parser.add_argument(
        "--out_buffers_geojson",
        default=None,
        help="Deprecated alias; if set, footprint GeoJSON is also written here.",
    )
    parser.add_argument(
        "--footprint_shape",
        choices=["square", "circle"],
        default="square",
        help="Footprint used for comparison. 'square' is recommended for 12.5x12.5 m plots.",
    )
    parser.add_argument(
        "--plot_size_m",
        type=float,
        default=DEFAULT_PLOT_SIZE_M,
        help="Square plot side length in metres (used when --footprint_shape square).",
    )
    parser.add_argument("--buffer_m", type=float, default=DEFAULT_BUFFER_M)
    parser.add_argument("--buffer_vertices", type=int, default=72)
    parser.add_argument(
        "--min_coverage_ratio",
        type=float,
        default=0.97,
        help="Minimum required valid-pixel coverage ratio inside footprint.",
    )
    parser.add_argument(
        "--rotation_deg",
        type=float,
        default=0.0,
        help="Clockwise rotation for square footprint in degrees (0 means North-up).",
    )
    args = parser.parse_args()

    plots = load_field_plots(args.plots_geojson)
    if not plots:
        raise SystemExit(f"No plots found in {args.plots_geojson}")

    # Load manifest for tile-based CHM lookup if provided
    manifest = load_manifest(args.manifest) if args.manifest else None

    footprints_path = write_footprint_geojson(
        plots,
        out_path=args.out_footprints_geojson,
        shape=args.footprint_shape,
        plot_size_m=args.plot_size_m,
        buffer_m=args.buffer_m,
        n_vertices=args.buffer_vertices,
        rotation_deg=args.rotation_deg,
    )
    if args.out_buffers_geojson:
        write_footprint_geojson(
            plots,
            out_path=args.out_buffers_geojson,
            shape=args.footprint_shape,
            plot_size_m=args.plot_size_m,
            buffer_m=args.buffer_m,
            n_vertices=args.buffer_vertices,
            rotation_deg=args.rotation_deg,
        )

    df = build_results_dataframe(
        plots_geojson=args.plots_geojson,
        chm_dir=args.chm_dir,
        manifest=manifest,
        footprint_shape=args.footprint_shape,
        plot_size_m=args.plot_size_m,
        buffer_m=args.buffer_m,
        n_vertices=args.buffer_vertices,
        rotation_deg=args.rotation_deg,
        min_coverage_ratio=args.min_coverage_ratio,
    )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    df_ok = df[df["status"] == "ok"].copy()
    metrics = compute_metrics(df_ok["h_avg"].to_numpy(float), df_ok["h_pred"].to_numpy(float))
    low_cov_count = int((df["status"] == "low_coverage").sum())
    metrics_payload = {
        "chmv2": metrics,
        "benchmark_polinsar_tsi": {"rmse": BENCHMARK_RMSE, "r": BENCHMARK_R},
        "footprint": {
            "shape": args.footprint_shape,
            "plot_size_m": args.plot_size_m,
            "buffer_m": args.buffer_m,
            "rotation_deg": args.rotation_deg,
            "min_coverage_ratio": args.min_coverage_ratio,
        },
        "counts": {
            "total_plots": int(len(df)),
            "ok_plots": int(len(df_ok)),
            "low_coverage_plots": low_cov_count,
        },
        "comparison": {
            "rmse_delta_vs_benchmark": float(metrics["rmse"] - BENCHMARK_RMSE) if np.isfinite(metrics["rmse"]) else np.nan,
            "r_delta_vs_benchmark": float(metrics["r"] - BENCHMARK_R) if np.isfinite(metrics["r"]) else np.nan,
            "beats_rmse_benchmark": bool(metrics["rmse"] < BENCHMARK_RMSE) if np.isfinite(metrics["rmse"]) else False,
            "matches_or_beats_r": bool(metrics["r"] >= BENCHMARK_R) if np.isfinite(metrics["r"]) else False,
        },
    }

    out_metrics = Path(args.out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    footprint_label = (
        f"square {args.plot_size_m} m x {args.plot_size_m} m"
        if args.footprint_shape == "square"
        else f"circle radius {args.buffer_m} m"
    )
    render_summary_plot(df, metrics, args.out_plot, footprint_label=footprint_label)

    print("\n" + "=" * 64)
    print(f"Footprints GeoJSON : {footprints_path}")
    print(f"Results CSV     : {out_csv}")
    print(f"Metrics JSON    : {out_metrics}")
    print(f"Summary plot    : {args.out_plot}")
    print(
        "Footprint setup : "
        f"shape={args.footprint_shape}, plot_size_m={args.plot_size_m}, "
        f"buffer_m={args.buffer_m}, rotation_deg={args.rotation_deg}, "
        f"min_coverage_ratio={args.min_coverage_ratio}"
    )
    print(
        "Coverage filter : "
        f"ok={len(df_ok)} / total={len(df)} "
        f"(low_coverage={low_cov_count})"
    )
    print(
        "CHMv2 metrics   : "
        f"RMSE={metrics['rmse']:.2f} m, r={metrics['r']:.3f}, bias={metrics['bias']:+.2f} m, n={metrics['n']}"
    )
    print(f"Benchmark       : RMSE={BENCHMARK_RMSE:.2f} m, r={BENCHMARK_R:.2f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
