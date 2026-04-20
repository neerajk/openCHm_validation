#!/usr/bin/env python3
"""Run CHMv2 ESRI inference for all plots listed in a manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from validation_common import plot_dir_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CHMv2 inference across all field plots")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manifest", default="data/input/plot_manifest.json")
    parser.add_argument("--out_root", default="data/output/esri_results")
    parser.add_argument(
        "--nested_dirs",
        action="store_true",
        help="Store outputs in per-plot subdirectories (legacy mode). Default is flat output directory.",
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10_000)
    parser.add_argument(
        "--first_n",
        type=int,
        default=None,
        help="Optionally process only the first N plots after start/end filtering.",
    )
    parser.add_argument(
        "--only_first_100",
        action="store_true",
        help="Shortcut for --first_n 100.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}. Run fetch_all_plots.py first.")

    with manifest_path.open("r", encoding="utf-8") as f:
        plots = json.load(f)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    ok: list[int] = []
    skip: list[int] = []
    fail: list[int] = []

    selected = [p for p in plots if args.start <= int(p["sr"]) <= args.end]
    if args.only_first_100:
        args.first_n = 100
    if args.first_n is not None:
        selected = selected[: max(0, args.first_n)]

    print(f"Selected plots for inference: {len(selected)}")

    for plot in selected:
        sr = int(plot["sr"])
        patch_png_raw = plot.get("patch_png")
        patch_png = Path(patch_png_raw) if patch_png_raw else None
        patch_dir = Path(plot["patch_dir"])

        if patch_png is not None and patch_png.exists():
            esri_input = patch_png
        elif patch_dir.exists():
            pngs = list(patch_dir.glob("*.png"))
            if not pngs:
                print(f"[{sr:3d}] skip (no PNGs for plot; patch_png={patch_png_raw}, patch_dir={patch_dir})")
                skip.append(sr)
                continue

            # Safety guard: a flat shared patch directory with many PNGs must use patch_png in manifest.
            if len(pngs) > 1 and not patch_dir.name.startswith("plot_"):
                print(
                    f"[{sr:3d}] skip (ambiguous shared dir {patch_dir} with {len(pngs)} PNGs;"
                    " missing plot-specific patch_png in manifest)"
                )
                skip.append(sr)
                continue

            esri_input = patch_dir
        else:
            print(f"[{sr:3d}] skip (no PNGs for plot; patch_png={patch_png_raw}, patch_dir={patch_dir})")
            skip.append(sr)
            continue

        if args.nested_dirs:
            target_out_dir = out_root / plot_dir_name(sr)
            existing_chm = sorted(target_out_dir.glob("*_CHM.tif")) if target_out_dir.exists() else []
            if existing_chm and not args.overwrite:
                print(f"[{sr:3d}] skip existing CHM ({existing_chm[0].name})")
                skip.append(sr)
                continue
        else:
            target_out_dir = out_root
            if esri_input.is_file():
                expected_chm = target_out_dir / f"{esri_input.stem}_CHM.tif"
                if expected_chm.exists() and not args.overwrite:
                    print(f"[{sr:3d}] skip existing CHM ({expected_chm.name})")
                    skip.append(sr)
                    continue
            elif not args.overwrite:
                # Conservative skip check for flat mode when input is a directory.
                existing_flat = sorted(target_out_dir.glob("*_CHM.tif"))
                if existing_flat:
                    print(f"[{sr:3d}] skip (flat dir already has CHM outputs; use --overwrite to rerun)")
                    skip.append(sr)
                    continue

        target_out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "run_inference.py",
            "--config",
            args.config,
            "--mode",
            "esri",
            "--esri_dir",
            str(esri_input),
            "--output_dir",
            str(target_out_dir),
        ]

        print(f"[{sr:3d}] running inference -> {target_out_dir}")

        if args.dry_run:
            print("  DRY RUN:", " ".join(cmd))
            ok.append(sr)
            continue

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            ok.append(sr)
        else:
            fail.append(sr)
            tail = (result.stderr or result.stdout or "").strip()[-1200:]
            print(f"  FAILED:\n{tail}", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f"Inference complete. ok={len(ok)} skip={len(skip)} fail={len(fail)}")
    if fail:
        print(f"Failed plot ids: {fail}")


if __name__ == "__main__":
    main()
