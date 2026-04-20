import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ── Paths ──────────────────────────────────────────────────────────────────────
# Since optimize_rmse.py is in 'scripts/', we go 1 level up to reach the root
_repo_root = Path(__file__).resolve().parent.parent

# Define the path to your data
# Based on your sidebar: scripts -> statistical_analysis -> data
STATS_ROOT = _repo_root / "scripts" / "statistical_analysis" / "data"

# Verify path for debugging
if not STATS_ROOT.exists():
    print(f"ERROR: STATS_ROOT not found at: {STATS_ROOT}")
else:
    print(f"Success: Connected to {STATS_ROOT}")
def extract_plot_data(pdir, meta):
    """Extracts features, filters (pct_below5, edge_dist), and height metrics."""
    bb = meta["bbox"]
    sr = meta["sr"]
    h_true = meta["h_avg_field"]
    
    # 1. Edge Distance Calculation
    tile_edge_dist = int(min(bb["row_min"], bb["col_min"], 512 - bb["row_max"], 512 - bb["col_max"]))
    
    # 2. Pixel Extraction
    pixels_path = pdir / "pixels.npy"
    if pixels_path.exists():
        pixels = np.load(pixels_path).astype(float)
        valid_px = pixels[np.isfinite(pixels) & (pixels > 0)]
        
        if len(valid_px) > 0:
            h_mean = np.mean(valid_px)
            h_p90 = np.percentile(valid_px, 90)
            pct_below5 = float((valid_px < 5).mean())
        else:
            h_mean, h_p90, pct_below5 = np.nan, np.nan, np.nan
    else:
        h_mean, h_p90, pct_below5 = np.nan, np.nan, np.nan
        
    # 3. Apply your statistical filter logic
    # Keep if pct_below5 < 0.50 AND edge_dist >= 12
    f1_keep = pct_below5 < 0.50
    f2_keep = tile_edge_dist >= 12
    keep_plot = f1_keep and f2_keep
    
    return {
        "sr": sr,
        "h_true": h_true,
        "h_pred_mean": h_mean,
        "h_pred_p90": h_p90,
        "pct_below5": pct_below5,
        "tile_edge_dist": tile_edge_dist,
        "keep": keep_plot
    }

def main():
    print("Loading data and applying statistical filters...\n")
    records = []
    
    for pdir in sorted(STATS_ROOT.iterdir()):
        mf = pdir / "metadata.json"
        if pdir.is_dir() and mf.exists():
            meta = json.loads(mf.read_text())
            records.append(extract_plot_data(pdir, meta))
            
    df = pd.DataFrame(records).dropna()
    
    # ── 1. Baseline (Raw Mean, All Plots) ───────────────────────────────────────
    rmse_raw = np.sqrt(mean_squared_error(df["h_true"], df["h_pred_mean"]))
    print(f"Total plots                 : {len(df)}")
    print(f"RMSE (Baseline Mean)        : {rmse_raw:.2f} m\n")
    
    # ── 2. Apply F1/F2 Filtering ────────────────────────────────────────────────
    df_filtered = df[df["keep"] == True].copy()
    dropped_sr = df[df["keep"] == False]["sr"].tolist()
    
    rmse_filtered_mean = np.sqrt(mean_squared_error(df_filtered["h_true"], df_filtered["h_pred_mean"]))
    
    print(f"Removed by F1+F2            : {len(dropped_sr)} plots")
    print(f"Kept                        : {len(df_filtered)}")
    print(f"RMSE (Filtered Mean)        : {rmse_filtered_mean:.2f} m\n")

    # ── 3. The P90 Shift ────────────────────────────────────────────────────────
    rmse_filtered_p90 = np.sqrt(mean_squared_error(df_filtered["h_true"], df_filtered["h_pred_p90"]))
    print(f"--- Strategy 1: The P90 Shift ---")
    print(f"RMSE (Filtered P90)         : {rmse_filtered_p90:.2f} m")
    print(f"P90 Improvement over Mean   : {rmse_filtered_mean - rmse_filtered_p90:.2f} m\n")

    # ── 4. Statistical Calibration ──────────────────────────────────────────────
    X = df_filtered[["h_pred_p90"]].values
    y = df_filtered["h_true"].values
    
    calibrator = LinearRegression()
    calibrator.fit(X, y)
    
    df_filtered["h_calibrated"] = calibrator.predict(X)
    rmse_calibrated = np.sqrt(mean_squared_error(df_filtered["h_true"], df_filtered["h_calibrated"]))
    
    print(f"--- Strategy 2: Linear Calibration ---")
    print(f"Calibration Equation        : H_true = {calibrator.coef_[0]:.2f} * H_p90 + {calibrator.intercept_:.2f}")
    print(f"RMSE (Calibrated)           : {rmse_calibrated:.2f} m\n")
    
    # ── Final Summary ───────────────────────────────────────────────────────────
    total_improvement = rmse_raw - rmse_calibrated
    percent_reduction = (total_improvement / rmse_raw) * 100
    print("="*50)
    print("                FINAL OPTIMIZATION SUMMARY")
    print("="*50)
    print(f"Starting RMSE               : {rmse_raw:.2f} m")
    print(f"Final Optimized RMSE        : {rmse_calibrated:.2f} m")
    print(f"Total RMSE Reduction        : {total_improvement:.2f} m ({percent_reduction:.1f}% improvement)")
    print("="*50)

if __name__ == "__main__":
    main()