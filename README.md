# openCHM: Satellite Canopy Height Mapping & Field Validation

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20Ready-ee4c2c.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**openCHM** is an end-to-end canopy height inference and field-plot validation pipeline built on Meta's **CHMv2** model (DINOv3-ViT-L + DPT head). It ingests globally available ESRI World Imagery, runs on Apple Silicon (MPS), and includes a full statistical analysis framework for validating CHM predictions against field-measured tree heights.

The target application is **forest carbon accounting** — specifically, validating whether optical satellite-based canopy height estimation can replace or supplement LiDAR/PolInSAR-derived CHMs for above-ground biomass estimation.

---

## Bigger Picture

```
ESRI World Imagery (zoom 18)
        │
        ▼
  CHMv2 Foundation Model          ← this pipeline
  (per-tile height inference)
        │
        ▼
  Canopy Height Map (GeoTIFF)
        │
        ├── validate vs field plots (Barkot, Uttarakhand)
        │
        └── carbon_accounting_tool  ← downstream consumer
              (AGB + carbon density from CHM)
```

Accurate CHM is the critical dependency for carbon estimates. A 10 m height error on a 30 m Sal tree translates to >50% error in above-ground biomass due to the non-linear allometric relationship. This pipeline quantifies that error, identifies its sources, and provides a roadmap to reduce it.

---

## Pipeline

```
  INPUT
  ─────
  field_plots.geojson          config.yaml
  (sr, lat, lon, h_avg)        (model, tiling, output settings)
        │
        ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 1 — FETCH  (scripts/fetch_all_plots.py)                   │
  │                                                                 │
  │  • Compute tile (x, y) for each plot at zoom 18                 │
  │  • Deduplicate: plots sharing a tile → one 512×512 PNG          │
  │  • Stitch 2×2 ESRI 256px tiles → esri_512_z18_{x}_{y}.png      │
  │  • Write manifest: sr → {tile_key, tile_png, lat, lon, h_avg}   │
  └──────────────────────────┬──────────────────────────────────────┘
                             │
               data/input/esri_patches/
               data/input/plot_manifest.json
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 2 — INFERENCE  (run_inference.py --mode validate)         │
  │                                                                 │
  │  • Load CHMv2 once (DINOv3-ViT-L + DPT head)                   │
  │  • Per unique tile: forward pass → height map (float32, metres) │
  │  • Save esri_512_z18_{x}_{y}_CHM.tif  (EPSG:3857 GeoTIFF)      │
  │  • Save esri_512_z18_{x}_{y}_EMB.png  (PCA embedding vis)       │
  └──────────────────────────┬──────────────────────────────────────┘
                             │
               data/output/esri_results/
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 3 — EXTRACT DATASET  (scripts/extract_plot_dataset.py)    │
  │                                                                 │
  │  • Per field plot: locate CHM GeoTIFF via tile key              │
  │  • Extract 12.5×12.5 m footprint (≈24 px at zoom 18)           │
  │  • Compute: mean, median, std, min, max, coverage ratio         │
  │  • Save: pixels.npy, metadata.json, bbox_overlay.png            │
  │  • Write: dataset_summary.csv (all 100 plots)                   │
  └──────────────────────────┬──────────────────────────────────────┘
                             │
         scripts/statistical_analysis/data/
         ├── plot_001/ … plot_100/     (per-plot pixel arrays)
         └── dataset_summary.csv
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │  STEP 4 — STATISTICAL ANALYSIS  (notebooks/)                    │
  │                                                                 │
  │  4a. Global metrics & diagnostics       visualize_comp.ipynb    │
  │  4b. Filter validation (F1, F2)         visualize_comp.ipynb    │
  │  4c. Publication figures                visualize_comp.ipynb    │
  │  4d. Worst-plot inspection              worst_plots_inspection  │
  └─────────────────────────────────────────────────────────────────┘
```

---

## Validation Results — Barkot Range (n=100 field plots)

**Field dataset:** Khati 2014, Barkot Range, Uttarakhand. 100 plots, 12.5×12.5 m footprint, measured h_avg (mean tree height per plot).  
**Model:** CHMv2, ESRI World Imagery zoom 18 (~0.6 m/px per 256px tile; each esri_512 covers 2×2 tiles).  
**Benchmark:** PolInSAR Three-Stage Inversion, same plots — RMSE = 2.28 m, r = 0.62.

### Measured Performance

| Metric | All 100 plots | After F1+F2 filter (n=82) |
|--------|:---:|:---:|
| RMSE | **8.08 m** | **6.80 m** |
| MAE | 6.34 m | — |
| Mean Bias | +1.49 m | — |
| Pearson r | 0.099 (p=0.33) | — |
| vs PolInSAR RMSE | 3.5× worse | 3.0× worse |

The Pearson r of 0.099 is **not statistically significant** (p=0.33). The model has essentially no linear correlation with field heights across these 100 plots.

### Failure Mode 1 — GPS Coordinate Errors

Visual inspection of the 5 worst plots (|Δ| > 17 m) reveals that field GPS coordinates for several plots land on **roads and forest clearings**, not on canopy. CHMv2 correctly predicts near-zero height for roads — the error is in the field dataset coordinates, not the model.

**Diagnostic signal:** `pct_below5` — fraction of CHM pixels within the bbox predicting height < 5 m.  
- Pearson r with |error| = **0.671** (strongest predictor by far)
- Plots with GPS errors cluster at pct_below5 ≈ 1.0 (100% of pixels sub-canopy)
- Distribution is bimodal: valid plots near 0, coordinate-error plots near 1.0

**Filter F1:** `pct_below5 < 0.50` — removes plots where the majority of the footprint is non-forest.

### Failure Mode 2 — Tile-Edge Truncation

Some GPS coordinates fall within 12 pixels of the 512px tile border. Because the esri_512 image covers a fixed 2×2 tile area, a plot near the edge has part of its 12.5 m footprint outside the fetched image. The extracted pixel array is truncated, biasing statistics.

**Filter F2:** `tile_edge_dist >= 12 px` — removes plots with insufficient margin to tile border.

### Failure Mode 3 — Optical Saturation (Model Limitation)

For the remaining 82 filtered plots, the model still underestimates tall mature Sal forest. Error increases with true height:

| True Height Class | Mean Bias (m) |
|---|---|
| < 20 m | near zero |
| 20–24 m | −2 to −4 m |
| 24–27 m | −5 to −8 m |
| > 27 m | **−8 to −12 m** |

**Why:** ESRI imagery sees only the top leaf layer. A 20 m and a 30 m closed-canopy Sal stand are visually indistinguishable from nadir — both present a continuous green texture. CHMv2 relies on shadows and crown gaps for depth inference, neither of which exists in dense multi-story tropical forest. The model's training distribution skews toward sparse boreal and urban trees where ground is visible.

This is a **fundamental limitation of monocular optical depth estimation** in closed-canopy tropical forest, not a fixable pipeline issue.

### Consequence for Carbon Accounting

Biomass scales non-linearly with height (allometric exponent ≈ 2.5 for Sal). Missing 10 m of a 30 m tree height does not mean missing 33% of biomass — it means missing **>50% of carbon weight**. Until RMSE approaches benchmark levels (~2–3 m), optical CHM from this pipeline should not be used as the primary height input for carbon estimates; it can serve as a spatial prior or auxiliary variable alongside GEDI/PolInSAR.

---

## RMSE Improvement Roadmap

Listed in priority order. Only F1+F2 filtering is currently implemented.

| # | Method | Expected RMSE gain | Status |
|---|--------|-------------------|--------|
| 1 | **F1+F2 coordinate & edge filter** | −1.28 m (−15.9%) | ✅ Done |
| 2 | **Tile centering fix** | unknown (moderate) | 🔲 Pending |
| 3 | **Aggregation: median or p25–p75 mean** | unknown (small–moderate) | 🔲 Pending |
| 4 | **GEDI L2A bias calibration** | potentially large | 🔲 Pending |
| 5 | **Zoom 19 refetch** (0.3 m/px) | unknown | 🔲 Pending |
| 6 | **Terrain correction** (Copernicus DEM) | unknown | 🔲 Pending |

**Tile centering fix:** `latlon_to_tile_xy` returns the tile whose top-left corner is nearest the plot coordinate. A plot near a tile edge may be in the wrong pixel region of the 512px canvas. Fix: compute sub-tile pixel offset and re-center the 2×2 stitch so the plot lands near pixel (256, 256).

**Aggregation:** Currently using per-bbox mean. Road/shadow pixels drag the mean down. Median or interquartile mean is more robust. Testable immediately from existing `pixels.npy` without re-fetching.

**GEDI calibration:** Linear rescaling `H_cal = a × H_CHM + b` fit on nearby GEDI L2A LiDAR shots. Can correct systematic bias without changing the model. Requires fetching GEDI data for the Barkot bbox.

**GPS accuracy floor:** Field GPS at Barkot likely has 3–10 m horizontal uncertainty. A 5 m GPS shift on a 12.5 m plot means ~40% of the bbox is on the wrong pixels. This sets a theoretical RMSE floor independent of all other improvements.

---

## Installation

```bash
git clone https://github.com/yourusername/openCHM.git
cd openCHM

micromamba env create -f environment.yml
micromamba activate chmv2

# Verify Apple Silicon GPU
python -c "import torch; print(torch.backends.mps.is_available())"
```

### HuggingFace model access

```bash
pip install -U "huggingface_hub[cli]"
hf auth login          # interactive — required once
hf auth whoami         # verify
```

Model: [facebook/dinov3-vitl16-chmv2-dpt-head](https://huggingface.co/facebook/dinov3-vitl16-chmv2-dpt-head) — request access before first use.

---

## Usage

### Full validation pipeline

```bash
# Step 1: fetch tiles (tile-deduplicated, idempotent)
python scripts/fetch_all_plots.py \
  --plots_geojson data/input/field_plots.geojson \
  --zoom 18 \
  --out_dir data/input/esri_patches \
  --manifest data/input/plot_manifest.json

# Step 2: infer + compare + visualise (model loads once)
python run_inference.py --mode validate

# Options
python run_inference.py --mode validate --overwrite              # re-run existing CHMs
python run_inference.py --mode validate --min_coverage_ratio 0.85
```

### Ad-hoc ESRI patch

```bash
python scripts/fetch_esri_patches.py \
  --lat 30.455 --lon 78.075 --zoom 18 \
  --out_dir data/input/esri_patches

python run_inference.py --mode esri --esri_dir data/input/esri_patches
```

### Extract per-plot pixel dataset

```bash
python scripts/extract_plot_dataset.py \
  --plots_geojson data/input/field_plots.geojson \
  --chm_dir data/output/esri_results \
  --output_dir scripts/statistical_analysis/data \
  --plot_size_m 12.5
```

### Statistical analysis notebooks

```bash
jupyter notebook scripts/statistical_analysis/notebooks/
```

| Notebook | Contents |
|---|---|
| `visualize_comp.ipynb` | Global metrics · 4-panel diagnostic · filter evidence (F1/F2) · publication figures · per-plot detail panels |
| `worst_plots_inspection.ipynb` | ESRI + CHM panels for top-5 worst plots with sr labels — visual GPS error confirmation |

---

## Project Structure

```
openCHM/
├── run_inference.py              ← entry point (stac / esri / validate)
├── config.yaml
├── environment.yml
│
├── pipeline/
│   ├── model.py                  ← CHMv2 loader
│   ├── inference.py              ← batched forward pass
│   ├── tiling.py                 ← patch extraction
│   ├── visualise.py              ← GeoTIFF writer + PCA vis
│   └── runner.py                 ← pipeline orchestration
│
├── scripts/
│   ├── fetch_esri_patches.py     ← single-tile ESRI fetch
│   ├── fetch_all_plots.py        ← tile-deduplicated fetch + manifest
│   ├── extract_plot_dataset.py   ← per-plot pixel extraction → pixels.npy
│   ├── validation_common.py      ← shared types, metrics, geometry
│   ├── compare_heights.py        ← CHM vs field → CSV + scatter
│   └── visualize_plots.py        ← tile panels + dashboard
│
├── scripts/statistical_analysis/
│   ├── data/
│   │   ├── plot_001/ … plot_100/
│   │   │   ├── pixels.npy        ← 2D CHM array (bbox footprint)
│   │   │   ├── metadata.json     ← sr, lat, lon, bbox, pred stats
│   │   │   └── bbox_overlay.png
│   │   └── dataset_summary.csv   ← aggregated stats (100 plots)
│   └── notebooks/
│       ├── visualize_comp.ipynb
│       └── worst_plots_inspection.ipynb
│
└── data/
    ├── input/
    │   ├── field_plots.geojson
    │   ├── plot_manifest.json
    │   └── esri_patches/             ← esri_512_z18_{x}_{y}.png
    └── output/
        ├── esri_results/             ← *_CHM.tif, *_EMB.png
        ├── pub_4panel_*.png          ← publication figure (ESRI + CHM + plot details)
        ├── worst_plots_inspection.png
        ├── filter_diagnostics.png    ← F1/F2 correlation evidence
        ├── statistical_diagnostics.png
        ├── validation_results.csv
        └── validation_metrics.json
```

---

## Known Limitations

| Issue | Root cause | Current mitigation |
|---|---|---|
| RMSE 8.08 m vs benchmark 2.28 m | Optical saturation in closed-canopy forest | Ongoing — see roadmap |
| Near-zero predictions (sr=41,42,92,94) | GPS coordinates land on roads | F1 filter removes |
| Tile-edge truncation (sr=94) | bbox at row_min=0 | F2 filter removes |
| No correlation with field heights (r=0.10) | Model training domain mismatch | GEDI calibration planned |
| Tall tree underestimation (>27 m, −8 to −12 m) | Closed-canopy optical opacity | Fundamental — needs LiDAR fusion |
| GPS accuracy floor | Field GPS ±3–10 m on 12.5 m plots | Sets theoretical RMSE minimum |

---

## References

- **CHMv2:** Tollefson et al., arXiv:2603.06382
- **DINOv3:** arXiv:2508.10104
- **Model weights:** [facebook/dinov3-vitl16-chmv2-dpt-head](https://huggingface.co/facebook/dinov3-vitl16-chmv2-dpt-head)
- **Field benchmark:** Khati et al. 2014 — PolInSAR Three-Stage Inversion, Barkot Range, Uttarakhand
- **Imagery:** ESRI World Imagery (zoom 18)
- **DEM:** Copernicus GLO-30 via Microsoft Planetary Computer

---

## License

MIT — see [LICENSE](LICENSE).
