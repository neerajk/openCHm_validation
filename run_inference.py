"""
CHMv2 Canopy Height Inference Pipeline
======================================
DINOv3 + DPT head for per-pixel canopy height estimation (metres).

Usage:
    python run_inference.py --config config.yaml --mode stac
    python run_inference.py --config config.yaml --mode esri --esri_dir data/input/esri_patches
"""

import argparse
import yaml
from pathlib import Path
from pipeline.runner import StacInferencePipeline, EsriPatchInferencePipeline

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(
        description="CHMv2 Canopy Height Inference — DINOv3 backbone"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--mode", 
        choices=["stac", "esri"], 
        default="stac", 
        help="Choose 'stac' for full-scene TIF upscaling, or 'esri' for directory of PNG patches."
    )
    parser.add_argument(
        "--esri_dir",
        type=str,
        default="data/input/esri_patches",
        help="Path to directory containing ESRI PNG patches (only used in 'esri' mode)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"\n{'='*60}")
    print("  CHMv2 Canopy Height Pipeline  (DINOv3 + DPT Head)")
    print(f"{'='*60}")
    print(f"  Mode   : {args.mode.upper()}")
    
    if args.mode == "stac":
        print(f"  Input  : {cfg['input']['image_path']}")
    else:
        print(f"  Input  : {args.esri_dir}/*.png")
        
    print(f"  Output : {cfg['output']['output_dir']}")
    print(f"  Device : {cfg['model']['device']}")
    print(f"{'='*60}\n")

    if args.mode == "stac":
        pipeline = StacInferencePipeline(cfg)
        pipeline.run()
        
    elif args.mode == "esri":
        pipeline = EsriPatchInferencePipeline(cfg, args.esri_dir)
        pipeline.run()

if __name__ == "__main__":
    main()