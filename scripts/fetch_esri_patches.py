import os
import math
import argparse
import requests
from PIL import Image
import io
from tqdm import tqdm

# ==========================================
# 🗺️ TILE MATH & FETCHING LOGIC
# ==========================================

def latlon_to_tile_xy(lat, lon, zoom):
    """Converts Latitude/Longitude to standard XYZ tile coordinates."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def fetch_single_tile(zoom, x, y):
    """Fetches a single 256x256 tile from the ESRI World Imagery server."""
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    # Headers are required so the server doesn't block us as a bot
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        print(f"⚠️ Warning: Failed to fetch tile Z:{zoom} X:{x} Y:{y} (HTTP {response.status_code})")
        return None

def stitch_512_patch(zoom, start_x, start_y, output_dir, identifier="patch"):
    """
    Fetches a 2x2 grid of 256px tiles and stitches them into one 512x512 PNG.
    """
    # Create a blank 512x512 canvas
    patch = Image.new('RGB', (512, 512))
    
    # Fetch the 4 tiles (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
    tl = fetch_single_tile(zoom, start_x, start_y)
    tr = fetch_single_tile(zoom, start_x + 1, start_y)
    bl = fetch_single_tile(zoom, start_x, start_y + 1)
    br = fetch_single_tile(zoom, start_x + 1, start_y + 1)
    
    # Paste them into the correct quadrants
    if tl: patch.paste(tl, (0, 0))
    if tr: patch.paste(tr, (256, 0))
    if bl: patch.paste(bl, (0, 256))
    if br: patch.paste(br, (256, 256))
    
    # Save the final 512x512 patch
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"esri_512_{identifier}_z{zoom}_{start_x}_{start_y}.png")
    patch.save(filename, format="PNG")
    return filename

# ==========================================
# 🚀 EXECUTION MODES (Single vs Bounding Box)
# ==========================================

def fetch_area_by_bbox(bbox, zoom, output_dir):
    """Calculates all tiles needed to cover a bounding box and fetches them in 512px chunks."""
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Get top-left and bottom-right tile coordinates
    min_x, min_y = latlon_to_tile_xy(max_lat, min_lon, zoom) # Top-Left
    max_x, max_y = latlon_to_tile_xy(min_lat, max_lon, zoom) # Bottom-Right
    
    print(f"\n🌍 Planning fetch for Bounding Box: {bbox}")
    print(f"   Tile Grid: X({min_x} to {max_x}), Y({min_y} to {max_y})")
    
    # Step by 2 because each 512px patch consumes 2x2 standard tiles
    patch_coordinates = []
    for x in range(min_x, max_x + 1, 2):
        for y in range(min_y, max_y + 1, 2):
            patch_coordinates.append((x, y))
            
    print(f"   Total 512x512 patches to generate: {len(patch_coordinates)}\n")
    
    for i, (x, y) in enumerate(tqdm(patch_coordinates, desc="Stitching Patches")):
        stitch_512_patch(zoom, x, y, output_dir, identifier=f"part_{i}")
        
    print(f"\n✅ All patches successfully downloaded to: {output_dir}")

# ==========================================
# 🛠️ CLI ROUTER
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch 512x512 ESRI Patches for CHMv2 Pipeline")
    
    # Arguments for a single center point
    parser.add_argument("--lat", type=float, help="Latitude for a single central patch.")
    parser.add_argument("--lon", type=float, help="Longitude for a single central patch.")
    
    # Arguments for an entire area (Bounding Box)
    parser.add_argument("--bbox", type=float, nargs=4, help="[min_lon, min_lat, max_lon, max_lat] to fetch a grid of patches.")
    
    # Global settings
    parser.add_argument("--zoom", type=int, default=18, help="Zoom level (18 is approx 0.6m/pixel). Default 18.")
    parser.add_argument("--out_dir", type=str, default="data/input/esri_patches", help="Output directory.")
    
    args = parser.parse_args()
    
    if args.bbox:
        fetch_area_by_bbox(args.bbox, args.zoom, args.out_dir)
    elif args.lat and args.lon:
        start_x, start_y = latlon_to_tile_xy(args.lat, args.lon, args.zoom)
        print(f"\n📍 Fetching single 512x512 patch centered near {args.lat}, {args.lon}...")
        path = stitch_512_patch(args.zoom, start_x, start_y, args.out_dir, identifier="center")
        print(f"✅ Saved to: {path}")
    else:
        print("❌ Error: You must provide either --bbox OR both --lat and --lon.")
        print("Example: python fetch_esri_patches.py --lat 30.455 --lon 78.075")