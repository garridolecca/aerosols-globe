"""
Encode aerosol-weighted wind U/V data as PNG images.
Same technique used by earth.nullschool.net and windy.com.

Each pixel encodes:
  R channel = U component (mapped from [-max, +max] to [0, 255])
  G channel = V component (mapped from [-max, +max] to [0, 255])
  B channel = magnitude (0-255)
  A channel = 255 (opaque)

Output: 360x181 PNG per species (1-degree resolution, ~30KB each)
Plus a metadata JSON with scale factors for decoding.
"""
import numpy as np
import json
import os

try:
    from PIL import Image
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "Pillow", "-q"])
    from PIL import Image

import xarray as xr

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Target resolution: 1 degree (360x181)
TARGET_LONS = np.arange(-180, 180, 1.0)  # 360
TARGET_LATS = np.arange(90, -91, -1.0)   # 181 (top to bottom for image)

SPECIES = ["composite", "black_carbon", "dust", "sea_salt"]


def encode_component(data, max_val):
    """Map [-max_val, +max_val] to [0, 255]."""
    normalized = (data / max_val + 1.0) * 0.5  # 0 to 1
    return np.clip(normalized * 255, 0, 255).astype(np.uint8)


def main():
    print("Encoding aerosol flow data as PNG images...")

    metadata = {
        "width": len(TARGET_LONS),
        "height": len(TARGET_LATS),
        "lon_min": -180, "lon_max": 179,
        "lat_min": -90, "lat_max": 90,
        "species": {}
    }

    for species in SPECIES:
        nc_path = os.path.join(OUTPUT_DIR, f"aerosol_flow_{species}.nc")
        if not os.path.exists(nc_path):
            print(f"  SKIP: {nc_path} not found")
            continue

        print(f"\n  Processing: {species}")
        ds = xr.open_dataset(nc_path)
        u = ds["U"].values  # (lat, lon) - lat is -90 to 90
        v = ds["V"].values

        # Flip lat to be top-to-bottom (90 to -90) for image encoding
        u = np.flipud(u)
        v = np.flipud(v)

        # Downsample from 0.25 to 1 degree using mean
        src_lats = np.flipud(ds["lat"].values)
        src_lons = ds["lon"].values

        # Simple nearest-neighbor resample to target grid
        lat_idx = np.searchsorted(-src_lats, -TARGET_LATS)  # descending
        lon_idx = np.searchsorted(src_lons, TARGET_LONS)
        lat_idx = np.clip(lat_idx, 0, u.shape[0] - 1)
        lon_idx = np.clip(lon_idx, 0, u.shape[1] - 1)

        u_resamp = u[np.ix_(lat_idx, lon_idx)]
        v_resamp = v[np.ix_(lat_idx, lon_idx)]

        # Replace NaN
        u_resamp = np.nan_to_num(u_resamp, 0)
        v_resamp = np.nan_to_num(v_resamp, 0)

        # Compute magnitude
        mag = np.sqrt(u_resamp**2 + v_resamp**2)

        # Find max for encoding scale
        max_uv = max(np.abs(u_resamp).max(), np.abs(v_resamp).max(), 0.01)
        max_mag = mag.max() if mag.max() > 0 else 1.0

        print(f"    Shape: {u_resamp.shape}")
        print(f"    Max UV: {max_uv:.4f}, Max Mag: {max_mag:.4f}")

        # Encode as RGBA image
        r = encode_component(u_resamp, max_uv)
        g = encode_component(v_resamp, max_uv)
        b = np.clip((mag / max_mag) * 255, 0, 255).astype(np.uint8)
        a = np.full_like(r, 255)

        rgba = np.stack([r, g, b, a], axis=-1)
        img = Image.fromarray(rgba, 'RGBA')

        png_path = os.path.join(DATA_DIR, f"{species}.png")
        img.save(png_path, optimize=True)
        size_kb = os.path.getsize(png_path) / 1024
        print(f"    Saved: {png_path} ({size_kb:.1f} KB)")

        metadata["species"][species] = {
            "max_uv": float(max_uv),
            "max_mag": float(max_mag),
            "file": f"data/{species}.png"
        }

        ds.close()

    # Save metadata
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata: {meta_path}")
    print("\nDone! PNG files ready for GitHub Pages.")


if __name__ == "__main__":
    main()
