"""
Aerosol Transport Flow Data Processor
=====================================
Downloads GEOS-FP atmospheric data via OPeNDAP (no auth required),
combines wind vectors with aerosol concentration to create
"aerosol flux" vectors, and exports to NetCDF for ArcGIS FlowRenderer.

Data source: NASA GEOS-FP (Forward Processing) 0.25° global
- Wind: U10M, V10M (10-meter wind components)
- Aerosols: BCSMASS (black carbon), DUSMASS (dust),
           SSSMASS (sea salt), OCSMASS (organic carbon)

Output: NetCDF with weighted U/V bands per aerosol species,
        ready to upload to ArcGIS Online as ImageryTileLayer.

Requirements:
    pip install xarray netcdf4 numpy
"""

import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import os

# ── Configuration ──
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# GEOS-FP OPeNDAP endpoints (no authentication needed)
GEOS_WIND_URL = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_2d_asm_Nx"
GEOS_AERO_URL = "https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/tavg3_2d_aer_Nx"

# Aerosol species mapping
AEROSOL_SPECIES = {
    "composite": {
        "vars": ["bcsmass", "dusmass", "sssmass", "ocsmass", "so4smass"],
        "label": "Total Aerosol",
        "description": "All aerosol species combined"
    },
    "black_carbon": {
        "vars": ["bcsmass"],
        "label": "Black Carbon",
        "description": "Combustion, wildfires, soot"
    },
    "dust": {
        "vars": ["dusmass"],
        "label": "Mineral Dust",
        "description": "Saharan, Gobi, Arabian desert dust"
    },
    "sea_salt": {
        "vars": ["sssmass"],
        "label": "Sea Salt",
        "description": "Ocean spray, maritime aerosols"
    }
}


def fetch_latest_data():
    """Fetch the most recent GEOS-FP wind + aerosol data via OPeNDAP."""
    print("=" * 60)
    print("  GEOS-FP Aerosol Transport Data Processor")
    print("=" * 60)

    # Open wind dataset
    print("\n[1/4] Connecting to GEOS-FP wind data (OPeNDAP)...")
    print(f"  URL: {GEOS_WIND_URL}")
    wind_ds = xr.open_dataset(GEOS_WIND_URL)

    # Get the latest available time step
    latest_time = wind_ds.time[-1].values
    print(f"  Latest available: {np.datetime_as_string(latest_time, unit='h')}")

    # Select latest time, load U10M and V10M
    print("  Loading U10M, V10M...")
    wind = wind_ds[["u10m", "v10m"]].sel(time=latest_time)
    u10m = wind["u10m"].values  # shape: (lat, lon)
    v10m = wind["v10m"].values
    print(f"  Wind shape: {u10m.shape}")
    wind_ds.close()

    # Open aerosol dataset
    print("\n[2/4] Connecting to GEOS-FP aerosol data (OPeNDAP)...")
    print(f"  URL: {GEOS_AERO_URL}")
    aero_ds = xr.open_dataset(GEOS_AERO_URL)

    # Select closest time to wind data
    aero_time = aero_ds.time.sel(time=latest_time, method="nearest").values
    print(f"  Matched time: {np.datetime_as_string(aero_time, unit='h')}")

    # Load all aerosol species
    aero_vars = ["bcsmass", "dusmass", "sssmass", "ocsmass", "so4smass"]
    print(f"  Loading: {', '.join(aero_vars)}...")
    aero = aero_ds[aero_vars].sel(time=aero_time)

    aerosol_data = {}
    for var in aero_vars:
        data = aero[var].values
        # Replace NaN/fill values with 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        aerosol_data[var] = data
        vmax = np.nanmax(data)
        print(f"    {var}: max={vmax:.2e} kg/m²")

    lats = aero["lat"].values
    lons = aero["lon"].values
    aero_ds.close()

    return u10m, v10m, aerosol_data, lats, lons, latest_time


def compute_aerosol_flux(u10m, v10m, aerosol_data, species_config):
    """
    Weight wind vectors by aerosol concentration.

    The result is "aerosol flux" — wind direction preserved,
    but magnitude modulated by how much aerosol is present.
    Areas with high concentration + strong wind = bright particles.
    Areas with no aerosol = no particles (near zero magnitude).
    """
    # Sum the relevant species
    total = np.zeros_like(u10m)
    for var in species_config["vars"]:
        total += aerosol_data[var]

    # Normalize to 0-1 range (using 95th percentile to avoid outlier stretching)
    p95 = np.percentile(total[total > 0], 95) if np.any(total > 0) else 1.0
    normalized = np.clip(total / p95, 0, 1)

    # Apply a power curve to enhance contrast (low values more visible)
    enhanced = np.power(normalized, 0.5)

    # Weight wind vectors
    u_weighted = u10m * enhanced
    v_weighted = v10m * enhanced

    return u_weighted, v_weighted


def export_netcdf(u_weighted, v_weighted, lats, lons, species_name, timestamp, output_dir):
    """Export weighted UV vectors to NetCDF-3 (ArcGIS compatible)."""
    filename = f"aerosol_flow_{species_name}.nc"
    filepath = os.path.join(output_dir, filename)

    ds = xr.Dataset(
        {
            "U": (["lat", "lon"], u_weighted.astype(np.float32)),
            "V": (["lat", "lon"], v_weighted.astype(np.float32)),
        },
        coords={
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "title": f"Aerosol Transport Flow - {AEROSOL_SPECIES[species_name]['label']}",
            "description": AEROSOL_SPECIES[species_name]["description"],
            "source": "NASA GEOS-FP 0.25deg via OPeNDAP",
            "timestamp": str(timestamp),
            "processing": "Wind (U10M, V10M) weighted by aerosol surface mass concentration",
            "conventions": "CF-1.6",
        }
    )

    # Set variable attributes for ArcGIS vector field recognition
    ds["U"].attrs = {"long_name": "U-component of aerosol flux", "units": "m/s * kg/m2"}
    ds["V"].attrs = {"long_name": "V-component of aerosol flux", "units": "m/s * kg/m2"}
    ds["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}
    ds["lon"].attrs = {"long_name": "longitude", "units": "degrees_east"}

    # Save as NetCDF-3 Classic (most compatible with ArcGIS)
    ds.to_netcdf(filepath, format="NETCDF3_CLASSIC")
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"    Saved: {filename} ({size_mb:.1f} MB)")
    return filepath


def main():
    # Step 1-2: Fetch data
    u10m, v10m, aerosol_data, lats, lons, timestamp = fetch_latest_data()

    # Step 3: Compute weighted vectors for each species
    print("\n[3/4] Computing aerosol-weighted wind vectors...")
    output_files = {}

    for species_name, config in AEROSOL_SPECIES.items():
        print(f"\n  Processing: {config['label']}")
        u_w, v_w = compute_aerosol_flux(u10m, v10m, aerosol_data, config)

        wind_mag = np.sqrt(u_w**2 + v_w**2)
        print(f"    Flux magnitude: mean={np.mean(wind_mag):.4f}, max={np.max(wind_mag):.4f}")

        filepath = export_netcdf(u_w, v_w, lats, lons, species_name, timestamp, OUTPUT_DIR)
        output_files[species_name] = filepath

    # Step 4: Summary
    print("\n[4/4] Done!")
    print("=" * 60)
    print("  Output files:")
    for name, path in output_files.items():
        print(f"    {name}: {path}")
    print()
    print("  Next steps:")
    print("  1. Sign in to ArcGIS Online (developers.arcgis.com)")
    print("  2. Go to Content > New Item > Imagery Layer")
    print("  3. Upload each .nc file, select 'Vector field' template")
    print("  4. Set input type to 'U-V' (two components)")
    print("  5. Publish as Tiled Imagery Layer")
    print("  6. Use the service URL in FlowRenderer")
    print()
    print("  Or run: python publish_to_arcgis.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
