# Earth in Action — Aerosols

GPU-accelerated visualization of real atmospheric aerosol transport using NASA GEOS-FP satellite data and NVIDIA WebGPU compute shaders.

**[View Live Demo](https://garridolecca.github.io/aerosols-globe/)**

## System Architecture

![System Architecture](diagram.svg)

## Features

- **500,000 GPU-accelerated particles** via WebGPU compute shaders (falls back to Canvas 2D with 8K particles)
- **Real NASA aerosol data** — wind vectors weighted by aerosol concentration from GEOS-FP
- **Four aerosol species** — Total Composite, Black Carbon, Mineral Dust, Sea Salt
- **4-pass GPU pipeline** — Compute (advection) → Fade (trails) → Render (SDF glow) → Screen (composite)
- **Interactive controls** — particle count (up to 1M), speed, trail length, particle size
- **ArcGIS basemap** — Dark Gray Vector via ArcGIS Maps SDK 5.0
- **Auto-fallback** — WebGPU → Canvas 2D for browser compatibility

## Data Pipeline

```
NASA GEOS-FP (OPeNDAP) → Python (weight wind × aerosol) → PNG encoding → WebGPU texture → Compute shader → Screen
```

1. **Data Acquisition** — Wind (U10M, V10M) + aerosol mass (BCSMASS, DUSMASS, SSSMASS, OCSMASS, SO4SMASS) from NASA GEOS-FP at 0.25° global resolution
2. **Flux Computation** — `flux = wind × normalized(aerosol_concentration)`
3. **PNG Encoding** — U→Red, V→Green, Magnitude→Blue (~100KB per species)
4. **GPU Rendering** — WebGPU compute shader advects 500K particles in parallel, rendered with SDF glow and additive blending

## Tech Stack

| Component | Technology |
|-----------|-----------|
| GPU Compute | WebGPU API + WGSL shaders |
| Particle Rendering | Instanced quads, SDF glow, additive blending |
| Basemap | ArcGIS Maps SDK 5.0 (Dark Gray Vector) |
| Data Source | NASA GMAO GEOS-FP 0.25° via OPeNDAP |
| Processing | Python (xarray, NumPy, Pillow) |
| Hosting | GitHub Pages |

## Refreshing Data

```bash
python process_aerosols.py    # Download latest GEOS-FP data
python encode_wind_png.py     # Encode as PNG
git add data/ && git commit -m "Update aerosol data" && git push
```

## Credits

Created by **Jhonatan Garrido-Lecca**

Powered by NASA GEOS-FP, NVIDIA WebGPU, and ArcGIS Maps SDK.
