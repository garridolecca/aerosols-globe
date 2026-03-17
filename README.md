# Earth in Action — Aerosols Globe

An interactive 3D globe visualization of atmospheric aerosols using ArcGIS Maps SDK for JavaScript.

## Live Demo

**[View Live](https://jhon9904.github.io/aerosols-globe/)**

## Features

- **3D rotating globe** with stars and atmospheric effects
- **Four aerosol layers** — toggle independently:
  - Aerosol Composite (MODIS Terra AOD)
  - Black Carbon (combustion & wildfire emissions)
  - Dust (wind-lofted desert particles)
  - Sea Salt (ocean spray & maritime aerosols)
- **Real-time data** from NASA GEOS-5 and ArcGIS Living Atlas
- Layer opacity control and animated time stepping
- Dark theme with screen-blended overlays

## Data Sources

- [NASA GEOS-5](https://gmao.gsfc.nasa.gov/GEOS_systems/) Earth system model
- [NASA NEO](https://neo.gsfc.nasa.gov/) WMS services (MODIS aerosol products)
- [ArcGIS Living Atlas](https://livingatlas.arcgis.com/) imagery tile services

## Tech Stack

- ArcGIS Maps SDK for JavaScript 4.31 (SceneView)
- Calcite Design System (dark theme)
- Vanilla HTML/CSS/JS — no build step required

## Usage

Open `index.html` in a browser, or visit the GitHub Pages deployment above.
