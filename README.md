# Earth in Action — Aerosols Globe

Interactive 3D globe visualization of atmospheric aerosols with animated ocean current flow, built with ArcGIS Maps SDK for JavaScript 5.0.

## Live Demo

**[View Live](https://garridolecca.github.io/aerosols-globe/)**

## Features

- **3D rotating globe** with stars, atmosphere, and weather effects
- **Four distinct aerosol layers** — toggle independently:
  - **Total AOD** — NASA MODIS Combined Value-Added AOD (daily satellite)
  - **Absorbing Aerosols** — NASA OMI Absorbing AOD (smoke, soot, black carbon)
  - **Dust** — NASA MERRA-2 Dust Surface Mass Concentration (monthly reanalysis)
  - **Sea Salt** — ECMWF CAMS Sea Salt AOD at 550nm (forecast model)
- **Animated ocean current flow** — FlowRenderer particle trails on Global Drifter Program data
- Layer opacity, flow density, and flow speed controls
- Auto-rotating globe with play/pause
- Live status badges per layer (LOADING / LIVE / ERROR)
- Dark theme with screen-blended glowing overlays

## Data Sources

| Layer | Source | Temporal | Service |
|-------|--------|----------|---------|
| Total AOD | NASA MODIS (Terra + Aqua) | Daily | GIBS WMTS |
| Absorbing AOD | NASA OMI (Aura) | Daily | GIBS WMTS |
| Dust | NASA MERRA-2 Reanalysis | Monthly | GIBS WMTS |
| Sea Salt | ECMWF Copernicus CAMS | Forecast | WMS |
| Ocean Flow | Global Drifter Program | Climatology | Esri Living Atlas |

## Tech Stack

- ArcGIS Maps SDK for JavaScript 5.0 (SceneView, ES modules)
- FlowRenderer for animated ocean current particles
- WebTileLayer (NASA GIBS), WMSLayer (CAMS), ImageryTileLayer (Esri)
- Vanilla HTML/CSS/JS — no build step required

## Usage

Open `index.html` in a browser, or visit the GitHub Pages deployment above.
