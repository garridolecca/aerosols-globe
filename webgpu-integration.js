// ============================================================================
//  WebGPU Integration for Aerosol Globe
//  Drop-in replacement for Canvas 2D particles with automatic fallback
// ============================================================================

import { WebGPUParticleSystem, SPECIES_COLOR_MODE } from "./webgpu-particles.js";

/**
 * Initialize the WebGPU particle system, overlaid on the ArcGIS MapView.
 *
 * Usage from index.html:
 *   import { initWebGPUParticles } from "./webgpu-integration.js";
 *   const gpu = await initWebGPUParticles(view, canvas);
 *   if (!gpu) { /* fall back to existing Canvas 2D system * / }
 *
 * @param {__esri.MapView} view - ArcGIS MapView instance
 * @param {HTMLCanvasElement} canvas - The particle overlay canvas (#particleCanvas)
 * @returns {object|null} Controller object, or null if WebGPU unavailable
 */
export async function initWebGPUParticles(view, canvas) {
  // --- Feature detection ---
  if (!navigator.gpu) {
    console.warn("[WebGPU] navigator.gpu not available. Using Canvas 2D fallback.");
    return null;
  }

  const gpu = new WebGPUParticleSystem(canvas, view);

  try {
    const ok = await gpu.init();
    if (!ok) {
      console.warn("[WebGPU] Initialization failed. Using Canvas 2D fallback.");
      return null;
    }
  } catch (err) {
    console.error("[WebGPU] Init error:", err);
    return null;
  }

  // --- Load initial species ---
  const meta = await fetch("data/metadata.json").then(r => r.json());

  async function loadSpecies(species) {
    const info = meta.species[species];
    if (!info) {
      console.error(`[WebGPU] Unknown species: ${species}`);
      return;
    }
    await gpu.loadWindTexture(info.file, info.max_uv, info.max_mag);
    gpu.colorMode = SPECIES_COLOR_MODE[species] ?? 0;
  }

  await loadSpecies("composite");

  // --- Handle resize ---
  const handleResize = () => gpu.resize();
  window.addEventListener("resize", handleResize);
  view.watch("extent", handleResize);

  // --- Start rendering ---
  gpu.start();

  // --- Return controller API ---
  return {
    /** Switch aerosol species */
    async setSpecies(species) {
      await loadSpecies(species);
    },

    /** Update particle count (e.g., 500000) */
    setParticleCount(n) {
      gpu.setParticleCount(n);
    },

    /** Speed multiplier (e.g., 1.0) */
    setSpeed(factor) {
      gpu.speedFactor = factor;
    },

    /** Trail fade opacity (e.g., 0.92) */
    setFadeOpacity(val) {
      gpu.fadeOpacity = val;
    },

    /** Point size in approximate pixels (e.g., 2.0) */
    setPointSize(val) {
      gpu.pointSize = val;
    },

    /** Stop and clean up */
    destroy() {
      window.removeEventListener("resize", handleResize);
      gpu.destroy();
    },

    /** Direct access to the engine */
    engine: gpu,
  };
}
