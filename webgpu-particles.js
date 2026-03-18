// ============================================================================
//  WebGPU Particle System for Aerosol Visualization
//  500,000 particles via compute shader + wind texture advection
// ============================================================================

// ---------------------------------------------------------------------------
//  WGSL Shaders (inline strings)
// ---------------------------------------------------------------------------

const COMPUTE_SHADER = /* wgsl */ `

struct Params {
  deltaT       : f32,   // time step
  speedFactor  : f32,   // user speed multiplier
  maxUV        : f32,   // max wind component (for decoding PNG)
  maxMag       : f32,   // max magnitude (for decoding PNG)
  texW         : f32,   // wind texture width  (360)
  texH         : f32,   // wind texture height (181)
  dropRate     : f32,   // random-reset probability per frame
  dropRateBump : f32,   // extra drop probability in slow wind
  seed         : f32,   // per-frame random seed
  numParticles : u32,   // total particle count
  _pad0        : f32,
  _pad1        : f32,
};

struct Particle {
  lon : f32,     // degrees [-180, 180]
  lat : f32,     // degrees [-85, 85]
  age : f32,     // frames alive
  speed : f32,   // last magnitude (for coloring)
};

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read>       particlesIn  : array<Particle>;
@group(0) @binding(2) var<storage, read_write> particlesOut : array<Particle>;
@group(0) @binding(3) var windTex : texture_2d<f32>;

// Simple hash-based PRNG
fn rand(co : vec2<f32>) -> f32 {
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

fn rand2(id : u32, offset : f32) -> f32 {
  return rand(vec2(f32(id) * 0.001 + offset, params.seed + offset * 7.31));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.numParticles) { return; }

  var p = particlesIn[idx];

  // --- Sample wind at particle position ---
  // lon [-180,180] -> texel x [0, texW)
  // lat [-90, 90]  -> texel y [0, texH)  (top = +90, row 0)
  let tx = i32(((p.lon + 180.0) / 360.0) * params.texW) % i32(params.texW);
  let ty = i32(((90.0 - p.lat) / 180.0) * params.texH);
  let txClamped = clamp(tx, 0, i32(params.texW) - 1);
  let tyClamped = clamp(ty, 0, i32(params.texH) - 1);

  let pixel = textureLoad(windTex, vec2<i32>(txClamped, tyClamped), 0);

  // Decode: R,G are U,V encoded as (val/255)*2-1 * maxUV
  // But textureLoad with rgba8unorm gives [0,1], so:
  let u = (pixel.r * 2.0 - 1.0) * params.maxUV;
  let v = (pixel.g * 2.0 - 1.0) * params.maxUV;
  let mag = pixel.b * params.maxMag;

  // --- Advect ---
  // Match original Canvas 2D behavior: p.x += u*scale, p.y -= v*scale
  // where p.y is latitude. The subtraction accounts for the PNG encoding
  // convention where positive V from GEOS-FP V10M is stored as > 0.5 in
  // the green channel, and the grid row order (row 0 = north pole).
  let scale = params.speedFactor * params.deltaT;
  p.lon = p.lon + u * scale;
  p.lat = p.lat - v * scale;
  p.age = p.age + 1.0;
  p.speed = mag;

  // --- Wrap longitude ---
  if (p.lon > 180.0) { p.lon = p.lon - 360.0; }
  if (p.lon < -180.0) { p.lon = p.lon + 360.0; }

  // --- Random reset (stochastic particle recycling) ---
  let normalizedSpeed = mag / max(params.maxMag, 0.001);
  let dropChance = params.dropRate + params.dropRateBump * (1.0 - normalizedSpeed);
  let shouldDrop = rand2(idx, 1.0) < dropChance;
  let tooOld = p.age > 80.0;
  let outOfBounds = p.lat > 85.0 || p.lat < -85.0;
  let deadWind = mag < 0.05;

  if (shouldDrop || tooOld || outOfBounds || deadWind) {
    // Respawn at random position
    p.lon = rand2(idx, 2.0) * 360.0 - 180.0;
    p.lat = rand2(idx, 3.0) * 150.0 - 75.0;  // [-75, 75] avoids polar extremes
    p.age = 0.0;
    p.speed = 0.0;
  }

  particlesOut[idx] = p;
}
`;


const RENDER_SHADER = /* wgsl */ `

struct Uniforms {
  // Projection: Web Mercator extent -> clip space
  extXmin  : f32,
  extXmax  : f32,
  extYmin  : f32,
  extYmax  : f32,
  maxMag   : f32,
  pointSize : f32,
  colorMode : u32,   // 0=composite, 1=black_carbon, 2=dust, 3=sea_salt
  opacity  : f32,
};

struct Particle {
  lon : f32,
  lat : f32,
  age : f32,
  speed : f32,
};

@group(0) @binding(0) var<uniform> uniforms : Uniforms;
@group(0) @binding(1) var<storage, read> particles : array<Particle>;

struct VertexOutput {
  @builtin(position) position : vec4<f32>,
  @location(0) color : vec4<f32>,
  @location(1) pointCoord : vec2<f32>,
};

const HALF_SIZE : f32 = 20037508.342789244;
const PI : f32 = 3.141592653589793;
const DEG2RAD : f32 = 0.017453292519943295;

fn lonToMerc(lon : f32) -> f32 {
  return lon * HALF_SIZE / 180.0;
}

fn latToMerc(lat : f32) -> f32 {
  let clamped = clamp(lat, -85.0, 85.0);
  let rad = clamped * DEG2RAD;
  return log(tan(PI / 4.0 + rad / 2.0)) * HALF_SIZE / PI;
}

// Color ramps per species (5 stops each)
fn getColor(t : f32, mode : u32) -> vec3<f32> {
  // Composite: purple tones
  // Black carbon: red-orange
  // Dust: amber-brown
  // Sea salt: blue

  var c0 : vec3<f32>; var c1 : vec3<f32>; var c2 : vec3<f32>; var c3 : vec3<f32>; var c4 : vec3<f32>;

  if (mode == 1u) {
    // Black carbon
    c0 = vec3(0.235, 0.059, 0.039);
    c1 = vec3(0.706, 0.157, 0.078);
    c2 = vec3(0.941, 0.314, 0.118);
    c3 = vec3(1.000, 0.627, 0.196);
    c4 = vec3(1.000, 0.902, 0.706);
  } else if (mode == 2u) {
    // Dust
    c0 = vec3(0.196, 0.118, 0.039);
    c1 = vec3(0.549, 0.314, 0.078);
    c2 = vec3(0.824, 0.588, 0.196);
    c3 = vec3(0.980, 0.784, 0.314);
    c4 = vec3(1.000, 0.922, 0.667);
  } else if (mode == 3u) {
    // Sea salt
    c0 = vec3(0.039, 0.078, 0.235);
    c1 = vec3(0.118, 0.314, 0.706);
    c2 = vec3(0.235, 0.627, 0.902);
    c3 = vec3(0.431, 0.863, 0.980);
    c4 = vec3(0.784, 0.961, 1.000);
  } else {
    // Composite (default)
    c0 = vec3(0.118, 0.059, 0.235);
    c1 = vec3(0.392, 0.196, 0.706);
    c2 = vec3(0.627, 0.392, 0.941);
    c3 = vec3(0.784, 0.627, 1.000);
    c4 = vec3(0.941, 0.824, 1.000);
  }

  let idx = t * 4.0;
  let lo = u32(floor(idx));
  let frac = idx - floor(idx);

  if (lo == 0u) { return mix(c0, c1, frac); }
  if (lo == 1u) { return mix(c1, c2, frac); }
  if (lo == 2u) { return mix(c2, c3, frac); }
  return mix(c3, c4, frac);
}

// We render each particle as a quad (2 triangles = 6 vertices).
// vertexIndex 0..5 maps to the quad corners.
@vertex
fn vs_main(
  @builtin(vertex_index) vertexIndex : u32,
  @builtin(instance_index) instanceIndex : u32,
) -> VertexOutput {
  let p = particles[instanceIndex];

  // Project lon/lat -> Web Mercator -> clip space
  let mx = lonToMerc(p.lon);
  let my = latToMerc(p.lat);

  // Map to NDC [-1, 1]
  let ndcX = (mx - uniforms.extXmin) / (uniforms.extXmax - uniforms.extXmin) * 2.0 - 1.0;
  let ndcY = (my - uniforms.extYmin) / (uniforms.extYmax - uniforms.extYmin) * 2.0 - 1.0;

  // Quad offsets (2 triangles forming a quad)
  var quadPos : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
  );

  // --- Particle size scales with concentration ---
  // Low concentration = tiny (0.3x), high = large (1.8x)
  let t = clamp(p.speed / max(uniforms.maxMag * 0.5, 0.001), 0.0, 1.0);
  let sizeMult = 0.3 + t * 1.5; // range [0.3, 1.8]
  let offset = quadPos[vertexIndex] * uniforms.pointSize * 0.001 * sizeMult;

  var out : VertexOutput;
  out.position = vec4(ndcX + offset.x, ndcY + offset.y, 0.0, 1.0);
  out.pointCoord = quadPos[vertexIndex] * 0.5 + 0.5;

  let baseColor = getColor(t, uniforms.colorMode);

  // Age-based alpha fade
  let ageFade = 1.0 - smoothstep(60.0, 80.0, p.age);
  let alpha = (0.4 + t * 0.6) * ageFade * uniforms.opacity;

  out.color = vec4(baseColor, alpha);
  return out;
}

@fragment
fn fs_main(in : VertexOutput) -> @location(0) vec4<f32> {
  // Circular glow SDF
  let center = in.pointCoord - vec2(0.5);
  let dist = length(center) * 2.0;  // 0 at center, 1 at edge

  if (dist > 1.0) { discard; }

  // Soft glow falloff
  let glow = 1.0 - smoothstep(0.0, 1.0, dist);
  let glowIntensity = glow * glow;  // quadratic falloff for a nice glow

  return vec4(in.color.rgb * glowIntensity, in.color.a * glowIntensity);
}
`;


const FULLSCREEN_QUAD_SHADER = /* wgsl */ `

// Full-screen triangle (3 vertices, no vertex buffer needed)
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0),
  );
  return vec4(pos[vertexIndex], 0.0, 1.0);
}

@group(0) @binding(0) var fadeTex : texture_2d<f32>;
@group(0) @binding(1) var fadeSampler : sampler;

struct FadeParams {
  fadeOpacity : f32,
};
@group(0) @binding(2) var<uniform> fadeParams : FadeParams;

@fragment
fn fs_main(@builtin(position) fragCoord : vec4<f32>) -> @location(0) vec4<f32> {
  let texSize = vec2<f32>(textureDimensions(fadeTex));
  let uv = fragCoord.xy / texSize;
  let color = textureSampleLevel(fadeTex, fadeSampler, uv, 0.0);
  return vec4(color.rgb, color.a * fadeParams.fadeOpacity);
}
`;


const BLOOM_SHADER = /* wgsl */ `

// Gaussian blur for bloom post-processing
// Two passes: horizontal then vertical (separable filter)

@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var inputSampler : sampler;

struct BloomParams {
  direction : vec2<f32>,  // (1,0) for horizontal, (0,1) for vertical
  intensity : f32,
  threshold : f32,
};
@group(0) @binding(2) var<uniform> bloomParams : BloomParams;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> @builtin(position) vec4<f32> {
  var pos = array<vec2<f32>, 3>(
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0),
  );
  return vec4(pos[vertexIndex], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) fragCoord : vec4<f32>) -> @location(0) vec4<f32> {
  let texSize = vec2<f32>(textureDimensions(inputTex));
  let uv = fragCoord.xy / texSize;
  let pixel = 1.0 / texSize;

  // 9-tap Gaussian weights
  let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

  // Sample center
  var result = textureSampleLevel(inputTex, inputSampler, uv, 0.0) * weights[0];

  // Sample neighbors along blur direction
  let dir = bloomParams.direction * pixel;
  for (var i = 1; i < 5; i++) {
    let off = dir * f32(i) * 2.0; // spread the blur wider
    result += textureSampleLevel(inputTex, inputSampler, uv + off, 0.0) * weights[i];
    result += textureSampleLevel(inputTex, inputSampler, uv - off, 0.0) * weights[i];
  }

  // Apply intensity and threshold (only bloom bright areas)
  let brightness = dot(result.rgb, vec3(0.2126, 0.7152, 0.0722));
  let factor = smoothstep(bloomParams.threshold, bloomParams.threshold + 0.1, brightness);
  return vec4(result.rgb * bloomParams.intensity * factor, result.a);
}
`;


// ---------------------------------------------------------------------------
//  WebGPU Particle Engine Class
// ---------------------------------------------------------------------------

export class WebGPUParticleSystem {

  constructor(canvas, arcgisView) {
    this.canvas = canvas;
    this.view = arcgisView;
    this.device = null;
    this.context = null;

    // Configuration
    this.numParticles = 500000;
    this.speedFactor = 1.0;
    this.fadeOpacity = 0.92;
    this.pointSize = 2.0;
    this.colorMode = 0;       // 0=composite, 1=black_carbon, 2=dust, 3=sea_salt
    this.maxUV = 21.0;
    this.maxMag = 22.0;

    // Internal state
    this._ready = false;
    this._frame = 0;
    this._pingPong = 0;
    this._animId = null;
  }

  // =========================================================================
  //  1. INITIALIZATION
  // =========================================================================

  async init() {
    // --- Feature detection ---
    if (!navigator.gpu) {
      console.warn("WebGPU not supported in this browser. Falling back to Canvas 2D.");
      return false;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.warn("WebGPU adapter not available. Falling back to Canvas 2D.");
      return false;
    }

    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      }
    });

    // Handle device loss
    this.device.lost.then((info) => {
      console.error(`WebGPU device lost: ${info.message}`);
      if (info.reason !== "destroyed") {
        // Attempt re-init
        this.init();
      }
    });

    // --- Configure canvas ---
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) {
      console.warn("Could not get WebGPU context from canvas.");
      return false;
    }

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    this._configureCanvas();

    // --- Create pipelines ---
    this._createComputePipeline();
    this._createRenderPipeline();
    this._createFadePipeline();
    this._createBloomPipeline();

    // --- Create particle buffers ---
    this._createParticleBuffers();

    // --- Create uniform buffers ---
    this._createUniformBuffers();

    // --- Create ping-pong textures for trails ---
    this._createTrailTextures();

    // --- Create bloom textures ---
    this._createBloomTextures();

    this._ready = true;
    console.log(`WebGPU particle system initialized: ${this.numParticles} particles`);
    return true;
  }

  // =========================================================================
  //  2. LOAD WIND TEXTURE FROM PNG
  // =========================================================================

  async loadWindTexture(imageUrl, maxUV, maxMag) {
    this.maxUV = maxUV;
    this.maxMag = maxMag;

    // Load image as ImageBitmap (preserves raw pixel values)
    const response = await fetch(imageUrl);
    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });

    // Create GPU texture
    if (this.windTexture) {
      this.windTexture.destroy();
    }

    this.windTexture = this.device.createTexture({
      label: "wind-data-texture",
      size: [bitmap.width, bitmap.height],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING |
             GPUTextureUsage.COPY_DST |
             GPUTextureUsage.RENDER_ATTACHMENT,
    });

    // Copy image data to GPU texture
    this.device.queue.copyExternalImageToTexture(
      { source: bitmap, flipY: false },  // PNG row 0 = lat +90 (north), no flip needed
      { texture: this.windTexture },
      { width: bitmap.width, height: bitmap.height },
    );

    this.texWidth = bitmap.width;
    this.texHeight = bitmap.height;

    // Rebuild compute bind groups (they reference the wind texture)
    this._createComputeBindGroups();

    console.log(`Wind texture loaded: ${bitmap.width}x${bitmap.height}`);
  }

  // =========================================================================
  //  3. INTERNAL: PIPELINE CREATION
  // =========================================================================

  _configureCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = Math.floor(rect.width * dpr);
    this.canvas.height = Math.floor(rect.height * dpr);

    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: "premultiplied",
    });
  }

  _createComputePipeline() {
    const module = this.device.createShaderModule({
      label: "compute-particle-advection",
      code: COMPUTE_SHADER,
    });

    this.computePipeline = this.device.createComputePipeline({
      label: "particle-compute-pipeline",
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  _createRenderPipeline() {
    const module = this.device.createShaderModule({
      label: "particle-render-shader",
      code: RENDER_SHADER,
    });

    this.renderPipeline = this.device.createRenderPipeline({
      label: "particle-render-pipeline",
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vs_main",
        // No vertex buffers - we use instance_index to read from storage buffer
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{
          format: "rgba8unorm",  // Render to offscreen texture (not swapchain)
          blend: {
            // Additive blending: result = src + dst
            color: {
              operation: "add",
              srcFactor: "src-alpha",
              dstFactor: "one",
            },
            alpha: {
              operation: "add",
              srcFactor: "one",
              dstFactor: "one",
            },
          },
        }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });
  }

  _createFadePipeline() {
    const module = this.device.createShaderModule({
      label: "fade-fullscreen-shader",
      code: FULLSCREEN_QUAD_SHADER,
    });

    this.fadePipeline = this.device.createRenderPipeline({
      label: "fade-pipeline",
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vs_main",
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{
          format: "rgba8unorm",
          blend: {
            color: {
              operation: "add",
              srcFactor: "one",
              dstFactor: "zero",  // Replace mode for fade copy
            },
            alpha: {
              operation: "add",
              srcFactor: "one",
              dstFactor: "zero",
            },
          },
        }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

    // Fade pipeline that writes to the swapchain (presentationFormat)
    this.fadeToScreenPipeline = this.device.createRenderPipeline({
      label: "fade-to-screen-pipeline",
      layout: "auto",
      vertex: {
        module,
        entryPoint: "vs_main",
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{
          format: this.presentationFormat,
          blend: {
            color: {
              operation: "add",
              srcFactor: "one",
              dstFactor: "zero",
            },
            alpha: {
              operation: "add",
              srcFactor: "one",
              dstFactor: "zero",
            },
          },
        }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });
  }

  // =========================================================================
  //  4. INTERNAL: BUFFER CREATION
  // =========================================================================

  _createParticleBuffers() {
    const stride = 4 * Float32Array.BYTES_PER_ELEMENT;  // lon, lat, age, speed
    const size = this.numParticles * stride;

    // Initialize with random positions
    const data = new Float32Array(this.numParticles * 4);
    for (let i = 0; i < this.numParticles; i++) {
      data[i * 4 + 0] = Math.random() * 360 - 180;        // lon
      data[i * 4 + 1] = Math.random() * 150 - 75;          // lat [-75, 75]
      data[i * 4 + 2] = Math.floor(Math.random() * 80);    // age
      data[i * 4 + 3] = 0;                                  // speed (will be set by compute)
    }

    // Double-buffer for compute ping-pong
    this.particleBuffers = [null, null];
    for (let i = 0; i < 2; i++) {
      this.particleBuffers[i] = this.device.createBuffer({
        label: `particle-buffer-${i}`,
        size: size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(this.particleBuffers[i].getMappedRange()).set(data);
      this.particleBuffers[i].unmap();
    }
  }

  _createUniformBuffers() {
    // Compute uniforms: 12 floats (48 bytes), aligned to 16
    this.computeUniformBuffer = this.device.createBuffer({
      label: "compute-uniforms",
      size: 48,  // 12 * 4 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Render uniforms: 8 values (32 bytes)
    this.renderUniformBuffer = this.device.createBuffer({
      label: "render-uniforms",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Fade uniforms: 1 float (4 bytes, padded to 16)
    this.fadeUniformBuffer = this.device.createBuffer({
      label: "fade-uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  _createTrailTextures() {
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Two offscreen textures for ping-pong trail effect
    this.trailTextures = [null, null];
    this.trailViews = [null, null];

    for (let i = 0; i < 2; i++) {
      this.trailTextures[i] = this.device.createTexture({
        label: `trail-texture-${i}`,
        size: [w, h],
        format: "rgba8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT |
               GPUTextureUsage.TEXTURE_BINDING,
      });
      this.trailViews[i] = this.trailTextures[i].createView();
    }

    this._trailWidth = w;
    this._trailHeight = h;

    // Create sampler for reading trail textures
    this.trailSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    // Create fade bind groups for both directions
    this._createFadeBindGroups();

    // Recreate bloom textures if they exist
    if (this.bloomPipeline) {
      this._createBloomTextures();
    }
  }

  _createBloomPipeline() {
    const module = this.device.createShaderModule({
      label: "bloom-shader",
      code: BLOOM_SHADER,
    });

    this.bloomPipeline = this.device.createRenderPipeline({
      label: "bloom-pipeline",
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [{
          format: "rgba8unorm",
          blend: {
            color: { operation: "add", srcFactor: "one", dstFactor: "one" }, // additive
            alpha: { operation: "add", srcFactor: "one", dstFactor: "one" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });
  }

  _createBloomTextures() {
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Bloom at half resolution for performance
    const bw = Math.floor(w / 2);
    const bh = Math.floor(h / 2);

    this.bloomTex = this.device.createTexture({
      label: "bloom-intermediate",
      size: [bw, bh],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.bloomView = this.bloomTex.createView();

    this.bloomSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });

    // Uniform buffer for bloom params (direction, intensity, threshold)
    this.bloomUniformBuffer = this.device.createBuffer({
      label: "bloom-uniforms",
      size: 16, // vec2 direction + f32 intensity + f32 threshold
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this._bloomWidth = bw;
    this._bloomHeight = bh;
  }

  _createComputeBindGroups() {
    if (!this.windTexture) return;

    this.computeBindGroups = [null, null];
    for (let i = 0; i < 2; i++) {
      this.computeBindGroups[i] = this.device.createBindGroup({
        label: `compute-bind-group-${i}`,
        layout: this.computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.computeUniformBuffer } },
          { binding: 1, resource: { buffer: this.particleBuffers[i] } },
          { binding: 2, resource: { buffer: this.particleBuffers[1 - i] } },
          { binding: 3, resource: this.windTexture.createView() },
        ],
      });
    }
  }

  _createRenderBindGroups() {
    this.renderBindGroups = [null, null];
    for (let i = 0; i < 2; i++) {
      this.renderBindGroups[i] = this.device.createBindGroup({
        label: `render-bind-group-${i}`,
        layout: this.renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.renderUniformBuffer } },
          { binding: 1, resource: { buffer: this.particleBuffers[i] } },
        ],
      });
    }
  }

  _createFadeBindGroups() {
    // Bind group for fading trail texture 0 -> writing to trail texture 1
    this.fadeBindGroups = [null, null];
    for (let i = 0; i < 2; i++) {
      this.fadeBindGroups[i] = this.device.createBindGroup({
        label: `fade-bind-group-${i}`,
        layout: this.fadePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.trailViews[i] },
          { binding: 1, resource: this.trailSampler },
          { binding: 2, resource: { buffer: this.fadeUniformBuffer } },
        ],
      });
    }

    // Also create bind groups for fadeToScreenPipeline (same layout)
    this.fadeToScreenBindGroups = [null, null];
    for (let i = 0; i < 2; i++) {
      this.fadeToScreenBindGroups[i] = this.device.createBindGroup({
        label: `fade-to-screen-bind-group-${i}`,
        layout: this.fadeToScreenPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.trailViews[i] },
          { binding: 1, resource: this.trailSampler },
          { binding: 2, resource: { buffer: this.fadeUniformBuffer } },
        ],
      });
    }
  }

  // =========================================================================
  //  5. RESIZE HANDLING
  // =========================================================================

  resize() {
    if (!this._ready) return;

    this._configureCanvas();

    // Recreate trail textures at new size
    for (let i = 0; i < 2; i++) {
      if (this.trailTextures[i]) this.trailTextures[i].destroy();
    }
    this._createTrailTextures();
  }

  // =========================================================================
  //  6. UPDATE PARTICLE COUNT
  // =========================================================================

  setParticleCount(count) {
    if (!this._ready) return;
    this.numParticles = count;

    // Destroy old buffers
    for (let i = 0; i < 2; i++) {
      if (this.particleBuffers[i]) this.particleBuffers[i].destroy();
    }

    this._createParticleBuffers();
    if (this.windTexture) {
      this._createComputeBindGroups();
    }
    this._createRenderBindGroups();
  }

  // =========================================================================
  //  7. FRAME RENDERING
  // =========================================================================

  frame() {
    if (!this._ready || !this.windTexture || !this.view.extent) return;

    const ext = this.view.extent;
    const readIdx = this._pingPong;
    const writeIdx = 1 - this._pingPong;

    // Ensure render bind groups exist
    if (!this.renderBindGroups || !this.renderBindGroups[0]) {
      this._createRenderBindGroups();
    }

    // --- Update compute uniforms ---
    const computeData = new ArrayBuffer(48);
    const cf = new Float32Array(computeData);
    const cu = new Uint32Array(computeData);
    cf[0] = 0.03;                           // deltaT
    cf[1] = this.speedFactor;               // speedFactor
    cf[2] = this.maxUV;                     // maxUV
    cf[3] = this.maxMag;                    // maxMag
    cf[4] = this.texWidth || 360;           // texW
    cf[5] = this.texHeight || 181;          // texH
    cf[6] = 0.003;                          // dropRate
    cf[7] = 0.01;                           // dropRateBump
    cf[8] = Math.random();                  // seed
    cu[9] = this.numParticles;              // numParticles (u32)
    cf[10] = 0;                             // pad
    cf[11] = 0;                             // pad
    this.device.queue.writeBuffer(this.computeUniformBuffer, 0, computeData);

    // --- Update render uniforms ---
    const renderData = new ArrayBuffer(32);
    const rf = new Float32Array(renderData);
    const ru = new Uint32Array(renderData);
    rf[0] = ext.xmin;
    rf[1] = ext.xmax;
    rf[2] = ext.ymin;
    rf[3] = ext.ymax;
    rf[4] = this.maxMag;
    rf[5] = this.pointSize;
    ru[6] = this.colorMode;
    rf[7] = 1.0;  // opacity
    this.device.queue.writeBuffer(this.renderUniformBuffer, 0, renderData);

    // --- Update fade uniforms ---
    this.device.queue.writeBuffer(
      this.fadeUniformBuffer, 0,
      new Float32Array([this.fadeOpacity])
    );

    // --- Check if trail textures need resize ---
    if (this._trailWidth !== this.canvas.width || this._trailHeight !== this.canvas.height) {
      this.resize();
      return;  // Skip this frame during resize
    }

    // --- Build command buffer ---
    const encoder = this.device.createCommandEncoder();

    // PASS 1: Compute - advect particles
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.computePipeline);
      pass.setBindGroup(0, this.computeBindGroups[readIdx]);
      pass.dispatchWorkgroups(Math.ceil(this.numParticles / 64));
      pass.end();
    }

    // PASS 2: Fade - copy old trail with reduced opacity to the write trail texture
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.trailViews[writeIdx],
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(this.fadePipeline);
      pass.setBindGroup(0, this.fadeBindGroups[readIdx]);
      pass.draw(3);  // Full-screen triangle
      pass.end();
    }

    // PASS 3: Render particles on top of faded trail (into write trail texture)
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.trailViews[writeIdx],
          loadOp: "load",      // Don't clear - draw on top of faded trail
          storeOp: "store",
        }],
      });
      pass.setPipeline(this.renderPipeline);
      pass.setBindGroup(0, this.renderBindGroups[writeIdx]);
      pass.draw(6, this.numParticles);  // 6 verts per quad, N instances
      pass.end();
    }

    // PASS 4: Bloom - blur bright areas and composite back
    if (this.bloomTex && this.bloomPipeline) {
      // 4a: Horizontal blur from trail → bloom texture
      this.device.queue.writeBuffer(this.bloomUniformBuffer, 0, new Float32Array([
        1.0, 0.0,  // direction: horizontal
        1.2,       // intensity
        0.08,      // threshold
      ]));

      const bloomBG_H = this.device.createBindGroup({
        layout: this.bloomPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.trailViews[writeIdx] },
          { binding: 1, resource: this.bloomSampler },
          { binding: 2, resource: { buffer: this.bloomUniformBuffer } },
        ],
      });

      const bloomPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.bloomView,
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      bloomPass.setPipeline(this.bloomPipeline);
      bloomPass.setBindGroup(0, bloomBG_H);
      bloomPass.draw(3);
      bloomPass.end();

      // 4b: Vertical blur from bloom → back onto trail (additive)
      this.device.queue.writeBuffer(this.bloomUniformBuffer, 0, new Float32Array([
        0.0, 1.0,  // direction: vertical
        1.2,       // intensity
        0.08,      // threshold
      ]));

      const bloomBG_V = this.device.createBindGroup({
        layout: this.bloomPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.bloomView },
          { binding: 1, resource: this.bloomSampler },
          { binding: 2, resource: { buffer: this.bloomUniformBuffer } },
        ],
      });

      const vBloomPass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.trailViews[writeIdx],
          loadOp: "load", // additive on top of existing trail
          storeOp: "store",
        }],
      });
      vBloomPass.setPipeline(this.bloomPipeline);
      vBloomPass.setBindGroup(0, bloomBG_V);
      vBloomPass.draw(3);
      vBloomPass.end();
    }

    // PASS 5: Copy result trail to the swapchain (screen)
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{
          view: this.context.getCurrentTexture().createView(),
          clearValue: [0, 0, 0, 0],
          loadOp: "clear",
          storeOp: "store",
        }],
      });
      pass.setPipeline(this.fadeToScreenPipeline);
      pass.setBindGroup(0, this.fadeToScreenBindGroups[writeIdx]);
      pass.draw(3);
      pass.end();
    }

    this.device.queue.submit([encoder.finish()]);

    // Flip ping-pong
    this._pingPong = writeIdx;
    this._frame++;
  }

  // =========================================================================
  //  8. ANIMATION LOOP
  // =========================================================================

  start() {
    const loop = () => {
      this.frame();
      this._animId = requestAnimationFrame(loop);
    };
    this._animId = requestAnimationFrame(loop);
  }

  stop() {
    if (this._animId !== null) {
      cancelAnimationFrame(this._animId);
      this._animId = null;
    }
  }

  // =========================================================================
  //  9. CLEANUP
  // =========================================================================

  destroy() {
    this.stop();
    if (this.device) {
      this.device.destroy();
      this.device = null;
    }
    this._ready = false;
  }
}


// ---------------------------------------------------------------------------
//  COLOR MODE MAPPING (matches species names to shader colorMode uniform)
// ---------------------------------------------------------------------------
export const SPECIES_COLOR_MODE = {
  composite: 0,
  black_carbon: 1,
  dust: 2,
  sea_salt: 3,
};
