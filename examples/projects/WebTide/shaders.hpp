#ifndef WEBTIDE_SHADERS_HPP
#define WEBTIDE_SHADERS_HPP

namespace webtide {

// ============================================================================
// Compute Shaders
// ============================================================================

constexpr const char* phillipsSpectrumWGSL = R"(
const PI: f32 = 3.1415926;

@group(0) @binding(0) var H0: texture_storage_2d<rgba32float, write>;
@group(0) @binding(1) var Noise: texture_2d<f32>;

struct Params {
    textureSize: u32,
    tileSize: f32,
    windTheta: f32,
    windSpeed: f32,
    smallWaveLengthCutoff: f32,
    kMin: f32,
    kMax: f32,
    _pad: f32,
};

@group(0) @binding(2) var<uniform> params: Params;

fn phillipsSpectrum2D(k: vec2<f32>) -> f32 {
    if(length(k) < 0.0001) {
        return 0.0;
    }

    let A: f32 = 1.0;
    let windDir = vec2<f32>(cos(params.windTheta), sin(params.windTheta));
    let g: f32 = 9.81;
    let L: f32 = params.windSpeed * params.windSpeed / g;
    let k2: f32 = dot(k, k);
    let kL2: f32 = k2 * L * L;
    let k4: f32 = k2 * k2;
    var kw2: f32 = dot(normalize(k), normalize(windDir));
    kw2 *= kw2;

    let l: f32 = params.smallWaveLengthCutoff;
    let cutoff: f32 = exp(-k2 * l * l);

    return A * exp(-1.0 / kL2) * kw2 * cutoff / k4;
}

@compute @workgroup_size(8,8,1)
fn computeSpectrum(@builtin(global_invocation_id) id: vec3<u32>) {
    let deltaK = 2.0 * PI / params.tileSize;
    let nx = f32(id.x) - f32(params.textureSize) / 2.0;
    let nz = f32(id.y) - f32(params.textureSize) / 2.0;
    let k = vec2<f32>(nx, nz) * deltaK;

    // Band-pass filter: reject wavenumbers outside [kMin, kMax]
    let kLen = length(k);
    if (kLen < params.kMin || (params.kMax > 0.0 && kLen > params.kMax)) {
        textureStore(H0, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let noise_k = textureLoad(Noise, vec2<i32>(id.xy), 0).xy;
    let h0_k = noise_k * sqrt(phillipsSpectrum2D(k) / 2.0);

    let noise_minus_k = textureLoad(Noise, vec2<i32>(params.textureSize - id.xy), 0).xy;
    let h0_minus_k = noise_minus_k * sqrt(phillipsSpectrum2D(-k) / 2.0);
    let h0_minus_k_conj = vec2<f32>(h0_minus_k.x, -h0_minus_k.y);

    textureStore(H0, vec2<i32>(id.xy), vec4<f32>(h0_k, h0_minus_k_conj));
}
)";

constexpr const char* dynamicSpectrumWGSL = R"(
const PI: f32 = 3.1415926;

@group(0) @binding(0) var H0: texture_2d<f32>;
@group(0) @binding(1) var HT: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var DHT: texture_storage_2d<rg32float, write>;
@group(0) @binding(3) var Displacement: texture_storage_2d<rg32float, write>;

struct Params {
    textureSize: u32,
    tileSize: f32,
    elapsedSeconds: f32,
};

@group(0) @binding(4) var<uniform> params: Params;

@group(0) @binding(5) var JacDiag: texture_storage_2d<rg32float, write>;

fn omega(k: vec2<f32>) -> f32 {
    return sqrt(length(k) * 9.81);
}

fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.r * b.r - a.g * b.g, a.r * b.g + a.g * b.r);
}

@compute @workgroup_size(8,8,1)
fn computeSpectrum(@builtin(global_invocation_id) id: vec3<u32>) {
    let iid = vec3<i32>(id);

    let deltaK = 2.0 * PI / params.tileSize;
    let n = f32(id.x) - f32(params.textureSize) / 2.0;
    let m = f32(id.y) - f32(params.textureSize) / 2.0;
    let k = vec2<f32>(n, m) * deltaK;

    let theta = params.elapsedSeconds * omega(k);
    let exponent = vec2<f32>(cos(theta), sin(theta));
    let h0: vec4<f32> = textureLoad(H0, iid.xy, 0);

    let h = complexMult(h0.xy, exponent) + complexMult(h0.zw, vec2<f32>(exponent.x, -exponent.y));

    let ih = vec2<f32>(-h.y, h.x);

    let ikh = complexMult(k, ih);

    let displacement = ikh / (length(k) + 0.001);

    // Packed Jacobian diagonal spectrum: .r = Jxx spectrum, .g = Jzz spectrum
    // Jxx = kx^2/|k| * H, Jzz = kz^2/|k| * H (packed using real/imag trick)
    // IFFT of this gives (.r=Jxx_spatial, .g=Jzz_spatial)
    let klen = length(k) + 0.001;
    let jac_r = (k.x*k.x * h.x - k.y*k.y * h.y) / klen;
    let jac_i = (k.x*k.x * h.y + k.y*k.y * h.x) / klen;
    textureStore(JacDiag, iid.xy, vec4<f32>(jac_r, jac_i, 0.0, 0.0));

    textureStore(HT, iid.xy, vec4<f32>(h, vec2(0.0)));
    textureStore(DHT, iid.xy, vec4<f32>(ikh, vec2(0.0)));
    textureStore(Displacement, iid.xy, vec4<f32>(displacement, vec2(0.0)));
}
)";

constexpr const char* twiddleFactorsWGSL = R"(
const PI: f32 = 3.1415926;

@group(0) @binding(0) var PrecomputeBuffer: texture_storage_2d<rgba32float, write>;

struct Params {
    step: i32,
    textureSize: i32,
};

@group(0) @binding(1) var<uniform> params: Params;

fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32>
{
    return vec2<f32>(a.r * b.r - a.g * b.g, a.r * b.g + a.g * b.r);
}

fn complexExp(a: vec2<f32>) -> vec2<f32>
{
    return vec2<f32>(cos(a.y), sin(a.y)) * exp(a.x);
}

@compute @workgroup_size(1,8,1)
fn precomputeTwiddleFactorsAndInputIndices(@builtin(global_invocation_id) id: vec3<u32>)
{
    let iid = vec3<i32>(id);
    let b = params.textureSize >> (id.x + 1u);
    let mult = 2.0 * PI * vec2<f32>(0.0, -1.0) / f32(params.textureSize);
    let i = (2 * b * (iid.y / b) + (iid.y % b)) % params.textureSize;
    let twiddle = complexExp(mult * vec2<f32>(f32((iid.y / b) * b)));

    textureStore(PrecomputeBuffer, iid.xy, vec4<f32>(twiddle.x, twiddle.y, f32(i), f32(i + b)));
    textureStore(PrecomputeBuffer, vec2<i32>(iid.x, iid.y + params.textureSize / 2), vec4<f32>(-twiddle.x, -twiddle.y, f32(i), f32(i + b)));
}
)";

constexpr const char* horizontalStepIfftWGSL = R"(
struct Params {
    step: i32,
    textureSize: i32,
};

@group(0) @binding(0) var<uniform> params: Params;

@group(0) @binding(1) var PrecomputedData: texture_2d<f32>;

@group(0) @binding(2) var InputBuffer: texture_2d<f32>;
@group(0) @binding(3) var OutputBuffer: texture_storage_2d<rg32float, write>;

fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.r * b.r - a.g * b.g, a.r * b.g + a.g * b.r);
}

@compute @workgroup_size(8,8,1)
fn horizontalStepInverseFFT(@builtin(global_invocation_id) id: vec3<u32>) {
    let iid = vec3<i32>(id);
    let data = textureLoad(PrecomputedData, vec2<i32>(params.step, iid.x), 0);
    let inputsIndices = vec2<i32>(data.ba);

    let input0 = textureLoad(InputBuffer, vec2<i32>(inputsIndices.x, iid.y), 0);
    let input1 = textureLoad(InputBuffer, vec2<i32>(inputsIndices.y, iid.y), 0);

    textureStore(OutputBuffer, iid.xy, vec4<f32>(
        input0.xy + complexMult(vec2<f32>(data.r, -data.g), input1.xy), 0.0, 0.0
    ));
}
)";

constexpr const char* verticalStepIfftWGSL = R"(
struct Params {
    step: i32,
    textureSize: i32,
};

@group(0) @binding(0) var<uniform> params: Params;

@group(0) @binding(1) var PrecomputedData: texture_2d<f32>;

@group(0) @binding(2) var InputBuffer: texture_2d<f32>;
@group(0) @binding(3) var OutputBuffer: texture_storage_2d<rg32float, write>;

fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.r * b.r - a.g * b.g, a.r * b.g + a.g * b.r);
}

@compute @workgroup_size(8,8,1)
fn verticalStepInverseFFT(@builtin(global_invocation_id) id: vec3<u32>) {
    let iid = vec3<i32>(id);
    let data = textureLoad(PrecomputedData, vec2<i32>(params.step, iid.y), 0);
    let inputsIndices = vec2<i32>(data.ba);

    let input0 = textureLoad(InputBuffer, vec2<i32>(iid.x, inputsIndices.x), 0);
    let input1 = textureLoad(InputBuffer, vec2<i32>(iid.x, inputsIndices.y), 0);

    textureStore(OutputBuffer, iid.xy, vec4<f32>(
        input0.xy + complexMult(vec2<f32>(data.r, -data.g), input1.xy), 0.0, 0.0
    ));
}
)";

constexpr const char* permutationWGSL = R"(
@group(0) @binding(0) var InputBuffer: texture_2d<f32>;
@group(0) @binding(1) var OutputBuffer: texture_storage_2d<rg32float, write>;

@compute @workgroup_size(8,8,1)
fn permute(@builtin(global_invocation_id) id: vec3<u32>) {
    let iid = vec3<i32>(id);
    let input = textureLoad(InputBuffer, iid.xy, 0);

    textureStore(OutputBuffer, iid.xy, input * (1.0 - 2.0 * f32((iid.x + iid.y) % 2)));
}
)";

constexpr const char* copyTextureWGSL = R"(
@group(0) @binding(0) var dest: texture_storage_2d<rg32float, write>;
@group(0) @binding(1) var src: texture_2d<f32>;

struct Params {
    width: u32,
    height: u32,
};
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.width || global_id.y >= params.height) {
        return;
    }
    let pixel: vec4<f32> = textureLoad(src, vec2<i32>(global_id.xy), 0);
    textureStore(dest, vec2<i32>(global_id.xy), pixel);
}
)";

// Foam update: combined 3-cascade Jacobian whitecap detection with temporal decay.
// J_total = (1 + lambda*(Jxx0+Jxx1+Jxx2)) * (1 + lambda*(Jzz0+Jzz1+Jzz2))
// Real breaking happens when the combined displacement field folds, not just one scale.
// Each cascade's raw IFFT Jacobian is normalised by its own jacScale before summing.
// Foam persists with exponential decay: foam = max(newFoam, prevFoam - decay*dt)
constexpr const char* foamUpdateWGSL = R"(
@group(0) @binding(0) var t_jacDiag0: texture_2d<f32>;  // cascade 0 .r=Jxx .g=Jzz (spatial)
@group(0) @binding(1) var t_jacDiag1: texture_2d<f32>;  // cascade 1 .r=Jxx .g=Jzz (spatial)
@group(0) @binding(2) var t_jacDiag2: texture_2d<f32>;  // cascade 2 .r=Jxx .g=Jzz (spatial)
@group(0) @binding(3) var t_foamIn:   texture_2d<f32>;  // previous frame foam [0,1]
@group(0) @binding(4) var t_foamOut:  texture_storage_2d<rgba16float, write>;

struct FoamParams {
    lambda:    f32,   // choppiness multiplier
    decay:     f32,   // foam fade per second
    dt:        f32,   // delta time in seconds
    jacScale0: f32,   // 1/(2*C0_TILE) — normalise cascade-0 IFFT Jacobian
    jacScale1: f32,   // 1/(2*C1_TILE) — normalise cascade-1 IFFT Jacobian
    jacScale2: f32,   // 1/(2*C2_TILE) — normalise cascade-2 IFFT Jacobian
    _pad0:     f32,
    _pad1:     f32,
};
@group(0) @binding(5) var<uniform> params: FoamParams;

@compute @workgroup_size(8,8,1)
fn updateFoam(@builtin(global_invocation_id) id: vec3<u32>) {
    let coord = vec2<i32>(id.xy);
    // Normalise each cascade's raw IFFT Jacobian then sum for the combined field.
    let jac0 = textureLoad(t_jacDiag0, coord, 0).xy * params.jacScale0;
    let jac1 = textureLoad(t_jacDiag1, coord, 0).xy * params.jacScale1;
    let jac2 = textureLoad(t_jacDiag2, coord, 0).xy * params.jacScale2;
    let jacSum = jac0 + jac1 + jac2;
    // Combined Jacobian — negative means the surface is folding (breaking wave).
    let J        = (1.0 + params.lambda * jacSum.x) * (1.0 + params.lambda * jacSum.y);
    let newFoam  = select(0.0, 1.0, J < 0.0);
    let prevFoam = textureLoad(t_foamIn, coord, 0).r;
    let foam     = max(newFoam, prevFoam - params.decay * params.dt);
    textureStore(t_foamOut, coord, vec4<f32>(foam, 0.0, 0.0, 0.0));
}
)";

// ============================================================================
// Water Rendering Shaders (WGSL, ported from GLSL)
// ============================================================================

// Vertex shader: displaces water mesh using compute-generated maps from 3 cascades.
// Normal computation is done per-pixel in the fragment shader for smooth results.
constexpr const char* waterVertexWGSL = R"(

struct TransformUniforms {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    normalCol0: vec4<f32>,
    normalCol1: vec4<f32>,
    normalCol2: vec4<f32>,
    cameraPos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> transform: TransformUniforms;

struct LightData {
    numDir: u32, numPoint: u32, numSpot: u32, numHemi: u32,
    ambient: vec3<f32>, _pad: f32,
    // Phase 1: useLegacyLights flag + 12 bytes pad inserted into the lights
    // uniform header. dirDirection0 must stay at byte offset 48.
    useLegacyLights: u32, _pdUL0: u32, _pdUL1: u32, _pdUL2: u32,
    dirDirection0: vec3<f32>, _pd0: f32,
    dirColor0: vec3<f32>, _pd1: f32,
};
@group(0) @binding(1) var<uniform> lights: LightData;

struct OceanUniforms {
    choppiness:    vec4<f32>,
    contactObj0:   vec4<f32>,   // xy = world XZ centre, z = radius (0 = inactive)
    contactObj1:   vec4<f32>,
    contactObj2:   vec4<f32>,
    contactObj3:   vec4<f32>,
    detailStrength:vec4<f32>,   // x = detail micro-ripple strength (fades with distance)
    foamStrength:  vec4<f32>,
    foamThreshold: vec4<f32>,
    fogDensity:    vec4<f32>,
    fresnelScale:  vec4<f32>,
    mieCoeff:      vec4<f32>,   // x = mie scattering coefficient
    mieG:          vec4<f32>,   // x = mie directional G
    normalStrength:vec4<f32>,
    rayleigh:      vec4<f32>,   // x = rayleigh coefficient
    seaColor:      vec4<f32>,
    specShininess: vec4<f32>,
    tileSize:      vec4<f32>,
    turbidity:     vec4<f32>,   // x = atmospheric turbidity
    waveScale:     vec4<f32>,
    wireframe:     vec4<f32>,   // x > 0.5 = wireframe debug mode
};
@group(0) @binding(2) var<uniform> ocean: OceanUniforms;

// Displacement + height textures (RG32Float — loaded via textureLoad, no sampler needed).
// cascade0Displacement (3), cascade0Height (7)
// cascade1Displacement (9), cascade1Height (13)
// cascade2Displacement (15), cascade2Height (19)
// (sampler slots 4,8,10,14,16,20 left unused; gradient textures 5,6,11,12,17,18 in fragment)
@group(0) @binding(3)  var t_c0Disp:   texture_2d<f32>;
@group(0) @binding(7)  var t_c0Height: texture_2d<f32>;
@group(0) @binding(9)  var t_c1Disp:   texture_2d<f32>;
@group(0) @binding(13) var t_c1Height: texture_2d<f32>;
@group(0) @binding(15) var t_c2Disp:   texture_2d<f32>;
@group(0) @binding(19) var t_c2Height: texture_2d<f32>;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,  // unused by fragment; kept for layout compat
    @location(2) uv: vec2<f32>,
    @location(3) height: f32,             // combined Y displacement (for crest colour)
    @location(4) foamHeight: f32,
    @location(5) undispXZ: vec2<f32>,     // undisplaced world XZ for stable gradient UVs
};

// Bilinear sample for RG32Float (UnfilterableFloat) textures.
// textureSample/textureSampleLevel are forbidden by WebGPU spec for this format,
// so we do 4 textureLoad calls and lerp manually — same as sampleGrad() in the
// fragment shader.  This removes the nearest-neighbour "staircase" on displaced
// vertices, especially visible on the coarser outer-LOD tiles.
fn loadRG(t: texture_2d<f32>, uv: vec2<f32>) -> vec2<f32> {
    let sz  = vec2<f32>(textureDimensions(t, 0));
    let szi = vec2<i32>(sz);
    // Map into texel space, shift by -0.5 to interpolate between texel centres.
    let p   = fract(uv) * sz - 0.5;
    let i   = vec2<i32>(floor(p));
    let f   = fract(p);
    let i00 = i & (szi - 1);
    let i10 = (i + vec2<i32>(1, 0)) & (szi - 1);
    let i01 = (i + vec2<i32>(0, 1)) & (szi - 1);
    let i11 = (i + vec2<i32>(1, 1)) & (szi - 1);
    let v00 = textureLoad(t, i00, 0).rg;
    let v10 = textureLoad(t, i10, 0).rg;
    let v01 = textureLoad(t, i01, 0).rg;
    let v11 = textureLoad(t, i11, 0).rg;
    return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    const C0_TILE: f32 = 5.0;
    const C1_TILE: f32 = 40.0;
    const C2_TILE: f32 = 400.0;

    // World-space XZ before displacement (used for stable UV sampling)
    let worldXZ = (transform.model * vec4<f32>(in.position, 1.0)).xz;
    let uv0 = worldXZ / C0_TILE;
    let uv1 = worldXZ / C1_TILE;
    let uv2 = worldXZ / C2_TILE;

    let camDist = length(worldXZ - transform.cameraPos.xz);
    let c0Fade  = clamp(1.0 - (camDist - 20.0) / 60.0, 0.0, 1.0);

    // Normalize each cascade by its own tile size so all three contribute
    // comparable amplitude regardless of Phillips spectrum scale at different k.
    let d0 = loadRG(t_c0Disp,   uv0) * (c0Fade * 0.25 / C0_TILE);
    let h0 = loadRG(t_c0Height, uv0).r * (c0Fade * 0.25 / C0_TILE);

    let d1 = loadRG(t_c1Disp,   uv1) * (1.0 / C1_TILE);
    let h1 = loadRG(t_c1Height, uv1).r * (1.0 / C1_TILE);

    let d2 = loadRG(t_c2Disp,   uv2) * (1.0 / C2_TILE);
    let h2 = loadRG(t_c2Height, uv2).r * (1.0 / C2_TILE);

    let totalDisp = d0 + d1 + d2;
    let totalH    = h0 + h1 + h2;

    var waterPosition = in.position;
    waterPosition.x += totalDisp.x * ocean.choppiness.x;
    waterPosition.z += totalDisp.y * ocean.choppiness.x;
    let yDisplace = totalH * ocean.waveScale.x;
    waterPosition.y += yDisplace;

    let worldPos4 = transform.model * vec4<f32>(waterPosition, 1.0);
    out.worldPos    = worldPos4.xyz;
    out.worldNormal = (transform.model * vec4<f32>(0.0, 1.0, 0.0, 0.0)).xyz;
    out.uv          = in.uv;
    out.height      = yDisplace;
    out.foamHeight  = yDisplace;
    out.undispXZ    = worldXZ;  // pre-displacement XZ for gradient texture lookup
    out.clipPos     = transform.proj * transform.view * worldPos4;
    return out;
}
)";

// Fragment shader: per-pixel normals from gradient textures + Fresnel + foam.
// Gradient textures are sampled here (not in vertex shader) for smooth normals.
constexpr const char* waterFragmentWGSL = R"(

struct TransformUniforms {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    normalCol0: vec4<f32>,
    normalCol1: vec4<f32>,
    normalCol2: vec4<f32>,
    cameraPos: vec3<f32>,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> transform: TransformUniforms;

struct LightData {
    numDir: u32, numPoint: u32, numSpot: u32, numHemi: u32,
    ambient: vec3<f32>, _pad: f32,
    // Phase 1: useLegacyLights flag + 12 bytes pad inserted into the lights
    // uniform header. dirDirection0 must stay at byte offset 48.
    useLegacyLights: u32, _pdUL0: u32, _pdUL1: u32, _pdUL2: u32,
    dirDirection0: vec3<f32>, _pd0: f32,
    dirColor0: vec3<f32>, _pd1: f32,
};
@group(0) @binding(1) var<uniform> lights: LightData;

struct OceanUniforms {
    choppiness:    vec4<f32>,
    contactObj0:   vec4<f32>,   // xy = world XZ centre, z = radius (0 = inactive)
    contactObj1:   vec4<f32>,
    contactObj2:   vec4<f32>,
    contactObj3:   vec4<f32>,
    detailStrength:vec4<f32>,   // x = detail micro-ripple strength (fades with distance)
    foamStrength:  vec4<f32>,
    foamThreshold: vec4<f32>,
    fogDensity:    vec4<f32>,
    fresnelScale:  vec4<f32>,
    mieCoeff:      vec4<f32>,   // x = mie scattering coefficient
    mieG:          vec4<f32>,   // x = mie directional G
    normalStrength:vec4<f32>,
    rayleigh:      vec4<f32>,   // x = rayleigh coefficient
    seaColor:      vec4<f32>,
    specShininess: vec4<f32>,
    tileSize:      vec4<f32>,
    turbidity:     vec4<f32>,   // x = atmospheric turbidity
    waveScale:     vec4<f32>,
    wireframe:     vec4<f32>,   // x > 0.5 = wireframe debug mode
};
@group(0) @binding(2) var<uniform> ocean: OceanUniforms;

// Gradient textures — shared binding slots with vertex stage (5,6 / 11,12 / 17,18)
@group(0) @binding(5)  var t_c0Grad: texture_2d<f32>;
@group(0) @binding(6)  var s_c0Grad: sampler;
@group(0) @binding(11) var t_c1Grad: texture_2d<f32>;
@group(0) @binding(12) var s_c1Grad: sampler;
@group(0) @binding(17) var t_c2Grad: texture_2d<f32>;
@group(0) @binding(18) var s_c2Grad: sampler;

@group(0) @binding(21) var t_foamMap: texture_2d<f32>;
@group(0) @binding(22) var s_foamMap: sampler;

struct VertexOutput {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) height: f32,
    @location(4) foamHeight: f32,
    @location(5) undispXZ: vec2<f32>,
};

// Narkowicz 2015 ACES filmic tone mapping — maps linear HDR to display [0,1].
// Applied manually because WgpuRenderer does not inject tone mapping into
// custom WGSL shaders (unlike the GL renderer with GLSL #include injection).
fn acesFilmic(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + vec3<f32>(0.03))) /
                 (x * (2.43 * x + vec3<f32>(0.59)) + vec3<f32>(0.14)),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---- Preetham Atmospheric Sky (ported from three.js Sky shader) ----
// Computes physically-based sky radiance for a given view direction.
// Used for both water reflections and horizon fog.
const SKY_EE:         f32 = 1000.0;                  // solar irradiance at TOA
const SKY_CUTOFF:     f32 = 1.6110731556870734;       // earth shadow cutoff angle
const SKY_STEEP:      f32 = 1.5;
const SKY_RL_ZENITH:  f32 = 8.4E3;                   // rayleigh zenith optical depth
const SKY_MIE_ZENITH: f32 = 1.25E3;                  // mie zenith optical depth
const SKY_SUN_COS:    f32 = 0.9999566769;             // cos of 0.5° sun disc radius
const SKY_3_16PI:     f32 = 0.05968310365946075;      // 3/(16π)
const SKY_1_4PI:      f32 = 0.07957747154594767;      // 1/(4π)
const SKY_PI:         f32 = 3.141592653589793;

fn skyIntensity(cosZenith: f32) -> f32 {
    let z = clamp(cosZenith, -1.0, 1.0);
    return SKY_EE * max(0.0, 1.0 - exp(-((SKY_CUTOFF - acos(z)) / SKY_STEEP)));
}

fn skyMieCoeffs(turbidity: f32, mieCoeff: f32) -> vec3<f32> {
    let MieConst = vec3<f32>(1.8399918514433978E14, 2.7798023919660528E14, 4.0790479543861094E14);
    let c = (0.2 * turbidity) * 10E-18;
    return 0.434 * c * MieConst * mieCoeff;
}

fn skyHgPhase(cosTheta: f32, g: f32) -> f32 {
    let g2  = g * g;
    let inv = 1.0 / pow(max(1.0 - 2.0 * g * cosTheta + g2, 0.0001), 1.5);
    return SKY_1_4PI * ((1.0 - g2) * inv);
}

// Returns tonemapped sky colour for view direction `dir` (need not be above horizon).
// sunDir must be normalized. Preetham 1999 two-term scattering model.
fn skyColor(dir: vec3<f32>, sunDir: vec3<f32>,
            turbidity: f32, rayleigh: f32, mieCoeff: f32, mieG: f32) -> vec3<f32> {
    let totalRayleigh = vec3<f32>(5.804542996261093E-6, 1.3562911419845635E-5, 3.0265902468824876E-5);
    let up   = vec3<f32>(0.0, 1.0, 0.0);
    let sunE = skyIntensity(dot(sunDir, up));

    let betaR = totalRayleigh * rayleigh;
    let betaM = skyMieCoeffs(turbidity, mieCoeff);

    // Optical path length — clamp zenith to upper hemisphere to avoid singularity.
    let cosZ    = max(0.0, dot(up, dir));
    let za      = acos(cosZ);
    let za_deg  = za * (180.0 / SKY_PI);
    let denom   = cos(za) + 0.15 * pow(max(93.885 - za_deg, 1.0), -1.253);
    let inv     = 1.0 / max(denom, 1e-6);
    let Fex     = exp(-(betaR * (SKY_RL_ZENITH * inv) + betaM * (SKY_MIE_ZENITH * inv)));

    let cosTheta   = dot(dir, sunDir);
    let betaRTheta = betaR * (SKY_3_16PI * (1.0 + cosTheta * cosTheta));
    let betaMTheta = betaM * skyHgPhase(cosTheta, mieG);

    let sumBeta    = max(betaR + betaM, vec3<f32>(1e-10));
    let scatRatio  = (betaRTheta + betaMTheta) / sumBeta;

    let Lin  = pow(max(sunE * scatRatio * (1.0 - Fex), vec3<f32>(0.0)), vec3<f32>(1.5));
    let Lin2 = Lin * mix(vec3<f32>(1.0),
                         pow(max(sunE * scatRatio * Fex, vec3<f32>(0.0)), vec3<f32>(0.5)),
                         clamp(pow(1.0 - dot(up, sunDir), 5.0), 0.0, 1.0));

    let L0   = vec3<f32>(0.1) * Fex;
    let L0f  = L0 + (sunE * 19000.0 * Fex) *
               smoothstep(SKY_SUN_COS, SKY_SUN_COS + 0.00002, cosTheta);

    // three.js Sky pre-correction (pow 1/2.4 matches its retColor step),
    // then ACES to map HDR → display [0,1].
    let texColor = max((Lin2 + L0f) * 0.04 + vec3<f32>(0.0, 0.0003, 0.00075), vec3<f32>(0.0));
    return acesFilmic(pow(texColor, vec3<f32>(1.0 / 2.4)));
}

// Anti-aliased world-space grid line: returns 1.0 on a line, 0.0 between lines.
// Uses fwidth() so lines stay 1-pixel wide regardless of zoom/angle.
fn gridFactor(pos: vec2<f32>, spacing: f32) -> f32 {
    let fw = fwidth(pos);
    let g  = abs(fract(pos / spacing - 0.5) - 0.5) / max(fw, vec2<f32>(0.0001));
    let line = 1.0 - min(g, vec2<f32>(1.0));
    return max(line.x, line.y);
}

// Simple value noise to break up foam edges
fn hash2(p: vec2<f32>) -> f32 {
    let p2 = fract(p * vec2<f32>(5.3983, 5.4427));
    let p3 = p2 + dot(p2, p2.yx + vec2<f32>(21.5351));
    return fract((p3.x + p3.y) * p3.x);
}

// 2D gradient hash — maps lattice point to a unit-ish vector in [-1, 1]^2.
fn hash2Vec(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)),
                      dot(p, vec2<f32>(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(q) * 43758.5453123);
}

// Perlin-style gradient noise with analytical derivatives (quintic C2 interpolant).
// Returns vec3(noise_value, dfdx, dfdz) — the .yz components are the gradient of the
// noise field and can be used directly as a normal-map perturbation.
fn gradNoise(p: vec2<f32>) -> vec3<f32> {
    let i  = floor(p);
    let f  = fract(p);
    // Quintic: u = 6t^5 - 15t^4 + 10t^3,  du = 30t^4 - 60t^3 + 30t^2
    let u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    // Gradient vectors at the four surrounding lattice corners
    let ga = hash2Vec(i);
    let gb = hash2Vec(i + vec2<f32>(1.0, 0.0));
    let gc = hash2Vec(i + vec2<f32>(0.0, 1.0));
    let gd = hash2Vec(i + vec2<f32>(1.0, 1.0));
    // Dot products with offset vectors
    let va = dot(ga, f);
    let vb = dot(gb, f - vec2<f32>(1.0, 0.0));
    let vc = dot(gc, f - vec2<f32>(0.0, 1.0));
    let vd = dot(gd, f - vec2<f32>(1.0, 1.0));
    let k  = va - vb - vc + vd;  // bilinear remainder
    return vec3<f32>(
        va + u.x*(vb - va) + u.y*(vc - va) + u.x*u.y*k,   // noise value
        du.x * (u.y*k + vb - va),                           // df/dx
        du.y * (u.x*k + vc - va)                            // df/dz
    );
}

// Bilinear sample from a repeating RG32Float texture (UnfilterableFloat).
// textureSample/Level are forbidden for RG32Float; we do 4 textureLoad calls instead.
// Assumes power-of-2 texture dimensions (bitwise AND wrap).
fn sampleGrad(t: texture_2d<f32>, uv: vec2<f32>) -> vec2<f32> {
    let sz  = vec2<f32>(textureDimensions(t, 0));
    let szi = vec2<i32>(sz);
    // Map UV into texel space, offset by -0.5 so we interpolate between texel centres
    let p  = fract(uv) * sz - 0.5;
    let i  = vec2<i32>(floor(p));
    let f  = fract(p);
    let i00 = i & (szi - 1);
    let i10 = (i + vec2<i32>(1, 0)) & (szi - 1);
    let i01 = (i + vec2<i32>(0, 1)) & (szi - 1);
    let i11 = (i + vec2<i32>(1, 1)) & (szi - 1);
    let v00 = textureLoad(t, i00, 0).rg;
    let v10 = textureLoad(t, i10, 0).rg;
    let v01 = textureLoad(t, i01, 0).rg;
    let v11 = textureLoad(t, i11, 0).rg;
    return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}
)"  // ---- MSVC 65535-char string literal split ----
R"(
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    const C0_TILE: f32 = 5.0;
    const C1_TILE: f32 = 40.0;
    const C2_TILE: f32 = 400.0;

    // Use undisplaced XZ for stable UV sampling (same reference frame as the FFT).
    let worldXZ = in.undispXZ;

    // Wireframe debug: animated world-space grid on the displaced surface.
    // Tile LOD is recovered from undisplaced XZ via round(pos/tileSize).
    // Three LOD zones shown in distinct colours with their actual vertex spacing:
    //   green  — inner (dist≤1, 256 subdiv, 0.156 m spacing)
    //   yellow — mid   (dist≤3, 128 subdiv, 0.313 m spacing)
    //   orange — outer (dist≤6,  64 subdiv, 0.625 m spacing)
    // Tile boundary lines are always drawn in white.
    if (ocean.wireframe.x > 0.5) {
        let snapTile  = round(transform.cameraPos.xz / C1_TILE);
        let tileIdx   = round(worldXZ / C1_TILE);
        let relIdx    = tileIdx - snapTile;
        let chebyshev = max(abs(relIdx.x), abs(relIdx.y));

        var spacing:  f32;
        var lodColor: vec3<f32>;
        if (chebyshev <= 1.0) {
            spacing  = C1_TILE / 256.0;            // 0.156 m — inner, full detail
            lodColor = vec3<f32>(0.10, 0.95, 0.35);
        } else if (chebyshev <= 3.0) {
            spacing  = C1_TILE / 128.0;            // 0.313 m — mid
            lodColor = vec3<f32>(0.95, 0.85, 0.10);
        } else {
            spacing  = C1_TILE / 64.0;             // 0.625 m — outer, coarsest
            lodColor = vec3<f32>(0.95, 0.40, 0.10);
        }

        let gGrid = gridFactor(worldXZ, spacing);
        let gTile = gridFactor(worldXZ, C1_TILE);  // white tile boundaries

        // Tile boundary overrides LOD grid colour
        let col = mix(lodColor * gGrid * 0.85, vec3<f32>(1.0), gTile);
        let alpha = max(gGrid * 0.85, gTile);
        return vec4<f32>(col, alpha);
    }

    let uv0 = worldXZ / C0_TILE;
    let uv1 = worldXZ / C1_TILE;
    let uv2 = worldXZ / C2_TILE;

    let camDist = length(worldXZ - transform.cameraPos.xz);
    let c0Fade  = clamp(1.0 - (camDist - 20.0) / 60.0, 0.0, 1.0);

    // Per-pixel normals via bilinear gradient sampling (RG32Float — textureLoad only).
    // Each cascade normalized by its tile size (same convention as vertex displacement).
    let g0 = sampleGrad(t_c0Grad, uv0) * (c0Fade * 0.8 / C0_TILE);
    let g1 = sampleGrad(t_c1Grad, uv1) * (1.0 / C1_TILE);
    let g2 = sampleGrad(t_c2Grad, uv2) * (0.25 / C2_TILE);

    let totalGrad = (g0 + g1 + g2) * ocean.normalStrength.x;
    var normal    = normalize(vec3<f32>(-totalGrad.x, 1.0, -totalGrad.y));

    // Detail micro-ripples: 3 octaves of Perlin gradient noise at sub-metre scales.
    // Adds capillary-wave texture that the FFT can't capture at such small wavelengths.
    // Strength fades with camera distance to avoid aliasing shimmer on far water.
    {
        let detailFade = clamp(1.0 - (camDist - 5.0) / 30.0, 0.0, 1.0)
                         * ocean.detailStrength.x;
        if (detailFade > 0.001) {
            // Three octaves spanning ~0.48 m → ~0.07 m wavelength
            let dn1 = gradNoise(worldXZ * 2.1);   // ~0.48 m — medium ripple
            let dn2 = gradNoise(worldXZ * 5.7);   // ~0.18 m — fine chop
            let dn3 = gradNoise(worldXZ * 14.0);  // ~0.07 m — capillary froth
            let detailGrad = (dn1.yz * 0.55 + dn2.yz * 0.30 + dn3.yz * 0.15)
                             * detailFade;
            // Add detail gradient on top of FFT gradient, then renormalise
            normal = normalize(vec3<f32>(
                -(totalGrad.x + detailGrad.x),
                1.0,
                -(totalGrad.y + detailGrad.y)));
        }
    }

    let lightDir = normalize(lights.dirDirection0);
    let sunDir   = lightDir;   // directional light IS the sun
    let viewRay  = normalize(in.worldPos - transform.cameraPos);

    let ndl = max(0.0, dot(normal, lightDir));

    let seaBase       = ocean.seaColor.xyz;
    let seaWaterColor = vec3<f32>(0.8, 0.9, 0.6);

    let nDotV   = max(dot(-viewRay, normal), 0.0);
    let fresnel = pow(1.0 - nDotV, 3.0) * ocean.fresnelScale.x;

    let sharpDiff = pow(ndl * 0.4 + 0.6, 80.0);
    let refracted = seaBase + sharpDiff * seaWaterColor * 0.12;

    let reflectedDir = reflect(viewRay, normal);
    let reflected    = skyColor(normalize(reflectedDir), sunDir,
                                ocean.turbidity.x, ocean.rayleigh.x,
                                ocean.mieCoeff.x,  ocean.mieG.x);

    var color = mix(refracted, reflected, fresnel);

    let dist  = length(in.worldPos - transform.cameraPos);
    let atten = max(1.0 - dist * dist * 0.0003, 0.0);
    color += seaWaterColor * in.height * 0.18 * atten;

    // GGX micro-facet sun disc specular.
    // Instead of Blinn-Phong we use Cook-Torrance with the reflected sky as the
    // light source tint: skyColor(reflectedDir) returns the actual sun disc color
    // (or sky at that angle) so the glint automatically matches the atmosphere.
    // specShininess maps to GGX roughness: higher shininess → sharper glint.
    //   shininess=256 → α≈0.008 (mirror-like)
    //   shininess=120 → α≈0.017 (typical calm sea)
    //   shininess=  4 → α≈0.33  (choppy, diffuse glint)
    {
        let alpha  = max(2.0 / (ocean.specShininess.x + 2.0), 0.004);
        let alpha2 = alpha * alpha;
        // Half-vector between sun and view (viewRay points away from camera)
        let H      = normalize(sunDir - viewRay);
        let NdotH  = max(dot(normal, H), 0.0);
        let NdotV  = max(dot(-viewRay, normal), 0.0);
        let NdotL  = max(dot(normal, sunDir), 0.0);
        let VdotH  = max(dot(-viewRay, H), 0.0);
        // GGX NDF — wide at low shininess, needle-sharp at high shininess
        let d      = NdotH * NdotH * (alpha2 - 1.0) + 1.0;
        let D      = alpha2 / (3.14159265 * d * d);
        // Schlick Fresnel — water F0 ≈ 0.02 at normal incidence
        let F      = 0.02 + 0.98 * pow(1.0 - VdotH, 5.0);
        // Smith height-correlated geometry term
        let k      = alpha * 0.5;
        let G      = (NdotV / (NdotV * (1.0 - k) + k))
                   * (NdotL / (NdotL * (1.0 - k) + k));
        // Cook-Torrance BRDF value
        let brdf   = (D * F * G) / max(4.0 * NdotV, 0.001) * NdotL;
        // Sample sky in the mirror-reflect direction for tint — the sun disc term
        // inside skyColor gives the blinding white when reflectedDir ≈ sunDir.
        // The returned [0,1] ACES value is used as a tint; the HDR scale (×8) lets
        // the final acesFilmic call compress it into a natural bright disc.
        let glintSky = skyColor(normalize(reflectedDir), sunDir,
                                ocean.turbidity.x, ocean.rayleigh.x,
                                ocean.mieCoeff.x,  ocean.mieG.x);
        color += glintSky * brdf * 8.0;
    }

    // Foam: sample at undisplaced XZ so it lines up with the Jacobian computation.
    // Multi-scale hash noise breaks up the hard 0.156m texel boundaries.
    let foamUV     = worldXZ / C1_TILE;  // sampler wraps (repeat mode)
    let foamRaw    = textureSample(t_foamMap, s_foamMap, foamUV).r;
    let fn1 = hash2(worldXZ * 0.4);   // ~2.5m coarse variation
    let fn2 = hash2(worldXZ * 2.1);   // ~0.5m medium grain
    let fn3 = hash2(worldXZ * 6.5);   // ~0.15m fine grain (texel-scale breakup)
    let foamNoise  = fn1 * 0.12 + fn2 * 0.20 + fn3 * 0.13;  // total range ~0-0.45
    let foamSample = foamRaw - foamNoise;
    let foamValue  = smoothstep(ocean.foamThreshold.x - 0.05,
                                ocean.foamThreshold.x + 0.40,
                                foamSample);
    let foamColor  = vec3<f32>(0.85, 0.90, 0.92);
    color = mix(color, foamColor, foamValue * ocean.foamStrength.x);

    // Contact foam around scene objects (buoys, ship hulls, rocks).
    // Physical model:
    //   - Wave crests expand the splash radius and drive intensity.
    //   - Gradient direction gives asymmetry: upstream face gets the most foam.
    //   - Two-octave hash noise breaks the perfect-circle artefact.
    //   - Existing Jacobian foam (foamRaw) seeds extra turbulence near the hull.
    {
        let dispXZ   = in.worldPos.xz;
        let grad2D   = totalGrad / (ocean.normalStrength.x + 0.001); // raw XZ gradient

        // Wave energy at this surface point drives overall foam brightness.
        let waveH    = saturate(in.height * 1.2 + 0.4);              // bias up so troughs ≠ zero
        let steepness = saturate(length(grad2D) * 0.5);
        let waveEnergy = mix(0.25, 1.0, max(waveH, steepness));

        var cFoam: f32 = 0.0;
        let objs = array<vec4<f32>, 4>(
            ocean.contactObj0, ocean.contactObj1,
            ocean.contactObj2, ocean.contactObj3);

        for (var i: i32 = 0; i < 4; i++) {
            let obj = objs[i];
            if (obj.z < 0.001) { continue; }

            let toObj  = obj.xy - dispXZ;
            let dist2D = length(toObj);

            // Radius pulses outward on wave crests (splash).
            let effR = obj.z + max(0.0, in.height) * 0.6;
            let d    = max(0.0, dist2D - effR);

            // Proximity: tight inner ring, soft outer halo.
            let proximity = exp(-d * d * 0.30);

            // Directionality: the wave face pointing toward the object gets more foam.
            // grad2D points uphill; dot with -toObj is positive when the slope rises
            // away from the object (i.e. we are on the upstream face).
            let toObjN    = toObj / (dist2D + 0.001);
            let waveFace  = saturate(dot(-normalize(grad2D + vec2<f32>(0.001)), toObjN));
            let dirFactor = mix(0.3, 1.0, waveFace);

            // Two-octave hash noise: coarse shape + fine froth.
            let n1 = hash2(dispXZ * 1.9 + vec2<f32>(f32(i) * 4.1));
            let n2 = hash2(dispXZ * 5.3 + vec2<f32>(f32(i) * 1.7));
            let noisy = 0.35 + 0.45 * n1 + 0.20 * n2;

                // Jacobian foam acts as a decay memory: foamRaw persists for several seconds
            // after a wave breaks, so it proxies "a wave was here recently".
            // We blend it with the instantaneous wave energy so the ring doesn't
            // snap to zero in every trough — it fades gradually like real foam.
            let jacDecay  = foamRaw * 2.0;                    // amplify — range [0, ~2]
            let sustained = max(waveEnergy, min(jacDecay, 1.0) * 0.7);

            cFoam = max(cFoam, proximity * dirFactor * sustained * noisy);
        }
        color = mix(color, foamColor, min(cFoam, 1.0) * ocean.foamStrength.x);
    }

    let fogFactor  = clamp(exp(-max(dist - 200.0, 0.0) * ocean.fogDensity.x), 0.0, 1.0);
    let horizonDir = normalize(vec3<f32>(viewRay.x, -0.02, viewRay.z));
    let fogColor   = skyColor(horizonDir, sunDir,
                              ocean.turbidity.x, ocean.rayleigh.x,
                              ocean.mieCoeff.x,  ocean.mieG.x);
    color = mix(fogColor, color, fogFactor);

    // ACES tonemapping — WgpuRenderer does not inject this for custom WGSL shaders.
    // Fresnel-based alpha: transparent when looking straight down, nearly opaque at grazing angles.
    // fogFactor drives alpha to 1.0 at distance — more water depth = fully opaque horizon.
    let alpha = mix(1.0, mix(0.9, 0.97, fresnel), fogFactor);
    return vec4<f32>(acesFilmic(max(color, vec3<f32>(0.0))), alpha);
}
)";

// ============================================================================
// Sky Rendering Shaders — Preetham analytical atmosphere
// ============================================================================

// Vertex shader: renders a large box from inside (Side::Back).
// The depth trick (clipPos.z = clipPos.w) forces every fragment to the far
// plane so the sky is always behind every scene object.
constexpr const char* skyVertexWGSL = R"(
struct TransformUniforms {
    model:      mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    normalCol0: vec4<f32>,
    normalCol1: vec4<f32>,
    normalCol2: vec4<f32>,
    cameraPos:  vec3<f32>,
    _pad:       f32,
};
@group(0) @binding(0) var<uniform> transform: TransformUniforms;

struct VertexOut {
    @builtin(position) clipPos:  vec4<f32>,
    @location(0)       worldPos: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VertexOut {
    let worldPos = (transform.model * vec4<f32>(position, 1.0)).xyz;
    var cp = transform.proj * transform.view * vec4<f32>(worldPos, 1.0);
    cp.z = cp.w; // always render at far plane — sky behind everything
    return VertexOut(cp, worldPos);
}
)";

// Fragment shader: Preetham atmospheric scattering (same model as water reflections).
// The sun direction is taken from LightData binding 1 so both sky and water agree.
constexpr const char* skyFragmentWGSL = R"(
struct TransformUniforms {
    model:      mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    normalCol0: vec4<f32>,
    normalCol1: vec4<f32>,
    normalCol2: vec4<f32>,
    cameraPos:  vec3<f32>,
    _pad:       f32,
};
@group(0) @binding(0) var<uniform> transform: TransformUniforms;

struct LightData {
    numDir: u32, numPoint: u32, numSpot: u32, numHemi: u32,
    ambient: vec3<f32>, _pad: f32,
    // Phase 1: useLegacyLights flag + 12 bytes pad inserted into the lights
    // uniform header. dirDirection0 must stay at byte offset 48.
    useLegacyLights: u32, _pdUL0: u32, _pdUL1: u32, _pdUL2: u32,
    dirDirection0: vec3<f32>, _pd0: f32,
    dirColor0:     vec3<f32>, _pd1: f32,
};
@group(0) @binding(1) var<uniform> lights: LightData;

// Custom sky-material uniforms (alphabetical, packed by WgpuRenderer)
struct SkyUniforms {
    mieCoeff:  vec4<f32>,   // x = mie scattering coefficient
    mieG:      vec4<f32>,   // x = mie directional G
    rayleigh:  vec4<f32>,   // x = rayleigh coefficient
    turbidity: vec4<f32>,   // x = turbidity (1–30)
};
@group(0) @binding(2) var<uniform> sky: SkyUniforms;

struct VertexOut {
    @builtin(position) clipPos:  vec4<f32>,
    @location(0)       worldPos: vec3<f32>,
};

// ACES filmic tone mapping — same as in water fragment shader.
fn acesFilmic(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + vec3<f32>(0.03))) /
                 (x * (2.43 * x + vec3<f32>(0.59)) + vec3<f32>(0.14)),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---- Preetham sky model (shared constants with water fragment shader) ----
const SKY_EE:         f32 = 1000.0;
const SKY_CUTOFF:     f32 = 1.6110731556870734;
const SKY_STEEP:      f32 = 1.5;
const SKY_RL_ZENITH:  f32 = 8.4E3;
const SKY_MIE_ZENITH: f32 = 1.25E3;
const SKY_SUN_COS:    f32 = 0.9999566769;
const SKY_3_16PI:     f32 = 0.05968310365946075;
const SKY_1_4PI:      f32 = 0.07957747154594767;
const SKY_PI:         f32 = 3.141592653589793;

fn skyIntensity(cosZenith: f32) -> f32 {
    let z = clamp(cosZenith, -1.0, 1.0);
    return SKY_EE * max(0.0, 1.0 - exp(-((SKY_CUTOFF - acos(z)) / SKY_STEEP)));
}

fn skyMieCoeffs(turbidity: f32, mieCoeff: f32) -> vec3<f32> {
    let MieConst = vec3<f32>(1.8399918514433978E14, 2.7798023919660528E14, 4.0790479543861094E14);
    let c = (0.2 * turbidity) * 10E-18;
    return 0.434 * c * MieConst * mieCoeff;
}

fn skyHgPhase(cosTheta: f32, g: f32) -> f32 {
    let g2  = g * g;
    let inv = 1.0 / pow(max(1.0 - 2.0 * g * cosTheta + g2, 0.0001), 1.5);
    return SKY_1_4PI * ((1.0 - g2) * inv);
}

fn skyColor(dir: vec3<f32>, sunDir: vec3<f32>,
            turbidity: f32, rayleigh: f32, mieCoeff: f32, mieG: f32) -> vec3<f32> {
    let totalRayleigh = vec3<f32>(5.804542996261093E-6, 1.3562911419845635E-5, 3.0265902468824876E-5);
    let up   = vec3<f32>(0.0, 1.0, 0.0);
    let sunE = skyIntensity(dot(sunDir, up));

    let betaR = totalRayleigh * rayleigh;
    let betaM = skyMieCoeffs(turbidity, mieCoeff);

    let cosZ   = max(0.0, dot(up, dir));
    let za     = acos(cosZ);
    let za_deg = za * (180.0 / SKY_PI);
    let denom  = cos(za) + 0.15 * pow(max(93.885 - za_deg, 1.0), -1.253);
    let inv    = 1.0 / max(denom, 1e-6);
    let Fex    = exp(-(betaR * (SKY_RL_ZENITH * inv) + betaM * (SKY_MIE_ZENITH * inv)));

    let cosTheta   = dot(dir, sunDir);
    let betaRTheta = betaR * (SKY_3_16PI * (1.0 + cosTheta * cosTheta));
    let betaMTheta = betaM * skyHgPhase(cosTheta, mieG);

    let sumBeta   = max(betaR + betaM, vec3<f32>(1e-10));
    let scatRatio = (betaRTheta + betaMTheta) / sumBeta;

    let Lin  = pow(max(sunE * scatRatio * (1.0 - Fex), vec3<f32>(0.0)), vec3<f32>(1.5));
    let Lin2 = Lin * mix(vec3<f32>(1.0),
                         pow(max(sunE * scatRatio * Fex, vec3<f32>(0.0)), vec3<f32>(0.5)),
                         clamp(pow(1.0 - dot(up, sunDir), 5.0), 0.0, 1.0));

    let L0  = vec3<f32>(0.1) * Fex;
    let L0f = L0 + (sunE * 19000.0 * Fex) *
              smoothstep(SKY_SUN_COS, SKY_SUN_COS + 0.00002, cosTheta);

    let texColor = max((Lin2 + L0f) * 0.04 + vec3<f32>(0.0, 0.0003, 0.00075), vec3<f32>(0.0));
    return acesFilmic(pow(texColor, vec3<f32>(1.0 / 2.4)));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let dir    = normalize(in.worldPos - transform.cameraPos);
    let sunDir = normalize(lights.dirDirection0);
    // skyColor already returns ACES-tonemapped [0,1] — return directly.
    let color  = skyColor(dir, sunDir,
                          sky.turbidity.x, sky.rayleigh.x,
                          sky.mieCoeff.x,  sky.mieG.x);
    return vec4<f32>(color, 1.0);
}
)";

}// namespace webtide

#endif//WEBTIDE_SHADERS_HPP
