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

// Foam update: Jacobian-based whitecap detection with temporal decay.
// J ~= (1 + lambda*Jxx)(1 + lambda*Jzz) -- when J < 0 the surface is folding (breaking wave).
// Foam persists with exponential decay: foam = max(newFoam, prevFoam - decay*dt)
constexpr const char* foamUpdateWGSL = R"(
@group(0) @binding(0) var t_jacDiag: texture_2d<f32>;   // .r=Jxx .g=Jzz (spatial)
@group(0) @binding(1) var t_foamIn:  texture_2d<f32>;   // previous frame foam [0,1]
@group(0) @binding(2) var t_foamOut: texture_storage_2d<rg32float, write>;

struct FoamParams {
    lambda: f32,   // choppiness multiplier (larger = more foam)
    decay:  f32,   // foam fade per second
    dt:     f32,   // delta time in seconds
    _pad:   f32,
};
@group(0) @binding(3) var<uniform> params: FoamParams;

@compute @workgroup_size(8,8,1)
fn updateFoam(@builtin(global_invocation_id) id: vec3<u32>) {
    let coord = vec2<i32>(id.xy);
    let jac   = textureLoad(t_jacDiag, coord, 0).xy;
    let J     = (1.0 + params.lambda * jac.x) * (1.0 + params.lambda * jac.y);
    let newFoam  = select(0.0, 1.0, J < 0.0);
    let prevFoam = textureLoad(t_foamIn, coord, 0).r;
    let foam     = max(newFoam, prevFoam - params.decay * params.dt);
    textureStore(t_foamOut, coord, vec4<f32>(foam, 0.0, 0.0, 0.0));
}
)";

// ============================================================================
// Water Rendering Shaders (WGSL, ported from GLSL)
// ============================================================================

// Vertex shader: displaces water mesh using compute-generated maps from 3 cascades
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
    // We only use numDir and the first directional light direction for specular
    dirDirection0: vec3<f32>, _pd0: f32,
    dirColor0: vec3<f32>, _pd1: f32,
    // Rest of light buffer not used by water shader (padded to 704 bytes)
};
@group(0) @binding(1) var<uniform> lights: LightData;

// Alphabetical order matches C++ Uniform packing (each entry = one 16-byte slot):
//   foamStrength, foamThreshold, fogDensity, seaColor, tileSize, waveScale
struct OceanUniforms {
    foamStrength:  vec4<f32>,  // .x = foam blend factor
    foamThreshold: vec4<f32>,  // .x = threshold for Jacobian foam (unused in vertex)
    fogDensity:    vec4<f32>,  // .x = fog exponential coefficient
    seaColor:      vec4<f32>,  // .xyz = deep water base colour
    tileSize:      vec4<f32>,  // .x = reference tile size (kept for uniform compat)
    waveScale:     vec4<f32>,  // .x = vertical displacement multiplier
};
@group(0) @binding(2) var<uniform> ocean: OceanUniforms;

// Cascade texture bindings (alphabetical by customTextures key):
// cascade0Displacement (3,4), cascade0Gradient (5,6), cascade0Height (7,8)
// cascade1Displacement (9,10), cascade1Gradient (11,12), cascade1Height (13,14)
// cascade2Displacement (15,16), cascade2Gradient (17,18), cascade2Height (19,20)
// foamMap (21,22), reflectionMap (23,24)
@group(0) @binding(3) var t_c0Disp:   texture_2d<f32>;
@group(0) @binding(4) var s_c0Disp:   sampler;
@group(0) @binding(5) var t_c0Grad:   texture_2d<f32>;
@group(0) @binding(6) var s_c0Grad:   sampler;
@group(0) @binding(7) var t_c0Height: texture_2d<f32>;
@group(0) @binding(8) var s_c0Height: sampler;
@group(0) @binding(9)  var t_c1Disp:   texture_2d<f32>;
@group(0) @binding(10) var s_c1Disp:   sampler;
@group(0) @binding(11) var t_c1Grad:   texture_2d<f32>;
@group(0) @binding(12) var s_c1Grad:   sampler;
@group(0) @binding(13) var t_c1Height: texture_2d<f32>;
@group(0) @binding(14) var s_c1Height: sampler;
@group(0) @binding(15) var t_c2Disp:   texture_2d<f32>;
@group(0) @binding(16) var s_c2Disp:   sampler;
@group(0) @binding(17) var t_c2Grad:   texture_2d<f32>;
@group(0) @binding(18) var s_c2Grad:   sampler;
@group(0) @binding(19) var t_c2Height: texture_2d<f32>;
@group(0) @binding(20) var s_c2Height: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) height: f32,      // combined Y displacement (for crest colour)
    @location(4) foamHeight: f32,  // kept for compat but foam now driven by Jacobian texture
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Tile sizes for each cascade (must match C++ constants)
    const C0_TILE: f32 = 5.0;    // ripples
    const C1_TILE: f32 = 40.0;   // main chop
    const C2_TILE: f32 = 400.0;  // long swell

    let worldXZ  = (transform.model * vec4<f32>(in.position, 1.0)).xz;
    let uv0 = worldXZ / C0_TILE;
    let uv1 = worldXZ / C1_TILE;
    let uv2 = worldXZ / C2_TILE;

    // Cascade 0 (fine ripples) fades out with camera distance
    let camDist  = length(worldXZ - transform.cameraPos.xz);
    let c0Fade   = clamp(1.0 - (camDist - 20.0) / 60.0, 0.0, 1.0);

    let sf = 1.0 / C2_TILE;  // reference scale factor

    // Sample all 3 cascades
    let d0 = textureSampleLevel(t_c0Disp,   s_c0Disp,   uv0, 0.0).rg * c0Fade * 0.25;
    let h0 = textureSampleLevel(t_c0Height, s_c0Height, uv0, 0.0).r  * c0Fade * 0.25;
    let g0 = textureSampleLevel(t_c0Grad,   s_c0Grad,   uv0, 0.0).rg * c0Fade * 0.5;

    let d1 = textureSampleLevel(t_c1Disp,   s_c1Disp,   uv1, 0.0).rg;
    let h1 = textureSampleLevel(t_c1Height, s_c1Height, uv1, 0.0).r;
    let g1 = textureSampleLevel(t_c1Grad,   s_c1Grad,   uv1, 0.0).rg;

    let d2 = textureSampleLevel(t_c2Disp,   s_c2Disp,   uv2, 0.0).rg * 1.0;
    let h2 = textureSampleLevel(t_c2Height, s_c2Height, uv2, 0.0).r  * 1.0;
    let g2 = textureSampleLevel(t_c2Grad,   s_c2Grad,   uv2, 0.0).rg * 1.5;

    let totalDisp = (d0 + d1 + d2) * sf;
    let totalH    = (h0 + h1 + h2) * sf;
    let totalGrad = (g0 + g1 + g2) * sf;

    var waterPosition = in.position;

    // Apply displacement
    waterPosition.x += totalDisp.x;
    waterPosition.z += totalDisp.y;
    let yDisplace = totalH * 0.5 * ocean.waveScale.x;
    waterPosition.y += yDisplace;

    let normal = normalize(vec3<f32>(-totalGrad.x * 0.5, 1.0, -totalGrad.y * 0.5));

    let worldPos4 = transform.model * vec4<f32>(waterPosition, 1.0);
    out.worldPos   = worldPos4.xyz;
    out.worldNormal = (transform.model * vec4<f32>(normal, 0.0)).xyz;
    out.uv     = in.uv;
    out.height = yDisplace;
    out.foamHeight = yDisplace;  // kept for compat but not used for foam anymore
    out.clipPos = transform.proj * transform.view * worldPos4;
    return out;
}
)";

// Fragment shader: Fresnel + cubemap reflection + specular + Jacobian foam
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
    dirDirection0: vec3<f32>, _pd0: f32,
    dirColor0: vec3<f32>, _pd1: f32,
};
@group(0) @binding(1) var<uniform> lights: LightData;

struct OceanUniforms {
    foamStrength:  vec4<f32>,
    foamThreshold: vec4<f32>,
    fogDensity:    vec4<f32>,
    seaColor:      vec4<f32>,
    tileSize:      vec4<f32>,
    waveScale:     vec4<f32>,
};
@group(0) @binding(2) var<uniform> ocean: OceanUniforms;

@group(0) @binding(21) var t_foamMap:       texture_2d<f32>;
@group(0) @binding(22) var s_foamMap:       sampler;
@group(0) @binding(23) var t_reflectionMap: texture_cube<f32>;
@group(0) @binding(24) var s_reflectionMap: sampler;

struct VertexOutput {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) worldPos: vec3<f32>,
    @location(1) worldNormal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) height: f32,
    @location(4) foamHeight: f32,
};

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let normal   = normalize(in.worldNormal);
    let lightDir = normalize(lights.dirDirection0);
    let viewRay  = normalize(in.worldPos - transform.cameraPos);

    let ndl = max(0.0, dot(normal, lightDir));

    // Seascape colour constants (seaBase tunable via ImGui slider)
    let seaBase       = ocean.seaColor.xyz;
    let seaWaterColor = vec3<f32>(0.8,  0.9,  0.6);

    // Fresnel -- power 3, 0.65 scale (seascape style)
    let nDotV   = max(dot(-viewRay, normal), 0.0);
    let fresnel = pow(1.0 - nDotV, 3.0) * 0.65;

    // Refracted body colour: seaBase + sharp wrapped-diffuse tinted by water colour
    let sharpDiff = pow(ndl * 0.4 + 0.6, 80.0);
    let refracted = seaBase + sharpDiff * seaWaterColor * 0.12;

    // Sky reflection from cubemap
    let reflectedDir = reflect(viewRay, normal);
    let reflected    = textureSample(t_reflectionMap, s_reflectionMap, reflectedDir).rgb;

    var color = mix(refracted, reflected, fresnel);

    // Wave-height colour contribution (seascape: adds water colour at crests)
    let dist  = length(in.worldPos - transform.cameraPos);
    let atten = max(1.0 - dist * dist * 0.0003, 0.0);
    color += seaWaterColor * in.height * 0.18 * atten;

    // Specular (seascape exponent 60 with normalisation)
    let specNorm = (60.0 + 8.0) / (3.14159 * 8.0);
    let spec     = pow(max(0.0, dot(reflect(lightDir, normal), viewRay)), 60.0) * specNorm;
    color += vec3<f32>(spec) * lights.dirColor0;

    // Jacobian foam from persistent foam texture
    let foamRaw   = textureSample(t_foamMap, s_foamMap, in.uv).r;
    let foamValue = clamp((foamRaw - ocean.foamThreshold.x) / (1.0 - ocean.foamThreshold.x + 0.001), 0.0, 1.0);
    let foamColor = vec3<f32>(0.85, 0.90, 0.92);
    color = mix(color, foamColor, foamValue * ocean.foamStrength.x);

    // Atmospheric haze -- starts beyond 200 units, density tunable via ImGui
    let fogFactor  = clamp(exp(-max(dist - 200.0, 0.0) * ocean.fogDensity.x), 0.0, 1.0);
    let horizonDir = normalize(vec3<f32>(viewRay.x, -0.02, viewRay.z));
    let fogColor   = textureSample(t_reflectionMap, s_reflectionMap, horizonDir).rgb;
    color = mix(fogColor, color, fogFactor);

    // Gamma / tone curve (seascape: pow(color, 0.75) to lift the result)
    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(0.75));

    return vec4<f32>(color, 1.0);
}
)";

}// namespace webtide

#endif//WEBTIDE_SHADERS_HPP
