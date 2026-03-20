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

// ============================================================================
// Water Rendering Shaders (WGSL, ported from GLSL)
// ============================================================================

// Vertex shader: displaces water mesh using compute-generated maps
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

struct OceanUniforms {
    tileSize: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};
@group(0) @binding(2) var<uniform> ocean: OceanUniforms;

// Bindings assigned alphabetically by customTextures key:
// displacementMap (3,4), gradientMap (5,6), heightMap (7,8)
@group(0) @binding(3) var t_displacementMap: texture_2d<f32>;
@group(0) @binding(4) var s_displacementMap: sampler;
@group(0) @binding(5) var t_gradientMap: texture_2d<f32>;
@group(0) @binding(6) var s_gradientMap: sampler;
@group(0) @binding(7) var t_heightMap: texture_2d<f32>;
@group(0) @binding(8) var s_heightMap: sampler;
@group(0) @binding(9) var t_reflectionMap: texture_cube<f32>;
@group(0) @binding(10) var s_reflectionMap: sampler;

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
    @location(4) foamHeight: f32,  // world-continuous height for foam (non-repeating)
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let scalingFactor = 1.0 / ocean.tileSize;

    var waterPosition = in.position;

    // World-space UV at 2x tile scale — adjacent tiles sample different halves
    // of the FFT texture, breaking the obvious tile-repetition pattern.
    let worldXZ = (transform.model * vec4<f32>(in.position, 1.0)).xz;
    let largeUV = worldXZ / (ocean.tileSize * 2.0);
    // 3.7x scale is incommensurate with integer tile offsets — no two tiles repeat
    let foamUV  = worldXZ / (ocean.tileSize * 3.7);

    // Primary (tile-local UV)
    let disp1  = textureSampleLevel(t_displacementMap, s_displacementMap, in.uv, 0.0).rg;
    let h1     = textureSampleLevel(t_heightMap, s_heightMap, in.uv, 0.0).r;
    let grad1  = textureSampleLevel(t_gradientMap, s_gradientMap, in.uv, 0.0).rg;

    // Large-scale swell overlay (half amplitude, world-continuous UV)
    let disp2  = textureSampleLevel(t_displacementMap, s_displacementMap, largeUV, 0.0).rg * 0.35;
    let h2     = textureSampleLevel(t_heightMap, s_heightMap, largeUV, 0.0).r * 0.35;
    let grad2  = textureSampleLevel(t_gradientMap, s_gradientMap, largeUV, 0.0).rg * 0.35;

    let displacement = disp1 + disp2;
    let height       = h1 + h2;
    let gradient     = grad1 + grad2;

    // Horizontal displacement (choppy waves)
    waterPosition.x += displacement.x * scalingFactor;
    waterPosition.z += displacement.y * scalingFactor;

    // Vertical displacement (height)
    let yDisplace = height * scalingFactor * 0.5;
    waterPosition.y += yDisplace;

    // Normal from gradient map
    let normal = normalize(vec3<f32>(-gradient.x * scalingFactor * 0.5, 1.0, -gradient.y * scalingFactor * 0.5));

    let worldPos4 = transform.model * vec4<f32>(waterPosition, 1.0);
    out.worldPos = worldPos4.xyz;
    out.worldNormal = (transform.model * vec4<f32>(normal, 0.0)).xyz;
    out.uv = in.uv;
    out.height = yDisplace;
    let foamH  = textureSampleLevel(t_heightMap, s_heightMap, foamUV, 0.0).r;
    out.foamHeight = foamH * scalingFactor * 0.5;
    out.clipPos = transform.proj * transform.view * worldPos4;

    return out;
}
)";

// Fragment shader: Fresnel + cubemap reflection + specular
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

@group(0) @binding(9) var t_reflectionMap: texture_cube<f32>;
@group(0) @binding(10) var s_reflectionMap: sampler;

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

    // Seascape colour constants
    let seaBase       = vec3<f32>(0.1,  0.19, 0.22);
    let seaWaterColor = vec3<f32>(0.8,  0.9,  0.6);

    // Fresnel — power 3, 0.65 scale (seascape style)
    let nDotV   = max(dot(-viewRay, normal), 0.0);
    let fresnel = pow(1.0 - nDotV, 3.0) * 0.65;

    // Refracted body colour: seaBase + sharp wrapped-diffuse tinted by water colour
    // The sharp diffuse (exp 80) puts a subtle sun-coloured glow in the water near
    // the specular region — gives the characteristic yellow-green depth look.
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

    // Foam at wave crests — driven by world-continuous height so it never
    // repeats the same pattern across tiles
    let foam = smoothstep(0.55, 1.1, in.foamHeight);
    color = mix(color, vec3<f32>(1.0), foam * 0.6);

    // Atmospheric haze — starts beyond 40 units, fades into horizon sky
    let fogFactor  = clamp(exp(-max(dist - 200.0, 0.0) * 0.008), 0.0, 1.0);
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
