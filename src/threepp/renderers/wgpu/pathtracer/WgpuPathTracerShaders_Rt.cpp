#include "WgpuPathTracerShaders.hpp"

namespace threepp::wgpu_pt {

// ---------------------------------------------------------------------------
// WGSL compute shader — common code shared by raycaster and path tracer
// ---------------------------------------------------------------------------

const char* const csCommonWGSL = R"(

// Named constants
const RAY_EPS:  f32 = 1e-3;   // surface offset for secondary rays
const TRI_MISS: f32 = 1e30;   // sentinel for no intersection

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

struct RtUniforms {
    camOri:     vec4<f32>,
    camFwd:     vec4<f32>,
    camRgt:     vec4<f32>,
    camUp:      vec4<f32>,
    prevCamOri: vec4<f32>,
    prevCamFwd: vec4<f32>,
    prevCamRgt: vec4<f32>,
    prevCamUp:  vec4<f32>,
    iRes:       vec4<f32>,
    tanHalfFov: vec4<f32>,
    frameCount: vec4<f32>,
    triCount:   vec4<f32>,
    mode:       vec4<f32>,
    lightCount: vec4<f32>,
    lightPos:   array<vec4<f32>, 4>,
    lightCol:   array<vec4<f32>, 4>,
    lightType:  array<vec4<f32>, 4>,  // x: 0=point, 1=directional, 2=spot; y=cosAngle; z=cosOuter; w=distance
    lightDir:   array<vec4<f32>, 4>,  // xyz: spotlight direction (normalized); w=decay
    spp:          vec4<f32>,
    movedMeshBits: vec4<u32>,  // bit i = mesh i moved (4 words cover meshes 0-127)
    envColor:      vec4<f32>,  // xyz = color/tint, w = mode: 0=none, 1=solid color, 2=equirect tex
    envIntensity:  vec4<f32>,  // x = intensity scale, y = envWidth, z = envHeight, w = hasEnvCDF
    bgColor:       vec4<f32>,  // xyz = color, w = mode: 0=sky gradient, 1=solid color, 2=equirect tex (bgTex)
    params:        vec4<f32>,  // x = maxBounces
    emissiveInfo:  vec4<f32>,  // x = emissive triangle count, y = total emissive power, z = fireflyCap
    restirParams:  vec4<f32>,  // x = enabled, y = M_clamp, z = emissiveMoved, w = reserved
    bvhAux:        vec4<u32>,  // .x = bvhRootIdx (0 = normal root, >0 = overlay combined root)
};

struct Bvh4NodeGpu {
    cMinX: vec4<f32>,
    cMinY: vec4<f32>,
    cMinZ: vec4<f32>,
    cMaxX: vec4<f32>,
    cMaxY: vec4<f32>,
    cMaxZ: vec4<f32>,
    cIdx:  vec4<u32>,  // child indices (bitcast to i32 for leaf encoding)
}

@group(0) @binding(0) var<uniform> rt:          RtUniforms;
@group(0) @binding(1) var accumRead:  texture_2d<f32>;
@group(0) @binding(2) var accumWrite: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> bvhNodes: array<Bvh4NodeGpu>;
@group(0) @binding(4) var matData:    texture_2d<f32>;
@group(0) @binding(5) var triData:       texture_2d<f32>;
@group(0) @binding(6) var texAtlas:      texture_2d_array<f32>;
@group(0) @binding(7) var hitMeshRead:   texture_2d<f32>;
@group(0) @binding(8) var hitMeshWrite:  texture_storage_2d<rgba16float, write>;
@group(0) @binding(9)  var envTex:      texture_2d<f32>;
@group(0) @binding(10) var gBufWrite:   texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var<storage, read> emissiveTris: array<vec4<f32>>;  // per tri: (triIndex, area, 0, 0)
@group(0) @binding(14) var albedoWrite: texture_storage_2d<rgba16float, write>;
@group(0) @binding(15) var gBufRead:    texture_2d<f32>;
@group(0) @binding(16) var bgTex:       texture_2d<f32>;
@group(0) @binding(17) var reservoirRead:   texture_2d<f32>;
@group(0) @binding(18) var reservoirWrite:  texture_storage_2d<rgba32float, write>;
@group(0) @binding(19) var reservoirWRead:  texture_2d<f32>;
@group(0) @binding(20) var reservoirWWrite: texture_storage_2d<rgba32float, write>;
@group(0) @binding(21) var momentsRead:     texture_2d<f32>;
@group(0) @binding(22) var momentsWrite:    texture_storage_2d<rgba16float, write>;
@group(0) @binding(23) var diffAccumRead:   texture_2d<f32>;
@group(0) @binding(24) var diffAccumWrite:  texture_storage_2d<rgba16float, write>;
@group(0) @binding(25) var specAccumRead:   texture_2d<f32>;
@group(0) @binding(26) var specAccumWrite:  texture_storage_2d<rgba16float, write>;
@group(0) @binding(27) var<storage, read> rtMotionMats: array<mat4x4<f32>>;  // prevWorld * inverse(curWorld) per mesh
@group(0) @binding(28) var giResRead:    texture_2d<f32>;
@group(0) @binding(29) var giResWrite:   texture_storage_2d<rgba32float, write>;
@group(0) @binding(30) var giResWRead:   texture_2d<f32>;
@group(0) @binding(31) var giResWWrite:  texture_storage_2d<rgba32float, write>;
@group(0) @binding(32) var giResLoRead:  texture_2d<f32>;
@group(0) @binding(33) var giResLoWrite: texture_storage_2d<rgba16float, write>;
// Persistent-thread work queue: one atomic<u32> counter.  Threads pull pixel
// indices by atomicAdd(1) until the counter exceeds width*height, at which
// point the thread exits.  Dispatch size is fixed (not proportional to screen
// resolution), so threads that finish a short path immediately grab the next
// work unit rather than idling for warp-mates tracing long paths.
@group(0) @binding(34) var<storage, read_write> pathCounter: atomic<u32>;

// Kernel-split primary-hit buffer.  The primary kernel (rt_primary_main) does
// BVH traversal for the camera ray and writes the minimal RawHit payload here.
// The shading kernel (rt_main) reads it, reconstructs the full Hit via
// loadHitMaterial, and continues with shading/NEE/ReSTIR/bounces — splitting
// register pressure across two smaller kernels rather than one megakernel.
struct PrimaryHitPacked {
    triIdx: i32,  // -1 = miss (sky/env)
    t:      f32,
    u:      f32,
    v:      f32,
}
@group(0) @binding(35) var<storage, read_write> primaryHitBuf: array<PrimaryHitPacked>;

// Dedicated work-queue counter for the primary kernel.  Both primary and rt
// need to traverse all pixels, but they share a command encoder — and GPU-side
// queue.writeBuffer calls to reset a shared counter only land BEFORE the
// encoder executes (not between passes), so a single counter would be
// consumed by the primary pass leaving the rt pass with no work.
@group(0) @binding(36) var<storage, read_write> primaryCounter: atomic<u32>;

// Kernel-split bounce state.
// Written by primaryShade (inline in rt_main), read by rt_bounces_main.  Carries all live state across the primary→bounce kernel
// boundary: the next ray, throughput, accumulated radiance, MIS context,
// adaptive bounce cap, and the bounce-0 surface properties needed by ReSTIR
// GI at bounce 1.  12 vec4s = 192 bytes per pixel.
//
// Packing legend (indices = element .x/.y/.z/.w):
//   [0] rayOrigin.xyz, seed (bitcast u32)
//   [1] rayDir.xyz,    effectiveBounces (bitcast i32)
//   [2] throughput.xyz, flags (bitcast u32:
//                                bit 0 firstBounceSpec
//                                bit 1 afterTransmission
//                                bit 2 touchedMoved
//                                bit 3 pathAlive
//                                bit 4 giResStored
//                                bit 5 skipAccum — foveated/sky/AOV early-exit wrote accum already)
//   [3] diffRad.xyz,   prevMetalness
//   [4] specRad.xyz,   prevAlpha
//   [5] prevNormal.xyz, primaryDepth   (Stage D: accumulation reprojection input)
//   [6] prevWo.xyz,    _unused
//   [7] b0Point.xyz,   b0Alpha
//   [8] b0Normal.xyz,  b0Metal
//   [9] b0Wo.xyz,      b0MeshIdx (bitcast i32, -1 = no bounce-0)
//   [10] b0Albedo.xyz, primaryMatIdx (bitcast i32 — Stage D: hitMesh.g channel)
//   [11] b0F0.xyz,     _unused
struct PathStateEntry {
    w0:  vec4<f32>,
    w1:  vec4<f32>,
    w2:  vec4<f32>,
    w3:  vec4<f32>,
    w4:  vec4<f32>,
    w5:  vec4<f32>,
    w6:  vec4<f32>,
    w7:  vec4<f32>,
    w8:  vec4<f32>,
    w9:  vec4<f32>,
    w10: vec4<f32>,
    w11: vec4<f32>,
}
@group(0) @binding(37) var<storage, read_write> pathStateBuf: array<PathStateEntry>;

// Dedicated work-queue counter for rt_bounces_main.  Same reasoning as
// primaryCounter: CPU-side atomic resets can't interleave with compute passes in
// one encoder, so each kernel in the sample dispatch chain needs its own counter.
@group(0) @binding(38) var<storage, read_write> bounceCounter: atomic<u32>;

const MAX_TEX_SLOTS: i32 = 1024;
const EMPTY_CHILD: i32 = -2147483648;  // INT_MIN — sentinel for unused BVH4 child slots

fn TILE_SIZE() -> i32 { return max(i32(rt.spp.y), 1); }

struct Ray  { origin: vec3<f32>, dir: vec3<f32> }
struct Isect { t: f32, u: f32, v: f32 }
struct RawHit { t: f32, triIdx: i32, u: f32, v: f32 }
struct Hit  {
    t:            f32,
    point:        vec3<f32>,
    normal:       vec3<f32>,
    geoNormal:    vec3<f32>,
    albedo:       vec3<f32>,
    shininess:    f32,
    uv:           vec2<f32>,
    texSlot:      f32,
    metalness:    f32,
    emissive:     vec3<f32>,
    meshIdx:      i32,
    matIdx:       i32,
    transmission: f32,
    ior:          f32,
    frontFace:    f32,
    clearcoat:        f32,
    clearcoatAlpha:   f32,
    sheenColor:       vec3<f32>,
    sheenRoughness:   f32,
    specularColor:    vec3<f32>,
    specularIntensity: f32,
    dispersion:       f32,
    thickness:        f32,
    triIdx:           i32,
}

fn pcg(v: u32) -> u32 {
    var s = v * 747796405u + 2891336453u;
    s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (s >> 22u) ^ s;
}
fn rand(seed: ptr<function, u32>) -> f32 {
    *seed = pcg(*seed);
    return f32(*seed) / 4294967296.0;
}

const PI: f32 = 3.14159265358979;

// ---------------------------------------------------------------------------
// Spatio-temporal blue noise (Heitz & Belcour 2019, "Distributing Monte Carlo
// Errors as a Blue Noise in Screen Space"). Two-tier construction:
//  1. Per-frame, per-dim QMC point from the R2 / golden-ratio plastic constants
//     (low-discrepancy across the integration sequence).
//  2. Per-pixel Cranley-Patterson rotation by a screen-space blue-noise pattern
//     (Interleaved Gradient Noise + golden-ratio temporal advance — a poor-man's
//     STBN that gives blue-noise spectral falloff per frame and decorrelates
//     across frames so the à-trous filter can resolve the residual).
//
// Net effect: the same rays remain Monte Carlo (unbiased), but the per-pixel
// error correlates spatially as blue noise rather than white noise, which
// SVGF-style spatial filters reconstruct dramatically better.
// ---------------------------------------------------------------------------
const R2_A1: f32 = 0.7548776662466927;  // 1/phi2  (2D plastic constant axis 1)
const R2_A2: f32 = 0.5698402909980532;  // 1/phi2^2 (2D plastic constant axis 2)
const R1_A1: f32 = 0.6180339887498949;  // 1/phi   (1D golden-ratio sequence)
const GOLDEN_F32: f32 = 1.6180339887498949;

fn r2Seq(n: u32) -> vec2<f32> {
    return fract(vec2<f32>(f32(n) * R2_A1, f32(n) * R2_A2));
}

// Interleaved Gradient Noise (Jimenez 2014) — approximate 2D blue noise.
// Same coefficients evaluated with (px,py) and (py,px) yield two largely
// uncorrelated blue-noise patterns suitable for X/Y rotation.
fn ign(px: u32, py: u32) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * f32(px) + 0.00583715 * f32(py)));
}

// Temporal-advanced IGN: same blue-noise spatial pattern, rotated each frame
// by golden-ratio (decorrelates frames evenly across the unit interval).
fn ign_t(px: u32, py: u32, fc: u32) -> f32 {
    return fract(ign(px, py) + f32(fc) * GOLDEN_F32);
}

// Per-thread BN state (var<private> = per-invocation in compute).
// `bnDim` advances on every consumed sample and indexes into the QMC sequence,
// so each integrator dimension gets its own well-distributed point.
var<private> bnPx: u32;
var<private> bnPy: u32;
var<private> bnFc: u32;
var<private> bnDim: u32;

fn bnInit(px: u32, py: u32, fc: u32) {
    bnPx = px; bnPy = py; bnFc = fc; bnDim = 0u;
}

fn bnNext1d() -> f32 {
    // Frame and dim both advance the QMC; per-pixel offset is blue-noise.
    let qmc = fract(f32(bnFc) * R1_A1 + f32(bnDim) * 0.7548776662466927);
    let off = ign_t(bnPx, bnPy, bnFc + bnDim * 17u);
    bnDim = bnDim + 1u;
    return fract(qmc + off);
}

fn bnNext2d() -> vec2<f32> {
    // R2 (plastic) sequence stratifies the (frame*K + dim) index in 2D;
    // per-pixel offset uses ign(x,y) and ign(y,x) for two decorrelated axes.
    let idx = bnFc * 8u + bnDim;
    let qmc = fract(vec2<f32>(f32(idx) * R2_A1, f32(idx) * R2_A2));
    let ox  = ign_t(bnPx, bnPy, bnFc + bnDim * 13u);
    let oy  = ign_t(bnPy, bnPx, bnFc + bnDim * 31u);
    bnDim = bnDim + 1u;
    return fract(qmc + vec2<f32>(ox, oy));
}

fn ggxD(NdotH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

fn ggxG1(NdotX: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    return 2.0 * NdotX / max(NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX), 1e-6);
}

fn schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(max(0.0, 1.0 - cosTheta), 5.0);
}
)"
R"(
// sRGB → linear conversion (for baseColor and emissive textures)
fn srgbToLinear(c: vec3<f32>) -> vec3<f32> {
    return select(
        pow((c + 0.055) / 1.055, vec3<f32>(2.4)),
        c / 12.92,
        c <= vec3<f32>(0.04045)
    );
}

// Decode encoded texture slot: slot*16 + wrapS*4 + wrapT
// Wrap modes: 0=repeat, 1=clamp, 2=mirror
fn applyWrap(u: f32, mode: i32) -> f32 {
    if (mode == 1) { return clamp(u, 0.0, 1.0); }
    if (mode == 2) {
        let m = abs(u) % 2.0;
        return select(m, 2.0 - m, m > 1.0);
    }
    return fract(u);
}

fn wrapCoord(v: i32, mode: i32) -> i32 {
    if (mode == 0) { return ((v % TILE_SIZE()) + TILE_SIZE()) % TILE_SIZE(); } // repeat
    return clamp(v, 0, TILE_SIZE() - 1); // clamp / mirror (already wrapped by applyWrap)
}

// Bilinear, wrap-aware atlas fetch — returns full RGBA texel.
// sampleAtlas / sampleAtlasAlpha delegate here to share the wrap/tile logic
// and avoid duplicating 4 texture loads each time both channels are needed.
fn sampleAtlasRGBA(uv: vec2<f32>, texSlot: f32) -> vec4<f32> {
    let enc  = i32(texSlot);
    let slot = enc / 16;
    let wS   = (enc % 16) / 4;
    let wT   = enc % 4;
    let atlasCols = i32(textureDimensions(texAtlas, 0).x) / TILE_SIZE();
    let slotsPerLayer = atlasCols * atlasCols;
    let layer = slot / slotsPerLayer;
    let localSlot = slot % slotsPerLayer;
    let col  = localSlot % atlasCols;
    let row  = localSlot / atlasCols;
    let ox   = col * TILE_SIZE();
    let oy   = row * TILE_SIZE();
    let ts   = f32(TILE_SIZE());
    let wu   = applyWrap(uv.x, wS);
    let wv   = applyWrap(uv.y, wT);
    let fp  = vec2<f32>(wu, wv) * ts - 0.5;
    let x0  = wrapCoord(i32(floor(fp.x)), wS);
    let y0  = wrapCoord(i32(floor(fp.y)), wT);
    let x1  = wrapCoord(i32(floor(fp.x)) + 1, wS);
    let y1  = wrapCoord(i32(floor(fp.y)) + 1, wT);
    let wx  = fp.x - floor(fp.x);
    let wy  = fp.y - floor(fp.y);
    let c00 = textureLoad(texAtlas, vec2<i32>(ox + x0, oy + y0), layer, 0);
    let c10 = textureLoad(texAtlas, vec2<i32>(ox + x1, oy + y0), layer, 0);
    let c01 = textureLoad(texAtlas, vec2<i32>(ox + x0, oy + y1), layer, 0);
    let c11 = textureLoad(texAtlas, vec2<i32>(ox + x1, oy + y1), layer, 0);
    return mix(mix(c00, c10, wx), mix(c01, c11, wx), wy);
}
fn sampleAtlas(uv: vec2<f32>, texSlot: f32) -> vec3<f32> {
    return sampleAtlasRGBA(uv, texSlot).xyz;
}
fn sampleAtlasAlpha(uv: vec2<f32>, texSlot: f32) -> f32 {
    return sampleAtlasRGBA(uv, texSlot).w;
}

// Apply per-channel UV transform from matData.
// channelRow: 6=baseColor, 8=metalRough, 10=normal, 12=emissive, 14=occlusion
fn transformUV(uv0: vec2<f32>, uv1: vec2<f32>, matIdx: i32, channelRow: i32) -> vec2<f32> {
    let r0 = textureLoad(matData, vec2<i32>(matIdx, channelRow), 0);
    let r1 = textureLoad(matData, vec2<i32>(matIdx, channelRow + 1), 0);
    // r0 = (a, b, tx, c),  r1 = (d, ty, texCoord, 0)
    let rawUV = select(uv0, uv1, i32(r1.z) == 1);
    return vec2<f32>(
        r0.x * rawUV.x + r0.y * rawUV.y + r0.z,
        r0.w * rawUV.x + r1.x * rawUV.y + r1.y
    );
}
)"
R"(
// Equirectangular lookup helper (shared by sampleEnv and sampleBackground).
fn equirectUV(d: vec3<f32>) -> vec2<f32> {
    let nd  = normalize(d);
    let phi = atan2(nd.z, nd.x);
    let theta = asin(clamp(nd.y, -1.0, 1.0));
    return vec2<f32>(0.5 + phi / (2.0 * PI), 0.5 - theta / PI);
}

// IBL environment lighting (scene.environment).  Returns BLACK when no environment is set.
// Callers apply rt.envIntensity.x themselves.
fn sampleEnv(d: vec3<f32>) -> vec3<f32> {
    let mode = i32(rt.envColor.w);
    if (mode == 1) {
        return rt.envColor.xyz;
    } else if (mode == 2) {
        let uv = equirectUV(d);
        let sz = vec2<f32>(textureDimensions(envTex, 0));
        let px = vec2<i32>(i32(uv.x * sz.x) % i32(sz.x),
                           clamp(i32(uv.y * sz.y), 0, i32(sz.y) - 1));
        return textureLoad(envTex, px, 0).xyz;
    }
    return vec3<f32>(0.0);  // no environment = no IBL
}

// Background color for ray misses (scene.background).
fn sampleBackground(d: vec3<f32>) -> vec3<f32> {
    let mode = i32(rt.bgColor.w);
    if (mode == 1) {
        return rt.bgColor.xyz;
    } else if (mode == 2) {
        let uv = equirectUV(d);
        let sz = vec2<f32>(textureDimensions(bgTex, 0));
        let px = vec2<i32>(i32(uv.x * sz.x) % i32(sz.x),
                           clamp(i32(uv.y * sz.y), 0, i32(sz.y) - 1));
        return textureLoad(bgTex, px, 0).xyz;
    }
    // Default: procedural sky gradient
    let t = clamp(0.5 * (normalize(d).y + 1.0), 0.0, 1.0);
    return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.32, 0.52, 1.0), t);
}

// Watertight ray-triangle intersection (Woop, Benthin, Wald 2013).
fn triIntersect(ray: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> Isect {
    var r: Isect; r.t = 1e30;
    let ad = abs(ray.dir);
    var kz: u32; var kx: u32; var ky: u32;
    if (ad.x > ad.y && ad.x > ad.z) { kz = 0u; kx = 1u; ky = 2u; }
    else if (ad.y > ad.z)            { kz = 1u; kx = 2u; ky = 0u; }
    else                              { kz = 2u; kx = 0u; ky = 1u; }
    let dz = ray.dir[kz];
    let sz = 1.0 / dz;
    let sx = ray.dir[kx] * sz;
    let sy = ray.dir[ky] * sz;
    let A = v0 - ray.origin;
    let B = v1 - ray.origin;
    let C = v2 - ray.origin;
    let Ax = A[kx] - sx * A[kz]; let Ay = A[ky] - sy * A[kz];
    let Bx = B[kx] - sx * B[kz]; let By = B[ky] - sy * B[kz];
    let Cx = C[kx] - sx * C[kz]; let Cy = C[ky] - sy * C[kz];
    var U = Cx * By - Cy * Bx;
    var V = Ax * Cy - Ay * Cx;
    var W = Bx * Ay - By * Ax;
    if (U == 0.0 || V == 0.0 || W == 0.0) {
        // FMA-based Dekker two-product: fma(a,b,-(a*b)) gives exact rounding error
        let CxBy_p = Cx * By; let CxBy_e = fma(Cx, By, -CxBy_p);
        let CyBx_p = Cy * Bx; let CyBx_e = fma(Cy, Bx, -CyBx_p);
        U = (CxBy_p - CyBx_p) + (CxBy_e - CyBx_e);
        let AxCy_p = Ax * Cy; let AxCy_e = fma(Ax, Cy, -AxCy_p);
        let AyCx_p = Ay * Cx; let AyCx_e = fma(Ay, Cx, -AyCx_p);
        V = (AxCy_p - AyCx_p) + (AxCy_e - AyCx_e);
        let BxAy_p = Bx * Ay; let BxAy_e = fma(Bx, Ay, -BxAy_p);
        let ByAx_p = By * Ax; let ByAx_e = fma(By, Ax, -ByAx_p);
        W = (BxAy_p - ByAx_p) + (BxAy_e - ByAx_e);
    }
    if ((U < 0.0 || V < 0.0 || W < 0.0) && (U > 0.0 || V > 0.0 || W > 0.0)) { return r; }
    let det = U + V + W;
    if (det == 0.0) { return r; }
    let Az = sz * A[kz]; let Bz = sz * B[kz]; let Cz = sz * C[kz];
    let T = U * Az + V * Bz + W * Cz;
    let signDet = select(-1.0, 1.0, det > 0.0);
    let absDet  = det * signDet;
    let sT      = T * signDet;
    if (sT < 1e-4 * absDet) { return r; }
    let rcpDet = 1.0 / det;
    r.u = V * rcpDet; r.v = W * rcpDet; r.t = T * rcpDet;
    return r;
}
)"
R"(
fn aabbDist(bmin: vec3<f32>, bmax: vec3<f32>, ray: Ray, tmax: f32) -> f32 {
    let invD  = vec3<f32>(1.0) / ray.dir;
    let t1    = (bmin - ray.origin) * invD;
    let t2    = (bmax - ray.origin) * invD;
    let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tFar  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z)) * 1.0000007;
    if (tFar >= max(tNear, 0.0) && tNear < tmax) { return max(tNear, 0.0); }
    return 1e30;
}

// Test 4 child AABBs simultaneously using SoA f32 layout.
// Returns vec4 of distances. invD is precomputed once per ray as 1.0/ray.dir.
fn aabbDist4(nd: Bvh4NodeGpu, ray: Ray, invD: vec3<f32>, tmax: f32) -> vec4<f32> {
    let ox = vec4<f32>(ray.origin.x); let oy = vec4<f32>(ray.origin.y); let oz = vec4<f32>(ray.origin.z);
    let idx = vec4<f32>(invD.x); let idy = vec4<f32>(invD.y); let idz = vec4<f32>(invD.z);

    let t1x = (nd.cMinX - ox) * idx;  let t2x = (nd.cMaxX - ox) * idx;
    let t1y = (nd.cMinY - oy) * idy;  let t2y = (nd.cMaxY - oy) * idy;
    let t1z = (nd.cMinZ - oz) * idz;  let t2z = (nd.cMaxZ - oz) * idz;

    let tNear = max(max(min(t1x, t2x), min(t1y, t2y)), min(t1z, t2z));
    // Robust slab test (PBRT): scale tFar by 1+2*gamma(3) to absorb FP rounding.
    let tFar  = min(min(max(t1x, t2x), max(t1y, t2y)), max(t1z, t2z)) * vec4<f32>(1.0000007);

    let hit = tFar >= max(tNear, vec4<f32>(0.0));
    let nearClamp = max(tNear, vec4<f32>(0.0));
    let inRange = nearClamp < vec4<f32>(tmax);
    return select(vec4<f32>(TRI_MISS), nearClamp, hit & inRange);
}

// Compute F0 incorporating PBR specular extension.
// glTF: dielectric F0 = specularIntensity * specularColor * 0.04
fn computeF0(albedo: vec3<f32>, metalness: f32,
             specularColor: vec3<f32>, specularIntensity: f32) -> vec3<f32> {
    let dielectricF0 = vec3<f32>(0.04) * specularColor * specularIntensity;
    return mix(dielectricF0, albedo, metalness);
}

// BRDF evaluation: GGX specular + Lambertian diffuse.
struct BrdfResult { f_diff: vec3<f32>, f_spec: vec3<f32> }
fn evalBrdf(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>,
            albedo: vec3<f32>, metalness: f32, alpha: f32,
            F0: vec3<f32>) -> BrdfResult {
    let hv    = normalize(wo + wi);
    let NdotH = max(0.0, dot(n, hv));
    let NdotV = max(0.001, dot(n, wo));
    let NdotL = max(0.001, dot(n, wi));
    let VdotH = max(0.0, dot(wo, hv));
    let D     = ggxD(NdotH, alpha);
    let F     = schlick(VdotH, F0);
    let G     = ggxG1(NdotV, alpha) * ggxG1(NdotL, alpha);
    return BrdfResult(
        (vec3<f32>(1.0) - F) * albedo * (1.0 - metalness) / PI,
        D * F * G / max(4.0 * NdotV * NdotL, 1e-6));
}

// Charlie sheen NDF (Estevez & Kulla, 2017)
fn charlieD(NdotH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let sinTheta2 = 1.0 - NdotH * NdotH;
    let sinTheta  = max(sqrt(sinTheta2), 1e-6);
    let invA  = 1.0 / a2;
    return (2.0 + invA) * pow(sinTheta, invA) / (2.0 * PI);
}

// Ashikhmin visibility for sheen (Neubelt & Pettineo, 2013)
fn sheenV(NdotV: f32, NdotL: f32) -> f32 {
    return 1.0 / (4.0 * (NdotL + NdotV - NdotL * NdotV));
}

// Evaluate sheen lobe: returns sheen contribution for given directions
fn evalSheen(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>,
             sheenColor: vec3<f32>, sheenRoughness: f32) -> vec3<f32> {
    let hv    = normalize(wo + wi);
    let NdotH = max(0.0, dot(n, hv));
    let NdotV = max(0.001, dot(n, wo));
    let NdotL = max(0.001, dot(n, wi));
    let alpha = max(sheenRoughness * sheenRoughness, 1e-4);
    let D = charlieD(NdotH, alpha);
    let V = sheenV(NdotV, NdotL);
    return sheenColor * D * V;
}

// Lightweight intersection — geometry + alpha test only.
// Full material loading deferred to loadHitMaterial() for the winning triangle.
fn testTriangle(ray: Ray, ti: i32, rh: ptr<function, RawHit>) {
    let r0 = textureLoad(triData, triCoord(ti, 0), 0);
    let v0 = r0.xyz;
    let v1 = textureLoad(triData, triCoord(ti, 1), 0).xyz;
    let v2 = textureLoad(triData, triCoord(ti, 2), 0).xyz;

    let isect = triIntersect(ray, v0, v1, v2);
    if (isect.t >= (*rh).t) { return; }

    let matIdx = i32(r0.w);
    let mat3   = textureLoad(matData, vec2<i32>(matIdx, 3), 0);

    // Side-aware face culling.
    //   mat3.z = 0 → Side::Front  : cull back faces   (dot(dir,geoN) > 0)
    //   mat3.z = 1 → Side::Double : no culling        (glass also forced to 1)
    //   mat3.z = 2 → Side::Back   : cull front faces  (dot(dir,geoN) < 0)
    // The cull applies uniformly to all ray types (primary, shadow, GI, bounce);
    // for the gallery scene this is consistent with rasterized three.js behaviour.
    let sideFlag = mat3.z;
    if (sideFlag < 0.5) {
        let geoNormal = cross(v1 - v0, v2 - v0);
        if (dot(ray.dir, geoNormal) > 0.0) { return; }
    } else if (sideFlag > 1.5) {
        let geoNormal = cross(v1 - v0, v2 - v0);
        if (dot(ray.dir, geoNormal) < 0.0) { return; }
    }

    // Alpha handling: alphaTest (cutoff) or stochastic alpha (blend mode)
    // Negative mat3.w signals BLEND mode; absolute value is opacity.
    let alphaTest = mat3.y;
    let blendMode = mat3.w < 0.0;
    let opacity   = abs(mat3.w);
    let needsAlpha = alphaTest > 0.0 || blendMode;
    if (needsAlpha) {
        var alpha = opacity;
        let mat1 = textureLoad(matData, vec2<i32>(matIdx, 1), 0);
        if (mat1.x >= 0.0) {
            let w  = 1.0 - isect.u - isect.v;
            let uv01 = textureLoad(triData, triCoord(ti, 6), 0);
            let uv2  = textureLoad(triData, triCoord(ti, 7), 0).xy;
            let iuv0 = vec2<f32>(uv01.x, uv01.y) * w
                     + vec2<f32>(uv01.z, uv01.w) * isect.u
                     + uv2                        * isect.v;
            let tuv = transformUV(iuv0, iuv0, matIdx, 6);
            alpha *= sampleAtlasAlpha(tuv, mat1.x);
        }
        if (alphaTest > 0.0) {
            // Hard cutoff
            if (alpha < alphaTest) { return; }
        } else {
            // Stochastic alpha with bias-zero early-outs.
            //
            // Variance reduction: when alpha is ~1.0 (effectively opaque),
            // skip the coin flip entirely — one probabilistic pass-through per
            // ~1000 frames is invisible but contributes zero variance.
            // Symmetric early-out for alpha≈0 (fully transparent).
            // This alone fixes stochastic-alpha fireflies on glTF materials
            // with alpha-packed textures whose alpha is 0.98-1.00.
            // For genuinely translucent regions (0.01 < alpha < 0.99), fall
            // back to the unbiased stochastic coin-flip.
            if (alpha >= 0.99) {
                // always accept — effectively opaque
            } else if (alpha <= 0.01) {
                return;  // always skip — effectively transparent
            } else {
                // XOR in the global frame counter so the coin-flip varies every
                // frame — without this the same triangle at the same hit distance
                // always accepts/rejects, creating a fixed pattern that never converges.
                let h = pcg(pcg(u32(ti) ^ u32(rt.params.y) * 2654435761u) ^ pcg(bitcast<u32>(isect.t)));
                let rng = f32(h) / 4294967295.0;
                if (rng > alpha) { return; }
            }
        }
    }

    (*rh).t = isect.t;
    (*rh).triIdx = ti;
    (*rh).u = isect.u;
    (*rh).v = isect.v;
}

// Full material loading — called once for the closest hit triangle.
fn loadHitMaterial(rh: RawHit, ray: Ray) -> Hit {
    var h: Hit;
    h.t = rh.t;
    h.transmission = 0.0; h.ior = 1.5; h.frontFace = 1.0;
    h.geoNormal = vec3<f32>(0.0);
    h.clearcoat = 0.0; h.clearcoatAlpha = 0.0;
    h.sheenColor = vec3<f32>(0.0); h.sheenRoughness = 0.0;
    h.specularColor = vec3<f32>(1.0); h.specularIntensity = 1.0;
    h.dispersion = 0.0; h.thickness = 0.0;
    h.meshIdx = -1; h.matIdx = -1; h.triIdx = -1;

    let ti = rh.triIdx;
    h.triIdx = ti;
    let r0  = textureLoad(triData, triCoord(ti, 0), 0);
    let v0  = r0.xyz;
    let r1  = textureLoad(triData, triCoord(ti, 1), 0);
    let v1  = r1.xyz;
    let v2  = textureLoad(triData, triCoord(ti, 2), 0).xyz;

    let w  = 1.0 - rh.u - rh.v;
    // Interpolate UV0
    let uv01 = textureLoad(triData, triCoord(ti, 6), 0);
    let uv2_full = textureLoad(triData, triCoord(ti, 7), 0);
    let uv2  = uv2_full.xy;
    let iuv0 = vec2<f32>(uv01.x, uv01.y) * w
             + vec2<f32>(uv01.z, uv01.w) * rh.u
             + uv2                        * rh.v;
    let matIdx = i32(r0.w);
    let mat0   = textureLoad(matData, vec2<i32>(matIdx, 0), 0);
    let mat1   = textureLoad(matData, vec2<i32>(matIdx, 1), 0);
    let mat2   = textureLoad(matData, vec2<i32>(matIdx, 2), 0);
    let mat3   = textureLoad(matData, vec2<i32>(matIdx, 3), 0);

    // Per-channel transformed UVs (UV1 removed; UV0 is the only UV set).
    // Skip all 10 matData reads when all channels use identity UV0.
    let mat18 = textureLoad(matData, vec2<i32>(matIdx, 18), 0);
    var bcUV = iuv0; var mrUV = iuv0; var nmUV = iuv0; var emUV = iuv0;
    if (mat18.z > 0.5) {
        bcUV = transformUV(iuv0, iuv0, matIdx, 6);   // baseColor
        mrUV = transformUV(iuv0, iuv0, matIdx, 8);   // metalRough
        nmUV = transformUV(iuv0, iuv0, matIdx, 10);  // normal
        emUV = transformUV(iuv0, iuv0, matIdx, 12);  // emissive
    }

    let n0 = textureLoad(triData, triCoord(ti, 3), 0).xyz;
    let n1 = textureLoad(triData, triCoord(ti, 4), 0).xyz;
    let n2 = textureLoad(triData, triCoord(ti, 5), 0).xyz;
    var sn = normalize(n0 * w + n1 * rh.u + n2 * rh.v);
    // Side::Back: flip the shading normal so the geometric back face (the
    // rendered side) is treated as the "front" by all downstream isFrontFace
    // logic.  This keeps BRDF, NEE and refraction code unchanged.
    if (mat3.z > 1.5) { sn = -sn; }

    let isFrontFace = dot(ray.dir, sn) < 0.0;
    var finalNorm = select(-sn, sn, isFrontFace);

    // Normal mapping (uses normal-channel UV)
    let normalSlot = mat1.z;
    if (normalSlot >= 0.0) {
        let nmSample = sampleAtlas(nmUV, normalSlot);
        // Row 11 W stores normalScale.y (1.0 = OpenGL convention, -1.0 = DirectX).
        let normalScaleY = textureLoad(matData, vec2<i32>(matIdx, 11), 0).w;
        let nm_xy = nmSample.xy * 2.0 - 1.0;
        // Reconstruct Z from XY — handles BC5 (2-channel) maps where B=0,
        // and is correct for any well-formed unit-vector normal map.
        let nm_z = sqrt(max(0.0, 1.0 - dot(nm_xy, nm_xy)));
        let nmTangent = vec3<f32>(nm_xy.x, nm_xy.y * normalScaleY, nm_z);
        let e1  = v1 - v0;
        let e2  = v2 - v0;
        // Tangent basis uses raw UV0 (geometry-defined)
        let tuv0 = vec2<f32>(uv01.x, uv01.y);
        let tuv1 = vec2<f32>(uv01.z, uv01.w);
        let duv1 = tuv1 - tuv0;
        let duv2 = uv2 - tuv0;
        let denom = duv1.x * duv2.y - duv2.x * duv1.y;
        if (abs(denom) > 1e-8) {
            let invD = 1.0 / denom;
            var T = normalize((e1 * duv2.y - e2 * duv1.y) * invD);
            T = normalize(T - finalNorm * dot(finalNorm, T));
            let B = cross(finalNorm, T);
            let smoothNorm = finalNorm;
            finalNorm = normalize(T * nmTangent.x + B * nmTangent.y + finalNorm * nmTangent.z);
            // Clamp perturbed normal to upper hemisphere of smooth normal.
            // Prevents GGX blow-out when the perturbation pushes the normal
            // below the surface horizon (large D / tiny NdotV → ∞ specular).
            finalNorm = normalize(finalNorm + max(0.0, 1e-3 - dot(finalNorm, smoothNorm)) * smoothNorm);
        }
    }

    // Roughness + metalness from metallicRoughness texture (uses metalRough UV)
    // glTF packs: G = roughness, B = metallic
    var shininess = mat0.w;
    var metalness = mat1.y;
    let roughSlot = mat1.w;
    if (roughSlot >= 0.0) {
        let roughSample = sampleAtlas(mrUV, roughSlot);
        // Final alpha = (baseRoughness * textureRoughness)²
        //             = baseRoughness² * textureRoughness²
        //             = mat0.w * sample² (since mat0.w already stores roughness²)
        shininess = max(1e-4, mat0.w * roughSample.y * roughSample.y);
        metalness = roughSample.z;
    }

    // Geometric (flat) normal — also flipped for Side::Back so it matches
    // the shading-normal convention established above.
    var geoNcross = cross(v1 - v0, v2 - v0);
    if (mat3.z > 1.5) { geoNcross = -geoNcross; }
    let geoNlen   = length(geoNcross);
    let geoN    = select(sn, geoNcross / geoNlen, geoNlen > 1e-8);
    let geoNorm = select(-geoN, geoN, isFrontFace);

    h.point     = ray.origin + rh.t * ray.dir;
    h.normal    = finalNorm;
    h.geoNormal = geoNorm;
    h.albedo    = mat0.xyz;
    h.shininess = shininess;
    h.uv        = bcUV;
    h.texSlot   = mat1.x;
    h.metalness = metalness;
    h.transmission = mat2.w;
    h.ior          = mat3.x;
    h.frontFace    = select(0.0, 1.0, isFrontFace);
    h.meshIdx      = i32(r1.w);
    h.matIdx       = matIdx;
    let mat5 = textureLoad(matData, vec2<i32>(matIdx, 5), 0);
    h.clearcoat      = mat5.x;
    h.clearcoatAlpha = mat5.y;

    // Emissive map (uses emissive UV)
    var emissive = mat2.xyz;
    let emissiveSlot = mat5.z;
    if (emissiveSlot >= 0.0) {
        emissive *= srgbToLinear(sampleAtlas(emUV, emissiveSlot));
    }
    h.emissive = emissive;

    // Advanced PBR features (sheen, specular extension, dispersion, thickness)
    // Skip 2 matData reads when material uses defaults — most common case.
    if (mat18.w > 0.5) {
        let mat16 = textureLoad(matData, vec2<i32>(matIdx, 16), 0);
        h.sheenColor     = mat16.xyz;
        h.sheenRoughness = mat16.w;
        let mat17 = textureLoad(matData, vec2<i32>(matIdx, 17), 0);
        h.specularColor     = mat17.xyz;
        h.specularIntensity = mat17.w;
        h.dispersion = mat18.x;
        h.thickness  = mat18.y;
    } else {
        h.sheenColor = vec3<f32>(0.0);
        h.sheenRoughness = 0.0;
        h.specularColor = vec3<f32>(1.0);
        h.specularIntensity = 1.0;
        h.dispersion = 0.0;
        h.thickness = 0.0;
    }

    return h;
}
)"
R"(
fn decodeLeaf(ci: i32, ray: Ray, rh: ptr<function, RawHit>) {
    let raw = -ci;
    let triStart = (raw - 1) / MAX_LEAF_TRIS;
    let triCount = ((raw - 1) % MAX_LEAF_TRIS) + 1;
    for (var t = triStart; t < triStart + triCount; t++) {
        testTriangle(ray, t, rh);
    }
}

// Shadow hit: lightweight struct for shadow rays (no normal maps, roughness, clearcoat, etc.)
struct ShadowHit {
    t:                f32,
    point:            vec3<f32>,
    normal:           vec3<f32>,
    albedo:           vec3<f32>,
    uv:               vec2<f32>,
    texSlot:          f32,
    meshIdx:          i32,
    transmission:     f32,
    frontFace:        f32,
    attenuationColor: vec3<f32>,
    attenuationDist:  f32,
}

// Shadow traversal reuses RawHit + decodeLeaf for geometry test.
// Material loading deferred to loadShadowHitMaterial for the closest hit only.
fn loadShadowHitMaterial(rh: RawHit, ray: Ray) -> ShadowHit {
    var h: ShadowHit;
    h.t = rh.t; h.meshIdx = -1; h.transmission = 0.0;
    let ti = rh.triIdx;
    let r0  = textureLoad(triData, triCoord(ti, 0), 0);
    let matIdx = i32(r0.w);
    let mat2 = textureLoad(matData, vec2<i32>(matIdx, 2), 0);
    // Fast path: opaque material — skip normals, UVs, attenuation.
    if (mat2.w < 0.01) {
        h.point = ray.origin + rh.t * ray.dir;
        return h;
    }
    let r1  = textureLoad(triData, triCoord(ti, 1), 0);
    let v0  = r0.xyz;
    let v1  = r1.xyz;
    let v2  = textureLoad(triData, triCoord(ti, 2), 0).xyz;
    let mat0 = textureLoad(matData, vec2<i32>(matIdx, 0), 0);
    let mat1 = textureLoad(matData, vec2<i32>(matIdx, 1), 0);
    let w  = 1.0 - rh.u - rh.v;
    let uv01 = textureLoad(triData, triCoord(ti, 6), 0);
    let uv2  = textureLoad(triData, triCoord(ti, 7), 0).xy;
    let iuv0 = vec2<f32>(uv01.x, uv01.y) * w
             + vec2<f32>(uv01.z, uv01.w) * rh.u
             + uv2                        * rh.v;
    let bcUV = transformUV(iuv0, iuv0, matIdx, 6);
    let n0 = textureLoad(triData, triCoord(ti, 3), 0).xyz;
    let n1 = textureLoad(triData, triCoord(ti, 4), 0).xyz;
    let n2 = textureLoad(triData, triCoord(ti, 5), 0).xyz;
    let sn = normalize(n0 * w + n1 * rh.u + n2 * rh.v);
    h.point        = ray.origin + rh.t * ray.dir;
    let isFrontFace = dot(ray.dir, sn) < 0.0;
    h.normal       = select(-sn, sn, isFrontFace);
    h.albedo       = mat0.xyz;
    h.uv           = bcUV;
    h.texSlot      = mat1.x;
    h.meshIdx      = i32(r1.w);
    h.transmission = mat2.w;
    h.frontFace    = select(0.0, 1.0, isFrontFace);
    let mat4 = textureLoad(matData, vec2<i32>(matIdx, 4), 0);
    h.attenuationColor = mat4.xyz;
    h.attenuationDist  = mat4.w;
    return h;
}

fn sceneHitRaw(ray: Ray, maxT: f32) -> RawHit {
    var rh: RawHit; rh.t = maxT; rh.triIdx = -1;
    let invD = vec3<f32>(1.0) / ray.dir;
    var stack: array<i32, 32>;
    var top: i32 = 0;
    stack[0] = i32(rt.bvhAux.x); top = 1;  // bvhAux.x = root node index (0=normal, >0=overlay)

    while (top > 0) {
        top -= 1;
        let nd = bvhNodes[stack[top]];

        let dists = aabbDist4(nd, ray, invD, rh.t);
        if (all(dists >= vec4<f32>(1e30))) { continue; }

        let ci0 = bitcast<i32>(nd.cIdx.x);
        let ci1 = bitcast<i32>(nd.cIdx.y);
        let ci2 = bitcast<i32>(nd.cIdx.z);
        let ci3 = bitcast<i32>(nd.cIdx.w);

        if (dists.x < 1e30 && ci0 < 0 && ci0 != EMPTY_CHILD) { decodeLeaf(ci0, ray, &rh); }
        if (dists.y < 1e30 && ci1 < 0 && ci1 != EMPTY_CHILD) { decodeLeaf(ci1, ray, &rh); }
        if (dists.z < 1e30 && ci2 < 0 && ci2 != EMPTY_CHILD) { decodeLeaf(ci2, ray, &rh); }
        if (dists.w < 1e30 && ci3 < 0 && ci3 != EMPTY_CHILD) { decodeLeaf(ci3, ray, &rh); }

        // Push internal children nearest-last (popped first).
        // Branchless 5-comparator sorting network using select() — avoids
        // conditional branches that serialise across GPU warps.
        var n0 = dists.x; var n1 = dists.y; var n2 = dists.z; var n3 = dists.w;
        var k0 = ci0; var k1 = ci1; var k2 = ci2; var k3 = ci3;
        if (k0 < 0) { n0 = 1e30; } if (k1 < 0) { n1 = 1e30; }
        if (k2 < 0) { n2 = 1e30; } if (k3 < 0) { n3 = 1e30; }
        // Each step: c = (na < nb); new_na = select(na,nb,c); new_nb = select(nb,na,c)
        // Result is descending order (n0 >= n1 >= n2 >= n3).
        var c: bool; var tn: f32; var tk: i32;
        c=n0<n1; tn=select(n0,n1,c); n1=select(n1,n0,c); n0=tn; tk=select(k0,k1,c); k1=select(k1,k0,c); k0=tk;
        c=n2<n3; tn=select(n2,n3,c); n3=select(n3,n2,c); n2=tn; tk=select(k2,k3,c); k3=select(k3,k2,c); k2=tk;
        c=n0<n2; tn=select(n0,n2,c); n2=select(n2,n0,c); n0=tn; tk=select(k0,k2,c); k2=select(k2,k0,c); k0=tk;
        c=n1<n3; tn=select(n1,n3,c); n3=select(n3,n1,c); n1=tn; tk=select(k1,k3,c); k3=select(k3,k1,c); k1=tk;
        c=n1<n2; tn=select(n1,n2,c); n2=select(n2,n1,c); n1=tn; tk=select(k1,k2,c); k2=select(k2,k1,c); k1=tk;
        if (n0 < 1e30) { stack[top] = k0; top++; }
        if (n1 < 1e30) { stack[top] = k1; top++; }
        if (n2 < 1e30) { stack[top] = k2; top++; }
        if (n3 < 1e30) { stack[top] = k3; top++; }
    }
    return rh;
}


fn sceneHit(ray: Ray) -> Hit {
    let rh = sceneHitRaw(ray, 1e30);
    if (rh.triIdx < 0) {
        var h: Hit; h.t = 1e30; h.meshIdx = -1; h.matIdx = -1; h.triIdx = -1; h.transmission = 0.0; h.ior = 1.5;
        h.frontFace = 1.0; h.geoNormal = vec3<f32>(0.0);
        h.clearcoat = 0.0; h.clearcoatAlpha = 0.0;
        h.sheenColor = vec3<f32>(0.0); h.sheenRoughness = 0.0;
        h.specularColor = vec3<f32>(1.0); h.specularIntensity = 1.0;
        h.dispersion = 0.0; h.thickness = 0.0;
        return h;
    }
    return loadHitMaterial(rh, ray);
}

)"
R"(
// Fast any-hit traversal for shadow rays — exits on first intersection.
// No sorting, no closest-hit search. Much faster for large scenes.
fn sceneAnyHit(ray: Ray, maxT: f32) -> RawHit {
    var rh: RawHit; rh.t = maxT; rh.triIdx = -1;
    let invD = vec3<f32>(1.0) / ray.dir;
    var stack: array<i32, 32>;
    var top: i32 = 0;
    stack[0] = i32(rt.bvhAux.x); top = 1;  // bvhAux.x = root node index (0=normal, >0=overlay)

    while (top > 0) {
        top -= 1;
        let nd = bvhNodes[stack[top]];
        let dists = aabbDist4(nd, ray, invD, rh.t);
        if (all(dists >= vec4<f32>(1e30))) { continue; }

        let ci0 = bitcast<i32>(nd.cIdx.x);
        let ci1 = bitcast<i32>(nd.cIdx.y);
        let ci2 = bitcast<i32>(nd.cIdx.z);
        let ci3 = bitcast<i32>(nd.cIdx.w);

        if (dists.x < 1e30 && ci0 < 0 && ci0 != EMPTY_CHILD) { decodeLeaf(ci0, ray, &rh); if (rh.triIdx >= 0) { return rh; } }
        if (dists.y < 1e30 && ci1 < 0 && ci1 != EMPTY_CHILD) { decodeLeaf(ci1, ray, &rh); if (rh.triIdx >= 0) { return rh; } }
        if (dists.z < 1e30 && ci2 < 0 && ci2 != EMPTY_CHILD) { decodeLeaf(ci2, ray, &rh); if (rh.triIdx >= 0) { return rh; } }
        if (dists.w < 1e30 && ci3 < 0 && ci3 != EMPTY_CHILD) { decodeLeaf(ci3, ray, &rh); if (rh.triIdx >= 0) { return rh; } }

        // Push internal children nearest-last (popped first) — front-to-back
        // traversal order maximises early-exit probability for shadow rays.
        var n0 = dists.x; var n1 = dists.y; var n2 = dists.z; var n3 = dists.w;
        var k0 = ci0; var k1 = ci1; var k2 = ci2; var k3 = ci3;
        if (k0 < 0) { n0 = 1e30; } if (k1 < 0) { n1 = 1e30; }
        if (k2 < 0) { n2 = 1e30; } if (k3 < 0) { n3 = 1e30; }
        if (n0 < n1) { let t0=n0;n0=n1;n1=t0; let t1=k0;k0=k1;k1=t1; }
        if (n2 < n3) { let t0=n2;n2=n3;n3=t0; let t1=k2;k2=k3;k3=t1; }
        if (n0 < n2) { let t0=n0;n0=n2;n2=t0; let t1=k0;k0=k2;k2=t1; }
        if (n1 < n3) { let t0=n1;n1=n3;n3=t0; let t1=k1;k1=k3;k3=t1; }
        if (n1 < n2) { let t0=n1;n1=n2;n2=t0; let t1=k1;k1=k2;k2=t1; }
        if (n0 < 1e30) { stack[top] = k0; top++; }
        if (n1 < 1e30) { stack[top] = k1; top++; }
        if (n2 < 1e30) { stack[top] = k2; top++; }
        if (n3 < 1e30) { stack[top] = k3; top++; }
    }
    return rh;
}

// Fast boolean occlusion test — true if anything blocks the ray.
fn sceneOccluded(ray: Ray, maxDist: f32) -> bool {
    let rh = sceneAnyHit(ray, maxDist);
    return rh.triIdx >= 0;
}

// Closest-hit shadow — needed for transmission shadow chains that walk front-to-back.
fn sceneHitShadow(ray: Ray, maxDist: f32) -> ShadowHit {
    let rh = sceneHitRaw(ray, maxDist);
    if (rh.triIdx < 0) {
        var h: ShadowHit;
        h.t = maxDist; h.meshIdx = -1; h.transmission = 0.0;
        h.frontFace = 1.0; h.attenuationColor = vec3<f32>(1.0); h.attenuationDist = 0.0;
        return h;
    }
    return loadShadowHitMaterial(rh, ray);
}

fn makeRay(px: vec2<f32>, res: vec2<f32>) -> Ray {
    let aspect = res.x / res.y;
    let ndc = vec2<f32>((px.x / res.x) * 2.0 - 1.0,
                         1.0 - (px.y / res.y) * 2.0);
    var ray: Ray;
    ray.origin = rt.camOri.xyz;
    ray.dir    = normalize(rt.camFwd.xyz
                         + rt.camRgt.xyz * (ndc.x * rt.tanHalfFov.x * aspect)
                         + rt.camUp.xyz  * (ndc.y * rt.tanHalfFov.x));
    return ray;
}

// ---------------------------------------------------------------------------
// Shared light/shadow helpers — used by both classic NEE and ReSTIR
// ---------------------------------------------------------------------------

// Evaluate analytical light: returns (direction, attenuated color).
// For directional lights, dir = normalized lightPos, dist is ignored.
struct LightEval { dir: vec3<f32>, color: vec3<f32>, dist: f32 }
fn evalAnalyticalLight(li: i32, point: vec3<f32>) -> LightEval {
    var le: LightEval;
    var lc    = rt.lightCol[li].xyz;
    let ltype = i32(rt.lightType[li].x);
    let lPos  = rt.lightPos[li].xyz;
    le.dir  = select(normalize(lPos - point), normalize(lPos), ltype == 1);
    le.dist = select(length(lPos - point), 1e30, ltype == 1);

    // Physically correct attenuation for point and spot lights:
    //   irradiance = intensity / r²  (inverse-square law)
    // Optional smooth distance window (Frostbite/UE4) avoids a hard cutoff:
    //   window(r, rmax) = saturate(1 - (r/rmax)^4)²
    if (ltype != 1) {
        let inv_sq = 1.0 / max(le.dist * le.dist, 1e-4);
        let lDist  = rt.lightType[li].w;
        let window = select(1.0,
                            pow(max(1.0 - pow(le.dist / lDist, 4.0), 0.0), 2.0),
                            lDist > 0.0);
        lc *= inv_sq * window;
    }

    if (ltype == 2) {
        let spotDir  = rt.lightDir[li].xyz;
        let cosTheta = dot(-le.dir, spotDir);
        let cosInner = rt.lightType[li].y;
        let cosOuter = rt.lightType[li].z;
        lc *= clamp((cosTheta - cosOuter) / max(cosInner - cosOuter, 1e-6), 0.0, 1.0);
    }
    le.color = lc;
    return le;
}

// Evaluate light radiance for a given light type code and position.
// Handles analytical lights, emissive triangles, and environment.
fn evalLightRadiance(lightPos: vec3<f32>, lightType: f32, point: vec3<f32>) -> vec3<f32> {
    let typeCode = i32(lightType);
    if (typeCode < 0) {
        return sampleEnv(lightPos) * rt.envIntensity.x;
    } else if (typeCode >= 1000) {
        let eTi = typeCode - 1000;
        let eMatIdx = i32(textureLoad(triData, triCoord(eTi, 0), 0).w);
        return textureLoad(matData, vec2<i32>(eMatIdx, 2), 0).xyz;
    } else {
        let lcount = i32(rt.lightCount.x);
        if (typeCode < lcount) {
            let le = evalAnalyticalLight(typeCode, point);
            return le.color;
        }
    }
    return vec3<f32>(0.0);
}

// Shadow ray with glass-aware Beer-Lambert volumetric absorption.
// Returns RGB attenuation (0 = fully occluded, 1 = fully visible).
//
// Glass light-leak fix: maxBounces must be large enough to survive passing through
// a glass object (front + back face = 2 surfaces) AND still test the opaque wall
// or occluder beyond it.  With maxBounces=2 a single glass sphere exhausts both
// iterations (si=0: front face, si=1: back face) before the wall is ever checked,
// so the wall behind is never tested → light leaks through the wall.
// With maxBounces=4 the shadow ray handles up to 2 glass objects in the path and
// still has budget left for the first opaque occluder.  All call sites use 4.
//
// Origin offset: `normal * 1e-3` keeps the origin just outside the originating
// surface, preventing self-intersection.  For NdotL > 0 (required for NEE to fire)
// the shadow direction has a positive outward component, so the convex surface is
// never re-entered at positive t regardless of offset strategy.
fn traceShadowRay(origin: vec3<f32>, normal: vec3<f32>, dir: vec3<f32>,
                  maxDist: f32, maxBounces: i32) -> vec3<f32> {
    var sr: Ray;
    sr.origin = origin + normal * 1e-3;   // stay just outside the originating surface
    sr.dir = dir;
    var atten = vec3<f32>(1.0);
    var glassAttCol = vec3<f32>(1.0);
    var glassAttDist = 0.0;
    var inGlass = false;
    for (var si = 0; si < maxBounces; si++) {
        let sh = sceneHitShadow(sr, maxDist);
        if (sh.t >= maxDist) { break; }
        if (sh.transmission < 0.01) { return vec3<f32>(0.0); }
        var shAlbedo = sh.albedo;
        if (sh.texSlot >= 0.0) { shAlbedo = srgbToLinear(sampleAtlas(sh.uv, sh.texSlot)); }
        atten *= shAlbedo * sh.transmission;
        if (sh.frontFace > 0.5) {
            glassAttCol = sh.attenuationColor;
            glassAttDist = sh.attenuationDist;
            inGlass = true;
        } else if (inGlass && glassAttDist > 0.0) {
            let absorbCoeff = -log(max(glassAttCol, vec3<f32>(1e-6))) / glassAttDist;
            atten *= exp(-absorbCoeff * sh.t);
            inGlass = false;
        }
        sr.origin = sh.point + sr.dir * 1e-3;
    }
    return atten;
}

// BRDF + sheen combined evaluation: returns full lobe sum (diffuse + specular + sheen).
// Clearcoat lobe evaluation: dielectric GGX specular layered on top of base.
// Returns the radiance multiplier for wi given wo at this surface point.
fn evalClearcoat(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>,
                 ccWeight: f32, ccAlpha: f32) -> f32 {
    if (ccWeight <= 0.0 || ccAlpha <= 0.0) { return 0.0; }
    let hv    = normalize(wo + wi);
    let NdotH = max(0.0, dot(n, hv));
    let NdotV = max(0.001, dot(n, wo));
    let NdotL = max(0.001, dot(n, wi));
    let D     = ggxD(NdotH, ccAlpha);
    let G     = ggxG1(NdotV, ccAlpha) * ggxG1(NdotL, ccAlpha);
    return D * G / max(4.0 * NdotV * NdotL, 1e-6) * ccWeight;
}

fn evalBrdfFull(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>,
                albedo: vec3<f32>, metalness: f32, alpha: f32, F0: vec3<f32>,
                sheenColor: vec3<f32>, sheenRoughness: f32,
                clearcoat: f32, clearcoatAlpha: f32) -> vec3<f32> {
    let brdf = evalBrdf(wo, wi, n, albedo, metalness, alpha, F0);
    // Clearcoat Fresnel attenuation of the base layer. ccF0 = 0.04 (dielectric IOR ~1.5).
    // Mirrors the stochastic split in the BRDF-sampling path: base lobes receive
    // (1 - ccWeight) of the energy; clearcoat receives ccWeight.
    var ccWeight = 0.0;
    if (clearcoat > 0.0) {
        let ccF0 = 0.04;
        let NdotV_cc = max(0.0, dot(n, wo));
        let ccFresnel = ccF0 + (1.0 - ccF0) * pow(1.0 - NdotV_cc, 5.0);
        ccWeight = clearcoat * ccFresnel;
    }
    var lobeSum = (brdf.f_diff + brdf.f_spec) * (1.0 - ccWeight);
    if (ccWeight > 0.0) {
        lobeSum += vec3<f32>(evalClearcoat(wo, wi, n, ccWeight, clearcoatAlpha));
    }
    let sheenLum = dot(sheenColor, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (sheenLum > 0.001) {
        lobeSum += evalSheen(wo, wi, n, sheenColor, sheenRoughness);
    }
    return lobeSum;
}

struct BrdfSplit { diff: vec3<f32>, spec: vec3<f32> }

fn evalBrdfFullSplit(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>,
                     albedo: vec3<f32>, metalness: f32, alpha: f32, F0: vec3<f32>,
                     sheenColor: vec3<f32>, sheenRoughness: f32,
                     clearcoat: f32, clearcoatAlpha: f32) -> BrdfSplit {
    let brdf = evalBrdf(wo, wi, n, albedo, metalness, alpha, F0);
    var ccWeight = 0.0;
    if (clearcoat > 0.0) {
        let ccF0 = 0.04;
        let NdotV_cc = max(0.0, dot(n, wo));
        let ccFresnel = ccF0 + (1.0 - ccF0) * pow(1.0 - NdotV_cc, 5.0);
        ccWeight = clearcoat * ccFresnel;
    }
    var d = brdf.f_diff * (1.0 - ccWeight);
    var s = brdf.f_spec * (1.0 - ccWeight);
    if (ccWeight > 0.0) {
        s += vec3<f32>(evalClearcoat(wo, wi, n, ccWeight, clearcoatAlpha));
    }
    let sheenLum = dot(sheenColor, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (sheenLum > 0.001) {
        // Sheen is a diffuse-like soft reflection; route to diffuse
        d += evalSheen(wo, wi, n, sheenColor, sheenRoughness);
    }
    return BrdfSplit(d, s);
}

// Compute direction and max distance from a reservoir light to a shading point.
struct ReservoirDir { dir: vec3<f32>, maxDist: f32 }
fn reservoirLightDir(lightPos: vec3<f32>, lightType: f32, point: vec3<f32>) -> ReservoirDir {
    var rd: ReservoirDir;
    let typeCode = i32(lightType);
    if (typeCode < 0) {
        rd.dir = normalize(lightPos);
        rd.maxDist = 1e30;
    } else if (typeCode >= 1000) {
        let toL = lightPos - point;
        let dist = length(toL);
        rd.dir = toL / dist;
        rd.maxDist = dist - 1e-2;
    } else {
        let ltype = i32(rt.lightType[typeCode].x);
        if (ltype == 1) {
            rd.dir = normalize(lightPos);
            rd.maxDist = 1e30;
        } else {
            let toL = lightPos - point;
            let dist = length(toL);
            rd.dir = toL / dist;
            rd.maxDist = dist - 1e-2;
        }
    }
    return rd;
}

// Sample an emissive triangle from the power-weighted CDF.
// Returns: xyz = sampled point on triangle, w = triangle index (as float).
struct EmissiveSample { point: vec3<f32>, normal: vec3<f32>, triIdx: i32, area: f32, power: f32 }
fn sampleEmissiveTriCdf(seed: ptr<function, u32>, totalPower: f32, emTriCount: i32) -> EmissiveSample {
    // Blue-noise: 1D pick + 2D barycentric (3 BN samples per NEE call).
    let xi = bnNext1d() * totalPower;
    var lo = 0;
    var hi = emTriCount - 1;
    while (lo < hi) {
        let mid = (lo + hi) >> 1;
        if (emissiveTris[mid].z < xi) { lo = mid + 1; } else { hi = mid; }
    }
    let emInfo = emissiveTris[lo];
    let eTi  = i32(emInfo.x);
    let ev0 = textureLoad(triData, triCoord(eTi, 0), 0).xyz;
    let ev1 = textureLoad(triData, triCoord(eTi, 1), 0).xyz;
    let ev2 = textureLoad(triData, triCoord(eTi, 2), 0).xyz;
    let bnBary = bnNext2d();
    let su1 = sqrt(bnBary.x);
    let u2  = bnBary.y;
    var es: EmissiveSample;
    es.point  = (1.0 - su1) * ev0 + su1 * (1.0 - u2) * ev1 + su1 * u2 * ev2;
    es.normal = normalize(cross(ev1 - ev0, ev2 - ev0));
    es.triIdx = eTi;
    es.area   = emInfo.y;
    es.power  = emInfo.w;
    return es;
}
)";

// ---------------------------------------------------------------------------
// WGSL compute shader — path tracer code
// ---------------------------------------------------------------------------
const char* const csPathTraceWGSL = R"(

@group(0) @binding(12) var envCdfTex:     texture_2d<f32>;  // conditional CDF (per-row), R32Float
@group(0) @binding(13) var envMargTex:    texture_2d<f32>;  // marginal CDF (1-column), R32Float

const HAS_ENV_CDF: bool = /*ENV_CDF_FLAG*/false;

// r2Seq, ign, ign_t and the bnNext1d/2d helpers are defined in csCommonWGSL.

fn cosineHemisphere(n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let bn  = bnNext2d();
    let u1  = bn.x;
    let u2  = bn.y;
    let r   = sqrt(u1);
    let phi = 6.28318530718 * u2;
    let lx  = r * cos(phi);
    let ly  = r * sin(phi);
    let lz  = sqrt(max(0.0, 1.0 - u1));
    let nt  = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.99);
    let rgt = normalize(cross(nt, n));
    let up  = cross(n, rgt);
    return normalize(lx * rgt + ly * up + lz * n);
}

// Heitz 2018 VNDF (Visible Normal Distribution Function) sampling.
// Samples half-vectors proportional to the visible microfacet area,
// eliminating wasted below-horizon samples from plain D(h) sampling.
fn sampleVNDF(wo: vec3<f32>, n: vec3<f32>, alpha: f32,
              seed: ptr<function, u32>) -> vec3<f32> {
    // Build local frame around n
    let nt  = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.99);
    let t1  = normalize(cross(nt, n));
    let t2  = cross(n, t1);
    // Transform wo to local frame (t1, t2, n)
    let woLocal = vec3<f32>(dot(wo, t1), dot(wo, t2), dot(wo, n));
    // Stretch to isotropic configuration (isotropic alpha)
    let woStr = normalize(vec3<f32>(alpha * woLocal.x, alpha * woLocal.y, woLocal.z));
    // Build orthonormal basis around stretched wo
    let lensq = woStr.x * woStr.x + woStr.y * woStr.y;
    let T1 = select(vec3<f32>(1.0, 0.0, 0.0),
                    vec3<f32>(-woStr.y, woStr.x, 0.0) / sqrt(lensq),
                    lensq > 1e-7);
    let T2 = cross(woStr, T1);
    // Sample projected disk — blue-noise stratified.
    let bn  = bnNext2d();
    let u1  = bn.x;
    let u2  = bn.y;
    let r   = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let t1s = r * cos(phi);
    let s   = 0.5 * (1.0 + woStr.z);
    let t2s = mix(sqrt(max(0.0, 1.0 - t1s * t1s)), r * sin(phi), s);
    // Compute half-vector in stretched space, then unstretch
    let nhLocal = t1s * T1 + t2s * T2
                + sqrt(max(0.0, 1.0 - t1s * t1s - t2s * t2s)) * woStr;
    let hLocal = normalize(vec3<f32>(alpha * nhLocal.x, alpha * nhLocal.y, max(1e-6, nhLocal.z)));
    // Transform back to world space
    let hm = hLocal.x * t1 + hLocal.y * t2 + hLocal.z * n;
    return reflect(-wo, hm);
}

// PDF of the VNDF sampling strategy, expressed in reflected-direction (wi) space.
fn vndfPdf(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>, alpha: f32) -> f32 {
    let hm    = normalize(wo + wi);
    let NdotH = max(0.0, dot(n, hm));
    let NdotV = max(1e-6, dot(n, wo));
    let D     = ggxD(NdotH, alpha);
    let G1v   = ggxG1(NdotV, alpha);
    // VNDF PDF_h = D * G1 * VdotH / NdotV; Jacobian to wi: / (4 * VdotH)
    // VdotH cancels → PDF_wi = D * G1 / (4 * NdotV)
    return D * G1v / (4.0 * NdotV);
}

// Combined BRDF PDF (mixed specular + diffuse lobes, matching the path tracer's sampling strategy).
fn brdfPdf(wo: vec3<f32>, wi: vec3<f32>, n: vec3<f32>, alpha: f32, metalness: f32) -> f32 {
    let NdotL = dot(n, wi);
    if (NdotL <= 0.0) { return 0.0; }
    let p_spec  = mix(0.5, 0.98, metalness);
    let specPdf = vndfPdf(wo, wi, n, alpha);
    let diffPdf = NdotL / PI;
    return p_spec * specPdf + (1.0 - p_spec) * diffPdf;
}

// Route a clamped contribution to diffuse or specular radiance buffer.
// Bounce 0 NEE: always diffuse (specular split handled at ReSTIR shade site).
// Bounce > 0:   follows firstBounceSpec flag.
fn addSplit(diff: ptr<function, vec3<f32>>,
            spec: ptr<function, vec3<f32>>,
            contrib: vec3<f32>, cap: f32,
            bounce: i32, firstSpec: bool) {
    var c = contrib;
    if (cap > 0.0) {
        let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
        if (lum > cap) { c *= cap / lum; }
    }
    if (bounce == 0 || !firstSpec) {
        *diff += c;
    } else {
        // Indirect specular: aggressive per-bounce clamp.
        // Bounce 1 = direct specular NEE (from bounce 0 split) → use full cap.
        // Bounce 2+ = indirect specular → clamp to 2.0 / bounce.
        // This is biased but eliminates the firefly speckle that spatial
        // filtering alone cannot remove at interactive sample counts.
        if (bounce >= 2) {
            let indirectCap = 2.0 / f32(bounce);
            let sLum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
            if (sLum > indirectCap) { c *= indirectCap / sLum; }
        }
        *spec += c;
    }
}

// ---------------------------------------------------------------------------
// ReSTIR DI — Reservoir data structure and helpers
// ---------------------------------------------------------------------------
struct Reservoir {
    lightPos:  vec3<f32>,   // world-space position (area/point) or direction (env/dir)
    lightType: f32,         // 0..999 = analytical light index, 1000+ = emissive tri (1000+triIdx), -1 = env
    W_sum:     f32,         // running weight sum
    M:         f32,         // candidate count
    W:         f32,         // final weight = W_sum / (M * p_hat)
    p_hat:     f32,         // target PDF of selected sample
}

fn emptyReservoir() -> Reservoir {
    return Reservoir(vec3<f32>(0.0), -1.0, 0.0, 0.0, 0.0, 0.0);
}

fn updateReservoir(r: ptr<function, Reservoir>,
                   pos: vec3<f32>, ltype: f32, w: f32,
                   p_hat_new: f32, seed: ptr<function, u32>) {
    (*r).W_sum += w;
    (*r).M += 1.0;
    if (rand(seed) < w / max((*r).W_sum, 1e-20)) {
        (*r).lightPos  = pos;
        (*r).lightType = ltype;
        (*r).p_hat     = p_hat_new;
    }
}

fn finalizeReservoir(r: ptr<function, Reservoir>) {
    (*r).W = (*r).W_sum / max((*r).M * (*r).p_hat, 1e-20);
}


// Evaluate unshadowed target function for a reservoir sample.
// Returns NdotL * luminance(Le) — geometry-weighted light intensity without BRDF.
//
// Using a BRDF-based target PDF introduced texture-pattern bias: the roughness clamp
// (safeAlpha) needed to tame GGX D spikes on smooth surfaces created a systematic
// mismatch between the reservoir weight and the actual shade-time BRDF evaluation.
// The mismatch was spatially correlated with the roughness texture, making the texture
// pattern visible in the lighting.
//
// A pure luminance target is unbiased, roughness-independent, and produces good
// reservoir quality for diffuse-dominant scenes. Specular variance is slightly higher
// but handled by the temporal accumulation.
fn restirTargetPdf(point: vec3<f32>, normal: vec3<f32>, wo: vec3<f32>,
                   albedo: vec3<f32>, metalness: f32, alpha: f32, F0: vec3<f32>,
                   lightPos: vec3<f32>, lightType: f32,
                   lightLe: vec3<f32>) -> f32 {
    // Determine direction to light
    let typeCode = i32(lightType);
    var ln: vec3<f32>;
    if (typeCode < 0) {
        // Environment: lightPos is direction
        ln = normalize(lightPos);
    } else if (typeCode < 1000) {
        // Analytical light
        let ltype = i32(rt.lightType[typeCode].x);
        if (ltype == 1) {
            ln = normalize(lightPos); // directional
        } else {
            ln = normalize(lightPos - point);
        }
    } else {
        // Emissive triangle
        ln = normalize(lightPos - point);
    }

    let NdotL = dot(normal, ln);
    if (NdotL <= 0.0) { return 0.0; }

    return NdotL * luminance(lightLe);
}
)"
R"(
// Binary search a 1D CDF stored in a texture row.
// Returns the index where cdf[index] >= xi.
fn cdfSearch(tex: texture_2d<f32>, row: i32, size: i32, xi: f32) -> i32 {
    var lo = 0;
    var hi = size - 1;
    for (var iter = 0; iter < 16; iter++) {
        if (lo >= hi) { break; }
        let mid = (lo + hi) >> 1;
        if (textureLoad(tex, vec2<i32>(mid, row), 0).x < xi) { lo = mid + 1; } else { hi = mid; }
    }
    return lo;
}

// Direction → equirectangular UV (matches sampleEnv mapping)
fn dirToUV(d: vec3<f32>) -> vec2<f32> {
    let nd = normalize(d);
    let phi   = atan2(nd.z, nd.x);
    let theta = asin(clamp(nd.y, -1.0, 1.0));
    return vec2<f32>(0.5 + phi / (2.0 * PI), 0.5 - theta / PI);
}

// UV → direction (inverse of dirToUV)
fn uvToDir(uv: vec2<f32>) -> vec3<f32> {
    let phi   = (uv.x - 0.5) * 2.0 * PI;
    let theta = (0.5 - uv.y) * PI;
    let ct = cos(theta);
    return vec3<f32>(ct * cos(phi), sin(theta), ct * sin(phi));
}

// Importance-sample the environment map using precomputed 2D CDF.
// Returns: xyz = sampled direction, w = PDF (in solid angle measure).
fn sampleEnvImportance(seed: ptr<function, u32>) -> vec4<f32> {
    let envW = i32(rt.envIntensity.y);
    let envH = i32(rt.envIntensity.z);

    // 1) Sample marginal CDF to pick a row (v)
    let xi_v = rand(seed);
    let row  = cdfSearch(envMargTex, 0, envH, xi_v);

    // 2) Sample conditional CDF at that row to pick a column (u)
    let xi_u = rand(seed);
    let col  = cdfSearch(envCdfTex, row, envW, xi_u);

    // 3) Convert to UV with sub-pixel centering
    let u = (f32(col) + 0.5) / f32(envW);
    let v = (f32(row) + 0.5) / f32(envH);
    let dir = uvToDir(vec2<f32>(u, v));

    // 4) Compute PDF = luminance(pixel) / totalLuminance, converted to solid angle
    let envCol = textureLoad(envTex, vec2<i32>(col, row), 0).xyz;
    let lum = 0.2126 * envCol.r + 0.7152 * envCol.g + 0.0722 * envCol.b + 1e-10;

    let theta = (0.5 - v) * PI;
    let sinTheta = max(abs(sin(theta)), 1e-6);
    let totalSum = rt.envIntensity.w;
    let pdf_uv = lum / max(totalSum, 1e-10);
    let pdf = pdf_uv * f32(envW * envH) / (2.0 * PI * PI * sinTheta);

    return vec4<f32>(dir, pdf);
}

// PDF for a given direction under env importance sampling (for MIS).
fn envImportancePdf(d: vec3<f32>) -> f32 {
    let envW = i32(rt.envIntensity.y);
    let envH = i32(rt.envIntensity.z);
    let uv = dirToUV(d);
    let envCol = sampleEnv(d);
    let lum = 0.2126 * envCol.r + 0.7152 * envCol.g + 0.0722 * envCol.b + 1e-10;
    let theta = (0.5 - uv.y) * PI;
    let sinTheta = max(abs(sin(theta)), 1e-6);
    let totalSum = rt.envIntensity.w;
    let pdf_uv = lum / max(totalSum, 1e-10);
    return pdf_uv * f32(envW * envH) / (2.0 * PI * PI * sinTheta);
}

// Per-contribution firefly clamp for injection-time MIS spikes.
fn addClamped(rad: ptr<function, vec3<f32>>, contrib: vec3<f32>, cap: f32) {
    if (!(contrib.x == contrib.x) || !(contrib.y == contrib.y) || !(contrib.z == contrib.z)) {
        return;
    }
    let lum = dot(contrib, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (lum > cap) { *rad += contrib * (cap / lum); }
    else           { *rad += contrib; }
}

fn isMeshMoved(idx: i32) -> bool {
    if (idx < 0 || idx >= 128) { return false; }
    let ui  = u32(idx);
    let bit = ui & 31u;
    let wi  = ui >> 5u;  // 0..3 — selects x/y/z/w of movedMeshBits
    return ((rt.movedMeshBits[wi] >> bit) & 1u) != 0u;
}

// -------- Octahedral normal encoding (for GI reservoir packing) --------
fn octEncode(n: vec3<f32>) -> vec2<f32> {
    let t = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) {
        return (1.0 - abs(t.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), t.xy >= vec2<f32>(0.0));
    }
    return t;
}
fn octDecode(e: vec2<f32>) -> vec3<f32> {
    var n = vec3<f32>(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (n.z < 0.0) {
        let xy = (1.0 - abs(n.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0));
        n = vec3<f32>(xy, n.z);
    }
    return normalize(n);
}
fn packOctNormal(n: vec3<f32>) -> f32 {
    let e = octEncode(n);
    return bitcast<f32>(pack2x16snorm(e));
}
fn unpackOctNormal(p: f32) -> vec3<f32> {
    return octDecode(unpack2x16snorm(bitcast<u32>(p)));
}

// -------- GI target PDF: importance of a secondary hit as a virtual light --------
fn giTargetPdf(point: vec3<f32>, normal: vec3<f32>, wo: vec3<f32>,
               albedo: vec3<f32>, metalness: f32, alpha: f32, F0: vec3<f32>,
               secHitPos: vec3<f32>, secHitNorm: vec3<f32>, Lo: vec3<f32>) -> f32 {
    let wi = normalize(secHitPos - point);
    let NdotL = dot(normal, wi);
    if (NdotL <= 0.0) { return 0.0; }
    let delta = secHitPos - point;
    let dist2 = dot(delta, delta);
    let cosTheta2 = max(dot(secHitNorm, -wi), 0.0);
    let G = cosTheta2 / max(dist2, 0.01);
    let brdf = evalBrdf(wo, wi, normal, albedo, metalness, max(alpha, 0.1), F0);
    return luminance((brdf.f_diff + brdf.f_spec) * NdotL * Lo * G);
}

// Jacobian of the reconnection shift: ratio of differential solid angles subtended
// by the secondary hit (secPos/secNorm) as seen from two different primary shading
// points (fromPrimary = source, toPrimary = current).
// J = (cosθ_to · d²_from) / (cosθ_from · d²_to)
// where θ = angle at secNorm vs. direction toward the respective primary.
// Equals 1 when from == to (static scene).  Clamped to [0, 4] to prevent
// fireflies from near-grazing secondary hits.
fn reconnJacobian(secPos: vec3<f32>, secNorm: vec3<f32>,
                  fromPrimary: vec3<f32>, toPrimary: vec3<f32>) -> f32 {
    let vFrom = fromPrimary - secPos;
    let vTo   = toPrimary   - secPos;
    let d2From = dot(vFrom, vFrom);
    let d2To   = dot(vTo,   vTo);
    let cosFrom = abs(dot(secNorm, vFrom) / max(sqrt(d2From), 1e-6));
    let cosTo   = abs(dot(secNorm, vTo)   / max(sqrt(d2To),   1e-6));
    return clamp((cosTo * d2From) / max(cosFrom * d2To, 1e-6), 0.0, 4.0);
}

struct SplitRadiance { diff: vec3<f32>, spec: vec3<f32> }

// PrimaryShadeResult: all state that flows from primaryShade (bounce-0) into
// runBounces (bounces 1..N).  Serialized to pathStateBuf at the kernel boundary
// between rt_main and rt_bounces_main — keep field set in sync with that layout.
struct PrimaryShadeResult {
    rayOrigin:         vec3<f32>,
    rayDir:            vec3<f32>,
    throughput:        vec3<f32>,
    diffRad:           vec3<f32>,  // radiance accumulated during primaryShade
    specRad:           vec3<f32>,
    prevNormal:        vec3<f32>,
    prevWo:            vec3<f32>,
    prevAlpha:         f32,
    prevMetalness:     f32,
    afterTransmission: bool,
    firstBounceSpec:   bool,
    effectiveBounces:  i32,
    // Bounce-0 surface snapshot — consumed by ReSTIR GI at bounce 1 inside runBounces.
    b0Point:   vec3<f32>,
    b0Normal:  vec3<f32>,
    b0Wo:      vec3<f32>,
    b0Albedo:  vec3<f32>,
    b0F0:      vec3<f32>,
    b0Metal:   f32,
    b0Alpha:   f32,
    b0MeshIdx: i32,
    giResStored: bool,
    // false = primaryShade terminated the path (miss/unlit/backface/RR) and
    // runBounces must be skipped.  Callers should return diffRad+specRad directly.
    pathAlive: bool,
    // Primary-hit metadata needed by rt_bounces_main's accumulation.
    // primaryDepth == 0.0 for sky/miss (denoiser sentinel).  primaryMatIdx is
    // propagated to hitMesh.g channel.
    primaryDepth:  f32,
    primaryMatIdx: i32,
}
)";

// runBounces (bounce split, Stage B): executes the i>=1 tail of the original
// pathTrace loop as a standalone function.  Called from pathTrace's wrapper
// after primaryShade completes bounce 0.  Structurally identical to the tail
// of the old loop — only i==0 branches are removed.  Split into two string
// literals for the MSVC 16380-byte limit.
const char* const csRunBouncesWGSL1 = R"(
fn runBounces(state:           PrimaryShadeResult,
              seed:             ptr<function, u32>,
              pixel:            vec2<i32>,
              primaryMeshIdx:   u32,
              touchedMoved:     ptr<function, bool>) -> SplitRadiance {
    // Reconstruct loop locals from primaryShade state.
    var ray: Ray;
    ray.origin = state.rayOrigin;
    ray.dir    = state.rayDir;
    var throughput    = state.throughput;
    // runBounces returns the BOUNCE contribution only; caller sums with state.diffRad/specRad.
    var diffRad       = vec3<f32>(0.0);
    var specRad       = vec3<f32>(0.0);
    var firstBounceSpec  = state.firstBounceSpec;
    var prevNormal       = state.prevNormal;
    var prevAlpha        = state.prevAlpha;
    var prevMetalness    = state.prevMetalness;
    var prevWo           = state.prevWo;
    var afterTransmission = state.afterTransmission;
    let effectiveBounces = state.effectiveBounces;
    let b0Point    = state.b0Point;
    let b0Normal   = state.b0Normal;
    let b0Wo       = state.b0Wo;
    let b0Albedo   = state.b0Albedo;
    let b0F0       = state.b0F0;
    let b0Metal    = state.b0Metal;
    let b0Alpha    = state.b0Alpha;
    let b0MeshIdx  = state.b0MeshIdx;
    var giResStored = state.giResStored;

    for (var i = 1; i < effectiveBounces; i++) {
        var h = sceneHit(ray);
        if (h.t >= 1e29) {
            // Bounced miss: env IBL with MIS complement to BRDF env-importance NEE.
            var envMisW = 1.0;
            if (HAS_ENV_CDF && rt.envColor.w > 1.5 && prevAlpha > 0.01) {
                let pdf_env  = envImportancePdf(ray.dir);
                let pdf_brdf = brdfPdf(prevWo, normalize(ray.dir), prevNormal, prevAlpha, prevMetalness);
                envMisW = pdf_brdf / max(pdf_brdf + pdf_env, 1e-8);
            }
            addSplit(&diffRad, &specRad,
                throughput * sampleEnv(ray.dir) * rt.envIntensity.x * envMisW,
                rt.emissiveInfo.z, i, firstBounceSpec);
            if (i == 1 && rt.spp.x > 0.5 && !giResStored) {
                textureStore(giResWrite,   pixel, vec4<f32>(0.0));
                textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
                textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
                giResStored = true;
            }
            break;
        }
        // Secondary bounce landed on a DIFFERENT moved mesh than primary — flag for
        // temporal-reprojection anti-lag so the denoiser won't trust history here.
        if (isMeshMoved(h.meshIdx) && u32(h.meshIdx) != primaryMeshIdx) {
            *touchedMoved = true;
        }

        let emTriCount = i32(rt.emissiveInfo.x);
        let totalPower = rt.emissiveInfo.y;

        // Emissive MIS at bounce >= 1.
        if (length(h.emissive) > 0.0) {
            if (emTriCount == 0 || afterTransmission) {
                // No NEE available, or after transmission: full weight to this path.
                addSplit(&diffRad, &specRad, throughput * h.emissive,
                         rt.emissiveInfo.z, i, firstBounceSpec);
            } else if (i == 1 && rt.restirParams.x > 0.5 && !firstBounceSpec) {
                // Diffuse bounce-0 used ReSTIR DI which already produced unbiased direct;
                // adding BRDF-sampled emissive here would double-count.  Specular bounce-0
                // still needs this contribution (handled via firstBounceSpec == true else).
            } else {
                let cosLight = abs(dot(h.geoNormal, -ray.dir));
                if (cosLight > 1e-6) {
                    let emLum = 0.2126 * h.emissive.r + 0.7152 * h.emissive.g + 0.0722 * h.emissive.b;
                    let pdf_light = (emLum * h.t * h.t) / (totalPower * cosLight);
                    let pdf_brdf  = brdfPdf(prevWo, normalize(ray.dir), prevNormal, prevAlpha, prevMetalness);
                    let w = pdf_brdf / max(pdf_brdf + pdf_light, 1e-8);
                    if (w == w && w < 1e10) {
                        addSplit(&diffRad, &specRad, throughput * h.emissive * w,
                                 rt.emissiveInfo.z, i, firstBounceSpec);
                    }
                }
            }
        }

        var albedo = h.albedo;
        if (h.texSlot >= 0.0) { albedo = srgbToLinear(sampleAtlas(h.uv, h.texSlot)); }

        // Unlit: flat colour, no bouncing.  No i==0 reservoir-zero branch here.
        if (h.shininess < 0.0) {
            diffRad += throughput * albedo;
            if (i == 1 && rt.spp.x > 0.5 && !giResStored) {
                textureStore(giResWrite,   pixel, vec4<f32>(0.0));
                textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
                textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
                giResStored = true;
            }
            break;
        }

        let wo = normalize(-ray.dir);
        let F0_h = computeF0(albedo, h.metalness, h.specularColor, h.specularIntensity);

        // ReSTIR GI gating: fires only at the first secondary hit (i == 1).
        let giAlphaMin = select(0.01, 0.1, b0Metal > 0.5);
        let useReSTIRGI = i == 1 && rt.spp.x > 0.5 && !afterTransmission
                          && h.transmission < 0.05 && b0Alpha > giAlphaMin;
        var giLo = vec3<f32>(0.0);

        // ==== Classic NEE ==== (useReSTIR is i==0-only, so always false here.)
        let lcount = i32(rt.lightCount.x);

        // Analytical lights
        for (var li = 0; li < 4; li++) {
            if (li >= lcount) { break; }
            let le = evalAnalyticalLight(li, h.point);
            let NdotL = dot(h.normal, le.dir);
            if (NdotL <= 0.0) { continue; }
            let shadowAtten = traceShadowRay(h.point, h.normal, le.dir, le.dist - 1e-3, 4);
            if (shadowAtten.x + shadowAtten.y + shadowAtten.z > 0.001) {
                let cap = rt.emissiveInfo.z;
                let lobeSum = evalBrdfFull(wo, le.dir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                           h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                let neeContrib = shadowAtten * lobeSum * NdotL * le.color;
                if (useReSTIRGI && !giResStored) {
                    giLo += neeContrib;
                } else {
                    addSplit(&diffRad, &specRad, throughput * neeContrib, cap, i, firstBounceSpec);
                }
            }
        }

        // Emissive triangle NEE (power-weighted CDF)
        if (emTriCount > 0) {
            let totalPower2 = rt.emissiveInfo.y;
            let es = sampleEmissiveTriCdf(seed, totalPower2, emTriCount);
            let toLight = es.point - h.point;
            let dist    = length(toLight);
            let ln      = toLight / dist;
            let NdotL   = dot(h.normal, ln);
            let cosLight = abs(dot(es.normal, -ln));
            if (NdotL > 0.0 && cosLight > 1e-6) {
                let emAtten = traceShadowRay(h.point, h.normal, ln, dist - 1e-2, 4);
                if (emAtten.x + emAtten.y + emAtten.z > 0.001) {
                    let eMatIdx = i32(textureLoad(triData, triCoord(es.triIdx, 0), 0).w);
                    let emColor = textureLoad(matData, vec2<i32>(eMatIdx, 2), 0).xyz;
                    let pdf = (es.power * dist * dist) / (totalPower2 * es.area * cosLight);
                    let pdf_brdf_nee = brdfPdf(wo, ln, h.normal, h.shininess, h.metalness);
                    let w_light = pdf / max(pdf + pdf_brdf_nee, 1e-8);
                    let cap = rt.emissiveInfo.z;
                    let lobeSum3 = evalBrdfFull(wo, ln, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                                 h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                    let emNeeContrib = emAtten * lobeSum3 * NdotL * emColor * w_light / pdf;
                    if (useReSTIRGI && !giResStored) {
                        giLo += emNeeContrib;
                    } else {
                        addSplit(&diffRad, &specRad, throughput * emNeeContrib, cap, i, firstBounceSpec);
                    }
                }
            }
        }

        // Env map NEE (importance-sampled).
        if (HAS_ENV_CDF && rt.envColor.w > 1.5 && h.shininess > 0.01 && i < 4) {
            let envSample = sampleEnvImportance(seed);
            let envDir    = envSample.xyz;
            let envPdf    = envSample.w;
            let envNdotL  = dot(h.normal, envDir);
            if (envNdotL > 0.0 && envPdf > 1e-8) {
                let envAtten = traceShadowRay(h.point, h.normal, envDir, 1e30, 4);
                if (envAtten.x + envAtten.y + envAtten.z > 0.001) {
                    let envCol = sampleEnv(envDir) * rt.envIntensity.x;
                    let pdf_brdf_env = brdfPdf(wo, envDir, h.normal, h.shininess, h.metalness);
                    let w_env = envPdf / max(envPdf + pdf_brdf_env, 1e-8);
                    let cap = rt.emissiveInfo.z;
                    let lobeSum4 = evalBrdfFull(wo, envDir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                                 h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                    let envNeeContrib = envAtten * lobeSum4 * envNdotL * envCol * w_env / envPdf;
                    if (useReSTIRGI && !giResStored) {
                        giLo += envNeeContrib;
                    } else {
                        addSplit(&diffRad, &specRad, throughput * envNeeContrib, cap, i, firstBounceSpec);
                    }
                }
            }
        }
)";

// Second chunk of runBounces: ReSTIR GI block + Russian roulette + lobes + BRDF sample.
const char* const csRunBouncesWGSL2 = R"(
        // ==== ReSTIR GI reservoir (runs only at i==1 when useReSTIRGI) ====
        let giSecDist2 = dot(h.point - b0Point, h.point - b0Point);
        if (useReSTIRGI && !giResStored && giSecDist2 > 0.04) {
            let giP_hat = giTargetPdf(b0Point, b0Normal, b0Wo, b0Albedo, b0Metal, b0Alpha, b0F0,
                                       h.point, h.normal, giLo);
            var giW_sum = giP_hat;
            var giM = 1.0;
            var giSecPos  = h.point;
            var giSecNorm = h.normal;
            var giLoRes   = giLo;
            var giPhat    = giP_hat;

            // Temporal reuse of previous-frame GI reservoir.
            if (rt.frameCount.x > 0.0) {
                let giMeshMoved = b0MeshIdx >= 0 && b0MeshIdx < 128 && isMeshMoved(b0MeshIdx);
                var giReprojPt = b0Point;
                var giExpectedN = b0Normal;
                if (giMeshMoved) {
                    let giMotMat = rtMotionMats[b0MeshIdx];
                    giReprojPt = (giMotMat * vec4<f32>(b0Point, 1.0)).xyz;
                    giExpectedN = normalize((giMotMat * vec4<f32>(b0Normal, 0.0)).xyz);
                }
                let giRelP = giReprojPt - vec3<f32>(rt.prevCamOri.x, rt.prevCamOri.y, rt.prevCamOri.z);
                let giPrevFwd = vec3<f32>(rt.prevCamFwd.x, rt.prevCamFwd.y, rt.prevCamFwd.z);
                let giPrevRgt = vec3<f32>(rt.prevCamRgt.x, rt.prevCamRgt.y, rt.prevCamRgt.z);
                let giPrevUp  = vec3<f32>(rt.prevCamUp.x, rt.prevCamUp.y, rt.prevCamUp.z);
                let giDz = dot(giRelP, giPrevFwd);
                if (giDz > 0.0) {
                    let giDx = dot(giRelP, giPrevRgt);
                    let giDy = dot(giRelP, giPrevUp);
                    let giThf = rt.tanHalfFov.x;
                    let giAspect = rt.iRes.x / rt.iRes.y;
                    let giPrevU = (giDx / (giDz * giThf * giAspect) * 0.5 + 0.5) * rt.iRes.x;
                    let giPrevV = (0.5 - giDy / (giDz * giThf) * 0.5) * rt.iRes.y;
                    let giPrevPx = vec2<i32>(i32(floor(giPrevU)), i32(floor(giPrevV)));

                    if (giPrevPx.x >= 0 && giPrevPx.y >= 0 &&
                        giPrevPx.x < i32(rt.iRes.x) && giPrevPx.y < i32(rt.iRes.y)) {

                        let giPrevSGB = textureLoad(gBufRead, giPrevPx, 0);
                        let giPrevN = giPrevSGB.xyz;
                        let giPrevD = giPrevSGB.w;
                        let giCurD  = length(giRelP);
                        let giPrevMesh = i32(textureLoad(hitMeshRead, giPrevPx, 0).r);
                        let giValid = dot(giExpectedN, giPrevN) > 0.95 &&
                                      abs(giCurD - giPrevD) / max(giCurD, 1e-6) < 0.05 &&
                                      giPrevMesh == b0MeshIdx;

                        if (giValid) {
                            let prevGiSample = textureLoad(giResRead, giPrevPx, 0);
                            let prevGiWeight = textureLoad(giResWRead, giPrevPx, 0);
                            let prevGiLo     = textureLoad(giResLoRead, giPrevPx, 0).xyz;

                            let prevSecPos  = prevGiSample.xyz;
                            let prevSecNorm = unpackOctNormal(prevGiSample.w);
                            let prevW_sum = prevGiWeight.x;
                            let prevM     = min(prevGiWeight.y, 20.0);
                            let prevW     = prevGiWeight.z;
                            let prevPhat  = prevGiWeight.w;

                            let prevSecDist2 = dot(prevSecPos - b0Point, prevSecPos - b0Point);
                            let giPhatPrev = select(0.0,
                                giTargetPdf(b0Point, b0Normal, b0Wo, b0Albedo, b0Metal, b0Alpha, b0F0,
                                            prevSecPos, prevSecNorm, prevGiLo),
                                prevSecDist2 > 0.04);
                            if (giPhatPrev > 0.0 && prevW > 0.0) {
                                let J = reconnJacobian(prevSecPos, prevSecNorm, giReprojPt, b0Point);
                                let giWPrev = giPhatPrev * prevM * prevW * J;
                                giW_sum += giWPrev;
                                giM += prevM;
                                if (rand(seed) < giWPrev / max(giW_sum, 1e-20)) {
                                    giSecPos  = prevSecPos;
                                    giSecNorm = prevSecNorm;
                                    giLoRes   = prevGiLo;
                                    giPhat    = giPhatPrev;
                                }
                            }
                        }
                    }
                }
            }

            // Snapshot pre-spatial reservoir (stored for next-frame temporal reuse).
            let giPreSpW_sum = giW_sum;
            let giPreSpM     = giM;
            let giPreSpPhat  = giPhat;
            let giPreSpSecPos  = giSecPos;
            let giPreSpSecNorm = giSecNorm;
            let giPreSpLoRes   = giLoRes;

            // Spatial reuse from 4 random neighbours (20px disk).
            {
                for (var spI = 0u; spI < 4u; spI++) {
                    let spAngle = rand(seed) * 2.0 * PI;
                    let spR     = sqrt(rand(seed)) * 20.0;
                    let spOff   = vec2<i32>(i32(spR * cos(spAngle)), i32(spR * sin(spAngle)));
                    if (all(spOff == vec2<i32>(0))) { continue; }
                    let spPx = clamp(pixel + spOff,
                                     vec2<i32>(0),
                                     vec2<i32>(i32(rt.iRes.x) - 1, i32(rt.iRes.y) - 1));

                    let spSGB = textureLoad(gBufRead, spPx, 0);
                    let spMesh = i32(textureLoad(hitMeshRead, spPx, 0).r);
                    let b0Depth = length(b0Point - vec3<f32>(rt.camOri.x, rt.camOri.y, rt.camOri.z));
                    if (dot(b0Normal, spSGB.xyz) < 0.95 ||
                        abs(b0Depth - spSGB.w) / max(b0Depth, 1e-3) > 0.05 ||
                        spMesh != b0MeshIdx) { continue; }

                    let spNdc = vec2<f32>(
                        (f32(spPx.x) + 0.5) / rt.iRes.x * 2.0 - 1.0,
                        1.0 - (f32(spPx.y) + 0.5) / rt.iRes.y * 2.0);
                    let spAspect = rt.iRes.x / rt.iRes.y;
                    let spDir = normalize(rt.prevCamFwd.xyz
                        + rt.prevCamRgt.xyz * (spNdc.x * rt.tanHalfFov.x * spAspect)
                        + rt.prevCamUp.xyz  * (spNdc.y * rt.tanHalfFov.x));
                    let spPrimary = rt.prevCamOri.xyz + spDir * spSGB.w;

                    let spGiSample = textureLoad(giResRead, spPx, 0);
                    let spGiWeight = textureLoad(giResWRead, spPx, 0);
                    let spGiLo     = textureLoad(giResLoRead, spPx, 0).xyz;

                    let spSecPos  = spGiSample.xyz;
                    let spSecNorm = unpackOctNormal(spGiSample.w);
                    let spM = min(spGiWeight.y, 4.0);
                    let spW = spGiWeight.z;
                    if (spW <= 0.0 || spM <= 0.0) { continue; }

                    let spSecDist2 = dot(spSecPos - b0Point, spSecPos - b0Point);
                    if (spSecDist2 < 0.04) { continue; }

                    let spGiPhat = giTargetPdf(b0Point, b0Normal, b0Wo, b0Albedo, b0Metal, b0Alpha, b0F0,
                                                spSecPos, spSecNorm, spGiLo);
                    if (spGiPhat > 0.0) {
                        let J = reconnJacobian(spSecPos, spSecNorm, spPrimary, b0Point);
                        let spGiW = spGiPhat * spM * spW * J;
                        giW_sum += spGiW;
                        giM += spM;
                        if (rand(seed) < spGiW / max(giW_sum, 1e-20)) {
                            giSecPos  = spSecPos;
                            giSecNorm = spSecNorm;
                            giLoRes   = spGiLo;
                            giPhat    = spGiPhat;
                        }
                    }
                }
            }
)"
R"(
            // Finalize + visibility + shade + store pre-spatial reservoir.
            var giW = select(0.0, giW_sum / max(giM * giPhat, 1e-20), giPhat > 0.0);
            giW = min(giW, 3.0);

            let giWi = normalize(giSecPos - b0Point);
            let giNdotL = max(dot(b0Normal, giWi), 0.0);
            var giVisAtten = vec3<f32>(0.0);
            if (giNdotL > 0.0 && giW > 0.0) {
                let giDist = length(giSecPos - b0Point);
                giVisAtten = traceShadowRay(b0Point, b0Normal, giWi, giDist - 1e-3, 4);
            }

            if (giVisAtten.x + giVisAtten.y + giVisAtten.z > 0.001 && giNdotL > 0.0) {
                let giDelta = giSecPos - b0Point;
                let giDist2 = dot(giDelta, giDelta);
                let giCosTheta2 = max(dot(giSecNorm, -giWi), 0.0);
                let giG = giCosTheta2 / max(giDist2, 0.01);
                let giBrdf = evalBrdfFullSplit(b0Wo, giWi, b0Normal, b0Albedo, b0Metal, b0Alpha, b0F0,
                                                vec3<f32>(0.0), 0.0, 0.0, 0.0);
                let giShade = giVisAtten * giW * giNdotL * giLoRes * giG;
                let giCap = rt.emissiveInfo.z;
                var giContribD = giShade * giBrdf.diff;
                var giContribS = giShade * giBrdf.spec;
                let giLum = luminance(giContribD + giContribS);
                let giHardCap = 2.0;
                if (giLum > giHardCap) {
                    let giScale = giHardCap / giLum;
                    giContribD *= giScale;
                    giContribS *= giScale;
                }
                addSplit(&diffRad, &specRad, giContribD, giCap, 1, false);
                addSplit(&diffRad, &specRad, giContribS, giCap, 1, true);
            }

            let giVisible = giVisAtten.x + giVisAtten.y + giVisAtten.z > 0.001;
            let giStoreW_f = select(0.0, select(0.0, giPreSpW_sum / max(giPreSpM * giPreSpPhat, 1e-20), giPreSpPhat > 0.0), giVisible);
            let giStoreM_f = select(0.0, giPreSpM, giVisible);
            textureStore(giResWrite,   pixel, vec4<f32>(giPreSpSecPos, packOctNormal(giPreSpSecNorm)));
            textureStore(giResWWrite,  pixel, vec4<f32>(giPreSpW_sum, giStoreM_f, min(giStoreW_f, 3.0), giPreSpPhat));
            textureStore(giResLoWrite, pixel, vec4<f32>(giPreSpLoRes, 0.0));
            giResStored = true;
        }
        // GI fallback — add captured Lo back if reservoir was skipped (close hit).
        if (useReSTIRGI && !giResStored && luminance(giLo) > 0.0) {
            addSplit(&diffRad, &specRad, throughput * giLo, rt.emissiveInfo.z, 1, firstBounceSpec);
        }

        // Russian roulette (i > 0 branch — always true in runBounces).
        {
            let p_base = max(max(throughput.r, throughput.g), throughput.b);
            let rough  = sqrt(h.shininess);
            let weight = mix(0.25, 1.0, 1.0 - rough);
            let p = clamp(p_base * weight, 0.02, 1.0);
            if (rand(seed) > p) { break; }
            throughput /= p;
        }

        // Clearcoat lobe.
        if (h.clearcoat > 0.0 && h.transmission < 0.01) {
            let ccF0 = 0.04;
            let ccCos = max(0.0, dot(wo, h.normal));
            let ccFresnel = ccF0 + (1.0 - ccF0) * pow(1.0 - ccCos, 5.0);
            let ccWeight = h.clearcoat * ccFresnel;
            let ccProb = max(ccWeight, 0.15 * h.clearcoat);
            if (rand(seed) < ccProb) {
                let wi_cc = sampleVNDF(wo, h.normal, h.clearcoatAlpha, seed);
                let cos_cc = dot(h.normal, wi_cc);
                if (cos_cc <= 0.0) { break; }
                let G1_cc = ggxG1(cos_cc, h.clearcoatAlpha);
                throughput *= vec3<f32>(ccWeight * G1_cc / ccProb);
                prevWo        = wo;
                prevNormal    = h.normal;
                prevAlpha     = h.clearcoatAlpha;
                prevMetalness = 0.0;
                afterTransmission = false;
                ray.origin = h.point + h.normal * 1e-3;
                ray.dir    = wi_cc;
                continue;
            }
            throughput *= (1.0 - ccWeight) / (1.0 - ccProb);
        }

        // Sheen.
        let sheenLumPT = dot(h.sheenColor, vec3<f32>(0.2126, 0.7152, 0.0722));
        if (sheenLumPT > 0.001 && h.transmission < 0.01) {
            let sheenAlpha = max(h.sheenRoughness * h.sheenRoughness, 1e-4);
            let NdotV_sh = max(0.001, dot(wo, h.normal));
            let sheenFresnel = sheenLumPT * pow(1.0 - NdotV_sh, 3.0);
            throughput *= (1.0 - sheenFresnel);
        }

        // Transmission.
        if (h.transmission > 0.0 && rand(seed) < h.transmission) {
            let entering = h.frontFace > 0.5;
            let tMat4 = textureLoad(matData, vec2<i32>(h.matIdx, 4), 0);
            let tAttColor = tMat4.xyz;
            let tAttDist  = tMat4.w;
            var channelMask = vec3<f32>(1.0);
            var ior_eff = h.ior;
            if (h.dispersion > 0.0) {
                let lambda = array<f32, 3>(0.6563, 0.55, 0.4861);
                let ref_inv_sq = 1.0 / (0.5893 * 0.5893);
                let ch = u32(rand(seed) * 3.0) % 3u;
                let inv_sq = 1.0 / (lambda[ch] * lambda[ch]);
                let B = (h.ior - 1.0) * h.dispersion / 38.2;
                ior_eff = h.ior + B * (inv_sq - ref_inv_sq);
                channelMask = vec3<f32>(0.0);
                channelMask[ch] = 3.0;
            }
            let eta = select(ior_eff, 1.0 / ior_eff, entering);
            var tNorm = h.normal;
            var usedMicrofacet = false;
            if (h.shininess > 1e-3) {
                let wo_t = normalize(-ray.dir);
                let hm = sampleVNDF(wo_t, h.normal, h.shininess, seed);
                if (dot(hm, h.normal) > 0.0) { tNorm = hm; usedMicrofacet = true; }
            }
            let cosI = abs(dot(normalize(ray.dir), tNorm));
            let r0   = pow((1.0 - ior_eff) / (1.0 + ior_eff), 2.0);
            let sin2I = max(0.0, 1.0 - cosI * cosI);
            let cosSchlick = select(cosI,
                sqrt(max(0.0, 1.0 - ior_eff * ior_eff * sin2I)),
                !entering);
            let fresnel = r0 + (1.0 - r0) * pow(1.0 - cosSchlick, 5.0);
            var wi_t: vec3<f32>;
            var didRefract = false;
            if (rand(seed) < fresnel) {
                wi_t = reflect(ray.dir, tNorm);
                ray.origin = h.point + h.normal * 1e-3;
            } else {
                let refracted = refract(normalize(ray.dir), tNorm, eta);
                if (length(refracted) < 0.001) {
                    wi_t = reflect(ray.dir, tNorm);
                    ray.origin = h.point + h.normal * 1e-3;
                } else {
                    wi_t = refracted;
                    ray.origin = h.point - h.normal * 1e-3;
                    didRefract = true;
                }
            }
            var microWeight = 1.0;
            if (usedMicrofacet) {
                let cosOut = abs(dot(wi_t, h.normal));
                microWeight = ggxG1(cosOut, h.shininess);
            }
            if (didRefract) {
                let glassTint = mix(albedo, vec3<f32>(1.0), h.transmission) * microWeight;
                var volAtten = vec3<f32>(1.0);
                if (tAttDist > 0.0 && !entering) {
                    let absorbCoeff = -log(max(tAttColor, vec3<f32>(1e-6))) / tAttDist;
                    let pathLen = select(h.t, h.thickness, h.t < 1e-2 && h.thickness > 0.0);
                    volAtten = exp(-absorbCoeff * pathLen);
                }
                throughput *= glassTint * volAtten / (eta * eta) * channelMask;
            } else {
                throughput *= vec3<f32>(microWeight) * channelMask;
            }
            afterTransmission = true;
            ray.dir = wi_t;
            continue;
        }

        // BRDF sample — firstBounceSpec is i==0-only in the original, so no update here.
        let F0_b = F0_h;
        var wi_b: vec3<f32>;
        let p_spec = mix(0.5, 0.98, h.metalness);
        let isSpecBounce = rand(seed) < p_spec;
        if (isSpecBounce) {
            wi_b = sampleVNDF(wo, h.normal, h.shininess, seed);
            let cos_b = dot(h.normal, wi_b);
            if (cos_b <= 0.0) { break; }
            let hb  = normalize(wo + wi_b);
            let Fb  = schlick(max(0.0, dot(wo, hb)), F0_b);
            let G1L = ggxG1(cos_b, h.shininess);
            throughput *= Fb * G1L / p_spec;
        } else {
            wi_b = cosineHemisphere(h.normal, seed);
            let cos_b = dot(h.normal, wi_b);
            if (cos_b <= 0.0) { break; }
            throughput *= albedo * (1.0 - h.metalness) / (1.0 - p_spec);
        }
        prevWo        = wo;
        prevNormal    = h.normal;
        prevAlpha     = h.shininess;
        prevMetalness = h.metalness;
        afterTransmission = false;
        ray.origin = h.point + h.normal * 1e-3;
        ray.dir    = wi_b;
    }

    // Safety net: write empty GI reservoir if none of the paths above stored one.
    if (rt.spp.x > 0.5 && !giResStored) {
        textureStore(giResWrite,   pixel, vec4<f32>(0.0));
        textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
        textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
    }

    return SplitRadiance(diffRad, specRad);
}
)";

// primaryShade (bounce split, Stage B): the i==0 iteration of the original
// pathTrace loop extracted as a standalone function.  Responsibilities:
//   - Consume the pre-computed primary hit from the primary kernel.
//   - Handle primary miss (background), emissive, unlit, and adaptive bounce cap.
//   - Capture bounce-0 surface state for ReSTIR GI (b0*).
//   - Run ReSTIR DI (or classic NEE at i==0) + env NEE.
//   - Sample the first BRDF bounce (firstBounceSpec, outgoing ray).
// `continue` in the original loop becomes "return pathAlive=true with outgoing
// ray/throughput set"; `break` becomes "return pathAlive=false".  The caller
// (pathTrace wrapper) either returns primaryShade's radiance directly (dead) or
// invokes runBounces to continue the path.
// Split across three R-literals for the MSVC 16380-byte limit.
const char* const csPrimaryShadeWGSL1 = R"(
fn primaryShade(primaryHit:     Hit,
                ray_in:         Ray,
                seed:           ptr<function, u32>,
                pixel:          vec2<i32>,
                maxBounces:     i32,
                primaryMeshIdx: ptr<function, u32>,
                primaryNormal:  ptr<function, vec3<f32>>,
                primaryDepth:   ptr<function, f32>,
                primaryAlbedo:  ptr<function, vec3<f32>>,
                primaryRough:   ptr<function, f32>,
                primaryMatIdx:  ptr<function, i32>,
                primaryTriIdx:  ptr<function, i32>) -> PrimaryShadeResult {
    // primary* out-param defaults (match original pathTrace for sky/miss case).
    *primaryMeshIdx = 128u;
    *primaryNormal  = vec3<f32>(0.0);
    *primaryDepth   = 0.0;
    *primaryAlbedo  = vec3<f32>(1.0);
    *primaryRough   = 1.0;
    *primaryMatIdx  = -1;
    *primaryTriIdx  = -1;

    var result: PrimaryShadeResult;
    result.rayOrigin        = ray_in.origin;
    result.rayDir           = ray_in.dir;
    result.throughput       = vec3<f32>(1.0);
    result.diffRad          = vec3<f32>(0.0);
    result.specRad          = vec3<f32>(0.0);
    result.prevNormal       = vec3<f32>(0.0, 1.0, 0.0);
    result.prevWo           = vec3<f32>(0.0);
    result.prevAlpha        = 0.0;
    result.prevMetalness    = 0.0;
    result.afterTransmission = false;
    result.firstBounceSpec  = false;
    result.effectiveBounces = maxBounces;
    result.b0Point   = vec3<f32>(0.0);
    result.b0Normal  = vec3<f32>(0.0, 1.0, 0.0);
    result.b0Wo      = vec3<f32>(0.0);
    result.b0Albedo  = vec3<f32>(1.0);
    result.b0F0      = vec3<f32>(0.04);
    result.b0Metal   = 0.0;
    result.b0Alpha   = 1.0;
    result.b0MeshIdx = -1;
    result.giResStored = false;
    result.pathAlive   = false;
    result.primaryDepth  = 0.0;   // sentinel for sky/miss (matches *primaryDepth default)
    result.primaryMatIdx = -1;

    var ray        = ray_in;
    var throughput = vec3<f32>(1.0);
    var diffRad    = vec3<f32>(0.0);
    var specRad    = vec3<f32>(0.0);

    let h = primaryHit;
    if (h.t >= 1e29) {
        // Primary ray miss: show background.
        diffRad += throughput * sampleBackground(ray.dir);
        if (rt.restirParams.x > 0.5) {
            textureStore(reservoirWrite,  pixel, vec4<f32>(0.0));
            textureStore(reservoirWWrite, pixel, vec4<f32>(0.0));
        }
        if (rt.spp.x > 0.5) {
            textureStore(giResWrite,   pixel, vec4<f32>(0.0));
            textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
            textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
            result.giResStored = true;
        }
        result.diffRad = diffRad;
        result.specRad = specRad;
        result.pathAlive = false;
        return result;
    }

    // Primary hit: fill out-params + adaptive bounce cap.
    *primaryMeshIdx = u32(h.meshIdx);
    *primaryNormal  = h.normal;
    *primaryDepth   = h.t;
    *primaryRough   = sqrt(h.shininess);
    *primaryMatIdx  = h.matIdx;
    *primaryTriIdx  = h.triIdx;
    // Mirror primary metadata into result so rt_bounces_main reads without out-params.
    result.primaryDepth  = h.t;
    result.primaryMatIdx = h.matIdx;
    // Capture b0MeshIdx/b0Alpha/b0Point/b0Normal EARLY so unlit early-exit paths
    // still propagate valid primary-hit metadata to the accumulation stage.  The
    // other b0* fields (b0Wo/b0Albedo/b0F0/b0Metal) depend on wo/albedo computed
    // below and are captured there.  Unlit paths don't need them — runBounces
    // isn't called, and the accumulation stage only reads b0Point/b0Normal/b0Alpha.
    result.b0MeshIdx = h.meshIdx;
    result.b0Alpha   = h.shininess;
    result.b0Point   = h.point;
    result.b0Normal  = h.normal;

    var effectiveBounces = maxBounces;
    {
        let isGlass  = h.transmission > 0.05;
        let isMetal  = h.metalness > 0.5;
        let isMirror = h.shininess < 0.05;
        let isGlossy = h.shininess < 0.25;
        if (!isGlass && !isMetal && !isMirror) {
            if (isGlossy) {
                effectiveBounces = min(maxBounces, 4);
            } else {
                effectiveBounces = min(maxBounces, 3);
            }
        }
    }
    result.effectiveBounces = effectiveBounces;

    let emTriCount = i32(rt.emissiveInfo.x);
    let totalPower = rt.emissiveInfo.y;

    // Emissive at bounce 0 — full weight, no clamp (primary ray is physically correct).
    if (length(h.emissive) > 0.0) {
        diffRad += throughput * h.emissive;
    }

    var albedo = h.albedo;
    if (h.texSlot >= 0.0) { albedo = srgbToLinear(sampleAtlas(h.uv, h.texSlot)); }
    *primaryAlbedo = albedo;

    // Unlit: flat colour, no bouncing.
    if (h.shininess < 0.0) {
        diffRad += throughput * albedo;
        if (rt.restirParams.x > 0.5) {
            textureStore(reservoirWrite,  pixel, vec4<f32>(0.0));
            textureStore(reservoirWWrite, pixel, vec4<f32>(0.0));
        }
        if (rt.spp.x > 0.5) {
            textureStore(giResWrite,   pixel, vec4<f32>(0.0));
            textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
            textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
            result.giResStored = true;
        }
        result.diffRad = diffRad;
        result.specRad = specRad;
        result.pathAlive = false;
        return result;
    }

    let wo   = normalize(-ray.dir);
    let F0_h = computeF0(albedo, h.metalness, h.specularColor, h.specularIntensity);

    // Capture bounce-0 surface fields that require wo/F0 (remaining b0* captures;
    // b0MeshIdx/b0Alpha/b0Point/b0Normal were set earlier for unlit compatibility).
    result.b0Wo      = wo;
    result.b0Albedo  = albedo;
    result.b0F0      = F0_h;
    result.b0Metal   = h.metalness;

    let lcount    = i32(rt.lightCount.x);
    // ReSTIR DI gating at bounce 0: skip for transmissive and near-mirror surfaces.
    let useReSTIR = rt.restirParams.x > 0.5 && h.transmission < 0.05;

    if (useReSTIR) {
        // ======= ReSTIR DI: Initial candidate generation =======
        var reservoir = emptyReservoir();

        // 1. Analytical lights
        {
            let p_source_a = 1.0 / max(f32(lcount), 1.0);
            for (var li = 0; li < 4; li++) {
                if (li >= lcount) { break; }
                let le = evalAnalyticalLight(li, h.point);
                let lightP = rt.lightPos[li].xyz;
                let p_hat_a = restirTargetPdf(h.point, h.normal, wo, albedo, h.metalness, h.shininess, F0_h,
                                              lightP, f32(li), le.color);
                reservoir.M += 1.0;
                if (p_hat_a > 0.0) {
                    let w = p_hat_a / p_source_a;
                    reservoir.W_sum += w;
                    if (rand(seed) < w / max(reservoir.W_sum, 1e-20)) {
                        reservoir.lightPos  = lightP;
                        reservoir.lightType = f32(li);
                        reservoir.p_hat     = p_hat_a;
                    }
                }
            }
        }

        // 2. Emissive triangles — CDF samples
        if (emTriCount > 0 && totalPower > 0.0) {
            for (var ei = 0; ei < 4; ei++) {
                let es = sampleEmissiveTriCdf(seed, totalPower, emTriCount);
                let toLight = es.point - h.point;
                let dist = length(toLight);
                let ln_e = toLight / dist;
                let cosLight = abs(dot(es.normal, -ln_e));
                let NdotL_e = dot(h.normal, ln_e);
                reservoir.M += 1.0;
                if (NdotL_e > 0.0 && cosLight > 1e-6) {
                    let eMatIdx = i32(textureLoad(triData, triCoord(es.triIdx, 0), 0).w);
                    let emColor = textureLoad(matData, vec2<i32>(eMatIdx, 2), 0).xyz;
                    let p_hat_e = restirTargetPdf(h.point, h.normal, wo, albedo, h.metalness, h.shininess, F0_h,
                                                  es.point, 1000.0 + f32(es.triIdx), emColor);
                    if (p_hat_e > 0.0) {
                        let p_source_e = (es.power / totalPower) * (dist * dist) / (es.area * cosLight);
                        let w = p_hat_e / p_source_e;
                        reservoir.W_sum += w;
                        if (rand(seed) < w / max(reservoir.W_sum, 1e-20)) {
                            reservoir.lightPos  = es.point;
                            reservoir.lightType = 1000.0 + f32(es.triIdx);
                            reservoir.p_hat     = p_hat_e;
                        }
                    }
                }
            }
        }

        // 3. Env — NOT sampled via reservoir (handled unconditionally by env NEE below).
        finalizeReservoir(&reservoir);
)"
R"(
        // === TEMPORAL REUSE ===
        if (rt.frameCount.x > 0.0) {
            let primaryMoved = h.meshIdx >= 0 && h.meshIdx < 128 && isMeshMoved(h.meshIdx);
            var reprojPoint = h.point;
            var expectedPrevN = h.normal;
            if (primaryMoved) {
                let M = rtMotionMats[h.meshIdx];
                reprojPoint = (M * vec4<f32>(h.point, 1.0)).xyz;
                expectedPrevN = normalize((M * vec4<f32>(h.normal, 0.0)).xyz);
            }
            let relP = reprojPoint - vec3<f32>(rt.prevCamOri.x, rt.prevCamOri.y, rt.prevCamOri.z);
            let prevFwd = vec3<f32>(rt.prevCamFwd.x, rt.prevCamFwd.y, rt.prevCamFwd.z);
            let prevRgt = vec3<f32>(rt.prevCamRgt.x, rt.prevCamRgt.y, rt.prevCamRgt.z);
            let prevUp  = vec3<f32>(rt.prevCamUp.x, rt.prevCamUp.y, rt.prevCamUp.z);
            let dz = dot(relP, prevFwd);
            if (dz > 0.0) {
                let dx = dot(relP, prevRgt);
                let dy = dot(relP, prevUp);
                let thf = rt.tanHalfFov.x;
                let aspect = rt.iRes.x / rt.iRes.y;
                let ndcX = dx / (dz * thf * aspect);
                let ndcY = dy / (dz * thf);
                let prevU = (ndcX * 0.5 + 0.5) * rt.iRes.x;
                let prevV = (0.5 - ndcY * 0.5) * rt.iRes.y;
                let prevPx = vec2<i32>(i32(floor(prevU)), i32(floor(prevV)));

                if (prevPx.x >= 0 && prevPx.y >= 0 &&
                    prevPx.x < i32(rt.iRes.x) && prevPx.y < i32(rt.iRes.y)) {

                    let prevSGB = textureLoad(gBufRead, prevPx, 0);
                    let prevN = prevSGB.xyz;
                    let prevD = prevSGB.w;
                    let curD = length(relP);
                    let valid = dot(expectedPrevN, prevN) > 0.9 &&
                                abs(curD - prevD) / max(curD, 1e-6) < 0.1;

                    if (valid) {
                        let prevSample = textureLoad(reservoirRead, prevPx, 0);
                        let prevWeight = textureLoad(reservoirWRead, prevPx, 0);

                        var rPrev: Reservoir;
                        rPrev.lightPos  = prevSample.xyz;
                        rPrev.lightType = prevSample.w;
                        rPrev.W_sum = prevWeight.x;
                        rPrev.M     = min(prevWeight.y, rt.restirParams.y) * 0.8;
                        rPrev.W     = prevWeight.z;
                        rPrev.p_hat = prevWeight.w;

                        let prevLe = evalLightRadiance(rPrev.lightPos, rPrev.lightType, h.point);
                        let p_hat_prev = restirTargetPdf(h.point, h.normal, wo, albedo,
                                                         h.metalness, h.shininess, F0_h,
                                                         rPrev.lightPos, rPrev.lightType, prevLe);
                        if (p_hat_prev > 0.0 && rPrev.W > 0.0) {
                            let w_prev = p_hat_prev * rPrev.M * rPrev.W;
                            reservoir.W_sum += w_prev;
                            reservoir.M += rPrev.M;
                            if (rand(seed) < w_prev / max(reservoir.W_sum, 1e-20)) {
                                reservoir.lightPos  = rPrev.lightPos;
                                reservoir.lightType = rPrev.lightType;
                                reservoir.p_hat     = p_hat_prev;
                            }
                            finalizeReservoir(&reservoir);
                        }
                    }
                }
            }
        }

        // Snapshot reservoir before spatial reuse.
        let preSpReservoir = reservoir;

        // === SPATIAL REUSE — random neighbours from previous frame ===
        {
            let camMoving = (u32(rt.params.w) & 2u) != 0u;
            let spMax = select(5u, 2u, camMoving);
            let mTarget = 20.0;
            for (var spI = 0u; spI < spMax; spI++) {
                if (reservoir.M >= mTarget) { break; }
                let spAngle = rand(seed) * 2.0 * PI;
                let spR     = sqrt(rand(seed)) * 20.0;
                let spOff   = vec2<i32>(i32(spR * cos(spAngle)), i32(spR * sin(spAngle)));
                if (all(spOff == vec2<i32>(0))) { continue; }
                let spPx = clamp(pixel + spOff,
                                 vec2<i32>(0),
                                 vec2<i32>(i32(rt.iRes.x) - 1, i32(rt.iRes.y) - 1));

                let spSGB = textureLoad(gBufRead, spPx, 0);
                if (dot(h.normal, spSGB.xyz) < 0.906 ||
                    abs(h.t - spSGB.w) / max(h.t, 1e-3) > 0.1) { continue; }

                let spSmp = textureLoad(reservoirRead,  spPx, 0);
                let spWt  = textureLoad(reservoirWRead, spPx, 0);
                var rSp: Reservoir;
                rSp.lightPos  = spSmp.xyz;
                rSp.lightType = spSmp.w;
                rSp.W_sum = spWt.x;
                rSp.M     = min(spWt.y, 4.0);
                rSp.W     = spWt.z;
                rSp.p_hat = spWt.w;
                if (rSp.W <= 0.0 || rSp.M <= 0.0) { continue; }

                let spLe = evalLightRadiance(rSp.lightPos, rSp.lightType, h.point);
                let p_hat_sp = restirTargetPdf(h.point, h.normal, wo, albedo,
                                                h.metalness, h.shininess, F0_h,
                                                rSp.lightPos, rSp.lightType, spLe);
                if (p_hat_sp > 0.0) {
                    let w_sp = p_hat_sp * rSp.M * rSp.W;
                    reservoir.W_sum += w_sp;
                    reservoir.M     += rSp.M;
                    if (rand(seed) < w_sp / max(reservoir.W_sum, 1e-20)) {
                        reservoir.lightPos  = rSp.lightPos;
                        reservoir.lightType = rSp.lightType;
                        reservoir.p_hat     = p_hat_sp;
                    }
                }
            }
            finalizeReservoir(&reservoir);
        }
        reservoir.W = min(reservoir.W, 5.0);

        // === VISIBILITY TEST ===
        var reservoirShadowAtten = vec3<f32>(0.0);
        if (reservoir.p_hat > 0.0 && reservoir.W > 0.0) {
            let rd = reservoirLightDir(reservoir.lightPos, reservoir.lightType, h.point);
            reservoirShadowAtten = traceShadowRay(h.point, h.normal, rd.dir, rd.maxDist, 4);
        }

        // === SHADE FROM RESERVOIR (split diffuse/specular) ===
        if (reservoirShadowAtten.x + reservoirShadowAtten.y + reservoirShadowAtten.z > 0.001) {
            let rd = reservoirLightDir(reservoir.lightPos, reservoir.lightType, h.point);
            let rLe = evalLightRadiance(reservoir.lightPos, reservoir.lightType, h.point);
            let rNdotL = max(dot(h.normal, rd.dir), 0.0);
            let rSplit = evalBrdfFullSplit(wo, rd.dir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                          h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
            let cap = rt.emissiveInfo.z;
            let shade = throughput * reservoirShadowAtten * reservoir.W * rNdotL * rLe;
            addSplit(&diffRad, &specRad, shade * rSplit.diff, cap, 0, false);
            addSplit(&diffRad, &specRad, shade * rSplit.spec, cap, 1, true);
        }

        // === STORE RESERVOIR (pre-spatial, visibility-gated) ===
        let visible = reservoirShadowAtten.x + reservoirShadowAtten.y + reservoirShadowAtten.z > 0.001;
        let rW = select(0.0, select(0.0, preSpReservoir.W, preSpReservoir.W == preSpReservoir.W), visible);
        let rM = select(0.0, preSpReservoir.M, visible);
        textureStore(reservoirWrite, pixel,
            vec4<f32>(preSpReservoir.lightPos, preSpReservoir.lightType));
        textureStore(reservoirWWrite, pixel,
            vec4<f32>(preSpReservoir.W_sum, rM, rW, preSpReservoir.p_hat));

    } else {
)";

// Second chunk of primaryShade: classic NEE (non-ReSTIR branch) + env NEE + reservoir zero-out.
const char* const csPrimaryShadeWGSL2 = R"(
        // ======= Classic NEE (ReSTIR disabled or skipped) =======
        // Analytical lights
        for (var li = 0; li < 4; li++) {
            if (li >= lcount) { break; }
            let le = evalAnalyticalLight(li, h.point);
            let NdotL = dot(h.normal, le.dir);
            if (NdotL <= 0.0) { continue; }
            let shadowAtten = traceShadowRay(h.point, h.normal, le.dir, le.dist - 1e-3, 4);
            if (shadowAtten.x + shadowAtten.y + shadowAtten.z > 0.001) {
                let cap = rt.emissiveInfo.z;
                let bs = evalBrdfFullSplit(wo, le.dir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                          h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                let shade = throughput * shadowAtten * NdotL * le.color;
                addSplit(&diffRad, &specRad, shade * bs.diff, cap, 0, false);
                addSplit(&diffRad, &specRad, shade * bs.spec, cap, 1, true);
            }
        }

        // Emissive surface NEE
        if (emTriCount > 0) {
            let totalPower2 = rt.emissiveInfo.y;
            let es = sampleEmissiveTriCdf(seed, totalPower2, emTriCount);
            let toLight = es.point - h.point;
            let dist    = length(toLight);
            let ln      = toLight / dist;
            let NdotL   = dot(h.normal, ln);
            let cosLight = abs(dot(es.normal, -ln));
            if (NdotL > 0.0 && cosLight > 1e-6) {
                let emAtten = traceShadowRay(h.point, h.normal, ln, dist - 1e-2, 4);
                if (emAtten.x + emAtten.y + emAtten.z > 0.001) {
                    let eMatIdx = i32(textureLoad(triData, triCoord(es.triIdx, 0), 0).w);
                    let emColor = textureLoad(matData, vec2<i32>(eMatIdx, 2), 0).xyz;
                    let pdf = (es.power * dist * dist) / (totalPower2 * es.area * cosLight);
                    let pdf_brdf_nee = brdfPdf(wo, ln, h.normal, h.shininess, h.metalness);
                    let w_light = pdf / max(pdf + pdf_brdf_nee, 1e-8);
                    let cap = rt.emissiveInfo.z;
                    let bs = evalBrdfFullSplit(wo, ln, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                               h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                    let shade = throughput * emAtten * NdotL * emColor * w_light / pdf;
                    addSplit(&diffRad, &specRad, shade * bs.diff, cap, 0, false);
                    addSplit(&diffRad, &specRad, shade * bs.spec, cap, 1, true);
                }
            }
        }

    } // end ReSTIR vs classic NEE

    // Env NEE (always runs at bounce 0 regardless of ReSTIR).
    if (HAS_ENV_CDF && rt.envColor.w > 1.5 && h.shininess > 0.01) {
        let envSample = sampleEnvImportance(seed);
        let envDir    = envSample.xyz;
        let envPdf    = envSample.w;
        let envNdotL  = dot(h.normal, envDir);
        if (envNdotL > 0.0 && envPdf > 1e-8) {
            let envAtten = traceShadowRay(h.point, h.normal, envDir, 1e30, 4);
            if (envAtten.x + envAtten.y + envAtten.z > 0.001) {
                let envCol = sampleEnv(envDir) * rt.envIntensity.x;
                let pdf_brdf_env = brdfPdf(wo, envDir, h.normal, h.shininess, h.metalness);
                let w_env = envPdf / max(envPdf + pdf_brdf_env, 1e-8);
                let cap = rt.emissiveInfo.z;
                let bs = evalBrdfFullSplit(wo, envDir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                           h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                let shade = throughput * envAtten * envNdotL * envCol * w_env / envPdf;
                addSplit(&diffRad, &specRad, shade * bs.diff, cap, 0, false);
                addSplit(&diffRad, &specRad, shade * bs.spec, cap, 1, true);
            }
        }
    }

    // Zero out reservoir when ReSTIR was globally enabled but skipped (glass/etc).
    if (rt.restirParams.x > 0.5 && !useReSTIR) {
        textureStore(reservoirWrite,  pixel, vec4<f32>(0.0));
        textureStore(reservoirWWrite, pixel, vec4<f32>(0.0));
    }

    // NOTE: ReSTIR GI block (i==1) and Russian roulette (i>0) are NOT present at bounce 0.

    // Clearcoat lobe.
    if (h.clearcoat > 0.0 && h.transmission < 0.01) {
        let ccF0 = 0.04;
        let ccCos = max(0.0, dot(wo, h.normal));
        let ccFresnel = ccF0 + (1.0 - ccF0) * pow(1.0 - ccCos, 5.0);
        let ccWeight = h.clearcoat * ccFresnel;
        let ccProb = max(ccWeight, 0.15 * h.clearcoat);
        if (rand(seed) < ccProb) {
            let wi_cc = sampleVNDF(wo, h.normal, h.clearcoatAlpha, seed);
            let cos_cc = dot(h.normal, wi_cc);
            if (cos_cc <= 0.0) {
                // Original: `break` — safety net writes empty GI.
                if (rt.spp.x > 0.5 && !result.giResStored) {
                    textureStore(giResWrite,   pixel, vec4<f32>(0.0));
                    textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
                    textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
                    result.giResStored = true;
                }
                result.diffRad = diffRad;
                result.specRad = specRad;
                result.pathAlive = false;
                return result;
            }
            let G1_cc = ggxG1(cos_cc, h.clearcoatAlpha);
            throughput *= vec3<f32>(ccWeight * G1_cc / ccProb);
            result.prevWo        = wo;
            result.prevNormal    = h.normal;
            result.prevAlpha     = h.clearcoatAlpha;
            result.prevMetalness = 0.0;
            result.afterTransmission = false;
            ray.origin = h.point + h.normal * 1e-3;
            ray.dir    = wi_cc;
            // Original `continue` — hand off to runBounces with outgoing ray set.
            result.rayOrigin  = ray.origin;
            result.rayDir     = ray.dir;
            result.throughput = throughput;
            result.diffRad    = diffRad;
            result.specRad    = specRad;
            result.pathAlive  = true;
            return result;
        }
        throughput *= (1.0 - ccWeight) / (1.0 - ccProb);
    }

    // Sheen.
    let sheenLumPT = dot(h.sheenColor, vec3<f32>(0.2126, 0.7152, 0.0722));
    if (sheenLumPT > 0.001 && h.transmission < 0.01) {
        let sheenAlpha = max(h.sheenRoughness * h.sheenRoughness, 1e-4);
        let NdotV_sh = max(0.001, dot(wo, h.normal));
        let sheenFresnel = sheenLumPT * pow(1.0 - NdotV_sh, 3.0);
        throughput *= (1.0 - sheenFresnel);
    }
)";

// Third chunk of primaryShade: transmission lobe + BRDF sample + final return.
const char* const csPrimaryShadeWGSL3 = R"(
    // Transmission lobe.
    if (h.transmission > 0.0 && rand(seed) < h.transmission) {
        let entering = h.frontFace > 0.5;
        let tMat4 = textureLoad(matData, vec2<i32>(h.matIdx, 4), 0);
        let tAttColor = tMat4.xyz;
        let tAttDist  = tMat4.w;
        var channelMask = vec3<f32>(1.0);
        var ior_eff = h.ior;
        if (h.dispersion > 0.0) {
            let lambda = array<f32, 3>(0.6563, 0.55, 0.4861);
            let ref_inv_sq = 1.0 / (0.5893 * 0.5893);
            let ch = u32(rand(seed) * 3.0) % 3u;
            let inv_sq = 1.0 / (lambda[ch] * lambda[ch]);
            let B = (h.ior - 1.0) * h.dispersion / 38.2;
            ior_eff = h.ior + B * (inv_sq - ref_inv_sq);
            channelMask = vec3<f32>(0.0);
            channelMask[ch] = 3.0;
        }
        let eta = select(ior_eff, 1.0 / ior_eff, entering);
        var tNorm = h.normal;
        var usedMicrofacet = false;
        if (h.shininess > 1e-3) {
            let wo_t = normalize(-ray.dir);
            let hm = sampleVNDF(wo_t, h.normal, h.shininess, seed);
            if (dot(hm, h.normal) > 0.0) { tNorm = hm; usedMicrofacet = true; }
        }
        let cosI = abs(dot(normalize(ray.dir), tNorm));
        let r0   = pow((1.0 - ior_eff) / (1.0 + ior_eff), 2.0);
        let sin2I = max(0.0, 1.0 - cosI * cosI);
        let cosSchlick = select(cosI,
            sqrt(max(0.0, 1.0 - ior_eff * ior_eff * sin2I)),
            !entering);
        let fresnel = r0 + (1.0 - r0) * pow(1.0 - cosSchlick, 5.0);
        var wi_t: vec3<f32>;
        var didRefract = false;
        if (rand(seed) < fresnel) {
            wi_t = reflect(ray.dir, tNorm);
            ray.origin = h.point + h.normal * 1e-3;
        } else {
            let refracted = refract(normalize(ray.dir), tNorm, eta);
            if (length(refracted) < 0.001) {
                wi_t = reflect(ray.dir, tNorm);
                ray.origin = h.point + h.normal * 1e-3;
            } else {
                wi_t = refracted;
                ray.origin = h.point - h.normal * 1e-3;
                didRefract = true;
            }
        }
        var microWeight = 1.0;
        if (usedMicrofacet) {
            let cosOut = abs(dot(wi_t, h.normal));
            microWeight = ggxG1(cosOut, h.shininess);
        }
        if (didRefract) {
            let glassTint = mix(albedo, vec3<f32>(1.0), h.transmission) * microWeight;
            var volAtten = vec3<f32>(1.0);
            if (tAttDist > 0.0 && !entering) {
                let absorbCoeff = -log(max(tAttColor, vec3<f32>(1e-6))) / tAttDist;
                let pathLen = select(h.t, h.thickness, h.t < 1e-2 && h.thickness > 0.0);
                volAtten = exp(-absorbCoeff * pathLen);
            }
            throughput *= glassTint * volAtten / (eta * eta) * channelMask;
        } else {
            throughput *= vec3<f32>(microWeight) * channelMask;
        }
        ray.dir = wi_t;
        // Original: set afterTransmission=true then `continue`.  Hand off to runBounces.
        result.afterTransmission = true;
        result.rayOrigin  = ray.origin;
        result.rayDir     = ray.dir;
        result.throughput = throughput;
        result.diffRad    = diffRad;
        result.specRad    = specRad;
        // NOTE: prevNormal/prevAlpha/prevMetalness/prevWo left at struct defaults
        // here — matches original, which also doesn't update prev* on the transmission
        // `continue` branch at i==0.  firstBounceSpec also stays false; see pathTrace
        // comment "Transmission at bounce 0 also counts as specular for routing" —
        // that code path is unreachable in the original due to the `continue` above
        // the BRDF-sample `if (i == 0 && afterTransmission)` line, so we preserve
        // that behaviour verbatim.
        result.pathAlive = true;
        return result;
    }

    // BRDF sample (non-transmission path).
    let F0_b = F0_h;
    var wi_b: vec3<f32>;
    let p_spec = mix(0.5, 0.98, h.metalness);
    let isSpecBounce = rand(seed) < p_spec;
    result.firstBounceSpec = isSpecBounce;
    // Note: `if (i == 0 && afterTransmission)` from original is unreachable here (see
    // transmission return above), so we don't replicate that assignment.
    if (isSpecBounce) {
        wi_b = sampleVNDF(wo, h.normal, h.shininess, seed);
        let cos_b = dot(h.normal, wi_b);
        if (cos_b <= 0.0) {
            if (rt.spp.x > 0.5 && !result.giResStored) {
                textureStore(giResWrite,   pixel, vec4<f32>(0.0));
                textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
                textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
                result.giResStored = true;
            }
            result.diffRad = diffRad;
            result.specRad = specRad;
            result.pathAlive = false;
            return result;
        }
        let hb  = normalize(wo + wi_b);
        let Fb  = schlick(max(0.0, dot(wo, hb)), F0_b);
        let G1L = ggxG1(cos_b, h.shininess);
        throughput *= Fb * G1L / p_spec;
    } else {
        wi_b = cosineHemisphere(h.normal, seed);
        let cos_b = dot(h.normal, wi_b);
        if (cos_b <= 0.0) {
            if (rt.spp.x > 0.5 && !result.giResStored) {
                textureStore(giResWrite,   pixel, vec4<f32>(0.0));
                textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
                textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
                result.giResStored = true;
            }
            result.diffRad = diffRad;
            result.specRad = specRad;
            result.pathAlive = false;
            return result;
        }
        throughput *= albedo * (1.0 - h.metalness) / (1.0 - p_spec);
    }
    result.prevWo        = wo;
    result.prevNormal    = h.normal;
    result.prevAlpha     = h.shininess;
    result.prevMetalness = h.metalness;
    result.afterTransmission = false;
    ray.origin = h.point + h.normal * 1e-3;
    ray.dir    = wi_b;

    result.rayOrigin  = ray.origin;
    result.rayDir     = ray.dir;
    result.throughput = throughput;
    result.diffRad    = diffRad;
    result.specRad    = specRad;
    result.pathAlive  = true;
    return result;
}
)";

// pathTrace wrapper (bounce split, Stage B): replaces the old monolithic loop.
// Calls primaryShade for bounce 0, then if the path is still alive, runBounces for
// bounces 1..maxBounces.  Signature is UNCHANGED so rt_main continues to call this
// without modification.  Note: rt_main no longer calls this wrapper — it inlines
// primaryShade + serialize then dispatches to rt_bounces_main for runBounces.
// The wrapper is kept as a reference implementation / fallback that proves the
// two helpers compose correctly when called together.
const char* const csPathTraceWGSL2 = R"(
fn pathTrace(ray_in: Ray,
             primaryHit: Hit,  // pre-computed by rt_primary_main; skip BVH for i==0
             seed: ptr<function, u32>,
             pixel: vec2<i32>,
             maxBounces:     i32,
             primaryMeshIdx: ptr<function, u32>,
             primaryNormal:  ptr<function, vec3<f32>>,
             primaryDepth:   ptr<function, f32>,
             primaryAlbedo:  ptr<function, vec3<f32>>,
             primaryRough:   ptr<function, f32>,
             primaryMatIdx:  ptr<function, i32>,
             primaryTriIdx:  ptr<function, i32>,
             touchedMoved:   ptr<function, bool>) -> SplitRadiance {
    *touchedMoved = false;
    let result = primaryShade(primaryHit, ray_in, seed, pixel, maxBounces,
                              primaryMeshIdx, primaryNormal, primaryDepth,
                              primaryAlbedo, primaryRough, primaryMatIdx, primaryTriIdx);
    if (!result.pathAlive) {
        return SplitRadiance(result.diffRad, result.specRad);
    }
    let bouncesResult = runBounces(result, seed, pixel, *primaryMeshIdx, touchedMoved);
    return SplitRadiance(result.diffRad + bouncesResult.diff,
                         result.specRad + bouncesResult.spec);
}
)";

// Third part of the path trace shader (rt_main entry point + accumulation)
const char* const csPathTraceWGSL3 = R"(
@compute @workgroup_size(8, 8)
fn rt_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res   = rt.iRes.xy;
    let totalPixels = u32(res.x) * u32(res.y);
    let resXu = u32(res.x);
    // Persistent-thread work-stealing loop.  Each iteration claims one pixel
    // from the global queue.  Batch-pull (N=8) was tried and found slower —
    // atomic contention wasn't the bottleneck; losing per-pixel work-stealing
    // granularity cost more than the saved atomic traffic.
    loop {
        let pixelIdx = atomicAdd(&pathCounter, 1u);
        if (pixelIdx >= totalPixels) { break; }
        let pixel = vec2<i32>(i32(pixelIdx % resXu), i32(pixelIdx / resXu));

        let fc         = u32(rt.frameCount.x);
        let foveatedOn = rt.params.z > 0.5;

    // Don't foveate sky/env-map pixels — they're cheap to trace (BVH miss)
    // and skipping creates visible zone boundaries in uniform backgrounds.
    // Use previous frame's gBuf depth: env/sky pixels have depth <= 0.
    let prevDepth = textureLoad(gBufRead, pixel, 0).w;
    let isEnvPixel = prevDepth <= 0.0;

    // --- Material classification from previous frame's G-buffer ---
    // Used for material-aware bounce cap (camera motion) and foveated scheduling.
    // matClass: 0 = specular (glass/metal/mirror), 1 = glossy, 2 = rough diffuse, 3 = sky
    var matClass = 3u;  // default: sky/unknown → conservative (full bounces, no aggressive skip)
    if (!isEnvPixel) {
        let prevMatIdx = i32(textureLoad(hitMeshRead, pixel, 0).g);
        if (prevMatIdx >= 0) {
            let m0 = textureLoad(matData, vec2<i32>(prevMatIdx, 0), 0);   // .w = shininess (alpha = roughness^2)
            let m1 = textureLoad(matData, vec2<i32>(prevMatIdx, 1), 0);   // .y = metalness
            let m2 = textureLoad(matData, vec2<i32>(prevMatIdx, 2), 0);   // .w = transmission
            let shininess    = m0.w;
            let metalness    = m1.y;
            let transmission = m2.w;
            let isGlass  = transmission > 0.05;
            let isMetal  = metalness > 0.5;
            let isMirror = shininess < 0.05;
            let isGlossy = shininess < 0.25;
            if (isGlass || isMetal || isMirror) { matClass = 0u; }
            else if (isGlossy)                  { matClass = 1u; }
            else                                { matClass = 2u; }
        }
    }

    // Material-aware first-frame bounce cap.  On the very first sample (fc==0,
    // i.e. after forceReset) reduce bounces per material class to warm up faster.
    // Glass/metal/mirror need 4 (refraction chains), diffuse only needs 2.
    var maxBounces = i32(rt.params.x);
    if (fc == 0u) {
        if      (matClass <= 0u) { maxBounces = min(maxBounces, 3); }  // specular
        else if (matClass == 1u) { maxBounces = min(maxBounces, 2); }  // glossy
        else if (matClass == 2u) { maxBounces = min(maxBounces, 1); }  // rough diffuse
        else                     { maxBounces = min(maxBounces, 3); }  // sky/unknown
    }

    // --- Foveated rendering: progressive spatial coarsening ---
    // Instead of temporal frame-skipping (which leaves stale pixels → ghosting
    // during camera motion), we reduce spatial resolution outside the center cone.
    // Pixels in the same NxN block share the trace from the block's top-left
    // corner (the "leader"). Every pixel gets a fresh value each frame — no
    // ghosting — but the periphery traces fewer unique rays.
    //
    // Zone layout (normalized distance from center):
    //   dist <= 0.30 : 1×1 (full resolution)
    //   dist <= 0.55 : 2×2 blocks
    //   dist >  0.55 : 4×4 blocks
    // During static camera: fall back to every-pixel tracing (fc > 0 handles
    // convergence naturally, no skip needed).
    let center = res * 0.5;
    let dxy = (vec2<f32>(f32(pixel.x), f32(pixel.y)) - center) / center;
    let dist = length(dxy);
    let camMovingFov = (u32(rt.params.w) & 2u) != 0u;
    var fovBlockSize = 1u;
    if (foveatedOn && camMovingFov && !isEnvPixel) {
        if (dist > 0.55) {
            fovBlockSize = 4u;
            maxBounces = min(maxBounces, 1);  // periphery: direct light only
        } else if (dist > 0.30) {
            fovBlockSize = 2u;
            maxBounces = min(maxBounces, 2);  // middle: one indirect bounce
        }
    }
    // Snap to block leader (top-left pixel of the block)
    let fovLeader = vec2<i32>(
        i32((u32(pixel.x) / fovBlockSize) * fovBlockSize),
        i32((u32(pixel.y) / fovBlockSize) * fovBlockSize));
    let isFovLeader = all(pixel == fovLeader);
    let foveatedSkip = fovBlockSize > 1u && !isFovLeader;

    // Checkerboard skip: during camera motion skip half the pixels each frame,
    // alternating which half via globalFrameCounter so both patterns are covered across
    // two consecutive frames.
    //
    // Fires during rotation, pan, and orbit.  Disabled for forward/backward dolly
    // motion: dolly changes depth of every pixel proportionally, and the screen-
    // space parallax scales inversely with depth, so near geometry moves many
    // pixels per frame — a single alternation can't reconstruct that cleanly and
    // produces visible ghosting on close surfaces.  Pan and orbit have motion
    // roughly perpendicular to the view direction, so screen-space parallax is
    // small and uniform across depths, safe for checkerboarding.
    //
    // Staleness during allowed motion is handled by the follower-copy branch
    // capping pixelFC to 4 at stamp time: the next fresh sample at that pixel
    // gets ~20% EMA weight and the stale value decays in ~5 frames.
    // Foveated coarsening still excludes checker to avoid compounding sparsity.
    let camMovedEarly = (u32(rt.params.w) & 2u) != 0u;  // params.w bit 1 = camMoved
    let dTrans = rt.camOri.xyz - rt.prevCamOri.xyz;
    let tLen = length(dTrans);
    // "Dolly" = translation aligned with view axis.  cos(45°)≈0.7.  Pan is
    // perpendicular (ratio ≈ 0), orbit is tangential (ratio ≈ 0), dolly is
    // parallel (ratio ≈ 1).
    let dollyRatio = select(0.0, abs(dot(dTrans, rt.camFwd.xyz)) / tLen, tLen > 1e-5);
    let isDolly = dollyRatio > 0.7;
    let checkerSkip = !foveatedOn && camMovedEarly && !isDolly && !isEnvPixel &&
        ((u32(pixel.x) + u32(pixel.y) + u32(rt.params.y)) & 1u) == 0u;

    // Foveated spatial coarsening: follower pixels copy from the block leader's
    // previous-frame result.  The leader itself always traces fresh this frame.
    // This gives spatially coherent blocks with at most 1-frame latency — no
    // multi-frame ghosting.
    //
    // Checkerboard skip: pass through previous accumulation unchanged.
    if (foveatedSkip || checkerSkip) {
        // Foveated followers read from the leader; checkerboard reads from self.
        let src = select(pixel, fovLeader, foveatedSkip);
        // Cap history on leader-copied samples.  The follower never truly sampled
        // its own world position — its color is spatially stamped from the leader.
        // Storing the leader's pixelFC (possibly 256) would give the next fresh
        // sample only ~0.4% weight at the moment motion ends, burning in whatever
        // noisy value the last foveated leader produced.  Cap to 4 so the first
        // post-motion sample gets ~20% weight and any outlier decays in ~20 frames.
        let leaderAccum = textureLoad(accumRead, src, 0);
        let leaderDiff  = textureLoad(diffAccumRead, src, 0);
        let leaderSpec  = textureLoad(specAccumRead, src, 0);
        let histCap = 4.0;
        textureStore(accumWrite,     pixel, vec4<f32>(leaderAccum.xyz, min(leaderAccum.w, histCap)));
        textureStore(diffAccumWrite, pixel, vec4<f32>(leaderDiff.xyz,  min(leaderDiff.w,  histCap)));
        textureStore(specAccumWrite, pixel, vec4<f32>(leaderSpec.xyz,  min(leaderSpec.w,  histCap)));
        textureStore(hitMeshWrite, pixel, textureLoad(hitMeshRead, src, 0));
        textureStore(gBufWrite,    pixel, textureLoad(gBufRead, src, 0));
        textureStore(albedoWrite,  pixel, vec4<f32>(vec3<f32>(0.0), 0.0));
        textureStore(momentsWrite, pixel, textureLoad(momentsRead, src, 0));
        if (rt.restirParams.x > 0.5) {
            textureStore(reservoirWrite,  pixel, textureLoad(reservoirRead,  src, 0));
            textureStore(reservoirWWrite, pixel, textureLoad(reservoirWRead, src, 0));
        }
        // Mark this pixel as skipAccum so rt_bounces_main doesn't re-accumulate
        // on top of the leader-copied values.  Only the flag bits matter; other
        // PathStateEntry fields are left zero (harmless because skipAccum
        // short-circuits the bounce + accumulation path).
        {
            var entry: PathStateEntry;
            let flagBits = 32u;  // bit 5 = skipAccum
            entry.w0 = vec4<f32>(0.0);
            entry.w1 = vec4<f32>(0.0);
            entry.w2 = vec4<f32>(0.0, 0.0, 0.0, bitcast<f32>(flagBits));
            entry.w3 = vec4<f32>(0.0);
            entry.w4 = vec4<f32>(0.0);
            entry.w5 = vec4<f32>(0.0);
            entry.w6 = vec4<f32>(0.0);
            entry.w7 = vec4<f32>(0.0);
            entry.w8 = vec4<f32>(0.0);
            entry.w9 = vec4<f32>(0.0);
            entry.w10 = vec4<f32>(0.0);
            entry.w11 = vec4<f32>(0.0);
            pathStateBuf[pixelIdx] = entry;
        }
        if (rt.spp.x > 0.5) {
            textureStore(giResWrite,   pixel, textureLoad(giResRead,   src, 0));
            textureStore(giResWWrite,  pixel, textureLoad(giResWRead,  src, 0));
            textureStore(giResLoWrite, pixel, textureLoad(giResLoRead, src, 0));
        }
        continue;
    }
)"
R"(
    // AOV mode: only need primary-hit geometry data — skip all secondary bounces.
    // Mode 6 also caps to 1 bounce (geometry is all the heatmap needs).
    var varianceReducedBounces = maxBounces;
    let aovMode = i32(rt.emissiveInfo.w);
    if (aovMode > 0) { varianceReducedBounces = 1; }

    // pixelHistory and camMovedNow are consumed by the AOV mode 6 eligibility
    // check below.
    let pixelHistory = textureLoad(accumRead, pixel, 0).w;
    let camMovedNow  = (u32(rt.params.w) & 2u) != 0u;

    // Spatio-temporal blue noise (Heitz & Belcour 2019).
    // Camera jitter and the integrator both consume the same per-pixel BN state
    // (defined in csCommonWGSL): R2/golden-ratio QMC across frames+dims, with a
    // per-pixel Cranley-Patterson rotation that has blue-noise spectral falloff.
    // Sub-pixel jitter centred on pixel centre (0.5, 0.5), range ±0.375.
    //
    // Jitter is always on: blue-noise decorrelation keeps sub-pixel offsets
    // spatio-temporally varied, so neighbouring pixels' stored gbuf values
    // differ by bounded (~1% depth, <5° normal on curved surfaces) amounts
    // that the à-trous edge-stops (depth scale 4, albedo-similarity stop)
    // tolerate. The RT accumulation's bilinear reprojection absorbs the
    // worldPos sub-pixel mismatch, accumulating correctly toward the true
    // super-sampled mean.
    bnInit(u32(pixel.x), u32(pixel.y), fc);
    let camBn = bnNext2d();
    let jx = (camBn.x - 0.5) * 0.75;
    let jy = (camBn.y - 0.5) * 0.75;
    // PCG seed kept for the many integrator dims that don't yet use BN
    // (RR termination, lobe selection, ReSTIR M-sampling, etc.).
    var seed = pcg(pcg(u32(pixel.x) + u32(pixel.y) * 65537u) + fc * 12979u);

    let ray = makeRay(vec2<f32>(f32(pixel.x) + 0.5 + jx, f32(pixel.y) + 0.5 + jy), res);

    // Read the primary hit from the buffer populated by rt_primary_main.
    let packed = primaryHitBuf[pixelIdx];

    // Sky fast-path (kernel-split optimisation): skip pathTrace entirely for
    // primary-miss pixels.  A full pathTrace call for a miss costs ~function
    // call + loop setup + default-Hit construction + miss branch + output-param
    // writes.  Handling it inline saves register allocation for the function-
    // scope Hit struct (~20 vec/scalar fields) and lets the thread pull the
    // next queued pixel sooner.
    //
    // This also handles AOV modes for sky (aovColor is black/zero for all
    // modes since there's no geometry).  The full AOV write path below is
    // skipped in favour of a minimal sentinel write set.
    if (packed.triIdx < 0) {
        let bgColor = sampleBackground(ray.dir);
        let lum3    = vec3<f32>(0.2126, 0.7152, 0.0722);
        // NaN guard
        let bgClean = select(vec3<f32>(0.0), bgColor, bgColor.x == bgColor.x);
        let bgLum   = dot(bgClean, lum3);

        // Progressive accumulation (same math as the non-miss tail below).
        let prev   = textureLoad(accumRead, pixel, 0);
        let old    = prev.xyz;
        var pxFC   = prev.w;
        let forceRst = (u32(rt.params.w) & 1u) != 0u;
        if (forceRst) { pxFC = 0.0; }
        let alpha = 1.0 / (pxFC + 1.0);
        var blended: vec3<f32>;
        var newFC:   f32;
        if (pxFC < 1024.0) {
            blended = old * (1.0 - alpha) + bgClean * alpha;
            newFC   = pxFC + 1.0;
        } else {
            blended = old;
            newFC   = pxFC;
        }
        // Sky is all diffuse (no specular component).
        textureStore(accumWrite,     pixel, vec4<f32>(blended, newFC));
        textureStore(diffAccumWrite, pixel, vec4<f32>(blended, newFC));
        textureStore(specAccumWrite, pixel, vec4<f32>(vec3<f32>(0.0), newFC));
        textureStore(momentsWrite,   pixel, vec4<f32>(bgLum, bgLum * bgLum, 0.0, 0.0));

        // Sentinel metadata — matches what pathTrace would write for a miss.
        textureStore(hitMeshWrite, pixel, vec4<f32>(128.0, -1.0, 0.0, 0.0));
        textureStore(albedoWrite,  pixel, vec4<f32>(1.0, 1.0, 1.0, 1.0));
        textureStore(gBufWrite,    pixel, vec4<f32>(0.0, 0.0, 0.0, 0.0));

        // Clear reservoirs on miss (mirrors pathTrace's i==0 miss handling).
        if (rt.restirParams.x > 0.5) {
            textureStore(reservoirWrite,  pixel, vec4<f32>(0.0));
            textureStore(reservoirWWrite, pixel, vec4<f32>(0.0));
        }
        if (rt.spp.x > 0.5) {
            textureStore(giResWrite,   pixel, vec4<f32>(0.0));
            textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
            textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
        }
        // Sky fast-path wrote accum itself — mark skipAccum for rt_bounces_main.
        {
            var entry: PathStateEntry;
            let flagBits = 32u;  // bit 5 = skipAccum
            entry.w0 = vec4<f32>(0.0);
            entry.w1 = vec4<f32>(0.0);
            entry.w2 = vec4<f32>(0.0, 0.0, 0.0, bitcast<f32>(flagBits));
            entry.w3 = vec4<f32>(0.0);
            entry.w4 = vec4<f32>(0.0);
            entry.w5 = vec4<f32>(0.0);
            entry.w6 = vec4<f32>(0.0);
            entry.w7 = vec4<f32>(0.0);
            entry.w8 = vec4<f32>(0.0);
            entry.w9 = vec4<f32>(0.0);
            entry.w10 = vec4<f32>(0.0);
            entry.w11 = vec4<f32>(0.0);
            pathStateBuf[pixelIdx] = entry;
        }
        continue;
    }

    let rh = RawHit(packed.t, packed.triIdx, packed.u, packed.v);
    var primaryHit = loadHitMaterial(rh, ray);

    var primaryMeshIdx: u32;
    var primaryNormal:  vec3<f32>;
    var primaryDepth:   f32;
    var primaryAlbedo:  vec3<f32>;
    var primaryRough:   f32;
    var primaryMatIdx:  i32;
    var primaryTriIdx:  i32;
    var touchedMoved:   bool = false;

    // Run primary shading (bounce 0 NEE/ReSTIR DI + first BRDF sample).
    let shadeResult = primaryShade(primaryHit, ray, &seed, pixel, varianceReducedBounces,
                                   &primaryMeshIdx, &primaryNormal, &primaryDepth,
                                   &primaryAlbedo, &primaryRough, &primaryMatIdx, &primaryTriIdx);

    // Serialize PrimaryShadeResult -> PathStateEntry for rt_bounces_main.  Packed
    // per the layout declared at binding 37.  Seed captured AFTER primaryShade so
    // rt_bounces_main resumes from the same PCG state primaryShade left behind.
    // touchedMoved is initialized here (primary never sets it) and updated
    // in-place in rt_bounces_main via bit 2 of the flags word.
    {
        let flagBits = (select(0u, 1u, shadeResult.firstBounceSpec))
                     | (select(0u, 2u, shadeResult.afterTransmission))
                     | (select(0u, 4u, touchedMoved))
                     | (select(0u, 8u, shadeResult.pathAlive))
                     | (select(0u, 16u, shadeResult.giResStored));
        var entry: PathStateEntry;
        entry.w0  = vec4<f32>(shadeResult.rayOrigin, bitcast<f32>(seed));
        entry.w1  = vec4<f32>(shadeResult.rayDir,    bitcast<f32>(shadeResult.effectiveBounces));
        entry.w2  = vec4<f32>(shadeResult.throughput, bitcast<f32>(flagBits));
        entry.w3  = vec4<f32>(shadeResult.diffRad,    shadeResult.prevMetalness);
        entry.w4  = vec4<f32>(shadeResult.specRad,    shadeResult.prevAlpha);
        entry.w5  = vec4<f32>(shadeResult.prevNormal, shadeResult.primaryDepth);
        entry.w6  = vec4<f32>(shadeResult.prevWo,     0.0);
        entry.w7  = vec4<f32>(shadeResult.b0Point,    shadeResult.b0Alpha);
        entry.w8  = vec4<f32>(shadeResult.b0Normal,   shadeResult.b0Metal);
        entry.w9  = vec4<f32>(shadeResult.b0Wo,       bitcast<f32>(shadeResult.b0MeshIdx));
        entry.w10 = vec4<f32>(shadeResult.b0Albedo,   bitcast<f32>(shadeResult.primaryMatIdx));
        entry.w11 = vec4<f32>(shadeResult.b0F0,       0.0);
        pathStateBuf[pixelIdx] = entry;
    }

    // runBounces + radiance accumulation live in rt_bounces_main.  rt_main's
    // responsibility ends with serialize + primary-metadata texture writes.
    // AOV / sky / foveated paths short-circuit earlier because they need
    // primary-derived data (primaryDepth/etc) that are rt_main locals.
)"
R"(
    // AOV visualization mode — write noise-free primary-hit data and return early.
    // Writes to diffAccumWrite (display shader reads diffTex + specTex).
    if (aovMode > 0) {
        var aovColor = vec3<f32>(0.0);
        if (aovMode == 1)      { aovColor = vec3<f32>(primaryDepth / (primaryDepth + 1.0)); }
        else if (aovMode == 2) { aovColor = primaryNormal * 0.5 + 0.5; }
        else if (aovMode == 3) { aovColor = primaryAlbedo; }
        else if (aovMode == 4) { aovColor = vec3<f32>(f32(primaryMeshIdx % 32u) / 32.0,
                                                       f32((primaryMeshIdx * 7u) % 32u) / 32.0,
                                                       f32((primaryMeshIdx * 13u) % 32u) / 32.0); }
        else if (aovMode == 5) { aovColor = vec3<f32>(primaryRough); }
        else if (aovMode == 6) {
            // Use actual per-pixel roughness (includes roughnessMap texture, not just base value).
            // threshold: roughness² > 0.05  ↔  roughness > 0.224  (matches isMirror in pathTrace)
            // Sky pixels: primaryDepth == 0 → always blue (no geometry, no saving)
            let isEligible = primaryRough > 0.224 && primaryDepth > 0.0 && !camMovedNow;
            aovColor = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), isEligible);
        }
        textureStore(diffAccumWrite, pixel, vec4<f32>(aovColor, pixelHistory + 1.0));
        textureStore(specAccumWrite, pixel, vec4<f32>(0.0, 0.0, 0.0, pixelHistory));
        // Preserve pixelHistory in .w so adaptive bounce reduction keeps accumulating
        // across AOV mode 6 frames and doesn't reset to 1 each frame.
        textureStore(accumWrite, pixel, vec4<f32>(aovColor, pixelHistory + 1.0));
        textureStore(hitMeshWrite, pixel, vec4<f32>(f32(primaryMeshIdx), f32(primaryMatIdx), 0.0, 0.0));
        textureStore(gBufWrite, pixel, vec4<f32>(primaryNormal, primaryDepth));
        textureStore(albedoWrite, pixel, vec4<f32>(primaryAlbedo, primaryRough));
        // AOV already wrote accum — OR skipAccum into the flags word written by
        // the earlier serialize.  Direct overwrite of pathStateBuf[pixelIdx].w2.w
        // preserves other bits we don't care about (path terminates at AOV).
        {
            let prevFlags = bitcast<u32>(pathStateBuf[pixelIdx].w2.w);
            pathStateBuf[pixelIdx].w2.w = bitcast<f32>(prevFlags | 32u);
        }
        continue;
    }

    // accumulation + hitMesh write live in rt_bounces_main (hitMesh depends on
    // touchedMoved).  rt_main still writes primary-only metadata here.
    textureStore(albedoWrite,  pixel, vec4<f32>(primaryAlbedo, primaryRough));

    textureStore(gBufWrite, pixel, vec4<f32>(primaryNormal, primaryDepth));
    }  // end of work-stealing loop
}

// ---------------------------------------------------------------------------
// Primary-hit kernel (kernel split, step 1 of 2).
// Traces the camera ray through the BVH and writes a compact RawHit record
// (triIdx, t, u, v) to primaryHitBuf.  No material lookup, no shading, no
// bounces — those happen in rt_main, which reconstructs the full Hit via
// loadHitMaterial.  Splitting primary traversal off the megakernel reduces
// register pressure in both halves: this kernel has only BVH state (~50
// registers), and rt_main no longer needs to carry primary-traversal
// locals alongside its shading/bounce state.
//
// Seeding note: this kernel consumes one bnNext2d() for camera jitter and may
// consume PCG randomness inside stochastic alpha tests in testTriangle().
// rt_main re-seeds from the same pixel+frame values (same initial seed) and
// re-derives the same ray — the primary is NOT re-traced, so correlation is
// inert.  See bnInit/seed setup in rt_main for the matching convention.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn rt_primary_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res   = rt.iRes.xy;
    let totalPixels = u32(res.x) * u32(res.y);
    let resXu = u32(res.x);
    loop {
        let pixelIdx = atomicAdd(&primaryCounter, 1u);
        if (pixelIdx >= totalPixels) { break; }
        let pixel = vec2<i32>(i32(pixelIdx % resXu), i32(pixelIdx / resXu));

        let fc = u32(rt.frameCount.x);
        // Jitter derivation must match rt_main exactly so both kernels trace the
        // same camera ray for this pixel.  sceneHitRaw's stochastic alpha uses
        // its own PCG seeded from (triIdx, frame, t) — it doesn't consume any
        // thread-local seed, so no seed variable is needed in this kernel.
        bnInit(u32(pixel.x), u32(pixel.y), fc);
        let camBn = bnNext2d();
        let jx = (camBn.x - 0.5) * 0.75;
        let jy = (camBn.y - 0.5) * 0.75;

        let ray = makeRay(vec2<f32>(f32(pixel.x) + 0.5 + jx, f32(pixel.y) + 0.5 + jy), res);
        let rh = sceneHitRaw(ray, 1e30);

        var packed: PrimaryHitPacked;
        packed.triIdx = rh.triIdx;
        packed.t      = rh.t;
        packed.u      = rh.u;
        packed.v      = rh.v;
        primaryHitBuf[pixelIdx] = packed;

        // Vestigial: keeps binding 37 (pathStateBuf) in primaryPipeline's
        // bind-group layout so C++ setStorageBuffer(37, ...) stays valid for the
        // shared bind-group helpers.  rt_main overwrites this entry immediately
        // in the next pass, so the value doesn't matter.
        pathStateBuf[pixelIdx].w0 = vec4<f32>(0.0);
    }
}

// ---------------------------------------------------------------------------
// rt_bounces_main (bounce split, Stage D).
// Consumes pathStateBuf entries written by primaryShade-in-rt_main, runs the
// bounce loop (runBounces), and performs the full accumulation / reprojection
// pipeline that previously lived inline at the end of rt_main.  Splits the
// megakernel into primary / shade / bounce phases, reducing register pressure
// in each and enabling future material-sorting optimisations (wavefront).
//
// skipAccum short-circuit: pixels handled by foveated/checker copy, sky fast-
// path, or AOV modes write their own accum and set bit 5 of the flags word
// so this kernel treats them as no-ops.
// ---------------------------------------------------------------------------
@compute @workgroup_size(8, 8)
fn rt_bounces_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = rt.iRes.xy;
    let totalPixels = u32(res.x) * u32(res.y);
    let resXu = u32(res.x);
    loop {
        let pixelIdx = atomicAdd(&bounceCounter, 1u);
        if (pixelIdx >= totalPixels) { break; }
        let pixel = vec2<i32>(i32(pixelIdx % resXu), i32(pixelIdx / resXu));

        let entry = pathStateBuf[pixelIdx];
        let flagBits = bitcast<u32>(entry.w2.w);
        let skipAccum = (flagBits & 32u) != 0u;
        if (skipAccum) { continue; }

        // Deserialize PrimaryShadeResult from PathStateEntry (see packing legend
        // at struct declaration above).
        var state: PrimaryShadeResult;
        state.rayOrigin        = entry.w0.xyz;
        var seed               = bitcast<u32>(entry.w0.w);
        state.rayDir           = entry.w1.xyz;
        state.effectiveBounces = bitcast<i32>(entry.w1.w);
        state.throughput       = entry.w2.xyz;
        state.firstBounceSpec  = (flagBits & 1u)  != 0u;
        state.afterTransmission = (flagBits & 2u) != 0u;
        var touchedMoved       = (flagBits & 4u)  != 0u;
        state.pathAlive        = (flagBits & 8u)  != 0u;
        state.giResStored      = (flagBits & 16u) != 0u;
        state.diffRad          = entry.w3.xyz;
        state.prevMetalness    = entry.w3.w;
        state.specRad          = entry.w4.xyz;
        state.prevAlpha        = entry.w4.w;
        state.prevNormal       = entry.w5.xyz;
        state.primaryDepth     = entry.w5.w;
        state.prevWo           = entry.w6.xyz;
        state.b0Point          = entry.w7.xyz;
        state.b0Alpha          = entry.w7.w;
        state.b0Normal         = entry.w8.xyz;
        state.b0Metal          = entry.w8.w;
        state.b0Wo             = entry.w9.xyz;
        state.b0MeshIdx        = bitcast<i32>(entry.w9.w);
        state.b0Albedo         = entry.w10.xyz;
        state.primaryMatIdx    = bitcast<i32>(entry.w10.w);
        state.b0F0             = entry.w11.xyz;

        // Run bounces if path survived primaryShade, otherwise fall through with
        // primary-only radiance.
        var ptResult: SplitRadiance;
        let primaryMeshIdx_u   = u32(state.b0MeshIdx);
        if (state.pathAlive) {
            let bouncesResult = runBounces(state, &seed, pixel, primaryMeshIdx_u, &touchedMoved);
            ptResult = SplitRadiance(state.diffRad + bouncesResult.diff,
                                     state.specRad + bouncesResult.spec);
        } else {
            ptResult = SplitRadiance(state.diffRad, state.specRad);
        }
        var sample = ptResult.diff + ptResult.spec;

        // Derived values (aliases for readability in the preserved accumulation
        // block below).  Uses b0* captured early in primaryShade (valid for all
        // primary hits, including unlit which leaves pathAlive=false).
        let primaryMeshIdx = primaryMeshIdx_u;
        let primaryDepth   = state.primaryDepth;
        let primaryRough   = sqrt(state.b0Alpha);

        // === Accumulation + reprojection (moved verbatim from rt_main) ===
        let prev    = textureLoad(accumRead, pixel, 0);
        var old     = prev.xyz;
        var pixelFC = prev.w;

        let prevDiffRaw = textureLoad(diffAccumRead, pixel, 0).xyz;
        let prevSpecRaw = textureLoad(specAccumRead, pixel, 0).xyz;
        var oldDiff = select(vec3<f32>(0.0), prevDiffRaw, prevDiffRaw.x == prevDiffRaw.x);
        var oldSpec = select(vec3<f32>(0.0), prevSpecRaw, prevSpecRaw.x == prevSpecRaw.x);
        let prevMom = textureLoad(momentsRead, pixel, 0);
        var prevMomM1 = prevMom.x;
        var prevMomM2 = prevMom.y;

        let forceReset   = (u32(rt.params.w) & 1u) != 0u;
        let camMovedFlag = (u32(rt.params.w) & 2u) != 0u;
        if (forceReset) { pixelFC = 0.0; }

        let prevMeshU = u32(textureLoad(hitMeshRead, pixel, 0).r);
        let primaryMoved = primaryMeshIdx < 128u && isMeshMoved(i32(primaryMeshIdx));

        if (primaryMeshIdx != prevMeshU && prevMeshU < 128u && isMeshMoved(i32(prevMeshU))) {
            pixelFC = 0.0;
        }
        if (touchedMoved) {
            let emissiveMoved = rt.restirParams.z > 0.5;
            pixelFC = min(pixelFC, select(8.0, 2.0, emissiveMoved));
        }
)"
R"(
        if (primaryMoved || camMovedFlag) {
            var reprojOk = false;
            if (pixelFC > 0.0 && primaryDepth > 0.0) {
                // b0Point == rt.camOri + cameraRayDir * primaryDepth (captured from
                // h.point at primary hit), so we read it directly instead of
                // reconstructing the camera ray.
                let worldPos = state.b0Point;
                var prevWorldPos = worldPos;
                if (primaryMoved) {
                    prevWorldPos = (rtMotionMats[primaryMeshIdx] * vec4<f32>(worldPos, 1.0)).xyz;
                }
                let toPoint = prevWorldPos - rt.prevCamOri.xyz;
                let prevZ   = dot(toPoint, rt.prevCamFwd.xyz);
                if (prevZ > 0.001) {
                    let aspect   = res.x / res.y;
                    let prevNdcX = dot(toPoint, rt.prevCamRgt.xyz) / (prevZ * rt.tanHalfFov.x * aspect);
                    let prevNdcY = dot(toPoint, rt.prevCamUp.xyz)  / (prevZ * rt.tanHalfFov.x);
                    let prevU = (prevNdcX + 1.0) * 0.5 * res.x;
                    let prevV = (1.0 - prevNdcY) * 0.5 * res.y;
                    let pxBase = vec2<i32>(i32(floor(prevU)), i32(floor(prevV)));
                    if (pxBase.x >= 0 && pxBase.x + 1 < i32(res.x) &&
                        pxBase.y >= 0 && pxBase.y + 1 < i32(res.y)) {
                        let fx = prevU - f32(pxBase.x);
                        let fy = prevV - f32(pxBase.y);
                        let w00 = (1.0 - fx) * (1.0 - fy);
                        let w10 = fx         * (1.0 - fy);
                        let w01 = (1.0 - fx) *         fy;
                        let w11 = fx         *         fy;
                        let p00 = pxBase;
                        let p10 = pxBase + vec2<i32>(1, 0);
                        let p01 = pxBase + vec2<i32>(0, 1);
                        let p11 = pxBase + vec2<i32>(1, 1);
                        let m00 = u32(textureLoad(hitMeshRead, p00, 0).r);
                        let m10 = u32(textureLoad(hitMeshRead, p10, 0).r);
                        let m01 = u32(textureLoad(hitMeshRead, p01, 0).r);
                        let m11 = u32(textureLoad(hitMeshRead, p11, 0).r);
                        let v00 = select(0.0, w00, m00 == primaryMeshIdx);
                        let v10 = select(0.0, w10, m10 == primaryMeshIdx);
                        let v01 = select(0.0, w01, m01 == primaryMeshIdx);
                        let v11 = select(0.0, w11, m11 == primaryMeshIdx);
                        let wSum = v00 + v10 + v01 + v11;
                        if (wSum > 1e-4) {
                            let a00 = textureLoad(accumRead, p00, 0);
                            let a10 = textureLoad(accumRead, p10, 0);
                            let a01 = textureLoad(accumRead, p01, 0);
                            let a11 = textureLoad(accumRead, p11, 0);
                            let inv = 1.0 / wSum;
                            old = (a00.xyz * v00 + a10.xyz * v10
                                 + a01.xyz * v01 + a11.xyz * v11) * inv;
                            let prevFC = (a00.w * v00 + a10.w * v10
                                        + a01.w * v01 + a11.w * v11) * inv;
                            let d00 = textureLoad(diffAccumRead, p00, 0).xyz;
                            let d10 = textureLoad(diffAccumRead, p10, 0).xyz;
                            let d01 = textureLoad(diffAccumRead, p01, 0).xyz;
                            let d11 = textureLoad(diffAccumRead, p11, 0).xyz;
                            oldDiff = (d00 * v00 + d10 * v10 + d01 * v01 + d11 * v11) * inv;
                            let s00 = textureLoad(specAccumRead, p00, 0).xyz;
                            let s10 = textureLoad(specAccumRead, p10, 0).xyz;
                            let s01 = textureLoad(specAccumRead, p01, 0).xyz;
                            let s11 = textureLoad(specAccumRead, p11, 0).xyz;
                            oldSpec = (s00 * v00 + s10 * v10 + s01 * v01 + s11 * v11) * inv;
                            let mom00 = textureLoad(momentsRead, p00, 0).xy;
                            let mom10 = textureLoad(momentsRead, p10, 0).xy;
                            let mom01 = textureLoad(momentsRead, p01, 0).xy;
                            let mom11 = textureLoad(momentsRead, p11, 0).xy;
                            prevMomM1 = (mom00.x*v00 + mom10.x*v10 + mom01.x*v01 + mom11.x*v11) * inv;
                            prevMomM2 = (mom00.y*v00 + mom10.y*v10 + mom01.y*v01 + mom11.y*v11) * inv;
                            let staticCap = mix(8.0, 256.0, smoothstep(0.15, 0.7, primaryRough));
                            let movingCap = mix(8.0, 64.0,  smoothstep(0.15, 0.7, primaryRough));
                            if (primaryMoved) {
                                pixelFC = min(prevFC * 0.5, movingCap);
                            } else {
                                pixelFC = min(prevFC * 0.5, staticCap);
                            }
                            reprojOk = true;
                        }
                    }
                }
            }
            if (!reprojOk) {
                pixelFC = 0.0;
                oldDiff = vec3<f32>(0.0);
                oldSpec = vec3<f32>(0.0);
            }
        }
)"
R"(
        // NaN guard + hard/relative clamp.
        let sampleClean = select(vec3<f32>(0.0), sample, sample.x == sample.x);
        let oldClean    = select(vec3<f32>(0.0), old,    old.x == old.x);
        let lum3 = vec3<f32>(0.2126, 0.7152, 0.0722);
        var clamped = sampleClean;
        let rawLum = dot(sampleClean, lum3);
        let hardCap = 12.0;
        if (rawLum > hardCap) {
            clamped = sampleClean * (hardCap / rawLum);
        }
        if (pixelFC > 8.0) {
            let avgLum = max(dot(oldClean, lum3), 0.05);
            let smpLum = dot(clamped, lum3);
            let maxLum = avgLum * 7.0;
            if (smpLum > maxLum) {
                clamped = clamped * (maxLum / smpLum);
            }
        }

        let alpha = 1.0 / (pixelFC + 1.0);
        if (pixelFC < 1024.0) {
            var blended = oldClean * (1.0 - alpha) + clamped * alpha;
            const DEBUG_VIS_PIXEL_FC = false;
            if (DEBUG_VIS_PIXEL_FC) {
                let fcN = clamp(log2(pixelFC + 1.0) / 8.0, 0.0, 1.0);
                blended = vec3<f32>(fcN, fcN * fcN, fcN * fcN * fcN);
            }
            const DEBUG_VIS_RAW_SAMPLE = false;
            if (DEBUG_VIS_RAW_SAMPLE) {
                blended = sampleClean / (sampleClean + vec3<f32>(1.0));
            }
            textureStore(accumWrite, pixel, vec4<f32>(blended, pixelFC + 1.0));
        } else {
            textureStore(accumWrite, pixel, vec4<f32>(oldClean, pixelFC));
        }

        // Moments
        let sampleLum = dot(clamped, lum3);
        let momAlpha = max(alpha, 0.1);
        var m1 = prevMomM1;
        var m2 = prevMomM2;
        if (pixelFC < 1.0) { m1 = sampleLum; m2 = sampleLum * sampleLum; }
        else {
            m1 = m1 * (1.0 - momAlpha) + sampleLum * momAlpha;
            m2 = m2 * (1.0 - momAlpha) + sampleLum * sampleLum * momAlpha;
        }
        textureStore(momentsWrite, pixel, vec4<f32>(m1, m2, 0.0, 0.0));

        // Split accum (diffuse/specular)
        {
            let diffSample = select(vec3<f32>(0.0), ptResult.diff, ptResult.diff.x == ptResult.diff.x);
            let specSample = select(vec3<f32>(0.0), ptResult.spec, ptResult.spec.x == ptResult.spec.x);
            var dClamped = diffSample;
            let dLum = dot(diffSample, lum3);
            if (dLum > hardCap) { dClamped = diffSample * (hardCap / dLum); }
            let specHardCap = mix(3.0, 8.0, primaryRough);
            var sClamped = specSample;
            let sLum = dot(specSample, lum3);
            if (sLum > specHardCap) { sClamped = specSample * (specHardCap / sLum); }
            if (pixelFC < 1024.0) {
                textureStore(diffAccumWrite, pixel, vec4<f32>(oldDiff * (1.0 - alpha) + dClamped * alpha, pixelFC + 1.0));
                textureStore(specAccumWrite, pixel, vec4<f32>(oldSpec * (1.0 - alpha) + sClamped * alpha, pixelFC + 1.0));
            } else {
                textureStore(diffAccumWrite, pixel, vec4<f32>(oldDiff, pixelFC));
                textureStore(specAccumWrite, pixel, vec4<f32>(oldSpec, pixelFC));
            }
        }

        // hitMesh includes touchedMoved — written here since it depends on runBounces output.
        textureStore(hitMeshWrite, pixel, vec4<f32>(f32(primaryMeshIdx), f32(state.primaryMatIdx), select(0.0, 1.0, touchedMoved), 0.0));
    }
}

)";

// Build a raycaster or path tracer shader by concatenating common + specific code.
std::string buildRtShader(bool hasEnvCdf) {
    std::string src = std::string(csSharedDefsWGSL) + "\n" +
                      csCommonWGSL + "\n" +
                      csPathTraceWGSL + "\n" +
                      csRunBouncesWGSL1 + "\n" +
                      csRunBouncesWGSL2 + "\n" +
                      csPrimaryShadeWGSL1 + "\n" +
                      csPrimaryShadeWGSL2 + "\n" +
                      csPrimaryShadeWGSL3 + "\n" +
                      csPathTraceWGSL2 + "\n" +
                      csPathTraceWGSL3;
    const std::string marker = "/*ENV_CDF_FLAG*/false";
    auto pos = src.find(marker);
    if (pos != std::string::npos) {
        src.replace(pos, marker.size(), hasEnvCdf ? "true" : "false");
    }
    return src;
}

}// namespace threepp::wgpu_pt
