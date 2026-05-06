

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
    envIntensity:  vec4<f32>,  // x = unused (always 1.0), y = envWidth, z = envHeight, w = hasEnvCDF
    bgColor:       vec4<f32>,  // xyz = color, w = mode: 0=sky gradient, 1=solid color, 2=equirect tex (bgTex)
    params:        vec4<f32>,  // x = maxBounces
    emissiveInfo:  vec4<f32>,  // x = emissive triangle count, y = total emissive power, z = fireflyCap
    restirParams:  vec4<f32>,  // x = enabled, y = M_clamp, z = emissiveMoved, w = unused (always 1.0)
    bvhAux:        vec4<u32>,  // .x = bvhRootIdx (0 = normal root, >0 = overlay combined root)
    lens:          vec4<f32>,  // x = fStop (0=pinhole), y = focusDistance, z = blades (f32), w = apertureRotation
    fog:           vec4<f32>,  // xyz = sigma_t (per-channel extinction); w = enabled (1=on, 0=off)
    fogColor:      vec4<f32>,  // xyz = inscatter tint; w = g (HG asymmetry, reserved)
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
// bindings 1 and 2 were the combined main accum (read/write). Removed 2026-04-23
// — normal display reads diffAccum+specAccum directly; combined accum was dead
// bandwidth + 64MB VRAM at 1080p. Sky/AOV/rt_accum_main now operate only on the
// split pair.
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
@group(0) @binding(20) var reservoirWWrite: texture_storage_2d<rgba16float, write>;
@group(0) @binding(21) var momentsRead:     texture_2d<f32>;
@group(0) @binding(22) var momentsWrite:    texture_storage_2d<rgba16float, write>;
@group(0) @binding(23) var diffAccumRead:   texture_2d<f32>;
@group(0) @binding(24) var diffAccumWrite:  texture_storage_2d<rgba32float, write>;
@group(0) @binding(25) var specAccumRead:   texture_2d<f32>;
@group(0) @binding(26) var specAccumWrite:  texture_storage_2d<rgba32float, write>;
@group(0) @binding(27) var<storage, read> rtMotionMats: array<mat4x4<f32>>;  // prevWorld * inverse(curWorld) per mesh
@group(0) @binding(28) var giResRead:    texture_2d<f32>;
@group(0) @binding(29) var giResWrite:   texture_storage_2d<rgba32float, write>;
@group(0) @binding(30) var giResWRead:   texture_2d<f32>;
@group(0) @binding(31) var giResWWrite:  texture_storage_2d<rgba16float, write>;
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

// Stage F1 (wavefront prep): dead-lane compaction.  A separate compaction kernel
// (rt_compact_main) scans pathStateBuf after rt_main and writes pixel indices of
// paths that need bounce+accumulation work into aliveQueue.  rt_bounces_main then
// pulls from aliveQueue[i] instead of iterating all pixels, so warps are packed
// with live work instead of ~30% skipAccum lanes.
@group(0) @binding(39) var<storage, read_write> aliveQueue: array<u32>;
@group(0) @binding(40) var<storage, read_write> aliveCount: atomic<u32>;
// F2a bounce1 split: rt_bounce1_main processes ONE bounce iteration (i=1) for
// each pixel in aliveQueue, writes updated state back to pathStateBuf.
// rt_bounces_main then reads the SAME aliveQueue but its runBounces loop
// starts at i=2 (the i=1 work is already done).  The architectural win is
// register-pressure relief: bounce1 kernel doesn't need to carry i=2+ state,
// and bounces kernel doesn't need to carry bounce1's ReSTIR GI reservoir
// state.  Lower register pressure → higher occupancy → more concurrent
// warps hiding BVH + shadow-ray latency.
// F2b update: alive1Queue added below for actual inter-bounce compaction so
// rt_bounces_main dispatches on ONLY bounce1-survivors (~12% of pixels on
// Bistro).  Accumulation moves to a new rt_accum_main kernel that iterates
// aliveQueue so dead-at-bounce1 paths still get their primary radiance
// written to accum textures.
@group(0) @binding(42) var<storage, read_write> bounce1Counter: atomic<u32>;
@group(0) @binding(43) var<storage, read_write> alive1Queue: array<u32>;
@group(0) @binding(44) var<storage, read_write> alive1Count: atomic<u32>;
@group(0) @binding(45) var<storage, read_write> accumCounter: atomic<u32>;
// F2c: material sort before bounce1 for warp coherence.  aliveQueue is
// bucket-sorted by primaryMatIdx % 256 into sortedAliveQueue.  bounce1 reads
// sortedAliveQueue instead so adjacent warp lanes execute coherent BRDF code.
// 256 buckets is enough to cover typical scene material counts (Bistro has
// ~100); hash collisions merely reduce coherence, not correctness.
@group(0) @binding(46) var<storage, read_write> matBucketCount: array<atomic<u32>, 256>;
@group(0) @binding(47) var<storage, read_write> matBucketOffset: array<u32, 256>;
@group(0) @binding(48) var<storage, read_write> sortedAliveQueue: array<u32>;
@group(0) @binding(49) var<storage, read_write> sortCounter: atomic<u32>;
// Dedicated work-queue counter for rt_compact_main.  Can't share with
// bounceCounter because a CPU-side reset between the two passes can't
// interleave with GPU work inside the same command encoder.
@group(0) @binding(41) var<storage, read_write> compactCounter: atomic<u32>;


const MAX_TEX_SLOTS: i32 = 1024;
const EMPTY_CHILD: i32 = -2147483648;  // INT_MIN — sentinel for unused BVH4 child slots

fn TILE_SIZE() -> i32 { return max(i32(rt.spp.y), 1); }

// --- Fog / homogeneous participating media (v1: Beer-Lambert only) ---
fn fogEnabled() -> bool { return rt.fog.w > 0.5; }
fn fogTransmittance(dist: f32) -> vec3<f32> {
    // exp(-sigma_t * dist); clamp dist to avoid overflow on 1e30 miss sentinels.
    let d = min(max(dist, 0.0), 1e6);
    return exp(-rt.fog.xyz * d);
}


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

// Turquin 2018 / Kulla-Conty 2017 multi-scattering energy compensation LUT.
// 32×32 R32Float: u = cos θ_o in (0,1], v = α = roughness² in (0,1].
// Stored value is E(cos_o, α) with F0=1 — the single-scattering energy
// retained by the VNDF-sampled GGX estimator (see WgpuPathTracerGgxLut.hpp).
@group(0) @binding(50) var ggxELut: texture_2d<f32>;

// Bilinear sample of the GGX LUT.  cos_o and alpha are clamped to (0,1].
// The CPU builder places cells at (i+0.5)/N for cos_o (column) and (i+1)/N
// for α (row), so we align UV to those centers before interpolating.
fn sampleGgxELut(cos_o: f32, alpha: f32) -> f32 {
    let N: f32 = 32.0;
    let co = clamp(cos_o, 1e-3, 1.0);
    let al = clamp(alpha, 1e-3, 1.0);
    let fx = clamp(co * N - 0.5, 0.0, N - 1.0);
    let fy = clamp(al * N - 1.0, 0.0, N - 1.0);
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    let x1 = min(x0 + 1, 31);
    let y1 = min(y0 + 1, 31);
    let tx = fx - f32(x0);
    let ty = fy - f32(y0);
    let e00 = textureLoad(ggxELut, vec2<i32>(x0, y0), 0).r;
    let e10 = textureLoad(ggxELut, vec2<i32>(x1, y0), 0).r;
    let e01 = textureLoad(ggxELut, vec2<i32>(x0, y1), 0).r;
    let e11 = textureLoad(ggxELut, vec2<i32>(x1, y1), 0).r;
    return mix(mix(e00, e10, tx), mix(e01, e11, tx), ty);
}

// Turquin 2018 multi-scattering compensation factor.
// Given E from the LUT and F_avg ≈ (20·F0 + 1)/21 (Kulla-Conty closed form),
// restores the missing (1-E) energy via (1 + F_avg * (1-E)/E).
fn msCompensation(F0: vec3<f32>, cos_o: f32, alpha: f32) -> vec3<f32> {
    let E = sampleGgxELut(cos_o, alpha);
    let invE = 1.0 / max(E, 1e-3);
    let F_avg = (20.0 * F0 + vec3<f32>(1.0)) / 21.0;
    let comp = vec3<f32>(1.0) + F_avg * (max(0.0, 1.0 - E) * invE);
    // Upper bound protects against LUT noise at extreme grazing angles.
    // 2.0 was too tight: at α=1 and cos_o < ~0.3, legitimate comp exceeds 2
    // and got clipped, costing ~20% on rough-metal furnace.
    return clamp(comp, vec3<f32>(1.0), vec3<f32>(4.0));
}

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

// Equirectangular lookup helper (shared by sampleEnv and sampleBackground).
// three.js / GL convention: u = atan2(z, x) / (2π) + 0.5; v = asin(y) / π + 0.5
fn equirectUV(d: vec3<f32>) -> vec2<f32> {
    let nd  = normalize(d);
    let phi = atan2(nd.z, nd.x);
    let theta = asin(clamp(nd.y, -1.0, 1.0));
    return vec2<f32>(0.5 + phi / (2.0 * PI), 0.5 + theta / PI);
}

// Bilinear equirect lookup: wrap X (periodic), clamp Y (poles).
// Nearest-neighbor caused sub-pixel jitter shimmer on primary-ray sky samples,
// since pxFC caps at 32 (alpha = 1/33) and can't fully smooth per-frame
// texel hopping on HDR skies with bright regions.
fn sampleEquirectBilinear(tex: texture_2d<f32>, uv: vec2<f32>) -> vec3<f32> {
    let sz    = vec2<f32>(textureDimensions(tex, 0));
    let texel = uv * sz - vec2<f32>(0.5);
    let t0    = floor(texel);
    let f     = texel - t0;
    let sx    = i32(sz.x);
    let sy    = i32(sz.y);
    let ix0   = ((i32(t0.x) % sx) + sx) % sx;
    let ix1   = (ix0 + 1) % sx;
    let iy0   = clamp(i32(t0.y),     0, sy - 1);
    let iy1   = clamp(i32(t0.y) + 1, 0, sy - 1);
    let c00   = textureLoad(tex, vec2<i32>(ix0, iy0), 0).xyz;
    let c10   = textureLoad(tex, vec2<i32>(ix1, iy0), 0).xyz;
    let c01   = textureLoad(tex, vec2<i32>(ix0, iy1), 0).xyz;
    let c11   = textureLoad(tex, vec2<i32>(ix1, iy1), 0).xyz;
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// IBL environment lighting (scene.environment).  Returns BLACK when no environment is set.
fn sampleEnv(d: vec3<f32>) -> vec3<f32> {
    let mode = i32(rt.envColor.w);
    if (mode == 1) {
        return rt.envColor.xyz;
    } else if (mode == 2) {
        return sampleEquirectBilinear(envTex, equirectUV(d));
    }
    return vec3<f32>(0.0);  // no environment = no IBL
}

// Background color for ray misses (scene.background).
fn sampleBackground(d: vec3<f32>) -> vec3<f32> {
    let mode = i32(rt.bgColor.w);
    if (mode == 1) {
        return rt.bgColor.xyz;
    } else if (mode == 2) {
        return sampleEquirectBilinear(bgTex, equirectUV(d));
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
    let F_spec = schlick(VdotH, F0);
    let G      = ggxG1(NdotV, alpha) * ggxG1(NdotL, alpha);
    // K-C additive diffuse ms-comp: energy the specular layer returns to the
    // diffuse substrate via multiple internal bounces (Kulla & Conty 2017).
    let E_v    = sampleGgxELut(NdotV, alpha);
    let F_avg  = (20.0 * F0 + vec3<f32>(1.0)) / 21.0;
    let kcDiff = albedo * (1.0 - metalness) * F_avg * max(0.0, 1.0 - E_v) / PI;
    return BrdfResult(
        albedo * (1.0 - metalness) / PI + kcDiff,
        D * F_spec * G / max(4.0 * NdotV * NdotL, 1e-6));
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

fn makeRay(px: vec2<f32>, res: vec2<f32>, apBn: vec2<f32>) -> Ray {
    let aspect = res.x / res.y;
    let ndc = vec2<f32>((px.x / res.x) * 2.0 - 1.0,
                         1.0 - (px.y / res.y) * 2.0);
    var ray: Ray;
    ray.origin = rt.camOri.xyz;
    ray.dir    = normalize(rt.camFwd.xyz
                         + rt.camRgt.xyz * (ndc.x * rt.tanHalfFov.x * aspect)
                         + rt.camUp.xyz  * (ndc.y * rt.tanHalfFov.x));

    let fStop = rt.lens.x;
    if (fStop <= 0.0) { return ray; }

    let apertureRadius = (rt.tanHalfFov.x / fStop) * 0.5;
    let focusDist      = rt.lens.y;
    let blades         = i32(rt.lens.z);
    let rot            = rt.lens.w;

    var apOffset: vec2<f32>;
    if (blades < 3) {
        let r   = apertureRadius * sqrt(apBn.x);
        let phi = 6.2831853 * apBn.y;
        apOffset = vec2<f32>(r * cos(phi), r * sin(phi));
    } else {
        let n      = f32(blades);
        let sector = floor(apBn.x * n);
        let u      = fract(apBn.x * n);
        var a      = u;
        var b      = apBn.y;
        if (a + b > 1.0) { a = 1.0 - a; b = 1.0 - b; }
        let t0 = 6.2831853 * sector / n + rot;
        let t1 = 6.2831853 * (sector + 1.0) / n + rot;
        let p0 = vec2<f32>(cos(t0), sin(t0));
        let p1 = vec2<f32>(cos(t1), sin(t1));
        apOffset = apertureRadius * (a * p0 + b * p1);
    }

    let cosFwd     = max(dot(ray.dir, rt.camFwd.xyz), 1e-6);
    let focalPoint = ray.origin + ray.dir * (focusDist / cosFwd);
    let newOrigin  = ray.origin + rt.camRgt.xyz * apOffset.x + rt.camUp.xyz * apOffset.y;
    ray.origin = newOrigin;
    ray.dir    = normalize(focalPoint - newOrigin);
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
        return sampleEnv(lightPos);
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
        if (sh.texSlot >= 0.0) { shAlbedo *= srgbToLinear(sampleAtlas(sh.uv, sh.texSlot)); }
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
    // Volumetric attenuation across the full shadow-ray path (Beer-Lambert).
    // The ray travels `maxDist` to reach the light; fog dims the light source
    // along the way. Short-circuited in the hit case above (atten already 0).
    // Directional lights and env NEE use maxDist=1e30 as an "infinite-ray"
    // sentinel — treat those sources as sitting beyond the fog volume (sun
    // through atmosphere), otherwise fogTransmittance clamps to 1e6 and the
    // exponent zeros out atten, killing god rays from distant lights.
    if (fogEnabled() && maxDist < 1e20) { atten *= fogTransmittance(maxDist); }
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
    // Turquin 2018 MS compensation on the specular lobe (Kulla-Conty) —
    // restores the energy lost by single-scattering GGX so NEE matches the
    // energy conservation applied at the BRDF-sampling site.
    let msC = msCompensation(F0, max(1e-4, dot(n, wo)), alpha);
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
    var lobeSum = (brdf.f_diff + brdf.f_spec * msC) * (1.0 - ccWeight);
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
    // Turquin 2018 MS compensation on the specular lobe (see evalBrdfFull).
    let msC = msCompensation(F0, max(1e-4, dot(n, wo)), alpha);
    var ccWeight = 0.0;
    if (clearcoat > 0.0) {
        let ccF0 = 0.04;
        let NdotV_cc = max(0.0, dot(n, wo));
        let ccFresnel = ccF0 + (1.0 - ccF0) * pow(1.0 - NdotV_cc, 5.0);
        ccWeight = clearcoat * ccFresnel;
    }
    var d = brdf.f_diff * (1.0 - ccWeight);
    var s = brdf.f_spec * msC * (1.0 - ccWeight);
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
