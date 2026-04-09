
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuBuffer.hpp"
#include "threepp/renderers/wgpu/WgpuComputePipeline.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"

#include <tuple>

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <webgpu/webgpu.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <iostream>
#ifndef __EMSCRIPTEN__
#include <future>
#endif
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <webgpu/wgpu.h>

using namespace threepp;

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------
namespace {

constexpr int MAX_TEX_SLOTS = 256;
constexpr int DEFAULT_TILE_SIZE = 1024;
constexpr int ATLAS_WIDTH = 8192;  // fixed atlas width; cols = ATLAS_WIDTH / tileSize
constexpr int TRI_TEX_HEIGHT = 12;
constexpr int MAT_TEX_HEIGHT = 19;
constexpr int TEX_PAGE_WIDTH = 8192;

// Initial placeholder capacities — grown dynamically as scenes demand more.
constexpr int INIT_TRI_CAP  = 1;
constexpr int INIT_MAT_CAP  = 1;
constexpr int INIT_MESH_CAP = 1;

// objTriBuf is 48 floats (192 bytes) per tri.
// Max tri count is computed at runtime from the device's maxStorageBufferBindingSize.
constexpr size_t BYTES_PER_TRI = 48 * sizeof(float);  // 192

static int nextPow2(int v) {
    if (v <= 0) return 1;
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}

static int triTexPages(int triCap) {
    return (triCap + TEX_PAGE_WIDTH - 1) / TEX_PAGE_WIDTH;
}

// ---------------------------------------------------------------------------
// WGSL compute shader — common code shared by raycaster and path tracer
// ---------------------------------------------------------------------------
// Shared WGSL definitions used by multiple shaders (VT, refit, RT).
constexpr const char* csSharedDefsWGSL = R"(
const TRI_PAGE_W:  i32 = 8192;
const TRI_PAGE_H:  i32 = 12;
const MAX_LEAF_TRIS: i32 = 8;

fn triCoord(ti: i32, row: i32) -> vec2<i32> {
    return vec2<i32>(ti % TRI_PAGE_W, (ti / TRI_PAGE_W) * TRI_PAGE_H + row);
}
)";

constexpr const char* csCommonWGSL = R"(

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
@group(0) @binding(27) var stableGBufWrite: texture_storage_2d<rgba16float, write>;
@group(0) @binding(28) var stableGBufRead:  texture_2d<f32>;
@group(0) @binding(29) var<storage, read> rtMotionMats: array<mat4x4<f32>>;  // prevWorld * inverse(curWorld) per mesh
@group(0) @binding(30) var giResRead:    texture_2d<f32>;
@group(0) @binding(31) var giResWrite:   texture_storage_2d<rgba32float, write>;
@group(0) @binding(32) var giResWRead:   texture_2d<f32>;
@group(0) @binding(33) var giResWWrite:  texture_storage_2d<rgba32float, write>;
@group(0) @binding(34) var giResLoRead:  texture_2d<f32>;
@group(0) @binding(35) var giResLoWrite: texture_storage_2d<rgba16float, write>;

const MAX_TEX_SLOTS: i32 = 256;
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
    attenuationColor: vec3<f32>,
    attenuationDist:  f32,
    clearcoat:        f32,
    clearcoatAlpha:   f32,
    ao:               f32,
    sheenColor:       vec3<f32>,
    sheenRoughness:   f32,
    specularColor:    vec3<f32>,
    specularIntensity: f32,
    dispersion:       f32,
    thickness:        f32,
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

fn sampleAtlas(uv: vec2<f32>, texSlot: f32) -> vec3<f32> {
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
    let blended = mix(mix(c00, c10, wx), mix(c01, c11, wx), wy);
    return blended.xyz;
}

fn sampleAtlasAlpha(uv: vec2<f32>, texSlot: f32) -> f32 {
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
    let a00 = textureLoad(texAtlas, vec2<i32>(ox + x0, oy + y0), layer, 0).w;
    let a10 = textureLoad(texAtlas, vec2<i32>(ox + x1, oy + y0), layer, 0).w;
    let a01 = textureLoad(texAtlas, vec2<i32>(ox + x0, oy + y1), layer, 0).w;
    let a11 = textureLoad(texAtlas, vec2<i32>(ox + x1, oy + y1), layer, 0).w;
    return mix(mix(a00, a10, wx), mix(a01, a11, wx), wy);
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
    let NdotL = max(0.001, abs(dot(n, wi)));
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

    // Back-face culling for single-sided materials (mat3.z = 0 means single-sided)
    if (mat3.z < 0.5) {
        let geoNormal = cross(v1 - v0, v2 - v0);
        if (dot(ray.dir, geoNormal) > 0.0) { return; }
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
            let uv1_01 = textureLoad(triData, triCoord(ti, 8), 0);
            let uv1_2  = textureLoad(triData, triCoord(ti, 9), 0).xy;
            let iuv1 = vec2<f32>(uv1_01.x, uv1_01.y) * w
                     + vec2<f32>(uv1_01.z, uv1_01.w) * isect.u
                     + uv1_2                          * isect.v;
            let tuv = transformUV(iuv0, iuv1, matIdx, 6);
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
                let h = pcg(pcg(u32(ti)) ^ pcg(bitcast<u32>(isect.t)));
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
    h.geoNormal = vec3<f32>(0.0); h.attenuationColor = vec3<f32>(1.0);
    h.attenuationDist = 0.0; h.clearcoat = 0.0; h.clearcoatAlpha = 0.0; h.ao = 1.0;
    h.sheenColor = vec3<f32>(0.0); h.sheenRoughness = 0.0;
    h.specularColor = vec3<f32>(1.0); h.specularIntensity = 1.0;
    h.dispersion = 0.0; h.thickness = 0.0;
    h.meshIdx = -1; h.matIdx = -1;

    let ti = rh.triIdx;
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

    // Per-channel transformed UVs — skip UV1 interpolation (2 triData reads) and
    // all 10 transformUV matData reads when all channels use identity UV0.
    let mat18 = textureLoad(matData, vec2<i32>(matIdx, 18), 0);
    var bcUV = iuv0; var mrUV = iuv0; var nmUV = iuv0; var emUV = iuv0; var aoUV = iuv0;
    if (mat18.z > 0.5) {
        let uv1_01 = textureLoad(triData, triCoord(ti, 8), 0);
        let uv1_2  = textureLoad(triData, triCoord(ti, 9), 0).xy;
        let iuv1   = vec2<f32>(uv1_01.x, uv1_01.y) * w
                   + vec2<f32>(uv1_01.z, uv1_01.w) * rh.u
                   + uv1_2                          * rh.v;
        bcUV = transformUV(iuv0, iuv1, matIdx, 6);   // baseColor
        mrUV = transformUV(iuv0, iuv1, matIdx, 8);   // metalRough
        nmUV = transformUV(iuv0, iuv1, matIdx, 10);  // normal
        emUV = transformUV(iuv0, iuv1, matIdx, 12);  // emissive
        aoUV = transformUV(iuv0, iuv1, matIdx, 14);  // occlusion
    }

    let n0 = textureLoad(triData, triCoord(ti, 3), 0).xyz;
    let n1 = textureLoad(triData, triCoord(ti, 4), 0).xyz;
    let n2 = textureLoad(triData, triCoord(ti, 5), 0).xyz;
    let sn = normalize(n0 * w + n1 * rh.u + n2 * rh.v);

    let isFrontFace = dot(ray.dir, sn) < 0.0;
    var finalNorm = select(-sn, sn, isFrontFace);

    // Normal mapping (uses normal-channel UV)
    let normalSlot = mat1.z;
    if (normalSlot >= 0.0) {
        let nmSample = sampleAtlas(nmUV, normalSlot);
        let nmTangent = nmSample * 2.0 - 1.0;
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
            finalNorm = normalize(T * nmTangent.x + B * nmTangent.y + finalNorm * nmTangent.z);
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

    // Geometric (flat) normal
    let geoNcross = cross(v1 - v0, v2 - v0);
    let geoNlen   = length(geoNcross);
    let geoN    = select(sn, geoNcross / geoNlen, geoNlen > 1e-8);
    let geoNorm = select(-geoN, geoN, isFrontFace);

    // Vertex color interpolation
    let vc01 = textureLoad(triData, triCoord(ti, 10), 0);
    let vc2  = textureLoad(triData, triCoord(ti, 11), 0);
    let cb2  = uv2_full.w;
    let col0 = vec3<f32>(vc01.x, vc01.y, vc01.z);
    let col1 = vec3<f32>(vc01.w, vc2.x, vc2.y);
    let col2 = vec3<f32>(vc2.z, vc2.w, cb2);
    let vcolor = col0 * w + col1 * rh.u + col2 * rh.v;

    h.point     = ray.origin + rh.t * ray.dir;
    h.normal    = finalNorm;
    h.geoNormal = geoNorm;
    h.albedo    = mat0.xyz * vcolor;
    h.shininess = shininess;
    h.uv        = bcUV;
    h.texSlot   = mat1.x;
    h.metalness = metalness;
    h.transmission = mat2.w;
    h.ior          = mat3.x;
    h.frontFace    = select(0.0, 1.0, isFrontFace);
    h.meshIdx      = i32(r1.w);
    h.matIdx       = matIdx;
    let mat4 = textureLoad(matData, vec2<i32>(matIdx, 4), 0);
    h.attenuationColor = mat4.xyz;
    h.attenuationDist  = mat4.w;
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

    // AO map (uses occlusion UV)
    h.ao = 1.0;
    let aoSlot = mat5.w;
    if (aoSlot >= 0.0) {
        h.ao = sampleAtlas(aoUV, aoSlot).x;
    }

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
    let uv1_01 = textureLoad(triData, triCoord(ti, 8), 0);
    let uv1_2  = textureLoad(triData, triCoord(ti, 9), 0).xy;
    let iuv1   = vec2<f32>(uv1_01.x, uv1_01.y) * w
               + vec2<f32>(uv1_01.z, uv1_01.w) * rh.u
               + uv1_2                          * rh.v;
    let bcUV = transformUV(iuv0, iuv1, matIdx, 6);
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
    stack[0] = 0; top = 1;

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

        // Push internal children — nearest last so it's popped first
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

// Lightweight primary-hit query: BVH traversal + normal/depth/meshIdx only.
// No material evaluation, no texture sampling, no UV interpolation.
// Used for the unjittered stable G-buffer that drives à-trous edge-stopping.
struct StableHit { t: f32, normal: vec3<f32>, meshIdx: i32, matIdx: i32 }

fn stablePrimaryHit(ray: Ray) -> StableHit {
    var sh: StableHit;
    sh.t = 0.0; sh.normal = vec3<f32>(0.0); sh.meshIdx = -1; sh.matIdx = -1;
    let rh = sceneHitRaw(ray, 1e30);
    if (rh.triIdx < 0) { return sh; }

    let ti = rh.triIdx;
    let w  = 1.0 - rh.u - rh.v;
    // Interpolate smooth normal from vertex normals
    let n0 = textureLoad(triData, triCoord(ti, 3), 0).xyz;
    let n1 = textureLoad(triData, triCoord(ti, 4), 0).xyz;
    let n2 = textureLoad(triData, triCoord(ti, 5), 0).xyz;
    let sn = normalize(n0 * w + n1 * rh.u + n2 * rh.v);

    // meshIdx from triData row 1, matIdx from row 0
    let r0 = textureLoad(triData, triCoord(ti, 0), 0);
    let r1 = textureLoad(triData, triCoord(ti, 1), 0);

    sh.t = rh.t;
    sh.normal = select(-sn, sn, dot(ray.dir, sn) < 0.0);
    sh.meshIdx = i32(r1.w);
    sh.matIdx  = i32(r0.w);
    return sh;
}

fn sceneHit(ray: Ray) -> Hit {
    let rh = sceneHitRaw(ray, 1e30);
    if (rh.triIdx < 0) {
        var h: Hit; h.t = 1e30; h.meshIdx = -1; h.matIdx = -1; h.transmission = 0.0; h.ior = 1.5;
        h.frontFace = 1.0; h.geoNormal = vec3<f32>(0.0); h.attenuationColor = vec3<f32>(1.0);
        h.attenuationDist = 0.0; h.clearcoat = 0.0; h.clearcoatAlpha = 0.0; h.ao = 1.0;
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
    stack[0] = 0; top = 1;

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

        // No sorting — just push all hit internal children
        if (dists.x < 1e30 && ci0 >= 0) { stack[top] = ci0; top++; }
        if (dists.y < 1e30 && ci1 >= 0) { stack[top] = ci1; top++; }
        if (dists.z < 1e30 && ci2 >= 0) { stack[top] = ci2; top++; }
        if (dists.w < 1e30 && ci3 >= 0) { stack[top] = ci3; top++; }
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
    let lDist = rt.lightType[li].w;
    let lDecay = rt.lightDir[li].w;
    if (lDist > 0.0 && ltype != 1) {
        lc *= pow(max(1.0 - le.dist / lDist, 0.0), lDecay);
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
fn traceShadowRay(origin: vec3<f32>, normal: vec3<f32>, dir: vec3<f32>,
                  maxDist: f32, maxBounces: i32) -> vec3<f32> {
    var sr: Ray;
    sr.origin = origin + normal * 1e-3;
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
    let xi = rand(seed) * totalPower;
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
    let su1 = sqrt(rand(seed));
    let u2  = rand(seed);
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
constexpr const char* csPathTraceWGSL = R"(

@group(0) @binding(12) var envCdfTex:     texture_2d<f32>;  // conditional CDF (per-row), R32Float
@group(0) @binding(13) var envMargTex:    texture_2d<f32>;  // marginal CDF (1-column), R32Float

const HAS_ENV_CDF: bool = /*ENV_CDF_FLAG*/false;

// R2 quasi-random sequence (Martin Roberts) — low-discrepancy 2D points
// Uses the plastic constant for optimal 2D stratification
const R2_A1: f32 = 0.7548776662466927;  // 1/phi2
const R2_A2: f32 = 0.5698402909980532;  // 1/phi2^2
fn r2Seq(n: u32) -> vec2<f32> {
    return fract(vec2<f32>(f32(n) * R2_A1, f32(n) * R2_A2));
}

fn cosineHemisphere(n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let u1  = rand(seed);
    let u2  = rand(seed);
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
    // Sample projected disk
    let u1  = rand(seed);
    let u2  = rand(seed);
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

fn restirLuminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Evaluate unshadowed target function for a reservoir sample.
// Returns luminance of (BRDF * Le * NdotL * geometry).
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

    // Clamp roughness floor for the target PDF to prevent extreme GGX D values
    // on low-roughness surfaces. Without this, the specular peak creates
    // p_hat swings of 1e5+ between smooth/rough regions, producing stable
    // roughness-correlated brightness patches in the reservoir weights.
    let safeAlpha = max(alpha, 0.1);
    let brdf = evalBrdf(wo, ln, normal, albedo, metalness, safeAlpha, F0);
    let lobeSum = brdf.f_diff + brdf.f_spec;

    return restirLuminance(lobeSum * NdotL * lightLe);
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
    let ui = u32(idx);
    let bit  = ui & 31u;
    let wi   = ui >> 5u;  // 0..3
    var word: u32 = 0u;
    if      (wi == 0u) { word = rt.movedMeshBits.x; }
    else if (wi == 1u) { word = rt.movedMeshBits.y; }
    else if (wi == 2u) { word = rt.movedMeshBits.z; }
    else               { word = rt.movedMeshBits.w; }
    return ((word >> bit) & 1u) != 0u;
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

struct SplitRadiance { diff: vec3<f32>, spec: vec3<f32> }
)";

// Second half of the path trace shader (split for MSVC 16380-byte string literal limit)
constexpr const char* csPathTraceWGSL2 = R"(
fn pathTrace(ray_in: Ray, seed: ptr<function, u32>,
             pixel: vec2<i32>,
             maxBounces:     i32,
             primaryMeshIdx: ptr<function, u32>,
             primaryNormal:  ptr<function, vec3<f32>>,
             primaryDepth:   ptr<function, f32>,
             primaryAlbedo:  ptr<function, vec3<f32>>,
             primaryRough:   ptr<function, f32>,
             primaryMatIdx:  ptr<function, i32>,
             touchedMoved:   ptr<function, bool>) -> SplitRadiance {
    *primaryMeshIdx = 128u;
    *primaryNormal  = vec3<f32>(0.0);
    *primaryDepth   = 0.0;  // 0 = sky/no-hit sentinel for denoiser
    *primaryAlbedo  = vec3<f32>(1.0);  // default white (sky/miss: no demodulation)
    *primaryRough   = 1.0;  // sky: treat as fully rough (max history)
    *primaryMatIdx  = -1;   // sky: no material
    *touchedMoved   = false;
    var ray        = ray_in;
    var throughput = vec3<f32>(1.0);
    var diffRad    = vec3<f32>(0.0);
    var specRad    = vec3<f32>(0.0);
    var firstBounceSpec = false;  // determines routing for bounces > 0

    // Previous-bounce surface properties for MIS weighting
    var prevNormal    = vec3<f32>(0.0, 1.0, 0.0);
    var prevAlpha     = 0.0;
    var prevMetalness = 0.0;
    var prevWo        = vec3<f32>(0.0);
    var afterTransmission = false;
    var effectiveBounces = maxBounces;

    // Bounce-0 surface data for ReSTIR GI (captured at i==0, used at i==1)
    var b0Point  = vec3<f32>(0.0);
    var b0Normal = vec3<f32>(0.0, 1.0, 0.0);
    var b0Wo     = vec3<f32>(0.0);
    var b0Albedo = vec3<f32>(1.0);
    var b0F0     = vec3<f32>(0.04);
    var b0Metal  = 0.0;
    var b0Alpha  = 1.0;
    var b0MeshIdx = -1;
    var giResStored = false;

    for (var i = 0; i < maxBounces; i++) {
        if (i >= effectiveBounces) { break; }
        var h = sceneHit(ray);
        if (h.t >= 1e29) {
            // Primary ray miss: show background.  Bounced misses: use env IBL.
            if (i == 0) {
                diffRad += throughput * sampleBackground(ray.dir);
                // Store empty reservoir on primary miss
                if (rt.restirParams.x > 0.5) {
                    textureStore(reservoirWrite, pixel, vec4<f32>(0.0));
                    textureStore(reservoirWWrite, pixel, vec4<f32>(0.0));
                }
                if (rt.spp.x > 0.5) {
                    textureStore(giResWrite,   pixel, vec4<f32>(0.0));
                    textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
                    textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
                }
            } else {
                var envMisW = 1.0;
                if (HAS_ENV_CDF && rt.envColor.w > 1.5 && prevAlpha > 0.01) {
                    let pdf_env  = envImportancePdf(ray.dir);
                    let pdf_brdf = brdfPdf(prevWo, normalize(ray.dir), prevNormal, prevAlpha, prevMetalness);
                    envMisW = pdf_brdf / max(pdf_brdf + pdf_env, 1e-8);
                }
                addSplit(&diffRad, &specRad,
                    throughput * sampleEnv(ray.dir) * rt.envIntensity.x * envMisW,
                    rt.emissiveInfo.z, i, firstBounceSpec);
            }
            // Empty GI reservoir on miss at bounce 1
            if (i == 1 && rt.spp.x > 0.5 && !giResStored) {
                textureStore(giResWrite,   pixel, vec4<f32>(0.0));
                textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
                textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
                giResStored = true;
            }
            break;
        }
        if (i == 0) {
            *primaryMeshIdx = u32(h.meshIdx);
            *primaryNormal  = h.normal;
            *primaryDepth   = h.t;
            *primaryRough   = sqrt(h.shininess);  // linear roughness for TAA blend
            *primaryMatIdx  = h.matIdx;

            // Adaptive bounce cap — reduce bounces for diffuse/glossy surfaces.
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
        } else if (isMeshMoved(h.meshIdx) && u32(h.meshIdx) != *primaryMeshIdx) {
            // Only flag "touched moved" when the secondary bounce lands on a
            // DIFFERENT moved mesh than the primary — in that case relative
            // motion between two independently-moving meshes cannot be
            // reprojected.  If primary and secondary are on the same moved
            // mesh, the mesh's own motion matrix already reprojects the whole
            // rigid interaction correctly → no cap needed.
            *touchedMoved = true;
        }

        let emTriCount = i32(rt.emissiveInfo.x);
        let totalPower = rt.emissiveInfo.y;

        // Emissive hit with MIS balance heuristic
        if (length(h.emissive) > 0.0) {
            if (i == 0 || emTriCount == 0 || afterTransmission) {
                // Primary ray, no NEE available, or after transmission: full weight
                if (i == 0) {
                    diffRad += throughput * h.emissive;  // primary: physically correct, don't clamp
                } else {
                    addSplit(&diffRad, &specRad, throughput * h.emissive,
                             rt.emissiveInfo.z, i, firstBounceSpec);
                }
            } else {
                // MIS: BRDF sampling hit emissive — weight by balance heuristic
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

        if (i == 0) { *primaryAlbedo = albedo; }

        // Unlit: return flat color, no bouncing
        if (h.shininess < 0.0) {
            diffRad += throughput * albedo;
            if (i == 0 && rt.restirParams.x > 0.5) {
                textureStore(reservoirWrite, pixel, vec4<f32>(0.0));
                textureStore(reservoirWWrite, pixel, vec4<f32>(0.0));
            }
            if (i <= 1 && rt.spp.x > 0.5 && !giResStored) {
                textureStore(giResWrite,   pixel, vec4<f32>(0.0));
                textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
                textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
                giResStored = true;
            }
            break;
        }

        let wo = normalize(-ray.dir);
        let F0_h = computeF0(albedo, h.metalness, h.specularColor, h.specularIntensity);

        // Capture bounce-0 surface data for ReSTIR GI
        if (i == 0) {
            b0Point  = h.point;
            b0Normal = h.normal;
            b0Wo     = wo;
            b0Albedo = albedo;
            b0F0     = F0_h;
            b0Metal  = h.metalness;
            b0Alpha  = h.shininess;
            b0MeshIdx = h.meshIdx;
        }

        // ReSTIR GI only for diffuse/glossy primary surfaces — specular primaries
        // (mirrors, metals with low roughness) are better served by BRDF sampling.
        let useReSTIRGI = i == 1 && rt.spp.x > 0.5 && !afterTransmission
                          && h.transmission < 0.05 && b0Alpha > 0.05;
        var giLo = vec3<f32>(0.0);

        // --- NEE: ReSTIR DI at bounce 0, classic NEE at deeper bounces ---
        let lcount = i32(rt.lightCount.x);
        // Skip ReSTIR for transmissive/glass surfaces — direct illumination is
        // mostly irrelevant there (primary interaction is transmission, not reflection).
        // Applying ReSTIR NEE to a glass surface over-brightens it and makes it
        // appear opaque. Fall back to classic NEE for those hits.
        let useReSTIR = i == 0 && rt.restirParams.x > 0.5
                        && h.transmission < 0.05;

        if (useReSTIR) {
            // ======= ReSTIR DI: Initial candidate generation =======
            var reservoir = emptyReservoir();

            // 1. Analytical lights — evaluate ALL (typically 1-4, cheap).
            // Sampling only one caused visible flicker during camera motion
            // when temporal reuse breaks and each frame randomly picks a different light.
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
)"
R"(
            // 2. Emissive triangles — CDF samples
            if (emTriCount > 0 && totalPower > 0.0) {
                for (var ei = 0; ei < 4; ei++) {
                    let es = sampleEmissiveTriCdf(seed, totalPower, emTriCount);
                    let toLight = es.point - h.point;
                    let dist = length(toLight);
                    let ln_e = toLight / dist;
                    let cosLight = abs(dot(es.normal, -ln_e));
                    let NdotL_e = dot(h.normal, ln_e);
                    // Always count in M for correct RIS normalization.
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

            // 3. Environment — 1 importance sample
            if (HAS_ENV_CDF && rt.envColor.w > 1.5) {
                let envSample = sampleEnvImportance(seed);
                let envDir = envSample.xyz;
                let envPdf = envSample.w;
                // Always count in M for correct RIS normalization.
                reservoir.M += 1.0;
                if (dot(h.normal, envDir) > 0.0 && envPdf > 1e-8) {
                    let envCol = sampleEnv(envDir) * rt.envIntensity.x;
                    let p_hat_env = restirTargetPdf(h.point, h.normal, wo, albedo, h.metalness, h.shininess, F0_h,
                                                    envDir, -1.0, envCol);
                    if (p_hat_env > 0.0) {
                        let w = p_hat_env / envPdf;
                        reservoir.W_sum += w;
                        if (rand(seed) < w / max(reservoir.W_sum, 1e-20)) {
                            reservoir.lightPos  = envDir;
                            reservoir.lightType = -1.0;
                            reservoir.p_hat     = p_hat_env;
                        }
                    }
                }
            }

            finalizeReservoir(&reservoir);
)"
R"(
            // === TEMPORAL REUSE ===
            if (rt.frameCount.x > 0.0) {
                // Motion-vector reprojection: transform the current hit point back into
                // the PREVIOUS frame's world space when the hit mesh moved (rotated/translated).
                // Without this the reservoir temporal reuse fails on moving meshes — we'd
                // sample the wrong pixel's reservoir and the validation check would reject it,
                // forcing reservoir restart every frame → visible light-sampling noise.
                let primaryMoved = h.meshIdx >= 0 && h.meshIdx < 128 && isMeshMoved(h.meshIdx);
                var reprojPoint = h.point;
                var expectedPrevN = h.normal;
                if (primaryMoved) {
                    let M = rtMotionMats[h.meshIdx];
                    reprojPoint = (M * vec4<f32>(h.point, 1.0)).xyz;
                    // Rotate normal via 3x3 part (assumes rigid motion — uniform scale).
                    expectedPrevN = normalize((M * vec4<f32>(h.normal, 0.0)).xyz);
                }
                // Reproject current hit to previous frame pixel
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
                    // Round to nearest pixel (not truncate) to cancel jitter bias —
                    // Primary ray is pixel-center (no jitter), so prevU/V land at i+0.5.
                    // floor() gives the correct pixel index; +0.5 would overshoot by one.
                    let prevPx = vec2<i32>(i32(floor(prevU)), i32(floor(prevV)));

                    if (prevPx.x >= 0 && prevPx.y >= 0 &&
                        prevPx.x < i32(rt.iRes.x) && prevPx.y < i32(rt.iRes.y)) {

                        // Use stable (unjittered) g-buffer for validation — jittered
                        // normals/depth flip at silhouette edges causing reservoir
                        // resets every frame → noise spikes during motion.
                        let prevSGB = textureLoad(stableGBufRead, prevPx, 0);
                        let prevN = prevSGB.xyz;
                        let prevD = prevSGB.w;
                        // Compare prev-camera distance to reprojected point, not h.t
                        // (h.t is distance from current camera, but prevD is stored
                        // from previous camera's viewpoint).
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
                            rPrev.M     = min(prevWeight.y, rt.restirParams.y);
                            rPrev.W     = prevWeight.z;
                            rPrev.p_hat = prevWeight.w;

                            // Re-evaluate prev sample's target PDF at current shading point
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
)"
R"(
            // Snapshot reservoir before spatial reuse — stored for next-frame temporal reuse.
            // Spatial reuse is used for shading only; if we stored the post-spatial reservoir
            // a shadowed pixel that borrows a lit neighbour's light would keep failing visibility
            // and permanently accumulate W=0 (feedback loop → fades to black).
            let preSpReservoir = reservoir;

            // === SPATIAL REUSE — 8 random neighbours from previous frame ===
            // Reads last-frame reservoirs from a disk of radius ~20 px.
            // Geometry check (normal, depth) prevents bleeding across surfaces.
            // Uses one-frame-lagged neighbours — avoids a second dispatch and is
            // visually indistinguishable from same-frame spatial reuse.
            {
                let spMax = 5u;
                let mTarget = 20.0; // stop borrowing once we have enough confidence
                for (var spI = 0u; spI < spMax; spI++) {
                    if (reservoir.M >= mTarget) { break; }
                    let spAngle = rand(seed) * 2.0 * PI;
                    let spR     = sqrt(rand(seed)) * 20.0;
                    let spOff   = vec2<i32>(i32(spR * cos(spAngle)), i32(spR * sin(spAngle)));
                    if (all(spOff == vec2<i32>(0))) { continue; }
                    let spPx = clamp(pixel + spOff,
                                     vec2<i32>(0),
                                     vec2<i32>(i32(rt.iRes.x) - 1, i32(rt.iRes.y) - 1));

                    // Reject across surface boundaries — use stable g-buffer for
                    // consistent validation regardless of sub-pixel jitter.
                    let spSGB = textureLoad(stableGBufRead, spPx, 0);
                    if (dot(h.normal, spSGB.xyz) < 0.906 ||
                        abs(h.t - spSGB.w) / max(h.t, 1e-3) > 0.1) { continue; }

                    let spSmp = textureLoad(reservoirRead,  spPx, 0);
                    let spWt  = textureLoad(reservoirWRead, spPx, 0);
                    var rSp: Reservoir;
                    rSp.lightPos  = spSmp.xyz;
                    rSp.lightType = spSmp.w;
                    rSp.W_sum = spWt.x;
                    rSp.M     = min(spWt.y, 4.0); // low cap: spatial neighbours must not drown out this pixel's own candidates
                    rSp.W     = spWt.z;
                    rSp.p_hat = spWt.w;
                    if (rSp.W <= 0.0 || rSp.M <= 0.0) { continue; }

                    // Re-evaluate neighbour's chosen light at the CURRENT shading point
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

            // === VISIBILITY TEST — shadow ray from shading point ===
            var reservoirShadowAtten = vec3<f32>(0.0);
            if (reservoir.p_hat > 0.0 && reservoir.W > 0.0) {
                let rd = reservoirLightDir(reservoir.lightPos, reservoir.lightType, h.point);
                reservoirShadowAtten = traceShadowRay(h.point, h.normal, rd.dir, rd.maxDist, 2);
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
            // === STORE RESERVOIR ===
            // If visibility failed, reset M and W to 0 so next frame does not inherit
            // an occluded sample via temporal reuse.  This eliminates the "smeared shadow"
            // bias where high-p̂ but occluded samples propagate indefinitely through the
            // temporal reservoir.  Lit pixels accumulate M normally; shadowed pixels
            // restart fresh each frame (path-tracer accumulation still converges them).
            let visible = reservoirShadowAtten.x + reservoirShadowAtten.y + reservoirShadowAtten.z > 0.001;
            // Store pre-spatial reservoir so temporal reuse next frame reflects what
            // THIS pixel actually selected (with known good visibility), not what a
            // neighbour selected that may be occluded here.
            let rW = select(0.0, select(0.0, preSpReservoir.W, preSpReservoir.W == preSpReservoir.W), visible);
            let rM = select(0.0, preSpReservoir.M, visible);
            textureStore(reservoirWrite, pixel,
                vec4<f32>(preSpReservoir.lightPos, preSpReservoir.lightType));
            textureStore(reservoirWWrite, pixel,
                vec4<f32>(preSpReservoir.W_sum, rM, rW, preSpReservoir.p_hat));

        } else {
)";

// Fourth part: classic NEE + bounce loop tail + rt_main accumulation
constexpr const char* csPathTraceWGSL2b = R"(
        // ======= Classic NEE (bounces > 0 or ReSTIR disabled) =======

        // --- Analytical light NEE ---
        for (var li = 0; li < 4; li++) {
            if (li >= lcount) { break; }
            let le = evalAnalyticalLight(li, h.point);
            let NdotL = dot(h.normal, le.dir);
            if (NdotL <= 0.0) { continue; }
            let shadowAtten = traceShadowRay(h.point, h.normal, le.dir, le.dist - 1e-3, 2);
            if (shadowAtten.x + shadowAtten.y + shadowAtten.z > 0.001) {
                let cap = rt.emissiveInfo.z;
                if (i == 0) {
                    let bs = evalBrdfFullSplit(wo, le.dir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                              h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                    let shade = throughput * shadowAtten * NdotL * le.color;
                    addSplit(&diffRad, &specRad, shade * bs.diff, cap, 0, false);
                    addSplit(&diffRad, &specRad, shade * bs.spec, cap, 1, true);
                } else {
                    let lobeSum = evalBrdfFull(wo, le.dir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                               h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                    let neeContrib = shadowAtten * lobeSum * NdotL * le.color;
                    if (useReSTIRGI) {
                        giLo += neeContrib;
                    } else {
                        addSplit(&diffRad, &specRad, throughput * neeContrib, cap, i, firstBounceSpec);
                    }
                }
            }
        }
)"
R"(
        // --- Emissive surface NEE (power-weighted CDF sampling) ---
        if (emTriCount > 0) {
            let totalPower2 = rt.emissiveInfo.y;
            let es = sampleEmissiveTriCdf(seed, totalPower2, emTriCount);
            let toLight = es.point - h.point;
            let dist    = length(toLight);
            let ln      = toLight / dist;
            let NdotL   = dot(h.normal, ln);
            let cosLight = abs(dot(es.normal, -ln));

            if (NdotL > 0.0 && cosLight > 1e-6) {
                let emAtten = traceShadowRay(h.point, h.normal, ln, dist - 1e-2, 2);
                if (emAtten.x + emAtten.y + emAtten.z > 0.001) {
                    let eMatIdx = i32(textureLoad(triData, triCoord(es.triIdx, 0), 0).w);
                    let emColor = textureLoad(matData, vec2<i32>(eMatIdx, 2), 0).xyz;
                    let pdf = (es.power * dist * dist) / (totalPower2 * es.area * cosLight);
                    let pdf_brdf_nee = brdfPdf(wo, ln, h.normal, h.shininess, h.metalness);
                    let w_light = pdf / max(pdf + pdf_brdf_nee, 1e-8);
                    let cap = rt.emissiveInfo.z;
                    if (i == 0) {
                        let bs = evalBrdfFullSplit(wo, ln, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                                   h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                        let shade = throughput * emAtten * NdotL * emColor * w_light / pdf;
                        addSplit(&diffRad, &specRad, shade * bs.diff, cap, 0, false);
                        addSplit(&diffRad, &specRad, shade * bs.spec, cap, 1, true);
                    } else {
                        let lobeSum3 = evalBrdfFull(wo, ln, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                                     h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                        let emNeeContrib = emAtten * lobeSum3 * NdotL * emColor * w_light / pdf;
                        if (useReSTIRGI) {
                            giLo += emNeeContrib;
                        } else {
                            addSplit(&diffRad, &specRad, throughput * emNeeContrib, cap, i, firstBounceSpec);
                        }
                    }
                }
            }
        }

        // --- Environment map NEE (importance-sampled) ---
        if (HAS_ENV_CDF && rt.envColor.w > 1.5 && h.shininess > 0.01 && i < 4) {
            let envSample = sampleEnvImportance(seed);
            let envDir    = envSample.xyz;
            let envPdf    = envSample.w;
            let envNdotL  = dot(h.normal, envDir);
            if (envNdotL > 0.0 && envPdf > 1e-8) {
                let envAtten = traceShadowRay(h.point, h.normal, envDir, 1e30, 2);
                if (envAtten.x + envAtten.y + envAtten.z > 0.001) {
                    let envCol = sampleEnv(envDir) * rt.envIntensity.x;
                    let pdf_brdf_env = brdfPdf(wo, envDir, h.normal, h.shininess, h.metalness);
                    let w_env = envPdf / max(envPdf + pdf_brdf_env, 1e-8);
                    let cap = rt.emissiveInfo.z;
                    if (i == 0) {
                        let bs = evalBrdfFullSplit(wo, envDir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                                   h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                        let shade = throughput * envAtten * envNdotL * envCol * w_env / envPdf;
                        addSplit(&diffRad, &specRad, shade * bs.diff, cap, 0, false);
                        addSplit(&diffRad, &specRad, shade * bs.spec, cap, 1, true);
                    } else {
                        let lobeSum4 = evalBrdfFull(wo, envDir, h.normal, albedo, h.metalness, h.shininess, F0_h,
                                                     h.sheenColor, h.sheenRoughness, h.clearcoat, h.clearcoatAlpha);
                        let envNeeContrib = envAtten * lobeSum4 * envNdotL * envCol * w_env / envPdf;
                        if (useReSTIRGI) {
                            giLo += envNeeContrib;
                        } else {
                            addSplit(&diffRad, &specRad, throughput * envNeeContrib, cap, i, firstBounceSpec);
                        }
                    }
                }
            }
        }

        } // end ReSTIR vs classic NEE

        // When ReSTIR is globally enabled but was skipped for this pixel (glass /
        // high-transmission surface), zero out the reservoir so stale data from a
        // prior opaque frame at the same pixel location never bleeds into temporal reuse.
        if (i == 0 && rt.restirParams.x > 0.5 && !useReSTIR) {
            textureStore(reservoirWrite,  pixel, vec4<f32>(0.0));
            textureStore(reservoirWWrite, pixel, vec4<f32>(0.0));
        }

        // ======= ReSTIR GI: Build reservoir, temporal reuse, shade, store =======
        // Skip when secondary hit is too close to primary (seam geometry → G factor spike)
        let giSecDist2 = dot(h.point - b0Point, h.point - b0Point);
        if (useReSTIRGI && !giResStored && giSecDist2 > 0.04) {
            // --- Build 1-sample reservoir from captured Lo ---
            let giP_hat = giTargetPdf(b0Point, b0Normal, b0Wo, b0Albedo, b0Metal, b0Alpha, b0F0,
                                       h.point, h.normal, giLo);
            var giW_sum = giP_hat;  // w = p_hat / p_source; p_source = 1 → w = p_hat
            var giM = 1.0;
            var giSecPos  = h.point;
            var giSecNorm = h.normal;
            var giLoRes   = giLo;
            var giPhat    = giP_hat;
)"
R"(
            // --- Temporal reuse ---
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

                        let giPrevSGB = textureLoad(stableGBufRead, giPrevPx, 0);
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
                            let prevM     = min(prevGiWeight.y, 20.0);  // M clamp
                            let prevW     = prevGiWeight.z;
                            let prevPhat  = prevGiWeight.w;
)"
R"(
                            // Reject prev reservoir if its secondary hit is too close to current primary
                            let prevSecDist2 = dot(prevSecPos - b0Point, prevSecPos - b0Point);
                            // Re-evaluate prev sample's GI target PDF at current primary surface
                            let giPhatPrev = select(0.0,
                                giTargetPdf(b0Point, b0Normal, b0Wo, b0Albedo, b0Metal, b0Alpha, b0F0,
                                            prevSecPos, prevSecNorm, prevGiLo),
                                prevSecDist2 > 0.04);
                            if (giPhatPrev > 0.0 && prevW > 0.0) {
                                let giWPrev = giPhatPrev * prevM * prevW;
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

            // --- Snapshot pre-spatial reservoir (stored for next-frame temporal reuse) ---
            let giPreSpW_sum = giW_sum;
            let giPreSpM     = giM;
            let giPreSpPhat  = giPhat;
            let giPreSpSecPos  = giSecPos;
            let giPreSpSecNorm = giSecNorm;
            let giPreSpLoRes   = giLoRes;
)"
R"(
            // --- Spatial reuse: 4 neighbours from 20px disk ---
            {
                for (var spI = 0u; spI < 4u; spI++) {
                    let spAngle = rand(seed) * 2.0 * PI;
                    let spR     = sqrt(rand(seed)) * 20.0;
                    let spOff   = vec2<i32>(i32(spR * cos(spAngle)), i32(spR * sin(spAngle)));
                    if (all(spOff == vec2<i32>(0))) { continue; }
                    let spPx = clamp(pixel + spOff,
                                     vec2<i32>(0),
                                     vec2<i32>(i32(rt.iRes.x) - 1, i32(rt.iRes.y) - 1));

                    // Geometry validation via stable g-buffer + mesh ID
                    let spSGB = textureLoad(stableGBufRead, spPx, 0);
                    let spMesh = i32(textureLoad(hitMeshRead, spPx, 0).r);
                    let b0Depth = length(b0Point - vec3<f32>(rt.camOri.x, rt.camOri.y, rt.camOri.z));
                    if (dot(b0Normal, spSGB.xyz) < 0.95 ||
                        abs(b0Depth - spSGB.w) / max(b0Depth, 1e-3) > 0.05 ||
                        spMesh != b0MeshIdx) { continue; }

                    let spGiSample = textureLoad(giResRead, spPx, 0);
                    let spGiWeight = textureLoad(giResWRead, spPx, 0);
                    let spGiLo     = textureLoad(giResLoRead, spPx, 0).xyz;

                    let spSecPos  = spGiSample.xyz;
                    let spSecNorm = unpackOctNormal(spGiSample.w);
                    let spM = min(spGiWeight.y, 4.0);  // spatial M cap
                    let spW = spGiWeight.z;
                    if (spW <= 0.0 || spM <= 0.0) { continue; }

                    // Reject if neighbour's secondary hit too close to our primary
                    let spSecDist2 = dot(spSecPos - b0Point, spSecPos - b0Point);
                    if (spSecDist2 < 0.04) { continue; }

                    let spGiPhat = giTargetPdf(b0Point, b0Normal, b0Wo, b0Albedo, b0Metal, b0Alpha, b0F0,
                                                spSecPos, spSecNorm, spGiLo);
                    if (spGiPhat > 0.0) {
                        let spGiW = spGiPhat * spM * spW;
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

            // --- Finalize reservoir weight ---
            var giW = select(0.0, giW_sum / max(giM * giPhat, 1e-20), giPhat > 0.0);
            giW = min(giW, 3.0);

            // --- Visibility test: shadow ray from primary hit to secondary hit ---
            let giWi = normalize(giSecPos - b0Point);
            let giNdotL = max(dot(b0Normal, giWi), 0.0);
            var giVisAtten = vec3<f32>(0.0);
            if (giNdotL > 0.0 && giW > 0.0) {
                let giDist = length(giSecPos - b0Point);
                giVisAtten = traceShadowRay(b0Point, b0Normal, giWi, giDist - 1e-3, 2);
            }

            // --- Shade from GI reservoir ---
            if (giVisAtten.x + giVisAtten.y + giVisAtten.z > 0.001 && giNdotL > 0.0) {
                let giDelta = giSecPos - b0Point;
                let giDist2 = dot(giDelta, giDelta);
                let giCosTheta2 = max(dot(giSecNorm, -giWi), 0.0);
                let giG = giCosTheta2 / max(giDist2, 0.01);
                let giBrdf = evalBrdfFullSplit(b0Wo, giWi, b0Normal, b0Albedo, b0Metal, b0Alpha, b0F0,
                                                vec3<f32>(0.0), 0.0, 0.0, 0.0);
                let giShade = giVisAtten * giW * giNdotL * giLoRes * giG;
                // Hard luminance cap on GI contribution to prevent seam/firefly spikes
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

            // --- Store PRE-SPATIAL reservoir (visibility-gated) ---
            // Same as DI: store what this pixel selected before spatial reuse.
            // Prevents feedback loops where a shadowed pixel borrows a lit neighbour.
            let giVisible = giVisAtten.x + giVisAtten.y + giVisAtten.z > 0.001;
            let giStoreW_f = select(0.0, select(0.0, giPreSpW_sum / max(giPreSpM * giPreSpPhat, 1e-20), giPreSpPhat > 0.0), giVisible);
            let giStoreM_f = select(0.0, giPreSpM, giVisible);
            textureStore(giResWrite,   pixel, vec4<f32>(giPreSpSecPos, packOctNormal(giPreSpSecNorm)));
            textureStore(giResWWrite,  pixel, vec4<f32>(giPreSpW_sum, giStoreM_f, min(giStoreW_f, 3.0), giPreSpPhat));
            textureStore(giResLoWrite, pixel, vec4<f32>(giPreSpLoRes, 0.0));
            giResStored = true;
        }
        // Fallback: if GI was active (Lo captured) but reservoir skipped (close hit),
        // add the captured Lo back via classic accumulation so energy isn't lost.
        if (useReSTIRGI && !giResStored && luminance(giLo) > 0.0) {
            addSplit(&diffRad, &specRad, throughput * giLo, rt.emissiveInfo.z, 1, firstBounceSpec);
        }

        if (i > 1) {
            // Roughness-weighted Russian roulette — smoother surfaces survive longer.
            let p_base = max(max(throughput.r, throughput.g), throughput.b);
            let rough  = sqrt(h.shininess);
            let weight = mix(0.6, 1.0, 1.0 - rough);
            let p = clamp(p_base * weight, 0.05, 1.0);
            if (rand(seed) > p) { break; }
            throughput /= p;
        }

        // Clearcoat lobe: dielectric specular layer on top of base material.
        // Skip on transmissive surfaces (clearcoat on glass is not physical).
        if (h.clearcoat > 0.0 && h.transmission < 0.01) {
            let ccF0 = 0.04;  // dielectric IOR ~1.5
            let ccCos = max(0.0, dot(wo, h.normal));
            let ccFresnel = ccF0 + (1.0 - ccF0) * pow(1.0 - ccCos, 5.0);
            let ccWeight = h.clearcoat * ccFresnel;
            // Sampling probability: clamp above 0.15 to bound variance on dark surfaces
            // where base contributes little and clearcoat reflection dominates visually.
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
            // Passed through clearcoat — attenuate base by energy not reflected
            throughput *= (1.0 - ccWeight) / (1.0 - ccProb);
        }

        // Sheen lobe: soft velvet-like reflection at grazing angles
        let sheenLumPT = dot(h.sheenColor, vec3<f32>(0.2126, 0.7152, 0.0722));
        if (sheenLumPT > 0.001 && h.transmission < 0.01) {
            let sheenAlpha = max(h.sheenRoughness * h.sheenRoughness, 1e-4);
            // Approximate sheen reflectance at grazing angle for energy conservation
            let NdotV_sh = max(0.001, dot(wo, h.normal));
            let sheenFresnel = sheenLumPT * pow(1.0 - NdotV_sh, 3.0);
            // Attenuate base layer by sheen energy
            throughput *= (1.0 - sheenFresnel);
        }

        // Transmission lobe: refract through transmissive surfaces
        if (h.transmission > 0.0 && rand(seed) < h.transmission) {
            let entering = h.frontFace > 0.5;

            // Dispersion: stochastic wavelength selection.
            // Pick one of R/G/B, compute per-channel IOR via Cauchy-like model,
            // and weight throughput by 3x for the selected channel.
            var channelMask = vec3<f32>(1.0);
            var ior_eff = h.ior;
            if (h.dispersion > 0.0) {
                // Cauchy dispersion from KHR_materials_dispersion (dispersion = 20/V_d).
                // B = (ior-1)*dispersion / (20*(1/λ_F² - 1/λ_C²)) where
                // λ_F=0.4861μm, λ_C=0.6563μm → denominator ≈ 38.2.
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

            let cosI    = abs(dot(normalize(ray.dir), tNorm));
            let r0      = pow((1.0 - ior_eff) / (1.0 + ior_eff), 2.0);
            let fresnel = r0 + (1.0 - r0) * pow(1.0 - cosI, 5.0);

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
                if (h.attenuationDist > 0.0 && !entering) {
                    let absorbCoeff = -log(max(h.attenuationColor, vec3<f32>(1e-6))) / h.attenuationDist;
                    let pathLen = select(h.t, h.thickness, h.t < 1e-2 && h.thickness > 0.0);
                    volAtten = exp(-absorbCoeff * pathLen);
                }
                // Non-symmetry correction: BTDF includes (η_t/η_i)² = 1/η² to account
                // for solid angle change at refractive interface.
                throughput *= glassTint * volAtten / (eta * eta) * channelMask;
            } else {
                // Fresnel reflection: no volume interaction, just surface bounce
                throughput *= vec3<f32>(microWeight) * channelMask;
            }
            afterTransmission = true;
            ray.dir = wi_t;
            continue;
        }

        let F0_b = F0_h;
        var wi_b: vec3<f32>;
        let p_spec = mix(0.5, 0.98, h.metalness);
        let isSpecBounce = rand(seed) < p_spec;
        if (i == 0) { firstBounceSpec = isSpecBounce; }
        // Transmission at bounce 0 also counts as specular for routing
        if (i == 0 && afterTransmission) { firstBounceSpec = true; }
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
    // Safety net: if GI enabled but no reservoir was stored (e.g. maxBounces < 2), write empty
    if (rt.spp.x > 0.5 && !giResStored) {
        textureStore(giResWrite,   pixel, vec4<f32>(0.0));
        textureStore(giResWWrite,  pixel, vec4<f32>(0.0));
        textureStore(giResLoWrite, pixel, vec4<f32>(0.0));
    }
    return SplitRadiance(diffRad, specRad);
}
)";

// Third part of the path trace shader (rt_main entry point + accumulation)
constexpr const char* csPathTraceWGSL3 = R"(
@compute @workgroup_size(8, 8)
fn rt_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(i32(gid.x), i32(gid.y));
    let res   = rt.iRes.xy;
    if (f32(pixel.x) >= res.x || f32(pixel.y) >= res.y) { return; }

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

    // Material-aware camera-motion bounce cap.  During movement (fc==0) reduce
    // bounces per material class instead of a flat cap of 4 for all pixels.
    // Glass/metal/mirror need 4 (refraction chains), diffuse only needs 2.
    var maxBounces = i32(rt.params.x);
    if (fc == 0u) {
        if      (matClass <= 0u) { maxBounces = min(maxBounces, 3); }  // specular
        else if (matClass == 1u) { maxBounces = min(maxBounces, 2); }  // glossy
        else if (matClass == 2u) { maxBounces = min(maxBounces, 1); }  // rough diffuse
        else                     { maxBounces = min(maxBounces, 3); }  // sky/unknown
    }

    // --- Material-aware foveated convergence ---
    // Rough diffuse surfaces converge fast (low-frequency BRDF, reprojects well)
    // so they tolerate more aggressive skipping.  Specular surfaces need more
    // temporal samples for reflections.
    let center = res * 0.5;
    let dxy = (vec2<f32>(f32(pixel.x), f32(pixel.y)) - center) / center;
    let dist = length(dxy);
    var skipMask = 0u;
    if (foveatedOn) {
        if (matClass <= 0u) {
            // Specular/glass: trace more often — reflections are view-dependent
            if (dist > 0.65) { skipMask = 1u; }       // periphery: every 2nd frame
            // middle + center: every frame
        } else if (matClass == 2u) {
            // Rough diffuse: converges fast, skip more aggressively
            if      (dist > 0.65) { skipMask = 7u; }  // periphery: every 8th frame
            else if (dist > 0.3)  { skipMask = 3u; }  // middle: every 4th frame
        } else {
            // Glossy / sky / unknown: default schedule
            if      (dist > 0.65) { skipMask = 3u; }  // periphery: every 4th frame
            else if (dist > 0.3)  { skipMask = 1u; }  // middle: every 2nd frame
        }
    }

    let foveatedSkip = skipMask > 0u && (fc & skipMask) != 0u && !isEnvPixel;

    // Checkerboard skip: during camera movement (fc==0) skip half the pixels each frame,
    // alternating which half via globalFrameCounter so both patterns are covered across
    // two consecutive frames.  Only during rotation — translation causes parallax that
    // makes stale checkerboard pixels visually wrong (foreground/background mismatch).
    // Also disabled when the EMA spatial denoiser is active (brightness mismatch).
    let emaDenoiserOn = rt.spp.z > 0.5;
    let camTranslated = length(rt.camOri.xyz - rt.prevCamOri.xyz) > 1e-5;
    let checkerSkip = !emaDenoiserOn && !camTranslated && fc == 0u && !isEnvPixel &&
        ((u32(pixel.x) + u32(pixel.y) + u32(rt.params.y)) & 1u) == 0u;

    // Foveated/checkerboard skip: pass through previous accumulation unchanged.
    // The pixel keeps its existing color & frame count; it will trace on a future frame.
    if (foveatedSkip || checkerSkip) {
        let rawMode = rt.restirParams.w > 0.5;
        if (rawMode) {
            // Raw mode: write sentinel .w = -1 so temporal denoiser knows to pass through history
            textureStore(accumWrite,     pixel, vec4<f32>(vec3<f32>(0.0), -1.0));
            textureStore(diffAccumWrite, pixel, vec4<f32>(vec3<f32>(0.0), -1.0));
            textureStore(specAccumWrite, pixel, vec4<f32>(vec3<f32>(0.0), -1.0));
        } else {
            textureStore(accumWrite,   pixel, textureLoad(accumRead, pixel, 0));
            textureStore(diffAccumWrite, pixel, textureLoad(diffAccumRead, pixel, 0));
            textureStore(specAccumWrite, pixel, textureLoad(specAccumRead, pixel, 0));
        }
        textureStore(hitMeshWrite, pixel, textureLoad(hitMeshRead, pixel, 0));
        // Preserve previous gBuf (depth) so display shader tone-maps correctly.
        textureStore(gBufWrite,    pixel, textureLoad(gBufRead, pixel, 0));
        textureStore(stableGBufWrite, pixel, textureLoad(stableGBufRead, pixel, 0));
        textureStore(albedoWrite,  pixel, vec4<f32>(vec3<f32>(0.0), 0.0));
        textureStore(momentsWrite, pixel, textureLoad(momentsRead, pixel, 0));
        // Pass reservoir through the ping-pong so the next traced frame sees a valid prior.
        // Without this the reservoir slot contains whatever was written 2-4 frames ago
        // (stale/garbage after multiple skips), producing huge wrong temporal weights → acne.
        if (rt.restirParams.x > 0.5) {
            textureStore(reservoirWrite,  pixel, textureLoad(reservoirRead,  pixel, 0));
            textureStore(reservoirWWrite, pixel, textureLoad(reservoirWRead, pixel, 0));
        }
        if (rt.spp.x > 0.5) {
            textureStore(giResWrite,   pixel, textureLoad(giResRead,   pixel, 0));
            textureStore(giResWWrite,  pixel, textureLoad(giResWRead,  pixel, 0));
            textureStore(giResLoWrite, pixel, textureLoad(giResLoRead, pixel, 0));
        }
        return;
    }
)"
R"(
    // --- Variance-driven bounce reduction for converged pixels ---
    // Once moments have stabilized (fc > 16), check if this pixel's relative
    // variance is low enough to reduce bounces to 1.  The single bounce keeps
    // the pixel "alive" to detect illumination changes while avoiding expensive
    // deep traces.  Camera motion resets fc to 0, automatically disabling this.
    var varianceReducedBounces = maxBounces;

    // AOV mode: only need primary-hit geometry data — skip all secondary bounces.
    let aovMode = i32(rt.emissiveInfo.w);
    if (aovMode > 0) { varianceReducedBounces = 1; }

    // Per-pixel Cranley-Patterson rotation: offset R2 by a spatial hash to decorrelate pixels.
    let pixHash = pcg(gid.x + gid.y * 65537u);
    let r2 = r2Seq(fc);
    // Sub-pixel jitter centred on pixel centre (0.5, 0.5), range ±0.375.
    // Gives TAA true sub-pixel information for silhouette AA.
    let jx = (fract(r2.x + f32(pixHash)       / 4294967296.0) - 0.5) * 0.75;
    let jy = (fract(r2.y + f32(pcg(pixHash))  / 4294967296.0) - 0.5) * 0.75;
    var seed = pcg(pcg(gid.x + gid.y * 65537u) + fc * 12979u);

    let ray = makeRay(vec2<f32>(f32(pixel.x) + 0.5 + jx, f32(pixel.y) + 0.5 + jy), res);
    var primaryMeshIdx: u32;
    var primaryNormal:  vec3<f32>;
    var primaryDepth:   f32;
    var primaryAlbedo:  vec3<f32>;
    var primaryRough:   f32;
    var primaryMatIdx:  i32;
    var touchedMoved:   bool;
    var ptResult = pathTrace(ray, &seed, pixel, varianceReducedBounces, &primaryMeshIdx, &primaryNormal, &primaryDepth, &primaryAlbedo, &primaryRough, &primaryMatIdx, &touchedMoved);
    var sample = ptResult.diff + ptResult.spec;

    // Unjittered primary ray for stable G-buffer: traces a pixel-center ray
    // (no jitter) to get stable normal/depth/meshID for à-trous edge-stopping.
    // BVH traversal only — no shading, no material eval. Keeps edge-stopping
    // weights consistent frame-to-frame despite sub-pixel jitter on the radiance ray.
    let centerRay = makeRay(vec2<f32>(f32(pixel.x) + 0.5, f32(pixel.y) + 0.5), res);
    let stableHit = stablePrimaryHit(centerRay);

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
        textureStore(diffAccumWrite, pixel, vec4<f32>(aovColor, 1.0));
        textureStore(specAccumWrite, pixel, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        textureStore(accumWrite, pixel, vec4<f32>(aovColor, 1.0));
        textureStore(hitMeshWrite, pixel, vec4<f32>(f32(primaryMeshIdx), f32(primaryMatIdx), 0.0, 0.0));
        textureStore(gBufWrite, pixel, vec4<f32>(primaryNormal, primaryDepth));
        textureStore(stableGBufWrite, pixel, vec4<f32>(stableHit.normal, stableHit.t));
        textureStore(albedoWrite, pixel, vec4<f32>(primaryAlbedo, primaryRough));
        return;
    }

    // Raw 1-spp output mode (temporal denoiser pipeline)
    let rawOutputMode = rt.restirParams.w > 0.5;
    if (rawOutputMode) {
        let lum3 = vec3<f32>(0.2126, 0.7152, 0.0722);
        // NaN guard
        var sampleClean = select(vec3<f32>(0.0), sample, sample.x == sample.x);
        // Hard firefly cap
        let hardCap = 12.0;
        let rawLum = dot(sampleClean, lum3);
        if (rawLum > hardCap) { sampleClean = sampleClean * (hardCap / rawLum); }

        // Split diff/spec with clamping
        let diffSample = select(vec3<f32>(0.0), ptResult.diff, ptResult.diff.x == ptResult.diff.x);
        let specSample = select(vec3<f32>(0.0), ptResult.spec, ptResult.spec.x == ptResult.spec.x);
        var dClamped = diffSample;
        let dLum = dot(diffSample, lum3);
        if (dLum > hardCap) { dClamped = diffSample * (hardCap / dLum); }
        let specHardCap = mix(3.0, 8.0, primaryRough);
        var sClamped = specSample;
        let sLum = dot(specSample, lum3);
        if (sLum > specHardCap) { sClamped = specSample * (specHardCap / sLum); }

        let fc = rt.frameCount.x;
        textureStore(accumWrite, pixel, vec4<f32>(sampleClean, fc));
        textureStore(diffAccumWrite, pixel, vec4<f32>(dClamped, fc));
        textureStore(specAccumWrite, pixel, vec4<f32>(sClamped, fc));

        // Raw single-sample moments
        let sLumMom = dot(sampleClean, lum3);
        textureStore(momentsWrite, pixel, vec4<f32>(sLumMom, sLumMom * sLumMom, 0.0, 0.0));

        // G-buffer, hit mesh (with touchedMoved in .b), albedo
        textureStore(hitMeshWrite, pixel, vec4<f32>(f32(primaryMeshIdx), f32(primaryMatIdx), select(0.0, 1.0, touchedMoved), 0.0));
        textureStore(albedoWrite, pixel, vec4<f32>(primaryAlbedo, primaryRough));
        textureStore(gBufWrite, pixel, vec4<f32>(primaryNormal, primaryDepth));
        textureStore(stableGBufWrite, pixel, vec4<f32>(stableHit.normal, stableHit.t));
        return;
    }

    let prev     = textureLoad(accumRead, pixel, 0);
    var old      = prev.xyz;
    var pixelFC  = prev.w;

    // Split accum old values — read from current pixel, overridden by reprojection block
    let prevDiffRaw = textureLoad(diffAccumRead, pixel, 0).xyz;
    let prevSpecRaw = textureLoad(specAccumRead, pixel, 0).xyz;
    var oldDiff = select(vec3<f32>(0.0), prevDiffRaw, prevDiffRaw.x == prevDiffRaw.x);
    var oldSpec = select(vec3<f32>(0.0), prevSpecRaw, prevSpecRaw.x == prevSpecRaw.x);
    // Moments reprojected from same position as color — set inside reprojection block.
    // Falls back to current-pixel read if reprojection fails.
    let prevMom = textureLoad(momentsRead, pixel, 0);
    var prevMomM1 = prevMom.x;
    var prevMomM2 = prevMom.y;

    // Force-reset on mode switch / topology rebuild (params.w flag)
    let forceReset = rt.params.w > 0.5;
    if (forceReset) { pixelFC = 0.0; }

    // Reproject accumulation buffer across camera/mesh motion using motion vectors.
    // Two reprojection cases, unified through the same math:
    //   (A) Primary ray hits a moved mesh: transform worldPos by the mesh's
    //       motion matrix to find where this surface point was last frame,
    //       then project into the prev camera's screen space. Lets the TAA
    //       history follow rotating/translating meshes instead of resetting.
    //   (B) Camera moved (fc == 0u), primary hit a static surface: worldPos is
    //       already in the prev frame's world space, so identity-transform
    //       before prev-camera reprojection.
    // Without the motion-matrix branch, rotating meshes (e.g. a car) get
    // pixelFC=0 every frame → 1 spp visible noise. With it, their accumulation
    // follows the surface and converges like static geometry.
    let prevMeshU = u32(textureLoad(hitMeshRead, pixel, 0).r);
    let primaryMoved = primaryMeshIdx < 128u && isMeshMoved(i32(primaryMeshIdx));

    // Disocclusion: moved mesh LEFT this pixel (was here last frame, gone now)
    // → history is stale regardless of what's here now.
    if (primaryMeshIdx != prevMeshU && prevMeshU < 128u && isMeshMoved(i32(prevMeshU))) {
        pixelFC = 0.0;
    }
    // Secondary bounce hit a moved mesh — cap rather than reset so static
    // surfaces can still converge while indirect lighting refreshes.
    // If the moved mesh is itself an emissive source, use a much tighter cap (2 frames)
    // so that reflections/shadows of moving lights clear in ~33ms rather than ~133ms.
    if (touchedMoved) {
        let emissiveMoved = rt.restirParams.z > 0.5;
        pixelFC = min(pixelFC, select(8.0, 2.0, emissiveMoved));
    }

    // Unified motion-vector reprojection.  Runs when either the camera moved
    // (fc==0u) OR the primary-hit mesh moved.  Handles both together: if the
    // car rotates AND the camera moves, we compose mesh motion (motionMats)
    // with camera motion (prevCam*) in a single reprojection.
    if (primaryMoved || fc == 0u) {
        var reprojOk = false;
        if (pixelFC > 0.0 && primaryDepth > 0.0) {
            let worldPos = rt.camOri.xyz + ray.dir * primaryDepth;
            // For moved meshes: transform curWorld→prevWorld using the mesh's
            // motion matrix (prevWorld * inverse(curWorld)).  For static surfaces,
            // prevWorld == worldPos.
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
                // Path tracer uses pixel-CORNER ray convention
                // (ray target = pixel.xy + jitter, jitter ∈ [-0.5,0.5] with
                // mean 0 → mean ray target is pixel corner, not centre).
                // So prevNdcX maps directly to pixel coordinate — no -0.5
                // offset like TAA (which uses centre-convention rays).
                let prevU = (prevNdcX + 1.0) * 0.5 * res.x;
                let prevV = (1.0 - prevNdcY) * 0.5 * res.y;
                let pxBase = vec2<i32>(i32(floor(prevU)), i32(floor(prevV)));
                // Need full 2×2 footprint in bounds for bilinear
                if (pxBase.x >= 0 && pxBase.x + 1 < i32(res.x) &&
                    pxBase.y >= 0 && pxBase.y + 1 < i32(res.y)) {
                    let fx = prevU - f32(pxBase.x);
                    let fy = prevV - f32(pxBase.y);
                    // Bilinear weights for the 4 corners
                    let w00 = (1.0 - fx) * (1.0 - fy);
                    let w10 = fx         * (1.0 - fy);
                    let w01 = (1.0 - fx) *         fy;
                    let w11 = fx         *         fy;
                    // Validate each corner: only blend from corners that
                    // belong to the same mesh as the current pixel. Corners
                    // on different meshes (disocclusion / silhouette) are
                    // dropped and weights renormalised.
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
                        // Reproject split accum from same reprojected position
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
                        // Reproject moments from same reprojected position as color
                        let mom00 = textureLoad(momentsRead, p00, 0).xy;
                        let mom10 = textureLoad(momentsRead, p10, 0).xy;
                        let mom01 = textureLoad(momentsRead, p01, 0).xy;
                        let mom11 = textureLoad(momentsRead, p11, 0).xy;
                        prevMomM1 = (mom00.x*v00 + mom10.x*v10 + mom01.x*v01 + mom11.x*v11) * inv;
                        prevMomM2 = (mom00.y*v00 + mom10.y*v10 + mom01.y*v01 + mom11.y*v11) * inv;
                        // Roughness-adaptive history cap.
                        //   Diffuse/matte (rough≈1): no view-dependent shading,
                        //     history stays valid as long as reprojection is
                        //     accurate → cap at 256 (√256 = 16× noise drop).
                        //   Mirror/metal (rough≈0): specular lobe is view-
                        //     dependent, history goes stale → cap tight at 8
                        //     so reflections refresh as camera moves.
                        //   primaryRough (= sqrt(alpha)) comes from pathTrace.
                        // Moving meshes: per-mesh motion reprojection compounds
                        // sub-pixel bilinear blur every frame, so cap tighter
                        // (32 diffuse / 4 mirror) to shed history faster and
                        // avoid motion smear on rotating/translating geometry.
                        let staticCap = mix(8.0, 256.0,
                                            smoothstep(0.15, 0.7, primaryRough));
                        let movingCap = mix(8.0, 64.0,
                                            smoothstep(0.15, 0.7, primaryRough));
                        if (primaryMoved) {
                            pixelFC = min(prevFC * 0.5, movingCap);
                        } else {
                            // Pure camera reprojection (static mesh): halve
                            // to absorb any residual sub-pixel drift.
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
    // NaN guard: reject corrupted samples to prevent permanent accumulation damage
    let sampleClean = select(vec3<f32>(0.0), sample, sample.x == sample.x);
    let oldClean    = select(vec3<f32>(0.0), old,    old.x == old.x);

    // Adaptive outlier rejection: clamp sample relative to running average.
    // Hard cap first (catches extreme fireflies regardless of history).
    let lum3 = vec3<f32>(0.2126, 0.7152, 0.0722);
    var clamped = sampleClean;
    let rawLum = dot(sampleClean, lum3);
    let hardCap = 12.0;
    if (rawLum > hardCap) {
        clamped = sampleClean * (hardCap / rawLum);
    }
    // Relative clamp: only once history is reliable (≥8 frames), 7× running average.
    // Do NOT start earlier — moving emissives land on pixels with dark history,
    // making avgLum artificially low and crushing legitimate bright samples.
    if (pixelFC > 8.0) {
        let avgLum = max(dot(oldClean, lum3), 0.05);
        let smpLum = dot(clamped, lum3);
        let maxLum = avgLum * 7.0;
        if (smpLum > maxLum) {
            clamped = clamped * (maxLum / smpLum);
        }
    }

    // Stop accumulating once float16 precision is exhausted (~1024 frames)
    let alpha = 1.0 / (pixelFC + 1.0);
    if (pixelFC < 1024.0) {
        var blended = oldClean * (1.0 - alpha) + clamped * alpha;
        // DEBUG: flip to true to visualize pixelFC (history length) instead of radiance.
        // log2 ramp covers the full 0..256 range:
        //   black = 0,  red ≈ 8,  orange ≈ 32,  yellow ≈ 128,  white ≥ 256.
        // Diffuse-capped (roughCap=256) surfaces should converge to white.
        // Specular-capped (roughCap=8) surfaces stay red.
        const DEBUG_VIS_PIXEL_FC = false;
        if (DEBUG_VIS_PIXEL_FC) {
            // log2(pixelFC+1)/8 maps pixelFC=0→0, 8→~0.39, 32→~0.63, 256→1.0
            let fcN = clamp(log2(pixelFC + 1.0) / 8.0, 0.0, 1.0);
            blended = vec3<f32>(fcN, fcN * fcN, fcN * fcN * fcN);
        }
        // DEBUG: flip to true to visualize RAW single-sample path trace output
        // (bypasses all accumulation). If fireflies are visible in this view,
        // they originate in the path tracer itself. If NOT visible here but
        // visible in the accumulated output, accumulation logic is injecting
        // noise (e.g. reprojection returning bad history, outlier rejection
        // not firing). Tonemap the sample so extreme fireflies don't just
        // paint the screen white.
        const DEBUG_VIS_RAW_SAMPLE = false;
        if (DEBUG_VIS_RAW_SAMPLE) {
            // Simple Reinhard to compress fireflies into [0,1] while
            // keeping relative brightness visible.
            blended = sampleClean / (sampleClean + vec3<f32>(1.0));
        }
        textureStore(accumWrite, pixel, vec4<f32>(blended, pixelFC + 1.0));
    } else {
        textureStore(accumWrite, pixel, vec4<f32>(oldClean, pixelFC));
    }

    // Temporal variance tracking: accumulate 1st and 2nd moments of luminance.
    // Use post-hard-cap, pre-relative-clamp luminance so variance reflects true
    // sample spread (relative clamp would artificially suppress it).
    let sampleLum = dot(clamped, lum3);
    // Use reprojected moments (prevMomM1/M2) set in the reprojection block above.
    // Floor alpha at 0.1 so variance stays responsive even at convergence.
    let momAlpha = max(alpha, 0.1);
    var m1 = prevMomM1;
    var m2 = prevMomM2;
    if (pixelFC < 1.0) { m1 = sampleLum; m2 = sampleLum * sampleLum; }
    else {
        m1 = m1 * (1.0 - momAlpha) + sampleLum * momAlpha;
        m2 = m2 * (1.0 - momAlpha) + sampleLum * sampleLum * momAlpha;
    }
    textureStore(momentsWrite, pixel, vec4<f32>(m1, m2, 0.0, 0.0));

    // --- Diffuse/specular split accumulation ---
    // oldDiff/oldSpec are already reprojected (bilinear from previous frame's position)
    {
        let diffSample = select(vec3<f32>(0.0), ptResult.diff, ptResult.diff.x == ptResult.diff.x);
        let specSample = select(vec3<f32>(0.0), ptResult.spec, ptResult.spec.x == ptResult.spec.x);
        // Diffuse: same hard cap as combined (diffuse has low variance)
        var dClamped = diffSample;
        let dLum = dot(diffSample, lum3);
        if (dLum > hardCap) { dClamped = diffSample * (hardCap / dLum); }
        // Specular: tighter roughness-adaptive cap.
        // Glossy/mirror (rough≈0): cap=3 — specular has extreme variance from
        // bright point reflections, aggressive clamping needed.
        // Rough metal (rough≈1): cap=8 — wider lobe averages over more surface,
        // less variance per sample.
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

    textureStore(hitMeshWrite, pixel, vec4<f32>(f32(primaryMeshIdx), f32(primaryMatIdx), select(0.0, 1.0, touchedMoved), 0.0));
    textureStore(albedoWrite,  pixel, vec4<f32>(primaryAlbedo, primaryRough));

    textureStore(gBufWrite, pixel, vec4<f32>(primaryNormal, primaryDepth));
    textureStore(stableGBufWrite, pixel, vec4<f32>(stableHit.normal, stableHit.t));
}
)";

// Build a raycaster or path tracer shader by concatenating common + specific code.
static std::string buildRtShader(bool hasEnvCdf) {
    std::string src = std::string(csSharedDefsWGSL) + "\n" +
                      csCommonWGSL + "\n" +
                      csPathTraceWGSL + "\n" +
                      csPathTraceWGSL2 + "\n" +
                      csPathTraceWGSL2b + "\n" +
                      csPathTraceWGSL3;
    const std::string marker = "/*ENV_CDF_FLAG*/false";
    auto pos = src.find(marker);
    if (pos != std::string::npos) {
        src.replace(pos, marker.size(), hasEnvCdf ? "true" : "false");
    }
    return src;
}

// ---------------------------------------------------------------------------
// WGSL vertex-transform compute shader
// ---------------------------------------------------------------------------
constexpr const char* vtWGSL_ = R"(
struct ObjTriData {
    v0:   vec4<f32>,
    v1:   vec4<f32>,
    v2:   vec4<f32>,
    n0:   vec4<f32>,
    n1:   vec4<f32>,
    n2:   vec4<f32>,
    uv01: vec4<f32>,
    uv2:  vec4<f32>,
    uv1_01: vec4<f32>,
    uv1_2:  vec4<f32>,
    vcol01: vec4<f32>,  // vertex color: v0.r, v0.g, v0.b, v1.r
    vcol2:  vec4<f32>,  // vertex color: v1.g, v1.b, v2.r, v2.g  (v2.b in uv2.w)
}
struct MeshMatrices {
    world:  mat4x4<f32>,
    normal: mat4x4<f32>,
}
struct VtUniforms {
    triCount: u32,
    groupsX:  u32,
    _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<storage, read> objTris:   array<ObjTriData>;
@group(0) @binding(1) var<storage, read> meshMats:  array<MeshMatrices>;
@group(0) @binding(2) var triOut: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> vtUni: VtUniforms;
@group(0) @binding(4) var<storage, read> objTris2:  array<ObjTriData>;  // overflow buffer

@compute @workgroup_size(64)
fn vt_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linearId = gid.x + gid.y * vtUni.groupsX * 64u;
    if (linearId >= vtUni.triCount) { return; }
    let ti  = i32(linearId);
    let splitAt = i32(vtUni._p1);
    var obj: ObjTriData;
    if (ti < splitAt) { obj = objTris[ti]; }
    else              { obj = objTris2[ti - splitAt]; }
    let mi  = i32(obj.v1.w);
    let mat = meshMats[mi];
    let v0  = (mat.world  * vec4<f32>(obj.v0.xyz, 1.0)).xyz;
    let v1  = (mat.world  * vec4<f32>(obj.v1.xyz, 1.0)).xyz;
    let v2  = (mat.world  * vec4<f32>(obj.v2.xyz, 1.0)).xyz;
    let n0  = normalize((mat.normal * vec4<f32>(obj.n0.xyz, 0.0)).xyz);
    let n1  = normalize((mat.normal * vec4<f32>(obj.n1.xyz, 0.0)).xyz);
    let n2  = normalize((mat.normal * vec4<f32>(obj.n2.xyz, 0.0)).xyz);
    textureStore(triOut, triCoord(ti, 0), vec4<f32>(v0, obj.v0.w));
    textureStore(triOut, triCoord(ti, 1), vec4<f32>(v1, f32(mi)));
    textureStore(triOut, triCoord(ti, 2), vec4<f32>(v2, 0.0));
    textureStore(triOut, triCoord(ti, 3), vec4<f32>(n0, 0.0));
    textureStore(triOut, triCoord(ti, 4), vec4<f32>(n1, 0.0));
    textureStore(triOut, triCoord(ti, 5), vec4<f32>(n2, 0.0));
    textureStore(triOut, triCoord(ti, 6), obj.uv01);
    textureStore(triOut, triCoord(ti, 7), obj.uv2);
    textureStore(triOut, triCoord(ti, 8), obj.uv1_01);
    textureStore(triOut, triCoord(ti, 9), obj.uv1_2);
    textureStore(triOut, triCoord(ti, 10), obj.vcol01);
    textureStore(triOut, triCoord(ti, 11), obj.vcol2);
}
)";

// ---------------------------------------------------------------------------
// WGSL BVH refit compute shader
// ---------------------------------------------------------------------------
constexpr const char* refitWGSL_ = R"(
struct Bvh4NodeGpu {
    cMinX: vec4<f32>,
    cMinY: vec4<f32>,
    cMinZ: vec4<f32>,
    cMaxX: vec4<f32>,
    cMaxY: vec4<f32>,
    cMaxZ: vec4<f32>,
    cIdx:  vec4<u32>,  // child indices (bitcast to i32 for leaf encoding)
}
struct RefitUniforms {
    leafCount: u32,
    groupsX:   u32,
    _p1: u32, _p2: u32,
}

@group(0) @binding(0) var triTex:                          texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bvhNodes:   array<Bvh4NodeGpu>;
@group(0) @binding(2) var<storage, read_write> bvhCounters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>       leafIdxBuf: array<i32>;
@group(0) @binding(4) var<uniform>             refitUni:   RefitUniforms;
@group(0) @binding(5) var<storage, read>       refitMeta:  array<vec4<i32>>;  // (parent, childCount, numInternal, 0)

fn writeChildAABB(ni: i32, c: i32, bmin: vec3<f32>, bmax: vec3<f32>) {
    // Scale-aware padding: max(1e-5, 1e-5 * extent) so it stays meaningful at any scale.
    let extent = max(abs(bmin), abs(bmax));
    let E = max(extent * vec3<f32>(1e-5), vec3<f32>(1e-6));
    var n = bvhNodes[ni];
    n.cMinX[c] = bmin.x - E.x;
    n.cMinY[c] = bmin.y - E.y;
    n.cMinZ[c] = bmin.z - E.z;
    n.cMaxX[c] = bmax.x + E.x;
    n.cMaxY[c] = bmax.y + E.y;
    n.cMaxZ[c] = bmax.z + E.z;
    bvhNodes[ni] = n;
}

@compute @workgroup_size(64)
fn bvh_refit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linearId = gid.x + gid.y * refitUni.groupsX * 64u;
    if (linearId >= refitUni.leafCount) { return; }
    let wideNi = leafIdxBuf[i32(linearId)];
    let nfo = refitMeta[wideNi];
    let childCount = nfo.y;

    // Refit all leaf children of this wide node
    let cIdxVec = bvhNodes[wideNi].cIdx;
    let leafIdx = array<i32, 4>(bitcast<i32>(cIdxVec.x), bitcast<i32>(cIdxVec.y),
                                 bitcast<i32>(cIdxVec.z), bitcast<i32>(cIdxVec.w));

    for (var c: i32 = 0; c < childCount; c++) {
        let ci = leafIdx[c];
        if (ci >= 0) { continue; }

        // Decode packed leaf: triStart and triCount
        let raw = -ci;
        let triStart = (raw - 1) / MAX_LEAF_TRIS;
        let triCount = ((raw - 1) % MAX_LEAF_TRIS) + 1;

        var bmin = vec3<f32>(1e30);
        var bmax = vec3<f32>(-1e30);
        for (var ti = triStart; ti < triStart + triCount; ti++) {
            for (var row = 0; row < 3; row++) {
                let v = textureLoad(triTex, triCoord(ti, row), 0).xyz;
                bmin = min(bmin, v);
                bmax = max(bmax, v);
            }
        }
        writeChildAABB(wideNi, c, bmin, bmax);
    }

    // If this node has internal children, we must wait for them before propagating.
    let numInternal = nfo.z;
    if (numInternal > 0) {
        let cnt = atomicAdd(&bvhCounters[wideNi], 1u);
        if (cnt < u32(numInternal)) { return; }
    }

    // Propagate up to parent
    var curNi = nfo.x;
    loop {
        if (curNi < 0) { break; }
        let curNfo = refitMeta[curNi];
        let numInt = u32(curNfo.z);
        let cnt = atomicAdd(&bvhCounters[curNi], 1u);
        let curChildCount = curNfo.y;
        let hasLeaves = u32(curChildCount) > numInt;
        let expected = select(numInt - 1u, numInt, hasLeaves);
        if (cnt < expected) { break; }

        // All children done — recompute internal children's AABBs from their sub-children
        let pIdxVec = bvhNodes[curNi].cIdx;
        let pIdx = array<i32, 4>(bitcast<i32>(pIdxVec.x), bitcast<i32>(pIdxVec.y),
                                  bitcast<i32>(pIdxVec.z), bitcast<i32>(pIdxVec.w));
        for (var c: i32 = 0; c < curChildCount; c++) {
            let ci = pIdx[c];
            if (ci < 0) { continue; }
            let child = bvhNodes[ci];
            let cc = refitMeta[ci].y;
            var bmin = vec3<f32>(1e30);
            var bmax = vec3<f32>(-1e30);
            // Read f32 child AABBs directly
            let cMinXa = array<f32, 4>(child.cMinX.x, child.cMinX.y, child.cMinX.z, child.cMinX.w);
            let cMinYa = array<f32, 4>(child.cMinY.x, child.cMinY.y, child.cMinY.z, child.cMinY.w);
            let cMinZa = array<f32, 4>(child.cMinZ.x, child.cMinZ.y, child.cMinZ.z, child.cMinZ.w);
            let cMaxXa = array<f32, 4>(child.cMaxX.x, child.cMaxX.y, child.cMaxX.z, child.cMaxX.w);
            let cMaxYa = array<f32, 4>(child.cMaxY.x, child.cMaxY.y, child.cMaxY.z, child.cMaxY.w);
            let cMaxZa = array<f32, 4>(child.cMaxZ.x, child.cMaxZ.y, child.cMaxZ.z, child.cMaxZ.w);
            for (var gc: i32 = 0; gc < cc; gc++) {
                bmin = min(bmin, vec3<f32>(cMinXa[gc], cMinYa[gc], cMinZa[gc]));
                bmax = max(bmax, vec3<f32>(cMaxXa[gc], cMaxYa[gc], cMaxZa[gc]));
            }
            writeChildAABB(curNi, c, bmin, bmax);
        }
        curNi = curNfo.x;
    }
}
)";

static std::string buildVtShader() { return std::string(csSharedDefsWGSL) + "\n" + vtWGSL_; }
static std::string buildRefitShader() { return std::string(csSharedDefsWGSL) + "\n" + refitWGSL_; }

// ---------------------------------------------------------------------------
// WGSL ReLAX-inspired temporal accumulation (Phase 1)
// Per-channel temporal filter with per-pixel history length, moment tracking,
// luminance-based anti-lag clamping, and mode-aware blending (diffuse vs specular).
// Runs separately on diffuse and specular split buffers.
// ---------------------------------------------------------------------------
constexpr const char* taaWGSL = R"(
struct TaaUniforms {
    prevCamOri: vec4<f32>, prevCamFwd: vec4<f32>, prevCamRgt: vec4<f32>, prevCamUp: vec4<f32>,
    curCamOri:  vec4<f32>, curCamFwd:  vec4<f32>, curCamRgt:  vec4<f32>, curCamUp:  vec4<f32>,
    iRes:       vec4<f32>,   // .xy = resolution, .zw = prevJx/prevJy
    tanHalfFov: vec4<f32>,
    frameCount: vec4<f32>,   // .x = global FC, .y = mode (0=diff, 1=spec), .z = curJx, .w = curJy
    movedMeshBits: vec4<u32>,
}

@group(0) @binding(0) var<uniform> taa:      TaaUniforms;
@group(0) @binding(1) var accumIn:   texture_2d<f32>;
@group(0) @binding(2) var gBufCur:   texture_2d<f32>;
@group(0) @binding(3) var taaHistIn: texture_2d<f32>;  // .w = history length
@group(0) @binding(4) var taaOut:    texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var hitMeshTex: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> motionMats: array<mat4x4<f32>>;
@group(0) @binding(7) var albedoTex:  texture_2d<f32>;  // .w = primary linear roughness
@group(0) @binding(8) var momentsIn:  texture_2d<f32>;  // (E[L], E[L²]) from RT pass
@group(0) @binding(9) var gBufPrev:   texture_2d<f32>;  // previous frame g-buffer (depth in .w)
@group(0) @binding(10) var stableGBuf:     texture_2d<f32>; // unjittered current: normal.xyz + depth.w
@group(0) @binding(11) var stableGBufPrev: texture_2d<f32>; // unjittered previous frame

fn pcg_t(v: u32) -> u32 {
    var s = v * 747796405u + 2891336453u;
    s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (s >> 22u) ^ s;
}
const R2_A1_T: f32 = 0.7548776662466927;
const R2_A2_T: f32 = 0.5698402909980532;
fn r2Seq_t(n: u32) -> vec2<f32> {
    return fract(vec2<f32>(f32(n) * R2_A1_T, f32(n) * R2_A2_T));
}

fn luminance_t(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn taa_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(gid.xy);
    let res   = taa.iRes.xy;
    let iRes  = vec2<i32>(i32(res.x), i32(res.y));
    if (pixel.x >= iRes.x || pixel.y >= iRes.y) { return; }

    let isSpec  = taa.frameCount.y > 0.5;
    let curSamp  = textureLoad(accumIn, pixel, 0);
    let curColor = curSamp.xyz;
    let curFC    = curSamp.w;
    let curGB    = textureLoad(gBufCur, pixel, 0);
    let curDepth = curGB.w;
    // Stable G-buffer: unjittered primary ray — consistent normal/depth for
    // reprojection and per-tap validation, eliminating silhouette shimmer.
    let stableGB   = textureLoad(stableGBuf, pixel, 0);
    let stableNorm = stableGB.xyz;
    let stableDepth = stableGB.w;

    // Sky pixels: pass through, history length = 0
    if (curDepth < 1e-6) {
        textureStore(taaOut, pixel, vec4<f32>(curColor, 0.0));
        return;
    }

    // Foveated/checkerboard-skipped pixels: .w = -1 sentinel in raw mode.
    // Instead of blindly passing through history at the current pixel position,
    // mark this pixel so that after reprojection we write the reprojected history
    // without blending a new sample.  This keeps brightness consistent with
    // neighboring traced pixels (which go through EMA) and avoids the checkerboard
    // flashing artefact caused by half the pixels being stale while the other half
    // get a fresh 1-spp blend.
    let isSkippedPixel = curFC < -0.5;

    let hitInfo  = textureLoad(hitMeshTex, pixel, 0);
    let curMeshId = u32(hitInfo.r);
    let touchedMoved = hitInfo.b > 0.5;

    // No variance box. SVGF design: temporal pass = pure EMA + disocclusion reset.
    // The moments-driven à-trous spatial filter handles all noise removal.
    // Box clamping in temporal passes is a rasterizer TAA technique — incorrect for
    // path tracing where every sample is independently noisy regardless of history.

    // Reproject current pixel into previous frame's screen space.
    // Use unjittered center ray + stable depth for consistent reprojection
    // across frames — eliminates shimmer from jitter-dependent world positions.
    let aspect = res.x / res.y;
    let ndc = vec2<f32>((f32(pixel.x) + 0.5) / res.x * 2.0 - 1.0,
                         1.0 - (f32(pixel.y) + 0.5) / res.y * 2.0);
    let rayDir   = normalize(taa.curCamFwd.xyz
                            + taa.curCamRgt.xyz * (ndc.x * taa.tanHalfFov.x * aspect)
                            + taa.curCamUp.xyz  * (ndc.y * taa.tanHalfFov.x));
    let worldPos = taa.curCamOri.xyz + rayDir * stableDepth;

    // For moved meshes: transform world pos by motion matrix
    let meshIdx = u32(textureLoad(hitMeshTex, pixel, 0).r);
    var prevWorldPos = worldPos;
    if (meshIdx < 128u) {
        let bit = meshIdx & 31u;
        let wi  = meshIdx >> 5u;
        var mbits: u32 = 0u;
        if      (wi == 0u) { mbits = taa.movedMeshBits.x; }
        else if (wi == 1u) { mbits = taa.movedMeshBits.y; }
        else if (wi == 2u) { mbits = taa.movedMeshBits.z; }
        else               { mbits = taa.movedMeshBits.w; }
        if (((mbits >> bit) & 1u) != 0u) {
            prevWorldPos = (motionMats[meshIdx] * vec4<f32>(worldPos, 1.0)).xyz;
        }
    }

    let toPoint  = prevWorldPos - taa.prevCamOri.xyz;
    let prevZ    = dot(toPoint, taa.prevCamFwd.xyz);

    var useHist = prevZ > 0.001;
    let prevNdcX = dot(toPoint, taa.prevCamRgt.xyz) / (prevZ * taa.tanHalfFov.x * aspect);
    let prevNdcY = dot(toPoint, taa.prevCamUp.xyz)  / (prevZ * taa.tanHalfFov.x);
    let prevPx   = vec2<f32>((prevNdcX + 1.0) * 0.5 * res.x - 0.5,
                              (1.0 - prevNdcY) * 0.5 * res.y - 0.5);
    let prevPixel = vec2<i32>(i32(floor(prevPx.x)), i32(floor(prevPx.y)));

    // Bounds check (need 2×2 footprint for bilinear)
    useHist = useHist && prevPixel.x >= 0 && prevPixel.x + 1 < iRes.x
                      && prevPixel.y >= 0 && prevPixel.y + 1 < iRes.y;

    // Depth-based disocclusion (two conditions):
    // 1. Standard: prevZ vs curZDepth — detects when current surface wasn't visible before.
    // 2. Revealed: current surface is much deeper than previous surface at prevPixel
    //    — detects background pixels revealed when a foreground object moved away.
    //    Without this, background pixels behind a moving object keep torus-color history.
    let curZDepth = dot(worldPos - taa.curCamOri.xyz, taa.curCamFwd.xyz);
    let depthRatio = abs(prevZ - curZDepth) / max(curZDepth, 0.01);
    // Previous frame's ray depth at prevPixel (g-buffer .w stores ray distance).
    // Use stable depth on both sides — consistent across frames regardless of jitter.
    let prevFrameStableDepth = textureLoad(stableGBufPrev, prevPixel, 0).w;
    let revealedRatio = abs(stableDepth - prevFrameStableDepth) / max(stableDepth, 0.01);
    let disoccluded = useHist && (depthRatio > 0.1 || revealedRatio > 0.15);

    var result = curColor;
    var newHistLen = 1.0;

    if (useHist && !disoccluded) {
        // Per-tap validated bilinear history fetch.
        // Blindly blending 4 taps at geometry edges mixes foreground/background
        // history → shimmer as the jitter shifts the bilinear footprint frame-to-frame.
        // Each tap is validated against: depth agreement + normal agreement with current pixel.
        let fx = fract(prevPx.x);
        let fy = fract(prevPx.y);
        let px00 = prevPixel;
        let px10 = prevPixel + vec2<i32>(1, 0);
        let px01 = prevPixel + vec2<i32>(0, 1);
        let px11 = prevPixel + vec2<i32>(1, 1);

        let h00 = textureLoad(taaHistIn, px00, 0);
        let h10 = textureLoad(taaHistIn, px10, 0);
        let h01 = textureLoad(taaHistIn, px01, 0);
        let h11 = textureLoad(taaHistIn, px11, 0);

        // Use stable (unjittered) g-buffer on BOTH sides for per-tap validation.
        // Current pixel: stableGBuf (geometric normals + unjittered depth).
        // Previous taps: stableGBufPrev (also geometric normals + unjittered depth).
        // This avoids mismatch between geometric vs normal-mapped normals on textured surfaces.
        let sg00 = textureLoad(stableGBufPrev, px00, 0);
        let sg10 = textureLoad(stableGBufPrev, px10, 0);
        let sg01 = textureLoad(stableGBufPrev, px01, 0);
        let sg11 = textureLoad(stableGBufPrev, px11, 0);

        // Slightly relaxed depth threshold vs global disocclusion to avoid over-rejection
        let depthTol = 0.2;
        let normTol  = 0.8;  // dot(stableNorm, tapNorm) must exceed this

        let v00 = select(0.0, (1.0-fx)*(1.0-fy), abs(sg00.w - stableDepth)/max(stableDepth,0.01) < depthTol && dot(stableNorm, sg00.xyz) > normTol);
        let v10 = select(0.0, fx      *(1.0-fy), abs(sg10.w - stableDepth)/max(stableDepth,0.01) < depthTol && dot(stableNorm, sg10.xyz) > normTol);
        let v01 = select(0.0, (1.0-fx)*fy,       abs(sg01.w - stableDepth)/max(stableDepth,0.01) < depthTol && dot(stableNorm, sg01.xyz) > normTol);
        let v11 = select(0.0, fx      *fy,        abs(sg11.w - stableDepth)/max(stableDepth,0.01) < depthTol && dot(stableNorm, sg11.xyz) > normTol);
        let vSum = v00 + v10 + v01 + v11;

        // No valid taps — treat as disoccluded
        if (vSum < 1e-6) {
            if (isSkippedPixel) {
                // Skipped pixel with no valid reprojection — pass through current-position history
                let fallback = textureLoad(taaHistIn, pixel, 0);
                textureStore(taaOut, pixel, vec4<f32>(fallback.xyz, fallback.w));
            } else {
                textureStore(taaOut, pixel, vec4<f32>(curColor, 1.0));
            }
            return;
        }
        let inv = 1.0 / vSum;
        let histColor   = (h00.xyz*v00 + h10.xyz*v10 + h01.xyz*v01 + h11.xyz*v11) * inv;
        let prevHistLen = (h00.w  *v00 + h10.w  *v10 + h01.w  *v01 + h11.w  *v11) * inv;

        // SVGF-style: pure EMA accumulation, no color-box clamping.
        // The moments-driven à-trous spatial filter handles all noise removal.
        // Ghosting is controlled by depth disocclusion (above) + touchedMoved cap.

        let primaryRoughForCap = textureLoad(albedoTex, pixel, 0).w;
        let roughT = smoothstep(0.05, 0.3, primaryRoughForCap);
        let maxHist = select(
            256.0,                    // diffuse: large history
            mix(8.0, 32.0, roughT),  // specular: 8 smooth → 32 rough
            isSpec
        );
        var effHistLen = min(prevHistLen, maxHist);
        if (touchedMoved) { effHistLen = min(effHistLen, 4.0); }

        // Alpha floor: 1/32 while converging / in motion; drop to 1/64 once stable
        // (history > 32 frames and pixel surface not moving) to halve noise injection
        // at convergence without slowing response to lighting changes.
        let stableFloor = select(1.0/32.0, 1.0/64.0, effHistLen > 32.0 && !touchedMoved);
        let alpha = max(stableFloor, 1.0 / (effHistLen + 1.0));
        var finalAlpha = alpha;
        if (isSpec) {
            let smoothness = 1.0 - clamp(primaryRoughForCap, 0.0, 1.0);
            let specBoost = mix(1.0, 2.5, smoothness * smoothness);
            finalAlpha = min(1.0, alpha * specBoost);
        }

        // Checkerboard/foveated skip: run the same EMA blend but substitute
        // reprojected history for the missing current sample.  This applies the
        // identical alpha-weighted transform that traced pixels receive, so both
        // halves of the checkerboard have matching brightness — no flash.
        let blendSrc = select(curColor, histColor, isSkippedPixel);

        result = mix(histColor, blendSrc, finalAlpha);
        newHistLen = effHistLen + 1.0;
    } else if (isSkippedPixel) {
        // Skipped pixel with no usable history (disoccluded or out of bounds)
        // — pass through whatever history exists at this pixel position.
        let fallback = textureLoad(taaHistIn, pixel, 0);
        textureStore(taaOut, pixel, vec4<f32>(fallback.xyz, max(fallback.w, 1.0)));
        return;
    }

    // .w = per-pixel history length (used by spatial filter for variance-adaptive sigma)
    textureStore(taaOut, pixel, vec4<f32>(result, newHistLen));
}
)";

// ---------------------------------------------------------------------------
// WGSL spatial pre-filter for temporal denoiser pipeline.
// Lightweight 5×5 cross-bilateral: stabilizes neighborhood statistics
// so the temporal filter's variance clamping box is meaningful.
// No luminance edge-stopping (signal too noisy), no demodulation.
// ---------------------------------------------------------------------------
constexpr const char* preFilterWGSL = R"(
struct PreFilterUni { stepSize: u32, _p0: u32, _p1: f32, _p2: f32, }

@group(0) @binding(0) var<uniform> uni: PreFilterUni;
@group(0) @binding(1) var colorIn:    texture_2d<f32>;
@group(0) @binding(2) var colorOut:   texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var gBuf:       texture_2d<f32>;
@group(0) @binding(4) var hitMeshBuf: texture_2d<f32>;

@compute @workgroup_size(8, 8)
fn prefilter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(gid.xy);
    let res   = vec2<i32>(textureDimensions(colorIn, 0));
    if (pixel.x >= res.x || pixel.y >= res.y) { return; }

    let step = i32(uni.stepSize);

    let cSamp  = textureLoad(colorIn, pixel, 0);
    let cColor = cSamp.xyz;
    let cFC    = cSamp.w;
    let cGB    = textureLoad(gBuf, pixel, 0);
    let cNorm  = cGB.xyz;
    let cDepth = cGB.w;
    let cMeshId = u32(textureLoad(hitMeshBuf, pixel, 0).r);

    // Sky or foveated-skip sentinel: pass through
    if (cDepth < 1e-6 || cFC < -0.5) {
        textureStore(colorOut, pixel, vec4<f32>(cColor, cFC));
        return;
    }

    // 5×5 Gaussian kernel weights (separable)
    let kw = array<f32, 5>(1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0);

    var colorSum = vec3<f32>(0.0);
    var weightSum = 0.0;
    let fireflyCap = 12.0;

    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let sp = clamp(pixel + vec2<i32>(dx, dy) * step, vec2<i32>(0), res - 1);

            let sMeshId = u32(textureLoad(hitMeshBuf, sp, 0).r);
            if (sMeshId != cMeshId) { continue; }

            let sGB    = textureLoad(gBuf, sp, 0);
            let sNorm  = sGB.xyz;
            let sDepth = sGB.w;

            // Spatial weight
            let w_s = kw[dy + 2] * kw[dx + 2];
            // Normal edge-stopping (relaxed — pow=4)
            let w_n = pow(max(0.0, dot(cNorm, sNorm)), 4.0);
            // Depth edge-stopping (relaxed)
            let w_d = exp(-abs(cDepth - sDepth) * 1.0 / (cDepth + 0.01));

            // Read and firefly-clamp neighbor
            var sColor = textureLoad(colorIn, sp, 0).xyz;
            let sLum = dot(sColor, vec3<f32>(0.2126, 0.7152, 0.0722));
            if (sLum > fireflyCap) { sColor = sColor * (fireflyCap / sLum); }

            let w = w_s * w_n * w_d;
            colorSum += sColor * w;
            weightSum += w;
        }
    }

    let result = select(cColor, colorSum / weightSum, weightSum > 1e-6);
    textureStore(colorOut, pixel, vec4<f32>(result, cFC));
}
)";

// ---------------------------------------------------------------------------
// WGSL variance-guided à-trous spatial filter
// Uses the frame count (w channel) to adapt filter strength:
//   low frame count → more aggressive blur to suppress MC noise
//   high frame count → gentle filter to preserve converged detail
// ---------------------------------------------------------------------------
constexpr const char* svgfAtrousWGSL = R"(
struct AtrousUni { stepSize: u32, mode: u32, frameCount: f32, _p1: f32, }
// mode: 0 = combined/diffuse (wide kernel, relaxed), 1 = specular (tight, aggressive firefly clamp)

@group(0) @binding(0) var<uniform> uni:      AtrousUni;
@group(0) @binding(1) var colorIn:  texture_2d<f32>;
@group(0) @binding(2) var colorOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var gBuf:     texture_2d<f32>;
@group(0) @binding(4) var albedoBuf: texture_2d<f32>;
@group(0) @binding(5) var hitMeshBuf: texture_2d<f32>;
@group(0) @binding(6) var momentsBuf: texture_2d<f32>;
@group(0) @binding(7) var stableGBuf: texture_2d<f32>;  // unjittered: normal.xyz + depth.w

fn luminance_a(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Demodulate: divide out albedo to isolate irradiance (smooth signal).
fn demod(color: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    return color / max(albedo, vec3<f32>(0.1));
}

@compute @workgroup_size(8, 8)
fn svgf_atrous_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(gid.xy);
    let res   = vec2<i32>(textureDimensions(colorIn, 0));
    if (pixel.x >= res.x || pixel.y >= res.y) { return; }

    let step = i32(uni.stepSize);

    let cSamp   = textureLoad(colorIn, pixel, 0);
    let cColor  = cSamp.xyz;
    let cGB     = textureLoad(gBuf, pixel, 0);
    let cNorm   = cGB.xyz;
    let cDepth  = cGB.w;
    // Stable G-buffer: unjittered primary ray hit — normal and depth are
    // consistent frame-to-frame, eliminating square patch artifacts from jitter.
    let cStableGB = textureLoad(stableGBuf, pixel, 0);
    let cStableN  = cStableGB.xyz;
    let cStableD  = cStableGB.w;
    let cAlbedoFull = textureLoad(albedoBuf, pixel, 0);
    let cAlbedo = cAlbedoFull.xyz;
    let cRough  = cAlbedoFull.w;  // linear roughness (0 = mirror, 1 = diffuse)
    let cHitId  = textureLoad(hitMeshBuf, pixel, 0);
    let cMeshId = u32(cHitId.r);
    let cMatId  = i32(cHitId.g);

    // mode: 0 = diffuse, 1 = specular, 2 = diffuse temporal, 3 = specular temporal
    let isSpec = (uni.mode & 1u) != 0u;
    let isTemporal = (uni.mode & 2u) != 0u;
    let cFC = cSamp.w;

    // Sky pixels: pass through
    if (cDepth < 1e-6) {
        textureStore(colorOut, pixel, vec4<f32>(cColor, cFC));
        return;
    }

    // Demodulate center pixel for accumulation path (diffuse only)
    let cIrr = select(demod(cColor, cAlbedo), cColor, isSpec);
    // Blend luminance source: demod for bright surfaces (texture-aware), raw for dark (stable)
    let albedoLum = luminance_a(cAlbedo);
    let demodBlend = select(smoothstep(0.05, 0.2, albedoLum), 0.0, isSpec);
    let cLum = mix(luminance_a(cColor), luminance_a(cIrr), demodBlend);

    // Variance-guided luminance sigma.
    // Temporal variance from moments (reprojected, stable after ~10 frames).
    let moments = textureLoad(momentsBuf, pixel, 0).xy;
    let temporalVar = max(moments.y - moments.x * moments.x, 0.0);
    let baseSigma = select(0.7, mix(0.5, 2.0, cRough), isSpec);
    let lumSigma  = max(select(0.02, 0.1, isSpec), sqrt(temporalVar) * baseSigma);

    // 5×5 bilateral filter — tracks both demodulated irradiance and raw color
    let kw = array<f32, 5>(1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0);

    var irrSum    = vec3<f32>(0.0);
    var rawSum    = vec3<f32>(0.0);
    var weightSum = 0.0;
    // Spatial luminance moments — accumulated during the filter loop below.
    // Used to bootstrap variance in early frames before temporal moments converge.
    var spatLumM1 = 0.0;
    var spatLumM2 = 0.0;
    var spatWsum  = 0.0;

    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let sp      = clamp(pixel + vec2<i32>(dx, dy) * step, vec2<i32>(0), res - 1);
            let sHitId  = textureLoad(hitMeshBuf, sp, 0);
            let sMeshId = u32(sHitId.r);
            let sMatId  = i32(sHitId.g);
            // Reject only when BOTH mesh and material differ. This allows the
            // filter to share samples across distinct meshes that have the same
            // material (two chairs, repeated walls, foliage instances), which
            // roughly doubles usable samples per pixel in scenes with repeated
            // elements, without introducing cross-material light leak.
            if (sMeshId != cMeshId && sMatId != cMatId) { continue; }

            let sColor  = textureLoad(colorIn, sp, 0).xyz;
            let sGB     = textureLoad(gBuf, sp, 0);
            let sAlbedo = textureLoad(albedoBuf, sp, 0).xyz;
            let sIrr    = demod(sColor, sAlbedo);
            let sNorm   = sGB.xyz;
            let sStableGB = textureLoad(stableGBuf, sp, 0);
            let sStableN  = sStableGB.xyz;
            let sStableD  = sStableGB.w;

            // Spatial weight (separable Gaussian)
            let w_s = kw[dy + 2] * kw[dx + 2];
            // Normal edge-stopping: FC-adaptive for BOTH modes.
            // Low cFC (noisy, moving): relaxed (pow=2) so the filter can smooth
            // curved surfaces (torus, sphere) where normals vary rapidly.
            // High cFC (converged): aggressive (pow=128) to preserve sharp edges.
            // The material ID check already prevents cross-object bleed.
            let specNormPow = mix(2.0, 16.0, smoothstep(0.0, 32.0, cFC));
            let diffNormPow = mix(2.0, 128.0, smoothstep(0.0, 48.0, cFC));
            let normPow = select(diffNormPow, specNormPow, isSpec);
            let w_n = pow(max(0.0, dot(cStableN, sStableN)), normPow);
            // Depth edge-stopping using stable (unjittered) depth — consistent
            // frame-to-frame, no square patch artifacts from jitter.
            let depthBase = select(4.0, mix(1.0, 3.0, cRough), isSpec);
            let depthScale = mix(1.0, depthBase, smoothstep(0.0, 32.0, cFC));
            let w_d = exp(-abs(cStableD - sStableD) * depthScale / (cStableD + 0.01));
            // Luminance edge-stopping: wider sigma while noisy, but not so wide
            // that shadow boundaries blur out
            let lumBoost = mix(4.0, 1.0, smoothstep(0.0, 16.0, cFC));
            let effectiveLumSigma = lumSigma * lumBoost;
            let sAlbLum = luminance_a(sAlbedo);
            let sDemod = smoothstep(0.05, 0.2, sAlbLum);
            let sLum = mix(luminance_a(sColor), luminance_a(sIrr), sDemod);
            // Temporal cleanup: widen luminance sigma to tolerate TAA noise residuals
            // without creating grid patches, but keep relative scaling so strong
            // edges (glass/transmission) are still preserved.
            let finalLumSigma = select(effectiveLumSigma, effectiveLumSigma * 3.0, isTemporal);
            let w_l = exp(-(sLum - cLum) * (sLum - cLum) / (finalLumSigma * finalLumSigma + 1e-6));

            // Per-sample outlier clamp: suppress extreme bright samples in filter window.
            // Specular mode: tighter cap to kill fireflies without blurring.
            var sIrrClamped  = sIrr;
            var sColorClamped = sColor;
            let sIrrLum = luminance_a(sIrr);
            let sColLum = luminance_a(sColor);
            // Specular firefly cap: roughness-adaptive — rough metals tolerate
            // higher values (wider lobe, more variance), glossy needs tight cap.
            // Aggressive clamping is critical since we don't have TAA temporal smoothing
            // on the specular split buffer.
            let specCap = mix(4.0, 10.0, cRough);
            let filterCap = select(12.0, specCap, isSpec);
            if (sIrrLum > filterCap) { sIrrClamped  = sIrr  * (filterCap / sIrrLum); }
            if (sColLum > filterCap) { sColorClamped = sColor * (filterCap / sColLum); }

            let w = w_s * w_n * w_d * w_l;
            irrSum    += sIrrClamped  * w;
            rawSum    += sColorClamped * w;
            weightSum += w;
            // Spatial luminance moments (unweighted by edge-stopping — spatial-only weight)
            // Used for variance bootstrapping in early frames before temporal moments converge.
            spatLumM1 += sLum * w_s;
            spatLumM2 += sLum * sLum * w_s;
            spatWsum  += w_s;
        }
    }

    // Blend temporal and spatial variance for diagnostics / future passes:
    // Early frames (cFC < 8): temporal moments haven't converged → rely on spatial estimate.
    // Converged (cFC > 32): temporal moments are stable → use them (more accurate, follows surfaces).
    let spatVar  = select(temporalVar, max(spatLumM2/spatWsum - (spatLumM1/spatWsum)*(spatLumM1/spatWsum), 0.0), spatWsum > 1e-6);
    let varBlend = smoothstep(4.0, 32.0, cFC);
    let variance = mix(spatVar, temporalVar, varBlend);

    // Blend: demod/remod path for bright surfaces, raw filter for dark surfaces
    let filteredIrr = select(cIrr, irrSum / weightSum, weightSum > 1e-6);
    let filteredRaw = select(cColor, rawSum / weightSum, weightSum > 1e-6);
    let demodResult = filteredIrr * cAlbedo;
    let spatialResult = mix(filteredRaw, demodResult, demodBlend);
    // Roughness-driven blend for specular: smooth metals (low roughness)
    // have clean reflections — don't blur them. Rough surfaces benefit
    // from spatial filtering. Diffuse always gets full filtering.
    let specRoughBlend = smoothstep(0.05, 0.3, cRough);
    let filterBlend = select(1.0, specRoughBlend, isSpec);
    let result = mix(cColor, spatialResult, filterBlend);
    textureStore(colorOut, pixel, vec4<f32>(result, cFC));
}
)";

// ---------------------------------------------------------------------------
// WGSL depth-fill shader — reconstruct rasterizer depth from path-tracer gBuffer.
// Reads primary-ray hit distance (t) from the gBuffer's w channel, reconstructs
// the world-space hit point, then projects it to WebGPU NDC [0,1] depth.
// ---------------------------------------------------------------------------
constexpr const char* depthFillWGSL = R"(
struct DepthFillUniforms {
    projView:   mat4x4<f32>,
    camOri:     vec4<f32>,
    camFwd:     vec4<f32>,
    camRgt:     vec4<f32>,
    camUp:      vec4<f32>,
    iRes:       vec4<f32>,
    tanHalfFov: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: DepthFillUniforms;
@group(0) @binding(1) var gBuf: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(vid & 1u) * 4.0 - 1.0;
    let y = f32((vid >> 1u) & 1u) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) fpos: vec4<f32>) -> @builtin(frag_depth) f32 {
    let px  = vec2<i32>(i32(fpos.x), i32(fpos.y));
    let t   = textureLoad(gBuf, px, 0).w;
    if (t <= 0.0) { return 1.0; }
    let ndc    = vec2<f32>((fpos.x / u.iRes.x) * 2.0 - 1.0,
                            1.0 - (fpos.y / u.iRes.y) * 2.0);
    let aspect = u.iRes.x / u.iRes.y;
    let rayDir = normalize(u.camFwd.xyz
                         + u.camRgt.xyz * (ndc.x * u.tanHalfFov.x * aspect)
                         + u.camUp.xyz  * (ndc.y * u.tanHalfFov.x));
    let worldPos = u.camOri.xyz + t * rayDir;
    let clip     = u.projView * vec4<f32>(worldPos, 1.0);
    if (clip.w <= 0.0) { return 1.0; }
    return clamp(clip.z / clip.w, 0.0, 1.0);
}
)";

// ---------------------------------------------------------------------------
// WGSL display shader — blit accumulated texture to screen with ACES + gamma
// ---------------------------------------------------------------------------
constexpr const char* displayWGSL = R"(
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
@group(0) @binding(2) var accumTex: texture_2d<f32>;
// binding 3 = accumTex sampler (unused)
@group(0) @binding(4) var diffTex:  texture_2d<f32>;   // diffuse radiance
// binding 5 = diffTex sampler (unused)
@group(0) @binding(6) var gBufTex:  texture_2d<f32>;  // gBuf: .w = primary hit t, 0 = background
// binding 7 = gBufTex sampler (unused)
@group(0) @binding(8) var specTex:     texture_2d<f32>;   // specular radiance
@group(0) @binding(10) var upscaleTex: texture_2d<f32>;   // TAAU full-res output (1×1 dummy when inactive)

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return transform.proj * transform.view * transform.model * vec4<f32>(position, 1.0);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

// Tonemap + gamma-correct a full-res TAAU pixel.
// histLen < 0 means sky sentinel — apply gamma but no ACES.
fn tonemapUpscaleAt(p: vec2<i32>, sz: vec2<i32>, expo: f32) -> vec3<f32> {
    let pc = clamp(p, vec2<i32>(0), sz - vec2<i32>(1, 1));
    let s  = textureLoad(upscaleTex, pc, 0);
    if (s.w < 0.0) { return pow(max(s.xyz, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)); }
    return pow(aces(s.xyz * expo), vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_main(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32> {
    // _pad encodes both pixelScale and AOV mode:
    //   _pad = pixelScale + aovMode * 10.0
    // Normal rendering: aovMode=0, _pad = pixelScale (0.1-2.0)
    // AOV mode N: _pad = pixelScale + N*10  (N in 1..5, so _pad > 10)
    let rawPad    = transform._pad;
    let aovMode   = i32(rawPad / 10.0);
    let pixScale  = max(rawPad - f32(aovMode) * 10.0, 0.01);
    let accumSize = vec2<i32>(textureDimensions(accumTex, 0));
    let exposure  = transform.cameraPos.z;

    // AOV mode: direct passthrough of diffTex, gamma only (no tone mapping)
    if (aovMode > 0) {
        let px = clamp(vec2<i32>(fragPos.xy * pixScale), vec2<i32>(0), accumSize - 1);
        let col = textureLoad(diffTex, px, 0).xyz;
        return vec4<f32>(pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
    }

    // Temporal upscale path: if upscaleTex is full-res (> 1×1), TAAU is active.
    let upscaleSize = vec2<i32>(textureDimensions(upscaleTex, 0));
    if (upscaleSize.x > 1) {
        let px = vec2<i32>(i32(fragPos.x), i32(fragPos.y));
        return vec4<f32>(tonemapUpscaleAt(px, upscaleSize, exposure), 1.0);
    }

    // When rendering at reduced resolution (pixelScale < 1), use joint bilateral
    // upsampling guided by the G-buffer (normal + depth). This reconstructs sharp
    // edges at mesh/material boundaries while smoothly interpolating interior
    // regions — effectively lossless for geometric edges, unlike nearest-neighbor
    // which produces blocky silhouettes or bilinear which blurs across edges.
    //
    // For each full-res pixel, compute its floating-point position in the low-res
    // buffer, then sample a 2×2 footprint of low-res texels. Weight each texel by
    // spatial proximity (bilinear) × normal similarity × depth similarity. Texels
    // on a different surface (normal mismatch or depth discontinuity) get near-zero
    // weight, so the filter never blends across edges.
    if (pixScale < 0.85) {
        // G-buffer is always at the same resolution as the accum buffer
        let pNearest = clamp(vec2<i32>(fragPos.xy * pixScale), vec2<i32>(0), accumSize - 1);
        let gRef = textureLoad(gBufTex, pNearest, 0);
        let nRef = gRef.xyz;
        let dRef = gRef.w;

        // Background: skip bilateral, just nearest-neighbor
        if (dRef <= 0.0) {
            let col = textureLoad(diffTex, pNearest, 0).xyz + textureLoad(specTex, pNearest, 0).xyz;
            return vec4<f32>(pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
        }

        // Low-res 2×2 footprint for bilateral interpolation
        let fp  = fragPos.xy * pixScale - 0.5;
        let p0  = vec2<i32>(i32(floor(fp.x)), i32(floor(fp.y)));
        let f   = fp - floor(fp);

        let bw = vec4<f32>((1.0 - f.x) * (1.0 - f.y),
                                  f.x  * (1.0 - f.y),
                           (1.0 - f.x) *        f.y,
                                  f.x  *        f.y);

        let offsets = array<vec2<i32>, 4>(
            vec2<i32>(0, 0), vec2<i32>(1, 0),
            vec2<i32>(0, 1), vec2<i32>(1, 1));

        var colSum  = vec3<f32>(0.0);
        var wSum    = 0.0;
        let bwa = array<f32, 4>(bw.x, bw.y, bw.z, bw.w);

        for (var i = 0u; i < 4u; i++) {
            let sp = clamp(p0 + offsets[i], vec2<i32>(0), accumSize - 1);
            let g  = textureLoad(gBufTex, sp, 0);
            let sn = g.xyz;
            let sd = g.w;

            if (sd <= 0.0) { continue; }

            let wn = pow(max(0.0, dot(nRef, sn)), 64.0);
            let wd = exp(-abs(dRef - sd) * 8.0 / max(dRef, 0.01));

            let w = bwa[i] * wn * wd;
            colSum += (textureLoad(diffTex, sp, 0).xyz + textureLoad(specTex, sp, 0).xyz) * w;
            wSum   += w;
        }

        var col: vec3<f32>;
        if (wSum > 1e-6) {
            col = colSum / wSum;
        } else {
            col = textureLoad(diffTex, pNearest, 0).xyz + textureLoad(specTex, pNearest, 0).xyz;
        }

        let gc = pow(aces(col * exposure), vec3<f32>(1.0 / 2.2));
        return vec4<f32>(gc, 1.0);
    }

    // Native resolution — tonemap and gamma correct.
    let px  = clamp(vec2<i32>(fragPos.xy * pixScale), vec2<i32>(0), accumSize - 1);
    let col = textureLoad(diffTex, px, 0).xyz + textureLoad(specTex, px, 0).xyz;
    if (textureLoad(gBufTex, px, 0).w <= 0.0) {
        return vec4<f32>(pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
    }
    return vec4<f32>(pow(aces(col * exposure), vec3<f32>(1.0 / 2.2)), 1.0);
}
)";

// ---------------------------------------------------------------------------
// CPU-side uniform struct (matches WGSL RtUniforms, 512 bytes)
// ---------------------------------------------------------------------------
struct alignas(16) RtGpuUniforms {
    float camOri[4];
    float camFwd[4];
    float camRgt[4];
    float camUp[4];
    float prevCamOri[4];
    float prevCamFwd[4];
    float prevCamRgt[4];
    float prevCamUp[4];
    float iRes[4];
    float tanHalfFov[4];
    float frameCount[4];
    float triCount[4];
    float mode[4];
    float lightCount[4];
    float lightPos[4][4];
    float lightCol[4][4];
    float lightType[4][4];      // [i][0]=type (0=pt,1=dir,2=spot), [i][1]=cosAngle, [i][2]=cosOuter, [i][3]=distance
    float lightDir[4][4];       // [i] = spotlight direction xyz, [i][3]=decay
    float    spp[4];
    uint32_t movedMeshBits[4];  // bit i = mesh i moved; 4 words cover meshes 0–127
    float    envColor[4];       // xyz = color, w = mode (0=none, 1=solid color, 2=equirect tex)
    float    envIntensity[4];   // x = intensity, y = envWidth, z = envHeight, w = totalLumSum (0 = no CDF)
    float    bgColor[4];       // xyz = color, w = mode (0=sky gradient, 1=solid color, 2=equirect bgTex)
    float    params[4];        // x = maxBounces
    float    emissiveInfo[4]; // x = emissive tri count, y = total emissive power, z = fireflyCap
    float    restirParams[4]; // x = enabled, y = M_clamp, z = reserved, w = reserved
};
static_assert(sizeof(RtGpuUniforms) == 608, "RtGpuUniforms must be 608 bytes");

struct alignas(16) VtGpuUniforms {
    uint32_t triCount, groupsX, splitAt, _p1;
};
struct alignas(16) RefitGpuUniforms {
    uint32_t leafCount, groupsX, _p[2];
};
struct alignas(16) AtrousGpuUniforms {
    uint32_t stepSize, mode;  // mode: 0=diffuse, 1=specular
    float    frameCount, _p1;
};
struct alignas(16) PreFilterGpuUniforms {
    uint32_t stepSize, _p0;
    float    _p1, _p2;
};
struct alignas(16) DepthFillUniforms {
    float projView[16];   // NDC-remapped proj * view  (64 bytes)
    float camOri[4];      // camera position            (16 bytes)
    float camFwd[4];      // camera forward             (16 bytes)
    float camRgt[4];      // camera right               (16 bytes)
    float camUp[4];       // camera up                  (16 bytes)
    float iRes[4];        // resolution xy              (16 bytes)
    float tanHalfFov[4];  // x = tanHalfFov             (16 bytes)
    // total = 160 bytes (16-byte aligned, no padding needed)
};
struct alignas(16) TaaGpuUniforms {
    float prevCamOri[4], prevCamFwd[4], prevCamRgt[4], prevCamUp[4];
    float curCamOri[4],  curCamFwd[4],  curCamRgt[4],  curCamUp[4];
    float iRes[4];          // [0]=w [1]=h [2]=prevJx [3]=prevJy
    float tanHalfFov[4];
    float frameCount[4];   // [0]=FC [1]=mode(0=diff,1=spec) [2]=curJx [3]=curJy
    uint32_t movedMeshBits[4];
};
static_assert(sizeof(TaaGpuUniforms) == 192, "TaaGpuUniforms must be 192 bytes");

struct alignas(16) UpscaleGpuUniforms {
    float prevCamOri[4], prevCamFwd[4], prevCamRgt[4], prevCamUp[4]; // 64 bytes
    float curCamOri[4],  curCamFwd[4],  curCamRgt[4],  curCamUp[4]; // 64 bytes
    float iRes[4];        // [0]=fullW [1]=fullH [2]=pixelScale [3]=0
    float tanHalfFov[4];
    float frameCount[4];  // [0]=FC
};
static_assert(sizeof(UpscaleGpuUniforms) == 176, "UpscaleGpuUniforms must be 176 bytes");

// ---------------------------------------------------------------------------
// Binary BVH node (used during build, then collapsed to BVH4)
// ---------------------------------------------------------------------------
struct BvhNode {
    float minX, minY, minZ;
    int left;
    float maxX, maxY, maxZ;
    int right;
    int parent;
    int _pad[3];
};

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build 2D CDF for importance-sampling an equirectangular environment map.
/// Returns {conditionalCDF (width*height floats), marginalCDF (height floats), totalLuminanceSum}.
/// Both CDFs are normalized to [0,1]. The sin(theta) Jacobian is baked in so that
/// pixels near the equator (more solid angle) are sampled proportionally more.
struct EnvCdfResult {
    std::vector<float> conditional; // width × height
    std::vector<float> marginal;    // height
    float totalSum = 0.f;
    int width = 0, height = 0;
};

template<typename T>
static EnvCdfResult buildEnvCdf(const std::vector<T>& pixels, int w, int h) {
    EnvCdfResult r;
    r.width = w;
    r.height = h;
    r.conditional.resize(static_cast<size_t>(w) * h);
    r.marginal.resize(h);

    // Row luminance sums (weighted by sin(theta) for solid angle)
    for (int y = 0; y < h; ++y) {
        const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(h);
        const float theta = (0.5f - v) * 3.14159265359f;
        const float sinTheta = std::max(std::abs(std::sin(theta)), 1e-6f);

        float rowSum = 0.f;
        for (int x = 0; x < w; ++x) {
            const size_t idx = (static_cast<size_t>(y) * w + x) * 4;
            float rf, gf, bf;
            if constexpr (std::is_same_v<T, float>) {
                rf = pixels[idx]; gf = pixels[idx + 1]; bf = pixels[idx + 2];
            } else {
                rf = pixels[idx] / 255.f; gf = pixels[idx + 1] / 255.f; bf = pixels[idx + 2] / 255.f;
            }
            const float lum = (0.2126f * rf + 0.7152f * gf + 0.0722f * bf) * sinTheta;
            rowSum += lum;
            r.conditional[static_cast<size_t>(y) * w + x] = rowSum;
        }
        // Normalize conditional CDF for this row
        if (rowSum > 1e-10f) {
            for (int x = 0; x < w; ++x)
                r.conditional[static_cast<size_t>(y) * w + x] /= rowSum;
        } else {
            // Uniform fallback for black rows
            for (int x = 0; x < w; ++x)
                r.conditional[static_cast<size_t>(y) * w + x] = static_cast<float>(x + 1) / static_cast<float>(w);
        }
        r.totalSum += rowSum;
        r.marginal[y] = r.totalSum;
    }
    // Normalize marginal CDF
    if (r.totalSum > 1e-10f) {
        for (int y = 0; y < h; ++y)
            r.marginal[y] /= r.totalSum;
    } else {
        for (int y = 0; y < h; ++y)
            r.marginal[y] = static_cast<float>(y + 1) / static_cast<float>(h);
    }
    return r;
}

// ---------------------------------------------------------------------------
// Instancing support: flatten InstancedMesh into individual entries
// ---------------------------------------------------------------------------

struct RtMeshEntry {
    Mesh* mesh;
    Matrix4 worldMatrix;
};

/// Expand the mesh list so that each InstancedMesh becomes N separate entries
/// (one per instance), each with its own effective world matrix.
/// Regular meshes produce a single entry.
static std::vector<RtMeshEntry> expandMeshEntries(const std::vector<Mesh*>& meshes) {
    std::vector<RtMeshEntry> entries;
    for (auto* mesh : meshes) {
        auto* inst = dynamic_cast<InstancedMesh*>(mesh);
        if (inst && inst->count() > 0) {
            for (size_t j = 0; j < inst->count(); ++j) {
                Matrix4 instMat;
                inst->getMatrixAt(j, instMat);
                Matrix4 world;
                world.multiplyMatrices(*mesh->matrixWorld, instMat);
                entries.push_back({mesh, world});
            }
        } else {
            entries.push_back({mesh, *mesh->matrixWorld});
        }
    }
    return entries;
}

/// Build texture atlas sized to actual slot usage (not MAX_TEX_SLOTS).
/// Returns {pixel data, rows, columns, tileSize}.
static std::tuple<std::vector<unsigned char>, int, int, int> buildAtlas(
        const std::vector<Mesh*>& meshes,
        std::unordered_map<Texture*, int>& texSlotMap,
        int TILE_SIZE = DEFAULT_TILE_SIZE) {
    const int ATLAS_COLS = ATLAS_WIDTH / TILE_SIZE;
    // First pass: count unique textures to determine atlas size.
    int slotCount = 0;
    std::unordered_set<Texture*> seen;
    for (auto* mesh : meshes) {
        if (slotCount >= MAX_TEX_SLOTS) break;
        auto countTex = [&](Texture* tex) {
            if (tex && slotCount < MAX_TEX_SLOTS && !seen.count(tex)) {
                auto& img = tex->image();
                if (img.width > 0 && img.height > 0) {
                    seen.insert(tex);
                    ++slotCount;
                }
            }
        };
        if (auto* mwm = dynamic_cast<MaterialWithMap*>(mesh->material().get()))
            countTex(mwm->map.get());
        if (auto* mnm = dynamic_cast<MaterialWithNormalMap*>(mesh->material().get()))
            countTex(mnm->normalMap.get());
        if (auto* mwr = dynamic_cast<MaterialWithRoughness*>(mesh->material().get()))
            countTex(mwr->roughnessMap.get());
        if (auto* mwe = dynamic_cast<MaterialWithEmissive*>(mesh->material().get()))
            countTex(mwe->emissiveMap.get());
        if (auto* mwa = dynamic_cast<MaterialWithAoMap*>(mesh->material().get()))
            countTex(mwa->aoMap.get());
    }

    // Layout: 8×8 grid of 1024px tiles per layer (8192×8192), multiple layers for >64 textures
    const int slotsPerLayer = ATLAS_COLS * ATLAS_COLS; // 64
    const int numLayers = std::max(1, (slotCount + slotsPerLayer - 1) / slotsPerLayer);
    const int atlasCols = ATLAS_COLS;
    const int rows = ATLAS_COLS; // square grid per layer
    const int atlasW = ATLAS_COLS * TILE_SIZE;
    const int atlasH = ATLAS_COLS * TILE_SIZE;
    const size_t layerBytes = static_cast<size_t>(atlasW) * atlasH * 4;
    std::vector<unsigned char> atlas(layerBytes * numLayers, 255);

    auto addTexture = [&](Texture* tex, int& slot) {
        if (!tex || slot >= MAX_TEX_SLOTS) return;
        if (texSlotMap.count(tex)) return;
        auto& img = tex->image();
        if (img.width == 0 || img.height == 0) return;
        const auto& src = img.data<unsigned char>();
        const int srcW = static_cast<int>(img.width);
        const int srcH = static_cast<int>(img.height);
        const int ch = static_cast<int>(src.size()) / (srcW * srcH);

        const int layer = slot / slotsPerLayer;
        const int localSlot = slot % slotsPerLayer;
        const int col = localSlot % ATLAS_COLS;
        const int row = localSlot / ATLAS_COLS;
        const int destX = col * TILE_SIZE;
        const int destY = row * TILE_SIZE;
        unsigned char* layerBase = atlas.data() + layer * layerBytes;

        if (srcW == TILE_SIZE && srcH == TILE_SIZE && ch == 4) {
            for (int ty = 0; ty < TILE_SIZE; ++ty) {
                const int di = ((destY + ty) * atlasW + destX) * 4;
                const int si = ty * srcW * 4;
                std::memcpy(layerBase + di, src.data() + si, TILE_SIZE * 4);
            }
        } else {
            std::vector<int> xmap(TILE_SIZE);
            for (int tx = 0; tx < TILE_SIZE; ++tx)
                xmap[tx] = tx * srcW / TILE_SIZE;
            for (int ty = 0; ty < TILE_SIZE; ++ty) {
                const int sy = ty * srcH / TILE_SIZE;
                const int srcRowOff = sy * srcW;
                unsigned char* dst = layerBase + ((destY + ty) * atlasW + destX) * 4;
                if (ch == 4) {
                    for (int tx = 0; tx < TILE_SIZE; ++tx) {
                        std::memcpy(dst + tx * 4, src.data() + (srcRowOff + xmap[tx]) * 4, 4);
                    }
                } else {
                    for (int tx = 0; tx < TILE_SIZE; ++tx) {
                        const int si = (srcRowOff + xmap[tx]) * ch;
                        const int di = tx * 4;
                        dst[di + 0] = src[si + 0];
                        dst[di + 1] = src[si + 1];
                        dst[di + 2] = src[si + 2];
                        dst[di + 3] = ch >= 4 ? src[si + 3] : 255u;
                    }
                }
            }
        }
        texSlotMap[tex] = slot++;
    };

    int slot = 0;
    for (auto& mesh : meshes) {
        if (slot >= MAX_TEX_SLOTS) break;
        auto* mwm = dynamic_cast<MaterialWithMap*>(mesh->material().get());
        if (mwm && mwm->map) addTexture(mwm->map.get(), slot);
        auto* mnm = dynamic_cast<MaterialWithNormalMap*>(mesh->material().get());
        if (mnm && mnm->normalMap) addTexture(mnm->normalMap.get(), slot);
        auto* mwr = dynamic_cast<MaterialWithRoughness*>(mesh->material().get());
        if (mwr && mwr->roughnessMap) addTexture(mwr->roughnessMap.get(), slot);
        auto* mwe = dynamic_cast<MaterialWithEmissive*>(mesh->material().get());
        if (mwe && mwe->emissiveMap) addTexture(mwe->emissiveMap.get(), slot);
        auto* mwa = dynamic_cast<MaterialWithAoMap*>(mesh->material().get());
        if (mwa && mwa->aoMap) addTexture(mwa->aoMap.get(), slot);
    }
    std::cerr << "[PathTracer] Atlas: " << slot << " unique textures in "
              << numLayers << " layer(s), " << ATLAS_COLS << "x" << ATLAS_COLS
              << " grid (" << atlasW << "x" << atlasH << " px/layer)" << std::endl;
    return {std::move(atlas), numLayers, ATLAS_COLS, TILE_SIZE};
}

// Encode texture slot index + wrap modes into a single float.
// Layout: slot * 16 + wrapS * 4 + wrapT  (wrap: 0=repeat, 1=clamp, 2=mirror)
static float encodeSlotWrap(int slot, const Texture* tex) {
    int ws = 0, wt = 0;
    if (tex) {
        if (tex->wrapS == TextureWrapping::ClampToEdge) ws = 1;
        else if (tex->wrapS == TextureWrapping::MirroredRepeat) ws = 2;
        if (tex->wrapT == TextureWrapping::ClampToEdge) wt = 1;
        else if (tex->wrapT == TextureWrapping::MirroredRepeat) wt = 2;
    }
    return static_cast<float>(slot * 16 + ws * 4 + wt);
}

struct ExtractedMaterial {
    Color albedo{0.8f, 0.8f, 0.8f};
    float shininess = 8.f;
    float metalness = 0.f;
    Color emissive{0.f, 0.f, 0.f};
    float transmission = 0.f;
    float ior = 1.5f;
    float alphaTest = 0.f;
    Color attenuationColor{1.f, 1.f, 1.f};
    float attenuationDistance = 0.f;
    float clearcoat = 0.f;
    float clearcoatRoughness = 0.f;
    Color sheenColor{0.f, 0.f, 0.f};
    float sheenRoughness = 0.f;
    float specularIntensity = 1.f;
    Color specularColor{1.f, 1.f, 1.f};
    float dispersion = 0.f;
    float thickness = 0.f;
};

static ExtractedMaterial extractMaterial(const Material* mat) {
    ExtractedMaterial m;
    if (!mat) return m;
    if (auto* c = dynamic_cast<const MaterialWithColor*>(mat))
        m.albedo = c->color;
    // MeshBasicMaterial → unlit: shininess = -1 signals no lighting
    if (dynamic_cast<const MeshBasicMaterial*>(mat)) {
        m.shininess = -1.f;
        return m;
    }
    if (auto* r = dynamic_cast<const MaterialWithRoughness*>(mat)) {
        const float rough = std::max(0.f, std::min(1.f, r->roughness));
        m.shininess = std::max(1e-4f, rough * rough);
    } else if (auto* s = dynamic_cast<const MaterialWithSpecular*>(mat)) {
        const float n = std::max(1.f, s->shininess);
        m.shininess = std::max(0.04f, std::sqrt(2.f / (n + 2.f)));
    }
    if (auto* mm = dynamic_cast<const MaterialWithMetalness*>(mat))
        m.metalness = std::max(0.f, std::min(1.f, mm->metalness));
    if (auto* e = dynamic_cast<const MaterialWithEmissive*>(mat))
        m.emissive = Color(e->emissive.r * e->emissiveIntensity,
                         e->emissive.g * e->emissiveIntensity,
                         e->emissive.b * e->emissiveIntensity);
    if (auto* t = dynamic_cast<const MaterialWithTransmission*>(mat)) {
        m.transmission = std::clamp(t->transmission, 0.f, 1.f);
        m.ior = std::max(1.f, t->ior);
        m.dispersion = std::max(0.f, t->dispersion);
    }
    if (auto* a = dynamic_cast<const MaterialWithAttenuation*>(mat)) {
        m.attenuationColor = a->attenuationColor;
        m.attenuationDistance = std::max(0.f, a->attenuationDistance);
    }
    if (auto* th = dynamic_cast<const MaterialWithThickness*>(mat)) {
        m.thickness = std::max(0.f, th->thickness);
    }
    if (auto* cc = dynamic_cast<const MaterialWithClearcoat*>(mat)) {
        m.clearcoat = std::clamp(cc->clearcoat, 0.f, 1.f);
        const float ccr = std::clamp(cc->clearcoatRoughness, 0.f, 1.f);
        m.clearcoatRoughness = std::max(1e-4f, ccr * ccr);
    }
    if (auto* sh = dynamic_cast<const MaterialWithSheen*>(mat)) {
        m.sheenColor = sh->sheenColor;
        m.sheenRoughness = std::clamp(sh->sheenRoughness, 0.f, 1.f);
    }
    if (auto* sp = dynamic_cast<const MaterialWithPbrSpecular*>(mat)) {
        m.specularIntensity = std::max(0.f, sp->specularIntensity);
        m.specularColor = sp->specularColor;
    }
    m.alphaTest = mat->alphaTest;
    return m;
}

// Compute flat index into a paged triBuffer for triangle `ti`, row `row`.
// The texture is TEX_PAGE_WIDTH columns wide; each triangle occupies TRI_TEX_HEIGHT rows.
static int pagedIdx(int ti, int row) {
    return ((ti / TEX_PAGE_WIDTH * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + ti % TEX_PAGE_WIDTH) * 4;
}

static int buildGeometryBuffers(
        const std::vector<RtMeshEntry>& entries,
        const std::unordered_map<Texture*, int>& texSlotMap,
        std::vector<float>& triBuffer,
        std::vector<float>& matBuffer,
        std::vector<float>& rawObjTriBuf,
        std::vector<float>& matrixBuf,
        int maxTris, int maxMats, int maxMeshes) {
    std::ranges::fill(triBuffer, 0.f);
    std::ranges::fill(matBuffer, 0.f);
    std::ranges::fill(rawObjTriBuf, 0.f);
    std::ranges::fill(matrixBuf, 0.f);

    int triCount = 0;
    int matCount = 0;
    int meshCount = 0;

    // Track which Mesh* has already had its material written
    std::unordered_map<Mesh*, int> meshToMatIdx;

    auto setTexel = [&](std::vector<float>& buf, int width, int col, int row,
                        float x, float y, float z, float w) {
        int idx;
        if (width > TEX_PAGE_WIDTH) {
            const int page = col / TEX_PAGE_WIDTH;
            const int pcol = col % TEX_PAGE_WIDTH;
            idx = ((page * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + pcol) * 4;
        } else {
            idx = (row * width + col) * 4;
        }
        buf[idx + 0] = x;
        buf[idx + 1] = y;
        buf[idx + 2] = z;
        buf[idx + 3] = w;
    };
    auto setObj = [&](int ti, int field, float x, float y, float z, float w) {
        float* p = rawObjTriBuf.data() + ti * 48 + field * 4;
        p[0] = x; p[1] = y; p[2] = z; p[3] = w;
    };

    for (auto& entry : entries) {
        if (triCount >= maxTris || meshCount >= maxMeshes) break;

        // Deduplicate material: write once per unique Mesh*
        int matIdx;
        auto matIt = meshToMatIdx.find(entry.mesh);
        if (matIt != meshToMatIdx.end()) {
            matIdx = matIt->second;
        } else {
            if (matCount >= maxMats) continue;
            matIdx = matCount++;
            meshToMatIdx[entry.mesh] = matIdx;

            auto em = extractMaterial(entry.mesh->material().get());
            setTexel(matBuffer, maxMats, matIdx, 0,
                     em.albedo.r, em.albedo.g, em.albedo.b, em.shininess);

            float texSlot = -1.f;
            float normalSlot = -1.f;
            if (auto* mwm = dynamic_cast<MaterialWithMap*>(entry.mesh->material().get())) {
                if (mwm->map) {
                    auto it = texSlotMap.find(mwm->map.get());
                    if (it != texSlotMap.end()) {
                        texSlot = encodeSlotWrap(it->second, mwm->map.get());
                    }
                }
            }
            if (auto* mnm = dynamic_cast<MaterialWithNormalMap*>(entry.mesh->material().get())) {
                if (mnm->normalMap) {
                    auto it = texSlotMap.find(mnm->normalMap.get());
                    if (it != texSlotMap.end()) {
                        normalSlot = encodeSlotWrap(it->second, mnm->normalMap.get());
                    }
                }
            }
            float roughSlot = -1.f;
            if (auto* mwr = dynamic_cast<MaterialWithRoughness*>(entry.mesh->material().get())) {
                if (mwr->roughnessMap) {
                    auto it = texSlotMap.find(mwr->roughnessMap.get());
                    if (it != texSlotMap.end()) {
                        roughSlot = encodeSlotWrap(it->second, mwr->roughnessMap.get());
                    }
                }
            }
            setTexel(matBuffer, maxMats, matIdx, 1, texSlot, em.metalness, normalSlot, roughSlot);
            setTexel(matBuffer, maxMats, matIdx, 2,
                    em.emissive.r, em.emissive.g, em.emissive.b, em.transmission);
            const float doubleSided = (entry.mesh->material()->side == Side::Double || em.transmission > 0.f) ? 1.f : 0.f;
            const float opacity = std::clamp(entry.mesh->material()->opacity, 0.f, 1.f);
            // Encode blend mode: negative opacity signals stochastic alpha (BLEND mode).
            // We must keep BLEND mode any time `transparent=true` without alphaTest,
            // even if scalar opacity==1.0 — because the baseColor texture may still
            // carry a non-trivial alpha channel (logos, decals, cutouts drawn with
            // soft edges). The GPU path has per-texel early-outs that silently
            // promote alpha≥0.99 texels to opaque (zero variance) while preserving
            // the transparent regions, so the logo shape survives without the
            // stochastic-noise cost on solid paint.
            const float opacityEnc = (entry.mesh->material()->transparent && em.alphaTest <= 0.f)
                                     ? -opacity : opacity;
            setTexel(matBuffer, maxMats, matIdx, 3, em.ior, em.alphaTest, doubleSided, opacityEnc);
            setTexel(matBuffer, maxMats, matIdx, 4,
                    em.attenuationColor.r, em.attenuationColor.g, em.attenuationColor.b, em.attenuationDistance);
            float emissiveSlot = -1.f;
            if (auto* mwe = dynamic_cast<MaterialWithEmissive*>(entry.mesh->material().get())) {
                if (mwe->emissiveMap) {
                    auto it = texSlotMap.find(mwe->emissiveMap.get());
                    if (it != texSlotMap.end()) {
                        emissiveSlot = encodeSlotWrap(it->second, mwe->emissiveMap.get());
                    }
                }
            }
            float aoSlot = -1.f;
            if (auto* mwa = dynamic_cast<MaterialWithAoMap*>(entry.mesh->material().get())) {
                if (mwa->aoMap) {
                    auto it = texSlotMap.find(mwa->aoMap.get());
                    if (it != texSlotMap.end()) {
                        aoSlot = encodeSlotWrap(it->second, mwa->aoMap.get());
                    }
                }
            }
            setTexel(matBuffer, maxMats, matIdx, 5, em.clearcoat, em.clearcoatRoughness, emissiveSlot, aoSlot);

            // Per-channel UV transforms (rows 6-15, 2 rows per channel)
            // Layout per channel: row N = (a, b, tx, c), row N+1 = (d, ty, texCoord, 0)
            // where u' = a*u + b*v + tx,  v' = c*u + d*v + ty
            auto writeUvTransform = [&](int row, const Texture* tex) {
                if (tex) {
                    const_cast<Texture*>(tex)->updateMatrix();
                    const auto& e = tex->matrix.elements;
                    // Column-major: e[0]=a, e[3]=b, e[6]=tx, e[1]=c, e[4]=d, e[7]=ty
                    setTexel(matBuffer, maxMats, matIdx, row,
                             e[0], e[3], e[6], e[1]);
                    setTexel(matBuffer, maxMats, matIdx, row + 1,
                             e[4], e[7], static_cast<float>(tex->texCoord), 0.f);
                } else {
                    // Identity transform, UV0
                    setTexel(matBuffer, maxMats, matIdx, row, 1.f, 0.f, 0.f, 0.f);
                    setTexel(matBuffer, maxMats, matIdx, row + 1, 1.f, 0.f, 0.f, 0.f);
                }
            };

            auto* mat = entry.mesh->material().get();
            auto* mwm = dynamic_cast<MaterialWithMap*>(mat);
            auto* mnm = dynamic_cast<MaterialWithNormalMap*>(mat);
            auto* mwr = dynamic_cast<MaterialWithRoughness*>(mat);
            auto* mwe = dynamic_cast<MaterialWithEmissive*>(mat);
            auto* mwa = dynamic_cast<MaterialWithAoMap*>(mat);

            const Texture* uvTextures[5] = {
                mwm ? mwm->map.get() : nullptr,
                mwr ? mwr->roughnessMap.get() : nullptr,
                mnm ? mnm->normalMap.get() : nullptr,
                mwe ? mwe->emissiveMap.get() : nullptr,
                mwa ? mwa->aoMap.get() : nullptr
            };
            writeUvTransform(6,  uvTextures[0]);   // baseColor
            writeUvTransform(8,  uvTextures[1]);   // metalRough
            writeUvTransform(10, uvTextures[2]);    // normal
            writeUvTransform(12, uvTextures[3]);    // emissive
            writeUvTransform(14, uvTextures[4]);    // occlusion

            // Check if any channel uses non-identity UV transform or UV1
            float hasCustomUV = 0.f;
            for (const auto* tex : uvTextures) {
                if (tex) {
                    const_cast<Texture*>(tex)->updateMatrix();
                    const auto& e = tex->matrix.elements;
                    // Identity check: a=1, b=0, tx=0, c=0, d=1, ty=0
                    if (e[0] != 1.f || e[3] != 0.f || e[6] != 0.f ||
                        e[1] != 0.f || e[4] != 1.f || e[7] != 0.f ||
                        tex->texCoord != 0) {
                        hasCustomUV = 1.f;
                        break;
                    }
                }
            }

            // Row 16: sheen (r, g, b, roughness)
            setTexel(matBuffer, maxMats, matIdx, 16,
                    em.sheenColor.r, em.sheenColor.g, em.sheenColor.b, em.sheenRoughness);
            // Row 17: PBR specular (r, g, b, intensity)
            setTexel(matBuffer, maxMats, matIdx, 17,
                    em.specularColor.r, em.specularColor.g, em.specularColor.b, em.specularIntensity);
            // Row 18: dispersion + thickness + hasCustomUV + hasAdvancedPBR flags
            const bool hasAdvanced = (em.sheenColor.r != 0.f || em.sheenColor.g != 0.f || em.sheenColor.b != 0.f ||
                                      em.specularIntensity != 1.f ||
                                      em.specularColor.r != 1.f || em.specularColor.g != 1.f || em.specularColor.b != 1.f ||
                                      em.dispersion != 0.f || em.thickness != 0.f);
            setTexel(matBuffer, maxMats, matIdx, 18, em.dispersion, em.thickness, hasCustomUV, hasAdvanced ? 1.f : 0.f);
        }

        const int meshIdx = meshCount++;

        // Per-entry world matrix
        const auto& world = entry.worldMatrix;
        Matrix4 normalMat(world);
        normalMat.invert().transpose();

        if (meshIdx < maxMeshes) {
            float* mp = matrixBuf.data() + meshIdx * 32;
            std::memcpy(mp, world.elements.data(), 16 * sizeof(float));
            std::memcpy(mp + 16, normalMat.elements.data(), 16 * sizeof(float));
        }

        auto* geo = entry.mesh->geometry().get();
        auto* pos = geo->getAttribute<float>("position");
        if (!pos) continue;
        auto* nrm = geo->getAttribute<float>("normal");
        auto* uvs = geo->getAttribute<float>("uv");
        auto* uv2s = geo->getAttribute<float>("uv2");
        auto* cols = geo->getAttribute<float>("color");
        auto* idx = geo->getIndex();

        auto vert = [&](int i) {
            Vector3 v(pos->getX(i), pos->getY(i), pos->getZ(i));
            v.applyMatrix4(world);
            return v;
        };
        auto norm = [&](int i) -> Vector3 {
            if (!nrm) return {0.f, 1.f, 0.f};
            Vector3 n(nrm->getX(i), nrm->getY(i), nrm->getZ(i));
            n.transformDirection(normalMat);
            return n;
        };
        auto objVert = [&](int i) -> Vector3 {
            return {pos->getX(i), pos->getY(i), pos->getZ(i)};
        };
        auto objNorm = [&](int i) -> Vector3 {
            if (!nrm) return {0.f, 1.f, 0.f};
            return {nrm->getX(i), nrm->getY(i), nrm->getZ(i)};
        };
        auto uv = [&](int i) -> std::pair<float, float> {
            if (!uvs) return {0.f, 0.f};
            return {uvs->getX(i), uvs->getY(i)};
        };
        auto uv1 = [&](int i) -> std::pair<float, float> {
            if (!uv2s) return {0.f, 0.f};
            return {uv2s->getX(i), uv2s->getY(i)};
        };
        auto vcol = [&](int i) -> std::tuple<float, float, float> {
            if (!cols) return {1.f, 1.f, 1.f};
            return {cols->getX(i), cols->getY(i), cols->getZ(i)};
        };
        auto vi = [&](int tri, int corner) -> int {
            return idx ? static_cast<int>(idx->getX(tri * 3 + corner)) : tri * 3 + corner;
        };

        const int nTris = idx ? static_cast<int>(idx->count()) / 3
                              : static_cast<int>(pos->count()) / 3;
        for (int i = 0; i < nTris && triCount < maxTris; ++i) {
            const int i0 = vi(i, 0), i1 = vi(i, 1), i2 = vi(i, 2);
            const Vector3 v0 = vert(i0), v1 = vert(i1), v2 = vert(i2);
            const Vector3 n0 = norm(i0), n1 = norm(i1), n2 = norm(i2);
            const auto [u0, v0uv] = uv(i0);
            const auto [u1, v1uv] = uv(i1);
            const auto [u2, v2uv] = uv(i2);
            const auto [u1_0, v1_0] = uv1(i0);
            const auto [u1_1, v1_1] = uv1(i1);
            const auto [u1_2, v1_2] = uv1(i2);
            const auto [cr0, cg0, cb0] = vcol(i0);
            const auto [cr1, cg1, cb1] = vcol(i1);
            const auto [cr2, cg2, cb2] = vcol(i2);

            // Use paged layout (matches triGet/pagedIdx/GPU triCoord)
            auto setTri = [&](int row, float x, float y, float z, float w) {
                int idx = pagedIdx(triCount, row);
                triBuffer[idx + 0] = x; triBuffer[idx + 1] = y;
                triBuffer[idx + 2] = z; triBuffer[idx + 3] = w;
            };
            setTri(0, v0.x, v0.y, v0.z, static_cast<float>(matIdx));
            setTri(1, v1.x, v1.y, v1.z, 0.f);
            setTri(2, v2.x, v2.y, v2.z, 0.f);
            setTri(3, n0.x, n0.y, n0.z, 0.f);
            setTri(4, n1.x, n1.y, n1.z, 0.f);
            setTri(5, n2.x, n2.y, n2.z, 0.f);
            setTri(6, u0, v0uv, u1, v1uv);
            setTri(7, u2, v2uv, 0.f, cb2);
            setTri(8, u1_0, v1_0, u1_1, v1_1);
            setTri(9, u1_2, v1_2, 0.f, 0.f);
            setTri(10, cr0, cg0, cb0, cr1);
            setTri(11, cg1, cb1, cr2, cg2);

            const Vector3 ov0 = objVert(i0), ov1 = objVert(i1), ov2 = objVert(i2);
            const Vector3 on0 = objNorm(i0), on1 = objNorm(i1), on2 = objNorm(i2);
            setObj(triCount, 0, ov0.x, ov0.y, ov0.z, static_cast<float>(matIdx));
            setObj(triCount, 1, ov1.x, ov1.y, ov1.z, static_cast<float>(meshIdx));
            setObj(triCount, 2, ov2.x, ov2.y, ov2.z, 0.f);
            setObj(triCount, 3, on0.x, on0.y, on0.z, 0.f);
            setObj(triCount, 4, on1.x, on1.y, on1.z, 0.f);
            setObj(triCount, 5, on2.x, on2.y, on2.z, 0.f);
            setObj(triCount, 6, u0, v0uv, u1, v1uv);
            setObj(triCount, 7, u2, v2uv, 0.f, cb2);
            setObj(triCount, 8, u1_0, v1_0, u1_1, v1_1);
            setObj(triCount, 9, u1_2, v1_2, 0.f, 0.f);
            setObj(triCount, 10, cr0, cg0, cb0, cr1);
            setObj(triCount, 11, cg1, cb1, cr2, cg2);

            ++triCount;
        }
    }
    return triCount;
}

// ---------------------------------------------------------------------------
// BVH builder
// ---------------------------------------------------------------------------

static float triGet(const std::vector<float>& buf, int ti, int row, int comp) {
    const int page = ti / TEX_PAGE_WIDTH;
    const int pcol = ti % TEX_PAGE_WIDTH;
    return buf[((page * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + pcol) * 4 + comp];
}

static float boxSurfaceArea(float mnX, float mnY, float mnZ,
                            float mxX, float mxY, float mxZ) {
    const float dx = mxX - mnX, dy = mxY - mnY, dz = mxZ - mnZ;
    return 2.f * (dx * dy + dy * dz + dz * dx);
}

// Compute centroid of triangle ti along a given axis (0=X, 1=Y, 2=Z).
static float triCentroid(const std::vector<float>& buf, int ti, int axis) {
    return (triGet(buf, ti, 0, axis) + triGet(buf, ti, 1, axis) + triGet(buf, ti, 2, axis)) / 3.f;
}

static int buildBvhNode(
        std::vector<BvhNode>& nodes,
        std::vector<int>& idx,
        const std::vector<float>& buf,
        int start, int end, int parentIdx = -1) {
    const int ni = static_cast<int>(nodes.size());
    nodes.emplace_back();

    // Compute full AABB of all triangles in [start, end)
    float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
    float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
    for (int i = start; i < end; i++) {
        const int ti = idx[i];
        for (int r = 0; r <= 2; r++) {
            minX = std::min(minX, triGet(buf, ti, r, 0));
            minY = std::min(minY, triGet(buf, ti, r, 1));
            minZ = std::min(minZ, triGet(buf, ti, r, 2));
            maxX = std::max(maxX, triGet(buf, ti, r, 0));
            maxY = std::max(maxY, triGet(buf, ti, r, 1));
            maxZ = std::max(maxZ, triGet(buf, ti, r, 2));
        }
    }
    nodes[ni] = {minX, minY, minZ, 0, maxX, maxY, maxZ, 0, parentIdx, {0, 0, 0}};

    const int count = end - start;
    if (count <= 2) {
        nodes[ni].left = -(start + 1);
        nodes[ni].right = count;
        return ni;
    }

    // Compute centroid AABB for tighter bucket distribution
    float cMinX = 1e30f, cMinY = 1e30f, cMinZ = 1e30f;
    float cMaxX = -1e30f, cMaxY = -1e30f, cMaxZ = -1e30f;
    for (int i = start; i < end; i++) {
        const int ti = idx[i];
        const float cx = triCentroid(buf, ti, 0);
        const float cy = triCentroid(buf, ti, 1);
        const float cz = triCentroid(buf, ti, 2);
        cMinX = std::min(cMinX, cx); cMaxX = std::max(cMaxX, cx);
        cMinY = std::min(cMinY, cy); cMaxY = std::max(cMaxY, cy);
        cMinZ = std::min(cMinZ, cz); cMaxZ = std::max(cMaxZ, cz);
    }

    constexpr int NB = 32;
    constexpr float C_trav = 1.0f;  // traversal cost relative to intersection
    struct Bucket {
        float mnX = 1e30f, mnY = 1e30f, mnZ = 1e30f;
        float mxX = -1e30f, mxY = -1e30f, mxZ = -1e30f;
        int cnt = 0;
    };

    float bestCost = 1e30f;
    int bestAxis = -1;
    int bestSplit = NB / 2;

    const float nodeArea = boxSurfaceArea(minX, minY, minZ, maxX, maxY, maxZ);
    const float leafCost = static_cast<float>(count);  // cost of making this a leaf

    const float centMin[3] = {cMinX, cMinY, cMinZ};
    const float centMax[3] = {cMaxX, cMaxY, cMaxZ};

    for (int axis = 0; axis < 3; axis++) {
        if (centMax[axis] - centMin[axis] < 1e-6f) continue;
        const float scale = NB / (centMax[axis] - centMin[axis]);

        Bucket buckets[NB];
        for (int i = start; i < end; i++) {
            const int ti = idx[i];
            const float c = triCentroid(buf, ti, axis);
            const int bi = std::clamp(static_cast<int>((c - centMin[axis]) * scale), 0, NB - 1);
            for (int r = 0; r <= 2; r++) {
                buckets[bi].mnX = std::min(buckets[bi].mnX, triGet(buf, ti, r, 0));
                buckets[bi].mnY = std::min(buckets[bi].mnY, triGet(buf, ti, r, 1));
                buckets[bi].mnZ = std::min(buckets[bi].mnZ, triGet(buf, ti, r, 2));
                buckets[bi].mxX = std::max(buckets[bi].mxX, triGet(buf, ti, r, 0));
                buckets[bi].mxY = std::max(buckets[bi].mxY, triGet(buf, ti, r, 1));
                buckets[bi].mxZ = std::max(buckets[bi].mxZ, triGet(buf, ti, r, 2));
            }
            buckets[bi].cnt++;
        }

        // Forward prefix scan: prefixArea[s] and prefixCnt[s] cover buckets [0..s]
        float prefixArea[NB];
        int prefixCnt[NB];
        {
            float pmnX = 1e30f, pmnY = 1e30f, pmnZ = 1e30f;
            float pmxX = -1e30f, pmxY = -1e30f, pmxZ = -1e30f;
            int pcnt = 0;
            for (int b = 0; b < NB; b++) {
                if (buckets[b].cnt) {
                    pmnX = std::min(pmnX, buckets[b].mnX);
                    pmnY = std::min(pmnY, buckets[b].mnY);
                    pmnZ = std::min(pmnZ, buckets[b].mnZ);
                    pmxX = std::max(pmxX, buckets[b].mxX);
                    pmxY = std::max(pmxY, buckets[b].mxY);
                    pmxZ = std::max(pmxZ, buckets[b].mxZ);
                    pcnt += buckets[b].cnt;
                }
                prefixArea[b] = pcnt > 0 ? boxSurfaceArea(pmnX, pmnY, pmnZ, pmxX, pmxY, pmxZ) : 0.f;
                prefixCnt[b] = pcnt;
            }
        }

        // Backward prefix scan: suffixArea[s] and suffixCnt[s] cover buckets [s..NB-1]
        float suffixArea[NB];
        int suffixCnt[NB];
        {
            float smnX = 1e30f, smnY = 1e30f, smnZ = 1e30f;
            float smxX = -1e30f, smxY = -1e30f, smxZ = -1e30f;
            int scnt = 0;
            for (int b = NB - 1; b >= 0; b--) {
                if (buckets[b].cnt) {
                    smnX = std::min(smnX, buckets[b].mnX);
                    smnY = std::min(smnY, buckets[b].mnY);
                    smnZ = std::min(smnZ, buckets[b].mnZ);
                    smxX = std::max(smxX, buckets[b].mxX);
                    smxY = std::max(smxY, buckets[b].mxY);
                    smxZ = std::max(smxZ, buckets[b].mxZ);
                    scnt += buckets[b].cnt;
                }
                suffixArea[b] = scnt > 0 ? boxSurfaceArea(smnX, smnY, smnZ, smxX, smxY, smxZ) : 0.f;
                suffixCnt[b] = scnt;
            }
        }

        // Evaluate split between bucket s-1 and s (left = [0..s-1], right = [s..NB-1])
        for (int s = 1; s < NB; s++) {
            const int lcnt = prefixCnt[s - 1];
            const int rcnt = suffixCnt[s];
            if (!lcnt || !rcnt) continue;

            const float cost = C_trav + (static_cast<float>(lcnt) * prefixArea[s - 1]
                             + static_cast<float>(rcnt) * suffixArea[s]) / nodeArea;
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestSplit = s;
            }
        }
    }

    // If no good split found, or splitting is more expensive than a leaf, make a leaf.
    // But never exceed 8 tris — the BVH4 leaf encoding can't hold more.
    if (count <= 8 && (bestAxis < 0 || bestCost >= leafCost)) {
        nodes[ni].left = -(start + 1);
        nodes[ni].right = count;
        return ni;
    }

    // Force split even if SAH says leaf is cheaper — pick longest axis, median split
    if (bestAxis < 0) {
        const float extents[3] = {maxX - minX, maxY - minY, maxZ - minZ};
        bestAxis = (extents[0] >= extents[1] && extents[0] >= extents[2]) ? 0
                 : (extents[1] >= extents[2]) ? 1 : 2;
    }

    const float splitPos = centMin[bestAxis]
        + (centMax[bestAxis] - centMin[bestAxis]) * static_cast<float>(bestSplit) / NB;

    auto mid = std::partition(idx.begin() + start, idx.begin() + end, [&](int ti) {
        return triCentroid(buf, ti, bestAxis) < splitPos;
    });
    int sp = static_cast<int>(mid - idx.begin());
    if (sp == start || sp == end) sp = (start + end) / 2;

    const int lc = buildBvhNode(nodes, idx, buf, start, sp, ni);
    const int rc = buildBvhNode(nodes, idx, buf, sp, end, ni);
    nodes[ni].left = lc;
    nodes[ni].right = rc;
    return ni;
}

// ---------------------------------------------------------------------------
// Wide BVH (BVH4): collapse binary BVH into 4-way tree
// ---------------------------------------------------------------------------

// BVH4 node: up to 4 children, stored in SoA layout for SIMD AABB testing
// Leaf encoding: childIdx = -(triStart * MAX_LEAF_TRIS + triCount), where triCount is 1-based.
// Decode: raw = -childIdx; triStart = (raw - 1) / MAX_LEAF_TRIS; triCount = ((raw - 1) % MAX_LEAF_TRIS) + 1
static constexpr int MAX_LEAF_TRIS = 8;

struct Bvh4Node {
    float childMinX[4], childMinY[4], childMinZ[4];
    float childMaxX[4], childMaxY[4], childMaxZ[4];
    int   childIdx[4];  // >= 0: internal node index, < 0: leaf (encoded triStart+triCount)
    int   childCount;        // 1..4 valid children
    int   numInternalChildren; // count of children with childIdx >= 0
    int   parent;
};

/// Collapse a binary BVH into a BVH4 tree.
/// Walk the binary tree and greedily expand internal children until each node
/// has up to 4 children (which may be leaves or further internal BVH4 nodes).
static void collapseBvh4(
        const std::vector<BvhNode>& bin,
        std::vector<Bvh4Node>& wide,
        std::vector<int>& leafIndicesOut,
        int binNodeIdx, int parentWide) {

    const int wi = static_cast<int>(wide.size());
    wide.emplace_back();

    // Collect children by expanding shallowest internal nodes first.
    // Start with the two binary children of binNodeIdx.
    struct Candidate {
        int binIdx;
        float area;
    };
    std::vector<Candidate> children;
    auto addChild = [&](int bi) {
        const auto& n = bin[bi];
        float a = boxSurfaceArea(n.minX, n.minY, n.minZ, n.maxX, n.maxY, n.maxZ);
        children.push_back({bi, a});
    };

    // Helper: encode a binary leaf (triStart, triCount) into a single negative int
    auto encodeLeaf = [](int binLeft, int binRight) -> int {
        const int triStart = -(binLeft + 1);
        const int triCount = binRight;
        return -(triStart * MAX_LEAF_TRIS + triCount);
    };

    const auto& root = bin[binNodeIdx];
    if (root.left < 0) {
        // The binary root itself is a leaf — make a BVH4 leaf wrapper
        Bvh4Node& w = wide[wi];
        w.childCount = 1;
        w.childIdx[0] = encodeLeaf(root.left, root.right);
        w.childMinX[0] = root.minX; w.childMinY[0] = root.minY; w.childMinZ[0] = root.minZ;
        w.childMaxX[0] = root.maxX; w.childMaxY[0] = root.maxY; w.childMaxZ[0] = root.maxZ;
        for (int c = 1; c < 4; c++) {
            w.childIdx[c] = INT_MIN;
            w.childMinX[c] = 1e30f; w.childMinY[c] = 1e30f; w.childMinZ[c] = 1e30f;
            w.childMaxX[c] = -1e30f; w.childMaxY[c] = -1e30f; w.childMaxZ[c] = -1e30f;
        }
        w.parent = parentWide;
        leafIndicesOut.push_back(wi);
        return;
    }

    addChild(root.left);
    addChild(root.right);

    // Greedily expand the largest-area internal child until we have 4 or all are leaves
    while (static_cast<int>(children.size()) < 4) {
        int bestIdx = -1;
        float bestArea = -1.f;
        for (int i = 0; i < static_cast<int>(children.size()); i++) {
            const auto& bn = bin[children[i].binIdx];
            if (bn.left >= 0 && children[i].area > bestArea) {  // is internal
                bestIdx = i;
                bestArea = children[i].area;
            }
        }
        if (bestIdx < 0) break;  // all children are leaves

        // Expand: replace the internal child with its two children
        int expandBin = children[bestIdx].binIdx;
        children.erase(children.begin() + bestIdx);
        const auto& expanded = bin[expandBin];
        // Insert in same position to preserve order
        auto addChildAt = [&](int bi, int pos) {
            const auto& n = bin[bi];
            float a = boxSurfaceArea(n.minX, n.minY, n.minZ, n.maxX, n.maxY, n.maxZ);
            children.insert(children.begin() + pos, {bi, a});
        };
        addChildAt(expanded.left, bestIdx);
        addChildAt(expanded.right, bestIdx + 1);
    }

    // Fill the BVH4 node
    Bvh4Node& w = wide[wi];
    w.childCount = static_cast<int>(children.size());
    w.numInternalChildren = 0;
    w.parent = parentWide;
    bool hasLeafChild = false;

    for (int c = 0; c < 4; c++) {
        if (c < w.childCount) {
            const auto& bn = bin[children[c].binIdx];
            w.childMinX[c] = bn.minX; w.childMinY[c] = bn.minY; w.childMinZ[c] = bn.minZ;
            w.childMaxX[c] = bn.maxX; w.childMaxY[c] = bn.maxY; w.childMaxZ[c] = bn.maxZ;
            if (bn.left < 0) {
                // Leaf child — encode triStart+triCount into single int
                w.childIdx[c] = encodeLeaf(bn.left, bn.right);
                hasLeafChild = true;
            } else {
                w.childIdx[c] = 0;  // placeholder, will be patched
                w.numInternalChildren++;
            }
        } else {
            // Empty slot — INT_MIN sentinel, skipped by WGSL EMPTY_CHILD check
            w.childIdx[c] = INT_MIN;
            w.childMinX[c] = 1e30f; w.childMinY[c] = 1e30f; w.childMinZ[c] = 1e30f;
            w.childMaxX[c] = -1e30f; w.childMaxY[c] = -1e30f; w.childMaxZ[c] = -1e30f;
        }
    }

    if (hasLeafChild) leafIndicesOut.push_back(wi);

    // Recurse on internal children and patch their indices
    for (int c = 0; c < w.childCount; c++) {
        const auto& bn = bin[children[c].binIdx];
        if (bn.left >= 0) {
            int childWideIdx = static_cast<int>(wide.size());
            wide[wi].childIdx[c] = childWideIdx;
            collapseBvh4(bin, wide, leafIndicesOut, children[c].binIdx, wi);
        }
    }
}

/// Pack BVH4 nodes into a flat GPU buffer (7 × vec4 = 28 floats per node).
///
/// Layout per node:
///   row0: childMinX[0..3]
///   row1: childMinY[0..3]
///   row2: childMinZ[0..3]
///   row3: childMaxX[0..3]
///   row4: childMaxY[0..3]
///   row5: childMaxZ[0..3]
///   row6: childIdx[0..3]  (i32 via bitcast: >=0 internal, <0 leaf packed triStart*8+triCount)
///
/// Refit metadata is stored in a separate buffer (4 ints per node):
///   (parent, childCount, numInternalChildren, 0)
static constexpr int BVH4_GPU_U32S = 28;  // 112 bytes per node (f32 AABBs: 6*vec4 + cIdx)
static constexpr int BVH4_REFIT_INTS = 4;  // per-node refit metadata

// Convert f32 to IEEE 754 half-precision (f16), returned as uint16_t.
// roundUp: true = round toward +∞ (for AABB max), false = round toward -∞ (for AABB min).
static uint16_t f32ToF16(float value, bool roundUp = false) {
    uint32_t f;
    std::memcpy(&f, &value, sizeof(f));
    const uint32_t sign = (f >> 16) & 0x8000u;
    const bool     neg  = (sign != 0);
    const int32_t  exp  = static_cast<int32_t>((f >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = f & 0x007FFFFFu;
    if (exp <= 0) {
        // Subnormal/zero: if rounding away from zero, return smallest representable
        if (roundUp && !neg && value > 0.f) return 0x0001u;          // +smallest subnormal
        if (!roundUp && neg && value < 0.f) return static_cast<uint16_t>(0x8001u); // -smallest subnormal
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); // infinity / overflow
    uint16_t h = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
    // If truncated bits are nonzero and we need to round away from zero, bump the result
    bool needsBump = (roundUp && !neg) || (!roundUp && neg);
    if (needsBump && (mant & 0x1FFFu) != 0) {
        h++;  // safe: won't overflow past inf (0x7C00) in practice for AABB values
    }
    return h;
}

// Pack two f32 values into a single u32 as two f16 halves (matches WGSL unpack2x16float).
static uint32_t packF16x2(float a, float b) {
    return uint32_t(f32ToF16(a)) | (uint32_t(f32ToF16(b)) << 16);
}
// Conservative packing for AABBs: min rounds down, max rounds up.
static uint32_t packF16x2Min(float a, float b) {
    return uint32_t(f32ToF16(a, false)) | (uint32_t(f32ToF16(b, false)) << 16);
}
static uint32_t packF16x2Max(float a, float b) {
    return uint32_t(f32ToF16(a, true)) | (uint32_t(f32ToF16(b, true)) << 16);
}

// Compute f16 ULP-based epsilon: expand AABB so thin boxes survive f16 quantization.
// At value v, f16 precision is ~|v|/1024. We pad by 2 ULPs + a small absolute floor.
static float f16Eps(float v) {
    return std::max(std::abs(v) * (2.f / 1024.f), 1e-4f);
}

static void packBvh4Buffer(const std::vector<Bvh4Node>& nodes, std::vector<uint32_t>& buf, int capacity) {
    std::ranges::fill(buf, 0u);
    const int nc = std::min(static_cast<int>(nodes.size()), capacity);
    for (int i = 0; i < nc; i++) {
        const auto& n = nodes[i];
        float* p = reinterpret_cast<float*>(buf.data() + static_cast<size_t>(i) * BVH4_GPU_U32S);

        // f32 SoA layout with small epsilon expansion to cover CPU/GPU float discrepancy.
        // BVH AABBs are built from CPU-transformed vertices (applyMatrix4), but the GPU
        // VT pass recomputes world positions via mat*vec4 which can differ by a few ULPs.
        constexpr float E = 1e-5f;
        for (int c = 0; c < 4; c++) p[0 + c]  = n.childMinX[c] - E;
        for (int c = 0; c < 4; c++) p[4 + c]  = n.childMinY[c] - E;
        for (int c = 0; c < 4; c++) p[8 + c]  = n.childMinZ[c] - E;
        for (int c = 0; c < 4; c++) p[12 + c] = n.childMaxX[c] + E;
        for (int c = 0; c < 4; c++) p[16 + c] = n.childMaxY[c] + E;
        for (int c = 0; c < 4; c++) p[20 + c] = n.childMaxZ[c] + E;
        // cIdx: child indices as bitcast u32
        uint32_t* pi = buf.data() + static_cast<size_t>(i) * BVH4_GPU_U32S;
        for (int c = 0; c < 4; c++) std::memcpy(pi + 24 + c, &n.childIdx[c], sizeof(int));
    }
}

static void packRefitMetadata(const std::vector<Bvh4Node>& nodes, std::vector<int32_t>& buf, int capacity) {
    std::ranges::fill(buf, 0);
    const int nc = std::min(static_cast<int>(nodes.size()), capacity);
    for (int i = 0; i < nc; i++) {
        const auto& n = nodes[i];
        int32_t* p = buf.data() + static_cast<size_t>(i) * BVH4_REFIT_INTS;
        p[0] = n.parent;
        p[1] = n.childCount;
        p[2] = n.numInternalChildren;
        p[3] = 0;
    }
}

static void buildBVH(std::vector<float>& triBuffer, int triCount,
                     std::vector<Bvh4Node>& wideNodes, std::vector<int>& indices,
                     std::vector<int>& leafIndices,
                     std::vector<float>& rawObjTriBuf) {
    indices.resize(triCount);
    std::iota(indices.begin(), indices.end(), 0);

    // Phase 1: build binary BVH
    std::vector<BvhNode> binNodes;
    binNodes.reserve(triCount * 2);
    buildBvhNode(binNodes, indices, triBuffer, 0, triCount, -1);

    // Phase 2: collapse binary → BVH4
    wideNodes.clear();
    wideNodes.reserve(binNodes.size() / 2);  // roughly half as many nodes
    leafIndices.clear();
    if (!binNodes.empty()) {
        collapseBvh4(binNodes, wideNodes, leafIndices, 0, -1);
    }

    // Sort triangle data to match BVH index ordering
    std::vector<float> sorted(triBuffer.size(), 0.f);
    std::vector<float> sortedObj(rawObjTriBuf.size(), 0.f);
    for (int ni = 0; ni < triCount; ni++) {
        const int oi = indices[ni];
        for (int row = 0; row < TRI_TEX_HEIGHT; row++)
            for (int c = 0; c < 4; c++)
                sorted[pagedIdx(ni, row) + c] = triBuffer[pagedIdx(oi, row) + c];
        std::memcpy(sortedObj.data() + ni * 48, rawObjTriBuf.data() + oi * 48, 48 * sizeof(float));
    }
    triBuffer = std::move(sorted);
    rawObjTriBuf = std::move(sortedObj);
}

}// anonymous namespace

// ---------------------------------------------------------------------------
// WGSL temporal upscale shader — runs at full resolution, accumulates low-res
// denoised frames into a full-res history buffer via reprojection + EMA.
// ---------------------------------------------------------------------------
constexpr const char* upscaleWGSL = R"(
struct UpscaleUniforms {
    prevCamOri: vec4<f32>,
    prevCamFwd: vec4<f32>,
    prevCamRgt: vec4<f32>,
    prevCamUp:  vec4<f32>,
    curCamOri:  vec4<f32>,
    curCamFwd:  vec4<f32>,
    curCamRgt:  vec4<f32>,
    curCamUp:   vec4<f32>,
    iRes:       vec4<f32>,   // [0]=fullW [1]=fullH [2]=pixelScale [3]=0
    tanHalfFov: vec4<f32>,
    frameCount: vec4<f32>,
};
@group(0) @binding(0) var<uniform> up: UpscaleUniforms;
@group(0) @binding(1) var denoisedDiff: texture_2d<f32>;
@group(0) @binding(2) var denoisedSpec: texture_2d<f32>;
@group(0) @binding(3) var gBufCurLow: texture_2d<f32>;  // normal.xyz + rayDist.w
@group(0) @binding(4) var historyIn:  texture_2d<f32>;  // previous full-res output
@group(0) @binding(5) var historyOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn upscale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let fullPx  = vec2<i32>(i32(gid.x), i32(gid.y));
    let fullW   = up.iRes.x;
    let fullH   = up.iRes.y;
    if (f32(fullPx.x) >= fullW || f32(fullPx.y) >= fullH) { return; }

    let pixScale   = up.iRes.z;
    let lowResSize = vec2<i32>(textureDimensions(denoisedDiff));

    // Map full-res pixel to low-res accum pixel
    let lowResPx = clamp(
        vec2<i32>(vec2<f32>(fullPx) * pixScale),
        vec2<i32>(0),
        lowResSize - vec2<i32>(1)
    );

    // Current denoised color at low-res pixel
    let curColor = textureLoad(denoisedDiff, lowResPx, 0).xyz
                 + textureLoad(denoisedSpec, lowResPx, 0).xyz;

    // G-buffer: normal.xyz + ray distance.w
    let gBuf     = textureLoad(gBufCurLow, lowResPx, 0);
    let curDepth = gBuf.w;

    // Sky pixel — no temporal history possible
    if (curDepth < 1e-6) {
        textureStore(historyOut, fullPx, vec4<f32>(curColor, -1.0));
        return;
    }

    // Reconstruct world position from low-res pixel-center ray + ray distance
    let aspect  = fullW / fullH;
    let tanHfov = up.tanHalfFov.x;
    let lowNdc  = vec2<f32>(
        (f32(lowResPx.x) + 0.5) / f32(lowResSize.x) * 2.0 - 1.0,
        1.0 - (f32(lowResPx.y) + 0.5) / f32(lowResSize.y) * 2.0
    );
    let rayDir  = normalize(up.curCamFwd.xyz
                          + up.curCamRgt.xyz * (lowNdc.x * tanHfov * aspect)
                          + up.curCamUp.xyz  * (lowNdc.y * tanHfov));
    let worldPos = up.curCamOri.xyz + rayDir * curDepth;

    // Reproject world position to previous frame's full-res screen
    let relP  = worldPos - up.prevCamOri.xyz;
    let prevZ = dot(relP, up.prevCamFwd.xyz);
    if (prevZ <= 0.001) {
        textureStore(historyOut, fullPx, vec4<f32>(curColor, 1.0));
        return;
    }

    let prevNdcX = dot(relP, up.prevCamRgt.xyz) / (prevZ * tanHfov * aspect);
    let prevNdcY = dot(relP, up.prevCamUp.xyz)  / (prevZ * tanHfov);
    let prevU    = (prevNdcX + 1.0) * 0.5 * fullW - 0.5;
    let prevV    = (1.0 - prevNdcY) * 0.5 * fullH - 0.5;
    let prevFullPx = vec2<i32>(i32(floor(prevU)), i32(floor(prevV)));

    if (prevFullPx.x < 0 || prevFullPx.y < 0 ||
        prevFullPx.x >= i32(fullW) || prevFullPx.y >= i32(fullH)) {
        textureStore(historyOut, fullPx, vec4<f32>(curColor, 1.0));
        return;
    }

    // Fetch full-res history (.w = histLen; -1 = sky sentinel; 0 = fresh/reset)
    let histSamp  = textureLoad(historyIn, prevFullPx, 0);
    let histColor = histSamp.xyz;
    let histLen   = histSamp.w;

    // Disocclusion: only reject sky-sentinel history (depth cross-frame comparison
    // is unreliable for moving cameras — let short max-history handle ghosting).
    var result     = curColor;
    var newHistLen = 1.0;
    if (histLen >= 0.0) {
        newHistLen = min(histLen + 1.0, 32.0);
        let alpha  = max(1.0 / 16.0, 1.0 / newHistLen);
        result = mix(histColor, curColor, alpha);
    }

    textureStore(historyOut, fullPx, vec4<f32>(result, newHistLen));
}
)";

// ---------------------------------------------------------------------------
// WgpuPathTracer::Impl
// ---------------------------------------------------------------------------
struct WgpuPathTracer::Impl {
    WgpuRenderer& renderer;
    WGPUDevice device;
    WGPUQueue queue;

    // GPU textures
    WgpuTexture triTex;
    WgpuTexture matTex;
    WgpuTexture texAtlasTex;
    int atlasLayers_ = 0;  // current atlas row count (0 = initial placeholder)
    int atlasCols_ = ATLAS_WIDTH / DEFAULT_TILE_SIZE;
    int tileSize_ = DEFAULT_TILE_SIZE;
    int textureResolution_ = DEFAULT_TILE_SIZE;  // user config: 1024 or 2048
    WgpuTexture accumA;
    WgpuTexture accumB;
    WgpuTexture* readAccum;
    WgpuTexture* writeAccum;
    WgpuTexture hitMeshA;
    WgpuTexture hitMeshB;
    WgpuTexture* readHitMesh;
    WgpuTexture* writeHitMesh;
    WgpuTexture envTexGpu;   // equirectangular env map for IBL (1x1 placeholder when unused)
    WgpuTexture bgTexGpu;   // equirectangular background for ray misses (1x1 placeholder when unused)
    WgpuTexture envCdfTex;  // conditional CDF (width × height), R32Float
    WgpuTexture envMargTex; // marginal CDF (height × 1), R32Float
    float       envLumSum_ = 0.f;  // total luminance sum for PDF normalization
    Texture*    prevEnvTex_ = nullptr;
    Texture*    prevBgTex_  = nullptr;

    // G-buffer ping-pong
    WgpuTexture gBufA;          // normal.xyz + depth.w
    WgpuTexture gBufB;
    WgpuTexture* gBufCur;       // current frame writes here
    WgpuTexture* gBufPrev;      // previous frame's G-buffer

    WgpuTexture stableGBufA;   // r32float — jitter-corrected depth for à-trous
    WgpuTexture stableGBufB;
    WgpuTexture* stableGBufCur;
    WgpuTexture* stableGBufPrev;

    // ReSTIR DI reservoir ping-pong
    WgpuTexture reservoirA;     // rgba32float — lightPos.xyz + encoded type/index
    WgpuTexture reservoirB;
    WgpuTexture reservoirWA;    // rgba32float — W_sum, M, W, p_hat
    WgpuTexture reservoirWB;
    WgpuTexture* reservoirRead;
    WgpuTexture* reservoirWrite;
    WgpuTexture* reservoirWRead;
    WgpuTexture* reservoirWWrite;

    // ReSTIR GI reservoir ping-pong
    WgpuTexture giResA;       // rgba32float — secHitPos.xyz + octahedral-packed normal
    WgpuTexture giResB;
    WgpuTexture giResWA;      // rgba32float — W_sum, M, W, p_hat
    WgpuTexture giResWB;
    WgpuTexture giResLoA;     // rgba16float — Lo radiance at secondary hit
    WgpuTexture giResLoB;
    WgpuTexture* giResRead;
    WgpuTexture* giResWrite;
    WgpuTexture* giResWRead;
    WgpuTexture* giResWWrite;
    WgpuTexture* giResLoRead;
    WgpuTexture* giResLoWrite;

    // Albedo buffer (primary-hit albedo for demodulation/remodulation)
    WgpuTexture albedoTex;

    // Temporal variance moments ping-pong (μ, μ² of luminance)
    WgpuTexture momentsA;
    WgpuTexture momentsB;
    WgpuTexture* momentsRead;
    WgpuTexture* momentsWrite;

    // TAA history ping-pong
    WgpuTexture taaHistA;       // previous TAA output
    WgpuTexture taaHistB;
    WgpuTexture* taaHistRead;
    WgpuTexture* taaHistWrite;

    // Diffuse/specular split accumulation
    WgpuTexture diffAccumA;
    WgpuTexture diffAccumB;
    WgpuTexture* readDiffAccum;
    WgpuTexture* writeDiffAccum;
    WgpuTexture specAccumA;
    WgpuTexture specAccumB;
    WgpuTexture* readSpecAccum;
    WgpuTexture* writeSpecAccum;

    // Spatial filter ping-pong (shared between diffuse and specular passes)
    WgpuTexture filteredA;
    WgpuTexture filteredB;
    // Denoised output textures for display
    WgpuTexture denoisedDiff;
    WgpuTexture denoisedSpec;

    // TAA history for split channels
    WgpuTexture taaHistDiffA;
    WgpuTexture taaHistDiffB;
    WgpuTexture* taaHistDiffRead;
    WgpuTexture* taaHistDiffWrite;
    WgpuTexture taaHistSpecA;
    WgpuTexture taaHistSpecB;
    WgpuTexture* taaHistSpecRead;
    WgpuTexture* taaHistSpecWrite;

    // Temporal upscale (TAAU) — full-res ping-pong, active when pixelScale < 0.85
    WgpuTexture upscaleTexA;
    WgpuTexture upscaleTexB;
    WgpuTexture zeroTex;           // 1×1 dummy; bound when upscale inactive
    WgpuTexture* upscaleRead;
    WgpuTexture* upscaleWrite;
    WgpuComputePipeline upscalePipeline;
    WgpuBuffer          upscaleUniBuf;
    UpscaleGpuUniforms  upscaleUBO{};

    // Denoiser pipelines
    WgpuComputePipeline taaPipeline;
    WgpuComputePipeline atrousPipeline;
    WgpuComputePipeline preFilterPipeline;
    WgpuBuffer  taaUniBuf;
    WgpuBuffer  atrousUniBuf;
    WgpuBuffer  preFilterUniBuf;
    bool denoiserEnabled_ = true;
    bool temporalDenoiserEnabled_ = false;
    bool restirEnabled_ = true;
    bool restirGiEnabled_ = false;
    int spp_ = 1;
    float envIntensity_ = 0.5f;
    int maxBounces_ = 4;
    float exposure_ = 1.0f;
    // Per-contribution firefly clamp (luminance cap) on indirect MIS paths.
    // Default 8.0 matches production renderers (Arnold/Cycles/RenderMan).
    // Set to a very large value (e.g. 1e30f) to disable clipping when
    // unbiased HDR output is required (ML training data, light-transport
    // validation). Primary-ray emissive hits are never clamped.
    float fireflyCap_ = 8.0f;
    int aovMode_ = 0;  // 0=off, 1=depth, 2=normals, 3=albedo, 4=instanceId, 5=roughness

    // Dynamic capacity tracking — buffers grow as scenes demand more
    int triCapacity_  = INIT_TRI_CAP;
    int matCapacity_  = INIT_MAT_CAP;
    int meshCapacity_ = INIT_MESH_CAP;
    int bvhCapacity_  = 2 * INIT_TRI_CAP - 1;
    float pixelScale_ = 1.0f;
    int fullWidth_ = 0;   // unscaled window size
    int fullHeight_ = 0;

    // Previous camera vectors for TAA reprojection
    float prevCamOri_[3] = {0.f, 0.f, 0.f};
    float prevCamFwd_[3] = {0.f, 0.f, -1.f};
    float prevCamRgt_[3] = {1.f, 0.f, 0.f};
    float prevCamUp_[3]  = {0.f, 1.f, 0.f};

    // GPU storage buffers
    WgpuBuffer bvhNodeBuf;
    WgpuBuffer bvhCounterBuf;
    WgpuBuffer refitMetaBuf;
    WgpuBuffer objTriBuf;
    WgpuBuffer objTriBuf2;  // overflow buffer for large scenes
    int objTriSplit_ = 0;   // split point: tris [0, split) in buf1, [split, count) in buf2
    WgpuBuffer matrixBuf;
    WgpuBuffer motionMatBuf;            // per-mesh motion matrices for TAA reprojection
    std::vector<float> motionMatCpu;    // CPU staging: prevWorld * inverse(curWorld) per mesh
    WgpuBuffer leafIndexBuf;

    // Emissive triangle NEE
    WgpuBuffer emissiveTriBuf;
    std::vector<float> emissiveTriCpu;  // packed vec4: (triIndex, area, 0, 0) per emissive tri
    int emissiveTriCount_ = 0;
    float emissiveTotalArea_ = 0.f;
    float emissiveTotalPower_ = 0.f;
    std::unordered_set<int> emissiveMeshSet_;  // mesh indices that contribute emissive light

    // GPU uniform buffers
    WgpuBuffer vtUniBuf;
    WgpuBuffer refitUniBuf;
    WgpuBuffer rtUniformBuf;

    // Compute pipelines
    WgpuComputePipeline vtPipeline;
    WgpuComputePipeline refitPipeline;
    WgpuComputePipeline rtPipeline;

    // Depth-fill pipeline — writes NDC depth from gBuffer primary-ray t values
    WGPURenderPipeline      depthFillPipeline_ = nullptr;
    WGPUPipelineLayout      depthFillPipeLayout_ = nullptr;
    WGPUBindGroupLayout     depthFillBGL_ = nullptr;
    WGPUShaderModule        depthFillShader_ = nullptr;
    WGPUBuffer              depthFillUniBuf_ = nullptr;
    uint32_t                depthFillSampleCount_ = 0;  // 0 = not yet built

    // Display pipeline
    OrthographicCamera displayCam;
    Scene displayScene;
    std::shared_ptr<ShaderMaterial> displayMat;

    // CPU staging buffers
    std::vector<float> triBuffer;
    std::vector<float> matBuffer;
    std::vector<float> rawObjTriBuf;
    std::vector<float> matrixCpuBuf;
    std::vector<uint32_t> bvhNodeCpuBuf;
    std::vector<int32_t> refitMetaCpuBuf;
    std::vector<uint32_t> bvhCounterZeros;

    // BVH state (wide BVH4)
    std::vector<Bvh4Node> bvhNodes;
    std::vector<int> bvhIndices;
    std::vector<int> leafIndices;

    // Async scene build result — CPU work done on background thread
    struct AsyncBuildResult {
        std::vector<unsigned char> atlasData;
        int atlasLayers = 0;
        int atlasCols = ATLAS_WIDTH / DEFAULT_TILE_SIZE;
        int tileSize = DEFAULT_TILE_SIZE;
        std::unordered_map<Texture*, int> texSlotMap;
        std::vector<float> triBuffer;
        std::vector<float> matBuffer;
        std::vector<float> rawObjTriBuf;
        std::vector<float> matrixCpuBuf;
        std::vector<Bvh4Node> bvhNodes;
        std::vector<int> bvhIndices;
        std::vector<int> leafIndices;
        std::vector<uint32_t> bvhNodeCpuBuf;
        std::vector<int32_t> refitMetaCpuBuf;
        std::vector<float> emissiveTriCpu;
        std::unordered_set<int> emissiveMeshSet;  // mesh indices with emissive contribution
        int triCount = 0;
        int numBvhNodes = 0;
        int emissiveTriCount = 0;
        float emissiveTotalArea = 0.f;
        float emissiveTotalPower = 0.f;
        std::vector<Mesh*> meshes;       // unique meshes (for atlas)
        std::vector<RtMeshEntry> entries; // expanded instances
        // Capacities used for buffer sizing (next-power-of-2)
        int triCapacity = 0;
        int matCapacity = 0;
        int meshCapacity = 0;
        int bvhCapacity = 0;
        int objTriSplit = 0;  // split point for two-buffer objTri scheme
    };
#ifndef __EMSCRIPTEN__
    // Async env CDF build
    std::future<EnvCdfResult> asyncEnvCdf_;
    bool envCdfPending_ = false;
#endif
    bool shaderHasEnvCdf_ = false;  // tracks which shader variant is active

    // Shader compilation wait tracking
    uint32_t shaderWaitFrames_ = 0;
    // True from when RT async build starts until the very first dispatch completes.
    // Ensures the VT pass (objTriBuf → triTex) runs on first dispatch even if no mesh moved.
    bool firstDispatchPending_ = false;

    // Frame state
    std::unordered_map<Texture*, int> texSlotMap;
    std::vector<Mesh*> prevMeshes;
    int prevEntryCount_ = 0;
    std::vector<Matrix4> prevEntryMatrices;
    int triCount_ = 0;
    int numBvhNodes_ = 0;
    uint32_t vtDispatchX_ = 1, vtDispatchY_ = 1;
    uint32_t rfDispatchX_ = 1, rfDispatchY_ = 1;
    float frameCount_ = 0.f;
    uint32_t globalFrameCounter_ = 0;
    bool foveatedEnabled_ = false;
    int foveatedConvergeFrames_ = 4;
    Vector3 prevCamPos_;
    Vector3 prevCamDir_;
    int overlayLayer_ = -1;  // -1 = disabled; objects on this layer bypass path tracing and go to raster overlay

    int width_, height_;

    int maxTriCap() const {
        WGPULimits limits{};
        wgpuDeviceGetLimits(device, &limits);
        return static_cast<int>(std::min(
            limits.maxStorageBufferBindingSize / BYTES_PER_TRI,
            uint64_t(INT_MAX)));
    }

    Impl(WgpuRenderer& r, int w, int h)
        : renderer(r),
          device(static_cast<WGPUDevice>(r.nativeDevice())),
          queue(static_cast<WGPUQueue>(r.nativeQueue())),
          // Geometry textures (small placeholders — grown dynamically on first build)
          triTex(r, TEX_PAGE_WIDTH, TRI_TEX_HEIGHT * triTexPages(INIT_TRI_CAP),
                 WgpuTexture::Format::RGBA32Float,
                 WgpuTexture::Storage | WgpuTexture::TextureBinding),
          matTex(r, INIT_MAT_CAP, MAT_TEX_HEIGHT,
                 WgpuTexture::Format::RGBA32Float,
                 WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          texAtlasTex(r, 1u, 1u,
                      WgpuTexture::Format::RGBA8Unorm,
                      WgpuTexture::Dimension::D2Array,
                      WgpuTexture::TextureBinding | WgpuTexture::CopyDst, 1u),
          // Accumulation textures
          accumA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                 WgpuTexture::Format::RGBA16Float),
          accumB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                 WgpuTexture::Format::RGBA16Float),
          readAccum(&accumA), writeAccum(&accumB),
          hitMeshA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                   WgpuTexture::Format::RGBA16Float),
          hitMeshB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                   WgpuTexture::Format::RGBA16Float),
          readHitMesh(&hitMeshA), writeHitMesh(&hitMeshB),
          envTexGpu(r, 1u, 1u, WgpuTexture::Format::RGBA8Unorm,
                    WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          bgTexGpu(r, 1u, 1u, WgpuTexture::Format::RGBA8Unorm,
                   WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          envCdfTex(r, 1u, 1u, WgpuTexture::Format::R32Float,
                    WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          envMargTex(r, 1u, 1u, WgpuTexture::Format::R32Float,
                     WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          // G-buffer ping-pong
          gBufA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                WgpuTexture::Format::RGBA16Float,
                WgpuTexture::Storage | WgpuTexture::TextureBinding | WgpuTexture::CopyDst | WgpuTexture::RenderAttachment),
          gBufB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                WgpuTexture::Format::RGBA16Float,
                WgpuTexture::Storage | WgpuTexture::TextureBinding | WgpuTexture::CopyDst | WgpuTexture::RenderAttachment),
          gBufCur(&gBufA), gBufPrev(&gBufB),
          // Stable G-buffer ping-pong (unjittered primary ray: normal.xyz + depth.w)
          stableGBufA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                       WgpuTexture::Format::RGBA16Float,
                       WgpuTexture::Storage | WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          stableGBufB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                       WgpuTexture::Format::RGBA16Float,
                       WgpuTexture::Storage | WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          stableGBufCur(&stableGBufA), stableGBufPrev(&stableGBufB),
          // ReSTIR DI reservoir ping-pong
          reservoirA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                     WgpuTexture::Format::RGBA32Float),
          reservoirB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                     WgpuTexture::Format::RGBA32Float),
          reservoirWA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                      WgpuTexture::Format::RGBA32Float),
          reservoirWB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                      WgpuTexture::Format::RGBA32Float),
          reservoirRead(&reservoirA), reservoirWrite(&reservoirB),
          reservoirWRead(&reservoirWA), reservoirWWrite(&reservoirWB),
          // ReSTIR GI reservoir ping-pong
          giResA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                 WgpuTexture::Format::RGBA32Float),
          giResB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                 WgpuTexture::Format::RGBA32Float),
          giResWA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                  WgpuTexture::Format::RGBA32Float),
          giResWB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                  WgpuTexture::Format::RGBA32Float),
          giResLoA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                   WgpuTexture::Format::RGBA16Float),
          giResLoB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                   WgpuTexture::Format::RGBA16Float),
          giResRead(&giResA), giResWrite(&giResB),
          giResWRead(&giResWA), giResWWrite(&giResWB),
          giResLoRead(&giResLoA), giResLoWrite(&giResLoB),
          // Albedo buffer for demodulation
          albedoTex(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                    WgpuTexture::Format::RGBA16Float),
          // Temporal variance moments ping-pong
          momentsA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                   WgpuTexture::Format::RGBA16Float),
          momentsB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                   WgpuTexture::Format::RGBA16Float),
          momentsRead(&momentsA), momentsWrite(&momentsB),
          // TAA history ping-pong
          taaHistA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                   WgpuTexture::Format::RGBA16Float),
          taaHistB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                   WgpuTexture::Format::RGBA16Float),
          taaHistRead(&taaHistA), taaHistWrite(&taaHistB),
          // Diffuse/specular split accumulation
          diffAccumA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                     WgpuTexture::Format::RGBA16Float),
          diffAccumB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                     WgpuTexture::Format::RGBA16Float),
          readDiffAccum(&diffAccumA), writeDiffAccum(&diffAccumB),
          specAccumA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                     WgpuTexture::Format::RGBA16Float),
          specAccumB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                     WgpuTexture::Format::RGBA16Float),
          readSpecAccum(&specAccumA), writeSpecAccum(&specAccumB),
          // Spatial filter ping-pong
          filteredA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                    WgpuTexture::Format::RGBA16Float),
          filteredB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                    WgpuTexture::Format::RGBA16Float),
          denoisedDiff(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                       WgpuTexture::Format::RGBA16Float),
          denoisedSpec(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                       WgpuTexture::Format::RGBA16Float),
          taaHistDiffA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                       WgpuTexture::Format::RGBA16Float),
          taaHistDiffB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                       WgpuTexture::Format::RGBA16Float),
          taaHistDiffRead(&taaHistDiffA), taaHistDiffWrite(&taaHistDiffB),
          taaHistSpecA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                       WgpuTexture::Format::RGBA16Float),
          taaHistSpecB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                       WgpuTexture::Format::RGBA16Float),
          taaHistSpecRead(&taaHistSpecA), taaHistSpecWrite(&taaHistSpecB),
          // TAAU full-res ping-pong textures
          upscaleTexA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                      WgpuTexture::Format::RGBA16Float,
                      WgpuTexture::Storage | WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          upscaleTexB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                      WgpuTexture::Format::RGBA16Float,
                      WgpuTexture::Storage | WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          zeroTex(r, 1u, 1u,
                  WgpuTexture::Format::RGBA16Float,
                  WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          upscaleRead(&upscaleTexA), upscaleWrite(&upscaleTexB),
          upscalePipeline(r, upscaleWGSL, "upscale_main"),
          upscaleUniBuf(r, sizeof(UpscaleGpuUniforms)),
          // Denoiser pipelines
          taaPipeline(r, taaWGSL, "taa_main"),
          atrousPipeline(r, svgfAtrousWGSL, "svgf_atrous_main"),
          preFilterPipeline(r, preFilterWGSL, "prefilter_main"),
          taaUniBuf(r, sizeof(TaaGpuUniforms)),
          atrousUniBuf(r, sizeof(AtrousGpuUniforms)),
          preFilterUniBuf(r, sizeof(PreFilterGpuUniforms)),
          // Storage buffers (small placeholders — grown dynamically on first build)
          bvhNodeBuf(r, static_cast<size_t>(2 * INIT_TRI_CAP - 1) * BVH4_GPU_U32S * sizeof(uint32_t),
                     WgpuBuffer::Usage::Storage),
          bvhCounterBuf(r, static_cast<size_t>(2 * INIT_TRI_CAP - 1) * sizeof(uint32_t),
                        WgpuBuffer::Usage::Storage),
          refitMetaBuf(r, static_cast<size_t>(2 * INIT_TRI_CAP - 1) * BVH4_REFIT_INTS * sizeof(int32_t),
                       WgpuBuffer::Usage::Storage),
          objTriBuf(r, static_cast<size_t>(INIT_TRI_CAP) * 48 * sizeof(float),
                    WgpuBuffer::Usage::Storage),
          objTriBuf2(r, 192u, WgpuBuffer::Usage::Storage),  // placeholder — grown when needed
          matrixBuf(r, static_cast<size_t>(INIT_MESH_CAP) * 32 * sizeof(float),
                    WgpuBuffer::Usage::Storage),
          motionMatBuf(r, static_cast<size_t>(INIT_MESH_CAP) * 16 * sizeof(float),
                       WgpuBuffer::Usage::Storage),
          leafIndexBuf(r, static_cast<size_t>(INIT_TRI_CAP) * sizeof(int),
                       WgpuBuffer::Usage::Storage),
          emissiveTriBuf(r, static_cast<size_t>(INIT_TRI_CAP) * 4 * sizeof(float),
                         WgpuBuffer::Usage::Storage),
          // Uniform buffers
          vtUniBuf(r, sizeof(VtGpuUniforms)),
          refitUniBuf(r, sizeof(RefitGpuUniforms)),
          rtUniformBuf(r, sizeof(RtGpuUniforms)),
          // Compute pipelines
          vtPipeline(r, buildVtShader(), "vt_main"),
          refitPipeline(r, buildRefitShader(), "bvh_refit"),
          rtPipeline(r, buildRtShader(false), "rt_main"),
          // Display pipeline
          displayCam(-1.f, 1.f, 1.f, -1.f, 0.1f, 10.f),
          // CPU buffers — empty; sized dynamically by async build
          width_(w), height_(h),
          fullWidth_(w), fullHeight_(h) {

        // Wire compute pipeline bindings
        vtPipeline.setStorageBufferRead(0, objTriBuf);
        vtPipeline.setStorageBufferRead(1, matrixBuf);
        vtPipeline.setStorageTexture(2, triTex);
        vtPipeline.setUniformBuffer(3, vtUniBuf);
        vtPipeline.setStorageBufferRead(4, objTriBuf2);

        refitPipeline.setTexture(0, triTex);
        refitPipeline.setStorageBuffer(1, bvhNodeBuf);
        refitPipeline.setStorageBuffer(2, bvhCounterBuf);
        refitPipeline.setStorageBufferRead(3, leafIndexBuf);
        refitPipeline.setUniformBuffer(4, refitUniBuf);
        refitPipeline.setStorageBufferRead(5, refitMetaBuf);

        // RT pipelines — set ALL bindings upfront (per-frame ones get overwritten)
        rtPipeline.setUniformBuffer(0, rtUniformBuf);
        rtPipeline.setTexture(1, *readAccum);
        rtPipeline.setStorageTexture(2, *writeAccum);
        rtPipeline.setStorageBufferRead(3, bvhNodeBuf);
        rtPipeline.setTexture(4, matTex);
        rtPipeline.setTexture(5, triTex);
        rtPipeline.setTexture(6, texAtlasTex);
        rtPipeline.setTexture(7, *readHitMesh);
        rtPipeline.setStorageTexture(8, *writeHitMesh);
        rtPipeline.setTexture(9, envTexGpu);
        rtPipeline.setStorageTexture(10, *gBufCur);
        rtPipeline.setStorageBufferRead(11, emissiveTriBuf);
        rtPipeline.setTexture(12, envCdfTex);
        rtPipeline.setTexture(13, envMargTex);
        rtPipeline.setStorageTexture(14, albedoTex);
        rtPipeline.setTexture(15, *gBufPrev);
        rtPipeline.setTexture(16, bgTexGpu);
        rtPipeline.setTexture(17, *reservoirRead);
        rtPipeline.setStorageTexture(18, *reservoirWrite);
        rtPipeline.setTexture(19, *reservoirWRead);
        rtPipeline.setStorageTexture(20, *reservoirWWrite);
        rtPipeline.setTexture(21, *momentsRead);
        rtPipeline.setStorageTexture(22, *momentsWrite);
        rtPipeline.setTexture(23, *readDiffAccum);
        rtPipeline.setStorageTexture(24, *writeDiffAccum);
        rtPipeline.setTexture(25, *readSpecAccum);
        rtPipeline.setStorageTexture(26, *writeSpecAccum);
        rtPipeline.setStorageTexture(27, *stableGBufCur);
        rtPipeline.setTexture(28, *stableGBufPrev);
        rtPipeline.setStorageBufferRead(29, motionMatBuf);
        rtPipeline.setTexture(30, *giResRead);
        rtPipeline.setStorageTexture(31, *giResWrite);
        rtPipeline.setTexture(32, *giResWRead);
        rtPipeline.setStorageTexture(33, *giResWWrite);
        rtPipeline.setTexture(34, *giResLoRead);
        rtPipeline.setStorageTexture(35, *giResLoWrite);

        // TAA pipeline — set ALL bindings upfront
        taaPipeline.setUniformBuffer(0, taaUniBuf);
        taaPipeline.setTexture(1, *readAccum);
        taaPipeline.setTexture(2, *gBufPrev);
        taaPipeline.setTexture(3, *taaHistRead);
        taaPipeline.setStorageTexture(4, *taaHistWrite);
        taaPipeline.setTexture(5, *readHitMesh);
        taaPipeline.setStorageBufferRead(6, motionMatBuf);
        taaPipeline.setTexture(7, albedoTex);
        taaPipeline.setTexture(8, *momentsRead);  // temporally accumulated (E[L], E[L²])
        taaPipeline.setTexture(9, *gBufCur);      // previous frame g-buffer (depth for revealed-bg check)
        taaPipeline.setTexture(10, *stableGBufPrev);  // current frame stable g-buffer
        taaPipeline.setTexture(11, *stableGBufCur);   // previous frame stable g-buffer

        // Spatial filter — set ALL bindings upfront
        atrousPipeline.setUniformBuffer(0, atrousUniBuf);
        atrousPipeline.setTexture(1, *readAccum);
        atrousPipeline.setStorageTexture(2, *writeAccum);
        atrousPipeline.setTexture(3, *gBufPrev);
        atrousPipeline.setTexture(4, albedoTex);
        atrousPipeline.setTexture(5, *readHitMesh);
        atrousPipeline.setTexture(6, *momentsRead);
        atrousPipeline.setTexture(7, *stableGBufPrev);

        // Pre-filter pipeline bindings
        preFilterPipeline.setUniformBuffer(0, preFilterUniBuf);
        preFilterPipeline.setTexture(1, *readDiffAccum);
        preFilterPipeline.setStorageTexture(2, filteredA);
        preFilterPipeline.setTexture(3, *gBufPrev);
        preFilterPipeline.setTexture(4, *readHitMesh);

        // TAAU pipeline — initial bindings (per-frame ones refreshed in render)
        upscalePipeline.setUniformBuffer(0, upscaleUniBuf);
        upscalePipeline.setTexture(1, denoisedDiff);
        upscalePipeline.setTexture(2, denoisedSpec);
        upscalePipeline.setTexture(3, *gBufCur);
        upscalePipeline.setTexture(4, *upscaleRead);
        upscalePipeline.setStorageTexture(5, *upscaleWrite);

        // Kick off async shader compilation for the small helper pipelines.
        // The large RT shaders are compiled after the first topology build so that
        // async uses the real (scene-sized) buffer bindings — this avoids a costly
        // double-compilation where the initial async result gets discarded because
        // layoutDirty=true (topology bindings changed) and the pipeline must be
        // rebuilt synchronously on the main thread a second time.
        vtPipeline.startAsyncBuild();
        refitPipeline.startAsyncBuild();
        taaPipeline.startAsyncBuild();
        preFilterPipeline.startAsyncBuild();
        atrousPipeline.startAsyncBuild();
        upscalePipeline.startAsyncBuild();
        std::cerr << "[PathTracer] Async shader compilation started for helper pipelines" << std::endl;

        // Zero-fill accumulators and SVGF textures
        {
            std::vector<float> zeros(w * h * 4, 0.f);
            accumA.write(zeros.data(), zeros.size() * sizeof(float));
            accumB.write(zeros.data(), zeros.size() * sizeof(float));
            gBufA.write(zeros.data(), zeros.size() * sizeof(float));
            gBufB.write(zeros.data(), zeros.size() * sizeof(float));
            stableGBufA.write(zeros.data(), zeros.size() * sizeof(float));
            stableGBufB.write(zeros.data(), zeros.size() * sizeof(float));
            taaHistA.write(zeros.data(), zeros.size() * sizeof(float));
            taaHistB.write(zeros.data(), zeros.size() * sizeof(float));
            momentsA.write(zeros.data(), zeros.size() * sizeof(float));
            momentsB.write(zeros.data(), zeros.size() * sizeof(float));
        }
        // Init albedo to white (no demodulation effect until first frame writes real values)
        {
            std::vector<float> ones(w * h * 4, 1.f);
            albedoTex.write(ones.data(), ones.size() * sizeof(float));
        }
        // Fill hitMesh textures with sentinel 128.0f (= "no hit")
        {
            std::vector<float> hitSentinel(w * h * 4, 128.f);
            hitMeshA.write(hitSentinel.data(), hitSentinel.size() * sizeof(float));
            hitMeshB.write(hitSentinel.data(), hitSentinel.size() * sizeof(float));
        }

        // Display quad
        displayCam.position.z = 1.f;
        displayMat = ShaderMaterial::create();
        displayMat->vertexShader = displayWGSL;
        displayMat->fragmentShader = displayWGSL;
        displayMat->customTextures["accumTex"] = readAccum;
        displayMat->customTextures["gBufTex"]  = gBufPrev;
        displayMat->customTextures["diffTex"]      = readDiffAccum;
        displayMat->customTextures["specTex"]      = readSpecAccum;
        displayMat->customTextures["upscaleTex"]   = &zeroTex;
        displayScene.add(Mesh::create(PlaneGeometry::create(2.f, 2.f), displayMat));

        // Depth-fill: build shader, BGL, layout, and uniform buffer now.
        // The pipeline itself is created lazily on first use (needs sample count from live frame).
        {
            WGPUShaderModuleDescriptor smDesc{};
            smDesc.label = WGPUStringView{"depth_fill_sm", WGPU_STRLEN};
            WGPUShaderSourceWGSL wgslSrc{};
            wgslSrc.chain.sType = WGPUSType_ShaderSourceWGSL;
            wgslSrc.code = WGPUStringView{depthFillWGSL, WGPU_STRLEN};
            smDesc.nextInChain = reinterpret_cast<WGPUChainedStruct*>(&wgslSrc);
            depthFillShader_ = wgpuDeviceCreateShaderModule(device, &smDesc);

            WGPUBindGroupLayoutEntry bglEntries[2]{};
            bglEntries[0].binding = 0;
            bglEntries[0].visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
            bglEntries[0].buffer.type = WGPUBufferBindingType_Uniform;
            bglEntries[0].buffer.minBindingSize = sizeof(DepthFillUniforms);
            bglEntries[1].binding = 1;
            bglEntries[1].visibility = WGPUShaderStage_Fragment;
            bglEntries[1].texture.sampleType = WGPUTextureSampleType_Float;  // RGBA16Float is filterable
            bglEntries[1].texture.viewDimension = WGPUTextureViewDimension_2D;
            WGPUBindGroupLayoutDescriptor bglDesc{};
            bglDesc.label = WGPUStringView{"depth_fill_bgl", WGPU_STRLEN};
            bglDesc.entryCount = 2;
            bglDesc.entries = bglEntries;
            depthFillBGL_ = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);

            WGPUPipelineLayoutDescriptor plDesc{};
            plDesc.label = WGPUStringView{"depth_fill_pl", WGPU_STRLEN};
            plDesc.bindGroupLayoutCount = 1;
            plDesc.bindGroupLayouts = &depthFillBGL_;
            depthFillPipeLayout_ = wgpuDeviceCreatePipelineLayout(device, &plDesc);

            WGPUBufferDescriptor ubDesc{};
            ubDesc.label = WGPUStringView{"depth_fill_uni", WGPU_STRLEN};
            ubDesc.size = sizeof(DepthFillUniforms);
            ubDesc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
            depthFillUniBuf_ = wgpuDeviceCreateBuffer(device, &ubDesc);
        }
    }

    // Build (or rebuild) the depth-fill render pipeline for a given MSAA sample count.
    void ensureDepthFillPipeline(uint32_t sampleCount) {
        if (depthFillPipeline_ && depthFillSampleCount_ == sampleCount) return;
        if (depthFillPipeline_) { wgpuRenderPipelineRelease(depthFillPipeline_); depthFillPipeline_ = nullptr; }
        WGPURenderPipelineDescriptor rpDesc{};
        rpDesc.label = WGPUStringView{"depth_fill_rp", WGPU_STRLEN};
        rpDesc.layout = depthFillPipeLayout_;
        rpDesc.vertex.module = depthFillShader_;
        rpDesc.vertex.entryPoint = WGPUStringView{"vs", WGPU_STRLEN};
        WGPUPrimitiveState prim{};
        prim.topology = WGPUPrimitiveTopology_TriangleList;
        rpDesc.primitive = prim;
        WGPUDepthStencilState ds{};
        ds.format = WGPUTextureFormat_Depth24Plus;
        ds.depthWriteEnabled = WGPUOptionalBool_True;
        ds.depthCompare = WGPUCompareFunction_Always;
        rpDesc.depthStencil = &ds;
        WGPUFragmentState frag{};
        frag.module = depthFillShader_;
        frag.entryPoint = WGPUStringView{"fs", WGPU_STRLEN};
        frag.targetCount = 0;
        rpDesc.fragment = &frag;
        rpDesc.multisample.count = sampleCount;
        rpDesc.multisample.mask  = 0xFFFFFFFF;
        depthFillPipeline_ = wgpuDeviceCreateRenderPipeline(device, &rpDesc);
        depthFillSampleCount_ = sampleCount;
    }

    void recreateAccumTextures(int w, int h) {
        width_ = w;
        height_ = h;
        auto uw = static_cast<uint32_t>(w);
        auto uh = static_cast<uint32_t>(h);
        auto fmt = WgpuTexture::Format::RGBA16Float;

        accumA = WgpuTexture(renderer, uw, uh, fmt);
        accumB = WgpuTexture(renderer, uw, uh, fmt);
        readAccum = &accumA;
        writeAccum = &accumB;

        hitMeshA = WgpuTexture(renderer, uw, uh, fmt);
        hitMeshB = WgpuTexture(renderer, uw, uh, fmt);
        readHitMesh  = &hitMeshA;
        writeHitMesh = &hitMeshB;

        const uint32_t gBufUsage = WgpuTexture::Storage | WgpuTexture::TextureBinding
                                 | WgpuTexture::CopyDst | WgpuTexture::RenderAttachment;
        gBufA = WgpuTexture(renderer, uw, uh, fmt, gBufUsage);
        gBufB = WgpuTexture(renderer, uw, uh, fmt, gBufUsage);
        gBufCur  = &gBufA;
        gBufPrev = &gBufB;

        const auto sdUsage = WgpuTexture::Storage | WgpuTexture::TextureBinding | WgpuTexture::CopyDst;
        stableGBufA = WgpuTexture(renderer, uw, uh, WgpuTexture::Format::RGBA16Float, sdUsage);
        stableGBufB = WgpuTexture(renderer, uw, uh, WgpuTexture::Format::RGBA16Float, sdUsage);
        stableGBufCur  = &stableGBufA;
        stableGBufPrev = &stableGBufB;

        auto fmt32 = WgpuTexture::Format::RGBA32Float;
        reservoirA  = WgpuTexture(renderer, uw, uh, fmt32);
        reservoirB  = WgpuTexture(renderer, uw, uh, fmt32);
        reservoirWA = WgpuTexture(renderer, uw, uh, fmt32);
        reservoirWB = WgpuTexture(renderer, uw, uh, fmt32);
        reservoirRead   = &reservoirA;
        reservoirWrite  = &reservoirB;
        reservoirWRead  = &reservoirWA;
        reservoirWWrite = &reservoirWB;

        giResA  = WgpuTexture(renderer, uw, uh, fmt32);
        giResB  = WgpuTexture(renderer, uw, uh, fmt32);
        giResWA = WgpuTexture(renderer, uw, uh, fmt32);
        giResWB = WgpuTexture(renderer, uw, uh, fmt32);
        giResLoA = WgpuTexture(renderer, uw, uh, fmt);
        giResLoB = WgpuTexture(renderer, uw, uh, fmt);
        giResRead   = &giResA;
        giResWrite  = &giResB;
        giResWRead  = &giResWA;
        giResWWrite = &giResWB;
        giResLoRead  = &giResLoA;
        giResLoWrite = &giResLoB;

        albedoTex = WgpuTexture(renderer, uw, uh, fmt);

        momentsA = WgpuTexture(renderer, uw, uh, fmt);
        momentsB = WgpuTexture(renderer, uw, uh, fmt);
        momentsRead  = &momentsA;
        momentsWrite = &momentsB;

        taaHistA = WgpuTexture(renderer, uw, uh, fmt);
        taaHistB = WgpuTexture(renderer, uw, uh, fmt);
        taaHistRead  = &taaHistA;
        taaHistWrite = &taaHistB;

        diffAccumA = WgpuTexture(renderer, uw, uh, fmt);
        diffAccumB = WgpuTexture(renderer, uw, uh, fmt);
        readDiffAccum  = &diffAccumA;
        writeDiffAccum = &diffAccumB;
        specAccumA = WgpuTexture(renderer, uw, uh, fmt);
        specAccumB = WgpuTexture(renderer, uw, uh, fmt);
        readSpecAccum  = &specAccumA;
        writeSpecAccum = &specAccumB;

        taaHistDiffA = WgpuTexture(renderer, uw, uh, fmt);
        taaHistDiffB = WgpuTexture(renderer, uw, uh, fmt);
        // Workaround: use displayable textures for TAA history ping-pong
        // (taaHistDiff/Spec textures have display issues in WebGPU abstraction)
        taaHistDiffRead  = &taaHistDiffA;
        taaHistDiffWrite = &taaHistDiffB;
        taaHistSpecA = WgpuTexture(renderer, uw, uh, fmt);
        taaHistSpecB = WgpuTexture(renderer, uw, uh, fmt);
        taaHistSpecRead  = &taaHistSpecA;
        taaHistSpecWrite = &taaHistSpecB;

        filteredA = WgpuTexture(renderer, uw, uh, fmt);
        filteredB = WgpuTexture(renderer, uw, uh, fmt);
        denoisedDiff = WgpuTexture(renderer, uw, uh, fmt);
        denoisedSpec = WgpuTexture(renderer, uw, uh, fmt);

        std::vector<float> zeros(w * h * 4, 0.f);
        accumA.write(zeros.data(), zeros.size() * sizeof(float));
        accumB.write(zeros.data(), zeros.size() * sizeof(float));
        gBufA.write(zeros.data(), zeros.size() * sizeof(float));
        gBufB.write(zeros.data(), zeros.size() * sizeof(float));
        stableGBufA.write(zeros.data(), zeros.size() * sizeof(float));
        stableGBufB.write(zeros.data(), zeros.size() * sizeof(float));
        taaHistA.write(zeros.data(), zeros.size() * sizeof(float));
        taaHistB.write(zeros.data(), zeros.size() * sizeof(float));
        taaHistDiffA.write(zeros.data(), zeros.size() * sizeof(float));
        taaHistDiffB.write(zeros.data(), zeros.size() * sizeof(float));
        taaHistSpecA.write(zeros.data(), zeros.size() * sizeof(float));
        taaHistSpecB.write(zeros.data(), zeros.size() * sizeof(float));
        momentsA.write(zeros.data(), zeros.size() * sizeof(float));
        momentsB.write(zeros.data(), zeros.size() * sizeof(float));
        diffAccumA.write(zeros.data(), zeros.size() * sizeof(float));
        diffAccumB.write(zeros.data(), zeros.size() * sizeof(float));
        specAccumA.write(zeros.data(), zeros.size() * sizeof(float));
        specAccumB.write(zeros.data(), zeros.size() * sizeof(float));
        denoisedDiff.write(zeros.data(), zeros.size() * sizeof(float));
        denoisedSpec.write(zeros.data(), zeros.size() * sizeof(float));
        filteredA.write(zeros.data(), zeros.size() * sizeof(float));
        filteredB.write(zeros.data(), zeros.size() * sizeof(float));
        std::vector<float> hitSentinel(w * h * 4, 128.f);
        hitMeshA.write(hitSentinel.data(), hitSentinel.size() * sizeof(float));
        hitMeshB.write(hitSentinel.data(), hitSentinel.size() * sizeof(float));

        rtPipeline.setStorageTexture(10, *gBufCur);
        rtPipeline.setTexture(15, *gBufPrev);
        rtPipeline.setStorageTexture(14, albedoTex);
        rtPipeline.setTexture(17, *reservoirRead);
        rtPipeline.setStorageTexture(18, *reservoirWrite);
        rtPipeline.setTexture(19, *reservoirWRead);
        rtPipeline.setStorageTexture(20, *reservoirWWrite);
        rtPipeline.setTexture(21, *momentsRead);
        rtPipeline.setStorageTexture(22, *momentsWrite);
        rtPipeline.setTexture(23, *readDiffAccum);
        rtPipeline.setStorageTexture(24, *writeDiffAccum);
        rtPipeline.setTexture(25, *readSpecAccum);
        rtPipeline.setStorageTexture(26, *writeSpecAccum);
        rtPipeline.setStorageTexture(27, *stableGBufCur);
        rtPipeline.setTexture(28, *stableGBufPrev);
        rtPipeline.setStorageBufferRead(29, motionMatBuf);
        rtPipeline.setTexture(30, *giResRead);
        rtPipeline.setStorageTexture(31, *giResWrite);
        rtPipeline.setTexture(32, *giResWRead);
        rtPipeline.setStorageTexture(33, *giResWWrite);
        rtPipeline.setTexture(34, *giResLoRead);
        rtPipeline.setStorageTexture(35, *giResLoWrite);
        atrousPipeline.setTexture(4, albedoTex);
        atrousPipeline.setTexture(6, *momentsRead);
        preFilterPipeline.setTexture(1, *readDiffAccum);
        preFilterPipeline.setStorageTexture(2, filteredA);
        preFilterPipeline.setTexture(3, *gBufPrev);
        preFilterPipeline.setTexture(4, *readHitMesh);
        taaPipeline.setTexture(7, albedoTex);
        taaPipeline.setTexture(8, *momentsRead);
        taaPipeline.setTexture(9, *gBufCur);  // prev frame depth
        taaPipeline.setTexture(10, *stableGBufPrev);
        taaPipeline.setTexture(11, *stableGBufCur);

        frameCount_ = 0.f;
    }

    void recreateUpscaleTextures(int fw, int fh) {
        auto ufw = static_cast<uint32_t>(fw);
        auto ufh = static_cast<uint32_t>(fh);
        const uint32_t usage = WgpuTexture::Storage | WgpuTexture::TextureBinding | WgpuTexture::CopyDst;
        auto fmt = WgpuTexture::Format::RGBA16Float;
        upscaleTexA = WgpuTexture(renderer, ufw, ufh, fmt, usage);
        upscaleTexB = WgpuTexture(renderer, ufw, ufh, fmt, usage);
        upscaleRead  = &upscaleTexA;
        upscaleWrite = &upscaleTexB;
        std::vector<float> zeros(fw * fh * 4, 0.f);
        upscaleTexA.write(zeros.data(), zeros.size() * sizeof(float));
        upscaleTexB.write(zeros.data(), zeros.size() * sizeof(float));
    }

    void resetUpscaleHistory() {
        // Zero-fill histLen (w=0 → first frame takes 100% current, no stale blending)
        const int fw = fullWidth_;
        const int fh = fullHeight_;
        if (fw <= 0 || fh <= 0) return;
        std::vector<float> zeros(fw * fh * 4, 0.f);
        upscaleTexA.write(zeros.data(), zeros.size() * sizeof(float));
        upscaleTexB.write(zeros.data(), zeros.size() * sizeof(float));
    }
};

// ---------------------------------------------------------------------------
// WgpuPathTracer public API
// ---------------------------------------------------------------------------

WgpuPathTracer::WgpuPathTracer(WgpuRenderer& renderer, std::pair<int, int> size)
    : pimpl_(std::make_unique<Impl>(renderer, size.first, size.second)) {}

WgpuPathTracer::~WgpuPathTracer() = default;

void WgpuPathTracer::render(Object3D& scene, Camera& camera) {
    auto& d = *pimpl_;

    // Derive camera vectors from camera world matrix (controller-agnostic)
    const Vector3& camPos = camera.position;
    Vector3 fwd;
    camera.getWorldDirection(fwd);
    Vector3 rgt = Vector3(fwd).cross(Vector3(0.f, 1.f, 0.f)).normalize();
    Vector3 up = Vector3(rgt).cross(fwd);

    // Reset accumulation on camera movement
    const bool camMoved =
            (camPos - d.prevCamPos_).length() > 1e-4f ||
            (fwd - d.prevCamDir_).length() > 1e-4f;
    if (camMoved) {
        d.frameCount_ = 0.f;
        d.prevCamPos_ = camPos;
        d.prevCamDir_ = fwd;
    }

    // Collect visible, non-wireframe, non-line meshes and lights
    std::vector<Mesh*> rtMeshes;
    scene.traverseVisible([&](Object3D& o) {
        auto* m_ptr = dynamic_cast<Mesh*>(&o);
        if (!m_ptr) return;
        auto& m = *m_ptr;
        if (d.overlayLayer_ >= 0 && m.layers.isEnabled(static_cast<unsigned>(d.overlayLayer_))) return;
        auto* mat = m.material().get();
        if (!mat->visible) return;
        auto* mww = mat->as<MaterialWithWireframe>();
        if (mww && mww->wireframe) return;
        if (mat->is<LineBasicMaterial>()) return;
        // Transparent meshes with no texture and no transmission are raster-only
        // overlay effects (e.g. separate clearcoat geometry layers). They have no
        // physical meaning in path tracing and render as opaque shells that occlude
        // everything beneath them.
        if (mat->transparent) {
            auto* mwm = dynamic_cast<MaterialWithMap*>(mat);
            auto* mwt = dynamic_cast<MaterialWithTransmission*>(mat);
            const bool hasMap = mwm && mwm->map;
            const bool hasTransmission = mwt && mwt->transmission > 0.f;
            const bool hasBlend = mat->opacity < 0.999f;
            if (!hasMap && !hasTransmission && !hasBlend) return;
        }
        rtMeshes.push_back(&m);
    });
    std::vector<PointLight*> pointLights;
    scene.traverseType<PointLight>([&](PointLight& l) { if (l.visible) pointLights.push_back(&l); });
    std::vector<DirectionalLight*> dirLights;
    scene.traverseType<DirectionalLight>([&](DirectionalLight& l) { if (l.visible) dirLights.push_back(&l); });
    std::vector<SpotLight*> spotLights;
    scene.traverseType<SpotLight>([&](SpotLight& l) { if (l.visible) spotLights.push_back(&l); });

    // Compute entry count for instanced mesh awareness
    int totalEntryCount = 0;
    for (auto* m : rtMeshes) {
        auto* inst = dynamic_cast<InstancedMesh*>(m);
        totalEntryCount += (inst && inst->count() > 0) ? static_cast<int>(inst->count()) : 1;
    }

    // Detect topology change (mesh list or instance configuration changed)
    const bool topoChanged = (rtMeshes != d.prevMeshes) || (totalEntryCount != d.prevEntryCount_);

    // Build scene data (BVH, geometry buffers, atlas, emissives) when topology changes.
    // Native: async on background thread.  Emscripten: synchronous (no pthreads).
    bool topoJustFinished = false;
#ifdef __EMSCRIPTEN__
    if (topoChanged) {
        d.prevMeshes = rtMeshes;
        d.prevEntryCount_ = totalEntryCount;
        auto meshes = rtMeshes;

        // Expand InstancedMesh objects into individual entries
        for (auto* m : meshes) m->updateWorldMatrix(true, true);
        auto entries = expandMeshEntries(meshes);

        Impl::AsyncBuildResult r;
        r.meshes = meshes;
        r.entries = entries;
        r.texSlotMap.clear();
        auto [atlasData, atlasLayers, atlasCols_, tileSize_] = buildAtlas(meshes, r.texSlotMap, d.textureResolution_);
        r.atlasData = std::move(atlasData);
        r.atlasLayers = atlasLayers;
        r.atlasCols = atlasCols_;
        r.tileSize = tileSize_;

        int totalTris = 0;
        for (auto& entry : entries) {
            auto* geo = entry.mesh->geometry().get();
            auto* idx = geo->getIndex();
            auto* pos = geo->getAttribute<float>("position");
            if (!pos) continue;
            totalTris += idx ? static_cast<int>(idx->count()) / 3
                             : static_cast<int>(pos->count()) / 3;
        }
        const int matCount = static_cast<int>(meshes.size());
        const int meshCount = static_cast<int>(entries.size());
        const int triCap = d.maxTriCap();
        const int maxTotalTris = triCap * 2;  // two split buffers
        r.triCapacity  = std::clamp(totalTris, 1, maxTotalTris);
        if (totalTris > maxTotalTris) {
            std::cerr << "[PathTracer] Warning: scene has " << totalTris
                      << " tris, capped to " << maxTotalTris << " (2x GPU buffer limit)\n";
        }
        r.objTriSplit = std::min(r.triCapacity, triCap);  // first buffer holds up to triCap
        r.matCapacity  = std::max(matCount, 1);
        r.meshCapacity = std::max(meshCount, 1);

        const int pages = triTexPages(r.triCapacity);
        r.triBuffer.resize(static_cast<size_t>(TEX_PAGE_WIDTH) * TRI_TEX_HEIGHT * pages * 4, 0.f);
        r.matBuffer.resize(static_cast<size_t>(r.matCapacity) * MAT_TEX_HEIGHT * 4, 0.f);
        r.rawObjTriBuf.resize(static_cast<size_t>(r.triCapacity) * 48, 0.f);
        r.matrixCpuBuf.resize(static_cast<size_t>(r.meshCapacity) * 32, 0.f);
        r.triCount = buildGeometryBuffers(entries, r.texSlotMap, r.triBuffer, r.matBuffer,
                                           r.rawObjTriBuf, r.matrixCpuBuf,
                                           r.triCapacity, r.matCapacity, r.meshCapacity);
        buildBVH(r.triBuffer, r.triCount, r.bvhNodes, r.bvhIndices, r.leafIndices, r.rawObjTriBuf);
        r.numBvhNodes = static_cast<int>(r.bvhNodes.size());
        r.bvhCapacity = std::max(r.numBvhNodes, 1);  // at least 1 to avoid 0-byte buffers
        r.bvhNodeCpuBuf.resize(static_cast<size_t>(r.bvhCapacity) * BVH4_GPU_U32S, 0u);
        packBvh4Buffer(r.bvhNodes, r.bvhNodeCpuBuf, r.bvhCapacity);
        r.refitMetaCpuBuf.resize(static_cast<size_t>(r.bvhCapacity) * BVH4_REFIT_INTS, 0);
        packRefitMetadata(r.bvhNodes, r.refitMetaCpuBuf, r.bvhCapacity);

        r.emissiveTriCount = 0;
        r.emissiveTotalArea = 0.f;
        r.emissiveTotalPower = 0.f;
        for (int ti = 0; ti < r.triCount; ti++) {
            const int matIdx = static_cast<int>(r.triBuffer[pagedIdx(ti, 0) + 3]);
            const float er = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 0];
            const float eg = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 1];
            const float eb = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 2];
            const float luminance = 0.2126f * er + 0.7152f * eg + 0.0722f * eb;
            if (luminance > 0.001f) {
                const float* v0p = r.triBuffer.data() + pagedIdx(ti, 0);
                const float* v1p = r.triBuffer.data() + pagedIdx(ti, 1);
                const float* v2p = r.triBuffer.data() + pagedIdx(ti, 2);
                Vector3 v0(v0p[0], v0p[1], v0p[2]);
                Vector3 v1(v1p[0], v1p[1], v1p[2]);
                Vector3 v2(v2p[0], v2p[1], v2p[2]);
                Vector3 cross;
                cross.crossVectors(v1 - v0, v2 - v0);
                const float area = cross.length() * 0.5f;
                if (area > 1e-8f) {
                    const float power = area * luminance;
                    r.emissiveTotalArea += area;
                    r.emissiveTotalPower += power;
                    r.emissiveTriCpu.push_back(static_cast<float>(ti));
                    r.emissiveTriCpu.push_back(area);
                    r.emissiveTriCpu.push_back(r.emissiveTotalPower);
                    r.emissiveTriCpu.push_back(power);
                    r.emissiveTriCount++;
                    // Record which mesh this emissive tri belongs to (triData row1.w = meshIdx)
                    r.emissiveMeshSet.insert(static_cast<int>(v1p[3]));
                }
            }
        }
        topoJustFinished = true;
        d.frameCount_ = 0.f;
#else
    if (topoChanged) {
        d.prevMeshes = rtMeshes;
        d.prevEntryCount_ = totalEntryCount;

        auto meshes = rtMeshes;
        for (auto* m : meshes) m->updateWorldMatrix(true, true);
        auto entries = expandMeshEntries(meshes);

        Impl::AsyncBuildResult r;
        r.meshes = meshes;
        r.entries = entries;
        r.texSlotMap.clear();
        auto [atlasData, atlasLayers, atlasCols_, tileSize_] = buildAtlas(meshes, r.texSlotMap, d.textureResolution_);
        r.atlasData = std::move(atlasData);
        r.atlasLayers = atlasLayers;
        r.atlasCols = atlasCols_;
        r.tileSize = tileSize_;

        int totalTris = 0;
        for (auto& entry : entries) {
            auto* geo = entry.mesh->geometry().get();
            auto* idx = geo->getIndex();
            auto* pos = geo->getAttribute<float>("position");
            if (!pos) continue;
            totalTris += idx ? static_cast<int>(idx->count()) / 3
                             : static_cast<int>(pos->count()) / 3;
        }
        const int matCount = static_cast<int>(meshes.size());
        const int meshCount = static_cast<int>(entries.size());
        const int triCap = d.maxTriCap();
        const int maxTotalTris = triCap * 2;  // two split buffers
        r.triCapacity  = std::clamp(totalTris, 1, maxTotalTris);
        if (totalTris > maxTotalTris) {
            std::cerr << "[PathTracer] Warning: scene has " << totalTris
                      << " tris, capped to " << maxTotalTris << " (2x GPU buffer limit)\n";
        }
        r.objTriSplit = std::min(r.triCapacity, triCap);  // first buffer holds up to triCap
        r.matCapacity  = std::max(matCount, 1);
        r.meshCapacity = std::max(meshCount, 1);

        const int pages = triTexPages(r.triCapacity);
        r.triBuffer.resize(static_cast<size_t>(TEX_PAGE_WIDTH) * TRI_TEX_HEIGHT * pages * 4, 0.f);
        r.matBuffer.resize(static_cast<size_t>(r.matCapacity) * MAT_TEX_HEIGHT * 4, 0.f);
        r.rawObjTriBuf.resize(static_cast<size_t>(r.triCapacity) * 48, 0.f);
        r.matrixCpuBuf.resize(static_cast<size_t>(r.meshCapacity) * 32, 0.f);
        r.triCount = buildGeometryBuffers(entries, r.texSlotMap, r.triBuffer, r.matBuffer,
                                           r.rawObjTriBuf, r.matrixCpuBuf,
                                           r.triCapacity, r.matCapacity, r.meshCapacity);
        buildBVH(r.triBuffer, r.triCount, r.bvhNodes, r.bvhIndices, r.leafIndices, r.rawObjTriBuf);
        r.numBvhNodes = static_cast<int>(r.bvhNodes.size());
        r.bvhCapacity = std::max(r.numBvhNodes, 1);
        r.bvhNodeCpuBuf.resize(static_cast<size_t>(r.bvhCapacity) * BVH4_GPU_U32S, 0u);
        packBvh4Buffer(r.bvhNodes, r.bvhNodeCpuBuf, r.bvhCapacity);
        r.refitMetaCpuBuf.resize(static_cast<size_t>(r.bvhCapacity) * BVH4_REFIT_INTS, 0);
        packRefitMetadata(r.bvhNodes, r.refitMetaCpuBuf, r.bvhCapacity);

        r.emissiveTriCount = 0;
        r.emissiveTotalArea = 0.f;
        r.emissiveTotalPower = 0.f;
        for (int ti = 0; ti < r.triCount; ti++) {
            const int matIdx = static_cast<int>(r.triBuffer[pagedIdx(ti, 0) + 3]);
            const float er = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 0];
            const float eg = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 1];
            const float eb = r.matBuffer[(2 * r.matCapacity + matIdx) * 4 + 2];
            const float luminance = 0.2126f * er + 0.7152f * eg + 0.0722f * eb;
            if (luminance > 0.001f) {
                const float* v0p = r.triBuffer.data() + pagedIdx(ti, 0);
                const float* v1p = r.triBuffer.data() + pagedIdx(ti, 1);
                const float* v2p = r.triBuffer.data() + pagedIdx(ti, 2);
                Vector3 v0(v0p[0], v0p[1], v0p[2]);
                Vector3 v1(v1p[0], v1p[1], v1p[2]);
                Vector3 v2(v2p[0], v2p[1], v2p[2]);
                Vector3 cross;
                cross.crossVectors(v1 - v0, v2 - v0);
                const float area = cross.length() * 0.5f;
                if (area > 1e-8f) {
                    const float power = area * luminance;
                    r.emissiveTotalArea += area;
                    r.emissiveTotalPower += power;
                    r.emissiveTriCpu.push_back(static_cast<float>(ti));
                    r.emissiveTriCpu.push_back(area);
                    r.emissiveTriCpu.push_back(r.emissiveTotalPower);
                    r.emissiveTriCpu.push_back(power);
                    r.emissiveTriCount++;
                    r.emissiveMeshSet.insert(static_cast<int>(v1p[3]));
                }
            }
        }
        topoJustFinished = true;
        d.frameCount_ = 0.f;
#endif

        // Move CPU results into Impl
        d.texSlotMap = std::move(r.texSlotMap);
        d.triBuffer = std::move(r.triBuffer);
        d.matBuffer = std::move(r.matBuffer);
        d.rawObjTriBuf = std::move(r.rawObjTriBuf);
        d.matrixCpuBuf = std::move(r.matrixCpuBuf);
        d.bvhNodes = std::move(r.bvhNodes);
        d.bvhIndices = std::move(r.bvhIndices);
        d.leafIndices = std::move(r.leafIndices);
        d.bvhNodeCpuBuf = std::move(r.bvhNodeCpuBuf);
        d.refitMetaCpuBuf = std::move(r.refitMetaCpuBuf);
        d.emissiveTriCpu = std::move(r.emissiveTriCpu);
        d.emissiveMeshSet_ = std::move(r.emissiveMeshSet);
        d.triCount_ = r.triCount;
        d.objTriSplit_ = r.objTriSplit;
        d.numBvhNodes_ = r.numBvhNodes;
        d.emissiveTriCount_ = r.emissiveTriCount;

        std::cerr << "[PathTracer] Scene: " << r.triCount << " tris, "
                  << r.numBvhNodes << " BVH nodes, "
                  << r.matCapacity << " materials, "
                  << r.meshCapacity << " meshes" << std::endl;
        d.emissiveTotalArea_ = r.emissiveTotalArea;
        d.emissiveTotalPower_ = r.emissiveTotalPower;

        // Grow GPU buffers when scene exceeds current capacity (same pattern as atlas)
        if (r.triCapacity != d.triCapacity_) {
            d.triCapacity_ = r.triCapacity;
            const int pages = triTexPages(r.triCapacity);
            d.triTex = WgpuTexture(d.renderer, TEX_PAGE_WIDTH, TRI_TEX_HEIGHT * pages,
                                    WgpuTexture::Format::RGBA32Float,
                                    WgpuTexture::Storage | WgpuTexture::TextureBinding);
            // Split objTri data across two buffers to stay within per-buffer size limits
            const size_t buf1Tris = static_cast<size_t>(r.objTriSplit);
            const size_t buf2Tris = static_cast<size_t>(std::max(r.triCapacity - r.objTriSplit, 0));
            d.objTriBuf = WgpuBuffer(d.renderer, buf1Tris * BYTES_PER_TRI,
                                      WgpuBuffer::Usage::Storage);
            d.objTriBuf2 = WgpuBuffer(d.renderer, std::max(buf2Tris * BYTES_PER_TRI, size_t(192)),
                                       WgpuBuffer::Usage::Storage);
            d.leafIndexBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.triCapacity) * sizeof(int),
                                         WgpuBuffer::Usage::Storage);
            d.emissiveTriBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.triCapacity) * 4 * sizeof(float),
                                           WgpuBuffer::Usage::Storage);
            d.vtPipeline.setStorageBufferRead(0, d.objTriBuf);
            d.vtPipeline.setStorageTexture(2, d.triTex);
            d.vtPipeline.setStorageBufferRead(4, d.objTriBuf2);
            d.refitPipeline.setTexture(0, d.triTex);
            d.refitPipeline.setStorageBufferRead(3, d.leafIndexBuf);
            d.rtPipeline.setTexture(5, d.triTex);
            d.rtPipeline.setStorageBufferRead(11, d.emissiveTriBuf);
        }
        if (r.bvhCapacity != d.bvhCapacity_) {
            d.bvhCapacity_ = r.bvhCapacity;
            d.bvhNodeBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.bvhCapacity) * BVH4_GPU_U32S * sizeof(uint32_t),
                                       WgpuBuffer::Usage::Storage);
            d.bvhCounterBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.bvhCapacity) * sizeof(uint32_t),
                                          WgpuBuffer::Usage::Storage);
            d.bvhCounterZeros.resize(r.bvhCapacity, 0u);
            d.refitMetaBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.bvhCapacity) * BVH4_REFIT_INTS * sizeof(int32_t),
                                         WgpuBuffer::Usage::Storage);
            d.refitPipeline.setStorageBuffer(1, d.bvhNodeBuf);
            d.refitPipeline.setStorageBuffer(2, d.bvhCounterBuf);
            d.refitPipeline.setStorageBufferRead(5, d.refitMetaBuf);
            d.rtPipeline.setStorageBufferRead(3, d.bvhNodeBuf);
        }
        if (r.matCapacity != d.matCapacity_) {
            d.matCapacity_ = r.matCapacity;
            d.matTex = WgpuTexture(d.renderer, r.matCapacity, MAT_TEX_HEIGHT,
                                    WgpuTexture::Format::RGBA32Float,
                                    WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
            d.rtPipeline.setTexture(4, d.matTex);
        }
        if (r.meshCapacity != d.meshCapacity_) {
            d.meshCapacity_ = r.meshCapacity;
            d.matrixBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.meshCapacity) * 32 * sizeof(float),
                                      WgpuBuffer::Usage::Storage);
            d.motionMatBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.meshCapacity) * 16 * sizeof(float),
                                         WgpuBuffer::Usage::Storage);
            d.vtPipeline.setStorageBufferRead(1, d.matrixBuf);
            d.taaPipeline.setStorageBufferRead(6, d.motionMatBuf);
            d.rtPipeline.setStorageBufferRead(29, d.motionMatBuf);
        }

        // Upload atlas
        if (r.atlasLayers != d.atlasLayers_ || r.atlasCols != d.atlasCols_ || r.tileSize != d.tileSize_) {
            d.atlasLayers_ = r.atlasLayers;
            d.atlasCols_ = r.atlasCols;
            d.tileSize_ = r.tileSize;
            const int layerW = d.atlasCols_ * d.tileSize_;
            const int layerH = d.atlasCols_ * d.tileSize_;
            d.texAtlasTex = WgpuTexture(d.renderer,
                    layerW, layerH,
                    WgpuTexture::Format::RGBA8Unorm,
                    WgpuTexture::Dimension::D2Array,
                    WgpuTexture::TextureBinding | WgpuTexture::CopyDst,
                    r.atlasLayers);
            d.rtPipeline.setTexture(6, d.texAtlasTex);
        }
        // Upload atlas layers
        const size_t layerBytes = static_cast<size_t>(d.atlasCols_ * d.tileSize_) * (d.atlasCols_ * d.tileSize_) * 4;
        for (int layer = 0; layer < r.atlasLayers; ++layer) {
            d.texAtlasTex.writeLayer(layer, r.atlasData.data() + layer * layerBytes, layerBytes);
        }

        // Upload geometry + BVH
        d.bvhNodeBuf.write(d.bvhNodeCpuBuf.data(), d.numBvhNodes_ * BVH4_GPU_U32S * sizeof(uint32_t));
        d.refitMetaBuf.write(d.refitMetaCpuBuf.data(), d.numBvhNodes_ * BVH4_REFIT_INTS * sizeof(int32_t));
        // Upload objTri data split across two buffers
        {
            const size_t splitAt = static_cast<size_t>(d.objTriSplit_);
            const size_t totalTris = static_cast<size_t>(d.triCount_);
            const size_t buf1Tris = std::min(totalTris, splitAt);
            const size_t buf2Tris = totalTris > splitAt ? totalTris - splitAt : 0;
            d.objTriBuf.write(d.rawObjTriBuf.data(), buf1Tris * BYTES_PER_TRI);
            if (buf2Tris > 0) {
                d.objTriBuf2.write(d.rawObjTriBuf.data() + splitAt * 48, buf2Tris * BYTES_PER_TRI);
            }
        }
        d.leafIndexBuf.write(d.leafIndices.data(), d.leafIndices.size() * sizeof(int));
        d.matTex.write(d.matBuffer.data(), d.matBuffer.size() * sizeof(float));

        if (d.emissiveTriCount_ > 0) {
            d.emissiveTriBuf.write(d.emissiveTriCpu.data(),
                                   d.emissiveTriCpu.size() * sizeof(float));
        }

        // Free large CPU-side build buffers now that data lives on GPU.
        { std::vector<float>().swap(d.triBuffer); }
        { std::vector<float>().swap(d.matBuffer); }
        { std::vector<float>().swap(d.rawObjTriBuf); }
        { std::vector<uint32_t>().swap(d.bvhNodeCpuBuf); }
        { std::vector<int32_t>().swap(d.refitMetaCpuBuf); }
        { std::vector<float>().swap(d.emissiveTriCpu); }

        // Start async RT pipeline compilation NOW — after all topology bindings (buffer sizes,
        // texture formats) are finalised.  This ensures:
        //  • layoutDirty stays false between now and the first dispatch (per-frame ping-pong
        //    swaps only change texture views, never the layout type/format → no dirty).
        //  • When async completes, the result is used directly without a second compile.
        // Previously these were started in the constructor with 1-tri placeholder bindings,
        // causing a layoutDirty discard → a second full synchronous compile on the main thread
        // (~30s extra freeze on first cold-cache launch).
        d.rtPipeline.startAsyncBuild();
        // Mark that triTex has not yet been populated via the VT pass.
        // The first real dispatch must run the VT pass even if no mesh moved.
        d.firstDispatchPending_ = true;
    }

    // Before first build, skip RT dispatch
    if (d.triCount_ == 0) {
        d.displayMat->customTextures["accumTex"] = d.readAccum;
        d.displayMat->customTextures["gBufTex"]  = d.gBufPrev;
        d.displayMat->customTextures["diffTex"]  = d.readDiffAccum;
        d.displayMat->customTextures["specTex"]  = d.readSpecAccum;
        d.displayMat->uniformsNeedUpdate = true;
        d.renderer.render(d.displayScene, d.displayCam);
        return;
    }

    scene.updateMatrixWorld();

    // Expand entries for movement detection (uses current world matrices)
    auto rtEntries = expandMeshEntries(rtMeshes);

    // Detect per-entry matrix changes; build bitmask of which entries moved.
    // movedBits is used by the GPU for per-pixel accumulation reset.
    // anyMeshMoved drives the vertex-transform and BVH-refit pipelines.
    uint32_t movedBits[4] = {0u, 0u, 0u, 0u};
    bool anyMeshMoved = (d.prevEntryMatrices.size() != rtEntries.size());

    if (topoJustFinished) {
        // Topology change: all pixels need to re-accumulate (mesh-to-triangle mapping changed)
        movedBits[0] = movedBits[1] = movedBits[2] = movedBits[3] = 0xFFFFFFFFu;
        anyMeshMoved = true;
    } else if (anyMeshMoved) {
        // Entry count mismatch (shouldn't happen without topo change, but be safe)
        movedBits[0] = movedBits[1] = movedBits[2] = movedBits[3] = 0xFFFFFFFFu;
    } else {
        for (size_t i = 0; i < rtEntries.size() && i < static_cast<size_t>(d.meshCapacity_) && i < 128u; ++i) {
            if (rtEntries[i].worldMatrix != d.prevEntryMatrices[i]) {
                anyMeshMoved = true;
                movedBits[i >> 5u] |= (1u << (i & 31u));
            }
        }
    }
    // Note: camMoved resets all pixels via fc==0u in the shader; no movedBits needed for that.

    // Compute per-entry motion matrices: prevWorld * inverse(curWorld)
    // Used by TAA to reproject pixels on moving objects to their previous-frame screen position.
    d.motionMatCpu.resize(static_cast<size_t>(d.meshCapacity_) * 16, 0.f);
    for (size_t i = 0; i < rtEntries.size() && i < static_cast<size_t>(d.meshCapacity_); ++i) {
        Matrix4 mot;  // identity by default
        if (i < d.prevEntryMatrices.size()) {
            Matrix4 curInv(rtEntries[i].worldMatrix);
            curInv.invert();
            mot.multiplyMatrices(d.prevEntryMatrices[i], curInv);
        }
        // else: identity (new entry, no previous frame)
        std::memcpy(d.motionMatCpu.data() + i * 16, mot.elements.data(), 16 * sizeof(float));
    }
    d.motionMatBuf.write(d.motionMatCpu.data(), d.motionMatCpu.size() * sizeof(float));

    d.prevEntryMatrices.resize(rtEntries.size());
    for (size_t i = 0; i < rtEntries.size(); ++i)
        d.prevEntryMatrices[i] = rtEntries[i].worldMatrix;
    if (anyMeshMoved) {
        if (!topoChanged) {
            std::ranges::fill(d.matrixCpuBuf, 0.f);
            int mi = 0;
            for (auto& entry : rtEntries) {
                if (mi >= d.meshCapacity_) break;
                const auto& w = entry.worldMatrix;
                Matrix4 nm(w);
                nm.invert().transpose();
                float* p = d.matrixCpuBuf.data() + mi * 32;
                std::memcpy(p, w.elements.data(), 16 * sizeof(float));
                std::memcpy(p + 16, nm.elements.data(), 16 * sizeof(float));
                ++mi;
            }
        }
        d.matrixBuf.write(d.matrixCpuBuf.data(), static_cast<size_t>(d.meshCapacity_) * 32 * sizeof(float));

        // Compute 2D dispatch dimensions (WebGPU max per-dimension is 65535)
        const uint32_t vtTotal = (static_cast<uint32_t>(d.triCount_) + 63u) / 64u;
        const uint32_t vtGx = (std::min)(vtTotal, 65535u);
        const uint32_t vtGy = (vtTotal + vtGx - 1u) / vtGx;
        d.vtDispatchX_ = vtGx;
        d.vtDispatchY_ = vtGy;

        VtGpuUniforms vtU{};
        vtU.triCount = static_cast<uint32_t>(d.triCount_);
        vtU.groupsX  = vtGx;
        vtU.splitAt  = static_cast<uint32_t>(d.objTriSplit_);
        d.vtUniBuf.write(&vtU, sizeof(vtU));

        const uint32_t rfTotal = (static_cast<uint32_t>(d.leafIndices.size()) + 63u) / 64u;
        const uint32_t rfGx = (std::min)(rfTotal, 65535u);
        const uint32_t rfGy = (rfTotal + rfGx - 1u) / rfGx;
        d.rfDispatchX_ = rfGx;
        d.rfDispatchY_ = rfGy;

        RefitGpuUniforms rfU{};
        rfU.leafCount = static_cast<uint32_t>(d.leafIndices.size());
        rfU.groupsX   = rfGx;
        d.refitUniBuf.write(&rfU, sizeof(rfU));
        d.bvhCounterBuf.write(d.bvhCounterZeros.data(),
                              static_cast<size_t>(d.numBvhNodes_) * sizeof(uint32_t));
    }

    // Compute tanHalfFov from the camera
    float tanHalfFov = 1.f;
    if (auto* persp = dynamic_cast<PerspectiveCamera*>(&camera)) {
        tanHalfFov = std::tan(persp->fov * 3.14159265358979f / 360.f);
    }

    // Pack uniform buffer
    RtGpuUniforms u{};
    u.camOri[0] = camPos.x; u.camOri[1] = camPos.y; u.camOri[2] = camPos.z;
    u.camFwd[0] = fwd.x; u.camFwd[1] = fwd.y; u.camFwd[2] = fwd.z;
    u.camRgt[0] = rgt.x; u.camRgt[1] = rgt.y; u.camRgt[2] = rgt.z;
    u.camUp[0] = up.x; u.camUp[1] = up.y; u.camUp[2] = up.z;
    u.prevCamOri[0] = d.prevCamOri_[0]; u.prevCamOri[1] = d.prevCamOri_[1]; u.prevCamOri[2] = d.prevCamOri_[2];
    u.prevCamFwd[0] = d.prevCamFwd_[0]; u.prevCamFwd[1] = d.prevCamFwd_[1]; u.prevCamFwd[2] = d.prevCamFwd_[2];
    u.prevCamRgt[0] = d.prevCamRgt_[0]; u.prevCamRgt[1] = d.prevCamRgt_[1]; u.prevCamRgt[2] = d.prevCamRgt_[2];
    u.prevCamUp[0]  = d.prevCamUp_[0];  u.prevCamUp[1]  = d.prevCamUp_[1];  u.prevCamUp[2]  = d.prevCamUp_[2];
    u.iRes[0] = static_cast<float>(d.width_);
    u.iRes[1] = static_cast<float>(d.height_);
    u.tanHalfFov[0] = tanHalfFov;
    u.frameCount[0] = d.frameCount_;
    u.triCount[0] = static_cast<float>(d.triCount_);
    u.mode[0] = 1.f;  // always path tracer
    u.spp[1] = static_cast<float>(d.tileSize_);
    u.movedMeshBits[0] = movedBits[0];
    u.movedMeshBits[1] = movedBits[1];
    u.movedMeshBits[2] = movedBits[2];
    u.movedMeshBits[3] = movedBits[3];
    // Detect if any moved mesh is an emissive source — triggers tighter accum cap in shader.
    bool anyEmissiveMoved = false;
    if (movedBits[0] | movedBits[1] | movedBits[2] | movedBits[3]) {
        for (int mi = 0; mi < 128 && !anyEmissiveMoved; ++mi) {
            const uint32_t word = movedBits[mi >> 5];
            const uint32_t bit  = static_cast<uint32_t>(mi & 31);
            if ((word >> bit) & 1u) {
                anyEmissiveMoved = d.emissiveMeshSet_.count(mi) > 0;
            }
        }
    }

    // Helper: upload an equirectangular texture to a GPU texture object.
    auto uploadEquirect = [&](Texture* tex, WgpuTexture& gpuTex) {
        auto& img = tex->image();
        if (img.width <= 0 || img.height <= 0) return;
        bool isHdr = false;
        try { (void)img.data<float>(); isHdr = true; }
        catch (const std::bad_variant_access&) {}
        if (isHdr) {
            const auto& src = img.data<float>();
            gpuTex = WgpuTexture(d.renderer,
                                 static_cast<uint32_t>(img.width),
                                 static_cast<uint32_t>(img.height),
                                 WgpuTexture::Format::RGBA32Float,
                                 WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
            gpuTex.write(src.data(), src.size() * sizeof(float));
        } else {
            const auto& src = img.data<unsigned char>();
            gpuTex = WgpuTexture(d.renderer,
                                 static_cast<uint32_t>(img.width),
                                 static_cast<uint32_t>(img.height),
                                 WgpuTexture::Format::RGBA8Unorm,
                                 WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
            gpuTex.write(src.data(), src.size());
        }
    };

    // --- Environment (IBL lighting): scene.environment ---
    // Fallback: use background texture for IBL when no environment is set.
    u.envColor[3] = 0.f;  // default: no IBL
    if (auto* s = dynamic_cast<Scene*>(&scene)) {
        Texture* envTex = s->environment.get();
        if (!envTex && s->background.isTexture()) {
            envTex = s->background.texture().get();
        }
        if (envTex) {
            if (envTex != d.prevEnvTex_) {
                uploadEquirect(envTex, d.envTexGpu);
                d.rtPipeline.setTexture(9, d.envTexGpu);

                // Build env CDF for importance sampling.
                auto& img = envTex->image();
                if (img.width > 0 && img.height > 0) {
                    int w = img.width, h = img.height;
                    bool isHdr = false;
                    try { (void)img.data<float>(); isHdr = true; }
                    catch (const std::bad_variant_access&) {}
#ifdef __EMSCRIPTEN__
                    EnvCdfResult cdf;
                    if (isHdr) {
                        const float* ptr = img.data<float>().data();
                        std::vector<float> tmp(ptr, ptr + static_cast<size_t>(w) * h * 4);
                        cdf = buildEnvCdf(tmp, w, h);
                    } else {
                        const unsigned char* ptr = img.data<unsigned char>().data();
                        std::vector<unsigned char> tmp(ptr, ptr + static_cast<size_t>(w) * h * 4);
                        cdf = buildEnvCdf(tmp, w, h);
                    }
                    d.envLumSum_ = cdf.totalSum;
                    d.envCdfTex = WgpuTexture(d.renderer,
                                              static_cast<uint32_t>(cdf.width),
                                              static_cast<uint32_t>(cdf.height),
                                              WgpuTexture::Format::R32Float,
                                              WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
                    d.envCdfTex.write(cdf.conditional.data(), cdf.conditional.size() * sizeof(float));
                    d.rtPipeline.setTexture(12, d.envCdfTex);
                    d.envMargTex = WgpuTexture(d.renderer,
                                               static_cast<uint32_t>(cdf.height), 1u,
                                               WgpuTexture::Format::R32Float,
                                               WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
                    d.envMargTex.write(cdf.marginal.data(), cdf.marginal.size() * sizeof(float));
                    d.rtPipeline.setTexture(13, d.envMargTex);
                    if (!d.shaderHasEnvCdf_) {
                        d.rtPipeline.replaceShader(buildRtShader(true));
                        d.shaderHasEnvCdf_ = true;
                    }
#else
                    if (isHdr) {
                        const float* ptr = img.data<float>().data();
                        d.asyncEnvCdf_ = std::async(std::launch::async, [ptr, w, h]() {
                            std::vector<float> tmp(ptr, ptr + static_cast<size_t>(w) * h * 4);
                            return buildEnvCdf(tmp, w, h);
                        });
                    } else {
                        const unsigned char* ptr = img.data<unsigned char>().data();
                        d.asyncEnvCdf_ = std::async(std::launch::async, [ptr, w, h]() {
                            std::vector<unsigned char> tmp(ptr, ptr + static_cast<size_t>(w) * h * 4);
                            return buildEnvCdf(tmp, w, h);
                        });
                    }
                    d.envCdfPending_ = true;
#endif
                }
                d.prevEnvTex_ = envTex;
            }
            u.envColor[3] = 2.f;
        } else {
            // No environment = no IBL; clear CDF
            if (d.prevEnvTex_) {
                d.prevEnvTex_ = nullptr;
                d.envLumSum_ = 0.f;
                if (d.shaderHasEnvCdf_) {
                    d.rtPipeline.replaceShader(buildRtShader(false));
                    d.shaderHasEnvCdf_ = false;
                }
            }
        }

        // --- Background (ray miss): scene.background ---
        u.bgColor[3] = 0.f;  // default: procedural sky gradient
        if (s->background.isTexture()) {
            Texture* bgTex = s->background.texture().get();
            if (bgTex != d.prevBgTex_) {
                uploadEquirect(bgTex, d.bgTexGpu);
                d.rtPipeline.setTexture(16, d.bgTexGpu);
                d.prevBgTex_ = bgTex;
            }
            u.bgColor[3] = 2.f;
        } else if (s->background.isColor()) {
            const Color& c = s->background.color();
            u.bgColor[0] = c.r; u.bgColor[1] = c.g; u.bgColor[2] = c.b;
            u.bgColor[3] = 1.f;
            d.prevBgTex_ = nullptr;
        }

    }
#ifndef __EMSCRIPTEN__
    // Check if async CDF build finished — upload to GPU
    if (d.envCdfPending_ && d.asyncEnvCdf_.valid() &&
        d.asyncEnvCdf_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        auto cdf = d.asyncEnvCdf_.get();
        d.envCdfPending_ = false;
        d.envLumSum_ = cdf.totalSum;
        d.envCdfTex = WgpuTexture(d.renderer,
                                  static_cast<uint32_t>(cdf.width),
                                  static_cast<uint32_t>(cdf.height),
                                  WgpuTexture::Format::R32Float,
                                  WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
        d.envCdfTex.write(cdf.conditional.data(), cdf.conditional.size() * sizeof(float));
        d.rtPipeline.setTexture(12, d.envCdfTex);
        d.envMargTex = WgpuTexture(d.renderer,
                                   static_cast<uint32_t>(cdf.height), 1u,
                                   WgpuTexture::Format::R32Float,
                                   WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
        d.envMargTex.write(cdf.marginal.data(), cdf.marginal.size() * sizeof(float));
        d.rtPipeline.setTexture(13, d.envMargTex);
        // Swap to env CDF shader variant
        if (!d.shaderHasEnvCdf_) {
            d.rtPipeline.replaceShader(buildRtShader(true));
            d.shaderHasEnvCdf_ = true;
        }
    }
#endif

    u.envIntensity[0] = d.envIntensity_;
    // Pass envmap dimensions and total luminance sum for importance sampling
    if (d.envLumSum_ > 0.f && d.prevEnvTex_) {
        auto& eImg = d.prevEnvTex_->image();
        u.envIntensity[1] = static_cast<float>(eImg.width);
        u.envIntensity[2] = static_cast<float>(eImg.height);
        u.envIntensity[3] = d.envLumSum_;
    } else {
        u.envIntensity[1] = 0.f;
        u.envIntensity[2] = 0.f;
        u.envIntensity[3] = 0.f;
    }
    u.params[0] = static_cast<float>(d.maxBounces_);
    u.params[1] = static_cast<float>(d.globalFrameCounter_++);
    u.params[2] = (d.foveatedEnabled_ && d.frameCount_ > 0.f) ? 1.f : 0.f;
    u.params[3] = (d.frameCount_ == 0.f) ? 1.f : 0.f;  // force-reset accum on first frame
    u.emissiveInfo[0] = static_cast<float>(d.emissiveTriCount_);
    u.emissiveInfo[1] = d.emissiveTotalPower_;
    u.emissiveInfo[2] = d.fireflyCap_;  // luminance cap for indirect MIS contributions
    u.emissiveInfo[3] = static_cast<float>(d.aovMode_);  // AOV visualization mode
    u.spp[0] = d.restirGiEnabled_ ? 1.f : 0.f;
    u.spp[2] = (d.denoiserEnabled_ && !d.temporalDenoiserEnabled_) ? 1.f : 0.f;  // EMA spatial denoiser active
    u.restirParams[0] = d.restirEnabled_ ? 1.f : 0.f;
    u.restirParams[1] = 20.f;  // M clamp — lower = crisper shadows, higher = lower variance
    u.restirParams[2] = anyEmissiveMoved ? 1.f : 0.f;  // emissive source moved → tight accum cap
    u.restirParams[3] = d.temporalDenoiserEnabled_ ? 1.f : 0.f;  // 1 = raw 1-spp output (no EMA)

    int nLights = 0;
    auto packLight = [&](float px, float py, float pz, float r, float g, float b, float type) {
        if (nLights >= 4) return;
        u.lightPos[nLights][0] = px; u.lightPos[nLights][1] = py; u.lightPos[nLights][2] = pz;
        u.lightCol[nLights][0] = r;  u.lightCol[nLights][1] = g;  u.lightCol[nLights][2] = b;
        u.lightType[nLights][0] = type;
        ++nLights;
    };
    for (auto* l : pointLights) {
        if (!l->visible) continue;
        const auto& lp = l->position;
        const auto& lc = l->color;
        const float li = l->intensity;
        packLight(lp.x, lp.y, lp.z, lc.r * li, lc.g * li, lc.b * li, 0.f);
        u.lightType[nLights - 1][3] = l->distance;
        u.lightDir[nLights - 1][3] = l->decay;
    }
    for (auto* l : dirLights) {
        if (!l->visible) continue;
        Vector3 dir = Vector3(l->position).sub(l->target().position).normalize();
        const auto& lc = l->color;
        const float li = l->intensity;
        packLight(dir.x, dir.y, dir.z, lc.r * li, lc.g * li, lc.b * li, 1.f);
    }
    for (auto* l : spotLights) {
        if (nLights >= 4 || !l->visible) break;
        const auto& lp = l->position;
        const auto& lc = l->color;
        const float li = l->intensity;
        Vector3 dir = Vector3(l->target().position).sub(lp).normalize();
        const float cosAngle = std::cos(l->angle);
        const float cosOuter = std::cos(l->angle * (1.f + l->penumbra));
        u.lightPos[nLights][0] = lp.x; u.lightPos[nLights][1] = lp.y; u.lightPos[nLights][2] = lp.z;
        u.lightCol[nLights][0] = lc.r * li; u.lightCol[nLights][1] = lc.g * li; u.lightCol[nLights][2] = lc.b * li;
        u.lightType[nLights][0] = 2.f;
        u.lightType[nLights][1] = cosAngle;
        u.lightType[nLights][2] = cosOuter;
        u.lightType[nLights][3] = l->distance;
        u.lightDir[nLights][0] = dir.x; u.lightDir[nLights][1] = dir.y; u.lightDir[nLights][2] = dir.z;
        u.lightDir[nLights][3] = l->decay;
        ++nLights;
    }
    u.lightCount[0] = static_cast<float>(nLights);
    d.rtUniformBuf.write(&u, sizeof(u));

    // Set per-frame texture bindings (accum + hitMesh ping-pong)
    auto& activePipeline = d.rtPipeline;

    // Skip RT dispatch if the active pipeline is still compiling asynchronously.
    // On cold shader cache (first launch / code change) this can take 10-30 seconds.
    if (!activePipeline.isReady()) {
        ++d.shaderWaitFrames_;
        if (d.shaderWaitFrames_ == 1) {
            std::cerr << "[PathTracer] Compiling RT shaders — first launch may take 10-30s "
                         "(subsequent launches are instant via driver cache)." << std::endl;
        } else if (d.shaderWaitFrames_ % 180 == 0) {
            std::cerr << "[PathTracer] Still compiling shaders... ("
                      << (d.shaderWaitFrames_ / 60) << "s elapsed)" << std::endl;
        }
        // Tick the device so backends that need main-thread event processing make progress.
        wgpuDevicePoll(d.device, false, nullptr);
        // Safety net: after ~60 seconds still black, force-block until async finishes.
        // This prevents a permanent black screen if something prevents the future from
        // being polled to completion in the normal render loop.
        if (d.shaderWaitFrames_ >= 3600) {
            std::cerr << "[PathTracer] 60s timeout — force-completing shader compilation." << std::endl;
            activePipeline.forceFinishBuild();
            d.shaderWaitFrames_ = 0;
            // Fall through: isReady() now returns true; encode() will do the fast sync rebuild.
        } else {
            d.displayMat->customTextures["accumTex"] = d.readAccum;
            d.displayMat->customTextures["gBufTex"]  = d.gBufPrev;
            d.displayMat->uniformsNeedUpdate = true;
            d.renderer.render(d.displayScene, d.displayCam);
            return;
        }
    }
    if (d.shaderWaitFrames_ > 0) {
        std::cerr << "[PathTracer] RT shaders ready after "
                  << (d.shaderWaitFrames_ / 60) << "s — rendering started." << std::endl;
        d.shaderWaitFrames_ = 0;
    }
    // First dispatch after async shader compilation: triTex has never been written by the VT
    // pass (every prior frame returned early).  Force the VT pass to run now so the RT shader
    // sees valid world-space triangle data, and reset accumulation for a clean start.
    if (d.firstDispatchPending_) {
        d.firstDispatchPending_ = false;
        anyMeshMoved = true;
        movedBits[0] = movedBits[1] = movedBits[2] = movedBits[3] = 0xFFFFFFFFu;
        d.frameCount_ = 0.f;
    }

    const uint32_t gx = (static_cast<uint32_t>(d.width_) + 7u) / 8u;
    const uint32_t gy = (static_cast<uint32_t>(d.height_) + 7u) / 8u;

    for (int sampleIdx = 0; sampleIdx < d.spp_; ++sampleIdx) {

        // Update frame count in uniforms for each sample (different random seed)
        if (sampleIdx > 0) {
            d.frameCount_ += 1.f;
            u.frameCount[0] = d.frameCount_;
            d.rtUniformBuf.write(&u, sizeof(u));
        }

        activePipeline.setTexture(1, *d.readAccum);
        activePipeline.setStorageTexture(2, *d.writeAccum);
        activePipeline.setTexture(7, *d.readHitMesh);
        activePipeline.setStorageTexture(8, *d.writeHitMesh);

        // GPU dispatch
        {
            WGPUCommandEncoderDescriptor encDesc{};
            encDesc.label = WGPUStringView{"pt_enc", WGPU_STRLEN};
            WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(d.device, &encDesc);

            WGPUComputePassDescriptor passDesc{};

            // VT + BVH refit only on first sample (geometry doesn't change between samples)
            if (sampleIdx == 0 && anyMeshMoved) {
                passDesc.label = WGPUStringView{"vt_pass", WGPU_STRLEN};
                WGPUComputePassEncoder vtPass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
                d.vtPipeline.encode(vtPass, d.vtDispatchX_, d.vtDispatchY_);
                wgpuComputePassEncoderEnd(vtPass);
                wgpuComputePassEncoderRelease(vtPass);

                if (!topoJustFinished) {
                    passDesc.label = WGPUStringView{"rf_pass", WGPU_STRLEN};
                    WGPUComputePassEncoder rfPass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
                    d.refitPipeline.encode(rfPass, d.rfDispatchX_, d.rfDispatchY_);
                    wgpuComputePassEncoderEnd(rfPass);
                    wgpuComputePassEncoderRelease(rfPass);
                }
            }

            // Ray trace
            passDesc.label = WGPUStringView{"rt_pass", WGPU_STRLEN};
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);
            activePipeline.encode(pass, gx, gy);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);

            WGPUCommandBufferDescriptor cmdDesc{};
            cmdDesc.label = WGPUStringView{"pt_cmd", WGPU_STRLEN};
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(encoder, &cmdDesc);
            wgpuQueueSubmit(d.queue, 1, &cmd);

            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(encoder);
        }

        // Swap all ping-pong buffers so next sample reads this sample's output
        std::swap(d.readAccum, d.writeAccum);
        std::swap(d.readHitMesh, d.writeHitMesh);
        std::swap(d.readDiffAccum, d.writeDiffAccum);
        std::swap(d.readSpecAccum, d.writeSpecAccum);

        std::swap(d.gBufCur, d.gBufPrev);
        d.rtPipeline.setStorageTexture(10, *d.gBufCur);
        d.rtPipeline.setTexture(15, *d.gBufPrev);

        std::swap(d.stableGBufCur, d.stableGBufPrev);
        d.rtPipeline.setStorageTexture(27, *d.stableGBufCur);
        d.rtPipeline.setTexture(28, *d.stableGBufPrev);

        std::swap(d.reservoirRead, d.reservoirWrite);
        std::swap(d.reservoirWRead, d.reservoirWWrite);
        d.rtPipeline.setTexture(17, *d.reservoirRead);
        d.rtPipeline.setStorageTexture(18, *d.reservoirWrite);
        d.rtPipeline.setTexture(19, *d.reservoirWRead);
        d.rtPipeline.setStorageTexture(20, *d.reservoirWWrite);

        std::swap(d.giResRead, d.giResWrite);
        std::swap(d.giResWRead, d.giResWWrite);
        std::swap(d.giResLoRead, d.giResLoWrite);
        d.rtPipeline.setTexture(30, *d.giResRead);
        d.rtPipeline.setStorageTexture(31, *d.giResWrite);
        d.rtPipeline.setTexture(32, *d.giResWRead);
        d.rtPipeline.setStorageTexture(33, *d.giResWWrite);
        d.rtPipeline.setTexture(34, *d.giResLoRead);
        d.rtPipeline.setStorageTexture(35, *d.giResLoWrite);

        std::swap(d.momentsRead, d.momentsWrite);
        d.rtPipeline.setTexture(21, *d.momentsRead);
        d.rtPipeline.setStorageTexture(22, *d.momentsWrite);
        d.rtPipeline.setTexture(23, *d.readDiffAccum);
        d.rtPipeline.setStorageTexture(24, *d.writeDiffAccum);
        d.rtPipeline.setTexture(25, *d.readSpecAccum);
        d.rtPipeline.setStorageTexture(26, *d.writeSpecAccum);
    }

    // Update denoiser's moments reference (reads converged moments)
    d.atrousPipeline.setTexture(6, *d.momentsRead);

    // Spatial denoiser (path tracer mode only) — à-trous wavelet filter on
    // split diffuse/specular channels independently. Each channel gets its own
    // mode (diffuse=0, specular=1) with roughness-adaptive edge-stopping.
    // 3 cascade passes with step sizes 1, 2, 4 give effective 25-pixel radius.
    WgpuTexture* displayTex = d.readAccum;
    WgpuTexture* displayDiff = d.readDiffAccum;
    WgpuTexture* displaySpec = d.readSpecAccum;
    const bool hasMotion = (movedBits[0] | movedBits[1] | movedBits[2] | movedBits[3]) != 0u;
    const bool needsDenoise = (d.frameCount_ < 64.f || hasMotion) && d.aovMode_ == 0;
    if (d.denoiserEnabled_ && needsDenoise && !d.temporalDenoiserEnabled_) {
        const uint32_t gx = (static_cast<uint32_t>(d.width_)  + 7u) / 8u;
        const uint32_t gy = (static_cast<uint32_t>(d.height_) + 7u) / 8u;

        AtrousGpuUniforms au{};
        au.frameCount = d.frameCount_;

        // Helper: run N-pass à-trous cascade on a channel.
        // Uses denoisedXxx and filteredX as ping-pong targets.
        // Input: srcAccum (raw accumulation), output lands in dstDenoised.
        auto runAtrous = [&](WgpuTexture& srcAccum, WgpuTexture& dstDenoised,
                             WgpuTexture& tmpFiltered, uint32_t mode, int passes) {
            au.mode = mode;
            d.atrousPipeline.setTexture(3, *d.gBufPrev);
            d.atrousPipeline.setTexture(5, *d.readHitMesh);
            d.atrousPipeline.setTexture(7, *d.stableGBufPrev);

            WgpuTexture* readTex  = &srcAccum;
            for (int p = 0; p < passes; ++p) {
                WgpuTexture* writeTex = (p % 2 == 0) ? &dstDenoised : &tmpFiltered;
                au.stepSize = static_cast<uint32_t>(1 << p);
                d.atrousUniBuf.write(&au, sizeof(au));
                d.atrousPipeline.setTexture(1, *readTex);
                d.atrousPipeline.setStorageTexture(2, *writeTex);
                d.atrousPipeline.dispatch(gx, gy);
                readTex = writeTex;
            }
        };

        runAtrous(*d.readDiffAccum, d.denoisedDiff, d.filteredA, 0, 2);
        runAtrous(*d.readSpecAccum, d.denoisedSpec, d.filteredB, 1, 2);

        displayDiff = &d.denoisedDiff;
        displaySpec = &d.denoisedSpec;
    } else if (d.temporalDenoiserEnabled_ && d.denoiserEnabled_ && d.aovMode_ == 0) {
        // New pipeline: 1-spp → pre-filter → temporal → spatial cleanup
        const uint32_t gx = (static_cast<uint32_t>(d.width_)  + 7u) / 8u;
        const uint32_t gy = (static_cast<uint32_t>(d.height_) + 7u) / 8u;

        // --- Phase 3: Temporal accumulation ---
        {
            TaaGpuUniforms tu{};
            tu.prevCamOri[0] = d.prevCamOri_[0]; tu.prevCamOri[1] = d.prevCamOri_[1]; tu.prevCamOri[2] = d.prevCamOri_[2];
            tu.prevCamFwd[0] = d.prevCamFwd_[0]; tu.prevCamFwd[1] = d.prevCamFwd_[1]; tu.prevCamFwd[2] = d.prevCamFwd_[2];
            tu.prevCamRgt[0] = d.prevCamRgt_[0]; tu.prevCamRgt[1] = d.prevCamRgt_[1]; tu.prevCamRgt[2] = d.prevCamRgt_[2];
            tu.prevCamUp[0]  = d.prevCamUp_[0];  tu.prevCamUp[1]  = d.prevCamUp_[1];  tu.prevCamUp[2]  = d.prevCamUp_[2];
            tu.curCamOri[0] = camPos.x; tu.curCamOri[1] = camPos.y; tu.curCamOri[2] = camPos.z;
            tu.curCamFwd[0] = fwd.x; tu.curCamFwd[1] = fwd.y; tu.curCamFwd[2] = fwd.z;
            tu.curCamRgt[0] = rgt.x; tu.curCamRgt[1] = rgt.y; tu.curCamRgt[2] = rgt.z;
            tu.curCamUp[0]  = up.x;  tu.curCamUp[1]  = up.y;  tu.curCamUp[2]  = up.z;
            tu.iRes[0] = static_cast<float>(d.width_);
            tu.iRes[1] = static_cast<float>(d.height_);
            tu.tanHalfFov[0] = tanHalfFov;
            tu.frameCount[0] = d.frameCount_;
            tu.movedMeshBits[0] = movedBits[0];
            tu.movedMeshBits[1] = movedBits[1];
            tu.movedMeshBits[2] = movedBits[2];
            tu.movedMeshBits[3] = movedBits[3];

            // Diffuse temporal: raw → taaHistDiffWrite, preFiltered = filteredA
            tu.frameCount[1] = 0.f;  // mode = diffuse
            d.taaUniBuf.write(&tu, sizeof(tu));
            d.taaPipeline.setTexture(1, *d.readDiffAccum);       // raw 1-spp
            d.taaPipeline.setTexture(2, *d.gBufPrev);            // current frame g-buffer (after swap: gBufPrev = just-written)
            d.taaPipeline.setTexture(3, *d.taaHistDiffRead);     // prev temporal history
            d.taaPipeline.setStorageTexture(4, *d.taaHistDiffWrite);
            d.taaPipeline.setTexture(5, *d.readHitMesh);
            d.taaPipeline.setTexture(8, *d.momentsRead);
            d.taaPipeline.setTexture(9, *d.gBufCur);             // previous frame g-buffer (after swap: gBufCur = prev frame)
            d.taaPipeline.setTexture(10, *d.stableGBufPrev);    // current frame stable g-buffer
            d.taaPipeline.setTexture(11, *d.stableGBufCur);     // previous frame stable g-buffer
            d.taaPipeline.dispatch(gx, gy);

            // Specular temporal: raw → taaHistSpecWrite
            tu.frameCount[1] = 1.f;  // mode = specular
            d.taaUniBuf.write(&tu, sizeof(tu));
            d.taaPipeline.setTexture(1, *d.readSpecAccum);
            d.taaPipeline.setTexture(3, *d.taaHistSpecRead);
            d.taaPipeline.setStorageTexture(4, *d.taaHistSpecWrite);
            d.taaPipeline.setTexture(8, *d.momentsRead);
            d.taaPipeline.dispatch(gx, gy);

            // Swap temporal history ping-pong
            std::swap(d.taaHistDiffRead, d.taaHistDiffWrite);
            std::swap(d.taaHistSpecRead, d.taaHistSpecWrite);
        }

        // --- Phase 4: Spatial cleanup on temporal output ---
        AtrousGpuUniforms au{};
        au.frameCount = d.frameCount_;

        auto runAtrous = [&](WgpuTexture& srcAccum, WgpuTexture& dstDenoised,
                             WgpuTexture& tmpFiltered, uint32_t mode, int passes) {
            au.mode = mode;
            d.atrousPipeline.setTexture(3, *d.gBufPrev);
            d.atrousPipeline.setTexture(5, *d.readHitMesh);
            d.atrousPipeline.setTexture(7, *d.stableGBufPrev);
            WgpuTexture* readTex = &srcAccum;
            for (int p = 0; p < passes; ++p) {
                WgpuTexture* writeTex = (p % 2 == 0) ? &dstDenoised : &tmpFiltered;
                au.stepSize = static_cast<uint32_t>(1 << p);
                d.atrousUniBuf.write(&au, sizeof(au));
                d.atrousPipeline.setTexture(1, *readTex);
                d.atrousPipeline.setStorageTexture(2, *writeTex);
                d.atrousPipeline.dispatch(gx, gy);
                readTex = writeTex;
            }
        };

        // Cleanup reads temporal output (taaHistDiffRead/SpecRead after swap = just-written).
        // Uses filteredA/B as ping-pong temp — safe, temporal already consumed them.
        runAtrous(*d.taaHistDiffRead, d.denoisedDiff, d.filteredA, 2, 2);  // 2 = diffuse | temporal
        runAtrous(*d.taaHistSpecRead, d.denoisedSpec, d.filteredB, 3, 2);  // 3 = specular | temporal

        displayDiff = &d.denoisedDiff;
        displaySpec = &d.denoisedSpec;
    }

    // --- TAAU: temporal upscale (active when pixelScale < 0.65, i.e. ≥1.5× upscale) ---
    if (d.pixelScale_ < 0.65f) {
        UpscaleGpuUniforms& uu = d.upscaleUBO;
        uu.prevCamOri[0] = d.prevCamOri_[0]; uu.prevCamOri[1] = d.prevCamOri_[1]; uu.prevCamOri[2] = d.prevCamOri_[2]; uu.prevCamOri[3] = 0.f;
        uu.prevCamFwd[0] = d.prevCamFwd_[0]; uu.prevCamFwd[1] = d.prevCamFwd_[1]; uu.prevCamFwd[2] = d.prevCamFwd_[2]; uu.prevCamFwd[3] = 0.f;
        uu.prevCamRgt[0] = d.prevCamRgt_[0]; uu.prevCamRgt[1] = d.prevCamRgt_[1]; uu.prevCamRgt[2] = d.prevCamRgt_[2]; uu.prevCamRgt[3] = 0.f;
        uu.prevCamUp[0]  = d.prevCamUp_[0];  uu.prevCamUp[1]  = d.prevCamUp_[1];  uu.prevCamUp[2]  = d.prevCamUp_[2];  uu.prevCamUp[3]  = 0.f;
        uu.curCamOri[0] = camPos.x; uu.curCamOri[1] = camPos.y; uu.curCamOri[2] = camPos.z; uu.curCamOri[3] = 0.f;
        uu.curCamFwd[0] = fwd.x;    uu.curCamFwd[1] = fwd.y;    uu.curCamFwd[2] = fwd.z;    uu.curCamFwd[3] = 0.f;
        uu.curCamRgt[0] = rgt.x;    uu.curCamRgt[1] = rgt.y;    uu.curCamRgt[2] = rgt.z;    uu.curCamRgt[3] = 0.f;
        uu.curCamUp[0]  = up.x;     uu.curCamUp[1]  = up.y;     uu.curCamUp[2]  = up.z;     uu.curCamUp[3]  = 0.f;
        uu.iRes[0] = static_cast<float>(d.fullWidth_);
        uu.iRes[1] = static_cast<float>(d.fullHeight_);
        uu.iRes[2] = d.pixelScale_;
        uu.iRes[3] = 0.f;
        uu.tanHalfFov[0] = tanHalfFov;
        uu.tanHalfFov[1] = uu.tanHalfFov[2] = uu.tanHalfFov[3] = 0.f;
        uu.frameCount[0] = d.frameCount_;
        uu.frameCount[1] = uu.frameCount[2] = uu.frameCount[3] = 0.f;
        d.upscaleUniBuf.write(&uu, sizeof(uu));

        d.upscalePipeline.setTexture(1, *displayDiff);
        d.upscalePipeline.setTexture(2, *displaySpec);
        d.upscalePipeline.setTexture(3, *d.gBufPrev);
        d.upscalePipeline.setTexture(4, *d.upscaleRead);
        d.upscalePipeline.setStorageTexture(5, *d.upscaleWrite);

        const int ugx = (d.fullWidth_  + 7) / 8;
        const int ugy = (d.fullHeight_ + 7) / 8;
        d.upscalePipeline.dispatch(ugx, ugy, 1);
        std::swap(d.upscaleRead, d.upscaleWrite);

        d.displayMat->customTextures["upscaleTex"] = d.upscaleRead;
    } else {
        d.displayMat->customTextures["upscaleTex"] = &d.zeroTex;
    }

    // Store camera for next frame's reprojection (must run every frame, not just when denoiser is active)
    d.prevCamOri_[0] = camPos.x; d.prevCamOri_[1] = camPos.y; d.prevCamOri_[2] = camPos.z;
    d.prevCamFwd_[0] = fwd.x;    d.prevCamFwd_[1] = fwd.y;    d.prevCamFwd_[2] = fwd.z;
    d.prevCamRgt_[0] = rgt.x;    d.prevCamRgt_[1] = rgt.y;    d.prevCamRgt_[2] = rgt.z;
    d.prevCamUp_[0]  = up.x;     d.prevCamUp_[1]  = up.y;     d.prevCamUp_[2]  = up.z;

    d.displayMat->customTextures["accumTex"] = displayTex;
    d.displayMat->customTextures["gBufTex"]  = d.gBufPrev;
    d.displayMat->customTextures["diffTex"]  = displayDiff;
    d.displayMat->customTextures["specTex"]  = displaySpec;
    d.displayMat->uniformsNeedUpdate = true;
    d.frameCount_ += 1.f;

    // Pass exposure via camera z (transform.cameraPos.z); always positive so the display quad
    // stays in front of the near plane.  AOV mode is encoded in _pad: _pad = pixelScale + aovMode*10,
    // allowing the display shader to extract both pixelScale and aovMode from a single float.
    d.displayCam.position.z = d.exposure_;
    d.renderer.setPixelRatioHint(d.pixelScale_ + static_cast<float>(d.aovMode_) * 10.f);

    // Blit to screen
    d.renderer.render(d.displayScene, d.displayCam);

    // Raster overlay: draw wireframe meshes and Line geometry on top.
    // Collect overlay objects and temporarily hide everything else so the
    // renderer only draws the overlay. Depth is cleared so overlays render
    // without being occluded by the blit quad's depth writes; color is loaded
    // so the path-traced image is preserved.
    {
        struct Entry { Object3D* obj; bool wasVisible; };
        std::vector<Entry> hidden;
        bool hasOverlay = false;

        // Only toggle renderable objects (Mesh / Line), never containers/groups,
        // so parent nodes remain visible and their children are reachable.
        scene.traverse([&](Object3D& obj) {
            if (!obj.visible) return;
            const bool onOverlayLayer = (d.overlayLayer_ >= 0 &&
                                         obj.layers.isEnabled(static_cast<unsigned>(d.overlayLayer_)));
            if (auto* mesh = obj.as<Mesh>()) {
                auto* mat = mesh->material().get();
                auto* mww = dynamic_cast<MaterialWithWireframe*>(mat);
                bool isOverlay = onOverlayLayer ||
                                 (mww && mww->wireframe) ||
                                 dynamic_cast<LineBasicMaterial*>(mat) != nullptr;
                if (isOverlay) {
                    hasOverlay = true;
                } else {
                    hidden.push_back({mesh, true});
                    mesh->visible = false;
                }
            } else if (obj.is<Line>() || onOverlayLayer) {
                hasOverlay = true;
            }
        });

        if (hasOverlay) {
            // Reconstruct rasterizer depth from path-traced primary-hit t values.
            // This lets wireframe/line objects be correctly occluded by path-traced geometry.
            auto* encoder = static_cast<WGPUCommandEncoder>(d.renderer.nativeRenderCommandEncoder());
            auto* depthView = static_cast<WGPUTextureView>(d.renderer.nativeFrameDepthView());
            const uint32_t depthSamples = d.renderer.nativeFrameDepthSampleCount();
            d.ensureDepthFillPipeline(depthSamples);
            if (encoder && depthView && d.depthFillPipeline_) {
                // Upload depth-fill uniforms (projView + camera vectors)
                DepthFillUniforms dfu{};
                {
                    Matrix4 proj = camera.projectionMatrix;
                    // Apply the same NDC z remap as WgpuRenderer::render()
                    auto& e = proj.elements;
                    e[2]  = 0.5f * e[2]  + 0.5f * e[3];
                    e[6]  = 0.5f * e[6]  + 0.5f * e[7];
                    e[10] = 0.5f * e[10] + 0.5f * e[11];
                    e[14] = 0.5f * e[14] + 0.5f * e[15];
                    Matrix4 pv;
                    pv.multiplyMatrices(proj, camera.matrixWorldInverse);
                    const auto& pe = pv.elements;
                    for (int i = 0; i < 16; ++i) dfu.projView[i] = pe[i];
                }
                dfu.camOri[0] = camPos.x; dfu.camOri[1] = camPos.y; dfu.camOri[2] = camPos.z;
                dfu.camFwd[0] = fwd.x;    dfu.camFwd[1] = fwd.y;    dfu.camFwd[2] = fwd.z;
                dfu.camRgt[0] = rgt.x;    dfu.camRgt[1] = rgt.y;    dfu.camRgt[2] = rgt.z;
                dfu.camUp[0]  = up.x;     dfu.camUp[1]  = up.y;     dfu.camUp[2]  = up.z;
                dfu.iRes[0]   = static_cast<float>(d.width_);
                dfu.iRes[1]   = static_cast<float>(d.height_);
                dfu.tanHalfFov[0] = tanHalfFov;
                wgpuQueueWriteBuffer(d.queue, d.depthFillUniBuf_, 0, &dfu, sizeof(dfu));

                // Build bind group: uniform (0) + gBufPrev (1)
                WGPUBindGroupEntry bgEntries[2]{};
                bgEntries[0].binding = 0;
                bgEntries[0].buffer  = d.depthFillUniBuf_;
                bgEntries[0].offset  = 0;
                bgEntries[0].size    = sizeof(DepthFillUniforms);
                bgEntries[1].binding    = 1;
                bgEntries[1].textureView = d.gBufPrev->view();
                WGPUBindGroupDescriptor bgDesc{};
                bgDesc.layout     = d.depthFillBGL_;
                bgDesc.entryCount = 2;
                bgDesc.entries    = bgEntries;
                WGPUBindGroup bg = wgpuDeviceCreateBindGroup(d.device, &bgDesc);

                // Depth-fill render pass: loads existing depth, then overwrites with
                // reconstructed path-traced depth. No color attachment.
                WGPURenderPassDepthStencilAttachment depthAtt{};
                depthAtt.view            = depthView;
                depthAtt.depthLoadOp     = WGPULoadOp_Clear;
                depthAtt.depthStoreOp    = WGPUStoreOp_Store;
                depthAtt.depthClearValue = 1.0f;
                WGPURenderPassDescriptor passDesc{};
                passDesc.label                    = WGPUStringView{"depth_fill_pass", WGPU_STRLEN};
                passDesc.colorAttachmentCount     = 0;
                passDesc.depthStencilAttachment   = &depthAtt;
                WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &passDesc);
                wgpuRenderPassEncoderSetPipeline(pass, d.depthFillPipeline_);
                wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
                wgpuRenderPassEncoderSetViewport(pass, 0.f, 0.f,
                    static_cast<float>(d.width_), static_cast<float>(d.height_), 0.f, 1.f);
                wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
                wgpuRenderPassEncoderEnd(pass);
                wgpuRenderPassEncoderRelease(pass);
                wgpuBindGroupRelease(bg);
            }

            // Overlay render: preserve color AND depth (use reconstructed depth for occlusion).
            const bool savedAutoClear       = d.renderer.autoClear;
            const bool savedShadowAutoUpdate = d.renderer.shadowMapAutoUpdate;
            d.renderer.autoClear           = false;  // no color or depth clear
            d.renderer.shadowMapAutoUpdate = false;  // skip shadow re-render
            d.renderer.render(scene, camera);
            d.renderer.autoClear           = savedAutoClear;
            d.renderer.shadowMapAutoUpdate = savedShadowAutoUpdate;
        }

        for (auto& e : hidden) e.obj->visible = e.wasVisible;
    }
}

void WgpuPathTracer::setEnvIntensity(float intensity) {
    if (intensity != pimpl_->envIntensity_) {
        pimpl_->envIntensity_ = intensity;
        pimpl_->frameCount_ = 0.f;
    }
}

float WgpuPathTracer::envIntensity() const {
    return pimpl_->envIntensity_;
}

void WgpuPathTracer::setMaxBounces(int bounces) {
    bounces = std::max(1, std::min(bounces, 16));
    if (pimpl_->maxBounces_ != bounces) {
        pimpl_->maxBounces_ = bounces;
        pimpl_->frameCount_ = 0.f;
    }
}

int WgpuPathTracer::maxBounces() const {
    return pimpl_->maxBounces_;
}

void WgpuPathTracer::setExposure(float exposure) {
    pimpl_->exposure_ = exposure;
}

float WgpuPathTracer::exposure() const {
    return pimpl_->exposure_;
}

void WgpuPathTracer::setDenoiserEnabled(bool enabled) {
    pimpl_->denoiserEnabled_ = enabled;
}

bool WgpuPathTracer::denoiserEnabled() const {
    return pimpl_->denoiserEnabled_;
}

void WgpuPathTracer::setTemporalDenoiser(bool enabled) {
    pimpl_->temporalDenoiserEnabled_ = enabled;
}

bool WgpuPathTracer::temporalDenoiser() const {
    return pimpl_->temporalDenoiserEnabled_;
}

void WgpuPathTracer::setReSTIREnabled(bool enabled) {
    pimpl_->frameCount_ = 0.f;  // flush stale reservoir before first temporal reuse
    pimpl_->restirEnabled_ = enabled;
}

bool WgpuPathTracer::restirEnabled() const {
    return pimpl_->restirEnabled_;
}

void WgpuPathTracer::setReSTIRGIEnabled(bool enabled) {
    pimpl_->frameCount_ = 0.f;
    pimpl_->restirGiEnabled_ = enabled;
}

bool WgpuPathTracer::restirGiEnabled() const {
    return pimpl_->restirGiEnabled_;
}

void WgpuPathTracer::setSamplesPerPixel(int spp) {
    pimpl_->spp_ = std::max(1, spp);
}

int WgpuPathTracer::samplesPerPixel() const {
    return pimpl_->spp_;
}

void WgpuPathTracer::setFireflyClamp(float cap) {
    pimpl_->fireflyCap_ = (cap > 0.f) ? cap : 1e30f;
}

float WgpuPathTracer::fireflyClamp() const {
    return pimpl_->fireflyCap_;
}

void WgpuPathTracer::setAOVMode(int mode) {
    mode = std::max(0, std::min(mode, 5));
    if (pimpl_->aovMode_ != mode) {
        pimpl_->aovMode_ = mode;
        pimpl_->frameCount_ = 0.f;
    }
}

int WgpuPathTracer::aovMode() const {
    return pimpl_->aovMode_;
}

void WgpuPathTracer::setSize(std::pair<int, int> size) {
    pimpl_->fullWidth_ = size.first;
    pimpl_->fullHeight_ = size.second;
    const int sw = std::max(1, static_cast<int>(size.first * pimpl_->pixelScale_));
    const int sh = std::max(1, static_cast<int>(size.second * pimpl_->pixelScale_));
    pimpl_->recreateAccumTextures(sw, sh);
    pimpl_->recreateUpscaleTextures(size.first, size.second);
}

std::pair<int, int> WgpuPathTracer::size() const {
    return {pimpl_->fullWidth_, pimpl_->fullHeight_};
}

void WgpuPathTracer::setPixelScale(float scale) {
    scale = std::clamp(scale, 0.1f, 2.0f);
    if (scale == pimpl_->pixelScale_) return;
    pimpl_->pixelScale_ = scale;
    // Set renderer hint so the display shader reads pixelScale from transform._pad
    pimpl_->renderer.setPixelRatioHint(scale);
    if (pimpl_->fullWidth_ > 0 && pimpl_->fullHeight_ > 0) {
        const int sw = std::max(1, static_cast<int>(pimpl_->fullWidth_ * scale));
        const int sh = std::max(1, static_cast<int>(pimpl_->fullHeight_ * scale));
        pimpl_->recreateAccumTextures(sw, sh);
        pimpl_->resetUpscaleHistory();
    }
}

float WgpuPathTracer::pixelScale() const {
    return pimpl_->pixelScale_;
}

void WgpuPathTracer::setFoveatedRendering(bool enabled) {
    pimpl_->foveatedEnabled_ = enabled;
}

bool WgpuPathTracer::foveatedRendering() const {
    return pimpl_->foveatedEnabled_;
}

int WgpuPathTracer::frameCount() const {
    return static_cast<int>(pimpl_->frameCount_);
}

void WgpuPathTracer::resetAccumulation() {
    pimpl_->frameCount_ = 0.f;
}

void WgpuPathTracer::setOverlayLayer(int channel) {
    pimpl_->overlayLayer_ = channel;
}

int WgpuPathTracer::overlayLayer() const {
    return pimpl_->overlayLayer_;
}

void WgpuPathTracer::setTextureResolution(int size) {
    size = (size >= 2048) ? 2048 : 1024;
    pimpl_->textureResolution_ = size;
}

int WgpuPathTracer::textureResolution() const {
    return pimpl_->textureResolution_;
}

void WgpuPathTracer::markDirty() {
    pimpl_->prevMeshes.clear();
    pimpl_->prevEntryCount_ = 0;
}

void WgpuPathTracer::dispose() {
    auto& d = *pimpl_;
    if (d.depthFillPipeline_)   { wgpuRenderPipelineRelease(d.depthFillPipeline_);   d.depthFillPipeline_ = nullptr; }
    if (d.depthFillPipeLayout_) { wgpuPipelineLayoutRelease(d.depthFillPipeLayout_); d.depthFillPipeLayout_ = nullptr; }
    if (d.depthFillBGL_)        { wgpuBindGroupLayoutRelease(d.depthFillBGL_);       d.depthFillBGL_ = nullptr; }
    if (d.depthFillShader_)     { wgpuShaderModuleRelease(d.depthFillShader_);       d.depthFillShader_ = nullptr; }
    if (d.depthFillUniBuf_)     { wgpuBufferRelease(d.depthFillUniBuf_);             d.depthFillUniBuf_ = nullptr; }
}
