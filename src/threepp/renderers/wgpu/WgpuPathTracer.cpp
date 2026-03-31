
#include "threepp/renderers/wgpu/WgpuPathTracer.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuBuffer.hpp"
#include "threepp/renderers/wgpu/WgpuComputePipeline.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/geometries/PlaneGeometry.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/materials/LineBasicMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <webgpu/webgpu.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>
#ifndef __EMSCRIPTEN__
#include <future>
#endif
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace threepp;

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------
namespace {

constexpr int MAX_TEX_SLOTS = 64;
constexpr int TILE_SIZE = 1024;
constexpr int ATLAS_COLS = 8;  // tiles per row in atlas (8 × 1024 = 8192 ≤ GPU max)
constexpr int TRI_TEX_HEIGHT = 8;
constexpr int MAT_TEX_HEIGHT = 6;
constexpr int TEX_PAGE_WIDTH = 8192;

// Initial placeholder capacities — grown dynamically as scenes demand more.
constexpr int INIT_TRI_CAP  = 1;
constexpr int INIT_MAT_CAP  = 1;
constexpr int INIT_MESH_CAP = 1;

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
constexpr const char* csCommonWGSL = R"(

struct RtUniforms {
    camOri:     vec4<f32>,
    camFwd:     vec4<f32>,
    camRgt:     vec4<f32>,
    camUp:      vec4<f32>,
    iRes:       vec4<f32>,
    tanHalfFov: vec4<f32>,
    frameCount: vec4<f32>,
    triCount:   vec4<f32>,
    mode:       vec4<f32>,
    lightCount: vec4<f32>,
    lightPos:   array<vec4<f32>, 4>,
    lightCol:   array<vec4<f32>, 4>,
    lightType:  array<vec4<f32>, 4>,  // x: 0=point, 1=directional, 2=spot; y=cosAngle; z=cosOuter
    lightDir:   array<vec4<f32>, 4>,  // xyz: spotlight direction (normalized)
    spp:          vec4<f32>,
    movedMeshBits: vec4<u32>,  // bit i = mesh i moved (words 0/1 cover meshes 0-63)
    envColor:      vec4<f32>,  // xyz = color/tint, w = mode: 0=sky gradient, 1=solid color, 2=equirect tex
    envIntensity:  vec4<f32>,  // x = intensity scale, y = envWidth, z = envHeight, w = hasEnvCDF
    params:        vec4<f32>,  // x = maxBounces, y = ambientFactor
    emissiveInfo:  vec4<f32>,  // x = emissive triangle count, y = total emissive power
};

struct BvhNodeGpu {
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
}

@group(0) @binding(0) var<uniform> rt:          RtUniforms;
@group(0) @binding(1) var accumRead:  texture_2d<f32>;
@group(0) @binding(2) var accumWrite: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> bvhNodes: array<BvhNodeGpu>;
@group(0) @binding(4) var matData:    texture_2d<f32>;
@group(0) @binding(5) var triData:       texture_2d<f32>;
@group(0) @binding(6) var texAtlas:      texture_2d<f32>;
@group(0) @binding(7) var hitMeshRead:   texture_2d<f32>;
@group(0) @binding(8) var hitMeshWrite:  texture_storage_2d<rgba16float, write>;
@group(0) @binding(9)  var envTex:      texture_2d<f32>;
@group(0) @binding(10) var gBufWrite:   texture_storage_2d<rgba16float, write>;
@group(0) @binding(11) var<storage, read> emissiveTris: array<vec4<f32>>;  // per tri: (triIndex, area, 0, 0)
@group(0) @binding(14) var albedoWrite: texture_storage_2d<rgba16float, write>;

const TRI_PAGE_W:  i32 = 8192;
const TRI_PAGE_H:  i32 = 8;
const TILE_SIZE:   i32 = 1024;
const MAX_TEX_SLOTS: i32 = 64;
const ATLAS_COLS:  i32 = 8;

fn triCoord(ti: i32, row: i32) -> vec2<i32> {
    return vec2<i32>(ti % TRI_PAGE_W, (ti / TRI_PAGE_W) * TRI_PAGE_H + row);
}

struct Ray  { origin: vec3<f32>, dir: vec3<f32> }
struct Isect { t: f32, u: f32, v: f32 }
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
    transmission: f32,
    ior:          f32,
    frontFace:    f32,
    attenuationColor: vec3<f32>,
    attenuationDist:  f32,
    clearcoat:        f32,
    clearcoatAlpha:   f32,
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
    let k = alpha * 0.5;
    return NdotX / max(NdotX * (1.0 - k) + k, 1e-6);
}

fn schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(max(0.0, 1.0 - cosTheta), 5.0);
}
)"
R"(
fn sampleAtlas(uv: vec2<f32>, texSlot: f32) -> vec3<f32> {
    let slot = i32(texSlot);
    let col  = slot % ATLAS_COLS;
    let row  = slot / ATLAS_COLS;
    let ox   = col * TILE_SIZE;
    let oy   = row * TILE_SIZE;
    let ts   = f32(TILE_SIZE);
    // Bilinear filter, clamped within the tile to avoid atlas bleeding
    let fp  = vec2<f32>(fract(uv.x), fract(uv.y)) * ts - 0.5;
    let x0  = clamp(i32(floor(fp.x)), 0, TILE_SIZE - 1);
    let y0  = clamp(i32(floor(fp.y)), 0, TILE_SIZE - 1);
    let x1  = clamp(x0 + 1,          0, TILE_SIZE - 1);
    let y1  = clamp(y0 + 1,          0, TILE_SIZE - 1);
    let wx  = fp.x - floor(fp.x);
    let wy  = fp.y - floor(fp.y);
    let c00 = textureLoad(texAtlas, vec2<i32>(ox + x0, oy + y0), 0);
    let c10 = textureLoad(texAtlas, vec2<i32>(ox + x1, oy + y0), 0);
    let c01 = textureLoad(texAtlas, vec2<i32>(ox + x0, oy + y1), 0);
    let c11 = textureLoad(texAtlas, vec2<i32>(ox + x1, oy + y1), 0);
    let blended = mix(mix(c00, c10, wx), mix(c01, c11, wx), wy);
    return blended.xyz;
}

fn sampleAtlasAlpha(uv: vec2<f32>, texSlot: f32) -> f32 {
    let slot = i32(texSlot);
    let col  = slot % ATLAS_COLS;
    let row  = slot / ATLAS_COLS;
    let ox   = col * TILE_SIZE;
    let oy   = row * TILE_SIZE;
    let ts   = f32(TILE_SIZE);
    let fp  = vec2<f32>(fract(uv.x), fract(uv.y)) * ts - 0.5;
    let x0  = clamp(i32(floor(fp.x)), 0, TILE_SIZE - 1);
    let y0  = clamp(i32(floor(fp.y)), 0, TILE_SIZE - 1);
    let x1  = clamp(x0 + 1,          0, TILE_SIZE - 1);
    let y1  = clamp(y0 + 1,          0, TILE_SIZE - 1);
    let wx  = fp.x - floor(fp.x);
    let wy  = fp.y - floor(fp.y);
    let a00 = textureLoad(texAtlas, vec2<i32>(ox + x0, oy + y0), 0).w;
    let a10 = textureLoad(texAtlas, vec2<i32>(ox + x1, oy + y0), 0).w;
    let a01 = textureLoad(texAtlas, vec2<i32>(ox + x0, oy + y1), 0).w;
    let a11 = textureLoad(texAtlas, vec2<i32>(ox + x1, oy + y1), 0).w;
    return mix(mix(a00, a10, wx), mix(a01, a11, wx), wy);
}
)"
R"(
// Returns the raw environment/background color with no intensity scaling.
// Callers apply rt.envIntensity.x themselves when using this as a light source.
fn sampleEnv(d: vec3<f32>) -> vec3<f32> {
    let mode = i32(rt.envColor.w);
    if (mode == 1) {
        return rt.envColor.xyz;
    } else if (mode == 2) {
        // Equirectangular texture
        let nd  = normalize(d);
        let phi = atan2(nd.z, nd.x);
        let theta = asin(clamp(nd.y, -1.0, 1.0));
        let uv  = vec2<f32>(0.5 + phi / (2.0 * PI),
                            0.5 - theta / PI);
        let sz  = vec2<f32>(textureDimensions(envTex, 0));
        let px  = vec2<i32>(i32(uv.x * sz.x) % i32(sz.x),
                            clamp(i32(uv.y * sz.y), 0, i32(sz.y) - 1));
        return textureLoad(envTex, px, 0).xyz;
    }
    // Default: procedural sky gradient
    let t = clamp(0.5 * (normalize(d).y + 1.0), 0.0, 1.0);
    return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.32, 0.52, 1.0), t);
}

fn triIntersect(ray: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> Isect {
    var r: Isect; r.t = 1e30;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let pv = cross(ray.dir, e2);
    let a  = dot(e1, pv);
    if (abs(a) < 1e-7) { return r; }
    let f  = 1.0 / a;
    let sv = ray.origin - v0;
    let u  = f * dot(sv, pv);
    if (u < 0.0 || u > 1.0) { return r; }
    let qv = cross(sv, e1);
    let v  = f * dot(ray.dir, qv);
    if (v < 0.0 || u + v > 1.0) { return r; }
    let t  = f * dot(e2, qv);
    if (t < 1e-4) { return r; }
    r.t = t; r.u = u; r.v = v;
    return r;
}

fn aabbDist(bmin: vec3<f32>, bmax: vec3<f32>, ray: Ray, tmax: f32) -> f32 {
    let invD  = vec3<f32>(1.0) / ray.dir;
    let t1    = (bmin - ray.origin) * invD;
    let t2    = (bmax - ray.origin) * invD;
    let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tFar  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tFar >= max(tNear, 0.0) && tNear < tmax) { return max(tNear, 0.0); }
    return 1e30;
}
fn aabbHit(bmin: vec3<f32>, bmax: vec3<f32>, ray: Ray, tmax: f32) -> bool {
    return aabbDist(bmin, bmax, ray, tmax) < 1e30;
}

fn testTriangle(ray: Ray, ti: i32, h: ptr<function, Hit>) {
    let r0  = textureLoad(triData, triCoord(ti, 0), 0);
    let v0  = r0.xyz;
    let r1  = textureLoad(triData, triCoord(ti, 1), 0);
    let v1  = r1.xyz;
    let v2  = textureLoad(triData, triCoord(ti, 2), 0).xyz;

    let isect = triIntersect(ray, v0, v1, v2);
    if (isect.t >= (*h).t) { return; }

    let w  = 1.0 - isect.u - isect.v;
    let n0 = textureLoad(triData, triCoord(ti, 3), 0).xyz;
    let n1 = textureLoad(triData, triCoord(ti, 4), 0).xyz;
    let n2 = textureLoad(triData, triCoord(ti, 5), 0).xyz;
    let sn = normalize(n0 * w + n1 * isect.u + n2 * isect.v);

    let uv01 = textureLoad(triData, triCoord(ti, 6), 0);
    let uv2  = textureLoad(triData, triCoord(ti, 7), 0).xy;
    let iuv  = vec2<f32>(uv01.x, uv01.y) * w
             + vec2<f32>(uv01.z, uv01.w) * isect.u
             + uv2                        * isect.v;

    let matIdx = i32(r0.w);
    let mat0   = textureLoad(matData, vec2<i32>(matIdx, 0), 0);
    let mat1   = textureLoad(matData, vec2<i32>(matIdx, 1), 0);
    let mat2   = textureLoad(matData, vec2<i32>(matIdx, 2), 0);
    let mat3   = textureLoad(matData, vec2<i32>(matIdx, 3), 0);

    // Alpha test: discard hit if texture alpha is below threshold
    let alphaTest = mat3.y;
    if (alphaTest > 0.0 && mat1.x >= 0.0) {
        let texAlpha = sampleAtlasAlpha(iuv, mat1.x);
        if (texAlpha < alphaTest) { return; }
    }

    let isFrontFace = dot(ray.dir, sn) < 0.0;
    var finalNorm = select(-sn, sn, isFrontFace);

    // Normal mapping
    let normalSlot = mat1.z;
    if (normalSlot >= 0.0) {
        let nmSample = sampleAtlas(iuv, normalSlot);
        let nmTangent = nmSample * 2.0 - 1.0;  // [0,1] → [-1,1]
        // Compute TBN from triangle edges and UVs
        let e1  = v1 - v0;
        let e2  = v2 - v0;
        let uv0 = vec2<f32>(uv01.x, uv01.y);
        let uv1 = vec2<f32>(uv01.z, uv01.w);
        let duv1 = uv1 - uv0;
        let duv2 = uv2 - uv0;
        let denom = duv1.x * duv2.y - duv2.x * duv1.y;
        if (abs(denom) > 1e-8) {
            let invD = 1.0 / denom;
            var T = normalize((e1 * duv2.y - e2 * duv1.y) * invD);
            // Gram-Schmidt orthogonalize T against N
            T = normalize(T - finalNorm * dot(finalNorm, T));
            let B = cross(finalNorm, T);
            finalNorm = normalize(T * nmTangent.x + B * nmTangent.y + finalNorm * nmTangent.z);
        }
    }

    // Roughness map
    var shininess = mat0.w;
    let roughSlot = mat1.w;
    if (roughSlot >= 0.0) {
        let roughSample = sampleAtlas(iuv, roughSlot);
        // roughnessMap green channel (glTF convention)
        let mappedRough = roughSample.y;
        shininess = max(1e-4, mappedRough * mappedRough);
    }

    // Geometric (flat) normal — needed for MIS PDF of emissive surfaces
    let geoNcross = cross(v1 - v0, v2 - v0);
    let geoNlen   = length(geoNcross);
    let geoN    = select(sn, geoNcross / geoNlen, geoNlen > 1e-8);
    let geoNorm = select(-geoN, geoN, isFrontFace);

    (*h).t         = isect.t;
    (*h).point     = ray.origin + isect.t * ray.dir;
    (*h).normal    = finalNorm;
    (*h).geoNormal = geoNorm;
    (*h).albedo    = mat0.xyz;
    (*h).shininess = shininess;
    (*h).uv        = iuv;
    (*h).texSlot   = mat1.x;
    (*h).metalness = mat1.y;
    (*h).emissive     = mat2.xyz;
    (*h).transmission = mat2.w;
    (*h).ior          = mat3.x;
    (*h).frontFace    = select(0.0, 1.0, isFrontFace);
    (*h).meshIdx      = i32(r1.w);
    // Row 4: attenuationColor.rgb + attenuationDistance
    let mat4 = textureLoad(matData, vec2<i32>(matIdx, 4), 0);
    (*h).attenuationColor = mat4.xyz;
    (*h).attenuationDist  = mat4.w;
    // Row 5: clearcoat + clearcoatAlpha
    let mat5 = textureLoad(matData, vec2<i32>(matIdx, 5), 0);
    (*h).clearcoat      = mat5.x;
    (*h).clearcoatAlpha = mat5.y;
}

fn sceneHit(ray: Ray) -> Hit {
    var h: Hit; h.t = 1e30; h.meshIdx = -1; h.transmission = 0.0; h.ior = 1.5; h.frontFace = 1.0; h.geoNormal = vec3<f32>(0.0); h.attenuationColor = vec3<f32>(1.0); h.attenuationDist = 0.0; h.clearcoat = 0.0; h.clearcoatAlpha = 0.0;
    var stack: array<i32, 64>;
    var top: i32 = 0;
    stack[0] = 0; top = 1;

    while (top > 0) {
        top -= 1;
        let ni  = stack[top];
        let nd  = bvhNodes[ni];
        if (!aabbHit(nd.row0.xyz, nd.row1.xyz, ray, h.t)) { continue; }
        let left  = bitcast<i32>(nd.row0.w);
        let right = bitcast<i32>(nd.row1.w);
        if (left < 0) {
            let triStart = -left - 1;
            let triCount = right;
            for (var ti = triStart; ti < triStart + triCount; ti++) {
                testTriangle(ray, ti, &h);
            }
        } else {
            let lnd = bvhNodes[left];
            let rnd = bvhNodes[right];
            let dL = aabbDist(lnd.row0.xyz, lnd.row1.xyz, ray, h.t);
            let dR = aabbDist(rnd.row0.xyz, rnd.row1.xyz, ray, h.t);
            if (dL < dR) {
                if (dR < 1e30) { stack[top] = right; top += 1; }
                if (dL < 1e30) { stack[top] = left;  top += 1; }
            } else {
                if (dL < 1e30) { stack[top] = left;  top += 1; }
                if (dR < 1e30) { stack[top] = right; top += 1; }
            }
        }
    }
    return h;
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
)";

// ---------------------------------------------------------------------------
// WGSL compute shader — raycaster-specific code
// ---------------------------------------------------------------------------
constexpr const char* csRaytraceWGSL = R"(

fn shade(h: Hit, rd: vec3<f32>) -> vec3<f32> {
    var albedo = h.albedo;
    if (h.texSlot >= 0.0) { albedo = sampleAtlas(h.uv, h.texSlot); }
    // Ambient: use env light in normal direction when envmap is available, otherwise flat ambient
    // Flat ambient, boosted by env average brightness when envmap is present
    let envBoost = select(1.0, max(1.0, rt.envIntensity.x * 3.0), rt.envColor.w > 1.5);
    var ambient = vec3<f32>(rt.params.y * envBoost);
    var col = albedo * ambient;
    let count = i32(rt.lightCount.x);
    let wo_s = normalize(-rd);
    for (var li = 0; li < 4; li++) {
        if (li >= count) { break; }
        var lc    = rt.lightCol[li].xyz;
        let ltype = i32(rt.lightType[li].x);  // 0=point, 1=directional, 2=spot
        let ln    = select(normalize(rt.lightPos[li].xyz - h.point),
                           normalize(rt.lightPos[li].xyz), ltype == 1);
        let ld    = select(length(rt.lightPos[li].xyz - h.point), 1e30, ltype == 1);

        // Spotlight cone attenuation
        if (ltype == 2) {
            let spotDir   = rt.lightDir[li].xyz;
            let cosTheta  = dot(-ln, spotDir);
            let cosInner  = rt.lightType[li].y;
            let cosOuter  = rt.lightType[li].z;
            let spotAtten = clamp((cosTheta - cosOuter) / max(cosInner - cosOuter, 1e-6), 0.0, 1.0);
            lc *= spotAtten;
        }

        var sr: Ray; sr.origin = h.point + h.normal * 1e-3; sr.dir = ln;
        var shAtten = vec3<f32>(1.0);
        for (var si = 0; si < 4; si++) {
            let sh = sceneHit(sr);
            if (sh.t >= ld - 1e-3) { break; }
            if (sh.transmission < 0.01) { shAtten = vec3<f32>(0.0); break; }
            var shAlb = sh.albedo;
            if (sh.texSlot >= 0.0) { shAlb = sampleAtlas(sh.uv, sh.texSlot); }
            shAtten *= shAlb * sh.transmission;
            sr.origin = sh.point - sh.normal * 1e-3;
        }
        if (shAtten.x + shAtten.y + shAtten.z > 0.001) {
            let NdotL = max(0.0, dot(h.normal, ln));
            let hv    = normalize(wo_s + ln);
            let NdotH = max(0.0, dot(h.normal, hv));
            let NdotV = max(0.001, dot(h.normal, wo_s));
            let NdotL2 = max(0.001, NdotL);
            let VdotH = max(0.0, dot(wo_s, hv));
            let F0_s  = mix(vec3<f32>(0.04), albedo, h.metalness);
            let Ds    = ggxD(NdotH, h.shininess);
            let Fs    = schlick(VdotH, F0_s);
            let Gs    = ggxG1(NdotV, h.shininess) * ggxG1(NdotL2, h.shininess);
            let f_spec = Ds * Fs * Gs / max(4.0 * NdotV * NdotL2, 1e-6);
            let f_diff = (vec3<f32>(1.0) - Fs) * albedo / PI;
            col += shAtten * lc * (f_diff + f_spec) * NdotL;
        }
    }
    return clamp(col + h.emissive, vec3<f32>(0.0), vec3<f32>(1.0));
}
)"
R"(
fn raytrace(ray: Ray) -> vec3<f32> {
    let h0 = sceneHit(ray);
    if (h0.t >= 1e30) { return sampleEnv(ray.dir); }

    // Transmission: one-level refraction for raytracer mode (smooth surfaces only)
    if (h0.transmission > 0.5 && h0.shininess < 0.1) {
        let glassTint = mix(h0.albedo, vec3<f32>(1.0), h0.transmission);
        let entering = h0.frontFace > 0.5;
        let eta = select(h0.ior, 1.0 / h0.ior, entering);
        let refDir = refract(normalize(ray.dir), h0.normal, eta);
        if (length(refDir) > 0.001) {
            var rr: Ray;
            rr.origin = h0.point - h0.normal * 1e-3;
            rr.dir    = refDir;
            let hr = sceneHit(rr);
            if (hr.t >= 1e30) { return sampleEnv(rr.dir) * glassTint; }
            // Beer's law attenuation through the medium
            var volAtten = vec3<f32>(1.0);
            if (h0.attenuationDist > 0.0) {
                let absorbCoeff = -log(max(h0.attenuationColor, vec3<f32>(1e-6))) / h0.attenuationDist;
                volAtten = exp(-absorbCoeff * hr.t);
            }
            // Second refraction (exit surface)
            let exitEnter = hr.frontFace > 0.5;
            let eta2 = select(hr.ior, 1.0 / hr.ior, exitEnter);
            let refDir2 = refract(normalize(rr.dir), hr.normal, eta2);
            if (length(refDir2) > 0.001) {
                var rr2: Ray;
                rr2.origin = hr.point - hr.normal * 1e-3;
                rr2.dir    = refDir2;
                let hr2 = sceneHit(rr2);
                if (hr2.t >= 1e30) { return sampleEnv(rr2.dir) * glassTint * volAtten; }
                return shade(hr2, rr2.dir) * glassTint * volAtten;
            }
            return shade(hr, rr.dir) * glassTint * volAtten;
        }
    }

    var col = shade(h0, ray.dir);

    // Specular mirror bounces (two levels, unrolled — deterministic, no seed needed).
    if (h0.shininess < 0.05 || (h0.metalness > 0.5 && h0.shininess < 0.3)) {
        let F0_0   = mix(vec3<f32>(0.04), h0.albedo, h0.metalness);
        let NdotV0 = max(0.001, dot(h0.normal, normalize(-ray.dir)));
        let roughFade = max(0.0, 1.0 - h0.shininess * 10.0);
        let k0     = schlick(NdotV0, F0_0) * roughFade;

        var r1: Ray;
        r1.origin = h0.point + h0.normal * 1e-3;
        r1.dir    = reflect(ray.dir, h0.normal);
        let h1    = sceneHit(r1);

        var rc1: vec3<f32>;
        if (h1.t >= 1e30) {
            rc1 = sampleEnv(r1.dir) * rt.envIntensity.x;
        } else {
            var base1 = shade(h1, r1.dir);
            // Second specular bounce: reflections-of-reflections
            if (h1.shininess < 0.5) {
                let F0_1   = mix(vec3<f32>(0.04), h1.albedo, h1.metalness);
                let NdotV1 = max(0.001, dot(h1.normal, normalize(-r1.dir)));
                let k1     = schlick(NdotV1, F0_1) * max(0.0, 1.0 - h1.shininess * 2.0);
                var r2: Ray;
                r2.origin = h1.point + h1.normal * 1e-3;
                r2.dir    = reflect(r1.dir, h1.normal);
                let h2    = sceneHit(r2);
                var rc2   = select(shade(h2, r2.dir), sampleEnv(r2.dir) * rt.envIntensity.x, h2.t >= 1e30);
                rc2   = max(rc2, mix(vec3<f32>(0.04), h1.albedo, h1.metalness) * 0.08);
                base1 = base1 * (vec3<f32>(1.0) - k1) + rc2 * k1;
            }
            rc1 = base1;
        }
        rc1 = max(rc1, F0_0 * 0.08);
        col = col * (vec3<f32>(1.0) - k0) + rc1 * k0;
    }
    return col;
}

@compute @workgroup_size(8, 8)
fn rt_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(i32(gid.x), i32(gid.y));
    let res   = rt.iRes.xy;
    if (f32(pixel.x) >= res.x || f32(pixel.y) >= res.y) { return; }

    let spp = i32(rt.spp.x);
    let fp  = vec2<f32>(f32(pixel.x), f32(pixel.y));
    var sample: vec3<f32>;
    if (spp >= 4) {
        let o0 = vec2<f32>( 0.125,  0.375);
        let o1 = vec2<f32>(-0.375,  0.125);
        let o2 = vec2<f32>( 0.375, -0.125);
        let o3 = vec2<f32>(-0.125, -0.375);
        sample = (raytrace(makeRay(fp + o0, res))
                + raytrace(makeRay(fp + o1, res))
                + raytrace(makeRay(fp + o2, res))
                + raytrace(makeRay(fp + o3, res))) * 0.25;
    } else if (spp >= 2) {
        let o0 = vec2<f32>( 0.25,  0.25);
        let o1 = vec2<f32>(-0.25, -0.25);
        sample = (raytrace(makeRay(fp + o0, res))
                + raytrace(makeRay(fp + o1, res))) * 0.5;
    } else {
        sample = raytrace(makeRay(fp + vec2<f32>(0.5, 0.5), res));
    }
    textureStore(accumWrite, pixel, vec4<f32>(sample, 1.0));
    let centerHit = sceneHit(makeRay(fp + vec2<f32>(0.5, 0.5), res));
    let tVal = select(centerHit.t, 0.0, centerHit.t >= 1e20);
    textureStore(gBufWrite, pixel, vec4<f32>(vec3<f32>(0.0), tVal));
    textureStore(albedoWrite, pixel, vec4<f32>(1.0, 1.0, 1.0, 1.0));
}
)";

// ---------------------------------------------------------------------------
// WGSL compute shader — path tracer-specific code
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

fn isMeshMoved(idx: i32) -> bool {
    if (idx < 0 || idx >= 64) { return false; }
    let ui = u32(idx);
    let bit  = ui & 31u;
    let word = select(rt.movedMeshBits.x, rt.movedMeshBits.y, ui >= 32u);
    return ((word >> bit) & 1u) != 0u;
}

fn pathTrace(ray_in: Ray, seed: ptr<function, u32>,
             primaryMeshIdx: ptr<function, u32>,
             primaryNormal:  ptr<function, vec3<f32>>,
             primaryDepth:   ptr<function, f32>,
             primaryAlbedo:  ptr<function, vec3<f32>>,
             touchedMoved:   ptr<function, bool>) -> vec3<f32> {
    *primaryMeshIdx = 64u;
    *primaryNormal  = vec3<f32>(0.0);
    *primaryDepth   = 0.0;  // 0 = sky/no-hit sentinel for denoiser
    *primaryAlbedo  = vec3<f32>(1.0);  // default white (sky/miss: no demodulation)
    *touchedMoved   = false;
    var ray        = ray_in;
    var throughput = vec3<f32>(1.0);
    var radiance   = vec3<f32>(0.0);

    // Previous-bounce surface properties for MIS weighting
    var prevNormal    = vec3<f32>(0.0, 1.0, 0.0);
    var prevAlpha     = 0.0;
    var prevMetalness = 0.0;
    var prevWo        = vec3<f32>(0.0);
    var afterTransmission = false;

    for (var i = 0; i < i32(rt.params.x); i++) {
        let h = sceneHit(ray);
        if (h.t >= 1e29) {
            // Primary ray miss = background pixel: show full color.
            // Indirect bounces: scale by envIntensity to control env light contribution.
            let envScale = select(rt.envIntensity.x, 1.0, i == 0);
            var envMisW = 1.0;
            // MIS: BRDF sampling hit the env — weight against env importance PDF
            if (HAS_ENV_CDF && rt.envColor.w > 1.5 && i > 0 && prevAlpha > 0.01) {
                let pdf_env  = envImportancePdf(ray.dir);
                let pdf_brdf = brdfPdf(prevWo, normalize(ray.dir), prevNormal, prevAlpha, prevMetalness);
                envMisW = pdf_brdf / max(pdf_brdf + pdf_env, 1e-8);
            }
            radiance += throughput * sampleEnv(ray.dir) * envScale * envMisW;
            break;
        }
        if (i == 0) {
            *primaryMeshIdx = u32(h.meshIdx);
            *primaryNormal  = h.normal;
            *primaryDepth   = h.t;
        } else if (isMeshMoved(h.meshIdx)) {
            *touchedMoved = true;
        }

        let emTriCount = i32(rt.emissiveInfo.x);
        let totalPower = rt.emissiveInfo.y;

        // Emissive hit with MIS balance heuristic
        if (length(h.emissive) > 0.0) {
            if (i == 0 || emTriCount == 0 || afterTransmission) {
                // Primary ray, no NEE available, or after transmission: full weight
                radiance += throughput * h.emissive;
            } else {
                // MIS: BRDF sampling hit emissive — weight by balance heuristic
                let cosLight = abs(dot(h.geoNormal, -ray.dir));
                if (cosLight > 1e-6) {
                    let emLum = 0.2126 * h.emissive.r + 0.7152 * h.emissive.g + 0.0722 * h.emissive.b;
                    let pdf_light = (emLum * h.t * h.t) / (totalPower * cosLight);
                    let pdf_brdf  = brdfPdf(prevWo, normalize(ray.dir), prevNormal, prevAlpha, prevMetalness);
                    let w = pdf_brdf / max(pdf_brdf + pdf_light, 1e-8);
                    if (w == w && w < 1e10) {
                        radiance += throughput * h.emissive * w;
                    }
                }
            }
        }

        var albedo = h.albedo;
        if (h.texSlot >= 0.0) { albedo = sampleAtlas(h.uv, h.texSlot); }

        if (i == 0) { *primaryAlbedo = albedo; }

        let wo = normalize(-ray.dir);

        // --- Analytical light NEE ---
        let lcount = i32(rt.lightCount.x);
        for (var li = 0; li < 4; li++) {
            if (li >= lcount) { break; }
            var lc    = rt.lightCol[li].xyz;
            let ltype = i32(rt.lightType[li].x);
            let ln    = select(normalize(rt.lightPos[li].xyz - h.point),
                               normalize(rt.lightPos[li].xyz), ltype == 1);
            let ld    = select(length(rt.lightPos[li].xyz - h.point), 1e30, ltype == 1);
            // Spotlight cone attenuation
            if (ltype == 2) {
                let spotDir   = rt.lightDir[li].xyz;
                let cosTheta  = dot(-ln, spotDir);
                let cosInner  = rt.lightType[li].y;
                let cosOuter  = rt.lightType[li].z;
                let spotAtten = clamp((cosTheta - cosOuter) / max(cosInner - cosOuter, 1e-6), 0.0, 1.0);
                lc *= spotAtten;
            }
            var sr: Ray;
            sr.origin = h.point + h.normal * 1e-3;
            sr.dir    = ln;
            var shadowAtten = vec3<f32>(1.0);
            for (var si = 0; si < 4; si++) {
                let sh = sceneHit(sr);
                if (sh.t >= ld - 1e-3) { break; }
                if (isMeshMoved(sh.meshIdx)) { *touchedMoved = true; }
                if (sh.transmission < 0.01) { shadowAtten = vec3<f32>(0.0); break; }
                var shAlbedo = sh.albedo;
                if (sh.texSlot >= 0.0) { shAlbedo = sampleAtlas(sh.uv, sh.texSlot); }
                shadowAtten *= shAlbedo * sh.transmission;
                sr.origin = sh.point - sh.normal * 1e-3;
            }
            if (shadowAtten.x + shadowAtten.y + shadowAtten.z > 0.001) {
                let NdotL = dot(h.normal, ln);
                if (NdotL <= 0.0) { continue; }
                let hv    = normalize(wo + ln);
                let NdotH = max(0.0, dot(h.normal, hv));
                let NdotV = max(0.001, dot(h.normal, wo));
                let VdotH = max(0.0, dot(wo, hv));
                let F0    = mix(vec3<f32>(0.04), albedo, h.metalness);
                let D     = ggxD(NdotH, h.shininess);
                let F     = schlick(VdotH, F0);
                let G     = ggxG1(NdotV, h.shininess) * ggxG1(max(0.001, NdotL), h.shininess);
                let f_spec = D * F * G / max(4.0 * NdotV * NdotL, 1e-6);
                let f_diff = (vec3<f32>(1.0) - F) * albedo / PI;
                radiance  += throughput * shadowAtten * (f_diff + f_spec) * NdotL * lc;
            }
        }
)"
R"(
        // --- Emissive surface NEE (power-weighted CDF sampling) ---
        if (emTriCount > 0) {
            let totalPower = rt.emissiveInfo.y;
            let xi = rand(seed) * totalPower;
            var lo = 0;
            var hi = emTriCount - 1;
            while (lo < hi) {
                let mid = (lo + hi) >> 1;
                if (emissiveTris[mid].z < xi) { lo = mid + 1; } else { hi = mid; }
            }
            let emInfo = emissiveTris[lo];
            let eTi   = i32(emInfo.x);
            let eArea = emInfo.y;
            let ePower = emInfo.w;

            let ev0 = textureLoad(triData, triCoord(eTi, 0), 0).xyz;
            let ev1 = textureLoad(triData, triCoord(eTi, 1), 0).xyz;
            let ev2 = textureLoad(triData, triCoord(eTi, 2), 0).xyz;

            let su1 = sqrt(rand(seed));
            let u2  = rand(seed);
            let lightPoint = (1.0 - su1) * ev0 + su1 * (1.0 - u2) * ev1 + su1 * u2 * ev2;

            let toLight = lightPoint - h.point;
            let dist    = length(toLight);
            let ln      = toLight / dist;
            let NdotL   = dot(h.normal, ln);

            let lightNormal = normalize(cross(ev1 - ev0, ev2 - ev0));
            let cosLight    = abs(dot(lightNormal, -ln));

            if (NdotL > 0.0 && cosLight > 1e-6) {
                var sr: Ray;
                sr.origin = h.point + h.normal * 1e-3;
                sr.dir    = ln;
                let sh    = sceneHit(sr);

                if (sh.t >= dist - 1e-2) {
                    let eMatIdx = i32(textureLoad(triData, triCoord(eTi, 0), 0).w);
                    let emColor = textureLoad(matData, vec2<i32>(eMatIdx, 2), 0).xyz;

                    let pdf = (ePower * dist * dist) / (totalPower * eArea * cosLight);

                    let hv    = normalize(wo + ln);
                    let NdotH = max(0.0, dot(h.normal, hv));
                    let NdotV = max(0.001, dot(h.normal, wo));
                    let VdotH = max(0.0, dot(wo, hv));
                    let F0    = mix(vec3<f32>(0.04), albedo, h.metalness);
                    let D     = ggxD(NdotH, h.shininess);
                    let F     = schlick(VdotH, F0);
                    let G     = ggxG1(NdotV, h.shininess) * ggxG1(max(0.001, NdotL), h.shininess);
                    let f_spec = D * F * G / max(4.0 * NdotV * NdotL, 1e-6);
                    let f_diff = (vec3<f32>(1.0) - F) * albedo / PI;

                    let pdf_brdf_nee = brdfPdf(wo, ln, h.normal, h.shininess, h.metalness);
                    let w_light = pdf / max(pdf + pdf_brdf_nee, 1e-8);
                    radiance += throughput * (f_diff + f_spec) * NdotL * emColor * w_light / pdf;
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
                var sr: Ray;
                sr.origin = h.point + h.normal * 1e-3;
                sr.dir    = envDir;
                let sh = sceneHit(sr);
                if (sh.t >= 1e29) {
                    let hv    = normalize(wo + envDir);
                    let NdotH = max(0.0, dot(h.normal, hv));
                    let NdotV = max(0.001, dot(h.normal, wo));
                    let VdotH = max(0.0, dot(wo, hv));
                    let F0    = mix(vec3<f32>(0.04), albedo, h.metalness);
                    let D     = ggxD(NdotH, h.shininess);
                    let F     = schlick(VdotH, F0);
                    let G     = ggxG1(NdotV, h.shininess) * ggxG1(max(0.001, envNdotL), h.shininess);
                    let f_spec = D * F * G / max(4.0 * NdotV * envNdotL, 1e-6);
                    let f_diff = (vec3<f32>(1.0) - F) * albedo * (1.0 - h.metalness) / PI;
                    let envCol = sampleEnv(envDir) * rt.envIntensity.x;
                    let pdf_brdf_env = brdfPdf(wo, envDir, h.normal, h.shininess, h.metalness);
                    let w_env = envPdf / max(envPdf + pdf_brdf_env, 1e-8);
                    radiance += throughput * (f_diff + f_spec) * envNdotL * envCol * w_env / envPdf;
                }
            }
        }

        // Flat ambient fallback — only when no environment map is providing indirect light
        if (rt.envColor.w < 1.5 && h.transmission < 0.5 && h.metalness < 0.5) {
            radiance += throughput * albedo * rt.params.y;
        }

        if (i > 0) {
            let p = max(max(throughput.r, throughput.g), throughput.b);
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

        // Transmission lobe: refract through transmissive surfaces
        if (h.transmission > 0.0 && rand(seed) < h.transmission) {
            let entering = h.frontFace > 0.5;
            let eta = select(h.ior, 1.0 / h.ior, entering);

            var tNorm = h.normal;
            var usedMicrofacet = false;
            if (h.shininess > 1e-3) {
                let wo_t = normalize(-ray.dir);
                let hm = sampleVNDF(wo_t, h.normal, h.shininess, seed);
                if (dot(hm, h.normal) > 0.0) { tNorm = hm; usedMicrofacet = true; }
            }

            let cosI    = abs(dot(normalize(ray.dir), tNorm));
            let r0      = pow((1.0 - h.ior) / (1.0 + h.ior), 2.0);
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
            let glassTint = mix(albedo, vec3<f32>(1.0), h.transmission) * microWeight;
            var volAtten = vec3<f32>(1.0);
            if (didRefract && h.attenuationDist > 0.0 && !entering) {
                let absorbCoeff = -log(max(h.attenuationColor, vec3<f32>(1e-6))) / h.attenuationDist;
                volAtten = exp(-absorbCoeff * h.t);
            }
            // Non-symmetry correction: BTDF includes (η_t/η_i)² = 1/η² to account
            // for solid angle change at refractive interface.
            throughput *= select(glassTint * volAtten, glassTint * volAtten / (eta * eta), didRefract);
            afterTransmission = true;
            ray.dir = wi_t;
            continue;
        }

        let F0_b = mix(vec3<f32>(0.04), albedo, h.metalness);
        var wi_b: vec3<f32>;
        let p_spec = mix(0.5, 0.98, h.metalness);
        if (rand(seed) < p_spec) {
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
    return radiance;
}

@compute @workgroup_size(8, 8)
fn rt_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(i32(gid.x), i32(gid.y));
    let res   = rt.iRes.xy;
    if (f32(pixel.x) >= res.x || f32(pixel.y) >= res.y) { return; }

    let fc   = u32(rt.frameCount.x);
    // R2 quasi-random sub-pixel jitter (low-discrepancy stratification)
    let r2  = r2Seq(fc);
    // Per-pixel Cranley-Patterson rotation: offset R2 by a spatial hash to decorrelate pixels
    let pixHash = pcg(gid.x + gid.y * 65537u);
    let jx  = fract(r2.x + f32(pixHash) / 4294967296.0) - 0.5;
    let jy  = fract(r2.y + f32(pcg(pixHash)) / 4294967296.0) - 0.5;
    var seed = pcg(pcg(gid.x + gid.y * 65537u) + fc * 12979u);
    let ray = makeRay(vec2<f32>(f32(pixel.x) + jx, f32(pixel.y) + jy), res);
    var primaryMeshIdx: u32;
    var primaryNormal:  vec3<f32>;
    var primaryDepth:   f32;
    var primaryAlbedo:  vec3<f32>;
    var touchedMoved:   bool;
    var sample  = pathTrace(ray, &seed, &primaryMeshIdx, &primaryNormal, &primaryDepth, &primaryAlbedo, &touchedMoved);

    let prev     = textureLoad(accumRead, pixel, 0);
    let old      = prev.xyz;
    var pixelFC  = prev.w;

    // Reset pixels whose primary ray hits a moved mesh.
    let prevMeshIdx = u32(textureLoad(hitMeshRead, pixel, 0).r);
    if (primaryMeshIdx < 64u && isMeshMoved(i32(primaryMeshIdx))) {
        pixelFC = 0.0;
    }
    // Moved mesh left this pixel (was covering it, now moved away) — flush stale color.
    if (primaryMeshIdx != prevMeshIdx && prevMeshIdx < 64u && isMeshMoved(i32(prevMeshIdx))) {
        pixelFC = 0.0;
    }
    // Secondary bounce hit a moved mesh — cap rather than reset so static
    // surfaces can still converge while indirect lighting refreshes.
    if (touchedMoved) { pixelFC = min(pixelFC, 8.0); }
    if (fc == 0u) { pixelFC = 0.0; }

    // NaN guard: reject corrupted samples to prevent permanent accumulation damage
    let sampleClean = select(vec3<f32>(0.0), sample, sample.x == sample.x);
    let oldClean    = select(vec3<f32>(0.0), old,    old.x == old.x);

    // Adaptive outlier rejection: clamp sample relative to running average.
    // Allows legitimate highlights to accumulate over time while suppressing
    // single-sample fireflies that are orders of magnitude brighter.
    var clamped = sampleClean;
    if (pixelFC > 8.0) {
        let avgLum = max(dot(oldClean, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.01);
        let smpLum = dot(sampleClean, vec3<f32>(0.2126, 0.7152, 0.0722));
        let maxLum = avgLum * 20.0;  // allow 20x brighter than average
        if (smpLum > maxLum) {
            clamped = sampleClean * (maxLum / smpLum);
        }
    }

    // Stop accumulating once float16 precision is exhausted (~1024 frames)
    if (pixelFC < 1024.0) {
        let alpha   = 1.0 / (pixelFC + 1.0);
        let blended = oldClean * (1.0 - alpha) + clamped * alpha;
        textureStore(accumWrite, pixel, vec4<f32>(blended, pixelFC + 1.0));
    } else {
        textureStore(accumWrite, pixel, vec4<f32>(oldClean, pixelFC));
    }
    textureStore(hitMeshWrite, pixel, vec4<f32>(f32(primaryMeshIdx), 0.0, 0.0, 0.0));
    textureStore(albedoWrite,  pixel, vec4<f32>(primaryAlbedo, 1.0));
    textureStore(gBufWrite,    pixel, vec4<f32>(primaryNormal, primaryDepth));
}
)";

// Build a raycaster or path tracer shader by concatenating common + specific code.
static std::string buildRtShader(bool pathTrace, bool hasEnvCdf) {
    std::string src = std::string(csCommonWGSL) + "\n" +
                      (pathTrace ? csPathTraceWGSL : csRaytraceWGSL);
    if (pathTrace) {
        const std::string marker = "/*ENV_CDF_FLAG*/false";
        auto pos = src.find(marker);
        if (pos != std::string::npos) {
            src.replace(pos, marker.size(), hasEnvCdf ? "true" : "false");
        }
    }
    return src;
}

// ---------------------------------------------------------------------------
// WGSL vertex-transform compute shader
// ---------------------------------------------------------------------------
constexpr const char* vtWGSL = R"(
const TRI_PAGE_W: i32 = 8192;
const TRI_PAGE_H: i32 = 8;

fn triCoord(ti: i32, row: i32) -> vec2<i32> {
    return vec2<i32>(ti % TRI_PAGE_W, (ti / TRI_PAGE_W) * TRI_PAGE_H + row);
}

struct ObjTriData {
    v0:   vec4<f32>,
    v1:   vec4<f32>,
    v2:   vec4<f32>,
    n0:   vec4<f32>,
    n1:   vec4<f32>,
    n2:   vec4<f32>,
    uv01: vec4<f32>,
    uv2:  vec4<f32>,
}
struct MeshMatrices {
    world:  mat4x4<f32>,
    normal: mat4x4<f32>,
}
struct VtUniforms {
    triCount: u32,
    _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var<storage, read> objTris:  array<ObjTriData>;
@group(0) @binding(1) var<storage, read> meshMats: array<MeshMatrices>;
@group(0) @binding(2) var triOut: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var<uniform> vtUni: VtUniforms;

@compute @workgroup_size(64)
fn vt_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= vtUni.triCount) { return; }
    let ti  = i32(gid.x);
    let obj = objTris[ti];
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
}
)";

// ---------------------------------------------------------------------------
// WGSL BVH refit compute shader
// ---------------------------------------------------------------------------
constexpr const char* refitWGSL = R"(
const TRI_PAGE_W: i32 = 8192;
const TRI_PAGE_H: i32 = 8;

fn triCoord(ti: i32, row: i32) -> vec2<i32> {
    return vec2<i32>(ti % TRI_PAGE_W, (ti / TRI_PAGE_W) * TRI_PAGE_H + row);
}

struct BvhNodeGpu {
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
}
struct RefitUniforms {
    leafCount: u32,
    _p0: u32, _p1: u32, _p2: u32,
}

@group(0) @binding(0) var triTex:                          texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> bvhNodes:   array<BvhNodeGpu>;
@group(0) @binding(2) var<storage, read_write> bvhCounters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>       leafIdxBuf: array<i32>;
@group(0) @binding(4) var<uniform>             refitUni:   RefitUniforms;

@compute @workgroup_size(64)
fn bvh_refit(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= refitUni.leafCount) { return; }
    let leafNi   = leafIdxBuf[i32(gid.x)];
    let triStart = -(bitcast<i32>(bvhNodes[leafNi].row0.w)) - 1;
    let triCount =   bitcast<i32>(bvhNodes[leafNi].row1.w);

    var bmin = vec3<f32>(1e30);
    var bmax = vec3<f32>(-1e30);
    for (var ti = triStart; ti < triStart + triCount; ti++) {
        for (var row = 0; row < 3; row++) {
            let v = textureLoad(triTex, triCoord(ti, row), 0).xyz;
            bmin = min(bmin, v);
            bmax = max(bmax, v);
        }
    }
    bvhNodes[leafNi].row0 = vec4<f32>(bmin, bvhNodes[leafNi].row0.w);
    bvhNodes[leafNi].row1 = vec4<f32>(bmax, bvhNodes[leafNi].row1.w);

    var curNi = bitcast<i32>(bvhNodes[leafNi].row2.x);
    loop {
        if (curNi < 0) { break; }
        let cnt = atomicAdd(&bvhCounters[curNi], 1u);
        if (cnt == 0u) { break; }
        let lNi = bitcast<i32>(bvhNodes[curNi].row0.w);
        let rNi = bitcast<i32>(bvhNodes[curNi].row1.w);
        bvhNodes[curNi].row0 = vec4<f32>(
            min(bvhNodes[lNi].row0.xyz, bvhNodes[rNi].row0.xyz),
            bvhNodes[curNi].row0.w);
        bvhNodes[curNi].row1 = vec4<f32>(
            max(bvhNodes[lNi].row1.xyz, bvhNodes[rNi].row1.xyz),
            bvhNodes[curNi].row1.w);
        curNi = bitcast<i32>(bvhNodes[curNi].row2.x);
    }
}
)";

// ---------------------------------------------------------------------------
// WGSL TAA-style temporal filter
// Reprojects previous frame's filtered output, clamps to current neighborhood
// color box, and blends. Robust against reprojection errors via clamping.
// ---------------------------------------------------------------------------
constexpr const char* taaWGSL = R"(
struct TaaUniforms {
    prevCamOri: vec4<f32>, prevCamFwd: vec4<f32>, prevCamRgt: vec4<f32>, prevCamUp: vec4<f32>,
    curCamOri:  vec4<f32>, curCamFwd:  vec4<f32>, curCamRgt:  vec4<f32>, curCamUp:  vec4<f32>,
    iRes:       vec4<f32>,
    tanHalfFov: vec4<f32>,
    frameCount: vec4<f32>,
    movedMeshBits: vec4<u32>,
}

@group(0) @binding(0) var<uniform> taa:      TaaUniforms;
@group(0) @binding(1) var accumIn:   texture_2d<f32>;
@group(0) @binding(2) var gBufCur:   texture_2d<f32>;
@group(0) @binding(3) var taaHistIn: texture_2d<f32>;
@group(0) @binding(4) var taaOut:    texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var hitMeshTex: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> motionMats: array<mat4x4<f32>>;

@compute @workgroup_size(8, 8)
fn taa_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pixel = vec2<i32>(gid.xy);
    let res   = taa.iRes.xy;
    let iRes  = vec2<i32>(i32(res.x), i32(res.y));
    if (pixel.x >= iRes.x || pixel.y >= iRes.y) { return; }

    let curSamp  = textureLoad(accumIn, pixel, 0);
    let curColor = curSamp.xyz;
    let curFC    = curSamp.w;
    let curGB    = textureLoad(gBufCur, pixel, 0);
    let curDepth = curGB.w;

    // Sky pixels: pass through
    if (curDepth < 1e-6) {
        textureStore(taaOut, pixel, vec4<f32>(curColor, curFC));
        return;
    }

    // Build 3×3 neighborhood color box for clamping (same mesh only)
    let curMeshId = u32(textureLoad(hitMeshTex, pixel, 0).r);
    var cMin = curColor;
    var cMax = curColor;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sp = clamp(pixel + vec2<i32>(dx, dy), vec2<i32>(0), iRes - 1);
            let nMeshId = u32(textureLoad(hitMeshTex, sp, 0).r);
            if (nMeshId != curMeshId) { continue; }
            let nc = textureLoad(accumIn, sp, 0).xyz;
            cMin = min(cMin, nc);
            cMax = max(cMax, nc);
        }
    }
    // Slightly widen box for robustness (avoids over-clamping due to noise)
    let boxCenter = (cMin + cMax) * 0.5;
    let boxExtent = (cMax - cMin) * 0.5;
    cMin = boxCenter - boxExtent * 1.25;
    cMax = boxCenter + boxExtent * 1.25;

    // Reproject current pixel into previous frame's screen space
    let aspect = res.x / res.y;
    let ndc = vec2<f32>((f32(pixel.x) + 0.5) / res.x * 2.0 - 1.0,
                         1.0 - (f32(pixel.y) + 0.5) / res.y * 2.0);
    let rayDir   = normalize(taa.curCamFwd.xyz
                            + taa.curCamRgt.xyz * (ndc.x * taa.tanHalfFov.x * aspect)
                            + taa.curCamUp.xyz  * (ndc.y * taa.tanHalfFov.x));
    let worldPos = taa.curCamOri.xyz + rayDir * curDepth;

    // For moved meshes: transform world pos by motion matrix (prevWorld * inverse(curWorld))
    // to find where this surface point was in the previous frame.
    let meshIdx = u32(textureLoad(hitMeshTex, pixel, 0).r);
    var prevWorldPos = worldPos;
    if (meshIdx < 64u) {
        let bit   = meshIdx & 31u;
        let mbits = select(taa.movedMeshBits.x, taa.movedMeshBits.y, meshIdx >= 32u);
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

    // Detect moved mesh (reuse meshIdx from reprojection above)
    var movedMesh = false;
    if (meshIdx < 64u) {
        let bit   = meshIdx & 31u;
        let mbits = select(taa.movedMeshBits.x, taa.movedMeshBits.y, meshIdx >= 32u);
        if (((mbits >> bit) & 1u) != 0u) { movedMesh = true; }
    }

    // First frame has no history
    if (taa.frameCount.x < 0.5) { useHist = false; }

    var result = curColor;
    if (useHist) {
        // Bilinear interpolation of history — avoids nearest-neighbor artifacts
        let fx = fract(prevPx.x);
        let fy = fract(prevPx.y);
        let h00 = textureLoad(taaHistIn, prevPixel, 0).xyz;
        let h10 = textureLoad(taaHistIn, prevPixel + vec2<i32>(1, 0), 0).xyz;
        let h01 = textureLoad(taaHistIn, prevPixel + vec2<i32>(0, 1), 0).xyz;
        let h11 = textureLoad(taaHistIn, prevPixel + vec2<i32>(1, 1), 0).xyz;
        let histColor = mix(mix(h00, h10, fx), mix(h01, h11, fx), fy);
        let clamped   = clamp(histColor, cMin, cMax);
        // Blend: use more history early (noisy), more current once converged.
        // For moved meshes: pixelFC is always 0 (reset) so the normal formula gives alpha=1.0
        // (ignore history). With motion vectors the history IS correctly reprojected, so
        // override to blend ~20% current + 80% history for effective temporal accumulation.
        var alpha = clamp(1.0 / (curFC * 0.5 + 1.0), 0.1, 1.0);
        if (movedMesh) { alpha = 0.2; }
        result = mix(clamped, curColor, alpha);
    }

    textureStore(taaOut, pixel, vec4<f32>(result, curFC));
}
)";

// ---------------------------------------------------------------------------
// WGSL variance-guided à-trous spatial filter
// Uses the frame count (w channel) to adapt filter strength:
//   low frame count → more aggressive blur to suppress MC noise
//   high frame count → gentle filter to preserve converged detail
// ---------------------------------------------------------------------------
constexpr const char* svgfAtrousWGSL = R"(
struct AtrousUni { stepSize: u32, _p0: u32, _p1: u32, _p2: u32, }

@group(0) @binding(0) var<uniform> uni:      AtrousUni;
@group(0) @binding(1) var colorIn:  texture_2d<f32>;
@group(0) @binding(2) var colorOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var gBuf:     texture_2d<f32>;
@group(0) @binding(4) var albedoBuf: texture_2d<f32>;
@group(0) @binding(5) var hitMeshBuf: texture_2d<f32>;

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
    let cFC     = cSamp.w;
    let cGB     = textureLoad(gBuf, pixel, 0);
    let cNorm   = cGB.xyz;
    let cDepth  = cGB.w;
    let cAlbedo = textureLoad(albedoBuf, pixel, 0).xyz;
    let cMeshId = u32(textureLoad(hitMeshBuf, pixel, 0).r);

    // Sky pixels: pass through
    if (cDepth < 1e-6) {
        textureStore(colorOut, pixel, vec4<f32>(cColor, cFC));
        return;
    }

    // Demodulate center pixel for accumulation path
    let cIrr = demod(cColor, cAlbedo);
    // Blend luminance source: demod for bright surfaces (texture-aware), raw for dark (stable)
    let albedoLum = luminance_a(cAlbedo);
    let demodBlend = smoothstep(0.05, 0.2, albedoLum);
    let cLum = mix(luminance_a(cColor), luminance_a(cIrr), demodBlend);

    // Luminance sigma: relaxed when noisy (low FC), tight when converged
    let lumSigma = max(0.02, 0.25 / sqrt(cFC + 1.0));

    // 5×5 bilateral filter — tracks both demodulated irradiance and raw color
    let kw = array<f32, 5>(1.0/16.0, 4.0/16.0, 6.0/16.0, 4.0/16.0, 1.0/16.0);

    var irrSum    = vec3<f32>(0.0);
    var rawSum    = vec3<f32>(0.0);
    var weightSum = 0.0;

    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let sp      = clamp(pixel + vec2<i32>(dx, dy) * step, vec2<i32>(0), res - 1);
            let sMeshId = u32(textureLoad(hitMeshBuf, sp, 0).r);
            if (sMeshId != cMeshId) { continue; }

            let sColor  = textureLoad(colorIn, sp, 0).xyz;
            let sGB     = textureLoad(gBuf, sp, 0);
            let sAlbedo = textureLoad(albedoBuf, sp, 0).xyz;
            let sIrr    = demod(sColor, sAlbedo);
            let sNorm   = sGB.xyz;
            let sDepth  = sGB.w;

            // Spatial weight (separable Gaussian)
            let w_s = kw[dy + 2] * kw[dx + 2];
            // Normal edge-stopping: relaxed for noisy pixels (low FC), sharp when converged
            let normPow = mix(16.0, 64.0, smoothstep(0.0, 8.0, cFC));
            let w_n = pow(max(0.0, dot(cNorm, sNorm)), normPow);
            // Depth edge-stopping
            let w_d = exp(-abs(cDepth - sDepth) * 2.0 / (cDepth + 0.01));
            // Luminance edge-stopping: demod for bright surfaces, raw for dark
            let sAlbLum = luminance_a(sAlbedo);
            let sDemod = smoothstep(0.05, 0.2, sAlbLum);
            let sLum = mix(luminance_a(sColor), luminance_a(sIrr), sDemod);
            let w_l  = exp(-(sLum - cLum) * (sLum - cLum) / (lumSigma * lumSigma + 1e-6));

            let w = w_s * w_n * w_d * w_l;
            irrSum    += sIrr * w;
            rawSum    += sColor * w;
            weightSum += w;
        }
    }

    // Blend: demod/remod path for bright surfaces, raw filter for dark surfaces
    let filteredIrr = select(cIrr, irrSum / weightSum, weightSum > 1e-6);
    let filteredRaw = select(cColor, rawSum / weightSum, weightSum > 1e-6);
    let demodResult = filteredIrr * cAlbedo;
    let result = mix(filteredRaw, demodResult, demodBlend);
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
@group(0) @binding(4) var gBufTex:  texture_2d<f32>;  // gBuf: .w = primary hit t, 0 = background

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return transform.proj * transform.view * transform.model * vec4<f32>(position, 1.0);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32> {
    // _pad carries the renderer's pixelRatio (== path tracer pixelScale).
    // Accum texture is at scaled resolution; screen is at full resolution.
    // Map screen pixel → accum pixel by multiplying by the scale factor.
    let pixScale  = max(transform._pad, 0.01);
    let accumSize = vec2<i32>(textureDimensions(accumTex, 0));
    let px  = clamp(vec2<i32>(fragPos.xy * pixScale), vec2<i32>(0), accumSize - 1);
    let col = textureLoad(accumTex, px, 0).xyz;
    let t   = textureLoad(gBufTex,  px, 0).w;
    let exposure = transform.cameraPos.z;
    // Background pixels: just gamma-correct, no tone mapping.
    // Geometry pixels: exposure + ACES tone mapping + gamma.
    var gc: vec3<f32>;
    if (t <= 0.0) {
        gc = pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    } else {
        gc = pow(aces(col * exposure), vec3<f32>(1.0 / 2.2));
    }
    return vec4<f32>(gc, 1.0);
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
    float iRes[4];
    float tanHalfFov[4];
    float frameCount[4];
    float triCount[4];
    float mode[4];
    float lightCount[4];
    float lightPos[4][4];
    float lightCol[4][4];
    float lightType[4][4];      // [i][0]=type (0=pt,1=dir,2=spot), [i][1]=cosAngle, [i][2]=cosOuter
    float lightDir[4][4];       // [i] = spotlight direction xyz
    float    spp[4];
    uint32_t movedMeshBits[4];  // bit i = mesh i moved; words 0/1 = meshes 0–63
    float    envColor[4];       // xyz = color, w = mode (0=sky, 1=solid color, 2=equirect tex)
    float    envIntensity[4];   // x = intensity, y = envWidth, z = envHeight, w = totalLumSum (0 = no CDF)
    float    params[4];        // x = maxBounces, y = ambientFactor
    float    emissiveInfo[4]; // x = emissive tri count, y = total emissive area
};
static_assert(sizeof(RtGpuUniforms) == 512, "RtGpuUniforms must be 512 bytes");

struct alignas(16) VtGpuUniforms {
    uint32_t triCount, _p[3];
};
struct alignas(16) RefitGpuUniforms {
    uint32_t leafCount, _p[3];
};
struct alignas(16) AtrousGpuUniforms {
    uint32_t stepSize, _p[3];
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
    float iRes[4];
    float tanHalfFov[4];
    float frameCount[4];
    uint32_t movedMeshBits[4];
};
static_assert(sizeof(TaaGpuUniforms) == 192, "TaaGpuUniforms must be 192 bytes");

// ---------------------------------------------------------------------------
// BVH node (48 bytes, matches GPU BvhNodeGpu)
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

/// Build texture atlas sized to actual slot usage (not MAX_TEX_SLOTS).
/// Returns {pixel data, rows used (>= 1)}.
static std::pair<std::vector<unsigned char>, int> buildAtlas(
        const std::vector<Mesh*>& meshes,
        std::unordered_map<Texture*, int>& texSlotMap) {
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
    }

    const int rows = std::max(1, (slotCount + ATLAS_COLS - 1) / ATLAS_COLS);
    const int atlasW = ATLAS_COLS * TILE_SIZE;
    const int atlasH = rows * TILE_SIZE;
    std::vector<unsigned char> atlas(atlasW * atlasH * 4, 255);

    auto addTexture = [&](Texture* tex, int& slot) {
        if (!tex || slot >= MAX_TEX_SLOTS) return;
        if (texSlotMap.count(tex)) return;
        auto& img = tex->image();
        if (img.width == 0 || img.height == 0) return;
        const auto& src = img.data<unsigned char>();
        const int srcW = static_cast<int>(img.width);
        const int srcH = static_cast<int>(img.height);
        const int ch = static_cast<int>(src.size()) / (srcW * srcH);

        const int col = slot % ATLAS_COLS;
        const int row = slot / ATLAS_COLS;
        const int destX = col * TILE_SIZE;
        const int destY = row * TILE_SIZE;

        if (srcW == TILE_SIZE && srcH == TILE_SIZE && ch == 4) {
            // Fast path: direct row memcpy when source matches tile dimensions and has 4 channels
            for (int ty = 0; ty < TILE_SIZE; ++ty) {
                const int di = ((destY + ty) * atlasW + destX) * 4;
                const int si = ty * srcW * 4;
                std::memcpy(atlas.data() + di, src.data() + si, TILE_SIZE * 4);
            }
        } else {
            // Precompute X mapping table to avoid repeated division in inner loop
            int xmap[TILE_SIZE];
            for (int tx = 0; tx < TILE_SIZE; ++tx)
                xmap[tx] = tx * srcW / TILE_SIZE;
            for (int ty = 0; ty < TILE_SIZE; ++ty) {
                const int sy = ty * srcH / TILE_SIZE;
                const int srcRowOff = sy * srcW;
                unsigned char* dst = atlas.data() + ((destY + ty) * atlasW + destX) * 4;
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
    }
    return {std::move(atlas), rows};
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
};

static ExtractedMaterial extractMaterial(const Material* mat) {
    ExtractedMaterial m;
    if (!mat) return m;
    if (auto* c = dynamic_cast<const MaterialWithColor*>(mat))
        m.albedo = c->color;
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
    }
    if (auto* a = dynamic_cast<const MaterialWithAttenuation*>(mat)) {
        m.attenuationColor = a->attenuationColor;
        m.attenuationDistance = std::max(0.f, a->attenuationDistance);
    }
    if (auto* cc = dynamic_cast<const MaterialWithClearcoat*>(mat)) {
        m.clearcoat = std::clamp(cc->clearcoat, 0.f, 1.f);
        const float ccr = std::clamp(cc->clearcoatRoughness, 0.f, 1.f);
        m.clearcoatRoughness = std::max(1e-4f, ccr * ccr);
    }
    m.alphaTest = mat->alphaTest;
    return m;
}

static int buildGeometryBuffers(
        const std::vector<Mesh*>& meshes,
        const std::unordered_map<Texture*, int>& texSlotMap,
        std::vector<float>& triBuffer,
        std::vector<float>& matBuffer,
        std::vector<float>& rawObjTriBuf,
        std::vector<float>& matrixBuf,
        float emissiveScale,
        int maxTris, int maxMats, int maxMeshes) {
    std::ranges::fill(triBuffer, 0.f);
    std::ranges::fill(matBuffer, 0.f);
    std::ranges::fill(rawObjTriBuf, 0.f);
    std::ranges::fill(matrixBuf, 0.f);

    int triCount = 0;
    int matCount = 0;

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
        float* p = rawObjTriBuf.data() + ti * 32 + field * 4;
        p[0] = x; p[1] = y; p[2] = z; p[3] = w;
    };

    for (auto& mesh : meshes) {
        if (triCount >= maxTris || matCount >= maxMats) break;

        auto em = extractMaterial(mesh->material().get());
        setTexel(matBuffer, maxMats, matCount, 0,
                 em.albedo.r, em.albedo.g, em.albedo.b, em.shininess);

        float texSlot = -1.f;
        float normalSlot = -1.f;
        float texOffsetX = 0.f, texOffsetY = 0.f;
        float texRepeatX = 1.f, texRepeatY = 1.f;
        if (auto* mwm = dynamic_cast<MaterialWithMap*>(mesh->material().get())) {
            if (mwm->map) {
                auto it = texSlotMap.find(mwm->map.get());
                if (it != texSlotMap.end()) {
                    texSlot = static_cast<float>(it->second);
                    texOffsetX = mwm->map->offset.x;
                    texOffsetY = mwm->map->offset.y;
                    texRepeatX = mwm->map->repeat.x;
                    texRepeatY = mwm->map->repeat.y;
                }
            }
        }
        if (auto* mnm = dynamic_cast<MaterialWithNormalMap*>(mesh->material().get())) {
            if (mnm->normalMap) {
                auto it = texSlotMap.find(mnm->normalMap.get());
                if (it != texSlotMap.end()) {
                    normalSlot = static_cast<float>(it->second);
                }
            }
        }
        float roughSlot = -1.f;
        if (auto* mwr = dynamic_cast<MaterialWithRoughness*>(mesh->material().get())) {
            if (mwr->roughnessMap) {
                auto it = texSlotMap.find(mwr->roughnessMap.get());
                if (it != texSlotMap.end()) {
                    roughSlot = static_cast<float>(it->second);
                }
            }
        }
        setTexel(matBuffer, maxMats, matCount, 1, texSlot, em.metalness, normalSlot, roughSlot);
        setTexel(matBuffer, maxMats, matCount, 2,
                em.emissive.r * emissiveScale, em.emissive.g * emissiveScale, em.emissive.b * emissiveScale, em.transmission);
        setTexel(matBuffer, maxMats, matCount, 3, em.ior, em.alphaTest, 0.f, 0.f);
        setTexel(matBuffer, maxMats, matCount, 4,
                em.attenuationColor.r, em.attenuationColor.g, em.attenuationColor.b, em.attenuationDistance);
        setTexel(matBuffer, maxMats, matCount, 5, em.clearcoat, em.clearcoatRoughness, 0.f, 0.f);

        const int matIdx = matCount++;
        const int meshIdx = matIdx;

        mesh->updateWorldMatrix(true, true);
        const auto& world = *mesh->matrixWorld;
        Matrix4 normalMat(world);
        normalMat.invert().transpose();

        if (meshIdx < maxMeshes) {
            float* mp = matrixBuf.data() + meshIdx * 32;
            std::memcpy(mp, world.elements.data(), 16 * sizeof(float));
            std::memcpy(mp + 16, normalMat.elements.data(), 16 * sizeof(float));
        }

        auto* geo = mesh->geometry().get();
        auto* pos = geo->getAttribute<float>("position");
        if (!pos) continue;
        auto* nrm = geo->getAttribute<float>("normal");
        auto* uvs = geo->getAttribute<float>("uv");
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
            return {uvs->getX(i) * texRepeatX + texOffsetX,
                    uvs->getY(i) * texRepeatY + texOffsetY};
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

            setTexel(triBuffer, maxTris, triCount, 0, v0.x, v0.y, v0.z, static_cast<float>(matIdx));
            setTexel(triBuffer, maxTris, triCount, 1, v1.x, v1.y, v1.z, 0.f);
            setTexel(triBuffer, maxTris, triCount, 2, v2.x, v2.y, v2.z, 0.f);
            setTexel(triBuffer, maxTris, triCount, 3, n0.x, n0.y, n0.z, 0.f);
            setTexel(triBuffer, maxTris, triCount, 4, n1.x, n1.y, n1.z, 0.f);
            setTexel(triBuffer, maxTris, triCount, 5, n2.x, n2.y, n2.z, 0.f);
            setTexel(triBuffer, maxTris, triCount, 6, u0, v0uv, u1, v1uv);
            setTexel(triBuffer, maxTris, triCount, 7, u2, v2uv, 0.f, 0.f);

            const Vector3 ov0 = objVert(i0), ov1 = objVert(i1), ov2 = objVert(i2);
            const Vector3 on0 = objNorm(i0), on1 = objNorm(i1), on2 = objNorm(i2);
            setObj(triCount, 0, ov0.x, ov0.y, ov0.z, static_cast<float>(matIdx));
            setObj(triCount, 1, ov1.x, ov1.y, ov1.z, static_cast<float>(meshIdx));
            setObj(triCount, 2, ov2.x, ov2.y, ov2.z, 0.f);
            setObj(triCount, 3, on0.x, on0.y, on0.z, 0.f);
            setObj(triCount, 4, on1.x, on1.y, on1.z, 0.f);
            setObj(triCount, 5, on2.x, on2.y, on2.z, 0.f);
            setObj(triCount, 6, u0, v0uv, u1, v1uv);
            setObj(triCount, 7, u2, v2uv, 0.f, 0.f);

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

    // If no good split found, or splitting is more expensive than a leaf, make a leaf
    if (bestAxis < 0 || bestCost >= leafCost) {
        nodes[ni].left = -(start + 1);
        nodes[ni].right = count;
        return ni;
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

static int pagedIdx(int ti, int row) {
    return ((ti / TEX_PAGE_WIDTH * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + ti % TEX_PAGE_WIDTH) * 4;
}

static void packBvhNodeBuffer(const std::vector<BvhNode>& nodes, std::vector<float>& buf, int bvhCapacity) {
    std::ranges::fill(buf, 0.f);
    const int nc = std::min(static_cast<int>(nodes.size()), bvhCapacity);
    for (int i = 0; i < nc; i++) {
        const auto& n = nodes[i];
        float* p = buf.data() + i * 12;
        p[0] = n.minX; p[1] = n.minY; p[2] = n.minZ;
        std::memcpy(p + 3, &n.left, sizeof(int));
        p[4] = n.maxX; p[5] = n.maxY; p[6] = n.maxZ;
        std::memcpy(p + 7, &n.right, sizeof(int));
        std::memcpy(p + 8, &n.parent, sizeof(int));
    }
}

static void buildBVH(std::vector<float>& triBuffer, int triCount,
                     std::vector<BvhNode>& nodes, std::vector<int>& indices,
                     std::vector<int>& leafIndices,
                     std::vector<float>& rawObjTriBuf) {
    indices.resize(triCount);
    std::iota(indices.begin(), indices.end(), 0);
    nodes.clear();
    nodes.reserve(triCount * 2);
    buildBvhNode(nodes, indices, triBuffer, 0, triCount, -1);

    leafIndices.clear();
    for (int i = 0; i < static_cast<int>(nodes.size()); i++) {
        if (nodes[i].left < 0) leafIndices.push_back(i);
    }

    std::vector<float> sorted(triBuffer.size(), 0.f);
    std::vector<float> sortedObj(rawObjTriBuf.size(), 0.f);
    for (int ni = 0; ni < triCount; ni++) {
        const int oi = indices[ni];
        for (int row = 0; row < TRI_TEX_HEIGHT; row++)
            for (int c = 0; c < 4; c++)
                sorted[pagedIdx(ni, row) + c] = triBuffer[pagedIdx(oi, row) + c];
        std::memcpy(sortedObj.data() + ni * 32, rawObjTriBuf.data() + oi * 32, 32 * sizeof(float));
    }
    triBuffer = std::move(sorted);
    rawObjTriBuf = std::move(sortedObj);
}

}// anonymous namespace

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
    int atlasRows_ = 0;  // current atlas row count (0 = initial placeholder)
    WgpuTexture accumA;
    WgpuTexture accumB;
    WgpuTexture* readAccum;
    WgpuTexture* writeAccum;
    WgpuTexture hitMeshA;
    WgpuTexture hitMeshB;
    WgpuTexture* readHitMesh;
    WgpuTexture* writeHitMesh;
    WgpuTexture envTexGpu;   // equirectangular env map (1x1 placeholder when unused)
    WgpuTexture envCdfTex;  // conditional CDF (width × height), R32Float
    WgpuTexture envMargTex; // marginal CDF (height × 1), R32Float
    float       envLumSum_ = 0.f;  // total luminance sum for PDF normalization
    Texture*    prevEnvTex_ = nullptr;

    // G-buffer ping-pong
    WgpuTexture gBufA;          // normal.xyz + depth.w
    WgpuTexture gBufB;
    WgpuTexture* gBufCur;       // current frame writes here
    WgpuTexture* gBufPrev;      // previous frame's G-buffer

    // Albedo buffer (primary-hit albedo for demodulation/remodulation)
    WgpuTexture albedoTex;

    // TAA history ping-pong
    WgpuTexture taaHistA;       // previous TAA output
    WgpuTexture taaHistB;
    WgpuTexture* taaHistRead;
    WgpuTexture* taaHistWrite;

    // Spatial filter ping-pong
    WgpuTexture filteredA;
    WgpuTexture filteredB;

    // Denoiser pipelines
    WgpuComputePipeline taaPipeline;
    WgpuComputePipeline atrousPipeline;
    WgpuBuffer  taaUniBuf;
    WgpuBuffer  atrousUniBuf;
    bool denoiserEnabled_ = false;
    float envIntensity_ = 1.0f;
    float emissiveScale_ = 1.0f;
    int maxBounces_ = 6;
    float exposure_ = 1.0f;
    float ambientFactor_ = 0.03f;

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
    WgpuBuffer objTriBuf;
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

    // GPU uniform buffers
    WgpuBuffer vtUniBuf;
    WgpuBuffer refitUniBuf;
    WgpuBuffer rtUniformBuf;

    // Compute pipelines
    WgpuComputePipeline vtPipeline;
    WgpuComputePipeline refitPipeline;
    WgpuComputePipeline rtPipeline;
    WgpuComputePipeline rtRaycastPipeline;

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
    std::vector<float> bvhNodeCpuBuf;
    std::vector<uint32_t> bvhCounterZeros;

    // BVH state
    std::vector<BvhNode> bvhNodes;
    std::vector<int> bvhIndices;
    std::vector<int> leafIndices;

    // Async scene build result — CPU work done on background thread
    struct AsyncBuildResult {
        std::vector<unsigned char> atlasData;
        int atlasRows = 0;
        std::unordered_map<Texture*, int> texSlotMap;
        std::vector<float> triBuffer;
        std::vector<float> matBuffer;
        std::vector<float> rawObjTriBuf;
        std::vector<float> matrixCpuBuf;
        std::vector<BvhNode> bvhNodes;
        std::vector<int> bvhIndices;
        std::vector<int> leafIndices;
        std::vector<float> bvhNodeCpuBuf;
        std::vector<float> emissiveTriCpu;
        int triCount = 0;
        int numBvhNodes = 0;
        int emissiveTriCount = 0;
        float emissiveTotalArea = 0.f;
        float emissiveTotalPower = 0.f;
        std::vector<Mesh*> meshes;  // snapshot of mesh list
        // Capacities used for buffer sizing (next-power-of-2)
        int triCapacity = 0;
        int matCapacity = 0;
        int meshCapacity = 0;
        int bvhCapacity = 0;
    };
#ifndef __EMSCRIPTEN__
    std::future<AsyncBuildResult> asyncBuild_;
    bool buildPending_ = false;

    // Async env CDF build
    std::future<EnvCdfResult> asyncEnvCdf_;
    bool envCdfPending_ = false;
#endif
    bool shaderHasEnvCdf_ = false;  // tracks which shader variant is active

    // Frame state
    std::unordered_map<Texture*, int> texSlotMap;
    std::vector<Mesh*> prevMeshes;
    std::vector<Matrix4> prevMeshMatrices;
    int triCount_ = 0;
    int numBvhNodes_ = 0;
    float frameCount_ = 0.f;
    Vector3 prevCamPos_;
    Vector3 prevCamDir_;
    Mode mode_ = Mode::Raytracer;
    int spp_ = 2;
    int overlayLayer_ = -1;  // -1 = disabled; objects on this layer bypass path tracing and go to raster overlay

    int width_, height_;

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
                      WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
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
          envCdfTex(r, 1u, 1u, WgpuTexture::Format::R32Float,
                    WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          envMargTex(r, 1u, 1u, WgpuTexture::Format::R32Float,
                     WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          // G-buffer ping-pong
          gBufA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                WgpuTexture::Format::RGBA16Float),
          gBufB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                WgpuTexture::Format::RGBA16Float),
          gBufCur(&gBufA), gBufPrev(&gBufB),
          // Albedo buffer for demodulation
          albedoTex(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                    WgpuTexture::Format::RGBA16Float),
          // TAA history ping-pong
          taaHistA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                   WgpuTexture::Format::RGBA16Float),
          taaHistB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                   WgpuTexture::Format::RGBA16Float),
          taaHistRead(&taaHistA), taaHistWrite(&taaHistB),
          // Spatial filter ping-pong
          filteredA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                    WgpuTexture::Format::RGBA16Float),
          filteredB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                    WgpuTexture::Format::RGBA16Float),
          // Denoiser pipelines
          taaPipeline(r, taaWGSL, "taa_main"),
          atrousPipeline(r, svgfAtrousWGSL, "svgf_atrous_main"),
          taaUniBuf(r, sizeof(TaaGpuUniforms)),
          atrousUniBuf(r, sizeof(AtrousGpuUniforms)),
          // Storage buffers (small placeholders — grown dynamically on first build)
          bvhNodeBuf(r, static_cast<size_t>(2 * INIT_TRI_CAP - 1) * 12 * sizeof(float),
                     WgpuBuffer::Usage::Storage),
          bvhCounterBuf(r, static_cast<size_t>(2 * INIT_TRI_CAP - 1) * sizeof(uint32_t),
                        WgpuBuffer::Usage::Storage),
          objTriBuf(r, static_cast<size_t>(INIT_TRI_CAP) * 32 * sizeof(float),
                    WgpuBuffer::Usage::Storage),
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
          vtPipeline(r, vtWGSL, "vt_main"),
          refitPipeline(r, refitWGSL, "bvh_refit"),
          rtPipeline(r, buildRtShader(true, false), "rt_main"),
          rtRaycastPipeline(r, buildRtShader(false, false), "rt_main"),
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

        refitPipeline.setTexture(0, triTex);
        refitPipeline.setStorageBuffer(1, bvhNodeBuf);
        refitPipeline.setStorageBuffer(2, bvhCounterBuf);
        refitPipeline.setStorageBufferRead(3, leafIndexBuf);
        refitPipeline.setUniformBuffer(4, refitUniBuf);

        rtPipeline.setUniformBuffer(0, rtUniformBuf);
        rtPipeline.setStorageBufferRead(3, bvhNodeBuf);
        rtPipeline.setTexture(4, matTex);
        rtPipeline.setTexture(5, triTex);
        rtPipeline.setTexture(6, texAtlasTex);
        rtPipeline.setTexture(9, envTexGpu);
        rtPipeline.setStorageTexture(10, *gBufCur);
        rtPipeline.setStorageBufferRead(11, emissiveTriBuf);
        rtPipeline.setTexture(12, envCdfTex);
        rtPipeline.setTexture(13, envMargTex);
        rtPipeline.setStorageTexture(14, albedoTex);

        rtRaycastPipeline.setUniformBuffer(0, rtUniformBuf);
        rtRaycastPipeline.setStorageBufferRead(3, bvhNodeBuf);
        rtRaycastPipeline.setTexture(4, matTex);
        rtRaycastPipeline.setTexture(5, triTex);
        rtRaycastPipeline.setTexture(6, texAtlasTex);
        rtRaycastPipeline.setTexture(9, envTexGpu);
        rtRaycastPipeline.setStorageTexture(10, *gBufCur);
        rtRaycastPipeline.setStorageBufferRead(11, emissiveTriBuf);
        rtRaycastPipeline.setStorageTexture(14, albedoTex);

        // TAA pipeline: uniform buffer is static binding
        taaPipeline.setUniformBuffer(0, taaUniBuf);
        taaPipeline.setStorageBufferRead(6, motionMatBuf);

        // Spatial filter: uniform buffer + albedo are static bindings
        atrousPipeline.setUniformBuffer(0, atrousUniBuf);
        atrousPipeline.setTexture(4, albedoTex);

        // Zero-fill accumulators and SVGF textures
        {
            std::vector<float> zeros(w * h * 4, 0.f);
            accumA.write(zeros.data(), zeros.size() * sizeof(float));
            accumB.write(zeros.data(), zeros.size() * sizeof(float));
            gBufA.write(zeros.data(), zeros.size() * sizeof(float));
            gBufB.write(zeros.data(), zeros.size() * sizeof(float));
            taaHistA.write(zeros.data(), zeros.size() * sizeof(float));
            taaHistB.write(zeros.data(), zeros.size() * sizeof(float));
        }
        // Init albedo to white (no demodulation effect until first frame writes real values)
        {
            std::vector<float> ones(w * h * 4, 1.f);
            albedoTex.write(ones.data(), ones.size() * sizeof(float));
        }
        // Fill hitMesh textures with sentinel 64.0f (= "no hit")
        {
            std::vector<float> hitSentinel(w * h * 4, 64.f);
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

        gBufA = WgpuTexture(renderer, uw, uh, fmt);
        gBufB = WgpuTexture(renderer, uw, uh, fmt);
        gBufCur  = &gBufA;
        gBufPrev = &gBufB;

        albedoTex = WgpuTexture(renderer, uw, uh, fmt);

        taaHistA = WgpuTexture(renderer, uw, uh, fmt);
        taaHistB = WgpuTexture(renderer, uw, uh, fmt);
        taaHistRead  = &taaHistA;
        taaHistWrite = &taaHistB;

        filteredA = WgpuTexture(renderer, uw, uh, fmt);
        filteredB = WgpuTexture(renderer, uw, uh, fmt);

        std::vector<float> zeros(w * h * 4, 0.f);
        accumA.write(zeros.data(), zeros.size() * sizeof(float));
        accumB.write(zeros.data(), zeros.size() * sizeof(float));
        gBufA.write(zeros.data(), zeros.size() * sizeof(float));
        gBufB.write(zeros.data(), zeros.size() * sizeof(float));
        taaHistA.write(zeros.data(), zeros.size() * sizeof(float));
        taaHistB.write(zeros.data(), zeros.size() * sizeof(float));

        std::vector<float> hitSentinel(w * h * 4, 64.f);
        hitMeshA.write(hitSentinel.data(), hitSentinel.size() * sizeof(float));
        hitMeshB.write(hitSentinel.data(), hitSentinel.size() * sizeof(float));

        rtPipeline.setStorageTexture(10, *gBufCur);
        rtPipeline.setStorageTexture(14, albedoTex);
        rtRaycastPipeline.setStorageTexture(10, *gBufCur);
        rtRaycastPipeline.setStorageTexture(14, albedoTex);
        atrousPipeline.setTexture(4, albedoTex);

        frameCount_ = 0.f;
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
            (camPos - d.prevCamPos_).length() > 1e-5f ||
            (fwd - d.prevCamDir_).length() > 1e-5f;
    if (camMoved) {
        d.frameCount_ = 0.f;
        d.prevCamPos_ = camPos;
        d.prevCamDir_ = fwd;
    }

    // Collect visible, non-wireframe, non-line meshes and lights
    std::vector<Mesh*> rtMeshes;
    bool hasEnclosingBox = false;
    Color enclosingBoxColor;
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
        // Side::Back meshes (enclosing boxes) poison the BVH with scene-spanning
        // triangles.  Exclude them and use their color as the ray-miss background.
        if (mat->side == Side::Back) {
            hasEnclosingBox = true;
            if (auto* mc = mat->as<MaterialWithColor>()) enclosingBoxColor = mc->color;
            return;
        }
        // Transparent meshes with no texture and no transmission are raster-only
        // overlay effects (e.g. separate clearcoat geometry layers). They have no
        // physical meaning in path tracing and render as opaque shells that occlude
        // everything beneath them.
        if (mat->transparent) {
            auto* mwm = dynamic_cast<MaterialWithMap*>(mat);
            auto* mwt = dynamic_cast<MaterialWithTransmission*>(mat);
            const bool hasMap = mwm && mwm->map;
            const bool hasTransmission = mwt && mwt->transmission > 0.f;
            if (!hasMap && !hasTransmission) return;
        }
        rtMeshes.push_back(&m);
    });
    std::vector<PointLight*> pointLights;
    scene.traverseType<PointLight>([&](PointLight& l) { if (l.visible) pointLights.push_back(&l); });
    std::vector<DirectionalLight*> dirLights;
    scene.traverseType<DirectionalLight>([&](DirectionalLight& l) { if (l.visible) dirLights.push_back(&l); });
    std::vector<SpotLight*> spotLights;
    scene.traverseType<SpotLight>([&](SpotLight& l) { if (l.visible) spotLights.push_back(&l); });

    // Detect topology change
    const bool topoChanged = (rtMeshes != d.prevMeshes);

    // Build scene data (BVH, geometry buffers, atlas, emissives) when topology changes.
    // Native: async on background thread.  Emscripten: synchronous (no pthreads).
    bool topoJustFinished = false;
#ifdef __EMSCRIPTEN__
    if (topoChanged) {
        d.prevMeshes = rtMeshes;
        auto meshes = rtMeshes;
        float emissiveScale = d.emissiveScale_;

        Impl::AsyncBuildResult r;
        r.meshes = meshes;
        r.texSlotMap.clear();
        auto [atlasData, atlasRows] = buildAtlas(meshes, r.texSlotMap);
        r.atlasData = std::move(atlasData);
        r.atlasRows = atlasRows;

        int totalTris = 0;
        for (auto* mesh : meshes) {
            auto* geo = mesh->geometry().get();
            auto* idx = geo->getIndex();
            auto* pos = geo->getAttribute<float>("position");
            if (!pos) continue;
            totalTris += idx ? static_cast<int>(idx->count()) / 3
                             : static_cast<int>(pos->count()) / 3;
        }
        const int matCount = static_cast<int>(meshes.size());
        r.triCapacity  = nextPow2(std::max(totalTris, 1));
        r.matCapacity  = nextPow2(std::max(matCount, 1));
        r.meshCapacity = r.matCapacity;
        r.bvhCapacity  = 2 * r.triCapacity - 1;

        const int pages = triTexPages(r.triCapacity);
        r.triBuffer.resize(static_cast<size_t>(TEX_PAGE_WIDTH) * TRI_TEX_HEIGHT * pages * 4, 0.f);
        r.matBuffer.resize(static_cast<size_t>(r.matCapacity) * MAT_TEX_HEIGHT * 4, 0.f);
        r.rawObjTriBuf.resize(static_cast<size_t>(r.triCapacity) * 32, 0.f);
        r.matrixCpuBuf.resize(static_cast<size_t>(r.meshCapacity) * 32, 0.f);

        r.triCount = buildGeometryBuffers(meshes, r.texSlotMap, r.triBuffer, r.matBuffer,
                                           r.rawObjTriBuf, r.matrixCpuBuf, emissiveScale,
                                           r.triCapacity, r.matCapacity, r.meshCapacity);
        buildBVH(r.triBuffer, r.triCount, r.bvhNodes, r.bvhIndices, r.leafIndices, r.rawObjTriBuf);
        r.numBvhNodes = static_cast<int>(r.bvhNodes.size());
        r.bvhNodeCpuBuf.resize(static_cast<size_t>(r.bvhCapacity) * 12, 0.f);
        packBvhNodeBuffer(r.bvhNodes, r.bvhNodeCpuBuf, r.bvhCapacity);

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
                }
            }
        }
        topoJustFinished = true;
#else
    if (topoChanged && !d.buildPending_) {
        d.buildPending_ = true;
        d.prevMeshes = rtMeshes;

        auto meshes = rtMeshes;
        float emissiveScale = d.emissiveScale_;
        d.asyncBuild_ = std::async(std::launch::async, [meshes, emissiveScale]() {
            Impl::AsyncBuildResult r;
            r.meshes = meshes;
            r.texSlotMap.clear();
            auto [atlasData, atlasRows] = buildAtlas(meshes, r.texSlotMap);
            r.atlasData = std::move(atlasData);
            r.atlasRows = atlasRows;

            int totalTris = 0;
            for (auto* mesh : meshes) {
                auto* geo = mesh->geometry().get();
                auto* idx = geo->getIndex();
                auto* pos = geo->getAttribute<float>("position");
                if (!pos) continue;
                totalTris += idx ? static_cast<int>(idx->count()) / 3
                                 : static_cast<int>(pos->count()) / 3;
            }
            const int matCount = static_cast<int>(meshes.size());
            r.triCapacity  = nextPow2(std::max(totalTris, 1));
            r.matCapacity  = nextPow2(std::max(matCount, 1));
            r.meshCapacity = r.matCapacity;
            r.bvhCapacity  = 2 * r.triCapacity - 1;

            const int pages = triTexPages(r.triCapacity);
            r.triBuffer.resize(static_cast<size_t>(TEX_PAGE_WIDTH) * TRI_TEX_HEIGHT * pages * 4, 0.f);
            r.matBuffer.resize(static_cast<size_t>(r.matCapacity) * MAT_TEX_HEIGHT * 4, 0.f);
            r.rawObjTriBuf.resize(static_cast<size_t>(r.triCapacity) * 32, 0.f);
            r.matrixCpuBuf.resize(static_cast<size_t>(r.meshCapacity) * 32, 0.f);

            r.triCount = buildGeometryBuffers(meshes, r.texSlotMap, r.triBuffer, r.matBuffer,
                                               r.rawObjTriBuf, r.matrixCpuBuf, emissiveScale,
                                               r.triCapacity, r.matCapacity, r.meshCapacity);
            buildBVH(r.triBuffer, r.triCount, r.bvhNodes, r.bvhIndices, r.leafIndices, r.rawObjTriBuf);
            r.numBvhNodes = static_cast<int>(r.bvhNodes.size());
            r.bvhNodeCpuBuf.resize(static_cast<size_t>(r.bvhCapacity) * 12, 0.f);
            packBvhNodeBuffer(r.bvhNodes, r.bvhNodeCpuBuf, r.bvhCapacity);

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
                    }
                }
            }
            return r;
        });
    }

    // Check if async build finished — upload results to GPU (must happen on main thread)
    if (d.buildPending_ && d.asyncBuild_.valid() &&
        d.asyncBuild_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {

        auto r = d.asyncBuild_.get();
        d.buildPending_ = false;
        topoJustFinished = true;
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
        d.emissiveTriCpu = std::move(r.emissiveTriCpu);
        d.triCount_ = r.triCount;
        d.numBvhNodes_ = r.numBvhNodes;
        d.emissiveTriCount_ = r.emissiveTriCount;
        d.emissiveTotalArea_ = r.emissiveTotalArea;
        d.emissiveTotalPower_ = r.emissiveTotalPower;

        // Grow GPU buffers when scene exceeds current capacity (same pattern as atlas)
        if (r.triCapacity > d.triCapacity_) {
            d.triCapacity_ = r.triCapacity;
            const int pages = triTexPages(r.triCapacity);
            d.triTex = WgpuTexture(d.renderer, TEX_PAGE_WIDTH, TRI_TEX_HEIGHT * pages,
                                    WgpuTexture::Format::RGBA32Float,
                                    WgpuTexture::Storage | WgpuTexture::TextureBinding);
            d.objTriBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.triCapacity) * 32 * sizeof(float),
                                      WgpuBuffer::Usage::Storage);
            d.leafIndexBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.triCapacity) * sizeof(int),
                                         WgpuBuffer::Usage::Storage);
            d.emissiveTriBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.triCapacity) * 4 * sizeof(float),
                                           WgpuBuffer::Usage::Storage);
            d.vtPipeline.setStorageBufferRead(0, d.objTriBuf);
            d.vtPipeline.setStorageTexture(2, d.triTex);
            d.refitPipeline.setTexture(0, d.triTex);
            d.refitPipeline.setStorageBufferRead(3, d.leafIndexBuf);
            d.rtPipeline.setTexture(5, d.triTex);
            d.rtPipeline.setStorageBufferRead(11, d.emissiveTriBuf);
            d.rtRaycastPipeline.setTexture(5, d.triTex);
            d.rtRaycastPipeline.setStorageBufferRead(11, d.emissiveTriBuf);
        }
        if (r.bvhCapacity > d.bvhCapacity_) {
            d.bvhCapacity_ = r.bvhCapacity;
            d.bvhNodeBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.bvhCapacity) * 12 * sizeof(float),
                                       WgpuBuffer::Usage::Storage);
            d.bvhCounterBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.bvhCapacity) * sizeof(uint32_t),
                                          WgpuBuffer::Usage::Storage);
            d.bvhCounterZeros.resize(r.bvhCapacity, 0u);
            d.refitPipeline.setStorageBuffer(1, d.bvhNodeBuf);
            d.refitPipeline.setStorageBuffer(2, d.bvhCounterBuf);
            d.rtPipeline.setStorageBufferRead(3, d.bvhNodeBuf);
            d.rtRaycastPipeline.setStorageBufferRead(3, d.bvhNodeBuf);
        }
        if (r.matCapacity > d.matCapacity_) {
            d.matCapacity_ = r.matCapacity;
            d.matTex = WgpuTexture(d.renderer, r.matCapacity, MAT_TEX_HEIGHT,
                                    WgpuTexture::Format::RGBA32Float,
                                    WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
            d.rtPipeline.setTexture(4, d.matTex);
            d.rtRaycastPipeline.setTexture(4, d.matTex);
        }
        if (r.meshCapacity > d.meshCapacity_) {
            d.meshCapacity_ = r.meshCapacity;
            d.matrixBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.meshCapacity) * 32 * sizeof(float),
                                      WgpuBuffer::Usage::Storage);
            d.motionMatBuf = WgpuBuffer(d.renderer, static_cast<size_t>(r.meshCapacity) * 16 * sizeof(float),
                                         WgpuBuffer::Usage::Storage);
            d.vtPipeline.setStorageBufferRead(1, d.matrixBuf);
            d.taaPipeline.setStorageBufferRead(6, d.motionMatBuf);
        }

        // Upload atlas
        if (r.atlasRows != d.atlasRows_) {
            d.atlasRows_ = r.atlasRows;
            d.texAtlasTex = WgpuTexture(d.renderer,
                    ATLAS_COLS * TILE_SIZE, r.atlasRows * TILE_SIZE,
                    WgpuTexture::Format::RGBA8Unorm,
                    WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
            d.rtPipeline.setTexture(6, d.texAtlasTex);
            d.rtRaycastPipeline.setTexture(6, d.texAtlasTex);
        }
        d.texAtlasTex.write(r.atlasData.data(), r.atlasData.size());

        // Upload geometry + BVH
        d.bvhNodeBuf.write(d.bvhNodeCpuBuf.data(), d.numBvhNodes_ * 12 * sizeof(float));
        d.objTriBuf.write(d.rawObjTriBuf.data(), static_cast<size_t>(d.triCount_) * 32 * sizeof(float));
        d.leafIndexBuf.write(d.leafIndices.data(), d.leafIndices.size() * sizeof(int));
        d.matTex.write(d.matBuffer.data(), d.matBuffer.size() * sizeof(float));

        if (d.emissiveTriCount_ > 0) {
            d.emissiveTriBuf.write(d.emissiveTriCpu.data(),
                                   d.emissiveTriCpu.size() * sizeof(float));
        }
    }

#ifndef __EMSCRIPTEN__
    // While building, skip RT dispatch — just return so the window stays responsive
    if (d.buildPending_) {
        return;
    }
#endif

    scene.updateMatrixWorld();

    // Detect per-mesh matrix changes; build bitmask of which meshes moved.
    // movedBits is used by the GPU for per-pixel accumulation reset.
    // anyMeshMoved drives the vertex-transform and BVH-refit pipelines.
    uint32_t movedBits[2] = {0u, 0u};
    bool anyMeshMoved = (d.prevMeshMatrices.size() != rtMeshes.size());

    if (topoJustFinished) {
        // Topology change: all pixels need to re-accumulate (mesh-to-triangle mapping changed)
        movedBits[0] = movedBits[1] = 0xFFFFFFFFu;
        anyMeshMoved = true;
    } else if (anyMeshMoved) {
        // Mesh count mismatch (shouldn't happen without topo change, but be safe)
        movedBits[0] = movedBits[1] = 0xFFFFFFFFu;
    } else {
        for (size_t i = 0; i < rtMeshes.size() && i < static_cast<size_t>(d.meshCapacity_); ++i) {
            if (*rtMeshes[i]->matrixWorld != d.prevMeshMatrices[i]) {
                anyMeshMoved = true;
                if (i < 32u) movedBits[0] |= (1u << i);
                else         movedBits[1] |= (1u << (i - 32u));
            }
        }
    }
    // Note: camMoved resets all pixels via fc==0u in the shader; no movedBits needed for that.

    // Compute per-mesh motion matrices: prevWorld * inverse(curWorld)
    // Used by TAA to reproject pixels on moving objects to their previous-frame screen position.
    d.motionMatCpu.resize(static_cast<size_t>(d.meshCapacity_) * 16, 0.f);
    for (size_t i = 0; i < rtMeshes.size() && i < static_cast<size_t>(d.meshCapacity_); ++i) {
        Matrix4 mot;  // identity by default
        if (i < d.prevMeshMatrices.size()) {
            Matrix4 curInv(*rtMeshes[i]->matrixWorld);
            curInv.invert();
            mot.multiplyMatrices(d.prevMeshMatrices[i], curInv);
        }
        // else: identity (new mesh, no previous frame)
        std::memcpy(d.motionMatCpu.data() + i * 16, mot.elements.data(), 16 * sizeof(float));
    }
    d.motionMatBuf.write(d.motionMatCpu.data(), d.motionMatCpu.size() * sizeof(float));

    d.prevMeshMatrices.resize(rtMeshes.size());
    for (size_t i = 0; i < rtMeshes.size(); ++i)
        d.prevMeshMatrices[i] = *rtMeshes[i]->matrixWorld;
    if (anyMeshMoved) {
        if (!topoChanged) {
            std::ranges::fill(d.matrixCpuBuf, 0.f);
            int mi = 0;
            for (auto* mesh : rtMeshes) {
                if (mi >= d.meshCapacity_) break;
                const auto& w = *mesh->matrixWorld;
                Matrix4 nm(w);
                nm.invert().transpose();
                float* p = d.matrixCpuBuf.data() + mi * 32;
                std::memcpy(p, w.elements.data(), 16 * sizeof(float));
                std::memcpy(p + 16, nm.elements.data(), 16 * sizeof(float));
                ++mi;
            }
        }
        d.matrixBuf.write(d.matrixCpuBuf.data(), static_cast<size_t>(d.meshCapacity_) * 32 * sizeof(float));

        VtGpuUniforms vtU{};
        vtU.triCount = static_cast<uint32_t>(d.triCount_);
        d.vtUniBuf.write(&vtU, sizeof(vtU));

        RefitGpuUniforms rfU{};
        rfU.leafCount = static_cast<uint32_t>(d.leafIndices.size());
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
    u.iRes[0] = static_cast<float>(d.width_);
    u.iRes[1] = static_cast<float>(d.height_);
    u.tanHalfFov[0] = tanHalfFov;
    u.frameCount[0] = d.frameCount_;
    u.triCount[0] = static_cast<float>(d.triCount_);
    u.mode[0] = (d.mode_ == Mode::PathTracer) ? 1.f : 0.f;
    u.spp[0] = static_cast<float>(d.spp_);
    u.movedMeshBits[0] = movedBits[0];
    u.movedMeshBits[1] = movedBits[1];

    // Environment light from scene.background / scene.environment
    u.envColor[3] = 0.f;  // default: sky gradient
    if (auto* s = dynamic_cast<Scene*>(&scene)) {
        // Priority: environment texture > background texture > background color
        Texture* envTex = s->environment.get();
        if (!envTex && s->background.isTexture())
            envTex = s->background.texture().get();

        if (envTex) {
            if (envTex != d.prevEnvTex_) {
                // Upload new equirectangular texture (LDR or HDR)
                auto& img = envTex->image();
                if (img.width > 0 && img.height > 0) {
                    bool isHdr = false;
                    try { (void)img.data<float>(); isHdr = true; }
                    catch (const std::bad_variant_access&) {}

                    if (isHdr) {
                        const auto& src = img.data<float>();
                        d.envTexGpu = WgpuTexture(d.renderer,
                                                  static_cast<uint32_t>(img.width),
                                                  static_cast<uint32_t>(img.height),
                                                  WgpuTexture::Format::RGBA32Float,
                                                  WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
                        d.envTexGpu.write(src.data(), src.size() * sizeof(float));
                    } else {
                        const auto& src = img.data<unsigned char>();
                        d.envTexGpu = WgpuTexture(d.renderer,
                                                  static_cast<uint32_t>(img.width),
                                                  static_cast<uint32_t>(img.height),
                                                  WgpuTexture::Format::RGBA8Unorm,
                                                  WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
                        d.envTexGpu.write(src.data(), src.size());
                    }
                    d.rtPipeline.setTexture(9, d.envTexGpu);
                    d.rtRaycastPipeline.setTexture(9, d.envTexGpu);

                    // Build env CDF for importance sampling.
                    // Native: async on background thread.  Emscripten: synchronous.
                    int w = img.width, h = img.height;
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
                        d.rtPipeline.replaceShader(buildRtShader(true, true));
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
        } else if (s->background.isColor()) {
            const Color& c = s->background.color();
            u.envColor[0] = c.r; u.envColor[1] = c.g; u.envColor[2] = c.b;
            u.envColor[3] = 1.f;
            d.prevEnvTex_ = nullptr;
            d.envLumSum_ = 0.f;  // no env map = no importance sampling
            if (d.shaderHasEnvCdf_) {
                d.rtPipeline.replaceShader(buildRtShader(true, false));
                d.shaderHasEnvCdf_ = false;
            }
        }
    }
    // Enclosing box overrides environment: use its color as solid-color miss background
    if (hasEnclosingBox) {
        u.envColor[0] = enclosingBoxColor.r;
        u.envColor[1] = enclosingBoxColor.g;
        u.envColor[2] = enclosingBoxColor.b;
        u.envColor[3] = 1.f;  // solid color mode
        d.envLumSum_ = 0.f;
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
            d.rtPipeline.replaceShader(buildRtShader(true, true));
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
    u.params[1] = d.ambientFactor_;
    u.params[2] = 0.f;
    u.params[3] = 0.f;
    u.emissiveInfo[0] = static_cast<float>(d.emissiveTriCount_);
    u.emissiveInfo[1] = d.emissiveTotalPower_;

    int nLights = 0;
    auto packLight = [&](float px, float py, float pz, float r, float g, float b, float type) {
        if (nLights >= 4) return;
        u.lightPos[nLights][0] = px; u.lightPos[nLights][1] = py; u.lightPos[nLights][2] = pz;
        u.lightCol[nLights][0] = r;  u.lightCol[nLights][1] = g;  u.lightCol[nLights][2] = b;
        u.lightType[nLights][0] = type;
        ++nLights;
    };
    for (auto* l : pointLights) {
        const auto& lp = l->position;
        const auto& lc = l->color;
        const float li = l->intensity;
        packLight(lp.x, lp.y, lp.z, lc.r * li, lc.g * li, lc.b * li, 0.f);
    }
    for (auto* l : dirLights) {
        Vector3 dir = Vector3(l->position).sub(l->target().position).normalize();
        const auto& lc = l->color;
        const float li = l->intensity;
        packLight(dir.x, dir.y, dir.z, lc.r * li, lc.g * li, lc.b * li, 1.f);
    }
    for (auto* l : spotLights) {
        if (nLights >= 4) break;
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
        u.lightDir[nLights][0] = dir.x; u.lightDir[nLights][1] = dir.y; u.lightDir[nLights][2] = dir.z;
        ++nLights;
    }
    u.lightCount[0] = static_cast<float>(nLights);
    d.rtUniformBuf.write(&u, sizeof(u));

    // Set per-frame texture bindings (accum + hitMesh ping-pong)
    const bool isPT = d.mode_ == Mode::PathTracer;
    auto& activePipeline = isPT ? d.rtPipeline : d.rtRaycastPipeline;
    activePipeline.setTexture(1, *d.readAccum);
    activePipeline.setStorageTexture(2, *d.writeAccum);
    activePipeline.setTexture(7, *d.readHitMesh);
    activePipeline.setStorageTexture(8, *d.writeHitMesh);

    // Batched GPU dispatch — single command encoder + compute pass
    {
        WGPUCommandEncoderDescriptor encDesc{};
        encDesc.label = WGPUStringView{"pt_enc", WGPU_STRLEN};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(d.device, &encDesc);

        WGPUComputePassDescriptor passDesc{};
        passDesc.label = WGPUStringView{"pt_pass", WGPU_STRLEN};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &passDesc);

        if (anyMeshMoved) {
            const uint32_t vtGx = (static_cast<uint32_t>(d.triCount_) + 63u) / 64u;
            d.vtPipeline.encode(pass, vtGx);

            const uint32_t rfGx = (static_cast<uint32_t>(d.leafIndices.size()) + 63u) / 64u;
            d.refitPipeline.encode(pass, rfGx);
        }

        const uint32_t gx = (static_cast<uint32_t>(d.width_) + 7u) / 8u;
        const uint32_t gy = (static_cast<uint32_t>(d.height_) + 7u) / 8u;
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

    // Swap ping-pong (accum + hitMesh)
    std::swap(d.readAccum, d.writeAccum);
    std::swap(d.readHitMesh, d.writeHitMesh);

    // Swap gBuf for next frame
    std::swap(d.gBufCur, d.gBufPrev);
    d.rtPipeline.setStorageTexture(10, *d.gBufCur);
    d.rtRaycastPipeline.setStorageTexture(10, *d.gBufCur);

    // TAA + spatial denoiser (path tracer mode only)
    // Skip entirely when fully converged — accumulation is already clean.
    WgpuTexture* displayTex = d.readAccum;
    const bool hasMotion = (movedBits[0] | movedBits[1]) != 0u;
    const bool needsDenoise = d.frameCount_ < 64.f || hasMotion;
    if (d.denoiserEnabled_ && d.mode_ == Mode::PathTracer && needsDenoise) {
        const uint32_t gx = (static_cast<uint32_t>(d.width_)  + 7u) / 8u;
        const uint32_t gy = (static_cast<uint32_t>(d.height_) + 7u) / 8u;

        // --- TAA pass: reproject + clamp + blend ---
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
        d.taaUniBuf.write(&tu, sizeof(tu));

        d.taaPipeline.setTexture(1, *d.readAccum);
        d.taaPipeline.setTexture(2, *d.gBufPrev);   // current frame's gBuf (after swap)
        d.taaPipeline.setTexture(3, *d.taaHistRead);
        d.taaPipeline.setStorageTexture(4, *d.taaHistWrite);
        d.taaPipeline.setTexture(5, *d.readHitMesh);
        d.taaPipeline.dispatch(gx, gy);

        std::swap(d.taaHistRead, d.taaHistWrite);

        // --- Spatial filter on TAA output ---
        WgpuTexture* src = d.taaHistRead;  // TAA output (after swap)
        WgpuTexture* dst = &d.filteredA;
        WgpuTexture* aux = &d.filteredB;

        // 5×5 kernel covers more area per pass, so fewer passes needed.
        // Scale down as image converges: sigma shrinks, filter becomes transparent.
        const int nPasses = (d.frameCount_ < 4.f)                ? 3 :
                            (d.frameCount_ < 32.f || hasMotion)   ? 2 : 1;

        for (int pass = 0; pass < nPasses; ++pass) {
            AtrousGpuUniforms au{static_cast<uint32_t>(1u << pass), {0u, 0u, 0u}};
            d.atrousUniBuf.write(&au, sizeof(au));
            d.atrousPipeline.setTexture(1, *src);
            d.atrousPipeline.setStorageTexture(2, *dst);
            d.atrousPipeline.setTexture(3, *d.gBufPrev);
            d.atrousPipeline.setTexture(5, *d.readHitMesh);
            d.atrousPipeline.dispatch(gx, gy);
            src = dst;
            std::swap(dst, aux);
        }
        displayTex = src;

        // Store camera for next frame's reprojection
        d.prevCamOri_[0] = camPos.x; d.prevCamOri_[1] = camPos.y; d.prevCamOri_[2] = camPos.z;
        d.prevCamFwd_[0] = fwd.x;    d.prevCamFwd_[1] = fwd.y;    d.prevCamFwd_[2] = fwd.z;
        d.prevCamRgt_[0] = rgt.x;    d.prevCamRgt_[1] = rgt.y;    d.prevCamRgt_[2] = rgt.z;
        d.prevCamUp_[0]  = up.x;     d.prevCamUp_[1]  = up.y;     d.prevCamUp_[2]  = up.z;
    }

    d.displayMat->customTextures["accumTex"] = displayTex;
    d.displayMat->customTextures["gBufTex"]  = d.gBufPrev;  // used to skip ACES on background pixels
    d.displayMat->uniformsNeedUpdate = true;
    d.frameCount_ += 1.f;

    // Pass exposure to display shader via ortho camera z-position (read as transform.cameraPos.z).
    // Default z=1.0 matches identity exposure; ortho projection is unaffected by z translation.
    d.displayCam.position.z = d.exposure_;

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

void WgpuPathTracer::setMode(Mode mode) {
    if (pimpl_->mode_ != mode) {
        pimpl_->mode_ = mode;
        pimpl_->frameCount_ = 0.f;
    }
}

WgpuPathTracer::Mode WgpuPathTracer::mode() const {
    return pimpl_->mode_;
}

void WgpuPathTracer::setEnvIntensity(float intensity) {
    pimpl_->envIntensity_ = intensity;
}

float WgpuPathTracer::envIntensity() const {
    return pimpl_->envIntensity_;
}

void WgpuPathTracer::setEmissiveScale(float scale) {
    pimpl_->emissiveScale_ = scale;
    pimpl_->prevMeshes.clear();
}

float WgpuPathTracer::emissiveScale() const {
    return pimpl_->emissiveScale_;
}

void WgpuPathTracer::setMaxBounces(int bounces) {
    bounces = std::max(1, std::min(bounces, 32));
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

void WgpuPathTracer::setAmbientFactor(float factor) {
    if (pimpl_->ambientFactor_ != factor) {
        pimpl_->ambientFactor_ = factor;
        pimpl_->frameCount_ = 0.f;
    }
}

float WgpuPathTracer::ambientFactor() const {
    return pimpl_->ambientFactor_;
}

void WgpuPathTracer::setDenoiserEnabled(bool enabled) {
    pimpl_->denoiserEnabled_ = enabled;
}

bool WgpuPathTracer::denoiserEnabled() const {
    return pimpl_->denoiserEnabled_;
}

void WgpuPathTracer::setSamplesPerPixel(int spp) {
    pimpl_->spp_ = (spp >= 4) ? 4 : (spp >= 2) ? 2 : 1;
}

int WgpuPathTracer::samplesPerPixel() const {
    return pimpl_->spp_;
}

void WgpuPathTracer::setSize(std::pair<int, int> size) {
    pimpl_->fullWidth_ = size.first;
    pimpl_->fullHeight_ = size.second;
    const int sw = std::max(1, static_cast<int>(size.first * pimpl_->pixelScale_));
    const int sh = std::max(1, static_cast<int>(size.second * pimpl_->pixelScale_));
    pimpl_->recreateAccumTextures(sw, sh);
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
    }
}

float WgpuPathTracer::pixelScale() const {
    return pimpl_->pixelScale_;
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

void WgpuPathTracer::markDirty() {
    pimpl_->prevMeshes.clear();
}

void WgpuPathTracer::dispose() {
    auto& d = *pimpl_;
    if (d.depthFillPipeline_)   { wgpuRenderPipelineRelease(d.depthFillPipeline_);   d.depthFillPipeline_ = nullptr; }
    if (d.depthFillPipeLayout_) { wgpuPipelineLayoutRelease(d.depthFillPipeLayout_); d.depthFillPipeLayout_ = nullptr; }
    if (d.depthFillBGL_)        { wgpuBindGroupLayoutRelease(d.depthFillBGL_);       d.depthFillBGL_ = nullptr; }
    if (d.depthFillShader_)     { wgpuShaderModuleRelease(d.depthFillShader_);       d.depthFillShader_ = nullptr; }
    if (d.depthFillUniBuf_)     { wgpuBufferRelease(d.depthFillUniBuf_);             d.depthFillUniBuf_ = nullptr; }
}
