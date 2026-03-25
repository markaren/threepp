
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
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include <webgpu/webgpu.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <vector>

using namespace threepp;

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------
namespace {

constexpr int MAX_TRIS = 131072;
constexpr int MAX_MATS = 64;
constexpr int MAX_BVH_NODES = 2 * MAX_TRIS - 1;
constexpr int MAX_TEX_SLOTS = 16;
constexpr int TILE_SIZE = 1024;
constexpr int ATLAS_COLS = 8;  // tiles per row in atlas (8 × 1024 = 8192 ≤ GPU max)
constexpr int ATLAS_ROWS = (MAX_TEX_SLOTS + ATLAS_COLS - 1) / ATLAS_COLS;
constexpr int TRI_TEX_HEIGHT = 8;
constexpr int MAT_TEX_HEIGHT = 3;
constexpr int TEX_PAGE_WIDTH = 8192;
constexpr int TRI_TEX_PAGES = (MAX_TRIS + TEX_PAGE_WIDTH - 1) / TEX_PAGE_WIDTH;
constexpr int MAX_MESHES = 64;

// ---------------------------------------------------------------------------
// WGSL compute shader — path tracer + accumulator
// ---------------------------------------------------------------------------
constexpr const char* csWGSL = R"(

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

const TRI_PAGE_W:  i32 = 8192;
const TRI_PAGE_H:  i32 = 8;
const TILE_SIZE:   i32 = 1024;
const MAX_TEX_SLOTS: i32 = 16;
const ATLAS_COLS:  i32 = 8;

fn triCoord(ti: i32, row: i32) -> vec2<i32> {
    return vec2<i32>(ti % TRI_PAGE_W, (ti / TRI_PAGE_W) * TRI_PAGE_H + row);
}

struct Ray  { origin: vec3<f32>, dir: vec3<f32> }
struct Isect { t: f32, u: f32, v: f32 }
struct Hit  {
    t:         f32,
    point:     vec3<f32>,
    normal:    vec3<f32>,
    albedo:    vec3<f32>,
    shininess: f32,
    uv:        vec2<f32>,
    texSlot:   f32,
    metalness: f32,
    emissive:  vec3<f32>,
    meshIdx:   i32,
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

fn sampleGGXDir(wo: vec3<f32>, n: vec3<f32>, alpha: f32,
                seed: ptr<function, u32>) -> vec3<f32> {
    let u1   = rand(seed);
    let u2   = rand(seed);
    let a2   = alpha * alpha;
    let phi  = 2.0 * PI * u1;
    let cosT = sqrt((1.0 - u2) / max(1.0 + (a2 - 1.0) * u2, 1e-6));
    let sinT = sqrt(max(0.0, 1.0 - cosT * cosT));
    let nt   = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.99);
    let rgt  = normalize(cross(nt, n));
    let up   = cross(n, rgt);
    let hm   = normalize(sinT * cos(phi) * rgt + sinT * sin(phi) * up + cosT * n);
    return reflect(-wo, hm);
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
    let c00 = textureLoad(texAtlas, vec2<i32>(ox + x0, oy + y0), 0).xyz;
    let c10 = textureLoad(texAtlas, vec2<i32>(ox + x1, oy + y0), 0).xyz;
    let c01 = textureLoad(texAtlas, vec2<i32>(ox + x0, oy + y1), 0).xyz;
    let c11 = textureLoad(texAtlas, vec2<i32>(ox + x1, oy + y1), 0).xyz;
    return mix(mix(c00, c10, wx), mix(c01, c11, wx), wy);
}

fn sampleEnv(d: vec3<f32>) -> vec3<f32> {
    let mode = i32(rt.envColor.w);
    if (mode == 1) {
        // Solid background color
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

    var finalNorm = select(sn, -sn, dot(ray.dir, sn) > 0.0);

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

    (*h).t         = isect.t;
    (*h).point     = ray.origin + isect.t * ray.dir;
    (*h).normal    = finalNorm;
    (*h).albedo    = mat0.xyz;
    (*h).shininess = shininess;
    (*h).uv        = iuv;
    (*h).texSlot   = mat1.x;
    (*h).metalness = mat1.y;
    (*h).emissive  = mat2.xyz;
    (*h).meshIdx   = i32(r1.w);
}

fn sceneHit(ray: Ray) -> Hit {
    var h: Hit; h.t = 1e30; h.meshIdx = -1;
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

fn shade(h: Hit, rd: vec3<f32>) -> vec3<f32> {
    var albedo = h.albedo;
    if (h.texSlot >= 0.0) { albedo = sampleAtlas(h.uv, h.texSlot); }
    var col = albedo * 0.02;
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
        let sh = sceneHit(sr);
        if (sh.t >= ld - 1e-3) {
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
            col += lc * (f_diff + f_spec) * NdotL;
        }
    }
    return clamp(col + h.emissive, vec3<f32>(0.0), vec3<f32>(1.0));
}
)"
R"(
fn raytrace(ray: Ray) -> vec3<f32> {
    let h0 = sceneHit(ray);
    if (h0.t >= 1e30) { return sampleEnv(ray.dir); }
    var col = shade(h0, ray.dir);

    // Specular mirror bounces (two levels, unrolled — deterministic, no seed needed).
    if (h0.shininess < 0.5) {
        let F0_0   = mix(vec3<f32>(0.04), h0.albedo, h0.metalness);
        let NdotV0 = max(0.001, dot(h0.normal, normalize(-ray.dir)));
        let k0     = schlick(NdotV0, F0_0) * max(0.0, 1.0 - h0.shininess * 2.0);

        var r1: Ray;
        r1.origin = h0.point + h0.normal * 1e-3;
        r1.dir    = reflect(ray.dir, h0.normal);
        let h1    = sceneHit(r1);

        var rc1: vec3<f32>;
        if (h1.t >= 1e30) {
            rc1 = sampleEnv(r1.dir);
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
                var rc2   = select(shade(h2, r2.dir), sampleEnv(r2.dir), h2.t >= 1e30);
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

fn pathTrace(ray_in: Ray, seed: ptr<function, u32>,
             primaryMeshIdx: ptr<function, u32>,
             primaryNormal:  ptr<function, vec3<f32>>,
             primaryDepth:   ptr<function, f32>) -> vec3<f32> {
    *primaryMeshIdx = 64u;
    *primaryNormal  = vec3<f32>(0.0);
    *primaryDepth   = 0.0;  // 0 = sky/no-hit sentinel for denoiser
    var ray        = ray_in;
    var throughput = vec3<f32>(1.0);
    var radiance   = vec3<f32>(0.0);

    for (var i = 0; i < 8; i++) {
        let h = sceneHit(ray);
        if (h.t >= 1e29) {
            radiance += throughput * sampleEnv(ray.dir);
            break;
        }
        if (i == 0) {
            *primaryMeshIdx = u32(h.meshIdx);
            *primaryNormal  = h.normal;
            *primaryDepth   = h.t;
        }

        radiance += throughput * h.emissive;

        var albedo = h.albedo;
        if (h.texSlot >= 0.0) { albedo = sampleAtlas(h.uv, h.texSlot); }

        let lcount = i32(rt.lightCount.x);
        let wo = normalize(-ray.dir);
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
            let sh = sceneHit(sr);
            if (sh.t >= ld - 1e-3) {
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
                radiance  += throughput * (f_diff + f_spec) * NdotL * lc;
            }
        }
        radiance += throughput * albedo * 0.03;

        if (i > 0) {
            let p = max(max(throughput.r, throughput.g), throughput.b);
            if (rand(seed) > p) { break; }
            throughput /= p;
        }

        let wo_b = normalize(-ray.dir);
        let F0_b = mix(vec3<f32>(0.04), albedo, h.metalness);
        var wi_b: vec3<f32>;
        // Importance-sample the lobe type by metalness:
        // metals are pure specular (p_spec→1), dielectrics split 50/50.
        let p_spec = mix(0.5, 0.98, h.metalness);
        if (rand(seed) < p_spec) {
            wi_b = sampleGGXDir(wo_b, h.normal, h.shininess, seed);
            let cos_b = dot(h.normal, wi_b);
            if (cos_b <= 0.0) { break; }
            let hb  = normalize(wo_b + wi_b);
            let Fb  = schlick(max(0.0, dot(wo_b, hb)), F0_b);
            let G1L = ggxG1(cos_b, h.shininess);
            throughput *= Fb * G1L / p_spec;
        } else {
            wi_b = cosineHemisphere(h.normal, seed);
            let cos_b = dot(h.normal, wi_b);
            if (cos_b <= 0.0) { break; }
            // Diffuse contribution is zero for pure metals, scales with (1-metalness)
            throughput *= albedo * (1.0 - h.metalness) / (1.0 - p_spec);
        }
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
    let isPT = rt.mode.x > 0.5;

    var sample: vec3<f32>;

    if (!isPT) {
        let spp = i32(rt.spp.x);
        let fp  = vec2<f32>(f32(pixel.x), f32(pixel.y));
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
    } else {
        var seed = pcg(gid.x * 1973u + 1u) ^ pcg(gid.y * 9277u + 1u) ^ pcg(fc * 26699u + 1u);
        let jx  = rand(&seed) - 0.5;
        let jy  = rand(&seed) - 0.5;
        let ray = makeRay(vec2<f32>(f32(pixel.x) + jx, f32(pixel.y) + jy), res);
        var primaryMeshIdx: u32;
        var primaryNormal:  vec3<f32>;
        var primaryDepth:   f32;
        sample  = pathTrace(ray, &seed, &primaryMeshIdx, &primaryNormal, &primaryDepth);

        let prev     = textureLoad(accumRead, pixel, 0);
        let old      = prev.xyz;
        var pixelFC  = prev.w;

        // Mesh-identity reset
        let prevHitMesh = u32(textureLoad(hitMeshRead, pixel, 0).r);
        if (prevHitMesh < 64u) {
            let bit   = prevHitMesh & 31u;
            let mbits = select(rt.movedMeshBits.x, rt.movedMeshBits.y,
                               prevHitMesh >= 32u);
            if (((mbits >> bit) & 1u) != 0u) { pixelFC = 0.0; }
        }
        if (primaryMeshIdx < 64u) {
            let bit   = primaryMeshIdx & 31u;
            let mbits = select(rt.movedMeshBits.x, rt.movedMeshBits.y,
                               primaryMeshIdx >= 32u);
            if (((mbits >> bit) & 1u) != 0u) { pixelFC = 0.0; }
        }
        if (fc == 0u) { pixelFC = 0.0; }

        let alpha   = 1.0 / (pixelFC + 1.0);
        let blended = old * (1.0 - alpha) + sample * alpha;
        textureStore(accumWrite,   pixel, vec4<f32>(blended, pixelFC + 1.0));
        textureStore(hitMeshWrite, pixel, vec4<f32>(f32(primaryMeshIdx), 0.0, 0.0, 0.0));
        textureStore(gBufWrite,    pixel, vec4<f32>(primaryNormal, primaryDepth));
    }
}
)";

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

    // Build 3×3 neighborhood color box for clamping
    var cMin = curColor;
    var cMax = curColor;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sp = clamp(pixel + vec2<i32>(dx, dy), vec2<i32>(0), iRes - 1);
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
    let toPoint  = worldPos - taa.prevCamOri.xyz;
    let prevZ    = dot(toPoint, taa.prevCamFwd.xyz);

    var useHist = prevZ > 0.001;
    let prevNdcX = dot(toPoint, taa.prevCamRgt.xyz) / (prevZ * taa.tanHalfFov.x * aspect);
    let prevNdcY = dot(toPoint, taa.prevCamUp.xyz)  / (prevZ * taa.tanHalfFov.x);
    let prevPx   = vec2<f32>((prevNdcX + 1.0) * 0.5 * res.x - 0.5,
                              (1.0 - prevNdcY) * 0.5 * res.y - 0.5);
    let prevPixel = vec2<i32>(i32(round(prevPx.x)), i32(round(prevPx.y)));

    // Bounds check
    useHist = useHist && prevPixel.x >= 0 && prevPixel.x < iRes.x
                      && prevPixel.y >= 0 && prevPixel.y < iRes.y;

    // Invalidate for moved meshes
    let meshIdx = u32(textureLoad(hitMeshTex, pixel, 0).r);
    if (meshIdx < 64u) {
        let bit   = meshIdx & 31u;
        let mbits = select(taa.movedMeshBits.x, taa.movedMeshBits.y, meshIdx >= 32u);
        if (((mbits >> bit) & 1u) != 0u) { useHist = false; }
    }

    // First frame has no history
    if (taa.frameCount.x < 0.5) { useHist = false; }

    var result = curColor;
    if (useHist) {
        let histColor = textureLoad(taaHistIn, prevPixel, 0).xyz;
        let clamped   = clamp(histColor, cMin, cMax);
        // Blend: use more history early (noisy), more current once converged
        let alpha = clamp(1.0 / (curFC * 0.5 + 1.0), 0.1, 1.0);
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

fn luminance_a(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

@compute @workgroup_size(8, 8)
fn svgf_atrous_main(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    // Sky pixels: pass through
    if (cDepth < 1e-6) {
        textureStore(colorOut, pixel, vec4<f32>(cColor, cFC));
        return;
    }

    let cLum = luminance_a(cColor);
    // lumSigma adapts to convergence: relaxed when noisy, tight when converged
    let lumSigma = max(0.02, 0.3 / sqrt(cFC + 1.0));

    // 3×3 bilateral filter with edge-preserving weights
    var colorSum  = vec3<f32>(0.0);
    var weightSum = 0.0;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let sp     = clamp(pixel + vec2<i32>(dx, dy) * step, vec2<i32>(0), res - 1);
            let sColor = textureLoad(colorIn, sp, 0).xyz;
            let sGB    = textureLoad(gBuf, sp, 0);
            let sNorm  = sGB.xyz;
            let sDepth = sGB.w;

            // Normal edge-stopping
            let w_n = pow(max(0.0, dot(cNorm, sNorm)), 64.0);
            // Depth edge-stopping
            let w_d = exp(-abs(cDepth - sDepth) * 2.0 / (cDepth + 0.01));
            // Luminance edge-stopping
            let sLum = luminance_a(sColor);
            let w_l  = exp(-(sLum - cLum) * (sLum - cLum) / (lumSigma * lumSigma + 1e-6));

            let w = w_n * w_d * w_l;
            colorSum  += sColor * w;
            weightSum += w;
        }
    }

    let filtered = select(cColor, colorSum / weightSum, weightSum > 1e-6);
    textureStore(colorOut, pixel, vec4<f32>(filtered, cFC));
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
    let col = textureLoad(accumTex, vec2<i32>(fragPos.xy), 0).xyz;
    let gc  = pow(aces(col), vec3<f32>(1.0 / 2.2));
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
    float    _pad[12];
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

static std::vector<unsigned char> buildAtlas(
        const std::vector<Mesh*>& meshes,
        std::unordered_map<Texture*, int>& texSlotMap) {
    const int atlasW = ATLAS_COLS * TILE_SIZE;
    const int atlasH = ATLAS_ROWS * TILE_SIZE;
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
        for (int ty = 0; ty < TILE_SIZE; ++ty) {
            const int sy = ty * srcH / TILE_SIZE;
            for (int tx = 0; tx < TILE_SIZE; ++tx) {
                const int sx = tx * srcW / TILE_SIZE;
                const int si = (sy * srcW + sx) * ch;
                const int di = ((destY + ty) * atlasW + destX + tx) * 4;
                atlas[di + 0] = src[si + 0];
                atlas[di + 1] = src[si + 1];
                atlas[di + 2] = src[si + 2];
                atlas[di + 3] = ch == 4 ? src[si + 3] : 255u;
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
    return atlas;
}

static std::tuple<Color, float, float, Color> extractMaterial(const Material* mat) {
    Color albedo(0.8f, 0.8f, 0.8f);
    float shininess = 8.f;
    float metalness = 0.f;
    Color emissive(0.f, 0.f, 0.f);
    if (!mat) return {albedo, shininess, metalness, emissive};
    if (auto* c = dynamic_cast<const MaterialWithColor*>(mat))
        albedo = c->color;
    if (auto* r = dynamic_cast<const MaterialWithRoughness*>(mat)) {
        const float rough = std::max(0.f, std::min(1.f, r->roughness));
        shininess = std::max(1e-4f, rough * rough);
    } else if (auto* s = dynamic_cast<const MaterialWithSpecular*>(mat)) {
        const float n = std::max(1.f, s->shininess);
        shininess = std::max(0.04f, std::sqrt(2.f / (n + 2.f)));
    }
    if (auto* m = dynamic_cast<const MaterialWithMetalness*>(mat))
        metalness = std::max(0.f, std::min(1.f, m->metalness));
    if (auto* e = dynamic_cast<const MaterialWithEmissive*>(mat))
        emissive = Color(e->emissive.r * e->emissiveIntensity,
                         e->emissive.g * e->emissiveIntensity,
                         e->emissive.b * e->emissiveIntensity);
    return {albedo, shininess, metalness, emissive};
}

static int buildGeometryBuffers(
        const std::vector<Mesh*>& meshes,
        const std::unordered_map<Texture*, int>& texSlotMap,
        std::vector<float>& triBuffer,
        std::vector<float>& matBuffer,
        std::vector<float>& rawObjTriBuf,
        std::vector<float>& matrixBuf) {
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
        if (triCount >= MAX_TRIS || matCount >= MAX_MATS) break;

        auto [albedo, shininess, metalness, emissive] = extractMaterial(mesh->material().get());
        setTexel(matBuffer, MAX_MATS, matCount, 0,
                 albedo.r, albedo.g, albedo.b, shininess);

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
        setTexel(matBuffer, MAX_MATS, matCount, 1, texSlot, metalness, normalSlot, roughSlot);
        setTexel(matBuffer, MAX_MATS, matCount, 2, emissive.r, emissive.g, emissive.b, 0.f);

        const int matIdx = matCount++;
        const int meshIdx = matIdx;

        mesh->updateWorldMatrix(true, true);
        const auto& world = *mesh->matrixWorld;
        Matrix4 normalMat(world);
        normalMat.invert().transpose();

        if (meshIdx < MAX_MESHES) {
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
        for (int i = 0; i < nTris && triCount < MAX_TRIS; ++i) {
            const int i0 = vi(i, 0), i1 = vi(i, 1), i2 = vi(i, 2);
            const Vector3 v0 = vert(i0), v1 = vert(i1), v2 = vert(i2);
            const Vector3 n0 = norm(i0), n1 = norm(i1), n2 = norm(i2);
            const auto [u0, v0uv] = uv(i0);
            const auto [u1, v1uv] = uv(i1);
            const auto [u2, v2uv] = uv(i2);

            setTexel(triBuffer, MAX_TRIS, triCount, 0, v0.x, v0.y, v0.z, static_cast<float>(matIdx));
            setTexel(triBuffer, MAX_TRIS, triCount, 1, v1.x, v1.y, v1.z, 0.f);
            setTexel(triBuffer, MAX_TRIS, triCount, 2, v2.x, v2.y, v2.z, 0.f);
            setTexel(triBuffer, MAX_TRIS, triCount, 3, n0.x, n0.y, n0.z, 0.f);
            setTexel(triBuffer, MAX_TRIS, triCount, 4, n1.x, n1.y, n1.z, 0.f);
            setTexel(triBuffer, MAX_TRIS, triCount, 5, n2.x, n2.y, n2.z, 0.f);
            setTexel(triBuffer, MAX_TRIS, triCount, 6, u0, v0uv, u1, v1uv);
            setTexel(triBuffer, MAX_TRIS, triCount, 7, u2, v2uv, 0.f, 0.f);

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

static int buildBvhNode(
        std::vector<BvhNode>& nodes,
        std::vector<int>& idx,
        const std::vector<float>& buf,
        int start, int end, int parentIdx = -1) {
    const int ni = static_cast<int>(nodes.size());
    nodes.emplace_back();

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
    if (count <= 4) {
        nodes[ni].left = -(start + 1);
        nodes[ni].right = count;
        return ni;
    }

    constexpr int NB = 8;
    struct Bucket {
        float mnX = 1e30f, mnY = 1e30f, mnZ = 1e30f;
        float mxX = -1e30f, mxY = -1e30f, mxZ = -1e30f;
        int cnt = 0;
    };

    float bestCost = 1e30f;
    int bestAxis = 0;
    int bestSplit = NB / 2;

    const float nodeArea = boxSurfaceArea(minX, minY, minZ, maxX, maxY, maxZ);

    for (int axis = 0; axis < 3; axis++) {
        const float axMin = (axis == 0) ? minX : (axis == 1 ? minY : minZ);
        const float axMax = (axis == 0) ? maxX : (axis == 1 ? maxY : maxZ);
        if (axMax - axMin < 1e-6f) continue;
        const float scale = NB / (axMax - axMin);

        Bucket buckets[NB];
        for (int i = start; i < end; i++) {
            const int ti = idx[i];
            const float c = (triGet(buf, ti, 0, axis) + triGet(buf, ti, 1, axis) + triGet(buf, ti, 2, axis)) / 3.f;
            const int bi = std::clamp(static_cast<int>((c - axMin) * scale), 0, NB - 1);
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

        for (int s = 1; s < NB; s++) {
            float lmnX = 1e30f, lmnY = 1e30f, lmnZ = 1e30f;
            float lmxX = -1e30f, lmxY = -1e30f, lmxZ = -1e30f;
            int lcnt = 0;
            for (int b = 0; b < s; b++) {
                if (!buckets[b].cnt) continue;
                lmnX = std::min(lmnX, buckets[b].mnX);
                lmnY = std::min(lmnY, buckets[b].mnY);
                lmnZ = std::min(lmnZ, buckets[b].mnZ);
                lmxX = std::max(lmxX, buckets[b].mxX);
                lmxY = std::max(lmxY, buckets[b].mxY);
                lmxZ = std::max(lmxZ, buckets[b].mxZ);
                lcnt += buckets[b].cnt;
            }
            float rmnX = 1e30f, rmnY = 1e30f, rmnZ = 1e30f;
            float rmxX = -1e30f, rmxY = -1e30f, rmxZ = -1e30f;
            int rcnt = 0;
            for (int b = s; b < NB; b++) {
                if (!buckets[b].cnt) continue;
                rmnX = std::min(rmnX, buckets[b].mnX);
                rmnY = std::min(rmnY, buckets[b].mnY);
                rmnZ = std::min(rmnZ, buckets[b].mnZ);
                rmxX = std::max(rmxX, buckets[b].mxX);
                rmxY = std::max(rmxY, buckets[b].mxY);
                rmxZ = std::max(rmxZ, buckets[b].mxZ);
                rcnt += buckets[b].cnt;
            }
            if (!lcnt || !rcnt) continue;

            const float cost = (static_cast<float>(lcnt) * boxSurfaceArea(lmnX, lmnY, lmnZ, lmxX, lmxY, lmxZ) + static_cast<float>(rcnt) * boxSurfaceArea(rmnX, rmnY, rmnZ, rmxX, rmxY, rmxZ)) / nodeArea;
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestSplit = s;
            }
        }
    }

    const float axMin = (bestAxis == 0) ? minX : (bestAxis == 1 ? minY : minZ);
    const float axMax = (bestAxis == 0) ? maxX : (bestAxis == 1 ? maxY : maxZ);
    const float splitPos = axMin + (axMax - axMin) * static_cast<float>(bestSplit) / NB;

    auto mid = std::partition(idx.begin() + start, idx.begin() + end, [&](int ti) {
        const float c = (triGet(buf, ti, 0, bestAxis) + triGet(buf, ti, 1, bestAxis) + triGet(buf, ti, 2, bestAxis)) / 3.f;
        return c < splitPos;
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

static void packBvhNodeBuffer(const std::vector<BvhNode>& nodes, std::vector<float>& buf) {
    std::ranges::fill(buf, 0.f);
    const int nc = std::min(static_cast<int>(nodes.size()), MAX_BVH_NODES);
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
    WgpuTexture accumA;
    WgpuTexture accumB;
    WgpuTexture* readAccum;
    WgpuTexture* writeAccum;
    WgpuTexture hitMeshA;
    WgpuTexture hitMeshB;
    WgpuTexture* readHitMesh;
    WgpuTexture* writeHitMesh;
    WgpuTexture envTexGpu;   // equirectangular env map (1x1 placeholder when unused)
    Texture*    prevEnvTex_ = nullptr;

    // G-buffer ping-pong
    WgpuTexture gBufA;          // normal.xyz + depth.w
    WgpuTexture gBufB;
    WgpuTexture* gBufCur;       // current frame writes here
    WgpuTexture* gBufPrev;      // previous frame's G-buffer

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
    bool denoiserEnabled_ = true;

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
    WgpuBuffer leafIndexBuf;

    // GPU uniform buffers
    WgpuBuffer vtUniBuf;
    WgpuBuffer refitUniBuf;
    WgpuBuffer rtUniformBuf;

    // Compute pipelines
    WgpuComputePipeline vtPipeline;
    WgpuComputePipeline refitPipeline;
    WgpuComputePipeline rtPipeline;

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

    int width_, height_;

    Impl(WgpuRenderer& r, int w, int h)
        : renderer(r),
          device(static_cast<WGPUDevice>(r.nativeDevice())),
          queue(static_cast<WGPUQueue>(r.nativeQueue())),
          // Geometry textures
          triTex(r, TEX_PAGE_WIDTH, TRI_TEX_HEIGHT * TRI_TEX_PAGES,
                 WgpuTexture::Format::RGBA32Float,
                 WgpuTexture::Storage | WgpuTexture::TextureBinding),
          matTex(r, MAX_MATS, MAT_TEX_HEIGHT,
                 WgpuTexture::Format::RGBA32Float,
                 WgpuTexture::TextureBinding | WgpuTexture::CopyDst),
          texAtlasTex(r, ATLAS_COLS * TILE_SIZE, ATLAS_ROWS * TILE_SIZE,
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
          // G-buffer ping-pong
          gBufA(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                WgpuTexture::Format::RGBA16Float),
          gBufB(r, static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                WgpuTexture::Format::RGBA16Float),
          gBufCur(&gBufA), gBufPrev(&gBufB),
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
          // Storage buffers
          bvhNodeBuf(r, static_cast<size_t>(MAX_BVH_NODES) * 12 * sizeof(float),
                     WgpuBuffer::Usage::Storage),
          bvhCounterBuf(r, static_cast<size_t>(MAX_BVH_NODES) * sizeof(uint32_t),
                        WgpuBuffer::Usage::Storage),
          objTriBuf(r, static_cast<size_t>(MAX_TRIS) * 32 * sizeof(float),
                    WgpuBuffer::Usage::Storage),
          matrixBuf(r, static_cast<size_t>(MAX_MESHES) * 32 * sizeof(float),
                    WgpuBuffer::Usage::Storage),
          leafIndexBuf(r, static_cast<size_t>(MAX_TRIS) * sizeof(int),
                       WgpuBuffer::Usage::Storage),
          // Uniform buffers
          vtUniBuf(r, sizeof(VtGpuUniforms)),
          refitUniBuf(r, sizeof(RefitGpuUniforms)),
          rtUniformBuf(r, sizeof(RtGpuUniforms)),
          // Compute pipelines
          vtPipeline(r, vtWGSL, "vt_main"),
          refitPipeline(r, refitWGSL, "bvh_refit"),
          rtPipeline(r, csWGSL, "rt_main"),
          // Display pipeline
          displayCam(-1.f, 1.f, 1.f, -1.f, 0.1f, 10.f),
          // CPU buffers
          triBuffer(TEX_PAGE_WIDTH * TRI_TEX_HEIGHT * TRI_TEX_PAGES * 4, 0.f),
          matBuffer(MAX_MATS * MAT_TEX_HEIGHT * 4, 0.f),
          rawObjTriBuf(static_cast<size_t>(MAX_TRIS) * 32, 0.f),
          matrixCpuBuf(static_cast<size_t>(MAX_MESHES) * 32, 0.f),
          bvhNodeCpuBuf(static_cast<size_t>(MAX_BVH_NODES) * 12, 0.f),
          bvhCounterZeros(MAX_BVH_NODES, 0u),
          width_(w), height_(h) {

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

        // TAA pipeline: uniform buffer is static binding
        taaPipeline.setUniformBuffer(0, taaUniBuf);

        // Spatial filter: uniform buffer is static binding
        atrousPipeline.setUniformBuffer(0, atrousUniBuf);

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
        displayScene.add(Mesh::create(PlaneGeometry::create(2.f, 2.f), displayMat));
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

    // Collect visible meshes and lights
    std::vector<Mesh*> rtMeshes;
    scene.traverseType<Mesh>([&](Mesh& m) {
        if (m.visible) rtMeshes.push_back(&m);
    });
    std::vector<PointLight*> pointLights;
    scene.traverseType<PointLight>([&](PointLight& l) { pointLights.push_back(&l); });
    std::vector<DirectionalLight*> dirLights;
    scene.traverseType<DirectionalLight>([&](DirectionalLight& l) { dirLights.push_back(&l); });
    std::vector<SpotLight*> spotLights;
    scene.traverseType<SpotLight>([&](SpotLight& l) { spotLights.push_back(&l); });

    // Detect topology change
    const bool topoChanged = (rtMeshes != d.prevMeshes);

    if (topoChanged) {
        d.texSlotMap.clear();
        const auto atlasData = buildAtlas(rtMeshes, d.texSlotMap);
        d.texAtlasTex.write(atlasData.data(), atlasData.size());
        d.prevMeshes = rtMeshes;
    }

    scene.updateMatrixWorld();

    // Detect per-mesh matrix changes; build bitmask of which meshes moved.
    // movedBits is used by the GPU for per-pixel accumulation reset.
    // anyMeshMoved drives the vertex-transform and BVH-refit pipelines.
    uint32_t movedBits[2] = {0u, 0u};
    bool anyMeshMoved = (d.prevMeshMatrices.size() != rtMeshes.size());

    if (topoChanged) {
        // Topology change: all pixels need to re-accumulate (mesh-to-triangle mapping changed)
        movedBits[0] = movedBits[1] = 0xFFFFFFFFu;
        anyMeshMoved = true;
    } else if (anyMeshMoved) {
        // Mesh count mismatch (shouldn't happen without topo change, but be safe)
        movedBits[0] = movedBits[1] = 0xFFFFFFFFu;
    } else {
        for (size_t i = 0; i < rtMeshes.size() && i < static_cast<size_t>(MAX_MESHES); ++i) {
            if (*rtMeshes[i]->matrixWorld != d.prevMeshMatrices[i]) {
                anyMeshMoved = true;
                if (i < 32u) movedBits[0] |= (1u << i);
                else         movedBits[1] |= (1u << (i - 32u));
            }
        }
    }
    // Note: camMoved resets all pixels via fc==0u in the shader; no movedBits needed for that.
    d.prevMeshMatrices.resize(rtMeshes.size());
    for (size_t i = 0; i < rtMeshes.size(); ++i)
        d.prevMeshMatrices[i] = *rtMeshes[i]->matrixWorld;

    // Three-tier geometry update
    if (topoChanged) {
        d.triCount_ = buildGeometryBuffers(rtMeshes, d.texSlotMap, d.triBuffer, d.matBuffer,
                                            d.rawObjTriBuf, d.matrixCpuBuf);
        buildBVH(d.triBuffer, d.triCount_, d.bvhNodes, d.bvhIndices, d.leafIndices, d.rawObjTriBuf);
        d.numBvhNodes_ = static_cast<int>(d.bvhNodes.size());
        packBvhNodeBuffer(d.bvhNodes, d.bvhNodeCpuBuf);
        d.bvhNodeBuf.write(d.bvhNodeCpuBuf.data(), d.numBvhNodes_ * 12 * sizeof(float));
        d.objTriBuf.write(d.rawObjTriBuf.data(), static_cast<size_t>(d.triCount_) * 32 * sizeof(float));
        d.leafIndexBuf.write(d.leafIndices.data(), d.leafIndices.size() * sizeof(int));
        d.matTex.write(d.matBuffer.data(), d.matBuffer.size() * sizeof(float));
        anyMeshMoved = true;
    }
    if (anyMeshMoved) {
        if (!topoChanged) {
            std::ranges::fill(d.matrixCpuBuf, 0.f);
            int mi = 0;
            for (auto* mesh : rtMeshes) {
                if (mi >= MAX_MESHES) break;
                const auto& w = *mesh->matrixWorld;
                Matrix4 nm(w);
                nm.invert().transpose();
                float* p = d.matrixCpuBuf.data() + mi * 32;
                std::memcpy(p, w.elements.data(), 16 * sizeof(float));
                std::memcpy(p + 16, nm.elements.data(), 16 * sizeof(float));
                ++mi;
            }
        }
        d.matrixBuf.write(d.matrixCpuBuf.data(), static_cast<size_t>(MAX_MESHES) * 32 * sizeof(float));

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
                }
                d.prevEnvTex_ = envTex;
            }
            u.envColor[3] = 2.f;
        } else if (s->background.isColor()) {
            const Color& c = s->background.color();
            u.envColor[0] = c.r; u.envColor[1] = c.g; u.envColor[2] = c.b;
            u.envColor[3] = 1.f;
            d.prevEnvTex_ = nullptr;
        }
    }

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
        Vector3 dir = Vector3(l->target().position).sub(l->position).normalize();
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
    d.rtPipeline.setTexture(1, *d.readAccum);
    d.rtPipeline.setStorageTexture(2, *d.writeAccum);
    d.rtPipeline.setTexture(7, *d.readHitMesh);
    d.rtPipeline.setStorageTexture(8, *d.writeHitMesh);

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
        d.rtPipeline.encode(pass, gx, gy);

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

    // TAA + spatial denoiser (path tracer mode only)
    WgpuTexture* displayTex = d.readAccum;
    if (d.denoiserEnabled_ && d.mode_ == Mode::PathTracer) {
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

        const int nPasses = (d.frameCount_ < 4.f)  ? 3 :
                            (d.frameCount_ < 20.f) ? 2 : 1;

        for (int pass = 0; pass < nPasses; ++pass) {
            AtrousGpuUniforms au{static_cast<uint32_t>(1u << pass), {0u, 0u, 0u}};
            d.atrousUniBuf.write(&au, sizeof(au));
            d.atrousPipeline.setTexture(1, *src);
            d.atrousPipeline.setStorageTexture(2, *dst);
            d.atrousPipeline.setTexture(3, *d.gBufPrev);
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
    d.displayMat->uniformsNeedUpdate = true;
    d.frameCount_ += 1.f;

    // Blit to screen
    d.renderer.render(d.displayScene, d.displayCam);
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
    pimpl_->recreateAccumTextures(size.first, size.second);
}

std::pair<int, int> WgpuPathTracer::size() const {
    return {pimpl_->width_, pimpl_->height_};
}

int WgpuPathTracer::frameCount() const {
    return static_cast<int>(pimpl_->frameCount_);
}

void WgpuPathTracer::resetAccumulation() {
    pimpl_->frameCount_ = 0.f;
}

void WgpuPathTracer::dispose() {
    // Resources are cleaned up by destructors
}
