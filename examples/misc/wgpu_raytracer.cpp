// GPU path tracer that reads real threepp geometry (BufferGeometry) and renders it
// entirely via a WGSL compute shader using WgpuComputePipeline.
//
// Architecture:
//   Each frame a compute shader traces one Monte Carlo path per pixel and blends
//   the result into a per-pixel accumulation texture (running average):
//
//     accumulation[pixel] = (accumulation[pixel] * frameCount + newSample) / (frameCount+1)
//
//   Two RGBA32Float textures ping-pong as accumulator (read one, write the other).
//   A simple blit fragment shader displays the accumulated result with gamma correction.
//   Moving the camera resets frameCount → accumulation restarts.
//
//   Geometry is packed into RGBA32Float WgpuTextures every frame (BVH rebuilt on CPU):
//     triData  rows 0-5: v0 v1 v2 n0 n1 n2  (paged, 8192 columns wide)
//     matData  row  0  : (albedo.r, albedo.g, albedo.b, shininess)
//     bvhData  rows 0-1: (aabbMin.xyz, left)  (aabbMax.xyz, right)  per node

#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
#include "threepp/renderers/wgpu/WgpuBuffer.hpp"
#include "threepp/renderers/wgpu/WgpuComputePipeline.hpp"
#include "threepp/renderers/wgpu/WgpuTexture.hpp"
#include "threepp/threepp.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <vector>

using namespace threepp;

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------
constexpr int MAX_TRIS = 65536;
constexpr int MAX_MATS = 64;
constexpr int MAX_BVH_NODES = 2 * MAX_TRIS - 1;
constexpr int MAX_TEX_SLOTS = 16;
constexpr int TILE_SIZE = 512;   // 16*512=8192=max WebGPU tex width; also update WGSL TILE_SIZE!
constexpr int TRI_TEX_HEIGHT = 8;// rows: v0 v1 v2 n0 n1 n2 uv01 uv2
constexpr int MAT_TEX_HEIGHT = 3;// row0: albedo+shininess, row1: texSlot+metalness, row2: emissive.rgb
constexpr int TEX_PAGE_WIDTH = 8192;
constexpr int TRI_TEX_PAGES = (MAX_TRIS + TEX_PAGE_WIDTH - 1) / TEX_PAGE_WIDTH;
constexpr int BVH_TEX_PAGES = (MAX_BVH_NODES + TEX_PAGE_WIDTH - 1) / TEX_PAGE_WIDTH;

// ---------------------------------------------------------------------------
// WGSL compute shader — path tracer + accumulator
//
// Bindings:
//   0: RtUniforms (uniform buffer)
//   1: accumRead  (texture_2d<f32>)
//   2: accumWrite (texture_storage_2d<rgba32float, write>)
//   3: bvhData    (texture_2d<f32>)
//   4: matData    (texture_2d<f32>)
//   5: triData    (texture_2d<f32>)
//   6: texAtlas   (texture_2d<f32>, RGBA8Unorm strip of TILE_SIZE×TILE_SIZE tiles)
// ---------------------------------------------------------------------------
constexpr const char* csWGSL = R"(

struct RtUniforms {
    camOri:     vec4<f32>,           // xyz = camera position
    camFwd:     vec4<f32>,           // xyz = forward direction
    camRgt:     vec4<f32>,           // xyz = right direction
    camUp:      vec4<f32>,           // xyz = up direction
    iRes:       vec4<f32>,           // xy  = viewport size in pixels
    tanHalfFov: vec4<f32>,           // x   = tan(fov/2)
    frameCount: vec4<f32>,           // x   = accumulated frames so far (float)
    triCount:   vec4<f32>,           // x   = number of triangles
    mode:       vec4<f32>,           // x   = 0 raytracer, 1 path tracer
    lightCount: vec4<f32>,           // x   = number of active lights (max 4)
    lightPos:   array<vec4<f32>, 4>, // xyz = position (point) or direction (directional)
    lightCol:   array<vec4<f32>, 4>, // xyz = color * intensity per light
    lightType:  array<vec4<f32>, 4>, // x   = 0 point, 1 directional
};

@group(0) @binding(0) var<uniform> rt:          RtUniforms;
@group(0) @binding(1) var accumRead:  texture_2d<f32>;
@group(0) @binding(2) var accumWrite: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var bvhData:    texture_2d<f32>;
@group(0) @binding(4) var matData:    texture_2d<f32>;
@group(0) @binding(5) var triData:    texture_2d<f32>;
@group(0) @binding(6) var texAtlas:   texture_2d<f32>;

const TRI_PAGE_W:  i32 = 8192;
const TRI_PAGE_H:  i32 = 8;
const TILE_SIZE:   i32 = 512;  // must match C++ TILE_SIZE constant
const MAX_TEX_SLOTS: i32 = 16;
const BVH_PAGE_W: i32 = 8192;

fn triCoord(ti: i32, row: i32) -> vec2<i32> {
    return vec2<i32>(ti % TRI_PAGE_W, (ti / TRI_PAGE_W) * TRI_PAGE_H + row);
}
fn bvhCoord(ni: i32, row: i32) -> vec2<i32> {
    return vec2<i32>(ni % BVH_PAGE_W, (ni / BVH_PAGE_W) * 2 + row);
}

// ---- Types ----
struct Ray  { origin: vec3<f32>, dir: vec3<f32> }
struct Isect { t: f32, u: f32, v: f32 }
struct Hit  {
    t:         f32,
    point:     vec3<f32>,
    normal:    vec3<f32>,
    albedo:    vec3<f32>,
    shininess: f32,
    uv:        vec2<f32>,
    texSlot:   f32,   // < 0 = no texture
    metalness: f32,
    emissive:  vec3<f32>,
}

// ---- PCG random number generator ----
fn pcg(v: u32) -> u32 {
    var s = v * 747796405u + 2891336453u;
    s = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (s >> 22u) ^ s;
}
fn rand(seed: ptr<function, u32>) -> f32 {
    *seed = pcg(*seed);
    return f32(*seed) / 4294967296.0;
}

// ---- Cosine-weighted hemisphere sample around n ----
fn cosineHemisphere(n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let u1  = rand(seed);
    let u2  = rand(seed);
    let r   = sqrt(u1);
    let phi = 6.28318530718 * u2;
    let lx  = r * cos(phi);
    let ly  = r * sin(phi);
    let lz  = sqrt(max(0.0, 1.0 - u1));
    // Build orthonormal basis around n
    let nt  = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.99);
    let rgt = normalize(cross(nt, n));
    let up  = cross(n, rgt);
    return normalize(lx * rgt + ly * up + lz * n);
}

// ---- GGX microfacet helpers ----
const PI: f32 = 3.14159265358979;

// GGX (Trowbridge-Reitz) Normal Distribution Function
fn ggxD(NdotH: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Smith G1 — Schlick-GGX approximation
fn ggxG1(NdotX: f32, alpha: f32) -> f32 {
    let k = alpha * 0.5;
    return NdotX / max(NdotX * (1.0 - k) + k, 1e-6);
}

// Schlick Fresnel
fn schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (vec3<f32>(1.0) - F0) * pow(max(0.0, 1.0 - cosTheta), 5.0);
}

// Sample a direction from the GGX NDF; returns the reflected direction.
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
)" R"(
// ---- Atlas colour lookup ----
fn sampleAtlas(uv: vec2<f32>, texSlot: f32) -> vec3<f32> {
    let tx = i32(texSlot) * TILE_SIZE
           + clamp(i32(fract(uv.x) * f32(TILE_SIZE)), 0, TILE_SIZE - 1);
    let ty = clamp(i32(fract(uv.y) * f32(TILE_SIZE)), 0, TILE_SIZE - 1);
    return textureLoad(texAtlas, vec2<i32>(tx, ty), 0).xyz;
}

// ---- Sky gradient ----
fn sky(d: vec3<f32>) -> vec3<f32> {
    let t = clamp(0.5 * (normalize(d).y + 1.0), 0.0, 1.0);
    return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.32, 0.52, 1.0), t);
}

// ---- Möller–Trumbore: returns (t, u, v); t=1e30 on miss ----
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

// ---- Slab AABB test ----
// Returns tNear if hit, else 1e30
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

// ---- Load triangle, update h if closer ----
fn testTriangle(ray: Ray, ti: i32, h: ptr<function, Hit>) {
    let r0 = textureLoad(triData, triCoord(ti, 0), 0);
    let v0 = r0.xyz;
    let v1 = textureLoad(triData, triCoord(ti, 1), 0).xyz;
    let v2 = textureLoad(triData, triCoord(ti, 2), 0).xyz;

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

    (*h).t         = isect.t;
    (*h).point     = ray.origin + isect.t * ray.dir;
    (*h).normal    = select(sn, -sn, dot(ray.dir, sn) > 0.0);
    (*h).albedo    = mat0.xyz;
    (*h).shininess = mat0.w;
    (*h).uv        = iuv;
    (*h).texSlot   = mat1.x;
    (*h).metalness = mat1.y;
    (*h).emissive  = mat2.xyz;
}

// ---- BVH traversal ----
fn sceneHit(ray: Ray) -> Hit {
    var h: Hit; h.t = 1e30;
    var stack: array<i32, 64>;
    var top: i32 = 0;
    stack[0] = 0; top = 1;

    while (top > 0) {
        top -= 1;
        let ni  = stack[top];
        let nd0 = textureLoad(bvhData, bvhCoord(ni, 0), 0);
        let nd1 = textureLoad(bvhData, bvhCoord(ni, 1), 0);
        if (!aabbHit(nd0.xyz, nd1.xyz, ray, h.t)) { continue; }
        let left  = i32(nd0.w);
        let right = i32(nd1.w);
        if (left < 0) {
            let triStart = -left - 1;
            let triCount = right;
            for (var ti = triStart; ti < triStart + triCount; ti++) {
                testTriangle(ray, ti, &h);
            }
        } else {
            let dL = aabbDist(textureLoad(bvhData, bvhCoord(left,  0), 0).xyz,
                               textureLoad(bvhData, bvhCoord(left,  1), 0).xyz, ray, h.t);
            let dR = aabbDist(textureLoad(bvhData, bvhCoord(right, 0), 0).xyz,
                               textureLoad(bvhData, bvhCoord(right, 1), 0).xyz, ray, h.t);
            // Push far child first so near child is popped first
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

// ---- Generate a camera ray for pixel px with viewport res ----
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

// ---- GGX shade with hard shadow (used by raytracer mode) ----
fn shade(h: Hit, rd: vec3<f32>) -> vec3<f32> {
    var albedo = h.albedo;
    if (h.texSlot >= 0.0) { albedo = sampleAtlas(h.uv, h.texSlot); }
    var col = albedo * 0.02;
    let count = i32(rt.lightCount.x);
    let wo_s = normalize(-rd);
    for (var li = 0; li < 4; li++) {
        if (li >= count) { break; }
        let lc    = rt.lightCol[li].xyz;
        let isDir = rt.lightType[li].x > 0.5;
        let ln    = select(normalize(rt.lightPos[li].xyz - h.point),
                           normalize(rt.lightPos[li].xyz), isDir);
        let ld    = select(length(rt.lightPos[li].xyz - h.point), 1e30, isDir);
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
)" R"(
// ---- Deterministic trace with one mirror bounce ----
fn raytrace(ray: Ray) -> vec3<f32> {
    let h0 = sceneHit(ray);
    if (h0.t >= 1e30) { return sky(ray.dir); }
    var col = shade(h0, ray.dir);
    if (h0.shininess < 0.5) {
        // Low GGX alpha = smooth/glossy surface: add a mirror bounce
        let k = max(0.0, 1.0 - h0.shininess * 2.0) * 0.55;
        var r1: Ray;
        r1.origin = h0.point + h0.normal * 1e-3;
        r1.dir    = reflect(ray.dir, h0.normal);
        let h1 = sceneHit(r1);
        let rc = select(shade(h1, r1.dir), sky(r1.dir), h1.t >= 1e30);
        col = col * (1.0 - k) + rc * k;
    }
    return col;
}

// ---- Path trace: next-event estimation + cosine-weighted indirect ----
fn pathTrace(ray_in: Ray, seed: ptr<function, u32>) -> vec3<f32> {
    var ray        = ray_in;
    var throughput = vec3<f32>(1.0);
    var radiance   = vec3<f32>(0.0);

    for (var i = 0; i < 5; i++) {
        let h = sceneHit(ray);
        if (h.t >= 1e29) {
            radiance += throughput * sky(ray.dir);
            break;
        }

        radiance += throughput * h.emissive;

        // Resolve albedo from texture atlas if this material has a texture
        var albedo = h.albedo;
        if (h.texSlot >= 0.0) { albedo = sampleAtlas(h.uv, h.texSlot); }

        // Next-event estimation: shadow ray to each light (GGX BSDF)
        let lcount = i32(rt.lightCount.x);
        let wo = normalize(-ray.dir);
        for (var li = 0; li < 4; li++) {
            if (li >= lcount) { break; }
            let lc    = rt.lightCol[li].xyz;
            let isDir = rt.lightType[li].x > 0.5;
            let ln    = select(normalize(rt.lightPos[li].xyz - h.point),
                               normalize(rt.lightPos[li].xyz), isDir);
            let ld    = select(length(rt.lightPos[li].xyz - h.point), 1e30, isDir);
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
        // Ambient term
        radiance += throughput * albedo * 0.03;

        // Russian roulette after first bounce
        if (i > 0) {
            let p = max(max(throughput.r, throughput.g), throughput.b);
            if (rand(seed) > p) { break; }
            throughput /= p;
        }

        // Sample next direction: 50/50 GGX specular vs. cosine-weighted diffuse
        let wo_b = normalize(-ray.dir);
        let F0_b = mix(vec3<f32>(0.04), albedo, h.metalness);
        var wi_b: vec3<f32>;
        if (rand(seed) < 0.5) {
            // GGX specular sample
            wi_b = sampleGGXDir(wo_b, h.normal, h.shininess, seed);
            let cos_b = dot(h.normal, wi_b);
            if (cos_b <= 0.0) { break; }
            let hb  = normalize(wo_b + wi_b);
            let Fb  = schlick(max(0.0, dot(wo_b, hb)), F0_b);
            let G1L = ggxG1(cos_b, h.shininess);
            throughput *= Fb * G1L * 2.0;   // divide by 0.5 selection probability
        } else {
            // Diffuse sample
            wi_b = cosineHemisphere(h.normal, seed);
            let cos_b = dot(h.normal, wi_b);
            if (cos_b <= 0.0) { break; }
            throughput *= albedo * 2.0;     // divide by 0.5 selection probability
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
        // ---- Raytracer mode: deterministic Blinn-Phong, RGSS 4x AA ----
        let o0 = vec2<f32>( 0.125,  0.375);
        let o1 = vec2<f32>(-0.375,  0.125);
        let o2 = vec2<f32>( 0.375, -0.125);
        let o3 = vec2<f32>(-0.125, -0.375);
        let fp = vec2<f32>(f32(pixel.x), f32(pixel.y));
        sample = (raytrace(makeRay(fp + o0, res))
                + raytrace(makeRay(fp + o1, res))
                + raytrace(makeRay(fp + o2, res))
                + raytrace(makeRay(fp + o3, res))) * 0.25;
        // No accumulation in raytracer mode — write fresh each frame
        textureStore(accumWrite, pixel, vec4<f32>(sample, 1.0));
    } else {
        // ---- Path tracer mode: stochastic, EMA accumulation ----
        var seed = pcg(gid.x * 1973u + 1u) ^ pcg(gid.y * 9277u + 1u) ^ pcg(fc * 26699u + 1u);
        let jx  = rand(&seed) - 0.5;
        let jy  = rand(&seed) - 0.5;
        let ray = makeRay(vec2<f32>(f32(pixel.x) + jx, f32(pixel.y) + jy), res);
        sample  = pathTrace(ray, &seed);

        // Exponential moving average (alpha=1 on reset to avoid ghosting old view)
        let old     = textureLoad(accumRead, pixel, 0).xyz;
        let alpha   = select(0.1, 1.0, fc == 0u);
        let blended = old * (1.0 - alpha) + sample * alpha;
        textureStore(accumWrite, pixel, vec4<f32>(blended, 1.0));
    }
}
)";

// ---------------------------------------------------------------------------
// WGSL display shader — blit accumulated texture to screen with gamma
//
// Bindings (ShaderMaterial, no custom uniforms):
//   0: TransformUniforms (always present)
//   1: LightData buffer (always present, unused here)
//   2: accumTex (first custom texture, texture_2d<f32>)
//   3: accumTex sampler (reserved, unused)
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
// CPU-side uniform struct (must match RtUniforms in csWGSL, 256-byte aligned)
// ---------------------------------------------------------------------------
// 22 vec4s used (352 bytes): + lightType[4] vs previous 18.
// Padded to 512 bytes.
struct alignas(16) RtGpuUniforms {
    float camOri[4];
    float camFwd[4];
    float camRgt[4];
    float camUp[4];
    float iRes[4];
    float tanHalfFov[4];
    float frameCount[4];
    float triCount[4];
    float mode[4];        // x: 0=raytracer 1=path tracer
    float lightCount[4];  // x: number of active lights (max 4)
    float lightPos[4][4]; // [lightIdx][xyzw] – position or toward-direction
    float lightCol[4][4]; // [lightIdx][xyzw]
    float lightType[4][4];// [lightIdx][x]: 0=point, 1=directional
    float _pad[40];       // 352 used + 160 pad = 512 bytes
};
static_assert(sizeof(RtGpuUniforms) == 512, "RtGpuUniforms must be 512 bytes");

// ---------------------------------------------------------------------------
// Build texture atlas: MAX_TEX_SLOTS tiles of TILE_SIZE×TILE_SIZE (RGBA8).
// Fills texSlotMap: Texture* → slot index.
// ---------------------------------------------------------------------------
static std::vector<unsigned char> buildAtlas(
        const std::vector<Mesh*>& meshes,
        std::unordered_map<Texture*, int>& texSlotMap) {
    const int atlasW = MAX_TEX_SLOTS * TILE_SIZE;
    std::vector<unsigned char> atlas(atlasW * TILE_SIZE * 4, 255);

    int slot = 0;
    for (auto& mesh : meshes) {
        if (slot >= MAX_TEX_SLOTS) break;
        auto* mwm = dynamic_cast<MaterialWithMap*>(mesh->material().get());
        if (!mwm || !mwm->map) continue;
        Texture* tex = mwm->map.get();
        if (texSlotMap.count(tex)) continue;

        auto& img = tex->image();
        if (img.width == 0 || img.height == 0) continue;
        const auto& src = img.data<unsigned char>();
        const int srcW = static_cast<int>(img.width);
        const int srcH = static_cast<int>(img.height);
        const int ch = static_cast<int>(src.size()) / (srcW * srcH);

        const int destX = slot * TILE_SIZE;
        for (int ty = 0; ty < TILE_SIZE; ++ty) {
            const int sy = ty * srcH / TILE_SIZE;
            for (int tx = 0; tx < TILE_SIZE; ++tx) {
                const int sx = tx * srcW / TILE_SIZE;
                const int si = (sy * srcW + sx) * ch;
                const int di = (ty * atlasW + destX + tx) * 4;
                atlas[di + 0] = src[si + 0];
                atlas[di + 1] = src[si + 1];
                atlas[di + 2] = src[si + 2];
                atlas[di + 3] = ch == 4 ? src[si + 3] : 255u;
            }
        }
        texSlotMap[tex] = slot++;
    }
    return atlas;
}

// ---------------------------------------------------------------------------
// Extract material properties
// ---------------------------------------------------------------------------
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
        shininess = std::max(0.04f, rough * rough);   // GGX alpha = roughness²
    } else if (auto* s = dynamic_cast<const MaterialWithSpecular*>(mat)) {
        const float n = std::max(1.f, s->shininess);
        shininess = std::max(0.04f, std::sqrt(2.f / (n + 2.f)));  // Phong → GGX alpha
    }
    if (auto* m = dynamic_cast<const MaterialWithMetalness*>(mat))
        metalness = std::max(0.f, std::min(1.f, m->metalness));
    if (auto* e = dynamic_cast<const MaterialWithEmissive*>(mat))
        emissive = Color(e->emissive.r * e->emissiveIntensity,
                         e->emissive.g * e->emissiveIntensity,
                         e->emissive.b * e->emissiveIntensity);
    return {albedo, shininess, metalness, emissive};
}

// ---------------------------------------------------------------------------
// Build CPU triangle + material buffers from Mesh objects.
// Returns the number of triangles written.
// ---------------------------------------------------------------------------
static int buildGeometryBuffers(
        const std::vector<Mesh*>& meshes,
        const std::unordered_map<Texture*, int>& texSlotMap,
        std::vector<float>& triBuffer,
        std::vector<float>& matBuffer) {
    std::ranges::fill(triBuffer, 0.f);
    std::ranges::fill(matBuffer, 0.f);

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

    for (auto& mesh : meshes) {
        if (triCount >= MAX_TRIS || matCount >= MAX_MATS) break;

        auto [albedo, shininess, metalness, emissive] = extractMaterial(mesh->material().get());
        setTexel(matBuffer, MAX_MATS, matCount, 0,
                 albedo.r, albedo.g, albedo.b, shininess);

        float texSlot = -1.f;
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
        setTexel(matBuffer, MAX_MATS, matCount, 1, texSlot, metalness, 0.f, 0.f);
        setTexel(matBuffer, MAX_MATS, matCount, 2, emissive.r, emissive.g, emissive.b, 0.f);

        const int matIdx = matCount++;

        mesh->updateWorldMatrix(true, true);
        const auto& world = *mesh->matrixWorld;
        // Normals must be transformed by the inverse-transpose of the world matrix
        // to stay perpendicular to surfaces under non-uniform scale.
        Matrix4 normalMat(world);
        normalMat.invert().transpose();
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
            ++triCount;
        }
    }
    return triCount;
}

// ---------------------------------------------------------------------------
// BVH builder
// ---------------------------------------------------------------------------
struct BvhNode {
    float minX, minY, minZ;
    int left;
    float maxX, maxY, maxZ;
    int right;
};

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

// SAH BVH: 8-bucket binned SAH, leaf threshold 4.
static int buildBvhNode(
        std::vector<BvhNode>& nodes,
        std::vector<int>& idx,
        const std::vector<float>& buf,
        int start, int end) {
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
    nodes[ni] = {minX, minY, minZ, 0, maxX, maxY, maxZ, 0};

    const int count = end - start;
    if (count <= 4) {
        nodes[ni].left = -(start + 1);
        nodes[ni].right = count;
        return ni;
    }

    // Binned SAH: evaluate 8 candidate splits along each axis.
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
            // Accumulate left [0..s-1] and right [s..NB-1]
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

    const int lc = buildBvhNode(nodes, idx, buf, start, sp);
    const int rc = buildBvhNode(nodes, idx, buf, sp, end);
    nodes[ni].left = lc;
    nodes[ni].right = rc;
    return ni;
}

static std::vector<float> buildBVH(std::vector<float>& triBuffer, int triCount) {
    std::vector<int> indices(triCount);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<BvhNode> nodes;
    nodes.reserve(triCount * 2);
    buildBvhNode(nodes, indices, triBuffer, 0, triCount);

    // Reorder triBuffer to match sorted leaf order
    auto pagedIdx = [](int ti, int row) -> int {
        const int page = ti / TEX_PAGE_WIDTH;
        const int pcol = ti % TEX_PAGE_WIDTH;
        return ((page * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + pcol) * 4;
    };
    std::vector<float> sorted(triBuffer.size(), 0.f);
    for (int ni = 0; ni < triCount; ni++) {
        const int oi = indices[ni];
        for (int row = 0; row < TRI_TEX_HEIGHT; row++)
            for (int c = 0; c < 4; c++)
                sorted[pagedIdx(ni, row) + c] = triBuffer[pagedIdx(oi, row) + c];
    }
    triBuffer = std::move(sorted);

    // Pack BVH nodes into paged RGBA32Float texture
    std::vector<float> bvhBuf(TEX_PAGE_WIDTH * 2 * BVH_TEX_PAGES * 4, 0.f);
    const int nc = std::min(static_cast<int>(nodes.size()), MAX_BVH_NODES);
    for (int i = 0; i < nc; i++) {
        const auto& n = nodes[i];
        const int page = i / TEX_PAGE_WIDTH;
        const int col = i % TEX_PAGE_WIDTH;
        bvhBuf[((page * 2 + 0) * TEX_PAGE_WIDTH + col) * 4 + 0] = n.minX;
        bvhBuf[((page * 2 + 0) * TEX_PAGE_WIDTH + col) * 4 + 1] = n.minY;
        bvhBuf[((page * 2 + 0) * TEX_PAGE_WIDTH + col) * 4 + 2] = n.minZ;
        bvhBuf[((page * 2 + 0) * TEX_PAGE_WIDTH + col) * 4 + 3] = static_cast<float>(n.left);
        bvhBuf[((page * 2 + 1) * TEX_PAGE_WIDTH + col) * 4 + 0] = n.maxX;
        bvhBuf[((page * 2 + 1) * TEX_PAGE_WIDTH + col) * 4 + 1] = n.maxY;
        bvhBuf[((page * 2 + 1) * TEX_PAGE_WIDTH + col) * 4 + 2] = n.maxZ;
        bvhBuf[((page * 2 + 1) * TEX_PAGE_WIDTH + col) * 4 + 3] = static_cast<float>(n.right);
    }
    return bvhBuf;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {

    Canvas canvas("Wgpu Path Tracer – Accumulation",
                  {{"graphicsApi", GraphicsAPI::WebGPU}});

    WgpuRenderer renderer(canvas);
    renderer.setClearColor(Color(0x000000));

    auto sz = canvas.size();

    // ---- Geometry textures (shared between frames) ----
    WgpuTexture bvhTex(renderer, TEX_PAGE_WIDTH, 2 * BVH_TEX_PAGES,
                       WgpuTexture::Format::RGBA32Float,
                       WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
    WgpuTexture matTex(renderer, MAX_MATS, MAT_TEX_HEIGHT,
                       WgpuTexture::Format::RGBA32Float,
                       WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
    WgpuTexture triTex(renderer, TEX_PAGE_WIDTH, TRI_TEX_HEIGHT * TRI_TEX_PAGES,
                       WgpuTexture::Format::RGBA32Float,
                       WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
    WgpuTexture texAtlasTex(renderer, MAX_TEX_SLOTS * TILE_SIZE, TILE_SIZE,
                            WgpuTexture::Format::RGBA8Unorm,
                            WgpuTexture::TextureBinding | WgpuTexture::CopyDst);

    // CPU geometry buffers
    std::vector<float> triBuffer(TEX_PAGE_WIDTH * TRI_TEX_HEIGHT * TRI_TEX_PAGES * 4, 0.f);
    std::vector<float> matBuffer(MAX_MATS * MAT_TEX_HEIGHT * 4, 0.f);

    // ---- Accumulation textures (ping-pong, RGBA32Float) ----
    // Default usage includes Storage | TextureBinding | CopyDst
    auto makeAccumTex = [&](uint32_t w, uint32_t h) {
        return WgpuTexture(renderer, w, h, WgpuTexture::Format::RGBA32Float);
    };
    auto accumA = makeAccumTex(static_cast<uint32_t>(sz.width()),
                               static_cast<uint32_t>(sz.height()));
    auto accumB = makeAccumTex(static_cast<uint32_t>(sz.width()),
                               static_cast<uint32_t>(sz.height()));
    WgpuTexture* readAccum = &accumA;
    WgpuTexture* writeAccum = &accumB;

    // Zero-fill for initial state
    {
        std::vector<float> zeros(sz.width() * sz.height() * 4, 0.f);
        accumA.write(zeros.data(), zeros.size() * sizeof(float));
        accumB.write(zeros.data(), zeros.size() * sizeof(float));
    }

    // ---- RT uniform buffer (256 bytes, matches RtGpuUniforms) ----
    WgpuBuffer rtUniformBuf(renderer, sizeof(RtGpuUniforms));

    // ---- Compute pipeline ----
    WgpuComputePipeline rtPipeline(renderer, csWGSL, "rt_main");
    rtPipeline.setUniformBuffer(0, rtUniformBuf);
    // bindings 1, 2 set per-frame (ping-pong)
    rtPipeline.setTexture(3, bvhTex);
    rtPipeline.setTexture(4, matTex);
    rtPipeline.setTexture(5, triTex);
    rtPipeline.setTexture(6, texAtlasTex);

    // ---- Camera + controls ----
    PerspectiveCamera rtCam(60.f, canvas.aspect(), 0.1f, 200.f);
    rtCam.position.set(0.f, 3.f, 8.f);
    OrbitControls controls{rtCam, canvas};
    controls.target.set(0.f, 0.f, 0.f);
    controls.update();
    const float tanHalfFov = std::tan(60.f * math::PI / 360.f);

    // ---- Display: fullscreen quad with blit shader ----
    OrthographicCamera displayCam(-1.f, 1.f, 1.f, -1.f, 0.1f, 10.f);
    displayCam.position.z = 1.f;

    auto displayMat = ShaderMaterial::create();
    displayMat->vertexShader = displayWGSL;
    displayMat->fragmentShader = displayWGSL;
    displayMat->customTextures["accumTex"] = readAccum;

    Scene displayScene;
    displayScene.add(Mesh::create(PlaneGeometry::create(2.f, 2.f), displayMat));

    // ---- Scene objects ----
    TextureLoader tl;
    auto tex = tl.load(std::string(DATA_FOLDER) + "/textures/uv_grid_opengl.jpg");

    auto boxMesh = Mesh::create(
            BoxGeometry::create(1.5f, 1.5f, 1.5f),
            MeshStandardMaterial::create({{"map", tex}, {"roughness", 0.9f}}));
    boxMesh->position.set(0.f, 1.f, -1.f);

    auto boxMesh2 = Mesh::create(
            BoxGeometry::create(),
            MeshStandardMaterial::create({{"color", Color::black}, {"roughness", 0.9f}}));
    boxMesh2->scale *= 1000;


    auto sphere1 = Mesh::create(
            SphereGeometry::create(0.85f, 32, 32),
            MeshStandardMaterial::create({{"color", Color::orangered},
                                          {"roughness", 0.85f}, {"emissive", Color::orangered}, {"emissiveIntensity", 0.9f}}));
    sphere1->position.set(-2.8f, 1.f, 0.f);

    auto sphere2 = Mesh::create(
            SphereGeometry::create(0.85f, 32, 32),
            MeshStandardMaterial::create({{"color", Color::steelblue},
                                          {"roughness", 0.1f},
                                          {"metalness", 0.9f}}));
    sphere2->position.set(2.8f, 1.f, 0.f);

    auto floor = Mesh::create(
            PlaneGeometry::create(16.f, 16.f, 4, 4),
            MeshStandardMaterial::create({{"color", Color::darkgrey},
                                          {"roughness", 0.99f}}));
    floor->rotation.x = -math::PI / 2.f;
    floor->position.y = -1.f;

    // ---- Scene graph — add all objects so traversal can find them ----
    Scene scene;
    scene.add(boxMesh);
    scene.add(sphere1);
    scene.add(sphere2);
    scene.add(floor);
    scene.add(boxMesh2);

    ModelLoader loader;
    auto obj = loader.load(std::string(DATA_FOLDER) + "/models/collada/stormtrooper/stormtrooper.dae");
    obj->traverseType<Mesh>([&](Mesh& m) {
        auto tex = m.material()->as<MaterialWithMap>();
        m.setMaterial(MeshStandardMaterial::create({{"map", tex ? tex->map : nullptr}, {"roughness", 0.9f}}));
    });
    obj->position.z = -4.f;
    obj->position.y = -1.f;
    // obj->scale *= 0.025f;
    scene.add(obj);

    // ---- Point light ----
    auto pointLight = PointLight::create(Color::white, 0.9f);
    pointLight->position.set(5.f, 6.f, -2.f);
    scene.add(pointLight);

    auto pointLight2 = PointLight::create(Color::white, 0.4f);
    pointLight2->position.set(-5.f, 6.f, 4.f);
    scene.add(pointLight2);

    // ---- Texture atlas (rebuilt when mesh list changes) ----
    std::unordered_map<Texture*, int> texSlotMap;
    std::vector<Mesh*> prevMeshes;// for change detection

    // ---- Mode + accumulation state ----
    bool pathTracerOn = false;// press T to toggle
    float frameCount = 0.f;
    Vector3 prevCamPos = rtCam.position;
    Vector3 prevTarget = controls.target;

    bool raster = false;
    KeyAdapter keyAdapter(KeyAdapter::Mode::KEY_PRESSED, [&](KeyEvent ev) {
        if (ev.key == Key::T) {
            pathTracerOn = !pathTracerOn;
            frameCount = 0.f;// reset accumulation on mode switch
        } else if (ev.key == Key::R) {
            raster = !raster;
        }
    });
    canvas.addKeyListener(keyAdapter);

    canvas.onWindowResize([&](const WindowSize& ns) {
        renderer.setSize(ns);
        // Recreate accum textures at new resolution
        accumA = makeAccumTex(static_cast<uint32_t>(ns.width()),
                              static_cast<uint32_t>(ns.height()));
        accumB = makeAccumTex(static_cast<uint32_t>(ns.width()),
                              static_cast<uint32_t>(ns.height()));
        readAccum = &accumA;
        writeAccum = &accumB;
        displayMat->customTextures["accumTex"] = readAccum;
        frameCount = 0.f;
    });

    // ---- Animation ----
    Clock clock;
    float elapsed = 0.f;

    canvas.animate([&] {
        const float dt = clock.getDelta();
        elapsed += dt;

        // Animate box and light
        boxMesh->rotation.y += dt * 0.6f;
        boxMesh->rotation.x += dt * 0.3f;
        pointLight->position.set(
                5.f * std::cos(elapsed * 0.6f),
                6.f + std::sin(elapsed * 0.3f),
                -2.f + 4.f * std::sin(elapsed * 0.6f));

        // Update camera
        controls.update();
        const Vector3& camPos = rtCam.position;
        Vector3 fwd = Vector3(controls.target).sub(camPos).normalize();
        Vector3 rgt = Vector3(fwd).cross(Vector3(0.f, 1.f, 0.f)).normalize();
        Vector3 up = Vector3(rgt).cross(fwd);

        // Reset accumulation on camera movement
        const bool camMoved =
                (camPos - prevCamPos).length() > 1e-5f ||
                (controls.target - prevTarget).length() > 1e-5f;
        if (camMoved) {
            frameCount = 0.f;
            prevCamPos = camPos;
            prevTarget = controls.target;
        }

        // Collect visible meshes and lights from scene graph
        std::vector<Mesh*> rtMeshes;
        scene.traverseType<Mesh>([&](Mesh& m) {
            if (m.visible) rtMeshes.push_back(&m);
        });
        std::vector<PointLight*> pointLights;
        scene.traverseType<PointLight>([&](PointLight& l) { pointLights.push_back(&l); });
        std::vector<DirectionalLight*> dirLights;
        scene.traverseType<DirectionalLight>([&](DirectionalLight& l) { dirLights.push_back(&l); });

        // Rebuild texture atlas only when mesh list changes
        if (rtMeshes != prevMeshes) {
            texSlotMap.clear();
            const auto atlasData = buildAtlas(rtMeshes, texSlotMap);
            texAtlasTex.write(atlasData.data(), atlasData.size());
            prevMeshes = rtMeshes;
        }

        // Rebuild geometry + BVH each frame (handles animated objects)
        const int triCount = buildGeometryBuffers(rtMeshes, texSlotMap, triBuffer, matBuffer);
        const auto bvhBuffer = buildBVH(triBuffer, triCount);

        bvhTex.write(bvhBuffer.data(), bvhBuffer.size() * sizeof(float));
        triTex.write(triBuffer.data(), triBuffer.size() * sizeof(float));
        matTex.write(matBuffer.data(), matBuffer.size() * sizeof(float));

        // Pack uniform buffer
        RtGpuUniforms u{};
        u.camOri[0] = camPos.x;
        u.camOri[1] = camPos.y;
        u.camOri[2] = camPos.z;
        u.camFwd[0] = fwd.x;
        u.camFwd[1] = fwd.y;
        u.camFwd[2] = fwd.z;
        u.camRgt[0] = rgt.x;
        u.camRgt[1] = rgt.y;
        u.camRgt[2] = rgt.z;
        u.camUp[0] = up.x;
        u.camUp[1] = up.y;
        u.camUp[2] = up.z;
        const auto curSz = canvas.size();
        u.iRes[0] = static_cast<float>(curSz.width());
        u.iRes[1] = static_cast<float>(curSz.height());
        u.tanHalfFov[0] = tanHalfFov;
        u.frameCount[0] = frameCount;
        u.triCount[0] = static_cast<float>(triCount);
        u.mode[0] = pathTracerOn ? 1.f : 0.f;

        // Pack up to 4 lights (point first, then directional)
        int nLights = 0;
        auto packLight = [&](float px, float py, float pz, float r, float g, float b, float type) {
            if (nLights >= 4) return;
            u.lightPos[nLights][0] = px;
            u.lightPos[nLights][1] = py;
            u.lightPos[nLights][2] = pz;
            u.lightCol[nLights][0] = r;
            u.lightCol[nLights][1] = g;
            u.lightCol[nLights][2] = b;
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
            // direction = normalize(target - position); store as "toward light" unit vector
            Vector3 dir = Vector3(l->target().position).sub(l->position).normalize();
            const auto& lc = l->color;
            const float li = l->intensity;
            packLight(dir.x, dir.y, dir.z, lc.r * li, lc.g * li, lc.b * li, 1.f);
        }
        u.lightCount[0] = static_cast<float>(nLights);

        rtUniformBuf.write(&u, sizeof(u));

        // Dispatch: read accumA, write accumB (ping-pong)
        rtPipeline.setTexture(1, *readAccum);
        rtPipeline.setStorageTexture(2, *writeAccum);
        const uint32_t gx = (static_cast<uint32_t>(curSz.width()) + 7u) / 8u;
        const uint32_t gy = (static_cast<uint32_t>(curSz.height()) + 7u) / 8u;
        rtPipeline.dispatch(gx, gy);

        // Swap ping-pong and point display at written result
        std::swap(readAccum, writeAccum);
        displayMat->customTextures["accumTex"] = readAccum;
        displayMat->uniformsNeedUpdate = true;

        frameCount += 1.f;

        if (raster) {
            renderer.render(scene, rtCam);
        } else {
            // Render display quad
            renderer.render(displayScene, displayCam);
        }
    });

    return 0;
}
