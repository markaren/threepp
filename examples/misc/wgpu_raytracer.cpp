// GPU raytracer that reads real threepp geometry (BufferGeometry) and renders it
// entirely in a WGSL fragment shader via WgpuRenderer.
//
// How it works:
//   Each frame, world-space triangle data is extracted from threepp Mesh objects
//   (position attribute + index buffer + matrixWorld), packed into two
//   RGBA32Float WgpuTextures, and uploaded to the GPU:
//
//     triData  (width=MAX_TRIS, height=4)
//       row 0 : col i = (v0.x, v0.y, v0.z, float(matIdx))
//       row 1 : col i = (v1.x, v1.y, v1.z, 0)
//       row 2 : col i = (v2.x, v2.y, v2.z, 0)
//       row 3 : col i = (nx,   ny,   nz,   0)   precomputed face normal
//
//     matData  (width=MAX_MATS, height=1)
//       row 0 : col i = (albedo.r, albedo.g, albedo.b, shininess)
//
//   The WGSL fragment shader loops over triCount triangles, runs
//   Möller–Trumbore intersection, and shades with Blinn-Phong + hard shadows
//   + one mirror-reflection bounce.  OrbitControls give interactive camera.

#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/materials/MeshLambertMaterial.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/renderers/WgpuRenderer.hpp"
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
constexpr int MAX_TRIS       = 4096;
constexpr int MAX_MATS       = 64;
constexpr int MAX_BVH_NODES  = 8192;  // 2*MAX_TRIS - 1 worst case
constexpr int TRI_TEX_HEIGHT = 8;     // rows: v0 v1 v2 n0 n1 n2 uv01 uv2
constexpr int MAT_TEX_HEIGHT = 2;     // row0: albedo+shininess, row1: texSlot
constexpr int MAX_TEX_SLOTS  = 16;
constexpr int TILE_SIZE      = 256;

// ---------------------------------------------------------------------------
// WGSL vertex shader
// ---------------------------------------------------------------------------
constexpr const char* vsWGSL = R"(
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

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return transform.proj * transform.view * transform.model * vec4<f32>(position, 1.0);
}
)";

// ---------------------------------------------------------------------------
// WGSL fragment shader
// Custom uniforms MUST be alphabetical: aspect camFwd camOri camRgt camUp iRes iTime tanHalfFov triCount
// Custom textures MUST be alphabetical: bvhData(3,4)  matData(5,6)  texAtlas(7,8)  triData(9,10)
// ---------------------------------------------------------------------------
constexpr const char* fsWGSL = R"(

struct RtUniforms {
    aspect:     vec4<f32>,   // x = width/height
    camFwd:     vec4<f32>,   // xyz = forward
    camOri:     vec4<f32>,   // xyz = camera position
    camRgt:     vec4<f32>,   // xyz = right
    camUp:      vec4<f32>,   // xyz = up
    iRes:       vec4<f32>,   // xy = viewport pixels
    iTime:      vec4<f32>,   // x = elapsed seconds
    tanHalfFov: vec4<f32>,   // x = tan(fov/2)
    triCount:   vec4<f32>,   // x = number of triangles (float→int)
};
@group(0) @binding(2) var<uniform> rt: RtUniforms;

// bvhData: row0 col i = (aabbMin.xyz, left)   left<0 → leaf, triStart=-left-1
//          row1 col i = (aabbMax.xyz, right)  right = rightChild OR triCount
@group(0) @binding(3) var bvhData: texture_2d<f32>;
// binding 4 = bvhData sampler (reserved)

// matData: row0 col i = (albedo.r, albedo.g, albedo.b, shininess)
//          row1 col i = (texSlot, 0, 0, 0)  — texSlot<0 means no texture
@group(0) @binding(5) var matData: texture_2d<f32>;
// binding 6 = matData sampler (reserved)

// texAtlas: horizontal strip of TILE_SIZE×TILE_SIZE tiles, one per texSlot
@group(0) @binding(7) var texAtlas:        texture_2d<f32>;
@group(0) @binding(8) var texAtlasSampler: sampler;

// triData: rows 0-2 = v0/v1/v2 (w=matIdx in row0), rows 3-5 = n0/n1/n2
//          row 6 = (u0,v0, u1,v1), row 7 = (u2,v2, 0,0)
@group(0) @binding(9) var triData: texture_2d<f32>;
// binding 10 = triData sampler (reserved)

const MAX_TEX_SLOTS: f32 = 16.0;

// ---- Types ----
struct Ray { origin: vec3<f32>, dir: vec3<f32>, }
struct Isect { t: f32, u: f32, v: f32, }
struct Hit {
    t:         f32,
    point:     vec3<f32>,
    normal:    vec3<f32>,
    albedo:    vec3<f32>,
    shininess: f32,
    uv:        vec2<f32>,
    texSlot:   f32,        // < 0 → no texture
}

// ---- Sky gradient ----
fn sky(d: vec3<f32>) -> vec3<f32> {
    let t = clamp(0.5 * (normalize(d).y + 1.0), 0.0, 1.0);
    return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.32, 0.52, 1.0), t);
}

// ---- Möller–Trumbore: returns (t, u, v) barycentric; t=1e30 on miss ----
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

// ---- Slab AABB test; returns true if ray hits box before tmax ----
fn aabbHit(bmin: vec3<f32>, bmax: vec3<f32>, ray: Ray, tmax: f32) -> bool {
    let invD = vec3<f32>(1.0) / ray.dir;
    let t1   = (bmin - ray.origin) * invD;
    let t2   = (bmax - ray.origin) * invD;
    let tNear = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tFar  = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    return tFar >= max(tNear, 0.0) && tNear < tmax;
}

// ---- Load and shade a triangle, update h if closer ----
fn testTriangle(ray: Ray, ti: i32, h: ptr<function, Hit>) {
    let r0 = textureLoad(triData, vec2<i32>(ti, 0), 0);
    let v0 = r0.xyz;
    let v1 = textureLoad(triData, vec2<i32>(ti, 1), 0).xyz;
    let v2 = textureLoad(triData, vec2<i32>(ti, 2), 0).xyz;

    let isect = triIntersect(ray, v0, v1, v2);
    if (isect.t >= (*h).t) { return; }

    let w   = 1.0 - isect.u - isect.v;
    let n0  = textureLoad(triData, vec2<i32>(ti, 3), 0).xyz;
    let n1  = textureLoad(triData, vec2<i32>(ti, 4), 0).xyz;
    let n2  = textureLoad(triData, vec2<i32>(ti, 5), 0).xyz;
    var sn  = normalize(n0 * w + n1 * isect.u + n2 * isect.v);

    let uv01 = textureLoad(triData, vec2<i32>(ti, 6), 0);
    let uv2  = textureLoad(triData, vec2<i32>(ti, 7), 0).xy;
    let iuv  = vec2<f32>(uv01.x, uv01.y) * w
             + vec2<f32>(uv01.z, uv01.w) * isect.u
             + uv2                        * isect.v;

    let matIdx = i32(r0.w);
    let mat0   = textureLoad(matData, vec2<i32>(matIdx, 0), 0);
    let mat1   = textureLoad(matData, vec2<i32>(matIdx, 1), 0);

    (*h).t         = isect.t;
    (*h).point     = ray.origin + isect.t * ray.dir;
    (*h).normal    = select(sn, -sn, dot(ray.dir, sn) > 0.0);
    (*h).albedo    = mat0.xyz;
    (*h).shininess = mat0.w;
    (*h).uv        = iuv;
    (*h).texSlot   = mat1.x;
}

// ---- BVH traversal ----
fn sceneHit(ray: Ray) -> Hit {
    var h: Hit;
    h.t = 1e30;

    var stack: array<i32, 32>;
    var top: i32 = 0;
    stack[0] = 0;
    top = 1;

    while (top > 0) {
        top -= 1;
        let ni  = stack[top];
        let nd0 = textureLoad(bvhData, vec2<i32>(ni, 0), 0);
        let nd1 = textureLoad(bvhData, vec2<i32>(ni, 1), 0);

        if (!aabbHit(nd0.xyz, nd1.xyz, ray, h.t)) { continue; }

        let left  = i32(nd0.w);
        let right = i32(nd1.w);

        if (left < 0) {
            // Leaf: test triangles
            let triStart = -left - 1;
            let triCount = right;
            for (var ti = triStart; ti < triStart + triCount; ti++) {
                testTriangle(ray, ti, &h);
            }
        } else {
            // Interior: push both children (closer last = popped first)
            stack[top] = right;
            top += 1;
            stack[top] = left;
            top += 1;
        }
    }
    return h;
}

// ---- Blinn-Phong shade with hard shadow ----
fn shade(h: Hit, rd: vec3<f32>, lp: vec3<f32>) -> vec3<f32> {
    // Resolve albedo: sample texture atlas or use material colour
    var albedo = h.albedo;
    if (h.texSlot >= 0.0) {
        let au = (h.texSlot + fract(h.uv.x)) / MAX_TEX_SLOTS;
        let av = fract(h.uv.y);
        albedo = textureSampleLevel(texAtlas, texAtlasSampler,
                                    vec2<f32>(au, av), 0.0).xyz;
    }

    let toL = lp - h.point;
    let ld  = length(toL);
    let ln  = toL / ld;

    var col = albedo * 0.10;

    var sr: Ray; sr.origin = h.point; sr.dir = ln;
    let sh = sceneHit(sr);
    if (sh.t >= ld - 1e-3) {
        col += albedo * max(0.0, dot(h.normal, ln)) * 0.80;
        let hv = normalize(normalize(-rd) + ln);
        col += pow(max(0.0, dot(h.normal, hv)), h.shininess + 1.0) * 0.65;
    }
    return clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---- Trace with one mirror-reflection bounce ----
fn trace(ray: Ray, lp: vec3<f32>) -> vec3<f32> {
    let h0 = sceneHit(ray);
    if (h0.t >= 1e30) { return sky(ray.dir); }

    var col = shade(h0, ray.dir, lp);

    if (h0.shininess > 20.0) {
        let k = min(0.55, h0.shininess / 100.0);
        var r1: Ray;
        r1.origin = h0.point;
        r1.dir    = reflect(ray.dir, h0.normal);
        let h1 = sceneHit(r1);
        var rc: vec3<f32>;
        if (h1.t >= 1e30) { rc = sky(r1.dir); }
        else               { rc = shade(h1, r1.dir, lp); }
        col = col * (1.0 - k) + rc * k;
    }
    return col;
}

@fragment
fn fs_main(@builtin(position) fragPos: vec4<f32>) -> @location(0) vec4<f32> {
    let res  = rt.iRes.xy;
    let time = rt.iTime.x;

    // NDC: WebGPU y=0 is at top, flip to get y=+1 at top of screen.
    let ndc = vec2<f32>(
        (fragPos.x / res.x) * 2.0 - 1.0,
        1.0 - (fragPos.y / res.y) * 2.0
    );

    let fwd = rt.camFwd.xyz;
    let rgt = rt.camRgt.xyz;
    let up  = rt.camUp.xyz;
    let tf  = rt.tanHalfFov.x;
    let asp = rt.aspect.x;

    var ray: Ray;
    ray.origin = rt.camOri.xyz;
    ray.dir    = normalize(fwd + rgt * (ndc.x * tf * asp) + up * (ndc.y * tf));

    let lp = vec3<f32>(
        5.0 * cos(time * 0.6),
        6.0 + sin(time * 0.3),
       -2.0 + 4.0 * sin(time * 0.6)
    );

    var col = trace(ray, lp);
    col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    return vec4<f32>(col, 1.0);
}
)";

// ---------------------------------------------------------------------------
// Extract material properties via interface casts
// ---------------------------------------------------------------------------
static std::pair<Color, float> extractMaterial(const Material* mat) {
    Color albedo(0.8f, 0.8f, 0.8f);
    float shininess = 8.f;
    if (!mat) return {albedo, shininess};

    if (auto* c = dynamic_cast<const MaterialWithColor*>(mat))
        albedo = c->color;

    if (auto* s = dynamic_cast<const MaterialWithSpecular*>(mat)) {
        shininess = std::max(1.f, s->shininess);
    } else if (auto* r = dynamic_cast<const MaterialWithRoughness*>(mat)) {
        shininess = std::max(1.f, (1.f - r->roughness) * 128.f);
    }
    return {albedo, shininess};
}

// ---------------------------------------------------------------------------
// Build texture atlas from mesh materials that have a map.
// Returns atlas pixel data (RGBA8, MAX_TEX_SLOTS*TILE_SIZE × TILE_SIZE)
// and fills texSlotMap: Texture* → slot index (0..MAX_TEX_SLOTS-1).
// ---------------------------------------------------------------------------
static std::vector<unsigned char> buildAtlas(
        const std::vector<std::shared_ptr<Mesh>>& meshes,
        std::unordered_map<Texture*, int>& texSlotMap)
{
    const int atlasW = MAX_TEX_SLOTS * TILE_SIZE;
    std::vector<unsigned char> atlas(atlasW * TILE_SIZE * 4, 255);

    int slot = 0;
    for (auto& mesh : meshes) {
        if (slot >= MAX_TEX_SLOTS) break;
        auto* mwm = dynamic_cast<MaterialWithMap*>(mesh->material().get());
        if (!mwm || !mwm->map) continue;
        Texture* tex = mwm->map.get();
        if (texSlotMap.count(tex)) continue;  // already assigned

        // Get raw image data (assume uchar RGBA/RGB)
        auto& img = tex->image();
        if (img.width == 0 || img.height == 0) continue;
        const auto& src = img.data<unsigned char>();
        const int srcW = static_cast<int>(img.width);
        const int srcH = static_cast<int>(img.height);
        const int channels = static_cast<int>(src.size()) / (srcW * srcH);

        // Blit into atlas slot with nearest-neighbour scale
        const int destX = slot * TILE_SIZE;
        for (int ty = 0; ty < TILE_SIZE; ++ty) {
            const int sy = ty * srcH / TILE_SIZE;
            for (int tx = 0; tx < TILE_SIZE; ++tx) {
                const int sx = tx * srcW / TILE_SIZE;
                const int si = (sy * srcW + sx) * channels;
                const int di = (ty * atlasW + destX + tx) * 4;
                atlas[di + 0] = src[si + 0];
                atlas[di + 1] = src[si + 1];
                atlas[di + 2] = src[si + 2];
                atlas[di + 3] = channels == 4 ? src[si + 3] : 255u;
            }
        }

        texSlotMap[tex] = slot++;
    }
    return atlas;
}

// ---------------------------------------------------------------------------
// Build CPU triangle/material buffers from a list of Mesh objects.
// Returns the number of triangles written.
// ---------------------------------------------------------------------------
static int buildGeometryBuffers(
        const std::vector<std::shared_ptr<Mesh>>& meshes,
        const std::unordered_map<Texture*, int>& texSlotMap,
        std::vector<float>& triBuffer,   // MAX_TRIS * TRI_TEX_HEIGHT rows * 4 floats
        std::vector<float>& matBuffer)   // MAX_MATS * MAT_TEX_HEIGHT rows * 4 floats
{
    std::ranges::fill(triBuffer, 0.f);
    std::ranges::fill(matBuffer, 0.f);

    int triCount = 0;
    int matCount = 0;

    auto setTexel = [&](std::vector<float>& buf, int width, int col, int row,
                        float x, float y, float z, float w) {
        const int idx = (row * width + col) * 4;
        buf[idx + 0] = x; buf[idx + 1] = y;
        buf[idx + 2] = z; buf[idx + 3] = w;
    };

    for (auto& mesh : meshes) {
        if (triCount >= MAX_TRIS || matCount >= MAX_MATS) break;

        // Material row 0: albedo + shininess
        auto [albedo, shininess] = extractMaterial(mesh->material().get());
        setTexel(matBuffer, MAX_MATS, matCount, 0,
                 albedo.r, albedo.g, albedo.b, shininess);

        // Material row 1: texture slot (-1 = none)
        float texSlot = -1.f;
        if (auto* mwm = dynamic_cast<MaterialWithMap*>(mesh->material().get())) {
            if (mwm->map) {
                auto it = texSlotMap.find(mwm->map.get());
                if (it != texSlotMap.end())
                    texSlot = static_cast<float>(it->second);
            }
        }
        setTexel(matBuffer, MAX_MATS, matCount, 1, texSlot, 0.f, 0.f, 0.f);

        const int matIdx = matCount++;

        // Geometry
        mesh->updateMatrixWorld(true);
        const auto& world = *mesh->matrixWorld;
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
            n.transformDirection(world);
            return n;
        };
        auto uv = [&](int i) -> std::pair<float, float> {
            if (!uvs) return {0.f, 0.f};
            return {uvs->getX(i), uvs->getY(i)};
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
    int   left;   // >= 0: left child index   < 0: leaf, triStart = -left-1
    float maxX, maxY, maxZ;
    int   right;  // interior: right child index   leaf: triangle count
};

// Read one float component from the flat triBuffer (row-major, RGBA texels)
static float triGet(const std::vector<float>& buf, int ti, int row, int comp) {
    return buf[(row * MAX_TRIS + ti) * 4 + comp];
}

static int buildBvhNode(
        std::vector<BvhNode>& nodes,
        std::vector<int>& idx,
        const std::vector<float>& buf,
        int start, int end)
{
    const int ni = static_cast<int>(nodes.size());
    nodes.emplace_back();

    // Compute AABB over triangles in [start, end)
    float minX = 1e30f, minY = 1e30f, minZ = 1e30f;
    float maxX =-1e30f, maxY =-1e30f, maxZ =-1e30f;
    for (int i = start; i < end; i++) {
        const int ti = idx[i];
        for (int r = 0; r <= 2; r++) {
            const float x = triGet(buf, ti, r, 0);
            const float y = triGet(buf, ti, r, 1);
            const float z = triGet(buf, ti, r, 2);
            minX = std::min(minX, x); minY = std::min(minY, y); minZ = std::min(minZ, z);
            maxX = std::max(maxX, x); maxY = std::max(maxY, y); maxZ = std::max(maxZ, z);
        }
    }
    nodes[ni].minX = minX; nodes[ni].minY = minY; nodes[ni].minZ = minZ;
    nodes[ni].maxX = maxX; nodes[ni].maxY = maxY; nodes[ni].maxZ = maxZ;

    const int count = end - start;
    if (count <= 2) {
        nodes[ni].left  = -(start + 1);  // leaf: encode triStart as negative
        nodes[ni].right = count;
        return ni;
    }

    // Split along longest axis at centroid midpoint
    const float dx = maxX - minX, dy = maxY - minY, dz = maxZ - minZ;
    const int axis = (dx >= dy && dx >= dz) ? 0 : (dy >= dz ? 1 : 2);
    const float axMins[3] = {minX, minY, minZ};
    const float axMaxs[3] = {maxX, maxY, maxZ};
    const float split = (axMins[axis] + axMaxs[axis]) * 0.5f;

    auto mid = std::partition(idx.begin() + start, idx.begin() + end, [&](int ti) {
        const float c = (triGet(buf, ti, 0, axis)
                       + triGet(buf, ti, 1, axis)
                       + triGet(buf, ti, 2, axis)) / 3.f;
        return c < split;
    });
    int sp = static_cast<int>(mid - idx.begin());
    if (sp == start || sp == end) sp = (start + end) / 2;

    const int lc = buildBvhNode(nodes, idx, buf, start, sp);
    const int rc = buildBvhNode(nodes, idx, buf, sp, end);
    nodes[ni].left  = lc;
    nodes[ni].right = rc;
    return ni;
}

// Build BVH over triCount triangles.  Reorders triBuffer in-place to match
// BVH leaf order.  Returns packed node data ready for GPU upload.
static std::vector<float> buildBVH(std::vector<float>& triBuffer, int triCount) {
    std::vector<int> indices(triCount);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<BvhNode> nodes;
    nodes.reserve(triCount * 2);
    buildBvhNode(nodes, indices, triBuffer, 0, triCount);

    // Reorder triBuffer to match the sorted index order
    std::vector<float> sorted(triBuffer.size(), 0.f);
    for (int ni = 0; ni < triCount; ni++) {
        const int oi = indices[ni];
        for (int row = 0; row < TRI_TEX_HEIGHT; row++)
            for (int c = 0; c < 4; c++)
                sorted[(row * MAX_TRIS + ni) * 4 + c] =
                    triBuffer[(row * MAX_TRIS + oi) * 4 + c];
    }
    triBuffer = std::move(sorted);

    // Pack into RGBA32Float texture rows (MAX_BVH_NODES wide, 2 tall)
    std::vector<float> bvhBuf(MAX_BVH_NODES * 2 * 4, 0.f);
    const int nc = std::min(static_cast<int>(nodes.size()), MAX_BVH_NODES);
    for (int i = 0; i < nc; i++) {
        const auto& n = nodes[i];
        bvhBuf[(0 * MAX_BVH_NODES + i) * 4 + 0] = n.minX;
        bvhBuf[(0 * MAX_BVH_NODES + i) * 4 + 1] = n.minY;
        bvhBuf[(0 * MAX_BVH_NODES + i) * 4 + 2] = n.minZ;
        bvhBuf[(0 * MAX_BVH_NODES + i) * 4 + 3] = static_cast<float>(n.left);
        bvhBuf[(1 * MAX_BVH_NODES + i) * 4 + 0] = n.maxX;
        bvhBuf[(1 * MAX_BVH_NODES + i) * 4 + 1] = n.maxY;
        bvhBuf[(1 * MAX_BVH_NODES + i) * 4 + 2] = n.maxZ;
        bvhBuf[(1 * MAX_BVH_NODES + i) * 4 + 3] = static_cast<float>(n.right);
    }
    return bvhBuf;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {

    Canvas canvas("Wgpu GPU Raytracer – Triangle Mesh",
                  {{"graphicsApi", GraphicsAPI::WebGPU}});

    WgpuRenderer renderer(canvas);
    renderer.setClearColor(Color(0x000000));

    // ---- GPU textures for geometry data ----
    WgpuTexture bvhTex(renderer, MAX_BVH_NODES, 2,
                       WgpuTexture::Format::RGBA32Float,
                       WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
    WgpuTexture matTex(renderer, MAX_MATS, MAT_TEX_HEIGHT,
                       WgpuTexture::Format::RGBA32Float,
                       WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
    // Texture atlas: MAX_TEX_SLOTS tiles of TILE_SIZE×TILE_SIZE (RGBA8Unorm)
    WgpuTexture texAtlasTex(renderer, MAX_TEX_SLOTS * TILE_SIZE, TILE_SIZE,
                            WgpuTexture::Format::RGBA8Unorm,
                            WgpuTexture::TextureBinding | WgpuTexture::CopyDst);
    WgpuTexture triTex(renderer, MAX_TRIS, TRI_TEX_HEIGHT,
                       WgpuTexture::Format::RGBA32Float,
                       WgpuTexture::TextureBinding | WgpuTexture::CopyDst);

    // CPU buffers for geometry data
    std::vector<float> triBuffer(MAX_TRIS * TRI_TEX_HEIGHT * 4, 0.f);
    std::vector<float> matBuffer(MAX_MATS * MAT_TEX_HEIGHT * 4, 0.f);

    // ---- Raytracer camera + OrbitControls ----
    PerspectiveCamera rtCam(60.f, canvas.aspect(), 0.1f, 200.f);
    rtCam.position.set(0.f, 3.f, 8.f);

    OrbitControls controls{rtCam, canvas};
    controls.target.set(0.f, 0.f, 0.f);
    controls.update();

    const float tanHalfFov = std::tan(60.f * math::PI / 360.f);
    float aspect = canvas.aspect();

    // ---- Fullscreen quad with WGSL ShaderMaterial ----
    OrthographicCamera displayCam(-1.f, 1.f, 1.f, -1.f, 0.1f, 10.f);
    displayCam.position.z = 1.f;

    auto rtMat = ShaderMaterial::create();
    rtMat->vertexShader   = vsWGSL;
    rtMat->fragmentShader = fsWGSL;

    // Custom textures bound alphabetically: bvhData(3,4) matData(5,6) texAtlas(7,8) triData(9,10)
    rtMat->customTextures["bvhData"]  = &bvhTex;
    rtMat->customTextures["matData"]  = &matTex;
    rtMat->customTextures["texAtlas"] = &texAtlasTex;
    rtMat->customTextures["triData"]  = &triTex;

    // Uniforms (alphabetical: aspect camFwd camOri camRgt camUp iRes iTime tanHalfFov triCount)
    auto sz = canvas.size();
    rtMat->uniforms["aspect"]     = Uniform(aspect);
    rtMat->uniforms["camFwd"]     = Uniform(Color(0.f, -0.35f, -0.94f));
    rtMat->uniforms["camOri"]     = Uniform(Color(0.f,  3.f,    8.f));
    rtMat->uniforms["camRgt"]     = Uniform(Color(1.f,  0.f,    0.f));
    rtMat->uniforms["camUp"]      = Uniform(Color(0.f,  1.f,    0.f));
    rtMat->uniforms["iRes"]       = Uniform(Color(float(sz.width()), float(sz.height()), 0.f));
    rtMat->uniforms["iTime"]      = Uniform(0.f);
    rtMat->uniforms["tanHalfFov"] = Uniform(tanHalfFov);
    rtMat->uniforms["triCount"]   = Uniform(0.f);

    Scene displayScene;
    displayScene.add(Mesh::create(PlaneGeometry::create(2.f, 2.f), rtMat));

    canvas.onWindowResize([&](WindowSize ns) {
        renderer.setSize(ns);
        aspect = ns.aspect();
        rtMat->uniforms["iRes"] = Uniform(Color(float(ns.width()), float(ns.height()), 0.f));
    });

    // -------------------------------------------------------------------------
    // Demo scene objects  — standard threepp Mesh + geometry + material
    // -------------------------------------------------------------------------

    TextureLoader tl;
    auto tex = tl.load(std::string(DATA_FOLDER) + "/textures/uv_grid_opengl.jpg");

    // Rotating gold box
    auto boxMesh = Mesh::create(
            BoxGeometry::create(1.5f, 1.5f, 1.5f),
            MeshStandardMaterial::create({{"map", tex},
                                          {"roughness", 1.f}}));
    boxMesh->position.set(0.f, 1.f, -1.f);

    // Shiny red sphere (left)
    auto sphere1 = Mesh::create(
            SphereGeometry::create(0.85f),
            MeshStandardMaterial::create({{"color", Color(0.90f, 0.18f, 0.10f)},
                                          {"roughness", 0.7f}}));
    sphere1->position.set(-2.8f, 1.f, 0.f);

    // Semi-shiny blue sphere (right)
    auto sphere2 = Mesh::create(
            SphereGeometry::create(0.85f ),
            MeshStandardMaterial::create({{"color", Color(0.15f, 0.45f, 0.95f)},
                                          {"roughness", 0.25f}}));
    sphere2->position.set(2.8f, 1.f, 0.f);

    // Large floor plane (rotated -90° around X)
    auto floor = Mesh::create(
            PlaneGeometry::create(16.f, 16.f, 4, 4),
            MeshStandardMaterial::create({{"color", Color(0.55f, 0.55f, 0.55f)},
                                          {"roughness", 0.9f}}));
    floor->rotation.x = -math::PI / 2.f;
    floor->position.y = -1.f;

    // Collect all RT scene meshes
    std::vector<std::shared_ptr<Mesh>> rtMeshes = {boxMesh, sphere1, sphere2, floor};

    // Build texture atlas (once — textures don't change per-frame)
    std::unordered_map<Texture*, int> texSlotMap;
    auto atlasData = buildAtlas(rtMeshes, texSlotMap);
    texAtlasTex.write(atlasData.data(), atlasData.size());

    // ---- Animation ----
    Clock clock;
    float elapsed = 0.f;

    canvas.animate([&] {
        const float dt = clock.getDelta();
        elapsed += dt;

        // Animate the box
        boxMesh->rotation.y += dt * 0.6f;
        boxMesh->rotation.x += dt * 0.3f;

        // Update camera orientation from OrbitControls
        controls.update();
        const Vector3& pos = rtCam.position;
        Vector3 fwd = Vector3(controls.target).sub(pos).normalize();
        Vector3 rgt = Vector3(fwd).cross(Vector3(0.f, 1.f, 0.f)).normalize();
        Vector3 up  = Vector3(rgt).cross(fwd);

        // Rebuild world-space triangle data, then BVH (triBuffer is reordered in-place)
        const int triCount = buildGeometryBuffers(rtMeshes, texSlotMap, triBuffer, matBuffer);
        const auto bvhBuffer = buildBVH(triBuffer, triCount);

        bvhTex.write(bvhBuffer.data(), bvhBuffer.size() * sizeof(float));
        triTex.write(triBuffer.data(), triBuffer.size() * sizeof(float));
        matTex.write(matBuffer.data(), matBuffer.size() * sizeof(float));

        // Push uniforms
        rtMat->uniforms["aspect"]     = Uniform(aspect);
        rtMat->uniforms["camFwd"]     = Uniform(Color(fwd.x, fwd.y, fwd.z));
        rtMat->uniforms["camOri"]     = Uniform(Color(pos.x, pos.y, pos.z));
        rtMat->uniforms["camRgt"]     = Uniform(Color(rgt.x, rgt.y, rgt.z));
        rtMat->uniforms["camUp"]      = Uniform(Color(up.x,  up.y,  up.z));
        rtMat->uniforms["iTime"]      = Uniform(elapsed);
        rtMat->uniforms["tanHalfFov"] = Uniform(tanHalfFov);
        rtMat->uniforms["triCount"]   = Uniform(static_cast<float>(triCount));
        rtMat->uniformsNeedUpdate     = true;

        renderer.render(displayScene, displayCam);
    });

    return 0;
}
