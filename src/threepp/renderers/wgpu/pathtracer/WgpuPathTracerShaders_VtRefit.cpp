#include "WgpuPathTracerShaders.hpp"

namespace threepp::wgpu_pt {

// Shared WGSL definitions used by multiple shaders (VT, refit, RT).
const char* const csSharedDefsWGSL = R"(
const TRI_PAGE_W:  i32 = 8192;
const TRI_PAGE_H:  i32 = 8;
const MAX_LEAF_TRIS: i32 = 8;

fn triCoord(ti: i32, row: i32) -> vec2<i32> {
    return vec2<i32>(ti % TRI_PAGE_W, (ti / TRI_PAGE_W) * TRI_PAGE_H + row);
}
)";

// ---------------------------------------------------------------------------
// WGSL vertex-transform compute shader
// ---------------------------------------------------------------------------
const char* const vtWGSL_ = R"(
struct ObjTriData {
    v0:   vec4<f32>,
    v1:   vec4<f32>,
    v2:   vec4<f32>,
    n0:   vec4<f32>,
    n1:   vec4<f32>,
    n2:   vec4<f32>,
    uv01: vec4<f32>,
    uv2:  vec4<f32>,   // .w unused (was cb2 vertex-color component)
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
}
)";

// ---------------------------------------------------------------------------
// WGSL BVH refit compute shader
// ---------------------------------------------------------------------------
const char* const refitWGSL_ = R"(
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

// ---------------------------------------------------------------------------
// WGSL BLAS refit compute shader (TLAS mode)
// ---------------------------------------------------------------------------
// Refits BLAS leaf AABBs from object-space triangles in `objTris`/`objTris2`,
// then propagates up parent AABBs within each BLAS sub-tree.  Propagation
// terminates at BLAS roots (parent == -1), so the single flat `blasNodes`
// buffer is walked correctly without crossing BLAS boundaries.
//
// Separate pipeline from bvh_refit because:
//   - reads object-space tri vertices (not the world-space triTex)
//   - writes to the `blasNodes` buffer (not `bvhNodes`)
//   - needs its own atomic counter + leaf-index + refit-metadata buffers
const char* const blasRefitWGSL_ = R"(
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
struct Bvh4NodeGpu {
    cMinX: vec4<f32>,
    cMinY: vec4<f32>,
    cMinZ: vec4<f32>,
    cMaxX: vec4<f32>,
    cMaxY: vec4<f32>,
    cMaxZ: vec4<f32>,
    cIdx:  vec4<u32>,
}
struct BlasRefitUniforms {
    leafCount: u32,
    groupsX:   u32,
    splitAt:   u32,   // objTri split point (in tris); tis < splitAt read objTris, else objTris2
    _p:        u32,
}

@group(0) @binding(0) var<storage, read>       objTris:    array<ObjTriData>;
@group(0) @binding(1) var<storage, read>       objTris2:   array<ObjTriData>;
@group(0) @binding(2) var<storage, read_write> blasNodes:  array<Bvh4NodeGpu>;
@group(0) @binding(3) var<storage, read_write> blasCtrs:   array<atomic<u32>>;
@group(0) @binding(4) var<storage, read>       leafIdxBuf: array<i32>;
@group(0) @binding(5) var<uniform>             refitUni:   BlasRefitUniforms;
@group(0) @binding(6) var<storage, read>       refitMeta:  array<vec4<i32>>;

fn loadObjTri(ti: i32) -> ObjTriData {
    let splitAt = i32(refitUni.splitAt);
    if (ti < splitAt) { return objTris[ti]; }
    return objTris2[ti - splitAt];
}

fn writeChildAABB(ni: i32, c: i32, bmin: vec3<f32>, bmax: vec3<f32>) {
    let extent = max(abs(bmin), abs(bmax));
    let E = max(extent * vec3<f32>(1e-5), vec3<f32>(1e-6));
    var n = blasNodes[ni];
    n.cMinX[c] = bmin.x - E.x;
    n.cMinY[c] = bmin.y - E.y;
    n.cMinZ[c] = bmin.z - E.z;
    n.cMaxX[c] = bmax.x + E.x;
    n.cMaxY[c] = bmax.y + E.y;
    n.cMaxZ[c] = bmax.z + E.z;
    blasNodes[ni] = n;
}

@compute @workgroup_size(64)
fn blas_refit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linearId = gid.x + gid.y * refitUni.groupsX * 64u;
    if (linearId >= refitUni.leafCount) { return; }
    let wideNi = leafIdxBuf[i32(linearId)];
    let nfo = refitMeta[wideNi];
    let childCount = nfo.y;

    let cIdxVec = blasNodes[wideNi].cIdx;
    let leafIdx = array<i32, 4>(bitcast<i32>(cIdxVec.x), bitcast<i32>(cIdxVec.y),
                                 bitcast<i32>(cIdxVec.z), bitcast<i32>(cIdxVec.w));

    for (var c: i32 = 0; c < childCount; c++) {
        let ci = leafIdx[c];
        if (ci >= 0) { continue; }

        let raw = -ci;
        let triStart = (raw - 1) / MAX_LEAF_TRIS;
        let triCount = ((raw - 1) % MAX_LEAF_TRIS) + 1;

        var bmin = vec3<f32>(1e30);
        var bmax = vec3<f32>(-1e30);
        for (var ti = triStart; ti < triStart + triCount; ti++) {
            let ot = loadObjTri(ti);
            bmin = min(bmin, ot.v0.xyz); bmax = max(bmax, ot.v0.xyz);
            bmin = min(bmin, ot.v1.xyz); bmax = max(bmax, ot.v1.xyz);
            bmin = min(bmin, ot.v2.xyz); bmax = max(bmax, ot.v2.xyz);
        }
        writeChildAABB(wideNi, c, bmin, bmax);
    }

    let numInternal = nfo.z;
    if (numInternal > 0) {
        let cnt = atomicAdd(&blasCtrs[wideNi], 1u);
        if (cnt < u32(numInternal)) { return; }
    }

    // Propagate up to parent — parent == -1 is the BLAS root, stop there.
    var curNi = nfo.x;
    loop {
        if (curNi < 0) { break; }
        let curNfo = refitMeta[curNi];
        let numInt = u32(curNfo.z);
        let cnt = atomicAdd(&blasCtrs[curNi], 1u);
        let curChildCount = curNfo.y;
        let hasLeaves = u32(curChildCount) > numInt;
        let expected = select(numInt - 1u, numInt, hasLeaves);
        if (cnt < expected) { break; }

        let pIdxVec = blasNodes[curNi].cIdx;
        let pIdx = array<i32, 4>(bitcast<i32>(pIdxVec.x), bitcast<i32>(pIdxVec.y),
                                  bitcast<i32>(pIdxVec.z), bitcast<i32>(pIdxVec.w));
        for (var c: i32 = 0; c < curChildCount; c++) {
            let ci = pIdx[c];
            if (ci < 0) { continue; }
            let child = blasNodes[ci];
            let cc = refitMeta[ci].y;
            var bmin = vec3<f32>(1e30);
            var bmax = vec3<f32>(-1e30);
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

std::string buildVtShader() { return std::string(csSharedDefsWGSL) + "\n" + vtWGSL_; }
std::string buildRefitShader() { return std::string(csSharedDefsWGSL) + "\n" + refitWGSL_; }
std::string buildBlasRefitShader() { return std::string(csSharedDefsWGSL) + "\n" + blasRefitWGSL_; }

}// namespace threepp::wgpu_pt
