
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
