#include "WgpuPathTracerBvh.hpp"
#include "WgpuPathTracerGeometry.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <numeric>

namespace threepp::wgpu_pt {

namespace {

float triGet(const std::vector<float>& buf, int ti, int row, int comp) {
    const int page = ti / TEX_PAGE_WIDTH;
    const int pcol = ti % TEX_PAGE_WIDTH;
    return buf[((page * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + pcol) * 4 + comp];
}

float boxSurfaceArea(float mnX, float mnY, float mnZ,
                     float mxX, float mxY, float mxZ) {
    const float dx = mxX - mnX, dy = mxY - mnY, dz = mxZ - mnZ;
    return 2.f * (dx * dy + dy * dz + dz * dx);
}

// Compute centroid of triangle ti along a given axis (0=X, 1=Y, 2=Z).
float triCentroid(const std::vector<float>& buf, int ti, int axis) {
    return (triGet(buf, ti, 0, axis) + triGet(buf, ti, 1, axis) + triGet(buf, ti, 2, axis)) / 3.f;
}

int buildBvhNode(
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
//
// Leaf encoding: childIdx = -(triStart * MAX_LEAF_TRIS + triCount), where triCount is 1-based.
// Decode: raw = -childIdx; triStart = (raw - 1) / MAX_LEAF_TRIS; triCount = ((raw - 1) % MAX_LEAF_TRIS) + 1

/// Collapse a binary BVH into a BVH4 tree.
/// Walk the binary tree and greedily expand internal children until each node
/// has up to 4 children (which may be leaves or further internal BVH4 nodes).
void collapseBvh4(
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

// ---------------------------------------------------------------------------
// F16 packing helpers (currently unused — preserved for future BVH packing
// experiments that may quantize AABBs to f16 to halve bandwidth).
// ---------------------------------------------------------------------------

// Convert f32 to IEEE 754 half-precision (f16), returned as uint16_t.
// roundUp: true = round toward +∞ (for AABB max), false = round toward -∞ (for AABB min).
[[maybe_unused]] uint16_t f32ToF16(float value, bool roundUp = false) {
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
[[maybe_unused]] uint32_t packF16x2(float a, float b) {
    return uint32_t(f32ToF16(a)) | (uint32_t(f32ToF16(b)) << 16);
}
// Conservative packing for AABBs: min rounds down, max rounds up.
[[maybe_unused]] uint32_t packF16x2Min(float a, float b) {
    return uint32_t(f32ToF16(a, false)) | (uint32_t(f32ToF16(b, false)) << 16);
}
[[maybe_unused]] uint32_t packF16x2Max(float a, float b) {
    return uint32_t(f32ToF16(a, true)) | (uint32_t(f32ToF16(b, true)) << 16);
}

// Compute f16 ULP-based epsilon: expand AABB so thin boxes survive f16 quantization.
// At value v, f16 precision is ~|v|/1024. We pad by 2 ULPs + a small absolute floor.
[[maybe_unused]] float f16Eps(float v) {
    return std::max(std::abs(v) * (2.f / 1024.f), 1e-4f);
}

}// namespace

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
void packBvh4Buffer(const std::vector<Bvh4Node>& nodes, std::vector<uint32_t>& buf, int capacity) {
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

void packRefitMetadata(const std::vector<Bvh4Node>& nodes, std::vector<int32_t>& buf, int capacity) {
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

void buildBVH(std::vector<float>& triBuffer, int triCount,
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

    // Sort triangle data to match BVH index ordering.
    // Uses in-place cycle-following permutation to avoid allocating two full copies
    // (~1.2 GB for large scenes). Extra memory: O(n/8) for the visited bitmap + 2 tri temps.
    {
        std::vector<bool> visited(triCount, false);
        std::vector<float> tmpTri(TRI_TEX_HEIGHT * 4);   // one triangle's worth of triBuffer rows (8*4=32)
        std::array<float, 32> tmpObj{};                   // one triangle's worth of rawObjTriBuf (32 floats)

        for (int i = 0; i < triCount; i++) {
            if (visited[i]) continue;
            visited[i] = true;
            if (indices[i] == i) continue;   // already in place

            // Save element at cycle start
            for (int row = 0; row < TRI_TEX_HEIGHT; row++)
                for (int c = 0; c < 4; c++)
                    tmpTri[row * 4 + c] = triBuffer[pagedIdx(i, row) + c];
            std::memcpy(tmpObj.data(), rawObjTriBuf.data() + i * 32, 32 * sizeof(float));

            int j = i;
            while (true) {
                const int k = indices[j];
                if (k == i) break;   // cycle closes back to start
                // Slide A[k] (not yet touched) into A[j]
                for (int row = 0; row < TRI_TEX_HEIGHT; row++)
                    for (int c = 0; c < 4; c++)
                        triBuffer[pagedIdx(j, row) + c] = triBuffer[pagedIdx(k, row) + c];
                std::memcpy(rawObjTriBuf.data() + j * 32,
                            rawObjTriBuf.data() + k * 32, 32 * sizeof(float));
                visited[k] = true;
                j = k;
            }
            // j is the last position in the cycle; receives the saved start element
            for (int row = 0; row < TRI_TEX_HEIGHT; row++)
                for (int c = 0; c < 4; c++)
                    triBuffer[pagedIdx(j, row) + c] = tmpTri[row * 4 + c];
            std::memcpy(rawObjTriBuf.data() + j * 32, tmpObj.data(), 32 * sizeof(float));
        }
    }
}

void buildOverlayBVH(
    std::vector<float>& triBuffer,
    std::vector<float>& rawObjTriBuf,
    int oldTriCount, int newTriCount,
    std::vector<Bvh4Node>& overlayNodes,
    std::vector<int>& overlayLeafIndices)
{
    if (newTriCount <= 0) { overlayNodes.clear(); overlayLeafIndices.clear(); return; }

    // Extract new tris into local buffers (indices 0..newTriCount-1).
    // localTriBuf must be sized for the paged layout: at least one full page even for small counts.
    const int localPages = triTexPages(newTriCount);
    const size_t localTrisWords = static_cast<size_t>(localPages) * TEX_PAGE_WIDTH * TRI_TEX_HEIGHT * 4;
    const size_t localObjWords  = static_cast<size_t>(newTriCount) * 32;
    std::vector<float> localTriBuf(localTrisWords, 0.f);
    std::vector<float> localObjBuf(localObjWords, 0.f);

    for (int li = 0; li < newTriCount; li++) {
        const int gi = oldTriCount + li;
        for (int row = 0; row < TRI_TEX_HEIGHT; row++) {
            const int lp = ((li / TEX_PAGE_WIDTH * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + li % TEX_PAGE_WIDTH) * 4;
            const int gp = pagedIdx(gi, row);
            for (int c = 0; c < 4; c++) localTriBuf[lp + c] = triBuffer[gp + c];
        }
        std::memcpy(localObjBuf.data() + li * 32, rawObjTriBuf.data() + gi * 32, 32 * sizeof(float));
    }

    // Build binary BVH over local indices
    std::vector<int> localIdx(newTriCount);
    std::iota(localIdx.begin(), localIdx.end(), 0);
    std::vector<BvhNode> binNodes;
    binNodes.reserve(newTriCount * 2);
    buildBvhNode(binNodes, localIdx, localTriBuf, 0, newTriCount, -1);

    // Collapse to BVH4
    overlayNodes.clear();
    overlayLeafIndices.clear();
    if (!binNodes.empty()) {
        overlayNodes.reserve(binNodes.size() / 2);
        collapseBvh4(binNodes, overlayNodes, overlayLeafIndices, 0, -1);
    }

    // Sort local buffers to BVH leaf order (same cycle-permutation as buildBVH)
    {
        std::vector<bool> visited(newTriCount, false);
        std::vector<float> tmpTri(TRI_TEX_HEIGHT * 4);
        std::array<float, 32> tmpObj{};

        for (int i = 0; i < newTriCount; i++) {
            if (visited[i]) continue;
            visited[i] = true;
            if (localIdx[i] == i) continue;

            for (int row = 0; row < TRI_TEX_HEIGHT; row++)
                for (int c = 0; c < 4; c++) {
                    const int lp = ((i / TEX_PAGE_WIDTH * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + i % TEX_PAGE_WIDTH) * 4;
                    tmpTri[row * 4 + c] = localTriBuf[lp + c];
                }
            std::memcpy(tmpObj.data(), localObjBuf.data() + i * 32, 32 * sizeof(float));

            int j = i;
            while (true) {
                const int k = localIdx[j];
                if (k == i) break;
                for (int row = 0; row < TRI_TEX_HEIGHT; row++)
                    for (int c = 0; c < 4; c++) {
                        const int jp = ((j / TEX_PAGE_WIDTH * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + j % TEX_PAGE_WIDTH) * 4;
                        const int kp = ((k / TEX_PAGE_WIDTH * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + k % TEX_PAGE_WIDTH) * 4;
                        localTriBuf[jp + c] = localTriBuf[kp + c];
                    }
                std::memcpy(localObjBuf.data() + j * 32, localObjBuf.data() + k * 32, 32 * sizeof(float));
                visited[k] = true;
                j = k;
            }
            for (int row = 0; row < TRI_TEX_HEIGHT; row++)
                for (int c = 0; c < 4; c++) {
                    const int jp = ((j / TEX_PAGE_WIDTH * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + j % TEX_PAGE_WIDTH) * 4;
                    localTriBuf[jp + c] = tmpTri[row * 4 + c];
                }
            std::memcpy(localObjBuf.data() + j * 32, tmpObj.data(), 32 * sizeof(float));
        }
    }

    // Offset all leaf triStart values in overlayNodes from local to global indices
    for (auto& node : overlayNodes) {
        for (int c = 0; c < 4; c++) {
            const int ci = node.childIdx[c];
            if (ci >= 0 || ci == INT_MIN) continue;  // internal or empty
            const int raw = -ci;
            const int localStart = (raw - 1) / MAX_LEAF_TRIS;
            const int cnt = ((raw - 1) % MAX_LEAF_TRIS) + 1;
            node.childIdx[c] = -(((localStart + oldTriCount) * MAX_LEAF_TRIS) + cnt);
        }
    }

    // Copy sorted local buffers back to global arrays at offset
    for (int li = 0; li < newTriCount; li++) {
        const int gi = oldTriCount + li;
        for (int row = 0; row < TRI_TEX_HEIGHT; row++) {
            const int lp = ((li / TEX_PAGE_WIDTH * TRI_TEX_HEIGHT + row) * TEX_PAGE_WIDTH + li % TEX_PAGE_WIDTH) * 4;
            const int gp = pagedIdx(gi, row);
            for (int c = 0; c < 4; c++) triBuffer[gp + c] = localTriBuf[lp + c];
        }
        std::memcpy(rawObjTriBuf.data() + gi * 32, localObjBuf.data() + li * 32, 32 * sizeof(float));
    }
}

}// namespace threepp::wgpu_pt
