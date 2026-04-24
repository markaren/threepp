#ifndef THREEPP_WGPUPATHTRACERBVH_HPP
#define THREEPP_WGPUPATHTRACERBVH_HPP

// Private header — BVH builder for the path tracer. Builds a binary SAH BVH,
// collapses it to a 4-way tree (BVH4), and serializes it into the GPU buffer
// layout sampled by the WGSL traversal kernel.

#include "threepp/renderers/wgpu/pathtracer/WgpuPathTracerTypes.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace threepp::wgpu_pt {

    struct RtMeshEntry;

    /// Build a BVH4 over `triCount` triangles already packed in `triBuffer`.
    /// Reorders `triBuffer` into BVH leaf order (cycle-permutation, in-place).
    /// `indices` is repurposed as scratch and also returned as the reorder
    /// permutation. `leafIndices` lists which BVH4 nodes contain leaf children
    /// (needed by the GPU refit pass).
    ///
    /// If `preserveObjTriOrder` is false (default), `rawObjTriBuf` is reordered
    /// alongside `triBuffer` — the single-level BVH path relies on this so its
    /// leaf triStart values address both buffers consistently.  When true, the
    /// object-space buffer is left in its original entry-contiguous order — used
    /// by the TLAS/BLAS path where per-entry BLAS records already address
    /// `rawObjTriBuf` directly and must not be invalidated.
    void buildBVH(std::vector<float>& triBuffer, int triCount,
                  std::vector<Bvh4Node>& wideNodes, std::vector<int>& indices,
                  std::vector<int>& leafIndices,
                  std::vector<float>& rawObjTriBuf,
                  bool preserveObjTriOrder = false);

    /// Build a BVH for newly appended triangles at indices
    /// [oldTriCount, oldTriCount+newTriCount). On return `overlayNodes`
    /// contains a BVH4 sub-tree with leaf triStart values already offset to
    /// global indices, and triBuffer / rawObjTriBuf are sorted into the new
    /// leaf order at the same offsets.
    void buildOverlayBVH(
            std::vector<float>& triBuffer,
            std::vector<float>& rawObjTriBuf,
            int oldTriCount, int newTriCount,
            std::vector<Bvh4Node>& overlayNodes,
            std::vector<int>& overlayLeafIndices);

    /// Build a BVH4 over a contiguous slice of object-space triangles in
    /// `objTriBuf` at [triStartLocal, triStartLocal + triCount).  Vertex
    /// positions are read from 32-float fields 0/1/2 (xyz of each vertex).
    ///
    /// Nodes are appended to `blasNodes` (a shared buffer across all BLASes).
    /// Emitted leaf `triStart` values are global — they index directly into
    /// `objTriBuf`.  The object-space slice is reordered in-place to match
    /// BVH leaf order; the world-space `triBuffer` is *not* touched (world
    /// positions will be regenerated per-frame from `objTriBuf` in the TLAS
    /// path once PR2b lands).
    ///
    /// Returns a `BlasRecord` describing the new BLAS: `rootNodeOffset`
    /// (absolute index into `blasNodes`), `nodeCount`, `triStart`/`triCount`
    /// within `objTriBuf`, and the object-space root AABB.  `leafIndicesOut`
    /// receives absolute `blasNodes` indices of nodes with leaf children —
    /// needed later for per-BLAS refit.
    ///
    /// PR2a plumbing — not yet wired into the scene build.  Callable in
    /// isolation for unit testing.
    BlasRecord buildBlas(
            std::vector<float>& objTriBuf,
            int triStartLocal,
            int triCount,
            std::vector<Bvh4Node>& blasNodes,
            std::vector<int>& leafIndicesOut);

    /// PR2b plumbing — build one BLAS per `RtMeshEntry` with non-zero tris and
    /// emit a matching TlasInstance record.  `entryTriRanges` comes from
    /// buildGeometryBuffers (see its docs); entries with zero tris are skipped.
    /// `objTriBuf` is reordered in-place per-slice by each buildBlas call.
    /// Outputs are appended to `blasNodes`, `blasLeafIndices`, `blasRecords`,
    /// `tlasInstances`, and `tlasToEntryIdx` (per-instance index back into the
    /// source `entries` vector — required for per-frame TLAS matrix refresh).
    void buildBlasesForEntries(
            const std::vector<RtMeshEntry>& entries,
            const std::vector<std::pair<int, int>>& entryTriRanges,
            std::vector<float>& objTriBuf,
            std::vector<Bvh4Node>& blasNodes,
            std::vector<int>& blasLeafIndices,
            std::vector<BlasRecord>& blasRecords,
            std::vector<TlasInstance>& tlasInstances,
            std::vector<std::uint32_t>& tlasToEntryIdx);

    /// Pack BVH4 nodes into a flat GPU buffer (7 × vec4 = 28 floats per node).
    /// `capacity` caps the number of nodes written; remaining bytes are zeroed.
    void packBvh4Buffer(const std::vector<Bvh4Node>& nodes, std::vector<std::uint32_t>& buf, int capacity);

    /// Pack per-node refit metadata (parent, childCount, numInternalChildren, 0)
    /// into 4 ints per node.
    void packRefitMetadata(const std::vector<Bvh4Node>& nodes, std::vector<std::int32_t>& buf, int capacity);

}// namespace threepp::wgpu_pt

#endif//THREEPP_WGPUPATHTRACERBVH_HPP
