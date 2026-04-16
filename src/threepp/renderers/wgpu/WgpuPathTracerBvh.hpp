#ifndef THREEPP_WGPUPATHTRACERBVH_HPP
#define THREEPP_WGPUPATHTRACERBVH_HPP

// Private header — BVH builder for the path tracer. Builds a binary SAH BVH,
// collapses it to a 4-way tree (BVH4), and serializes it into the GPU buffer
// layout sampled by the WGSL traversal kernel.

#include "WgpuPathTracerTypes.hpp"

#include <cstdint>
#include <vector>

namespace threepp::wgpu_pt {

    /// Build a BVH4 over `triCount` triangles already packed in `triBuffer`.
    /// Reorders both `triBuffer` and `rawObjTriBuf` into BVH leaf order
    /// (cycle-permutation, in-place). `indices` is repurposed as scratch and
    /// also returned as the reorder permutation. `leafIndices` lists which
    /// BVH4 nodes contain leaf children (needed by the GPU refit pass).
    void buildBVH(std::vector<float>& triBuffer, int triCount,
                  std::vector<Bvh4Node>& wideNodes, std::vector<int>& indices,
                  std::vector<int>& leafIndices,
                  std::vector<float>& rawObjTriBuf);

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

    /// Pack BVH4 nodes into a flat GPU buffer (7 × vec4 = 28 floats per node).
    /// `capacity` caps the number of nodes written; remaining bytes are zeroed.
    void packBvh4Buffer(const std::vector<Bvh4Node>& nodes, std::vector<std::uint32_t>& buf, int capacity);

    /// Pack per-node refit metadata (parent, childCount, numInternalChildren, 0)
    /// into 4 ints per node.
    void packRefitMetadata(const std::vector<Bvh4Node>& nodes, std::vector<std::int32_t>& buf, int capacity);

}// namespace threepp::wgpu_pt

#endif//THREEPP_WGPUPATHTRACERBVH_HPP
