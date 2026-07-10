// Renderer-agnostic index-only LOD chain generator, built on the bundled
// meshoptimizer (src/external/meshoptimizer). Used by the Vulkan renderer's
// auto-LOD feature (VulkanCoreScene.cpp / VulkanCoreGeometry.cpp) to fight
// sub-pixel geometric coverage flicker under TAA jitter: far-away triangles
// that cover less than a pixel still rasterize/trace at full density, and
// their sub-pixel silhouette motion under camera jitter never converges.
//
// Deliberately narrow: this header/cpp knows nothing about Vulkan, buffers,
// or BLAS. It takes a position + index array and returns a chain of
// alternative INDEX buffers only — vertex positions/normals/uvs are never
// touched, reordered, or duplicated. Every returned index still refers into
// the CALLER's original vertex array, so a consumer that already keeps
// vertex attributes in a separate, tightly-packed, vertex-id-addressed
// buffer (as the Vulkan renderer's BlasRecord does) can swap in a coarser
// level by rebinding just the index buffer — unchanged in both the raster
// vertex-pulling path and the ray-tracing BLAS/TLAS.
#ifndef THREEPP_GEOMETRYLOD_HPP
#define THREEPP_GEOMETRYLOD_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

namespace threepp::geometrylod {

    // One simplified index level in a chain.
    //   indices : the level's triangle list, indexing into the ORIGINAL
    //             vertex array passed to generateChain (never remapped).
    //   error   : ABSOLUTE object-space deviation this level introduces
    //             relative to the source mesh (meshopt's relative error
    //             converted via meshopt_simplifyScale). Monotonically
    //             non-decreasing across the returned chain, so a caller
    //             doing screen-space-error LOD selection can walk the
    //             chain coarsest-first / finest-first without the error
    //             metric ever going backwards.
    struct Level {
        std::vector<uint32_t> indices;
        float error = 0.f;
    };

    // Generates a progressive simplification chain from a single geometry's
    // position + index arrays.
    //
    // Ordering: chain[0] is the FIRST (finest) simplification beyond the
    // source mesh (~50% of the source triangle count); chain.back() is the
    // COARSEST level produced. Each level is generated from the PREVIOUS
    // level's index buffer (not always from the original), so simplification
    // cost is roughly linear in the chain length rather than growing with
    // level count — standard "progressive chain" construction. A caller
    // pairs chain[i] with LOD-select level (i+1); level 0 (not present in
    // the returned vector) is always the source mesh itself.
    //
    // positions: tightly packed xyz floats (stride 3 floats/vertex).
    // indices: the source mesh's triangle list (indices into `positions`).
    //
    // Policy (tuned for ~sub-pixel-error targeting, see VulkanCoreScene.cpp's
    // selection code for how `error` is consumed):
    //   - Each level targets ~50% of the previous level's index count.
    //   - meshopt_simplify runs with a generous relative target_error (0.05)
    //     so the INDEX-COUNT target dominates the stopping point, not the
    //     error bound — the chain is meant to be dense (small count steps),
    //     with the actual screen-space decision made later from `error`.
    //   - meshopt_SimplifyLockBorder is set: open-mesh boundary vertices
    //     (common on partial/clipped assets) never move, avoiding cracks
    //     between a simplified region and untouched neighboring geometry.
    //   - A level is discarded (chain generation stops) when: meshopt
    //     returns 0 indices (degenerate), the new index count doesn't drop
    //     below ~85% of the previous count (refuses to simplify further —
    //     continuing would waste a level on an imperceptible reduction), or
    //     the next target would fall below 384 indices (128 triangles).
    //   - Capped at 4 levels beyond the source mesh.
    //   - `error` is clamped non-decreasing (error[i] = max(error[i], error[i-1]))
    //     so a later, coarser level can never report a smaller screen-space
    //     error than an earlier, finer one (meshopt's greedy simplifier
    //     doesn't strictly guarantee monotonic error growth level-to-level).
    //
    // Returns an empty vector if the source is too small/degenerate to
    // simplify at all (already below the minimum index count, unindexed
    // triangle soup with < 3 indices, or a null pointer).
    std::vector<Level> generateChain(const float* positions, size_t vertexCount,
                                      const uint32_t* indices, size_t indexCount);

}// namespace threepp::geometrylod

#endif//THREEPP_GEOMETRYLOD_HPP
