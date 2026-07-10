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
    //   - `error` accumulates ADDITIVELY down the chain: meshopt's
    //     result_error is measured against each call's INPUT (the previous
    //     level), so the honest — conservative — bound vs the ORIGINAL mesh
    //     is the running sum. Monotonically non-decreasing by construction.
    //
    // sparse: pass TRUE when `indices` references only a small subset of the
    // vertex array — i.e. canonical welded-soup indices from
    // buildCanonicalIndices below, where ~5/6 of the vertices are unreferenced
    // duplicates. It maps to meshopt_SimplifySparse, which is REQUIRED for
    // correctness there, not just speed: without it, meshopt's wedge analysis
    // runs over ALL vertices (simplifier.cpp buildPositionRemap), so every
    // referenced vertex picks up its unreferenced same-position duplicates as
    // extra wedges, classifies as complex/locked, and the simplifier returns
    // the mesh untouched. Sparse excludes unreferenced vertices from the
    // analysis and hands back indices in ORIGINAL vid space. Error caveat:
    // meshopt then reports error relative to the REFERENCED subset's extents;
    // we still convert with the full mesh's simplifyScale, which can only
    // OVER-estimate (subset extents <= full extents) — conservative for
    // screen-space-error selection. Leave FALSE for regular indexed input.
    //
    // Returns an empty vector if the source is too small/degenerate to
    // simplify at all (already below the minimum index count, unindexed
    // triangle soup with < 3 indices, or a null pointer).
    std::vector<Level> generateChain(const float* positions, size_t vertexCount,
                                      const uint32_t* indices, size_t indexCount,
                                      bool sparse = false);

    // Welds NON-indexed triangle soup into a canonical index buffer so
    // generateChain has topology to collapse. Soup (three unshared vertices
    // per triangle — what FBX-style loaders that never call setIndex
    // produce) has no shared edges, so the quadric simplifier sees every
    // triangle as a bordered island and refuses to reduce anything; welding
    // reconnects the mesh WITHOUT touching the vertex data.
    //
    // Vertices are grouped by binary equality over ALL the streams passed
    // (positions required, 12B stride; normals/uvs optional, 12B/8B — via
    // meshopt_generateVertexRemapMulti), and each group elects its FIRST-
    // occurring original vertex id as representative. The returned buffer
    // has one entry per input vertex (it IS the soup's implicit triangle
    // list, 3 entries per triangle, truncated to whole triangles) whose
    // VALUES are those representative ORIGINAL vertex ids — so it can drive
    // an indexed draw or BLAS against the caller's UNCHANGED soup vertex
    // buffer, and feed straight into generateChain as `indices`.
    //
    // Only identical-attribute duplicates weld: smooth-surface soup
    // reconnects into real shared edges, while genuine seams and hard edges
    // (position matches but normal/uv differs) stay split — those remain
    // topological borders, which generateChain's meshopt_SimplifyLockBorder
    // then protects, exactly like an indexed mesh with the same seams.
    //
    // Returns empty on degenerate input (< 1 whole triangle, null positions).
    std::vector<uint32_t> buildCanonicalIndices(const float* positions,
                                                 const float* normals,
                                                 const float* uvs,
                                                 size_t vertexCount);

}// namespace threepp::geometrylod

#endif//THREEPP_GEOMETRYLOD_HPP
