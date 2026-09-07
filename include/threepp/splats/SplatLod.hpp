// Dynamic level-of-detail for a multi-level splat cloud — the TABLE (where each
// level and each of its chunks sits inside one resident SplatCloud) and the
// per-frame POLICY (which level, which chunks) that turns a camera into the
// range list SplatCloud::setSubmitRanges consumes.
//
// One resident cloud with submission ranges, not N clouds and not a re-packed
// buffer, because both alternatives were measured and rejected: a second
// SplatCloud costs ~1.3 ms flat (the Vulkan pass runs end to end per cloud) and
// re-packing on selection change re-uploads up to 1.2 GB. doc/vulkan_splats.md
// carries the numbers.
//
// The policy in one sentence: pick the coarsest level whose splat count still
// covers the cloud's projected footprint at about one splat per pixel, then
// submit only the chunks of that level whose bounds survive the frustum. The
// footprint rule makes "close up" saturate to the finest level by construction
// — no tuned distance constant — and the frustum test is all the chunk culling
// there is, an if-statement per chunk.
//
// Levels are ALTERNATIVES sharing no chunk partition (a real scan carries
// 9/5/3/2 chunks at levels 0-3), so the policy picks ONE level for the whole
// cloud. Per-chunk level MIXING needs a cross-level correspondence the SOG
// format does not provide; when a format does, it slots in here.

#ifndef THREEPP_SPLATS_SPLATLOD_HPP
#define THREEPP_SPLATS_SPLATLOD_HPP

#include "threepp/math/Box3.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/splats/SplatData.hpp"

#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace threepp {
    class Camera;
    class SplatCloud;
}

namespace threepp::splats {

    // Where one loaded detail level sits inside the resident cloud. `base` is
    // the level's first splat; chunk offsets are ABSOLUTE indices into the
    // cloud (base already folded in), which is what setSubmitRanges wants.
    struct LodChunk {
        std::size_t offset = 0;
        std::size_t count = 0;
        Box3 bound;// cloud-local
    };
    struct LodLevel {
        int lod = 0;         // the asset's own level number
        std::size_t base = 0;
        std::size_t count = 0;
        std::vector<LodChunk> chunks;
    };

    struct LodTable {
        // Finest first, matching load order. Empty = no dynamic LOD.
        std::vector<LodLevel> levels;
        // Selection hysteresis state: the level currently held. Lives in the
        // table so a caller holding one table per cloud gets per-cloud
        // hysteresis without a side map.
        int heldLevel = 0;
        // The cloud's PERCENTILE footprint (median centre, p90 radius,
        // cloud-local), computed at load. The policy's projected-area estimate
        // must use this and NEVER the chunk bounds: photogrammetry carries
        // outlier splats far outside the subject (a 167-unit stray in a
        // 20-unit scene is on record), the octree's chunk bounds contain them,
        // and a bounds-derived sphere inflated the footprint enough to pin the
        // policy at level 0 in every framing — measured at 5x the frame time
        // before this field existed.
        Vector3 center;
        float radius = 1.f;

        [[nodiscard]] bool empty() const { return levels.empty(); }
    };

    // One frame's decision. `viewportHeightPx` at the RENDER resolution the
    // cloud is judged against; targetSplatsPerPixel ~1 is the measured sweet
    // spot (levels differ by 2x, so precision past that buys nothing).
    //
    // Writes the chosen ranges INTO the cloud (setSubmitRanges) and returns the
    // level index it settled on. A caller that wants to decide differently can
    // read the table and call setSubmitRanges itself; this is the policy, not
    // the only policy.
    int selectLod(SplatCloud& cloud, LodTable& table, const Camera& camera,
                  int viewportHeightPx, float targetSplatsPerPixel = 1.f,
                  float hysteresis = 1.25f);

    // Load a multi-level SOG asset for dynamic LOD: EVERY OTHER level, finest
    // first, concatenated into one SplatData, with the table describing where
    // each level and chunk landed. Every other because resident memory is the
    // sum of the levels and it is paid twice (GL data textures + the Vulkan
    // pass's buffers): all four levels of a 5M-splat scan killed a 12 GB card,
    // adjacent levels differ by only 2x anyway, and skipping odd levels keeps
    // level 0 — the one the close-up invariant needs. A single-level asset
    // comes back with an EMPTY table: render it plainly, there is nothing to
    // select between.
    //
    // No decoding beyond the levels loaded: the chunk table is a prefix sum
    // over describe()'s counts, which lists chunks in exactly the order load()
    // concatenates them.
    struct SogLodResult {
        SplatData data;
        LodTable table;
    };
    [[nodiscard]] SogLodResult loadSogWithLod(const std::filesystem::path& path);

}// namespace threepp::splats

#endif// THREEPP_SPLATS_SPLATLOD_HPP
