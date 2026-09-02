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
// 9/5/3/2 chunks at levels 0-3), so a policy working from the CHUNK lists can
// only pick ONE level for the whole cloud.
//
// The SSOG tree is the cross-level correspondence that chunk lists are missing:
// each of its leaves names, per level, the range of that level's splats
// belonging to that leaf's piece of space. selectLodPerNode uses it to give
// every leaf its own level, which is what the whole-cloud rule cannot do —
// measured on the calico_tanks scan from inside the canyon, one framing
// submits 4.81 M splats and another 1.07 M for the same rule and the same
// camera, because a dense near view and a sparse one get the same level.

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

    // ── The SSOG tree, one entry per leaf ────────────────────────────────────
    //
    // Where one leaf's splats for ONE resident level landed in the concatenated
    // cloud. `level` indexes LodTable::levels, not the asset's own lod number.
    // A leaf normally has exactly one range per level it carries; the vector
    // allows more so a writer that splits a leaf across two chunk files still
    // round-trips (loadSogWithLod checks and would produce several entries).
    struct LodNodeRange {
        int level = 0;
        std::size_t offset = 0;// absolute index into the cloud
        std::size_t count = 0;
    };

    struct LodNode {
        Box3 bound;    // cloud-local, the tree's own leaf bound
        Vector3 center;// of `bound`
        float radius = 0.f;
        // Ascending level, only the RESIDENT levels this leaf has splats at.
        // A leaf can be absent from a coarse level entirely (135 of
        // calico_tanks' 1210 leaves are absent from level 6): the coarse
        // sampling dropped it, and selecting that level for it would make it
        // disappear. Only levels present here are candidates.
        std::vector<LodNodeRange> ranges;
        // Hysteresis state, per node, -1 until the first selection. Lives here
        // for the reason LodTable::heldLevel does: no side map keyed by
        // anything, so nothing has to be cleaned up and the order it is walked
        // in is the vector's.
        int heldLevel = -1;
        // The level this node was last submitted at, -1 when it was culled.
        // Read by the bench and by anything wanting a level histogram.
        int frameLevel = -1;
    };

    struct LodTable {
        // Finest first, matching load order. Empty = no dynamic LOD.
        std::vector<LodLevel> levels;
        // The tree's leaves. EMPTY when the asset carries no usable tree, in
        // which case selectLod falls back to the whole-cloud rule.
        std::vector<LodNode> nodes;
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

    // At most this many ranges reach the backend (SplatCloud::setSubmitRanges).
    // The per-node path coalesces down to it rather than dropping nodes.
    inline constexpr std::size_t kMaxSubmitRanges = 64;

    // One frame's decision. `viewportHeightPx` at the RENDER resolution the
    // cloud is judged against; targetSplatsPerPixel is splats per SCREEN pixel
    // for the whole visible cloud — ~1 for a subject seen from outside, and
    // the calico demo runs 8 with the camera inside the scan.
    //
    // Writes the chosen ranges INTO the cloud (setSubmitRanges) and returns the
    // level index it settled on. A caller that wants to decide differently can
    // read the table and call setSubmitRanges itself; this is the policy, not
    // the only policy.
    //
    // DISPATCHES: per node when the table carries the tree, whole-cloud when it
    // does not. The two obey the same targetSplatsPerPixel — see
    // selectLodPerNode for why that number keeps its meaning across the switch
    // — so a caller that was passing 8 keeps passing 8 and gets a frame with
    // the same total splat budget spent where it is visible instead of spread
    // flat.
    int selectLod(SplatCloud& cloud, LodTable& table, const Camera& camera,
                  int viewportHeightPx, float targetSplatsPerPixel = 1.f,
                  float hysteresis = 1.25f);

    // The original policy, kept whole: ONE level for the cloud, from the
    // cloud's percentile footprint, then the chosen level's chunks against the
    // frustum. The fallback when there is no tree, and the A/B baseline.
    int selectLodWholeCloud(SplatCloud& cloud, LodTable& table, const Camera& camera,
                            int viewportHeightPx, float targetSplatsPerPixel = 1.f,
                            float hysteresis = 1.25f);

    // Per-node selection. Requires table.nodes; returns 0 and does nothing when
    // the table has no tree.
    //
    // THE RULE, in three parts.
    //
    // 1. A node's projected area is its bounding sphere at the distance from
    //    the camera to its BOX (zero when the camera is inside it), CLAMPED TO
    //    THE SCREEN. The clamp is what makes the near field behave: without it
    //    a node the camera stands in projects to an unbounded area, its density
    //    collapses, and every rule reads it as "hopelessly undersampled". With
    //    it, a screen-filling node of the calico scan measures 0.7 splats per
    //    pixel at level 0 and 0.15 at level 2 — real numbers a threshold can
    //    sit between.
    //
    // 2. The per-node threshold is derived, not tuned:
    //
    //        t = targetSplatsPerPixel * screenArea / sum(area of visible nodes)
    //
    //    Requiring every node to carry t splats per pixel of ITS OWN footprint
    //    spends, in total, about targetSplatsPerPixel splats per pixel of the
    //    SCREEN — which is exactly what the whole-cloud rule's number means.
    //    So the same argument value survives the switch, the total submitted
    //    count lands near targetSplatsPerPixel * screenArea by construction,
    //    and the budget goes to the nodes that occupy screen rather than being
    //    shared equally by 1210 leaves most of which are specks. The divisor is
    //    the frame's OVERDRAW factor (18-35 on the calico scan): a canyon wall
    //    seen from a metre away has thirty nodes stacked over every pixel, and
    //    the rule notices.
    //
    // 3. Coarsest level meeting t wins, per node, with per-node hysteresis; a
    //    node no level satisfies gets the finest it has, which is the close-up
    //    invariant kept per node instead of per cloud.
    //
    // Emission: one range per selected node, sorted by offset, adjacent ranges
    // merged. Above kMaxSubmitRanges the neighbouring pairs separated by the
    // SMALLEST gaps are bridged — the union is drawn at the finer of the two
    // levels, since a gap inside one level's block is splats of that same
    // level — until 64 remain. A visible node is never dropped; bridging only
    // ever ADDS splats, and it costs 26 k of them on the calico back view.
    //
    // Returns the finest level index in use this frame (and leaves it in
    // table.heldLevel), so an existing caller printing "the level" still sees
    // a number that moves the same way.
    int selectLodPerNode(SplatCloud& cloud, LodTable& table, const Camera& camera,
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
    //
    // The SSOG tree comes along when the asset has one (LodTable::nodes), each
    // leaf's per-level ranges resolved to absolute indices by the same prefix
    // sum. The leaves of one chunk are CHECKED to tile it exactly — sorted by
    // offset they must run 0..count with no gap and no overlap — because a
    // tree that only nearly tiles would render a subset and look merely thin.
    struct SogLodResult {
        SplatData data;
        LodTable table;
    };
    [[nodiscard]] SogLodResult loadSogWithLod(const std::filesystem::path& path);

}// namespace threepp::splats

#endif// THREEPP_SPLATS_SPLATLOD_HPP
