// A network of road ribbons draped over one terrain (CPU, renderer-agnostic).
//
// GeoTerrainPack hands us N road polylines; RoadNetwork owns one
// road::RoadGenerator per polyline (for the ribbon geometry + baked asphalt
// texture) and composes their corridor flattenings into the SINGLE ground-height
// function that both the terrain tiles and any object placement query against.
//
// Two responsibilities, kept apart on purpose:
//
//   • buildMeshes()  — one Mesh per road (buildSurface() + bakeSurfaceTexture()
//                      as an sRGB DataTexture on a MeshStandardMaterial). On the
//                      Vulkan deferred renderer ONLY MeshStandardMaterial is lit,
//                      so the ribbons must not use Basic/Phong.
//
//   • groundHeight() / corridorWeight() — the FLATTENING query. These run from
//     TileTerrain's async bake WORKER THREADS, so they must be pure and
//     thread-safe. road::RoadGenerator's own const queries keep a `mutable`
//     scratch buffer (fine single-threaded, a data race across bake threads), so
//     RoadNetwork does NOT call them on the hot path. Instead conformTo() snap-
//     shots every road's conformed centerline into an IMMUTABLE segment soup with
//     a coarse uniform XZ grid; the queries touch only that (stack-local scratch,
//     no shared mutation). A query at (x,z) scans just the segments whose inflated
//     corridor footprint covers its grid cell — O(nearby roads), not O(all).
//
// Order of operations (mirrors the Drive demo): build the network, conformTo()
// the RAW terrain grid height, THEN assemble the provider whose height() folds
// groundHeight() over the same grid. conformTo() must run before groundHeight()/
// corridorWeight()/buildMeshes() — before it the segment heights are unset.
//
// Header-only, threepp core + road::RoadGenerator only.

#ifndef THREEPP_EXTRAS_ROAD_ROADNETWORK_HPP
#define THREEPP_EXTRAS_ROAD_ROADNETWORK_HPP

#include "threepp/math/MathUtils.hpp"
#include "threepp/extras/road/RoadGenerator.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <array>
#include <vector>

namespace threepp::road {

    // One road as fed to the network: a centerline plus its physical width and a
    // category letter that scales the verge/tessellation. Mirrors
    // terrain::GeoRoad but keeps this header independent of the pack loader.
    struct RoadSpec {
        std::string id;
        std::string category;// "E" | "R" | "F" | "K" | "P" | "S"
        float width = 6.f;   // total carriageway width (m)
        // Centerline (world XZ). Y: ignored by the legacy drape (conformTo sets
        // it from the ground), but when RoadProfileOptions.enabled the point Y
        // is the PACK's true road elevation and drives the bridge/tunnel
        // classification (a road whose data height spans a ravine/fjord must
        // NOT be draped down into the vertical U the ground makes there).
        std::vector<Vector3> points;
    };

    // Which N302 marking class a road falls into. Derived from its category and
    // its asphalt width (see MarkingRules); NVDB's own vegoppmerking data would
    // override this per road once it is fetched.
    enum class MarkingClass {
        Main,    // >= 6 m: yellow centre line, solid white edges
        Narrow,  // 5-6 m: no centre, DASHED white edges
        Unmarked,// < 5 m: no paint at all
        Track    // category S: gravel forest road
    };

    // The knobs behind that derivation. Defaults are the N302 numbers; a demo
    // can move centreMin to A/B a road it believes is mis-classified (Fv63's
    // hairpins carry the 7 m category default but are narrower in reality).
    struct MarkingRules {
        float centreMin = 6.0f;// asphalt width at/above which a centre line exists
        float edgeMin = 5.0f;  // ...at/above which edge lines exist at all
        float wideMin = 8.5f;  // ...at/above which the lines are 0.15 m not 0.10
        float lineWidth = 0.10f, lineWidthWide = 0.15f;
        float centreOn = 3.f, centreOff = 9.f;// skillelinje 3+9
        float edgeOn = 3.f, edgeOff = 3.f;    // stiplet kantlinje 3+3
        float shoulderInset = 0.25f;          // asphalt outside the edge line
        float shoulderInsetMain = 0.5f;       // ...on a wide road
        std::array<float, 3> whiteColor = {0.86f, 0.86f, 0.84f};
        std::array<float, 3> yellowColor = {0.88f, 0.70f, 0.12f};
        float wear = 0.6f;        // 0 = fresh paint on new asphalt, 1 = ruined
        bool gravelEdge = true;   // narrow grusskulder at the pavement edge
        unsigned int seed = 1337u;// variant seed
        int texWidth = 256, texHeight = 2048;
        float tileLength = 96.f;// 8 x (3+9) and 16 x (3+3): both patterns tile exactly
                                // 96 m at the same ~4.7 cm texel as the old 48/1024,
                                // so the tile's own repeat is 96 m of road (192 m with
                                // the two seeded variants, 384 m with mirroring) at
                                // 2 MB per map instead of 1.
        int patchRes = 256;     // repair-patch atlas: 4 variants of this square
        unsigned int patchSeed = 4711u;
        float patchesPer100m = 1.f;// upper bound; the per-piece hash picks 0..this (one patch every ~200 m)
    };

    // Elevation-profile handling for conformTo (all opt-in; default = the
    // legacy pure drape, byte-identical behaviour for existing callers).
    //
    // The rules, in plain terms:
    //   • a road whose DATA height runs well ABOVE the ground (or whose ground
    //     is water) is a BRIDGE there — the deck follows the data height and
    //     spans the vertical U instead of draping down into it;
    //   • a road whose data height runs well BELOW the ground is a TUNNEL —
    //     excluded outright (the road ends at the portal and reappears on the
    //     far side; there is nothing to render in between);
    //   • a road is never submerged: over water the deck is clamped to at
    //     least seaLevel + deckMin. A long water run whose data height never
    //     clears the water (a ferry leg, or garbage elevations) is excluded —
    //     rendering it would draw a road lying ON the sea.
    struct RoadProfileOptions {
        bool  enabled = false;  // false = legacy drape (flags stay clear)
        float seaLevel = 0.f;   // the pack's water-sheet elevation (m)
        float waterEps = 0.05f; // ground <= seaLevel+eps counts as water
        float bridgeThresh = 3.f;// packY - ground above this => bridge span
        float tunnelThresh = 3.f;// ground - packY above this => tunnel (excluded)
        float deckMin = 1.2f;    // minimum deck height above the water (m)
        float ferryMinLen = 60.f;// water run longer than this with packY never
                                 // clearing deckMin => ferry/garbage => excluded
        float minSpanLen = 30.f; // bridge runs shorter than this demote to ground
                                 // (culvert/causeway — carve fills it like a real
                                 // roadbed; a stub deck ribbon would be exactly
                                 // the floating-shard artifact class) — UNLESS the
                                 // run clears shortSpanMaxLift (a deep ravine
                                 // crossing is a real short viaduct, not a culvert)
        float shortSpanMaxLift = 6.f;// a short bridge run whose deck rises more than
                                 // this above ground is a genuine span (a narrow
                                 // gorge/stream ravine crossed by a real bridge) and
                                 // is KEPT even under minSpanLen — draping it to the
                                 // ravine floor makes a hard vertical-U dive instead
        float maxDataLift = 60.f;// packY more than this above ground = garbage
                                 // elevation data, NOT a viaduct — treat as ground
        float deckAnchorLift = 0.6f;// after classification, each bridge span EXTENDS
                                 // outward while dataY still rides this far above the
                                 // ground, so the deck anchors on solid ground. The
                                 // lift>bridgeThresh test alone starts the span only
                                 // once the deck is 3 m up — the approaches (lift
                                 // 0..3 m) would stay ground-draped, and near a gorge
                                 // that ground is already collapsing (rim steps,
                                 // lips), so the road dives into every sub-3 m fold
                                 // beside the deck and the crossing reads as a warped
                                 // hump instead of a taut span.
        float deckAnchorMax = 30.f;// extension bound per side (m) — a long shallow
                                 // drape is an embankment for the carve, not a deck
        float gradeSmoothing = 0.f;// ARC-LENGTH grade-smoothing window (m); 0 = off.
                                 // A real road follows a SMOOTH vertical alignment,
                                 // not the DEM's 5-30 m micro-undulations that are
                                 // invisible at a crawl but BOUNCE a car at speed.
                                 // The per-sample 1-2-1 smoothing can't remove them
                                 // (its reach scales with sample density, which is
                                 // sub-metre here); this averages each GROUND sample's
                                 // grade over a fixed metric window (bridge decks and
                                 // excluded spans keep their own heights). ~25 m halves
                                 // the mid-band roughness while the grade moves <~0.2 m.
    };

    class RoadNetwork {

    public:
        // The ribbon mesh's paved verts sit this far above the conformed grade
        // (RoadParams::surfaceRaise). Shared so surfaceHeight() reports the same
        // elevation the mesh is built at.
        static constexpr float kSurfaceRaise = 0.12f;

        // flattenMargin: metres of graded shoulder-out over which the terrain
        // relaxes from the road elevation back to natural ground (beyond the
        // paved+verge corridor). Wider than the ribbon so the road never floats.
        // trenchDepth: metres the terrain UNDER the pavement is dropped below the
        // conformed road grade, feathering back to 0 across the shoulder. This
        // gives coarse-LOD terrain interpolation a margin (trenchDepth+surfaceRaise)
        // to err within before it shears up through the ribbon. The ribbon mesh
        // itself stays at the un-trenched conformed height — only terrain drops.
        explicit RoadNetwork(std::vector<RoadSpec> specs, float flattenMargin = 8.f,
                             float trenchDepth = 0.35f)
            : flattenMargin_(std::max(flattenMargin, 0.5f)),
              trenchDepth_(std::max(trenchDepth, 0.f)) {
            roads_.reserve(specs.size());
            for (auto& s : specs) {
                if (s.points.size() < 2) continue;
                Road road;
                road.spec = std::move(s);
                road.gen = std::make_unique<RoadGenerator>(road.spec.points, paramsFor(road.spec));
                road.pavedHalf = road.gen->pavedHalfWidth();
                road.corridorHalf = road.gen->corridorHalfWidth();
                roads_.push_back(std::move(road));
            }
            buildXZGeometry();
            buildGrid();
        }

        [[nodiscard]] size_t roadCount() const { return roads_.size(); }

        // ── N302 marking classes and the shared surface materials ────────────
        // Changing the rules invalidates the baked material cache (the class, the
        // line widths and the wear level are all baked INTO the texture).
        void setMarkingRules(const MarkingRules& mr) {
            rules_ = mr;
            surfaceCache_.clear();
            patchMat_.reset();
            meanCached_[0] = meanCached_[1] = false;
        }
        [[nodiscard]] const MarkingRules& markingRules() const { return rules_; }

        // Marking class of a road: category S is always a forest track, otherwise
        // the ASPHALT WIDTH decides — that is exactly how N302 reads.
        [[nodiscard]] MarkingClass classOf(const RoadSpec& s) const {
            const char c = s.category.empty() ? '?' : s.category[0];
            if (c == 'S') return MarkingClass::Track;
            if (s.width >= rules_.centreMin) return MarkingClass::Main;
            if (s.width >= rules_.edgeMin) return MarkingClass::Narrow;
            return MarkingClass::Unmarked;
        }

        // How many distinct baked surface sets the network has handed out — the
        // whole point of the cache (one bake per class/width, not per chunk).
        [[nodiscard]] size_t surfaceSetCount() const { return surfaceCache_.size(); }

        // Mean sRGB of the paved band for a surface kind, straight out of the
        // bake. The far road is painted into the terrain albedo with a single
        // colour; taking it from here is what keeps the 600 m ribbon-to-paint
        // hand-off from stepping.
        [[nodiscard]] std::array<float, 3> meanSurfaceColor(SurfaceKind kind) const {
            const int k = (kind == SurfaceKind::Gravel) ? 1 : 0;
            if (!meanCached_[k]) {
                RoadSurfaceStyle st;
                st.kind = kind;
                st.centre = LinePattern::None;// the paint is the SURFACE, not the lines
                st.edge = LinePattern::None;
                st.wear = rules_.wear;
                st.seed = rules_.seed;
                st.tileLength = rules_.tileLength;
                st.pavedWidth = 6.f;
                st.fullWidth = 7.f;
                meanValue_[k] = RoadGenerator::bakeSurfaceMaps(st, 64, 128).meanPaved;
                meanCached_[k] = true;
            }
            return meanValue_[k];
        }

        // Per-road summary for a demo's --list-roads / view picking.
        struct RoadInfo {
            std::string id;
            std::string category;
            float width = 0.f;
            float length = 0.f;
            MarkingClass cls = MarkingClass::Unmarked;
            Vector3 first;
        };
        [[nodiscard]] std::vector<RoadInfo> roadInfos() const {
            std::vector<RoadInfo> out;
            out.reserve(roads_.size());
            for (const auto& r : roads_) {
                const auto& cl = r.gen->centerlineSamples();
                RoadInfo i;
                i.id = r.spec.id;
                i.category = r.spec.category;
                i.width = r.spec.width;
                i.cls = classOf(r.spec);
                for (size_t k = 1; k < cl.size(); ++k) i.length += cl[k].distanceTo(cl[k - 1]);
                if (!cl.empty()) i.first = cl.front();
                out.push_back(std::move(i));
            }
            return out;
        }

        // Conformed centreline of one road by id (empty if unknown).
        [[nodiscard]] std::vector<Vector3> roadCenterline(const std::string& id) const {
            for (const auto& r : roads_)
                if (r.spec.id == id) return r.gen->centerlineSamples();
            return {};
        }

        void setTrenchDepth(float d) { trenchDepth_ = std::max(d, 0.f); }
        [[nodiscard]] float trenchDepth() const { return trenchDepth_; }

        // Drape every road onto the ground function (typically the raw terrain
        // grid height), then snapshot conformed centerlines into the immutable
        // segment soup the thread-safe queries use. Call once before the queries.
        //
        // With profile.enabled, the drape becomes elevation-aware (see
        // RoadProfileOptions): where the spec's data height says the road spans
        // a ravine or water, the generator drapes onto max(ground, data height)
        // — the 1-2-1 grade smoothing then rounds the deck into its approaches —
        // and every dense segment is classified bridge / tunnel-excluded /
        // ferry-excluded, which the terrain queries, carveRoads and the mesh
        // builders consume via the Seg flags.
        void conformTo(const std::function<float(float, float)>& groundFn, int smoothingPasses = 14,
                       const RoadProfileOptions& profile = {}) {
            if (!profile.enabled) {
                for (auto& r : roads_) r.gen->conformTo(groundFn, smoothingPasses);
                snapshotHeights();
                conformed_ = true;
                return;
            }
            // Classify FIRST (on the raw ground + the spec's data heights), then
            // conform each road to explicit per-sample target elevations — so a
            // span the classification demotes (blip, garbage data) is draped at
            // GROUND, never left carrying a lifted deck height it no longer owns.
            for (auto& r : roads_) conformRoadProfiled(r, groundFn, smoothingPasses, profile);
            snapshotHeights();
            foldSegFlags();
            conformed_ = true;
        }

        // Unified ground height: terrain flattened into every nearby road
        // corridor (paved+verge held at the road grade, then graded back to
        // `terrainH` over flattenMargin), MINUS a trench under the pavement that
        // feathers to 0 at the shoulder edge (see trenchDepth). Thread-safe /
        // read-only after conformTo(). O(segments in the local grid cell).
        [[nodiscard]] float groundHeight(float terrainH, float x, float z) const {
            const float outer = flattenMargin_;
            float bestW = 0.f;
            float bestH = terrainH;
            float bestTrench = 0.f;// trench profile [0..1] of the winning segment
            forEachNearbySegment(x, z, [&](const Seg& s) {
                if (s.flags) return;// bridge deck spans / excluded roads don't shape the ground
                float t;
                const float d = distToSegment(x, z, s, t);
                const float corridorHalf = s.corridorHalf;
                if (d >= corridorHalf + outer) return;
                const float w = 1.f - math::smoothstep(corridorHalf, corridorHalf + outer, d);
                if (w > bestW) {
                    bestW = w;
                    bestH = s.ha + (s.hb - s.ha) * t;
                    // Trench profile: full depth under the paved band, smoothstep
                    // back to 0 at the corridor (shoulder-outer) edge so terrain
                    // meets the ribbon shoulder flush. C1-smooth so coarse tiles
                    // don't alias the drop into a crease.
                    bestTrench = (d <= s.pavedHalf)
                                         ? 1.f
                                         : math::smoothstep(s.corridorHalf, s.pavedHalf, d);
                }
            });
            const float flattened = terrainH + (bestH - terrainH) * bestW;
            return flattened - trenchDepth_ * bestTrench;
        }

        // Paved SURFACE elevation at (x,z): the nearest road's conformed
        // centerline height + the ribbon's surfaceRaise — i.e. the height the
        // ribbon MESH sits at, NOT trenched. Use this (not groundHeight) to
        // cross-check conformance or to place objects ON the road. Returns
        // `fallback` (NaN by default) when no road is near.
        [[nodiscard]] float surfaceHeight(float x, float z,
                                          float fallback = std::numeric_limits<float>::quiet_NaN()) const {
            float bestD = std::numeric_limits<float>::max();
            float bestH = fallback;
            forEachNearbySegment(x, z, [&](const Seg& s) {
                if (s.flags & kSegExcluded) return;// tunnels/ferries have no surface
                float t;
                const float d = distToSegment(x, z, s, t);
                if (d < bestD) {
                    bestD = d;
                    bestH = s.ha + (s.hb - s.ha) * t + kSurfaceRaise;
                }
            });
            return bestH;
        }

        // Does the axis-aligned tile [cx±half, cz±half] overlap any road corridor
        // (paved+verge inflated by the flatten margin)? Cheap grid-bounded AABB
        // test — used by demos to bias terrain LOD refinement toward road tiles.
        // Thread-safe / read-only after construction.
        [[nodiscard]] bool corridorIntersects(float cx, float cz, float half) const {
            if (grid_.empty()) return false;
            const float tMinX = cx - half, tMaxX = cx + half;
            const float tMinZ = cz - half, tMaxZ = cz + half;
            const int gx0 = cellOf(tMinX), gx1 = cellOf(tMaxX);
            const int gz0 = cellOf(tMinZ), gz1 = cellOf(tMaxZ);
            for (int gz = gz0; gz <= gz1; ++gz)
                for (int gx = gx0; gx <= gx1; ++gx) {
                    const auto it = grid_.find(packKey(gx, gz));
                    if (it == grid_.end()) continue;
                    for (int idx : it->second) {
                        const Seg& s = segs_[static_cast<size_t>(idx)];
                        const float reach = s.corridorHalf + flattenMargin_;
                        const float sMinX = std::min(s.ax, s.bx) - reach;
                        const float sMaxX = std::max(s.ax, s.bx) + reach;
                        const float sMinZ = std::min(s.az, s.bz) - reach;
                        const float sMaxZ = std::max(s.az, s.bz) + reach;
                        if (sMaxX >= tMinX && sMinX <= tMaxX && sMaxZ >= tMinZ && sMinZ <= tMaxZ)
                            return true;
                    }
                }
            return false;
        }

        // 1.0 on paved road, smoothstep to 0 across the verge, 0 beyond — the
        // MAX over nearby roads. Fades detail relief / tints the roadside albedo.
        // Thread-safe / read-only after conformTo().
        [[nodiscard]] float corridorWeight(float x, float z) const {
            float best = 0.f;
            forEachNearbySegment(x, z, [&](const Seg& s) {
                if (s.flags) return;// no corridor under bridge decks / excluded roads
                float t;
                const float d = distToSegment(x, z, s, t);
                float w;
                if (d <= s.pavedHalf) w = 1.f;
                else if (d >= s.corridorHalf) w = 0.f;
                else w = math::smoothstep(s.corridorHalf, s.pavedHalf, d);
                best = std::max(best, w);
            });
            return best;
        }

        // 1.0 on the PAVED band, feathered to 0 over `edgeFeather` metres just
        // beyond the pavement edge, 0 elsewhere — the albedo-paint query for
        // baked roads (terrain::makeGeoProvider paintRoads). Narrower than
        // corridorWeight on purpose: painting the whole shoulder corridor reads
        // as a phantom second road wherever roads run close (hairpins, dual
        // carriageways). Bridge/excluded segments never paint — there is no
        // road ON THE GROUND there. Thread-safe / read-only after conformTo().
        [[nodiscard]] float pavedWeight(float x, float z, float edgeFeather = 0.8f) const {
            float best = 0.f;
            forEachNearbySegment(x, z, [&](const Seg& s) {
                if (s.flags) return;
                float t;
                const float d = distToSegment(x, z, s, t);
                float w;
                if (d <= s.pavedHalf) w = 1.f;
                else if (d >= s.pavedHalf + edgeFeather) w = 0.f;
                else w = math::smoothstep(s.pavedHalf + edgeFeather, s.pavedHalf, d);
                best = std::max(best, w);
            });
            return best;
        }

        // One Mesh per road: paved ribbon + graded shoulders with a baked sRGB
        // asphalt/markings texture on a MeshStandardMaterial (Vulkan-deferred
        // safe). Call after conformTo().
        [[nodiscard]] std::shared_ptr<Group> buildMeshes(int texWidth = 128, int texHeight = 256) const {
            auto group = Group::create();
            group->name = "road_network";
            for (const auto& r : roads_) {
                auto geo = r.gen->buildSurface();
                if (!geo->getAttribute<float>("position") ||
                    geo->getAttribute<float>("position")->count() == 0)
                    continue;

                auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                                .color(Color::white)
                                                                .roughness(0.95f)
                                                                .metalness(0.f));
                auto tex = DataTexture::create(
                        ImageData{r.gen->bakeSurfaceTexture(texWidth, texHeight)},
                        static_cast<unsigned int>(texWidth), static_cast<unsigned int>(texHeight));
                tex->colorSpace = ColorSpace::sRGB;
                tex->magFilter = Filter::Linear;
                tex->minFilter = Filter::LinearMipmapLinear;
                tex->generateMipmaps = true;// DataTexture defaults false → GL black
                // A road ribbon is viewed at PERMANENT grazing incidence — without
                // aniso, trilinear collapses to the along-road mip axis and the
                // lane dashes smear into giant blobs a few car-lengths out (the
                // Vulkan material sampler forces 16× on unjittered paths, which is
                // why only GL showed it — it honors this per-texture value).
                tex->anisotropy = 16;
                tex->wrapS = TextureWrapping::ClampToEdge;
                tex->wrapT = TextureWrapping::Repeat;// tiles along the road length
                mat->map = tex;

                auto mesh = Mesh::create(geo, mat);
                mesh->name = "road_" + (r.spec.id.empty() ? std::string("?") : r.spec.id);
                mesh->receiveShadow = true;
                // The ribbon IS its own thin strip — the renderer's auto-LOD
                // chain simplifies it into flickering slivers, so pin it off.
                mesh->autoLod = false;
                group->add(mesh);
            }
            return group;
        }

        // Ribbon meshes for BRIDGE SPANS only (profile mode — see
        // RoadProfileOptions). Every maximal run of bridge-classified centerline
        // samples becomes one deck ribbon, extended a couple of samples onto
        // ground at each end so the deck meets the baked road flush. Roads fully
        // on ground produce NOTHING here: with carveRoads(bakeSurface) + the
        // provider's albedo paint the terrain itself is the road — mip-filtered
        // texture instead of sub-pixel ribbon geometry, which is what kills the
        // distant road shimmer (a 1 px jittered ribbon flips raster coverage
        // every frame; a baked texel does not). Call after conformTo() with
        // profile.enabled; returns an empty group in legacy mode.
        [[nodiscard]] std::shared_ptr<Group> buildBridgeMeshes(int texWidth = 128, int texHeight = 256) const {
            auto group = Group::create();
            group->name = "road_bridges";
            for (const auto& r : roads_) {
                const auto& cl = r.gen->centerlineSamples();
                const auto& f = r.sampleFlags;
                if (f.size() != cl.size()) continue;// legacy mode — no classification ran
                const std::vector<float> cum = stationsAlong(cl);
                size_t i = 0;
                int span = 0;
                while (i < cl.size()) {
                    if (f[i] != kSegBridge) { ++i; continue; }
                    size_t j = i;
                    while (j < cl.size() && f[j] == kSegBridge) ++j;
                    // Two-sample abutment overlap onto ground at both ends.
                    const size_t lo = (i >= 2) ? i - 2 : 0;
                    const size_t hi = std::min(cl.size(), j + 2);
                    std::vector<Vector3> deck(cl.begin() + static_cast<std::ptrdiff_t>(lo),
                                              cl.begin() + static_cast<std::ptrdiff_t>(hi));
                    i = j;
                    if (deck.size() < 2) continue;
                    if (auto mesh = buildRibbonPiece(r, deck, "bridge", span, texWidth, texHeight,
                                                     cum[lo])) {
                        group->add(mesh);
                        ++span;
                    }
                }
            }
            return group;
        }

        // Ribbon meshes for the ON-GROUND runs, CHUNKED (~chunkLen metres each)
        // so a caller can distance-cull them per frame: near the camera the
        // ribbon supplies stand-on detail (crisp geometric edges, baked lane
        // markings — everything the painted terrain's ~1 m splat texels cannot
        // hold), and beyond the cull distance it vanishes and the baked+painted
        // roadbed underneath carries the visual, which is mip-filtered texture
        // and therefore cannot coverage-shimmer the way a sub-pixel ribbon
        // does. Pair with carveRoads(bakeSurface): the baked bed sits at the
        // conformed grade, the ribbon kSurfaceRaise above it — no z-fight.
        // Default chunk length is a multiple of RoadParams::textureTileLength
        // (12 m) so the dash phase stays roughly continuous across seams.
        // Bridge/excluded samples never chunk (buildBridgeMeshes owns decks).
        // In legacy (non-profile) mode every sample counts as ground.
        [[nodiscard]] std::shared_ptr<Group> buildGroundChunkMeshes(float chunkLen = 240.f,
                                                                    int texWidth = 128,
                                                                    int texHeight = 256) const {
            auto group = Group::create();
            group->name = "road_chunks";
            for (const auto& r : roads_) {
                const auto& cl = r.gen->centerlineSamples();
                const auto& f = r.sampleFlags;
                const bool flagged = f.size() == cl.size();
                const std::vector<float> cum = stationsAlong(cl);
                int piece = 0;
                size_t i = 0;
                while (i < cl.size()) {
                    if (flagged && f[i] != 0) { ++i; continue; }
                    // Ground run [i, j).
                    size_t j = i;
                    while (j < cl.size() && (!flagged || f[j] == 0)) ++j;
                    // Slice the run into ~chunkLen pieces, sharing the boundary
                    // sample so consecutive chunks meet without a gap.
                    size_t s = i;
                    while (s + 1 < j) {
                        size_t e = s + 1;
                        float len = 0.f;
                        while (e + 1 < j && len < chunkLen) {
                            const float dx = cl[e].x - cl[e - 1].x, dz = cl[e].z - cl[e - 1].z;
                            len += std::sqrt(dx * dx + dz * dz);
                            ++e;
                        }
                        std::vector<Vector3> pts(cl.begin() + static_cast<std::ptrdiff_t>(s),
                                                 cl.begin() + static_cast<std::ptrdiff_t>(e + 1));
                        if (auto mesh = buildRibbonPiece(r, pts, "roadchunk", piece, texWidth,
                                                         texHeight, cum[s], /*patchDecals*/ true)) {
                            group->add(mesh);
                            ++piece;
                        }
                        s = e;
                    }
                    i = j;
                }
            }
            return group;
        }

        // Longest road by summed segment length (with its conformed midpoint +
        // heading) — used by demos to place a fly-over camera. Returns false if
        // the network is empty.
        [[nodiscard]] bool longestRoad(Vector3& midWorld, Vector3& headingXZ) const {
            int best = -1;
            float bestLen = -1.f;
            for (size_t i = 0; i < roads_.size(); ++i) {
                const auto& cl = roads_[i].gen->centerlineSamples();
                if (cl.size() < 2) continue;
                float len = 0.f;
                for (size_t k = 1; k < cl.size(); ++k) len += cl[k].distanceTo(cl[k - 1]);
                if (len > bestLen) {
                    bestLen = len;
                    best = static_cast<int>(i);
                }
            }
            if (best < 0) return false;
            const auto& cl = roads_[static_cast<size_t>(best)].gen->centerlineSamples();
            const Vector3 a = cl.front();
            const Vector3 b = cl.back();
            midWorld = cl[cl.size() / 2];
            headingXZ.set(b.x - a.x, 0.f, b.z - a.z);
            if (headingXZ.length() < 1e-4f) headingXZ.set(0.f, 0.f, 1.f);
            headingXZ.normalize();
            return true;
        }

        // Visit every conformed centerline segment in the network:
        //   f(ax, az, ha, bx, bz, hb, pavedHalf, corridorHalf)
        // with (ax,az)→(bx,bz) the XZ endpoints, ha/hb the conformed centerline
        // heights, and pavedHalf/corridorHalf the road's half pavement / half
        // pavement+shoulder widths. Only meaningful after conformTo(). Read-only
        // → thread-safe. Used by terrain::carveRoads (DEM road cut) and by
        // physics demos to build collision aprons along the ribbon edges.
        template<class F>
        void forEachSegment(F&& f) const {
            for (const auto& s : segs_)
                f(s.ax, s.az, s.ha, s.bx, s.bz, s.hb, s.pavedHalf, s.corridorHalf);
        }

        // Same visit with the profile-classification flags appended:
        //   f(ax, az, ha, bx, bz, hb, pavedHalf, corridorHalf, flags)
        // flags: kSegBridge / kSegExcluded, always 0 in legacy (non-profile)
        // mode. carveRoads uses this so bridge decks never carve the ground
        // beneath them and excluded roads never carve at all.
        template<class F>
        void forEachSegmentFlagged(F&& f) const {
            for (const auto& s : segs_)
                f(s.ax, s.az, s.ha, s.bx, s.bz, s.hb, s.pavedHalf, s.corridorHalf, s.flags);
        }

        // Dense conformed centerline polyline (world coords, y = conformed grade)
        // of the longest road — for a demo to auto-steer a car ALONG the road
        // (pure-pursuit). Ordered start→end so a progress index can follow it
        // through stacked hairpins without snapping to a switchback overhead.
        // Empty if the network has no road.
        [[nodiscard]] std::vector<Vector3> longestRoadCenterline() const {
            int best = -1;
            float bestLen = -1.f;
            for (size_t i = 0; i < roads_.size(); ++i) {
                const auto& cl = roads_[i].gen->centerlineSamples();
                if (cl.size() < 2) continue;
                float len = 0.f;
                for (size_t k = 1; k < cl.size(); ++k) len += cl[k].distanceTo(cl[k - 1]);
                if (len > bestLen) {
                    bestLen = len;
                    best = static_cast<int>(i);
                }
            }
            if (best < 0) return {};
            return roads_[static_cast<size_t>(best)].gen->centerlineSamples();
        }

        // The longest CONTIGUOUS run of DRIVABLE (non-excluded) conformed
        // centerline samples across all roads — ordered along the road. A driving
        // demo spawns on and follows this so it never lands on a ferry/tunnel span
        // that profile classification excluded (there is no collider or surface
        // there — the car would fall through). In legacy (non-profile) mode every
        // sample is drivable, so this is simply the longest road's centerline
        // (same as longestRoadCenterline). Empty if the network has no drivable
        // road. Call after conformTo().
        [[nodiscard]] std::vector<Vector3> longestDrivableRun() const {
            std::vector<Vector3> best;
            float bestLen = -1.f;
            for (const auto& r : roads_) {
                const auto& cl = r.gen->centerlineSamples();
                const auto& f = r.sampleFlags;
                const bool flagged = f.size() == cl.size();
                size_t i = 0;
                while (i < cl.size()) {
                    if (flagged && (f[i] & kSegExcluded)) { ++i; continue; }
                    size_t j = i;// extend the run while the NEXT sample stays drivable
                    float len = 0.f;
                    while (j + 1 < cl.size() && !(flagged && (f[j + 1] & kSegExcluded))) {
                        len += cl[j + 1].distanceTo(cl[j]);
                        ++j;
                    }
                    if (len > bestLen) {
                        bestLen = len;
                        best.assign(cl.begin() + static_cast<std::ptrdiff_t>(i),
                                    cl.begin() + static_cast<std::ptrdiff_t>(j + 1));
                    }
                    i = j + 1;
                }
            }
            return best;
        }

    private:
        // ── per-road ownership ───────────────────────────────────────────────
        struct Road {
            RoadSpec spec;
            std::unique_ptr<RoadGenerator> gen;
            float pavedHalf = 0.f, corridorHalf = 0.f;
            // Per dense-centerline-sample classification (kSegBridge/kSegExcluded),
            // filled by classifySegments in profile mode; empty in legacy mode.
            std::vector<std::uint8_t> sampleFlags;
        };

        // ── immutable segment soup (thread-safe query backing) ───────────────
        struct Seg {
            float ax = 0.f, az = 0.f, bx = 0.f, bz = 0.f;// endpoints (XZ)
            float ha = 0.f, hb = 0.f;                    // conformed heights (set by snapshotHeights)
            float pavedHalf = 0.f, corridorHalf = 0.f;   // per-road widths
            std::uint8_t flags = 0;                      // kBridge / kExcluded (profile mode only)
        };

    public:
        // Seg classification flags (set by conformTo when profile.enabled).
        static constexpr std::uint8_t kSegBridge = 1;  // deck span — no terrain flatten/carve/paint
        static constexpr std::uint8_t kSegExcluded = 2;// tunnel / ferry / garbage — not rendered at all

    private:

        // Map the RoadParams for a spec: laneWidth = width/2 with laneCount 2 so
        // the paved band == width; verge + tessellation scale with category.
        static RoadParams paramsFor(const RoadSpec& s) {
            RoadParams p;
            p.laneCount = 2;
            p.laneWidth = std::max(s.width * 0.5f, 1.5f);
            const char c = s.category.empty() ? '?' : s.category[0];
            switch (c) {
                case 'E': p.shoulderWidth = 3.0f; p.samplesPerSegment = 24; break;// europavei
                case 'R': p.shoulderWidth = 2.5f; p.samplesPerSegment = 22; break;// riksvei
                case 'F': p.shoulderWidth = 2.0f; p.samplesPerSegment = 18; break;// fylkesvei
                case 'K': p.shoulderWidth = 1.2f; p.samplesPerSegment = 14; break;// kommunal
                case 'P': p.shoulderWidth = 0.6f; p.samplesPerSegment = 10; break;// privat
                case 'S': p.shoulderWidth = 0.6f; p.samplesPerSegment = 10; break;// skogsbilvei
                default:  p.shoulderWidth = 1.5f; p.samplesPerSegment = 16; break;
            }
            p.surfaceRaise = kSurfaceRaise;// sit above the flattened corridor (anti z-fight)
            // Keep camber under raise/pavedHalf so banked curves never poke terrain.
            p.maxBanking = 0.03f;
            p.textureTileLength = 12.f;
            return p;
        }

        // Build the XZ segment list from each road's DENSE centerline (the same
        // tessellation the ribbon uses). Heights are filled in by snapshotHeights.
        void buildXZGeometry() {
            segs_.clear();
            for (const auto& r : roads_) {
                const auto& cl = r.gen->centerlineSamples();// XZ + provisional Y
                for (size_t k = 1; k < cl.size(); ++k) {
                    Seg s;
                    s.ax = cl[k - 1].x; s.az = cl[k - 1].z;
                    s.bx = cl[k].x;     s.bz = cl[k].z;
                    s.pavedHalf = r.pavedHalf;
                    s.corridorHalf = r.corridorHalf;
                    segs_.push_back(s);
                }
            }
        }

        // After conformTo, copy the conformed centerline heights onto the
        // segment endpoints (segment ordering matches buildXZGeometry).
        void snapshotHeights() {
            size_t si = 0;
            for (const auto& r : roads_) {
                const auto& cl = r.gen->centerlineSamples();// now carries conformed Y
                for (size_t k = 1; k < cl.size(); ++k, ++si) {
                    if (si >= segs_.size()) return;// defensive (shouldn't happen)
                    segs_[si].ha = cl[k - 1].y;
                    segs_[si].hb = cl[k].y;
                }
            }
        }

        // Coarse uniform grid: each segment is inserted into every cell its
        // corridor+margin-inflated AABB touches, so a single-cell query at (x,z)
        // sees every segment that can possibly influence it.
        void buildGrid() {
            grid_.clear();
            if (segs_.empty()) return;
            // Cell a bit larger than the widest corridor keeps the per-segment
            // cell span small while still bounding query fan-out.
            float maxReach = 8.f;
            for (const auto& s : segs_) maxReach = std::max(maxReach, s.corridorHalf + flattenMargin_);
            cellSize_ = std::max(maxReach * 1.5f, 16.f);

            for (int i = 0; i < static_cast<int>(segs_.size()); ++i) {
                const Seg& s = segs_[static_cast<size_t>(i)];
                const float reach = s.corridorHalf + flattenMargin_;
                const float minX = std::min(s.ax, s.bx) - reach, maxX = std::max(s.ax, s.bx) + reach;
                const float minZ = std::min(s.az, s.bz) - reach, maxZ = std::max(s.az, s.bz) + reach;
                const int cx0 = cellOf(minX), cx1 = cellOf(maxX);
                const int cz0 = cellOf(minZ), cz1 = cellOf(maxZ);
                for (int cz = cz0; cz <= cz1; ++cz)
                    for (int cx = cx0; cx <= cx1; ++cx)
                        grid_[packKey(cx, cz)].push_back(i);
            }
        }

        [[nodiscard]] int cellOf(float v) const {
            return static_cast<int>(std::floor(v / cellSize_));
        }
        static std::int64_t packKey(int cx, int cz) {
            return (static_cast<std::int64_t>(static_cast<std::uint32_t>(cx)) << 32) |
                   static_cast<std::uint32_t>(cz);
        }

        // Visit every segment registered in the grid cell containing (x,z). The
        // functor takes (const Seg&). Read-only → thread-safe.
        template<class F>
        void forEachNearbySegment(float x, float z, F&& f) const {
            if (grid_.empty()) return;
            const auto it = grid_.find(packKey(cellOf(x), cellOf(z)));
            if (it == grid_.end()) return;
            for (int idx : it->second) f(segs_[static_cast<size_t>(idx)]);
        }

        // Cumulative XZ arc length along a centreline — the GLOBAL station each
        // ribbon piece starts at. XZ, not 3D, because that is what
        // RoadGenerator's own arcLength (and therefore the v coordinate) is.
        static std::vector<float> stationsAlong(const std::vector<Vector3>& cl) {
            std::vector<float> cum(cl.size(), 0.f);
            for (size_t k = 1; k < cl.size(); ++k) {
                const float dx = cl[k].x - cl[k - 1].x, dz = cl[k].z - cl[k - 1].z;
                cum[k] = cum[k - 1] + std::sqrt(dx * dx + dz * dz);
            }
            return cum;
        }

        // FNV-1a over a road id: picks the surface of the roads that have no
        // marking to tell them apart (half the private roads in a Norwegian
        // valley are asphalt, half are gravel, and it is stable per road).
        static std::uint32_t idHash(const std::string& s) {
            std::uint32_t h = 2166136261u;
            for (char c : s) {
                h ^= static_cast<std::uint32_t>(static_cast<unsigned char>(c));
                h *= 16777619u;
            }
            return h;
        }

        // The marking style of one road: its class decides the paint, its width
        // decides the geometry the paint sits in. Widths come from paramsFor +
        // the ribbon piece's narrowed sealed edge, so the texture's metre-space
        // mapping matches the mesh exactly.
        [[nodiscard]] RoadSurfaceStyle styleFor(const RoadSpec& spec, int variant) const {
            RoadParams p = paramsFor(spec);
            const float shoulder = std::min(p.shoulderWidth, 0.5f);
            RoadSurfaceStyle st;
            st.pavedWidth = p.laneWidth * static_cast<float>(std::max(p.laneCount, 1));
            st.fullWidth = st.pavedWidth + 2.f * shoulder;
            st.tileLength = rules_.tileLength;
            st.wear = rules_.wear;
            st.gravelEdge = rules_.gravelEdge;
            st.whiteColor = rules_.whiteColor;
            st.yellowColor = rules_.yellowColor;
            st.centreOn = rules_.centreOn;
            st.centreOff = rules_.centreOff;
            st.edgeOn = rules_.edgeOn;
            st.edgeOff = rules_.edgeOff;
            st.seed = rules_.seed + 7919u * static_cast<unsigned int>(variant) + idHash(spec.category);
            const MarkingClass cls = classOf(spec);
            st.lineWidth = (spec.width >= rules_.wideMin) ? rules_.lineWidthWide : rules_.lineWidth;
            st.shoulderInset = (cls == MarkingClass::Main) ? rules_.shoulderInsetMain
                                                           : rules_.shoulderInset;
            switch (cls) {
                case MarkingClass::Main:
                    st.centre = LinePattern::Dashed;// skillelinje 3+9, yellow
                    st.edge = LinePattern::Solid;   // heltrukken kantlinje
                    break;
                case MarkingClass::Narrow:
                    st.centre = LinePattern::None;// too narrow for two lanes
                    st.edge = LinePattern::Dashed;// stiplet kantlinje 3+3
                    break;
                case MarkingClass::Unmarked:
                    // A 4 m private road carries NO paint at all. Both patterns
                    // must be cleared explicitly: RoadSurfaceStyle defaults its
                    // edge to Solid (the common case), so falling through here
                    // with only the surface set leaves phantom kantlinjer on
                    // every farm track — which is exactly what the first
                    // road-aerial of a P road showed.
                    st.centre = LinePattern::None;
                    st.edge = LinePattern::None;
                    st.kind = (idHash(spec.id) & 1u) ? SurfaceKind::Gravel : SurfaceKind::Asphalt;
                    break;
                case MarkingClass::Track:
                    st.centre = LinePattern::None;
                    st.edge = LinePattern::None;
                    st.kind = SurfaceKind::Gravel;
                    break;
            }
            return st;
        }

        // The SHARED material for a road's class/width/variant. 1397 aalesund
        // chunks used to mean 1397 bakes and ~180 MB of texture; there are only a
        // handful of distinct (class, width) pairs in a pack, so bake each once
        // and hand the same three maps to every piece that wants them.
        [[nodiscard]] std::shared_ptr<MeshStandardMaterial> surfaceMaterialFor(const RoadSpec& spec,
                                                                               int variant) const {
            const RoadSurfaceStyle st = styleFor(spec, variant);
            const std::uint64_t key =
                    (static_cast<std::uint64_t>(static_cast<int>(classOf(spec))) << 40) |
                    (static_cast<std::uint64_t>(st.kind == SurfaceKind::Gravel ? 1 : 0) << 36) |
                    ((static_cast<std::uint64_t>(std::lround(st.pavedWidth * 4.f)) & 0xFFFFull) << 16) |
                    ((static_cast<std::uint64_t>(std::lround(st.wear * 10.f)) & 0xFFull) << 8) |
                    (static_cast<std::uint64_t>(variant) & 0xFFull);
            if (const auto it = surfaceCache_.find(key); it != surfaceCache_.end()) return it->second;

            const auto maps = RoadGenerator::bakeSurfaceMaps(st, rules_.texWidth, rules_.texHeight);
            const auto mkTex = [&](std::vector<unsigned char> px, bool srgb) {
                auto t = DataTexture::create(ImageData{std::move(px)},
                                             static_cast<unsigned int>(maps.width),
                                             static_cast<unsigned int>(maps.height));
                t->colorSpace = srgb ? ColorSpace::sRGB : ColorSpace::NoColorSpace;
                t->magFilter = Filter::Linear;
                t->minFilter = Filter::LinearMipmapLinear;
                t->generateMipmaps = true;// DataTexture defaults false -> GL black
                t->anisotropy = 16;       // permanent grazing incidence (see buildMeshes)
                t->wrapS = TextureWrapping::ClampToEdge;
                t->wrapT = TextureWrapping::Repeat;// tiles along the road length
                return t;
            };
            auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                            .color(Color::white)
                                                            .roughness(1.f)
                                                            .metalness(1.f));// the maps carry both
            mat->map = mkTex(maps.albedo, true);
            mat->normalMap = mkTex(maps.normal, false);
            auto rm = mkTex(maps.roughMetal, false);
            mat->roughnessMap = rm;// g = roughness
            mat->metalnessMap = rm;// b = metalness (zero: asphalt is a dielectric)
            surfaceCache_.emplace(key, mat);
            return mat;
        }

        // The ONE shared repair-patch material (albedo + roughMetal atlas, four
        // variants). Baked on first use, like the surface sets.
        [[nodiscard]] std::shared_ptr<MeshStandardMaterial> patchMaterial() const {
            if (patchMat_) return patchMat_;
            const auto atlas = RoadGenerator::bakePatchAtlas(rules_.patchRes, rules_.patchSeed);
            const auto mkTex = [&](std::vector<unsigned char> px, bool srgb) {
                auto t = DataTexture::create(ImageData{std::move(px)},
                                             static_cast<unsigned int>(atlas.width),
                                             static_cast<unsigned int>(atlas.height));
                t->colorSpace = srgb ? ColorSpace::sRGB : ColorSpace::NoColorSpace;
                t->magFilter = Filter::Linear;
                t->minFilter = Filter::LinearMipmapLinear;
                t->generateMipmaps = true;
                t->anisotropy = 16;
                t->wrapS = TextureWrapping::ClampToEdge;
                t->wrapT = TextureWrapping::ClampToEdge;
                return t;
            };
            auto mat = MeshStandardMaterial::create(MeshStandardMaterial::Params{}
                                                            .color(Color::white)
                                                            .roughness(1.f)
                                                            .metalness(1.f));
            mat->map = mkTex(atlas.albedo, true);
            auto rm = mkTex(atlas.roughMetal, false);
            mat->roughnessMap = rm;
            mat->metalnessMap = rm;
            patchMat_ = mat;
            return patchMat_;
        }

        // Repair-patch DECALS for ONE ribbon piece: 0-3 patches per 100 m, each
        // a quad lying on the ribbon surface (same cross-section formula, lifted
        // 1.5 cm) with one of the atlas's four variants and a random u/v flip.
        //
        // This is where the patches moved TO. In the shared 96 m tile a patch was
        // one rectangle that every chunk of every road of that class carried at
        // the same lateral position every 96 m — "these patches are just repeated
        // the same way at an interval". Here the count, the stations, the lanes,
        // the sizes, the yaws and the variants are all functions of (road id,
        // piece index), so two pieces never place the same patch in the same
        // place, and a road that has no piece boundary in frame still gets
        // patches that are unique to that stretch.
        //
        // All of a piece's patches are ONE indexed geometry (one draw, one entry)
        // and the mesh is a CHILD of the ribbon piece, so the demo's distance
        // cull — which toggles the piece's `visible` — hides them with it (both
        // renderers skip an invisible node's whole subtree).
        [[nodiscard]] std::shared_ptr<Mesh> buildPatchDecals(const RoadGenerator& gen,
                                                             const RoadSpec& spec,
                                                             const RoadSurfaceStyle& st,
                                                             int idx) const {
            if (st.kind != SurfaceKind::Asphalt || rules_.wear <= 0.05f) return nullptr;
            const float len = gen.totalLength();
            if (len < 8.f) return nullptr;
            const float pavedHalf = std::max(st.pavedWidth, 1.f) * 0.5f;
            const std::uint32_t base = idHash(spec.id) ^ (static_cast<std::uint32_t>(idx) * 2654435761u);
            // Cheap deterministic stream: h(k) in [0,1).
            const auto h01 = [base](int k) {
                std::uint32_t x = base + static_cast<std::uint32_t>(k) * 2246822519u;
                x ^= x >> 15;
                x *= 2654435761u;
                x ^= x >> 13;
                x *= 3266489917u;
                x ^= x >> 16;
                return static_cast<float>(x & 0xFFFFFFu) / static_cast<float>(0x1000000);
            };

            std::vector<float> pos, nrm, uv;
            std::vector<unsigned int> idxs;
            const int windows = std::max(1, static_cast<int>(std::ceil(len / 100.f)));
            int key = 0;
            for (int wI = 0; wI < windows; ++wI) {
                const float w0 = static_cast<float>(wI) * 100.f;
                const float w1 = std::min(w0 + 100.f, len);
                if (w1 - w0 < 6.f) continue;
                const int count = static_cast<int>(h01(key++) * (rules_.patchesPer100m + 0.999f));
                for (int p = 0; p < count; ++p) {
                    const float halfWid = 0.5f * (1.5f + 2.5f * h01(key++));// 1.5-4 m across
                    const float halfLen = 0.5f * (2.0f + 4.0f * h01(key++));// 2-6 m along
                    const float station = w0 + halfLen + h01(key++) * std::max(w1 - w0 - 2.f * halfLen, 0.1f);
                    if (station - halfLen < 0.5f || station + halfLen > len - 0.5f) continue;
                    // Centred on a lane, jittered enough that some straddle the
                    // centre line — a resurfacing does not respect the paint.
                    const float laneC = (st.centre != LinePattern::None)
                                                ? (h01(key++) > 0.5f ? 1.f : -1.f) * pavedHalf * 0.5f
                                                : (++key, 0.f);
                    float lat = laneC + (h01(key++) * 2.f - 1.f) * 0.9f;
                    const float limit = std::max(pavedHalf - 0.1f - halfWid, 0.f);
                    lat = std::clamp(lat, -limit, limit);
                    const float yaw = (h01(key++) * 2.f - 1.f) * 3.f * math::DEG2RAD;
                    const int variant = static_cast<int>(h01(key++) * RoadGenerator::kPatchVariants) %
                                        RoadGenerator::kPatchVariants;
                    const bool flipU = h01(key++) > 0.5f, flipV = h01(key++) > 0.5f;

                    const float cy = std::cos(yaw), sy = std::sin(yaw);
                    const unsigned int v0 = static_cast<unsigned int>(pos.size() / 3);
                    // Corner order (along, across): (-,-) (+,-) (-,+) (+,+).
                    std::array<Vector3, 4> corner{};
                    for (int c = 0; c < 4; ++c) {
                        const float ds = ((c & 1) ? 1.f : -1.f) * halfLen;
                        const float dl = ((c & 2) ? 1.f : -1.f) * halfWid;
                        const float ds2 = ds * cy - dl * sy;
                        const float dl2 = ds * sy + dl * cy;
                        corner[c] = gen.surfacePointAt(station + ds2, lat + dl2, 0.03f);
                    }
                    // The TRUE quad normal, not (0,1,0). A patch lies on a road
                    // that is cambered and on a grade, so its plane is tilted by
                    // up to ~0.1 rad from vertical — and a shading normal that
                    // disagrees with the geometry by that much puts the shadow
                    // ray's bias origin BELOW the triangle it came from, which
                    // self-shadows the quad. Measured: with up-normals the patch
                    // rendered 78 -> 57 -> 35 across its own face against a road
                    // at 85, i.e. classic acne, not the -0.08 albedo step it is
                    // supposed to be.
                    Vector3 nq = corner[2].clone().sub(corner[0])
                                         .cross(corner[1].clone().sub(corner[0]));
                    if (nq.length() < 1e-8f) nq.set(0.f, 1.f, 0.f);
                    nq.normalize();
                    if (nq.y < 0.f) nq.multiplyScalar(-1.f);
                    for (int c = 0; c < 4; ++c) {
                        const Vector3& P = corner[c];
                        pos.push_back(P.x);
                        pos.push_back(P.y);
                        pos.push_back(P.z);
                        nrm.push_back(nq.x);
                        nrm.push_back(nq.y);
                        nrm.push_back(nq.z);
                        // Half-texel inset so the atlas cells never bleed.
                        const float e = 0.5f / static_cast<float>(std::max(rules_.patchRes, 16));
                        float fu = (c & 1) ? 1.f - e : e;
                        float fv = (c & 2) ? 1.f - e : e;
                        if (flipU) fu = 1.f - fu;
                        if (flipV) fv = 1.f - fv;
                        uv.push_back((static_cast<float>(variant) + fu) /
                                     static_cast<float>(RoadGenerator::kPatchVariants));
                        uv.push_back(fv);
                    }
                    // Wound to face UP, matching buildSurface's convention.
                    idxs.push_back(v0 + 0); idxs.push_back(v0 + 2); idxs.push_back(v0 + 1);
                    idxs.push_back(v0 + 1); idxs.push_back(v0 + 2); idxs.push_back(v0 + 3);
                }
            }
            if (idxs.empty()) return nullptr;

            auto geo = std::make_shared<BufferGeometry>();
            geo->setIndex(idxs);
            geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
            geo->computeBoundingBox();
            geo->computeBoundingSphere();
            auto mesh = Mesh::create(geo, patchMaterial());
            mesh->name = "roadpatch_" + (spec.id.empty() ? std::string("?") : spec.id) + "_" +
                         std::to_string(idx);
            mesh->receiveShadow = true;
            mesh->autoLod = false;
            return mesh;
        }

        // One ribbon sub-mesh over `pts` (a slice of a road's conformed dense
        // centerline, y = final grade): sub-generator draped onto the slice's
        // own profile + the same material/texture recipe as buildMeshes.
        // Shared by buildBridgeMeshes (deck spans) and buildGroundChunkMeshes
        // (distance-culled near-detail chunks). Returns nullptr on degenerate
        // geometry.
        [[nodiscard]] std::shared_ptr<Mesh> buildRibbonPiece(const Road& r,
                                                             const std::vector<Vector3>& pts,
                                                             const char* namePrefix, int idx,
                                                             int texWidth, int texHeight,
                                                             float station = 0.f,
                                                             bool patchDecals = false) const {
            (void) texWidth;
            (void) texHeight;
            if (pts.size() < 2) return nullptr;
            RoadParams p = paramsFor(r.spec);
            p.samplesPerSegment = 2;// slice points are already dense
            // Baked pipeline: the TERRAIN owns the verge (carveRoads bakes the
            // roadbed and makeGeoProvider paints asphalt→grass over pavedWeight),
            // so these ribbons need only a THIN SEALED edge to grade their
            // kSurfaceRaise lip down to grade — not the wide GRAVEL shoulder the
            // legacy ribbon-is-the-road look bakes, which reads as a dirt band
            // beside a road that is already painted into the ground. Narrow the
            // shoulder and colour it asphalt (a sealed road edge). buildMeshes
            // (legacy full ribbons) keeps the gravel verge.
            p.shoulderWidth = std::min(p.shoulderWidth, 0.5f);
            p.shoulderColor = p.asphaltColor;
            // The marking tile is SHARED (one bake per class/width, see
            // surfaceMaterialFor) and 48 m long — an exact whole number of both
            // the 3+9 centre and the 3+3 edge dash periods — so the dashes carry
            // across a chunk seam as long as the piece's v starts at its GLOBAL
            // station along the road rather than at zero.
            p.textureTileLength = rules_.tileLength;
            RoadGenerator gen(pts, p);
            // "Ground" for the piece is its own profile — light smoothing keeps
            // a deck taut instead of re-draping it into the ravine/water it
            // exists to cross, and is a near-no-op for ground chunks (their
            // heights are already the smoothed conformed grade).
            gen.conformTo([&pts](float x, float z) {
                return polylineHeightAt(pts, x, z);
            }, 2);
            // Mirror u on odd tiles: the tile's own repeat becomes 2 x tileLength
            // and comes back left-right flipped rather than identical. Safe here
            // because every baked class layout is symmetric (see buildSurface).
            auto geo = gen.buildSurface(station, /*mirrorAlternateTiles*/ true);
            if (!geo->getAttribute<float>("position") ||
                geo->getAttribute<float>("position")->count() == 0)
                return nullptr;

            // Two seeded variants per class alternate by piece index, so the
            // shared tile's own repeat is 192 m of road rather than 96.
            auto mat = surfaceMaterialFor(r.spec, idx & 1);

            auto mesh = Mesh::create(geo, mat);
            mesh->name = std::string(namePrefix) + "_" +
                         (r.spec.id.empty() ? std::string("?") : r.spec.id) +
                         "_" + std::to_string(idx);
            mesh->receiveShadow = true;
            mesh->autoLod = false;// thin strip — LOD would sliver it (see buildMeshes)
            // GROUND chunks only. A bridge deck is a structure, not a stretch of
            // pavement that has been dug up and refilled — patching one reads as
            // a mistake, and the deck's own piece indices would collide with the
            // ground chunks' anyway.
            if (patchDecals) {
                if (auto patches = buildPatchDecals(gen, r.spec, styleFor(r.spec, idx & 1), idx))
                    mesh->add(patches);
            }
            return mesh;
        }

        // Interpolated polyline height at the nearest XZ point on `pts`.
        // O(points) — used only at load time (conform/classification/bridges).
        static float polylineHeightAt(const std::vector<Vector3>& pts, float x, float z) {
            float bestD = std::numeric_limits<float>::max();
            float bestY = 0.f;
            for (size_t k = 1; k < pts.size(); ++k) {
                const Vector3& a = pts[k - 1];
                const Vector3& b = pts[k];
                const float abx = b.x - a.x, abz = b.z - a.z;
                const float lsq = abx * abx + abz * abz;
                float t = (lsq > 1e-12f) ? ((x - a.x) * abx + (z - a.z) * abz) / lsq : 0.f;
                t = std::clamp(t, 0.f, 1.f);
                const float dx = x - (a.x + t * abx), dz = z - (a.z + t * abz);
                const float d = dx * dx + dz * dz;
                if (d < bestD) {
                    bestD = d;
                    bestY = a.y + (b.y - a.y) * t;
                }
            }
            return bestY;
        }
        // Data height (spec point Y — the pack's true road elevation) at (x,z).
        static float specHeightAt(const RoadSpec& s, float x, float z) {
            return polylineHeightAt(s.points, x, z);
        }

        // Profile-mode classify + conform for one road (see RoadProfileOptions).
        // Per dense centerline sample: bridge where the data height spans
        // plausibly above the ground or the ground is water; tunnel-excluded
        // where the data height runs well below the ground; ferry-excluded for
        // long water runs whose data height never clears the deck minimum;
        // short bridge runs and implausible (garbage) data lifts demote back to
        // ground. Every sample then gets an explicit TARGET elevation for its
        // final role — deck height for spans, raw ground for everything else —
        // and the road conforms to those targets (conformToHeights), so demoted
        // spans never carry a stale lift. Sample flags are kept on the Road
        // (for buildBridgeMeshes); foldSegFlags() mirrors them onto the Seg
        // soup for the terrain queries + carveRoads.
        void conformRoadProfiled(Road& r, const std::function<float(float, float)>& groundFn,
                                 int smoothingPasses, const RoadProfileOptions& po) {
            const auto& cl = r.gen->centerlineSamples();
            const size_t n = cl.size();
            std::vector<std::uint8_t> f(n, 0);
            std::vector<float> ground(n), dataY(n);
            for (size_t i = 0; i < n; ++i) {
                ground[i] = groundFn(cl[i].x, cl[i].z);
                dataY[i] = specHeightAt(r.spec, cl[i].x, cl[i].z);
                const bool water = ground[i] <= po.seaLevel + po.waterEps;
                const float lift = dataY[i] - ground[i];
                // A plausible lift is a span; an implausible one is garbage
                // elevation data and stays on the ground. Water always spans
                // (a road is never submerged) — ferry runs are culled below.
                if ((lift > po.bridgeThresh && lift <= po.maxDataLift) || water) f[i] |= kSegBridge;
                // Tunnel wins outright — including SUBSEA tunnels, whose data
                // height runs far below the water sheet.
                if (ground[i] - dataY[i] > po.tunnelThresh) f[i] = kSegExcluded;
            }
            const auto runLen = [&cl](size_t i, size_t j) {
                float len = 0.f;
                for (size_t k = i + 1; k < j; ++k) {
                    const float dx = cl[k].x - cl[k - 1].x, dz = cl[k].z - cl[k - 1].z;
                    len += std::sqrt(dx * dx + dz * dz);
                }
                return len;
            };
            // Ferry / tunnel-flat / garbage: a long water crossing is a REAL
            // bridge only if its data heights are ELEVATED along the span (a
            // deck arches over the water) — judged by the run's MEDIAN data
            // height against the bridge threshold. A ferry leg or a flattened
            // subsea-tunnel polyline runs at quay/portal height (median ≈ sea
            // even when an endpoint pokes a few metres up), and rendering it
            // would lay a road ON the sea — exclude the run outright.
            for (size_t i = 0; i < n;) {
                if (ground[i] > po.seaLevel + po.waterEps) { ++i; continue; }
                size_t j = i;
                while (j < n && ground[j] <= po.seaLevel + po.waterEps) ++j;
                if (runLen(i, j) > po.ferryMinLen) {
                    std::vector<float> med(dataY.begin() + static_cast<std::ptrdiff_t>(i),
                                           dataY.begin() + static_cast<std::ptrdiff_t>(j));
                    std::nth_element(med.begin(), med.begin() + static_cast<std::ptrdiff_t>(med.size() / 2),
                                     med.end());
                    if (med[med.size() / 2] < po.seaLevel + po.bridgeThresh)
                        for (size_t k = i; k < j; ++k) f[k] = kSegExcluded;
                }
                i = j;
            }
            // Demote short bridge runs (shore blips where the centerline grazes
            // a water cell, culvert-scale stream hops, single-point data noise).
            // ON LAND the carve fills them as a causeway — how a real roadbed
            // crosses a dip. OVER WATER a demoted blip must be EXCLUDED, never
            // ground: a "ground" stub standing on the open sea is exactly the
            // floating-shard artifact class this feature exists to kill.
            for (size_t i = 0; i < n;) {
                if (f[i] != kSegBridge) { ++i; continue; }
                size_t j = i;
                while (j < n && f[j] == kSegBridge) ++j;
                // A short run is a blip to demote ONLY if it is also SHALLOW; a
                // short run whose deck rises well above ground is a real bridge
                // over a narrow ravine (draping it makes the hard vertical U).
                float peakLift = 0.f;
                for (size_t k = i; k < j; ++k) peakLift = std::max(peakLift, dataY[k] - ground[k]);
                if (runLen(i, j) < po.minSpanLen && peakLift <= po.shortSpanMaxLift)
                    for (size_t k = i; k < j; ++k)
                        f[k] = (ground[k] <= po.seaLevel + po.waterEps)
                                       ? kSegExcluded
                                       : static_cast<std::uint8_t>(f[k] & ~kSegBridge);
                i = j;
            }
            // Absorb short keep-islands flanked by exclusion: the fjord DTM has
            // skerries/shoals that split a ferry crossing into fragments — a
            // few samples of "land" (or a sub-minSpan bridge) in the middle of
            // an excluded crossing is data noise, not a road you can stand on.
            for (size_t i = 0; i < n;) {
                if (f[i] == kSegExcluded) { ++i; continue; }
                size_t j = i;
                while (j < n && f[j] != kSegExcluded) ++j;
                const bool flankedL = (i == 0) || f[i - 1] == kSegExcluded;
                const bool flankedR = (j == n) || f[j] == kSegExcluded;
                // Interior islands only — a run touching the polyline START and
                // END is the whole road, never absorbed.
                const bool interior = !(i == 0 && j == n);
                if (interior && flankedL && flankedR && runLen(i, j) < po.minSpanLen)
                    for (size_t k = i; k < j; ++k) f[k] = kSegExcluded;
                i = j;
            }
            // Anchor decks on solid ground (see deckAnchorLift): extend each
            // surviving span outward while the data line still rides clear of
            // the ground, bounded per side. Runs are collected first so an
            // extension never feeds the next run's scan.
            {
                std::vector<std::pair<size_t, size_t>> runs;
                for (size_t i = 0; i < n;) {
                    if (f[i] != kSegBridge) { ++i; continue; }
                    size_t j = i;
                    while (j < n && f[j] == kSegBridge) ++j;
                    runs.emplace_back(i, j);
                    i = j;
                }
                const auto stepLen = [&cl](size_t a, size_t b) {
                    const float dx = cl[a].x - cl[b].x, dz = cl[a].z - cl[b].z;
                    return std::sqrt(dx * dx + dz * dz);
                };
                for (const auto& [ri, rj] : runs) {
                    float len = 0.f;
                    for (size_t k = ri; k > 0 && f[k - 1] == 0 &&
                                        dataY[k - 1] - ground[k - 1] > po.deckAnchorLift;) {
                        len += stepLen(k, k - 1);
                        if (len > po.deckAnchorMax) break;
                        f[--k] = kSegBridge;
                    }
                    len = 0.f;
                    for (size_t k = rj; k < n && f[k] == 0 &&
                                        dataY[k] - ground[k] > po.deckAnchorLift;) {
                        len += stepLen(k, k > 0 ? k - 1 : k);
                        if (len > po.deckAnchorMax) break;
                        f[k++] = kSegBridge;
                    }
                }
            }
            // Explicit target elevation per sample for its FINAL role. Demoted
            // water blips still hold the deck minimum (a causeway pad, never a
            // submerged roadbed); everything else on ground drapes the ground.
            std::vector<float> target(n);
            for (size_t i = 0; i < n; ++i) {
                const bool water = ground[i] <= po.seaLevel + po.waterEps;
                if (f[i] & kSegBridge)
                    target[i] = std::max(dataY[i], po.seaLevel + po.deckMin);
                else
                    target[i] = water ? po.seaLevel + po.deckMin : ground[i];
            }
            // Arc-length grade smoothing (drivable vertical alignment): average
            // each GROUND sample's target over a fixed metric window so the road
            // rides a smooth grade instead of draping over the DEM's mid-scale
            // undulation. Bridge/excluded samples keep their classified height and
            // are skipped both as targets and as neighbours (a deck must not drag
            // the approach grade up, nor the approach flatten the deck).
            if (po.gradeSmoothing > 0.f && n > 2) {
                std::vector<float> arc(n, 0.f);
                for (size_t i = 1; i < n; ++i) {
                    const float dx = cl[i].x - cl[i - 1].x, dz = cl[i].z - cl[i - 1].z;
                    arc[i] = arc[i - 1] + std::sqrt(dx * dx + dz * dz);
                }
                const float half = po.gradeSmoothing * 0.5f;
                std::vector<float> sm = target;
                for (size_t i = 0; i < n; ++i) {
                    if (f[i] != 0) continue;// keep deck / excluded targets
                    float sum = target[i];
                    int cnt = 1;
                    for (size_t k = i + 1; k < n && arc[k] - arc[i] <= half; ++k)
                        if (f[k] == 0) { sum += target[k]; ++cnt; }
                    for (size_t k = i; k-- > 0 && arc[i] - arc[k] <= half;)
                        if (f[k] == 0) { sum += target[k]; ++cnt; }
                    sm[i] = sum / static_cast<float>(cnt);
                }
                target.swap(sm);
            }
            r.gen->conformToHeights(target, smoothingPasses);
            r.sampleFlags = std::move(f);
        }

        // Mirror the per-sample classification onto the segment soup (exclusion
        // dominates). Ordering matches snapshotHeights/buildXZGeometry.
        void foldSegFlags() {
            size_t si = 0;
            for (const auto& r : roads_) {
                const auto& f = r.sampleFlags;
                for (size_t k = 1; k < f.size(); ++k, ++si) {
                    if (si >= segs_.size()) return;// defensive (shouldn't happen)
                    const std::uint8_t u = static_cast<std::uint8_t>(f[k - 1] | f[k]);
                    segs_[si].flags = (u & kSegExcluded) ? kSegExcluded
                                                         : static_cast<std::uint8_t>(u & kSegBridge);
                }
            }
        }

        // Point-to-segment distance in XZ; writes the clamped projection param t
        // (0 at a, 1 at b) for height interpolation.
        static float distToSegment(float x, float z, const Seg& s, float& t) {
            const float abx = s.bx - s.ax, abz = s.bz - s.az;
            const float apx = x - s.ax, apz = z - s.az;
            const float abLenSq = abx * abx + abz * abz;
            t = (abLenSq > 1e-12f) ? (apx * abx + apz * abz) / abLenSq : 0.f;
            t = std::clamp(t, 0.f, 1.f);
            const float cx = s.ax + t * abx, cz = s.az + t * abz;
            const float dx = x - cx, dz = z - cz;
            return std::sqrt(dx * dx + dz * dz);
        }


        float flattenMargin_;
        float trenchDepth_ = 0.35f;
        bool conformed_ = false;
        MarkingRules rules_{};
        // Baked surface sets, keyed on (class, surface, width, wear, variant) —
        // built lazily by surfaceMaterialFor from the const mesh builders, hence
        // mutable. Not part of the logical const state.
        mutable std::unordered_map<std::uint64_t, std::shared_ptr<MeshStandardMaterial>> surfaceCache_;
        mutable std::shared_ptr<MeshStandardMaterial> patchMat_;// shared repair-patch atlas
        mutable bool meanCached_[2] = {false, false};
        mutable std::array<float, 3> meanValue_[2] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
        std::vector<Road> roads_;

        std::vector<Seg> segs_;
        std::unordered_map<std::int64_t, std::vector<int>> grid_;
        float cellSize_ = 32.f;
    };

}// namespace threepp::road

#endif//THREEPP_EXTRAS_ROAD_ROADNETWORK_HPP
