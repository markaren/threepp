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
#include <vector>

namespace threepp::road {

    // One road as fed to the network: a centerline plus its physical width and a
    // category letter that scales the verge/tessellation. Mirrors
    // terrain::GeoRoad but keeps this header independent of the pack loader.
    struct RoadSpec {
        std::string id;
        std::string category;// "E" | "R" | "F" | "K" | "P" | "S"
        float width = 6.f;   // total carriageway width (m)
        std::vector<Vector3> points;// centerline (world XZ; Y ignored — conformTo sets it)
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

        void setTrenchDepth(float d) { trenchDepth_ = std::max(d, 0.f); }
        [[nodiscard]] float trenchDepth() const { return trenchDepth_; }

        // Drape every road onto the ground function (typically the raw terrain
        // grid height), then snapshot conformed centerlines into the immutable
        // segment soup the thread-safe queries use. Call once before the queries.
        void conformTo(const std::function<float(float, float)>& groundFn, int smoothingPasses = 14) {
            for (auto& r : roads_) r.gen->conformTo(groundFn, smoothingPasses);
            snapshotHeights();
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
                float t;
                const float d = distToSegment(x, z, s, t);
                const float corridorHalf = s.corridorHalf;
                if (d >= corridorHalf + outer) return;
                const float w = 1.f - smoothstep(corridorHalf, corridorHalf + outer, d);
                if (w > bestW) {
                    bestW = w;
                    bestH = s.ha + (s.hb - s.ha) * t;
                    // Trench profile: full depth under the paved band, smoothstep
                    // back to 0 at the corridor (shoulder-outer) edge so terrain
                    // meets the ribbon shoulder flush. C1-smooth so coarse tiles
                    // don't alias the drop into a crease.
                    bestTrench = (d <= s.pavedHalf)
                                         ? 1.f
                                         : smoothstep(s.corridorHalf, s.pavedHalf, d);
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
                float t;
                const float d = distToSegment(x, z, s, t);
                float w;
                if (d <= s.pavedHalf) w = 1.f;
                else if (d >= s.corridorHalf) w = 0.f;
                else w = smoothstep(s.corridorHalf, s.pavedHalf, d);
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

    private:
        // ── per-road ownership ───────────────────────────────────────────────
        struct Road {
            RoadSpec spec;
            std::unique_ptr<RoadGenerator> gen;
            float pavedHalf = 0.f, corridorHalf = 0.f;
        };

        // ── immutable segment soup (thread-safe query backing) ───────────────
        struct Seg {
            float ax = 0.f, az = 0.f, bx = 0.f, bz = 0.f;// endpoints (XZ)
            float ha = 0.f, hb = 0.f;                    // conformed heights (set by snapshotHeights)
            float pavedHalf = 0.f, corridorHalf = 0.f;   // per-road widths
        };

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

        static float smoothstep(float e0, float e1, float x) {
            const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
            return t * t * (3.f - 2.f * t);
        }

        float flattenMargin_;
        float trenchDepth_ = 0.35f;
        bool conformed_ = false;
        std::vector<Road> roads_;

        std::vector<Seg> segs_;
        std::unordered_map<std::int64_t, std::vector<int>> grid_;
        float cellSize_ = 32.f;
    };

}// namespace threepp::road

#endif//THREEPP_EXTRAS_ROAD_ROADNETWORK_HPP
