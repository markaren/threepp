// Distance-streamed content cells around the LIVE camera.
//
// The pattern `TerrainScatter` uses for grass tufts, lifted out so the forest,
// the bushes and the urban props can share it. The defect it exists to kill:
// every one of those systems used to build its content ONCE, inside a square
// ROI centred on the STARTUP camera target. That is a still-frame trick. Fly
// two kilometres and the world is bare — no trees, no cars — because the ROI
// never moved. A town the camera can walk needs content everywhere, streamed
// by distance from where the camera IS.
//
//   • cells are keyed by (level, cx, cz) and built ONCE, on first activation,
//     by a caller-supplied builder. The result is CACHED for the life of the
//     streamer: leaving a cell removes its subtree from the scene, coming back
//     re-adds the same objects. A cell is never rebuilt, and a cell that built
//     nothing is remembered as built-and-empty so it is never re-scanned;
//   • two levels. Fine cells (`fineCellSize`) inside `fineRadius`, coarse
//     cells (`coarseFactor` × that edge) out to `coarseRadius`. The two tile
//     the same ground EXACTLY — a coarse cell is a whole block of
//     coarseFactor² fine cells, and a block is represented either by its
//     coarse cell or by its fine cells, never by both — so the seam cannot
//     double-plant a tree or leave a hole;
//   • at most `maxCellBuildsPerFrame` cells BUILD per update, so a fly-in
//     stays smooth. Activating a cached cell is free and is not budgeted.
//     The first update builds everything in range in one go: a startup frame
//     is allowed to be slow, and a headless --shot must not photograph a
//     half-grown world;
//   • removal has hysteresis (`removeSlack`). It matters more than the adds:
//     removing a scene entry clears the Vulkan renderer's temporal history,
//     so a cell that flickers in and out at the ring boundary would strobe the
//     whole frame. Adds are appends and are cheap.
//
// Header-only, threepp core only. update() is main-thread only.

#ifndef THREEPP_EXTRAS_TERRAIN_CELLSTREAMER_HPP
#define THREEPP_EXTRAS_TERRAIN_CELLSTREAMER_HPP

#include "threepp/math/Vector3.hpp"
#include "threepp/objects/Group.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace threepp::terrain {

    struct CellStreamerOptions {
        float fineCellSize = 128.f;
        float fineRadius = 800.f;
        float coarseRadius = 1600.f;// <= fineRadius disables the coarse band
        int coarseFactor = 3;       // coarse cell edge = coarseFactor × fine edge
        float removeSlack = 1.3f;   // cells drop beyond radius × this
        int maxCellBuildsPerFrame = 2;
        bool burstFirstUpdate = true;// build the whole first ring in one update
    };

    struct CellStreamerStats {
        int active = 0;   // cells currently in the scene
        int cached = 0;   // cells ever built (kept for revisits)
        int builds = 0;   // built by the last update()
        int adds = 0;     // cached cells re-attached by the last update()
        int removes = 0;  // detached by the last update()
        int pending = 0;  // wanted but not yet built (budget)
        long long totalBuilds = 0, totalRemoves = 0;
    };

    class CellStreamer: public Group {

    public:
        // level 0 = fine, 1 = coarse. Return nullptr for "this cell holds
        // nothing" — that answer is cached too.
        using Builder = std::function<std::shared_ptr<Object3D>(int level, int cx, int cz,
                                                                float cellSize)>;

        CellStreamer(Builder builder, CellStreamerOptions options = {})
            : o_(options), build_(std::move(builder)) {}

        static std::shared_ptr<CellStreamer> create(Builder builder,
                                                    CellStreamerOptions options = {}) {
            return std::make_shared<CellStreamer>(std::move(builder), options);
        }

        // Call once per frame with the primary camera position.
        void update(const Vector3& camPos) {
            st_.builds = st_.adds = st_.removes = 0;
            if (!build_) return;

            const float fcs = o_.fineCellSize;
            const int cf = std::max(1, o_.coarseFactor);
            const float ccs = fcs * static_cast<float>(cf);
            const bool twoLevel = o_.coarseRadius > o_.fineRadius * 1.01f;
            const float outer = twoLevel ? o_.coarseRadius : o_.fineRadius;
            const float outerKeep = outer * o_.removeSlack;

            // ── which cells does this camera want? ─────────────────────────
            // Decided per BLOCK (one coarse cell = cf² fine cells) so the two
            // levels partition the ground instead of overlapping.
            std::map<std::int64_t, float> want;// key → distance² (build order)
            const int b0x = ifloor((camPos.x - outerKeep) / ccs);
            const int b1x = ifloor((camPos.x + outerKeep) / ccs);
            const int b0z = ifloor((camPos.z - outerKeep) / ccs);
            const int b1z = ifloor((camPos.z + outerKeep) / ccs);
            ++gen_;
            for (int bz = b0z; bz <= b1z; ++bz) {
                for (int bx = b0x; bx <= b1x; ++bx) {
                    const float d = boxDistance(camPos, static_cast<float>(bx) * ccs,
                                                static_cast<float>(bz) * ccs, ccs);
                    const std::int64_t bk = key(0, bx, bz);
                    auto it = blocks_.find(bk);
                    const bool known = it != blocks_.end();
                    // Hysteresis on both decisions: whether the block is in
                    // range at all, and whether it is drawn fine or coarse.
                    const bool inRange = known ? d <= outerKeep : d <= outer;
                    if (!inRange) {
                        if (known) blocks_.erase(it);
                        continue;
                    }
                    bool fine = !twoLevel;
                    if (twoLevel) {
                        const bool was = known && it->second.fine;
                        fine = was ? d <= o_.fineRadius * o_.removeSlack : d <= o_.fineRadius;
                    }
                    if (known) {
                        it->second.fine = fine;
                        it->second.gen = gen_;
                    } else {
                        blocks_[bk] = Block{fine, gen_};
                    }
                    if (fine) {
                        for (int i = 0; i < cf; ++i)
                            for (int j = 0; j < cf; ++j) {
                                const int cx = bx * cf + j, cz = bz * cf + i;
                                want[key(0, cx, cz)] =
                                        boxDistance(camPos, static_cast<float>(cx) * fcs,
                                                    static_cast<float>(cz) * fcs, fcs);
                            }
                    } else {
                        want[key(1, bx, bz)] = d;
                    }
                }
            }
            for (auto it = blocks_.begin(); it != blocks_.end();)
                it = (it->second.gen != gen_) ? blocks_.erase(it) : std::next(it);

            // ── detach what is no longer wanted ────────────────────────────
            for (auto& [k, c] : cells_) {
                if (!c.active || want.count(k)) continue;
                if (c.content) remove(*c.content);
                c.active = false;
                ++st_.removes;
                ++st_.totalRemoves;
            }

            // ── attach cached cells (free), build missing ones (budgeted) ──
            std::vector<std::pair<float, std::int64_t>> missing;
            for (const auto& [k, d] : want) {
                auto it = cells_.find(k);
                if (it == cells_.end()) {
                    missing.emplace_back(d, k);
                    continue;
                }
                if (!it->second.active) {
                    if (it->second.content) addRef(*it->second.content);
                    it->second.active = true;
                    ++st_.adds;
                }
            }
            std::sort(missing.begin(), missing.end());
            int budget = (first_ && o_.burstFirstUpdate) ? static_cast<int>(missing.size())
                                                         : std::max(0, o_.maxCellBuildsPerFrame);
            first_ = false;
            for (const auto& [d, k] : missing) {
                if (budget-- <= 0) break;
                const int level = static_cast<int>((k >> 48) & 0xff);
                const float cs = level ? ccs : fcs;
                auto content = build_(level, cxOf(k), czOf(k), cs);
                if (content) addRef(*content);
                cells_.emplace(k, Cell{std::move(content), true});
                ++st_.builds;
                ++st_.totalBuilds;
            }
            st_.pending = static_cast<int>(missing.size()) - st_.builds;
            st_.cached = static_cast<int>(cells_.size());
            st_.active = 0;
            for (const auto& [k, c] : cells_)
                if (c.active) ++st_.active;
        }

        [[nodiscard]] const CellStreamerStats& stats() const { return st_; }
        [[nodiscard]] const CellStreamerOptions& options() const { return o_; }

    private:
        struct Cell {
            std::shared_ptr<Object3D> content;// null = built, empty
            bool active = false;
        };
        struct Block {
            bool fine = true;
            unsigned int gen = 0;
        };

        static int ifloor(float v) { return static_cast<int>(std::floor(v)); }

        // Distance from the camera to a cell's XZ footprint (0 inside it).
        static float boxDistance(const Vector3& p, float x0, float z0, float size) {
            const float dx = std::max({x0 - p.x, 0.f, p.x - (x0 + size)});
            const float dz = std::max({z0 - p.z, 0.f, p.z - (z0 + size)});
            return std::sqrt(dx * dx + dz * dz);
        }

        static std::int64_t key(int level, int cx, int cz) {
            return (static_cast<std::int64_t>(level) << 48) |
                   ((static_cast<std::int64_t>(cx) & 0xffffff) << 24) |
                   (static_cast<std::int64_t>(cz) & 0xffffff);
        }
        static int sext24(std::int64_t v) {
            const auto u = static_cast<std::int32_t>(v & 0xffffff);
            return (u & 0x800000) ? u - 0x1000000 : u;
        }
        static int cxOf(std::int64_t k) { return sext24(k >> 24); }
        static int czOf(std::int64_t k) { return sext24(k); }

        CellStreamerOptions o_;
        Builder build_;
        std::map<std::int64_t, Cell> cells_;
        std::map<std::int64_t, Block> blocks_;
        CellStreamerStats st_;
        unsigned int gen_ = 0;
        bool first_ = true;
    };

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_CELLSTREAMER_HPP
