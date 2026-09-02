
#include "threepp/splats/SplatLod.hpp"

#include "threepp/cameras/Camera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/loaders/SogLoader.hpp"
#include "threepp/math/Frustum.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Sphere.hpp"
#include "threepp/objects/SplatCloud.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace threepp::splats {

    namespace {

        // Pixels per world unit at `dist`. An orthographic camera has no
        // distance falloff; a perspective one does. zoom participates in both,
        // which is what makes the editor's scroll-zoom drive the policy.
        float pixelsPerUnit(const Camera& camera, int viewportHeightPx, float dist) {

            if (const auto* pc = dynamic_cast<const PerspectiveCamera*>(&camera)) {

                return 0.5f * static_cast<float>(viewportHeightPx) /
                       (std::tan(math::degToRad(pc->fov * 0.5f / std::max(pc->zoom, 1e-3f))) *
                        std::max(dist, 1e-3f));
            }
            // Orthographic: projectionMatrix[5] = 2/(top-bottom) * zoom.
            return 0.5f * static_cast<float>(viewportHeightPx) * camera.projectionMatrix.elements[5] * 0.5f;
        }

        // The frame's pixel budget. Only the height is passed in (a render
        // scale is one number), so the width comes from the camera's aspect
        // where there is one and from 16:9 where there is not.
        float screenAreaPx(const Camera& camera, int viewportHeightPx) {

            const auto h = static_cast<float>(std::max(viewportHeightPx, 1));
            float aspect = 16.f / 9.f;
            if (const auto* pc = dynamic_cast<const PerspectiveCamera*>(&camera)) {

                aspect = std::max(pc->aspect, 1e-3f);
            }
            return std::max(h * h * aspect, 1.f);
        }

    }// namespace

    int selectLod(SplatCloud& cloud, LodTable& table, const Camera& camera,
                  int viewportHeightPx, float targetSplatsPerPixel, float hysteresis) {

        if (!table.nodes.empty()) {

            return selectLodPerNode(cloud, table, camera, viewportHeightPx, targetSplatsPerPixel,
                                    hysteresis);
        }
        return selectLodWholeCloud(cloud, table, camera, viewportHeightPx, targetSplatsPerPixel,
                                   hysteresis);
    }

    int selectLodWholeCloud(SplatCloud& cloud, LodTable& table, const Camera& camera,
                            int viewportHeightPx, float targetSplatsPerPixel, float hysteresis) {

        if (table.empty()) return 0;
        const auto nLevels = static_cast<int>(table.levels.size());
        table.heldLevel = std::clamp(table.heldLevel, 0, nLevels - 1);

        // The cloud's PERCENTILE footprint from the table (see the field's
        // comment for why never the chunk bounds), taken into world space.
        Sphere sphere(table.center, table.radius);
        sphere.applyMatrix4(*cloud.matrixWorld);

        const auto camPos = Vector3().setFromMatrixPosition(*camera.matrixWorld);
        const float pxPerUnit = pixelsPerUnit(camera, viewportHeightPx, camPos.distanceTo(sphere.center));
        const float rPx = sphere.radius * pxPerUnit;
        const float area = std::max(math::PI * rPx * rPx, 1.f);

        // Coarsest level still covering the footprint at the target density;
        // walk from the coarsest so the first hit is the answer. Close up the
        // area outgrows every count and the walk lands on level 0 — the
        // "always highest quality when you lean in" invariant, by construction.
        int want = 0;
        for (int l = nLevels - 1; l >= 0; --l) {
            if (static_cast<float>(table.levels[static_cast<std::size_t>(l)].count) / area >=
                targetSplatsPerPixel) {
                want = l;
                break;
            }
        }
        // Hysteresis: move only past a clear margin, so an orbit hovering on a
        // threshold does not flip level every frame.
        if (want != table.heldLevel) {
            const float held = static_cast<float>(
                                       table.levels[static_cast<std::size_t>(table.heldLevel)].count) /
                               area;
            const float cand = static_cast<float>(
                                       table.levels[static_cast<std::size_t>(want)].count) /
                               area;
            const bool coarser = want > table.heldLevel;
            if (coarser ? (cand >= targetSplatsPerPixel * hysteresis)
                        : (held < targetSplatsPerPixel / hysteresis))
                table.heldLevel = want;
        }

        // The chosen level's chunks against the frustum; adjacent survivors
        // merge, so a fully visible level is ONE range and the 64-range backend
        // ceiling is never in play.
        Matrix4 vp;
        vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        Frustum frustum;
        frustum.setFromProjectionMatrix(vp);

        const auto& lvl = table.levels[static_cast<std::size_t>(table.heldLevel)];
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        for (const auto& ch : lvl.chunks) {
            Box3 world = ch.bound;
            world.applyMatrix4(*cloud.matrixWorld);
            if (!frustum.intersectsBox(world)) continue;
            const auto off = static_cast<uint32_t>(ch.offset);
            const auto cnt = static_cast<uint32_t>(ch.count);
            if (!ranges.empty() && ranges.back().first + ranges.back().second == off)
                ranges.back().second += cnt;
            else
                ranges.emplace_back(off, cnt);
        }
        // Same reason as the per-node path: an EMPTY list means "draw all of
        // me", so a framing where every chunk falls outside the frustum has to
        // submit one empty range rather than none.
        if (ranges.empty()) ranges.emplace_back(0u, 0u);
        cloud.setSubmitRanges(std::move(ranges));
        return table.heldLevel;
    }

    int selectLodPerNode(SplatCloud& cloud, LodTable& table, const Camera& camera,
                         int viewportHeightPx, float targetSplatsPerPixel, float hysteresis) {

        if (table.empty() || table.nodes.empty()) return 0;

        const auto nNodes = table.nodes.size();
        const auto nLevels = static_cast<int>(table.levels.size());

        Matrix4 vp;
        vp.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
        Frustum frustum;
        frustum.setFromProjectionMatrix(vp);

        const auto camPos = Vector3().setFromMatrixPosition(*camera.matrixWorld);
        const float screen = screenAreaPx(camera, viewportHeightPx);

        // PASS 1: visibility and footprint. Both are needed before any level
        // can be chosen, because the threshold is a function of the whole
        // visible set (see the header's part 2).
        std::vector<float> area(nNodes, 0.f);
        std::vector<unsigned char> visible(nNodes, 0);
        double areaSum = 0.0;

        for (std::size_t i = 0; i < nNodes; ++i) {

            auto& node = table.nodes[i];
            node.frameLevel = -1;
            if (node.ranges.empty()) continue;

            Box3 world = node.bound;
            world.applyMatrix4(*cloud.matrixWorld);
            if (!frustum.intersectsBox(world)) continue;

            Sphere sphere(node.center, node.radius);
            sphere.applyMatrix4(*cloud.matrixWorld);

            // Distance to the BOX, not to the centre: a node whose near face is
            // a metre away has to be treated as a metre away however deep it
            // runs, or a wall the camera is standing against coarsens because
            // its centre is twenty metres off.
            const float dist = world.distanceToPoint(camPos);
            const float rPx = sphere.radius * pixelsPerUnit(camera, viewportHeightPx, dist);
            area[i] = std::clamp(math::PI * rPx * rPx, 1.f, screen);
            visible[i] = 1;
            areaSum += static_cast<double>(area[i]);
        }

        // Nothing visible. NOT an empty list: empty means "draw all of me" to
        // the backend, so turning away from the scan would submit all 25.6 M
        // splats. One empty range is how you say nothing.
        if (areaSum <= 0.0) {

            cloud.setSubmitRanges({{0u, 0u}});
            return table.heldLevel;
        }

        // The derived per-node threshold. Guard the divisor so a single tiny
        // visible node cannot drive it to infinity.
        const float t = targetSplatsPerPixel * screen /
                        static_cast<float>(std::max(areaSum, static_cast<double>(screen)));

        // PASS 2: per node, the coarsest resident level it HAS that meets t,
        // with hysteresis on the held level. Densities are computed from the
        // node's own counts, so a node absent from a coarse level simply never
        // offers it.
        std::vector<std::pair<uint32_t, uint32_t>> segs;
        segs.reserve(nNodes);
        std::vector<float> perLevel(static_cast<std::size_t>(nLevels), 0.f);
        int finest = nLevels - 1;

        for (std::size_t i = 0; i < nNodes; ++i) {

            if (!visible[i]) continue;
            auto& node = table.nodes[i];

            // Splats per level for THIS node. Summed rather than read off one
            // range, because a level split across two chunk files arrives as
            // two ranges and half a level would read as half the density.
            const float a = area[i];
            perLevel.assign(static_cast<std::size_t>(nLevels), 0.f);
            for (const auto& r : node.ranges)
                perLevel[static_cast<std::size_t>(r.level)] += static_cast<float>(r.count);

            const int finestHere = node.ranges.front().level;
            int want = finestHere;
            for (int l = nLevels - 1; l >= 0; --l) {

                if (perLevel[static_cast<std::size_t>(l)] > 0.f &&
                    perLevel[static_cast<std::size_t>(l)] / a >= t) {

                    want = l;
                    break;
                }
            }
            // Nothing is dense enough leaves `want` at the finest level this
            // node has — the close-up invariant, kept per node.

            // Hysteresis must never hold a level this node does not carry.
            if (node.heldLevel >= 0 && perLevel[static_cast<std::size_t>(node.heldLevel)] <= 0.f)
                node.heldLevel = -1;

            if (node.heldLevel < 0) {

                node.heldLevel = want;

            } else if (want != node.heldLevel) {

                // Move only past a clear margin, so a node hovering on a
                // threshold does not flip level every frame.
                const bool coarser = want > node.heldLevel;
                if (coarser ? (perLevel[static_cast<std::size_t>(want)] / a >= t * hysteresis)
                            : (perLevel[static_cast<std::size_t>(node.heldLevel)] / a < t / hysteresis))
                    node.heldLevel = want;
            }

            node.frameLevel = node.heldLevel;
            finest = std::min(finest, node.heldLevel);

            for (const auto& r : node.ranges) {

                if (r.level != node.heldLevel) continue;
                segs.emplace_back(static_cast<uint32_t>(r.offset), static_cast<uint32_t>(r.count));
            }
        }

        // Sort by offset and merge what is already adjacent. The tree stores a
        // chunk's leaves back to back, so a run of neighbouring leaves at one
        // level collapses to a single range here.
        std::sort(segs.begin(), segs.end());
        std::vector<std::pair<uint32_t, uint32_t>> merged;
        merged.reserve(segs.size());
        for (const auto& s : segs) {

            if (!merged.empty() && merged.back().first + merged.back().second == s.first)
                merged.back().second += s.second;
            else
                merged.push_back(s);
        }

        // Above the backend's ceiling, bridge the smallest gaps. Every gap
        // inside one level's block is splats of that same level, so bridging
        // draws the in-between nodes at the finer of the two neighbours'
        // levels rather than dropping anything. Gaps are independent — merging
        // one pair changes no other pair's gap — so this is one partial sort,
        // not a loop of re-scans.
        if (merged.size() > kMaxSubmitRanges) {

            const auto nGaps = merged.size() - 1;
            std::vector<std::pair<uint64_t, std::size_t>> gaps(nGaps);
            for (std::size_t i = 0; i < nGaps; ++i) {

                const auto end = static_cast<uint64_t>(merged[i].first) + merged[i].second;
                gaps[i] = {static_cast<uint64_t>(merged[i + 1].first) - end, i};
            }
            const auto nBridge = merged.size() - kMaxSubmitRanges;
            std::nth_element(gaps.begin(), gaps.begin() + static_cast<std::ptrdiff_t>(nBridge) - 1,
                             gaps.end());
            std::vector<unsigned char> bridge(nGaps, 0);
            for (std::size_t k = 0; k < nBridge; ++k) bridge[gaps[k].second] = 1;

            std::vector<std::pair<uint32_t, uint32_t>> out;
            out.reserve(kMaxSubmitRanges);
            auto cur = merged.front();
            for (std::size_t i = 0; i < nGaps; ++i) {

                if (bridge[i])
                    cur.second = merged[i + 1].first + merged[i + 1].second - cur.first;
                else {
                    out.push_back(cur);
                    cur = merged[i + 1];
                }
            }
            out.push_back(cur);
            merged = std::move(out);
        }

        cloud.setSubmitRanges(std::move(merged));
        table.heldLevel = std::clamp(finest, 0, nLevels - 1);
        return table.heldLevel;
    }

    SogLodResult loadSogWithLod(const std::filesystem::path& path) {

        SogLodResult out;
        const auto info = SogLoader::describe(path);
        if (info.lodLevels <= 1) {
            out.data = SogLoader::load(path);
            return out;// empty table: nothing to select between
        }

        // asset lod -> index into out.table.levels, -1 for a level not resident.
        std::vector<int> levelIndex(static_cast<std::size_t>(info.lodLevels), -1);
        // asset lod -> prefix sums over that level's chunk counts, in the order
        // describe() lists them, which is the order load() decodes them.
        std::vector<std::vector<std::size_t>> chunkBase(static_cast<std::size_t>(info.lodLevels));

        for (int l = 0; l < info.lodLevels; l += 2) {

            SplatData d = SogLoader::load(path, {l});
            LodLevel lvl;
            lvl.lod   = l;
            lvl.base  = out.data.count();
            lvl.count = d.count();
            std::size_t within = 0;
            auto& bases = chunkBase[static_cast<std::size_t>(l)];
            for (const auto& ch : info.levels[static_cast<std::size_t>(l)].chunks) {
                bases.push_back(within);
                lvl.chunks.push_back({lvl.base + within, ch.count, ch.bound});
                within += ch.count;
            }
            // describe() and load() walk the same chunk list in the same order,
            // so the prefix sum must land exactly on the level's total; a
            // mismatch means the table would misaddress every later chunk.
            if (within != lvl.count)
                throw std::runtime_error("SOG level " + std::to_string(l) +
                                         ": chunk counts sum to " + std::to_string(within) +
                                         " but the level holds " + std::to_string(lvl.count));

            if (out.data.count() == 0) {
                out.data = std::move(d);
            } else if (d.shDegree != out.data.shDegree) {
                // Concatenation is only valid at one SH stride; a level that
                // disagrees would silently shear every coefficient after it.
                throw std::runtime_error("SOG level " + std::to_string(l) + " has SH degree " +
                                         std::to_string(d.shDegree) + " but level 0 has " +
                                         std::to_string(out.data.shDegree));
            } else {
                out.data.means.insert(out.data.means.end(), d.means.begin(), d.means.end());
                out.data.scales.insert(out.data.scales.end(), d.scales.begin(), d.scales.end());
                out.data.rotations.insert(out.data.rotations.end(), d.rotations.begin(),
                                          d.rotations.end());
                out.data.opacities.insert(out.data.opacities.end(), d.opacities.begin(),
                                          d.opacities.end());
                out.data.sh.insert(out.data.sh.end(), d.sh.begin(), d.sh.end());
            }
            levelIndex[static_cast<std::size_t>(l)] = static_cast<int>(out.table.levels.size());
            out.table.levels.push_back(std::move(lvl));
        }

        // ── The tree ─────────────────────────────────────────────────────────
        // Every leaf's per-level range, resolved to an absolute index by the
        // same prefix sum the chunk table uses, and CHECKED to tile its chunk:
        // sorted by offset, one chunk's leaves must run 0..count with no gap
        // and no overlap. They do on every asset seen (33/9/3/1 chunks of the
        // calico scan, exact), and a tree that only nearly tiled would render a
        // subset and look merely thin, which is the failure this refuses to
        // ship silently.
        if (!info.nodes.empty()) {

            // chunk -> the leaf ranges landing in it, for the tiling check.
            std::vector<std::vector<std::pair<std::size_t, std::size_t>>> perChunk(
                    static_cast<std::size_t>(info.lodLevels));
            std::vector<std::vector<std::size_t>> perChunkOf(static_cast<std::size_t>(info.lodLevels));

            out.table.nodes.reserve(info.nodes.size());
            for (const auto& n : info.nodes) {

                LodNode node;
                node.bound = n.bound;
                node.center = n.bound.getCenter();
                node.radius = std::max(0.5f * n.bound.getSize().length(), 1e-4f);

                for (const auto& r : n.lods) {

                    const auto li = levelIndex[static_cast<std::size_t>(r.lod)];
                    if (li < 0) continue;// not a resident level
                    if (r.count == 0) continue;

                    const auto& lvl = out.table.levels[static_cast<std::size_t>(li)];
                    const auto base = lvl.base + chunkBase[static_cast<std::size_t>(r.lod)][r.chunk];
                    node.ranges.push_back({li, base + r.offset, r.count});

                    perChunk[static_cast<std::size_t>(r.lod)].emplace_back(r.offset, r.count);
                    perChunkOf[static_cast<std::size_t>(r.lod)].push_back(r.chunk);
                }
                if (!node.ranges.empty()) out.table.nodes.push_back(std::move(node));
            }

            for (int l = 0; l < info.lodLevels; l += 2) {

                const auto ll = static_cast<std::size_t>(l);
                const auto& chunks = info.levels[ll].chunks;
                for (std::size_t c = 0; c < chunks.size(); ++c) {

                    std::vector<std::pair<std::size_t, std::size_t>> mine;
                    for (std::size_t k = 0; k < perChunk[ll].size(); ++k)
                        if (perChunkOf[ll][k] == c) mine.push_back(perChunk[ll][k]);

                    std::sort(mine.begin(), mine.end());
                    std::size_t at = 0;
                    for (const auto& [off, cnt] : mine) {

                        if (off != at) {

                            throw std::runtime_error(
                                    "SOG level " + std::to_string(l) + " chunk " + std::to_string(c) +
                                    ": tree leaves do not tile it — expected offset " +
                                    std::to_string(at) + ", found " + std::to_string(off));
                        }
                        at += cnt;
                    }
                    if (at != chunks[c].count) {

                        throw std::runtime_error("SOG level " + std::to_string(l) + " chunk " +
                                                 std::to_string(c) + ": tree leaves cover " +
                                                 std::to_string(at) + " of " +
                                                 std::to_string(chunks[c].count) + " splats");
                    }
                }
            }
        }

        // The percentile footprint, over the finest level only (the coarser
        // copies of the same scene would just be duplicate samples). Median
        // centre per component, p90 radius about it — the same estimator the
        // example's framing has always used, robust to the outliers the chunk
        // bounds are not.
        const auto& l0 = out.table.levels.front();
        const auto n = l0.count;
        std::vector<float> xs(n), ys(n), zs(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto& m = out.data.means[l0.base + i];
            xs[i] = m.x; ys[i] = m.y; zs[i] = m.z;
        }
        auto median = [](std::vector<float>& v) {
            std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
            return v[v.size() / 2];
        };
        out.table.center.set(median(xs), median(ys), median(zs));
        std::vector<float> radii(n);
        for (std::size_t i = 0; i < n; ++i)
            radii[i] = out.data.means[l0.base + i].distanceTo(out.table.center);
        const auto p90 = static_cast<std::ptrdiff_t>(static_cast<double>(n) * 0.90);
        std::nth_element(radii.begin(), radii.begin() + p90, radii.end());
        out.table.radius = std::max(radii[static_cast<std::size_t>(p90)], 1e-3f);

        return out;
    }

}// namespace threepp::splats
