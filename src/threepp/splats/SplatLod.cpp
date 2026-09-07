
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

    int selectLod(SplatCloud& cloud, LodTable& table, const Camera& camera,
                  int viewportHeightPx, float targetSplatsPerPixel, float hysteresis) {

        if (table.empty()) return 0;
        const auto nLevels = static_cast<int>(table.levels.size());
        table.heldLevel = std::clamp(table.heldLevel, 0, nLevels - 1);

        // The cloud's PERCENTILE footprint from the table (see the field's
        // comment for why never the chunk bounds), taken into world space.
        Sphere sphere(table.center, table.radius);
        sphere.applyMatrix4(*cloud.matrixWorld);

        // Pixels per world unit at the sphere's centre. An orthographic camera
        // has no distance falloff; a perspective one does. zoom participates in
        // both, which is what makes the editor's scroll-zoom drive the policy.
        float pxPerUnit;
        if (const auto* pc = dynamic_cast<const PerspectiveCamera*>(&camera)) {
            const auto camPos = Vector3().setFromMatrixPosition(*camera.matrixWorld);
            const float dist = std::max(camPos.distanceTo(sphere.center), 1e-3f);
            pxPerUnit = 0.5f * static_cast<float>(viewportHeightPx) /
                        (std::tan(math::degToRad(pc->fov * 0.5f / std::max(pc->zoom, 1e-3f))) * dist);
        } else {
            // Orthographic: projectionMatrix[5] = 2/(top-bottom) * zoom.
            pxPerUnit = 0.5f * static_cast<float>(viewportHeightPx) *
                        camera.projectionMatrix.elements[5] * 0.5f;
        }
        const float rPx  = sphere.radius * pxPerUnit;
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
        cloud.setSubmitRanges(std::move(ranges));
        return table.heldLevel;
    }

    SogLodResult loadSogWithLod(const std::filesystem::path& path) {

        SogLodResult out;
        const auto info = SogLoader::describe(path);
        if (info.lodLevels <= 1) {
            out.data = SogLoader::load(path);
            return out;// empty table: nothing to select between
        }

        for (int l = 0; l < info.lodLevels; l += 2) {

            SplatData d = SogLoader::load(path, {l});
            LodLevel lvl;
            lvl.lod   = l;
            lvl.base  = out.data.count();
            lvl.count = d.count();
            std::size_t within = 0;
            for (const auto& ch : info.levels[static_cast<std::size_t>(l)].chunks) {
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
            out.table.levels.push_back(std::move(lvl));
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
