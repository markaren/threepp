// Build a large grass field as a grid of independent GrassMesh tiles for the
// Vulkan renderer, so the renderer can frustum/occlusion-cull the off-screen
// tiles and freeze the wind on the far ones. A single merged GrassMesh (see
// GrassMesh.hpp / the fjord example's makeGrassGeometry) scales its per-frame
// cost — the wind dispatch and the BLAS refit — linearly with area and is
// exempt from culling (one giant AABB is almost always on screen). Splitting
// the field into KxK world-space tiles turns that into a per-tile cost the
// renderer can skip when a tile is off screen (raster) or out of range (wind).
//
// Sway continuity across tile seams is free: the wind phase in grass_wind.comp
// derives from each blade's WORLD XZ rest position, and this helper bakes every
// blade in world/scene space exactly as the single-mesh path does — so two
// blades either side of a tile boundary compute the same phase and there is no
// visible seam. Each tile is otherwise a standalone GrassMesh (its own merged
// geometry + BLAS + one TLAS instance).
//
// Backends: the wind + per-tile culling/freeze are Vulkan-path features. On the
// GL / WebGPU raster backends a GrassMesh renders as a plain static Mesh, so the
// tiles simply draw as static grass there (no wind, no per-tile freeze).

#ifndef THREEPP_GRASSTILES_HPP
#define THREEPP_GRASSTILES_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/GrassMesh.hpp"

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace threepp::vegetation {

    // One placed blade: its base position (world/scene space), a non-uniform
    // scale (x/z = width, y = height), and an orientation (typically a yaw about
    // +Y, but any rotation is honoured).
    struct GrassBlade {
        Vector3 position;
        Vector3 scale{1.f, 1.f, 1.f};
        Quaternion yaw;
    };

    // Look of a single blade — a tapered quad-strip with a bottom→top colour
    // gradient. Defaults match the fjord meadow so the tiled path is a drop-in
    // replacement for its makeGrassGeometry().
    struct GrassBladeStyle {
        int segments = 4;                        // blade tessellation
        float halfWidthBase = 0.05f;             // half-width at the base (tapers to 0 at tip)
        Vector3 bottomColor{0.055f, 0.11f, 0.035f};
        Vector3 topColor{0.19f, 0.30f, 0.10f};
    };

    // Merge a set of blades into ONE BufferGeometry carrying the attributes the
    // wind path needs: position (rest pose, world space), normal, uv, color, and
    // the custom per-vertex float "heightFrac" (0 at a blade's base, 1 at its
    // tip) that grass_wind.comp reads to weight the sway. This is the exact bake
    // the fjord example used for its single merged mesh, factored out for reuse.
    inline std::shared_ptr<BufferGeometry> buildGrassGeometry(
            const std::vector<GrassBlade>& blades, const GrassBladeStyle& style = {}) {
        const int seg = style.segments < 1 ? 1 : style.segments;
        const float wBase = style.halfWidthBase;
        const Vector3& bottom = style.bottomColor;
        const Vector3& top = style.topColor;

        // Blade template in local space (base at origin, unit height along +Y).
        struct V {
            Vector3 p, n;
            float u, vy;
            Vector3 c;
        };
        std::vector<V> tmpl;
        std::vector<unsigned int> tidx;
        for (int i = 0; i <= seg; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(seg);
            const float w = wBase * (1.f - t);// taper to a point at the tip
            const Vector3 c{bottom.x + (top.x - bottom.x) * t, bottom.y + (top.y - bottom.y) * t,
                            bottom.z + (top.z - bottom.z) * t};
            for (int s = 0; s < 2; ++s)
                tmpl.push_back({Vector3{(s == 0 ? -w : w), t, 0.f}, Vector3{0.f, 0.85f, 0.53f},
                                (s == 0 ? 0.f : 1.f), t, c});
        }
        for (int i = 0; i < seg; ++i) {
            const auto a = static_cast<unsigned int>(i * 2);
            tidx.insert(tidx.end(), {a, a + 1u, a + 2u, a + 1u, a + 3u, a + 2u});
        }

        std::vector<float> pos, nrm, uv, col, hfrac;
        std::vector<unsigned int> idx;
        pos.reserve(blades.size() * tmpl.size() * 3);
        Matrix4 m;
        for (const auto& bl : blades) {
            const auto base = static_cast<unsigned int>(pos.size() / 3);
            m.compose(bl.position, bl.yaw, bl.scale);
            for (const auto& tv : tmpl) {
                Vector3 p = tv.p;
                p.applyMatrix4(m);
                Vector3 n = tv.n;
                n.applyQuaternion(bl.yaw);
                n.normalize();
                pos.insert(pos.end(), {p.x, p.y, p.z});
                nrm.insert(nrm.end(), {n.x, n.y, n.z});
                uv.insert(uv.end(), {tv.u, tv.vy});
                col.insert(col.end(), {tv.c.x, tv.c.y, tv.c.z});
                hfrac.push_back(tv.vy);
            }
            for (unsigned int t : tidx) idx.push_back(base + t);
        }

        auto geo = BufferGeometry::create();
        geo->setIndex(idx);
        geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
        geo->setAttribute("color", FloatBufferAttribute::create(col, 3));
        geo->setAttribute("heightFrac", FloatBufferAttribute::create(hfrac, 1));
        return geo;
    }

    // Bucket blades into a world grid of `tileSize`-metre cells (by their XZ base
    // position) and build one GrassMesh per NON-EMPTY cell. Empty cells produce
    // no mesh. Every tile shares `material` and receives a copy of `meshParams`
    // (windDir / windStrength / maxAnimDistance), so per-frame wind updates just
    // iterate the returned vector and set params.time on each. For the valley a
    // tileSize giving ~16–128 non-empty tiles is a good range: enough to cull
    // meaningfully, not so many that the per-tile TLAS-instance / draw overhead
    // dominates.
    //
    // Callers that want distance-gated wind freeze set meshParams.maxAnimDistance
    // (e.g. a couple of tile diagonals); the renderer then freezes tiles beyond
    // that range while keeping them in the TLAS for shadows/reflections.
    inline std::vector<std::shared_ptr<GrassMesh>> buildGrassTiles(
            const std::vector<GrassBlade>& blades, float tileSize,
            const std::shared_ptr<Material>& material,
            const GrassMesh::Params& meshParams = {}, const GrassBladeStyle& style = {}) {
        std::vector<std::shared_ptr<GrassMesh>> tiles;
        if (blades.empty() || tileSize <= 0.f) return tiles;

        // Group blade indices by integer grid cell (floor division so negative
        // world coordinates bucket correctly).
        std::map<std::pair<int, int>, std::vector<const GrassBlade*>> cells;
        for (const auto& bl : blades) {
            const int cx = static_cast<int>(std::floor(bl.position.x / tileSize));
            const int cz = static_cast<int>(std::floor(bl.position.z / tileSize));
            cells[{cx, cz}].push_back(&bl);
        }

        tiles.reserve(cells.size());
        for (auto& [cell, ptrs] : cells) {
            std::vector<GrassBlade> tileBlades;
            tileBlades.reserve(ptrs.size());
            for (const auto* p : ptrs) tileBlades.push_back(*p);

            auto tile = GrassMesh::create(buildGrassGeometry(tileBlades, style), material);
            tile->params = meshParams;
            tile->name = "grass_tile_" + std::to_string(cell.first) + "_" + std::to_string(cell.second);
            tiles.push_back(std::move(tile));
        }
        return tiles;
    }

}// namespace threepp::vegetation

#endif// THREEPP_GRASSTILES_HPP
