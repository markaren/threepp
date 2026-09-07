// Weight-driven ground-cover scatter (grass tufts) for TileTerrain.
//
// Texture structure alone cannot carry the last metres to the camera: at
// grazing angles the world-XZ band projection minifies away, and a texture is
// still flat. What separates "modern game ground" from a painted carpet up
// close is STUFF — physical tufts breaking the silhouette. This helper
// scatters instanced props on a grid of CELLS that follows the camera:
//
//   • placement is driven by the SAME TerrainProvider the tiles bake from —
//     height for position, weights for coverage (tufts on the grass band),
//     local slope as a sanity gate — so props always match the ground they
//     stand on, painted roads excluded automatically (weights are suppressed
//     over pavement);
//   • cells build DETERMINISTICALLY from their coordinates (hash-seeded): the
//     same cell always regrows the same props, no popping differences on
//     revisit;
//   • the radius is deliberately SHORT. Tufts are centimetre-thin geometry —
//     beyond ~60 m they are sub-pixel, and sub-pixel triangles rasterize under
//     a different TAA/DLSS jitter phase every frame: the pixel alternates
//     blade↔ground, the temporal accumulator can never converge, and the
//     distant field sparkles white. Structure past the ring is the band
//     texture's job;
//   • at most maxCellBuildsPerFrame cells build per update() and cell removal
//     has hysteresis, so scene-entry churn stays bounded (cell adds are
//     appends — they do NOT trip the renderer's mid-list-change history
//     clear the way removals do);
//   • one InstancedMesh per cell: instances frustum-cull as a cell unit via
//     an explicit bounding sphere.
//
// (An earlier revision also scattered stones. Removed: a stone placed on the
// PROVIDER surface visibly floats or sinks wherever the rendered tile mesh
// deviates from it, and weight-driven talus kept landing where scree WEIGHT
// is high rather than where scree physically collects — below walls, in
// chutes and fans. Stones need placement derived from the actual landform
// (flow accumulation / wall-base detection) and mesh-conformed heights to
// read right; tufts are forgiving because grass grows anywhere grass-band
// ground is flat-ish and each blade's ground contact is visually fuzzy.)
//
// Usage:
//   auto scatter = terrain::TerrainScatter::create(provider, {});
//   scene.add(scatter);
//   scatter->update(camera.position);   // per frame, same pos as tiles
//
// Header-only, threepp core only. Provider callbacks are invoked on the
// update() thread only.

#ifndef THREEPP_EXTRAS_TERRAIN_TERRAINSCATTER_HPP
#define THREEPP_EXTRAS_TERRAIN_TERRAINSCATTER_HPP

#include "threepp/extras/terrain/TerrainTiles.hpp"// TerrainProvider
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Rng.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace threepp::terrain {

    struct TerrainScatterOptions {
        float cellSize = 40.f;  // metres per scatter cell
        float radius = 55.f;    // SHORT ring — see header (sub-pixel sparkle)
        float removeSlack = 1.3f;// cells drop beyond radius*removeSlack (hysteresis)
        int maxCellBuildsPerFrame = 2;

        unsigned int seed = 20260731u;

        // Per-cell candidate count (survivors depend on local weights/slope).
        int tuftCandidates = 320;

        // Gates. Weight is the provider's GRASS band coverage; density scales
        // with coverage, so sparse transition zones get sparse props instead
        // of a hard cutoff.
        float tuftWeightMin = 0.35f;// grass coverage below this → no tuft
        float tuftSlopeMax = 0.50f;

        // Prop dimensions (uniform-random within range, metres).
        float tuftSizeMin = 0.14f, tuftSizeMax = 0.34f;

        std::array<float, 3> tuftColor = {0.20f, 0.28f, 0.11f};
    };

    class TerrainScatter : public Group {

    public:
        explicit TerrainScatter(TerrainProvider provider, TerrainScatterOptions options = {})
            : provider_(std::move(provider)), o_(options) {
            buildPrototype();
        }

        static std::shared_ptr<TerrainScatter> create(TerrainProvider provider,
                                                      TerrainScatterOptions options = {}) {
            return std::make_shared<TerrainScatter>(std::move(provider), options);
        }

        // Call once per frame with the (primary) camera position.
        void update(const Vector3& camPos) {
            if (!provider_.height) return;
            const float cs = o_.cellSize;
            // Drop cells past the hysteresis ring.
            const float dropR = o_.radius * o_.removeSlack;
            for (auto it = cells_.begin(); it != cells_.end();) {
                const float dx = cellCenter(it->first.first) - camPos.x;
                const float dz = cellCenter(it->first.second) - camPos.z;
                if (dx * dx + dz * dz > dropR * dropR) {
                    if (it->second) remove(*it->second);
                    it = cells_.erase(it);
                } else {
                    ++it;
                }
            }
            // Build the nearest missing cells, bounded per frame.
            const int cx0 = static_cast<int>(std::floor((camPos.x - o_.radius) / cs));
            const int cx1 = static_cast<int>(std::floor((camPos.x + o_.radius) / cs));
            const int cz0 = static_cast<int>(std::floor((camPos.z - o_.radius) / cs));
            const int cz1 = static_cast<int>(std::floor((camPos.z + o_.radius) / cs));
            struct Cand {
                float d2;
                int cx, cz;
            };
            std::vector<Cand> missing;
            for (int cz = cz0; cz <= cz1; ++cz)
                for (int cx = cx0; cx <= cx1; ++cx) {
                    if (cells_.count({cx, cz})) continue;
                    const float dx = cellCenter(cx) - camPos.x;
                    const float dz = cellCenter(cz) - camPos.z;
                    const float d2 = dx * dx + dz * dz;
                    if (d2 <= o_.radius * o_.radius) missing.push_back({d2, cx, cz});
                }
            std::sort(missing.begin(), missing.end(),
                      [](const Cand& a, const Cand& b) { return a.d2 < b.d2; });
            int builds = o_.maxCellBuildsPerFrame;
            for (const auto& c : missing) {
                if (builds-- <= 0) break;
                cells_.emplace(std::make_pair(c.cx, c.cz), buildCell(c.cx, c.cz));
            }
        }

        [[nodiscard]] int activeCells() const { return static_cast<int>(cells_.size()); }

    private:
        // A built cell may legitimately hold no props (wrong band/slope) —
        // presence in the map marks it BUILT so it is never re-scanned.
        using CellMap = std::map<std::pair<int, int>, std::shared_ptr<InstancedMesh>>;

        [[nodiscard]] float cellCenter(int c) const {
            return (static_cast<float>(c) + 0.5f) * o_.cellSize;
        }

        // math::Rng's single-key counter hash; only the composite cell key
        // below stays domain-specific.
        static float hash01(unsigned int x) {
            return math::Rng::hash01(static_cast<std::uint64_t>(x));
        }
        [[nodiscard]] float cellHash(int cx, int cz, int i, int ch) const {
            return hash01(static_cast<unsigned int>(cx * 73856093) ^
                          static_cast<unsigned int>(cz * 19349663) ^
                          static_cast<unsigned int>(i * 83492791) ^
                          (o_.seed + static_cast<unsigned int>(ch) * 0x9e3779b9u));
        }

        void weightsAt(float x, float z, float h, float slope, float* w4) const {
            if (provider_.weights) {
                provider_.weights(x, z, h, slope, w4);
                return;
            }
            // Slope-only fallback when the provider has no band weights.
            w4[0] = std::clamp(1.f - slope / 0.4f, 0.f, 1.f);// "grass"
            w4[1] = w4[2] = w4[3] = 0.f;
        }

        std::shared_ptr<InstancedMesh> buildCell(int cx, int cz) {
            const float cs = o_.cellSize;
            const float x0 = static_cast<float>(cx) * cs;
            const float z0 = static_cast<float>(cz) * cs;
            const float e = 0.6f;// slope probe half-width (m)

            std::vector<Matrix4> mats;
            Matrix4 m;
            Quaternion q;
            Vector3 pos, scl;
            const Vector3 up(0, 1, 0);
            float meanH = 0.f;
            for (int i = 0; i < o_.tuftCandidates; ++i) {
                const float fx = cellHash(cx, cz, i, 0);
                const float fz = cellHash(cx, cz, i, 1);
                const float x = x0 + fx * cs;
                const float z = z0 + fz * cs;
                const float h = provider_.height(x, z);
                const float hx = provider_.height(x + e, z) - provider_.height(x - e, z);
                const float hz = provider_.height(x, z + e) - provider_.height(x, z - e);
                const float ny = (2.f * e) / std::sqrt(hx * hx + hz * hz + 4.f * e * e);
                const float slope = 1.f - ny;

                float w4[4] = {0, 0, 0, 0};
                weightsAt(x, z, h, slope, w4);
                if (slope > o_.tuftSlopeMax || w4[0] < o_.tuftWeightMin) continue;
                // Density ∝ coverage: a candidate survives with probability w.
                if (cellHash(cx, cz, i, 2) > w4[0]) continue;

                const float t = cellHash(cx, cz, i, 3);
                const float s = o_.tuftSizeMin + (o_.tuftSizeMax - o_.tuftSizeMin) * t;
                q.setFromAxisAngle(up, cellHash(cx, cz, i, 4) * math::TWO_PI);
                pos.set(x, h, z);// a tuft's origin is its root
                scl.set(s, s, s);
                m.compose(pos, q, scl);
                mats.push_back(m);
                meanH += h;
            }
            if (mats.empty()) return nullptr;

            const float cellMidH = meanH / static_cast<float>(mats.size());
            auto im = InstancedMesh::create(tuftGeo_, tuftMat_, mats.size());
            for (size_t i = 0; i < mats.size(); ++i) im->setMatrixAt(i, mats[i]);
            im->instanceMatrix()->needsUpdate();
            // Cull as a cell unit: sphere over the cell footprint (props are
            // ≤ ~0.4 m — the +2 m pad covers height spread + prop size).
            im->boundingSphere = Sphere(Vector3(x0 + cs * 0.5f, cellMidH, z0 + cs * 0.5f),
                                        cs * 0.75f + 2.f);
            im->autoLod = false;// sub-metre props: simplification chains cost more than they save
            add(im);
            return im;
        }

        void buildPrototype() {
            // Tuft: a fan of single-triangle blades leaning outward from the
            // root. Blades are deliberately WIDE for their height — thin
            // slivers go sub-pixel a few dozen metres out and sparkle under
            // jittered AA (see the radius note in the header).
            auto geo = BufferGeometry::create();
            std::vector<float> pos;
            std::vector<unsigned int> idx;
            const int blades = 9;
            for (int b = 0; b < blades; ++b) {
                const float ang = (static_cast<float>(b) + 0.5f) / blades * math::TWO_PI;
                const float lean = 0.35f + 0.3f * hash01(static_cast<unsigned int>(b) * 2654435761u + o_.seed);
                const float dx = std::cos(ang), dz = std::sin(ang);
                const float bw = 0.10f;// half-width at the root
                const auto base = static_cast<unsigned int>(pos.size() / 3);
                pos.insert(pos.end(), {dx * 0.05f - dz * bw, 0.f, dz * 0.05f + dx * bw});
                pos.insert(pos.end(), {dx * 0.05f + dz * bw, 0.f, dz * 0.05f - dx * bw});
                pos.insert(pos.end(), {dx * lean, 1.f, dz * lean});// tip
                idx.insert(idx.end(), {base, base + 1, base + 2});
            }
            geo->setAttribute("position", FloatBufferAttribute::create(std::move(pos), 3));
            geo->setIndex(std::move(idx));
            geo->computeVertexNormals();
            geo->computeBoundingSphere();
            tuftGeo_ = geo;
            auto mat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}
                            .color(Color(o_.tuftColor[0], o_.tuftColor[1], o_.tuftColor[2]))
                            .roughness(0.97f)// matte — sun-catching blades read as white specks
                            .metalness(0.f));
            mat->side = Side::Double;
            tuftMat_ = mat;
        }

        TerrainProvider provider_;
        TerrainScatterOptions o_;
        std::shared_ptr<BufferGeometry> tuftGeo_;
        std::shared_ptr<Material> tuftMat_;
        CellMap cells_;
    };

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_TERRAINSCATTER_HPP
