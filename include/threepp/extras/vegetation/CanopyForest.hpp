// CANOPY FOREST — plant instanced trees where a canopy height model says forest
// stands, at the height it measured.
//
// A lidar surface model minus the terrain model (DOM - DTM) is a canopy height
// model (CHM): metres of vegetation above the ground, per cell. That is not a
// hint about where forest *could* grow — it is a measurement of where forest IS,
// and how tall it is. Scattering trees by slope/elevation rules invents a forest;
// reading the CHM reproduces the one that was flown.
//
// Three steps, each usable on its own:
//
//   detectTreeSites()      CHM local maxima (a crown is a bump) → (x, z, height),
//                          thinned so crowns of the found heights can coexist.
//   makeForestTreeVariant()  one prototype: trunk + leaf geometry, bark + leaf
//                          materials, and the prototype's own height so an
//                          instance can be scaled to the measured canopy value.
//   buildCanopyForest()    sites × species rules × variants → InstancedMesh pairs
//                          (trunks, leaves) under a parent, two LOD tiers.
//
// The height function passed to the builder MUST be the same one the terrain
// tiles bake from (the provider's `height`), not the raw DEM: a provider that
// adds cliff relief or trenches roads moves the surface the trees have to stand
// on, and a base sampled from the raw grid then floats or sinks by that delta.
//
// No terrain dependency beyond terrain::HeightGrid (read-only), so this header
// works with any float grid: a DEM pack, a procedural field, or a synthetic test
// grid.

#ifndef THREEPP_CANOPYFOREST_HPP
#define THREEPP_CANOPYFOREST_HPP

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"
#include "threepp/extras/vegetation/TreeGenerator.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"
#include "threepp/math/Vector3.hpp"
#include "threepp/objects/InstancedMesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

namespace threepp::vegetation {

    // ── Sites ───────────────────────────────────────────────────────────────

    struct TreeSite {
        float x = 0.f, z = 0.f;    // world position (the grid's own frame)
        float canopyHeight = 0.f;  // metres of vegetation at this cell (CHM value)
    };

    struct CanopySiteOptions {
        // A cell is a crown apex when it is the maximum of a (2·windowRadius+1)²
        // window. 5×5 on a 1 m grid ⇒ crowns no closer than ~2 m before thinning.
        int windowRadius = 2;
        float minHeight = 2.5f;// below this the CHM is scrub, noise, or a wall edge

        // Crown radius as a fraction of tree height, and how much of the summed
        // radii two neighbours may overlap. Spruce crowns run ~0.2·H wide; letting
        // them overlap 30% keeps a closed canopy without stacking trunks.
        float crownRadiusFactor = 0.2f;
        float spacingFactor = 0.7f;

        // DOM - DTM on a near-vertical face is largely REGISTRATION NOISE: the two
        // models disagree by metres horizontally, and on a 70° wall a 1 m
        // horizontal shift reads as 3 m of "canopy". Gate on the DEM's own slope.
        float maxSlopeDeg = 65.f;
        float slopeEpsilon = 3.f;// metres, central-difference arm for the slope

        // Ground gate: the sea sheet and any carved bathymetry sit at/below this.
        float seaLevel = 0.f;
        float minGroundHeight = 0.5f;// metres above seaLevel

        // Region of interest (world units). A 4 km pack is 16 M cells; a shot only
        // ever looks at part of it, and the instance budget is spent there.
        float centerX = 0.f, centerZ = 0.f;
        float halfExtent = 1e9f;
    };

    // Local maxima of `canopy`, gated by `dem` slope/height, thinned by crown size.
    // Both grids must share the same frame (a pack's CHM and DEM do by
    // construction). Pure: neither grid is modified.
    inline std::vector<TreeSite> detectTreeSites(const terrain::HeightGrid& canopy,
                                                 const terrain::HeightGrid& dem,
                                                 const CanopySiteOptions& o) {
        std::vector<TreeSite> out;
        if (!canopy.valid() || !dem.valid()) return out;

        const int dim = canopy.dim();
        const float world = canopy.worldSize();
        const float step = world / static_cast<float>(dim - 1);
        const float half = world * 0.5f;
        // Pack grids are centred on the pack origin (GeoTerrainPack contract), and
        // HeightGrid does not expose its centre; the caller's world frame is that
        // frame, so index → world is the plain centred mapping.
        const auto gx = [&](int ix) { return -half + static_cast<float>(ix) * step; };
        const auto gz = [&](int iz) { return -half + static_cast<float>(iz) * step; };

        const auto& c = canopy.data();
        const auto at = [&](int ix, int iz) { return c[static_cast<size_t>(iz) * dim + ix]; };

        // Restrict the scan to the region of interest, in index space.
        const int r = std::max(1, o.windowRadius);
        const auto clampIx = [&](float v) {
            return std::clamp(static_cast<int>(std::floor((v + half) / step)), r, dim - 1 - r);
        };
        const int ix0 = clampIx(o.centerX - o.halfExtent), ix1 = clampIx(o.centerX + o.halfExtent);
        const int iz0 = clampIx(o.centerZ - o.halfExtent), iz1 = clampIx(o.centerZ + o.halfExtent);

        const float cosMax = std::cos(o.maxSlopeDeg * math::DEG2RAD);

        std::vector<TreeSite> peaks;
        peaks.reserve(4096);
        for (int iz = iz0; iz <= iz1; ++iz) {
            for (int ix = ix0; ix <= ix1; ++ix) {
                const float h = at(ix, iz);
                if (h < o.minHeight) continue;
                // Strict on the already-visited half of the window, non-strict on
                // the rest: a flat plateau of equal values then yields exactly one
                // peak (its first cell) instead of one per cell.
                bool isMax = true;
                for (int dz = -r; dz <= r && isMax; ++dz)
                    for (int dx = -r; dx <= r; ++dx) {
                        if (dx == 0 && dz == 0) continue;
                        const float n = at(ix + dx, iz + dz);
                        const bool earlier = dz < 0 || (dz == 0 && dx < 0);
                        if (earlier ? n >= h : n > h) {
                            isMax = false;
                            break;
                        }
                    }
                if (!isMax) continue;

                const float x = gx(ix), z = gz(iz);
                if (dem.slopeNy(x, z, o.slopeEpsilon) < cosMax) continue;// too steep to trust
                if (dem.sampleBilinear(x, z) < o.seaLevel + o.minGroundHeight) continue;
                peaks.push_back({x, z, h});
            }
        }
        if (peaks.empty()) return out;

        // Thin tallest-first: a big crown wins the space, and a small tree under it
        // is what the CHM would have registered as one crown anyway.
        std::sort(peaks.begin(), peaks.end(),
                  [](const TreeSite& a, const TreeSite& b) { return a.canopyHeight > b.canopyHeight; });

        float maxH = peaks.front().canopyHeight;
        const float cell = std::max(4.f, o.spacingFactor * 2.f * o.crownRadiusFactor * maxH);
        // With cell ≥ the largest possible spacing, a 3×3 neighbourhood of the hash
        // is a complete rejection test.
        std::unordered_map<std::int64_t, std::vector<int>> bins;
        bins.reserve(peaks.size());
        const auto key = [](int bx, int bz) {
            return (static_cast<std::int64_t>(bx) << 32) ^ static_cast<std::uint32_t>(bz);
        };

        out.reserve(peaks.size());
        for (const auto& p : peaks) {
            const int bx = static_cast<int>(std::floor(p.x / cell));
            const int bz = static_cast<int>(std::floor(p.z / cell));
            bool ok = true;
            for (int dz = -1; dz <= 1 && ok; ++dz)
                for (int dx = -1; dx <= 1 && ok; ++dx) {
                    auto it = bins.find(key(bx + dx, bz + dz));
                    if (it == bins.end()) continue;
                    for (int idx : it->second) {
                        const TreeSite& q = out[static_cast<size_t>(idx)];
                        const float need = o.spacingFactor * o.crownRadiusFactor *
                                           (p.canopyHeight + q.canopyHeight);
                        const float ddx = p.x - q.x, ddz = p.z - q.z;
                        if (ddx * ddx + ddz * ddz < need * need) {
                            ok = false;
                            break;
                        }
                    }
                }
            if (!ok) continue;
            bins[key(bx, bz)].push_back(static_cast<int>(out.size()));
            out.push_back(p);
        }
        return out;
    }

    // ── Prototypes ──────────────────────────────────────────────────────────

    enum class TreeSpecies {
        ScrubBirch = 0,// low bright thicket on benches and the shoreline
        Birch = 1,     // broadleaf, bright green, low ground
        Spruce = 2,    // conifer, dark, higher ground
    };

    struct TreeVariant {
        std::shared_ptr<BufferGeometry> trunkGeo;
        std::shared_ptr<BufferGeometry> leafGeo;
        std::shared_ptr<MeshStandardMaterial> barkMat;
        std::shared_ptr<MeshStandardMaterial> leafMat;
        // Prototype height in metres, measured from the geometry — the divisor
        // that turns a measured canopy height into an instance scale.
        float height = 10.f;
    };

    namespace detail {

        inline float geometryTopY(const std::shared_ptr<BufferGeometry>& g) {
            if (!g) return 0.f;
            g->computeBoundingBox();
            return g->boundingBox ? g->boundingBox->max().y : 0.f;
        }

    }// namespace detail

    // One prototype. `cheapBlob` swaps the card/frond canopy for low-poly puffs —
    // the distance tier, where a card atlas costs alpha-test bandwidth for
    // sub-pixel leaves nobody can resolve.
    inline TreeVariant makeForestTreeVariant(TreeSpecies sp, unsigned int seed, bool cheapBlob) {
        TreeParams tp;
        applyPreset(sp == TreeSpecies::Spruce ? 1 : 2, tp);// 1 = Norway spruce, 2 = birch
        tp.seed = seed;

        if (sp == TreeSpecies::Spruce && !cheapBlob) {
            // Slim serrated conifer: height ≈ 12.8 m, width ≈ 3.8 m, whorl shelves
            // close enough to overlap over the bole (the fjord-demo silhouette).
            tp.trunkHeight = 1.8f;
            tp.trunkRadius = 0.22f;
            tp.crownRadiusX = tp.crownRadiusZ = 1.9f;
            tp.crownHeight = 11.f;
            tp.whorlSpacing = 0.72f;
            tp.branchesPerWhorl = 5;
            tp.branchDroop = 0.44f;
            tp.branchTipUpturn = 0.42f;
            tp.crownProfileExponent = 1.25f;
            tp.sideTwigDensity = 0.6f;
            tp.leafSize = 0.75f;
            tp.leafDensity = 0.92f;
            tp.leafClumping = 0.f;
            tp.leafColor = {0.11f, 0.28f, 0.09f};// darker than the birches
        }
        if (sp != TreeSpecies::Spruce) {
            tp.barkColor = {0.72f, 0.71f, 0.67f};// mute the preset's pure white
            tp.leafDensity = 0.95f;
            tp.leafClumping = 0.35f;
        }
        if (sp == TreeSpecies::ScrubBirch) {
            // Shoreline/bench thicket: the bright yellow-green band the reference
            // has under the darker conifer wall. Short and wide, so the 0.35-2.2
            // instance scale lands its 2.5-6 m CHM values near 1.
            tp.trunkHeight = 1.0f;
            tp.crownRadiusX = tp.crownRadiusZ = 1.6f;
            tp.crownHeight = 3.0f;
            tp.leafColor = {0.42f, 0.62f, 0.16f};
        }
        if (cheapBlob) {
            // Distance silhouette: space colonisation + blob puffs. Whorl+frond
            // would balloon node and card counts for a canopy that is 3 px tall.
            tp.branchingMode = BranchingMode::Colonise;
            tp.crownShape = CrownShape::Cone;
            tp.trunkHeight = sp == TreeSpecies::ScrubBirch ? 1.2f : 3.5f;
            // Wider than the near prototype: at ~1 km a 200 stems/ha stand only
            // closes into one canopy mass if the crowns actually touch, and a
            // blob puff has no twig fringe to bridge the gap for it.
            tp.crownRadiusX = tp.crownRadiusZ = sp == TreeSpecies::ScrubBirch ? 1.8f : 2.35f;
            tp.crownHeight = sp == TreeSpecies::ScrubBirch ? 3.0f : 7.0f;
            tp.influenceDistance = 3.5f;
            tp.killDistance = 0.7f;
            tp.segmentLength = 0.45f;
            tp.maxIterations = 200;
            tp.tropism = -0.04f;
            tp.leafStyle = LeafStyle::Blob;
            // Several SMALL puffs per node: two big spheres merge into one smooth
            // dome, and the cone profile stops showing through.
            tp.leavesPerCluster = 3;
            tp.leafSize = sp == TreeSpecies::ScrubBirch ? 0.62f : 0.92f;
            tp.attractorCount = 320;
            tp.radialSegments = 5;
            if (sp == TreeSpecies::Spruce) tp.crownShape = CrownShape::Cone;
        }

        TreeGenerator gen(seed);
        gen.buildSkeleton(tp);

        TreeVariant v;
        v.trunkGeo = gen.makeTrunkGeometry(tp);
        v.leafGeo = gen.makeLeafGeometry(tp);
        v.height = std::max(1.f, std::max(detail::geometryTopY(v.trunkGeo),
                                          detail::geometryTopY(v.leafGeo)));

        auto bark = makeBarkTextures(cheapBlob ? 128 : 256, seed, tp.barkColor, tp.barkStyle);
        bark.first->repeat.set(3.f, 0.5f);
        bark.second->repeat.set(3.f, 0.5f);
        v.barkMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.92f).metalness(0.f));
        v.barkMat->map = bark.first;
        v.barkMat->normalMap = bark.second;
        v.barkMat->vertexColors = true;// twig darkening, baked per-vertex

        v.leafMat = MeshStandardMaterial::create(
                MeshStandardMaterial::Params{}.color(Color::white).roughness(0.85f).metalness(0.f));
        // Backlit canopies glow through instead of going flat-dark (deferred).
        v.leafMat->translucencyColor = Color(0.50f, 0.80f, 0.28f);
        v.leafMat->translucency = cheapBlob ? 0.35f : 0.5f;
        if (cheapBlob) {
            // leafColor is an sRGB hint; material->color is LINEAR working space.
            // Without the conversion the blobs render ~4× too bright and read
            // "always lit".
            v.leafMat->color = Color(tp.leafColor[0], tp.leafColor[1], tp.leafColor[2])
                                       .convertSRGBToLinear();
            v.leafMat->vertexColors = true;// canopy tint gradient baked per-vertex
        } else {
            // The atlas grid must be the one the cards were UV'd for
            // (TreeParams::leafAtlasCells) or every card samples the wrong cell.
            v.leafMat->map = (tp.leafStyle == LeafStyle::Frond)
                    ? makeNeedleFrondTexture(256, seed, tp.leafColor, tp.leafAtlasCells)
                    : makeLeafClusterTexture(256, seed, tp.leafColor, tp.leafShape, 8, tp.leafAtlasCells);
            v.leafMat->alphaTest = kLeafAlphaTest;
            v.leafMat->side = Side::Double;
            v.leafMat->vertexColors = true;
        }
        return v;
    }

    // ── Builder ─────────────────────────────────────────────────────────────

    struct SpeciesVariants {
        std::vector<TreeVariant> near;// card/frond canopies
        std::vector<TreeVariant> far; // blob canopies
    };

    struct ForestOptions {
        Vector3 cameraPos;             // tier split is measured from here
        float nearDistance = 380.f;    // beyond this, blob canopies

        // Species rule (this pack): scrub below `scrubMaxHeight`, birch below
        // `birchMaxHeight` AND below `birchMaxElevation`, spruce otherwise.
        float scrubMaxHeight = 6.f;
        float birchMaxHeight = 14.f;
        float birchMaxElevation = 350.f;

        // The CHM measures the canopy TOP; the instance is scaled so the
        // prototype's own height matches it. Clamped: a 0.1× tree is a bush with
        // 12 m of detail in it, and a 4× tree is a redwood on a fjord bench.
        float minScale = 0.35f, maxScale = 2.2f;

        // Sink the base so the downhill half of a trunk flare does not float off a
        // slope (the base is a point sample of a surface that keeps dropping).
        float sink = 0.4f;

        int cap = 40000;// instance budget; sites are shuffled before the trim
        unsigned int seed = 20260904u;
    };

    struct ForestStats {
        int sites = 0;    // sites offered
        int planted = 0;  // instances actually emitted
        int nearTier = 0; // of which card/frond
        int farTier = 0;  // of which blob
        int meshes = 0;   // InstancedMesh objects added (2 per non-empty bucket)
    };

    // Emits one InstancedMesh pair (trunks, leaves) per (species, tier, variant)
    // bucket under `parent`. `heightFn` must be the terrain provider's height.
    inline ForestStats buildCanopyForest(Object3D& parent,
                                         const std::vector<TreeSite>& sites,
                                         const std::array<SpeciesVariants, 3>& species,
                                         const std::function<float(float, float)>& heightFn,
                                         const ForestOptions& o) {
        ForestStats st;
        st.sites = static_cast<int>(sites.size());
        if (sites.empty() || !heightFn) return st;

        std::mt19937 rng(o.seed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);

        std::vector<int> order(sites.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
        if (static_cast<int>(order.size()) > o.cap) {
            // Shuffle then trim: a scan-order cap spends the whole budget on the
            // first rows of the grid and leaves the rest of the wall bare.
            std::shuffle(order.begin(), order.end(), rng);
            order.resize(static_cast<size_t>(o.cap));
        }

        // [species][tier][variant] → transforms
        std::array<std::array<std::vector<std::vector<Matrix4>>, 2>, 3> buckets;
        for (int s = 0; s < 3; ++s) {
            buckets[s][0].resize(species[s].near.size());
            buckets[s][1].resize(species[s].far.size());
        }

        Quaternion q;
        const Vector3 up{0.f, 1.f, 0.f};
        for (int idx : order) {
            const TreeSite& p = sites[static_cast<size_t>(idx)];
            const float y = heightFn(p.x, p.z);

            int s;
            if (p.canopyHeight < o.scrubMaxHeight) s = static_cast<int>(TreeSpecies::ScrubBirch);
            else if (p.canopyHeight < o.birchMaxHeight && y < o.birchMaxElevation)
                s = static_cast<int>(TreeSpecies::Birch);
            else
                s = static_cast<int>(TreeSpecies::Spruce);

            const float dx = p.x - o.cameraPos.x, dy = y - o.cameraPos.y, dz = p.z - o.cameraPos.z;
            const bool far = (dx * dx + dy * dy + dz * dz) > o.nearDistance * o.nearDistance;
            int tier = far ? 1 : 0;
            const auto& vars = tier == 0 ? species[s].near : species[s].far;
            if (vars.empty()) {// fall back to whichever tier this species has
                tier = 1 - tier;
                if ((tier == 0 ? species[s].near : species[s].far).empty()) continue;
            }
            const auto& use = tier == 0 ? species[s].near : species[s].far;

            const size_t vi = static_cast<size_t>(u01(rng) * static_cast<float>(use.size())) % use.size();
            const float scale = std::clamp(p.canopyHeight / use[vi].height, o.minScale, o.maxScale);
            // Trunks stay VERTICAL. Trees grow against gravity, not normal to the
            // slope: a leaned trunk on a steep bank is the one thing that reads as
            // "scattered props" from a kilometre out.
            q.setFromAxisAngle(up, u01(rng) * math::TWO_PI);
            Matrix4 m;
            m.compose(Vector3(p.x, y - o.sink, p.z), q, Vector3(scale, scale, scale));
            buckets[s][tier][vi].push_back(m);
            ++st.planted;
            if (tier == 0) ++st.nearTier;
            else
                ++st.farTier;
        }

        for (int s = 0; s < 3; ++s)
            for (int t = 0; t < 2; ++t) {
                const auto& vars = t == 0 ? species[s].near : species[s].far;
                for (size_t vi = 0; vi < buckets[s][t].size(); ++vi) {
                    const auto& xf = buckets[s][t][vi];
                    if (xf.empty()) continue;
                    auto trunks = InstancedMesh::create(vars[vi].trunkGeo, vars[vi].barkMat, xf.size());
                    auto leaves = InstancedMesh::create(vars[vi].leafGeo, vars[vi].leafMat, xf.size());
                    for (size_t i = 0; i < xf.size(); ++i) {
                        trunks->setMatrixAt(i, xf[i]);
                        leaves->setMatrixAt(i, xf[i]);
                    }
                    trunks->instanceMatrix()->needsUpdate();
                    leaves->instanceMatrix()->needsUpdate();
                    parent.add(trunks);
                    parent.add(leaves);
                    st.meshes += 2;
                }
            }
        return st;
    }

}// namespace threepp::vegetation

#endif// THREEPP_CANOPYFOREST_HPP
