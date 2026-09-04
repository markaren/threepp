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
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/LOD.hpp"
#include "threepp/objects/Mesh.hpp"

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
    //
    // `distant` cheapens the blob AGAIN, and it is the whole reason a canopy-model
    // forest can move the camera. The fjord demo's blob was tuned for ~1000 far
    // trees: 320 attractors × 3 puffs × an 80-triangle UV sphere is tens of
    // thousands of triangles, which is a bargain once and a frame-killer instanced
    // 30 000 times. At 300-800 m a whole crown is 5-15 px, so what has to survive
    // is the OUTLINE and the mass — not the puff count. Fewer, bigger, coarser
    // puffs on a sparser skeleton keep both at a twentieth of the triangles.
    inline TreeVariant makeForestTreeVariant(TreeSpecies sp, unsigned int seed, bool cheapBlob,
                                             bool distant = false) {
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
        if (cheapBlob && distant) {
            // Coarser skeleton: a longer kill distance retires an attractor after
            // fewer segments, so the node count (and with it the puff count, which
            // is what the triangles are) falls roughly linearly.
            tp.attractorCount = 190;
            tp.influenceDistance = 4.2f;
            tp.killDistance = 1.05f;
            tp.segmentLength = 0.6f;
            tp.maxIterations = 130;
            tp.radialSegments = 4;
            tp.leafDensity = 0.78f;
            tp.leavesPerCluster = 2;
            // Slightly bigger puffs so two per tip still close the crown. Pushed
            // much past this the crown loses its SHAPE — measured by looking: at
            // 1.55× on a 110-attractor skeleton a spruce becomes four bright
            // balloons and the conifer silhouette is gone.
            tp.leafSize *= 1.2f;
            tp.blobLatSegs = 3;
            tp.blobLonSegs = 5;// 30 triangles per puff instead of 80
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

    // ── Camera-following cell LOD ───────────────────────────────────────────
    //
    // `buildCanopyForest` above splits the tiers ONCE, against the camera the
    // scene was built with. That is a still-frame optimisation: move the camera
    // and every near tree stays a card atlas at 2 km while every far tree stays a
    // blob at 40 m. It also submits one InstancedMesh spanning the whole region,
    // so nothing is ever skipped.
    //
    // The fix is spatial: cut the sites into cells and give each cell its own
    // `threepp::LOD` node, whose level is chosen per frame from the camera. Three
    // levels, because the interesting range spans two orders of magnitude:
    //
    //   L0  0-300 m    card/frond crowns — the only range where a leaf is a pixel
    //   L1  300-800 m  distant blob puffs (~2-3 k tris) — outline and mass only
    //   L2  > 800 m    ONE canopy surface mesh per cell, built from the CHM
    //
    // L2 is the level that makes the whole thing affordable, and it is not an
    // approximation of the trees — it is the same measurement the trees came
    // from. A canopy height model already IS a surface: ground + CHM, sampled on
    // a few-metre lattice, keeps the crown bumps that give a forest its texture
    // at a kilometre, for a few thousand triangles per 128 m cell instead of a
    // few hundred million. What it must not do is end in mid-air, so every
    // boundary edge of the valid region gets a skirt down toward the ground: seen
    // side-on from across a fjord, a forest edge is a WALL of foliage, and a
    // floating sheet reads instantly as a hack.
    //
    // Hidden levels are `visible = false`, so the renderer's traverseVisible walk
    // never reaches them: no draws, no TLAS instances, no BLAS residency churn.

    struct CanopyMeshOptions {
        float step = 3.f;      // lattice spacing (world units)
        float minCanopy = 2.f; // below this the cell is not forest
        float maxSlopeDeg = 65.f;// same gate the SITES use, or L2 grows forest
        float slopeEpsilon = 3.f;// that L1 does not have and the handoff pops
        float seaLevel = 0.f;
        float minGroundHeight = 0.5f;
        // The CHM's local max is the crown APEX; the surface through the apexes
        // sits a little above the canopy an observer sees between them.
        float topFraction = 0.92f;
        // Metres of foliage wall at a forest edge. Kept SHORT: on a benched wall
        // the 65° gate below already cuts the valid region into tread-wide
        // strips, and a deep skirt under every strip turns the stand into a
        // ladder of bright rungs over dark risers (measured by looking at the
        // 2 km shot with 9 m).
        float skirtDepth = 7.f;
        float uvPeriod = 4.5f;// world metres per leaf-atlas tile
        // Species rule — must be the ForestOptions rule, or L2 recolours the
        // forest at the handoff.
        float scrubMaxHeight = 6.f, birchMaxHeight = 14.f, birchMaxElevation = 350.f;
        // Vertex AO: a vertex this far below its 3×3 neighbourhood maximum is
        // fully buried. Canopy gaps are where a forest gets its value structure.
        float aoDepth = 6.f, aoStrength = 0.45f;
        // The leaf atlas is used as a GRAIN map (neutral, near-white), so the
        // species colour can live in the vertex colour. Its mean is well below 1,
        // hence the gain.
        float texGain = 0.85f;
    };

    // One canopy surface for the world-XZ rectangle [x0,x1) × [z0,z1), with all
    // vertices expressed relative to `origin` (the LOD node's own position).
    // Returns nullptr when the rectangle holds no forest.
    inline std::shared_ptr<BufferGeometry> buildCanopySurface(
            const terrain::HeightGrid& canopy, const terrain::HeightGrid& dem,
            const std::function<float(float, float)>& heightFn,
            const std::array<Color, 3>& speciesColor,
            float x0, float z0, float x1, float z1,
            const Vector3& origin, const CanopyMeshOptions& o) {

        const float s = std::max(0.5f, o.step);
        // Quad indices on a GLOBAL lattice, half-open so a quad belongs to exactly
        // one cell: neighbouring cells share their boundary node positions, hence
        // sample the same CHM there, hence meet without a crack or an overlap.
        const int qi0 = static_cast<int>(std::floor(x0 / s));
        const int qi1 = static_cast<int>(std::floor(x1 / s));
        const int qj0 = static_cast<int>(std::floor(z0 / s));
        const int qj1 = static_cast<int>(std::floor(z1 / s));
        const int nx = qi1 - qi0 + 1, nz = qj1 - qj0 + 1;// nodes
        if (nx < 2 || nz < 2) return nullptr;

        const float cosMax = std::cos(o.maxSlopeDeg * math::DEG2RAD);
        const size_t nn = static_cast<size_t>(nx) * static_cast<size_t>(nz);
        std::vector<float> hCan(nn, 0.f), hGnd(nn, 0.f);
        std::vector<unsigned char> ok(nn, 0u);

        for (int j = 0; j < nz; ++j)
            for (int i = 0; i < nx; ++i) {
                const float x = static_cast<float>(qi0 + i) * s;
                const float z = static_cast<float>(qj0 + j) * s;
                // MAX over the lattice cell, not a point sample: the CHM is 1 m
                // and the crowns ARE its local maxima, so averaging (or picking
                // one texel in nine) throws away exactly the bumps that make the
                // surface read as canopy rather than as a tarpaulin.
                float hc = 0.f;
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dx = -1; dx <= 1; ++dx)
                        hc = std::max(hc, canopy.sampleBilinear(x + static_cast<float>(dx) * s / 3.f,
                                                                z + static_cast<float>(dz) * s / 3.f));
                const float g = heightFn ? heightFn(x, z) : dem.sampleBilinear(x, z);
                hCan[static_cast<size_t>(j) * nx + i] = hc;
                hGnd[static_cast<size_t>(j) * nx + i] = g;
                ok[static_cast<size_t>(j) * nx + i] =
                        (hc >= o.minCanopy && g > o.seaLevel + o.minGroundHeight &&
                         dem.slopeNy(x, z, o.slopeEpsilon) >= cosMax)
                                ? 1u
                                : 0u;
            }

        std::vector<float> positions, normals, uvs, colors;
        std::vector<unsigned int> indices;
        std::vector<int> vidx(nn, -1);// node → emitted vertex, -1 = not emitted

        const auto topY = [&](size_t k) { return hGnd[k] + hCan[k] * o.topFraction; };
        const auto emitTop = [&](int i, int j) {
            const size_t k = static_cast<size_t>(j) * nx + i;
            if (vidx[k] >= 0) return vidx[k];
            const float x = static_cast<float>(qi0 + i) * s;
            const float z = static_cast<float>(qj0 + j) * s;
            const float y = topY(k);
            // Burial AO from the 3×3 neighbourhood: a crown that sits well below
            // its neighbours is in their shade.
            float localMax = hCan[k];
            for (int dj = -1; dj <= 1; ++dj)
                for (int di = -1; di <= 1; ++di) {
                    const int ii = i + di, jj = j + dj;
                    if (ii < 0 || jj < 0 || ii >= nx || jj >= nz) continue;
                    localMax = std::max(localMax, hCan[static_cast<size_t>(jj) * nx + ii]);
                }
            const float ao = 1.f - o.aoStrength * std::clamp((localMax - hCan[k]) / o.aoDepth, 0.f, 1.f);
            int sp;
            if (hCan[k] < o.scrubMaxHeight) sp = 0;
            else if (hCan[k] < o.birchMaxHeight && hGnd[k] < o.birchMaxElevation)
                sp = 1;
            else
                sp = 2;
            const Color& c = speciesColor[static_cast<size_t>(sp)];
            positions.push_back(x - origin.x);
            positions.push_back(y - origin.y);
            positions.push_back(z - origin.z);
            normals.push_back(0.f);
            normals.push_back(1.f);
            normals.push_back(0.f);
            uvs.push_back(x / o.uvPeriod);
            uvs.push_back(z / o.uvPeriod);
            colors.push_back(c.r * ao * o.texGain);
            colors.push_back(c.g * ao * o.texGain);
            colors.push_back(c.b * ao * o.texGain);
            vidx[k] = static_cast<int>(positions.size() / 3) - 1;
            return vidx[k];
        };

        const int qw = nx - 1, qh = nz - 1;
        std::vector<unsigned char> quad(static_cast<size_t>(qw) * static_cast<size_t>(qh), 0u);
        for (int j = 0; j < qh; ++j)
            for (int i = 0; i < qw; ++i) {
                const size_t k = static_cast<size_t>(j) * nx + i;
                if (ok[k] && ok[k + 1] && ok[k + nx] && ok[k + nx + 1])
                    quad[static_cast<size_t>(j) * qw + i] = 1u;
            }
        // Drop ISOLATED and single-file quads. The 65° gate slices a benched
        // wall into tread-wide slivers, and a lone 3 m quad with a skirt under it
        // is a floating shelf — stack a slope's worth of them and the ridge line
        // becomes a staircase of green trays against the sky (looked at, 2 km
        // shot). A quad needs two orthogonal neighbours to be part of a STAND.
        {
            std::vector<unsigned char> keep(quad.size(), 0u);
            for (int j = 0; j < qh; ++j)
                for (int i = 0; i < qw; ++i) {
                    if (!quad[static_cast<size_t>(j) * qw + i]) continue;
                    int n = 0;
                    if (i > 0) n += quad[static_cast<size_t>(j) * qw + i - 1];
                    if (i + 1 < qw) n += quad[static_cast<size_t>(j) * qw + i + 1];
                    if (j > 0) n += quad[static_cast<size_t>(j - 1) * qw + i];
                    if (j + 1 < qh) n += quad[static_cast<size_t>(j + 1) * qw + i];
                    keep[static_cast<size_t>(j) * qw + i] = n >= 2 ? 1u : 0u;
                }
            quad.swap(keep);
        }
        for (int j = 0; j < qh; ++j)
            for (int i = 0; i < qw; ++i) {
                if (!quad[static_cast<size_t>(j) * qw + i]) continue;
                const unsigned int a = static_cast<unsigned int>(emitTop(i, j));
                const unsigned int b = static_cast<unsigned int>(emitTop(i + 1, j));
                const unsigned int c = static_cast<unsigned int>(emitTop(i + 1, j + 1));
                const unsigned int d = static_cast<unsigned int>(emitTop(i, j + 1));
                indices.push_back(a);
                indices.push_back(d);
                indices.push_back(c);
                indices.push_back(a);
                indices.push_back(c);
                indices.push_back(b);
            }
        if (indices.empty()) return nullptr;

        // ── Edge skirt ──────────────────────────────────────────────────────
        // A lattice edge with a quad on exactly one side is the forest boundary.
        // Hang a curtain from it: the canopy top down to (canopy top − skirtDepth)
        // clamped to the ground, so a stand seen from the side is a wall of
        // foliage and not a sheet with daylight under it.
        const auto quadAt = [&](int i, int j) {
            if (i < 0 || j < 0 || i >= qw || j >= qh) return 0;
            return static_cast<int>(quad[static_cast<size_t>(j) * qw + i]);
        };
        const auto emitSkirt = [&](int i0, int j0, int i1, int j1) {
            const size_t ka = static_cast<size_t>(j0) * nx + i0;
            const size_t kb = static_cast<size_t>(j1) * nx + i1;
            const float xa = static_cast<float>(qi0 + i0) * s, za = static_cast<float>(qj0 + j0) * s;
            const float xb = static_cast<float>(qi0 + i1) * s, zb = static_cast<float>(qj0 + j1) * s;
            const float ya = topY(ka), yb = topY(kb);
            const float la = std::max(hGnd[ka], ya - o.skirtDepth);
            const float lb = std::max(hGnd[kb], yb - o.skirtDepth);
            const unsigned int base = static_cast<unsigned int>(positions.size() / 3);
            const float px[4] = {xa, xb, xb, xa};
            const float py[4] = {ya, yb, lb, la};
            const float pz[4] = {za, zb, zb, za};
            // Colour from the taller end, dimmed: a canopy wall is shaded by the
            // crowns above it, and the darker band is what reads as depth.
            const size_t kc = hCan[ka] >= hCan[kb] ? ka : kb;
            int sp;
            if (hCan[kc] < o.scrubMaxHeight) sp = 0;
            else if (hCan[kc] < o.birchMaxHeight && hGnd[kc] < o.birchMaxElevation)
                sp = 1;
            else
                sp = 2;
            const Color& c = speciesColor[static_cast<size_t>(sp)];
            for (int v = 0; v < 4; ++v) {
                const float dim = (v < 2) ? 1.f : 1.f - o.aoStrength;
                positions.push_back(px[v] - origin.x);
                positions.push_back(py[v] - origin.y);
                positions.push_back(pz[v] - origin.z);
                normals.push_back(0.f);
                normals.push_back(1.f);
                normals.push_back(0.f);// replaced by computeVertexNormals
                // Vertical faces get a (horizontal, height) parametrisation —
                // world-XZ UVs on a wall are the vertical smear this whole demo
                // has been fighting since phase 1b.
                uvs.push_back((px[v] + pz[v]) / o.uvPeriod);
                uvs.push_back(py[v] / o.uvPeriod);
                colors.push_back(c.r * dim * o.texGain);
                colors.push_back(c.g * dim * o.texGain);
                colors.push_back(c.b * dim * o.texGain);
            }
            indices.push_back(base);
            indices.push_back(base + 1);
            indices.push_back(base + 2);
            indices.push_back(base);
            indices.push_back(base + 2);
            indices.push_back(base + 3);
        };
        for (int j = 0; j < nz; ++j)
            for (int i = 0; i < nx - 1; ++i)// edge along +x, between quads (i,j-1) and (i,j)
                if (quadAt(i, j - 1) + quadAt(i, j) == 1) emitSkirt(i, j, i + 1, j);
        for (int j = 0; j < nz - 1; ++j)
            for (int i = 0; i < nx; ++i)// edge along +z, between quads (i-1,j) and (i,j)
                if (quadAt(i - 1, j) + quadAt(i, j) == 1) emitSkirt(i, j, i, j + 1);

        auto geo = std::make_shared<BufferGeometry>();
        geo->setIndex(indices);
        geo->setAttribute("position", FloatBufferAttribute::create(positions, 3));
        geo->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
        geo->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
        geo->setAttribute("color", FloatBufferAttribute::create(colors, 3));
        geo->computeVertexNormals();
        geo->computeBoundingBox();
        geo->computeBoundingSphere();
        return geo;
    }

    struct ForestLodOptions {
        float cellSize = 128.f;// world metres per LOD cell
        float l0Distance = 300.f;
        float l1Distance = 800.f;
        // Same site → same species/scale/yaw at every level, so a handoff moves
        // no tree. Shared with ForestOptions.
        float scrubMaxHeight = 6.f, birchMaxHeight = 14.f, birchMaxElevation = 350.f;
        float minScale = 0.35f, maxScale = 2.2f;
        float sink = 0.4f;
        int cap = 40000;
        unsigned int seed = 20260904u;

        // L2 = THINNED L1, not a surface. `buildCanopySurface` above is the
        // prettier idea and it works where the ground is gentle, but on a benched
        // cliff the slope gate cuts the valid region into tread-wide strips and a
        // lit horizontal sheet on each tread reads as a STAIRCASE OF GREEN TRAYS
        // against the sky (looked at: aaa_caps/geiranger_lod_2km_crop_ridge.png,
        // and it survives an isolated-quad filter because the strips really are
        // connected). Blobs on the same treads read as trees because they have a
        // silhouette in three dimensions. So beyond `l1Distance` keep the blobs
        // and drop 3 in 4, widening the survivors to hold the canopy MASS — the
        // thing the user said "did wonders for the look".
        int l2Keep = 4;          // keep 1 instance in l2Keep
        float l2ScaleBoost = 1.4f;// survivors widen to close the gaps
        bool buildCanopyMesh = false;
        CanopyMeshOptions mesh;
    };

    struct ForestLodStats {
        int sites = 0, planted = 0, cells = 0;
        int l0Meshes = 0, l1Meshes = 0, l2Meshes = 0;
        int canopyTris = 0;
        int l1ProtoTris = 0;// triangles in ONE L1 prototype (trunk + leaves)
    };

    // One LOD node per non-empty cell under `parent`.
    inline ForestLodStats buildCanopyForestLod(
            Object3D& parent, const std::vector<TreeSite>& sites,
            const std::array<SpeciesVariants, 3>& species,
            const std::shared_ptr<MeshStandardMaterial>& canopyMat,
            const terrain::HeightGrid& canopy, const terrain::HeightGrid& dem,
            const std::function<float(float, float)>& heightFn,
            const ForestLodOptions& o) {

        ForestLodStats st;
        st.sites = static_cast<int>(sites.size());
        if (sites.empty() || !heightFn) return st;

        std::mt19937 rng(o.seed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);
        std::vector<int> order(sites.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);
        if (static_cast<int>(order.size()) > o.cap) {
            std::shuffle(order.begin(), order.end(), rng);
            order.resize(static_cast<size_t>(o.cap));
        }

        const float cs = std::max(16.f, o.cellSize);
        struct Cell {
            int cx = 0, cz = 0;
            std::array<std::vector<Matrix4>, 3> xf;// per species, WORLD transforms
            Vector3 sum;
            int n = 0;
        };
        std::unordered_map<std::int64_t, Cell> cells;
        const auto key = [](int bx, int bz) {
            return (static_cast<std::int64_t>(bx) << 32) ^ static_cast<std::uint32_t>(bz);
        };

        Quaternion q;
        const Vector3 up{0.f, 1.f, 0.f};
        for (int idx : order) {
            const TreeSite& p = sites[static_cast<size_t>(idx)];
            const float y = heightFn(p.x, p.z);
            int s;
            if (p.canopyHeight < o.scrubMaxHeight) s = 0;
            else if (p.canopyHeight < o.birchMaxHeight && y < o.birchMaxElevation)
                s = 1;
            else
                s = 2;
            // The scale divisor is the L0 prototype's height; L1/L0 prototypes are
            // measured separately below and re-scaled per level, so a tree keeps
            // its metre height across the handoff.
            const int cx = static_cast<int>(std::floor(p.x / cs));
            const int cz = static_cast<int>(std::floor(p.z / cs));
            auto& cell = cells[key(cx, cz)];
            cell.cx = cx;
            cell.cz = cz;
            q.setFromAxisAngle(up, u01(rng) * math::TWO_PI);
            Matrix4 m;
            // Scale field carries the site's CANOPY HEIGHT, not a ratio: each level
            // divides by its own prototype height when it writes its matrices.
            m.compose(Vector3(p.x, y - o.sink, p.z), q, Vector3(p.canopyHeight, p.canopyHeight, p.canopyHeight));
            cell.xf[static_cast<size_t>(s)].push_back(m);
            cell.sum.add(Vector3(p.x, y, p.z));
            ++cell.n;
            ++st.planted;
        }
        st.cells = static_cast<int>(cells.size());

        // Prototype triangle count, for the budget line the demo prints.
        const auto triCount = [](const TreeVariant& v) {
            int t = 0;
            for (const auto& g : {v.trunkGeo, v.leafGeo}) {
                if (!g) continue;
                if (auto* ix = g->getIndex()) t += static_cast<int>(ix->count()) / 3;
                else if (auto* pa = g->getAttribute<float>("position"))
                    t += static_cast<int>(pa->count()) / 3;
            }
            return t;
        };
        if (!species[2].far.empty()) st.l1ProtoTris = triCount(species[2].far.front());

        Matrix4 local;
        for (auto& [k, cell] : cells) {
            const Vector3 origin(cell.sum.x / static_cast<float>(cell.n),
                                 cell.sum.y / static_cast<float>(cell.n),
                                 cell.sum.z / static_cast<float>(cell.n));
            auto lod = LOD::create();
            lod->name = "forest_cell";
            lod->position.copy(origin);

            // One variant per (species, level) per CELL, not per tree: variety
            // still comes from the cell mosaic, but the mesh (and entry) count
            // stays at 2 per species per level instead of 2 per variant.
            auto levelGroup = [&](bool nearTier, int keep, float boost) {
                auto g = Group::create();
                int meshes = 0;
                for (int s = 0; s < 3; ++s) {
                    const auto& xf = cell.xf[static_cast<size_t>(s)];
                    if (xf.empty()) continue;
                    const auto& vars = nearTier ? species[s].near : species[s].far;
                    if (vars.empty()) continue;
                    const size_t vi = static_cast<size_t>(
                            (cell.cx * 31 + cell.cz * 17 + s * 7) & 0x7fffffff) % vars.size();
                    // Thinning is a stride over a list that was already shuffled by
                    // the cap pass, so the survivors are spread over the cell
                    // rather than being its first rows.
                    const size_t n = keep <= 1 ? xf.size()
                                               : (xf.size() + static_cast<size_t>(keep) - 1) /
                                                         static_cast<size_t>(keep);
                    if (n == 0) continue;
                    auto trunks = InstancedMesh::create(vars[vi].trunkGeo, vars[vi].barkMat, n);
                    auto leaves = InstancedMesh::create(vars[vi].leafGeo, vars[vi].leafMat, n);
                    for (size_t oi = 0, i = 0; oi < n; ++oi, i += static_cast<size_t>(std::max(1, keep))) {
                        // Re-derive the instance from the stored (position, yaw,
                        // canopy height) so each level divides by ITS prototype's
                        // measured height: the two prototypes are not the same
                        // number of metres tall, and reusing one matrix would make
                        // every tree jump in size at the handoff.
                        Vector3 p, sc;
                        Quaternion rq;
                        xf[i].decompose(p, rq, sc);
                        const float scale = boost * std::clamp(sc.x / vars[vi].height,
                                                               o.minScale, o.maxScale);
                        local.compose(Vector3(p.x - origin.x, p.y - origin.y, p.z - origin.z), rq,
                                      Vector3(scale, scale, scale));
                        trunks->setMatrixAt(oi, local);
                        leaves->setMatrixAt(oi, local);
                    }
                    trunks->instanceMatrix()->needsUpdate();
                    leaves->instanceMatrix()->needsUpdate();
                    g->add(trunks);
                    g->add(leaves);
                    meshes += 2;
                }
                return std::pair{g, meshes};
            };

            auto [g0, m0] = levelGroup(true, 1, 1.f);
            auto [g1, m1] = levelGroup(false, 1, 1.f);
            st.l0Meshes += m0;
            st.l1Meshes += m1;
            lod->addLevel(g0, 0.f);
            lod->addLevel(g1, o.l0Distance);

            if (!o.buildCanopyMesh && o.l2Keep > 1) {
                auto [g2, m2] = levelGroup(false, o.l2Keep, o.l2ScaleBoost);
                st.l2Meshes += m2;
                lod->addLevel(g2, o.l1Distance);
            }
            if (o.buildCanopyMesh && canopyMat) {
                std::array<Color, 3> sc{};
                for (int s = 0; s < 3; ++s)
                    sc[static_cast<size_t>(s)] = species[s].far.empty()
                                                         ? Color(0.2f, 0.35f, 0.12f)
                                                         : species[s].far.front().leafMat->color;
                auto geo = buildCanopySurface(canopy, dem, heightFn, sc,
                                              static_cast<float>(cell.cx) * cs,
                                              static_cast<float>(cell.cz) * cs,
                                              static_cast<float>(cell.cx + 1) * cs,
                                              static_cast<float>(cell.cz + 1) * cs,
                                              origin, o.mesh);
                if (geo) {
                    if (auto* ix = geo->getIndex()) st.canopyTris += static_cast<int>(ix->count()) / 3;
                    auto m = Mesh::create(geo, canopyMat);
                    m->name = "canopy_surface";
                    lod->addLevel(m, o.l1Distance);
                    ++st.l2Meshes;
                }
            }
            parent.add(lod);
        }
        return st;
    }

}// namespace threepp::vegetation

#endif// THREEPP_CANOPYFOREST_HPP
