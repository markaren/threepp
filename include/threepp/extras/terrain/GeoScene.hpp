// GEO SCENE — one object for "put this real place in my scene".
//
// The geodata terrain stack (GeoTerrainPack → RoadNetwork → makeGeoProvider →
// TileTerrain → band sets → CliffShell → CanopyForest) is nine headers and
// ~300 lines of wiring, all of which examples/extras/terrain/norway_terrain.cpp
// spells out by hand. Every other consumer — another demo, a Python script, a
// sensor rig — needs the SAME wiring with the same defaults, and re-deriving it
// is how two scenes end up looking like two different planets.
//
// GeoScene is that wiring, verbatim, behind one call:
//
//     auto geo = terrain::GeoScene::create({.packDir = "geodata/norddal"});
//     scene.add(geo);
//     ...
//     geo->update(camera.position);   // once per frame: tile LOD + scatter
//
// It is a Group, so it drops into any scene graph; it OWNS the pack, the road
// network and the provider (all three are captured by reference inside the
// provider's callbacks, so their lifetime has to be the object's, not the
// caller's).
//
// ONE deliberate addition over the demo: SYNTHETIC BATHYMETRY. A Kartverket DTM
// stores water as a dead-flat sheet at exactly seaLevel — there are no
// soundings — so a boat, a net pen or a ROV placed offshore hangs 15 cm above
// the "seabed". norway_terrain sinks every sea cell by a constant 6 m, which is
// enough to stop the terrain poking through wave troughs but still reads as a
// knee-deep pond and gives an underwater camera a flat grey floor. Here the
// sink follows a DISTANCE-TO-SHORE profile instead: a fjord wall does not stop
// at the waterline, it keeps going down, so depth grows with distance from the
// nearest land cell and saturates in open water. See makeBathymetry() below.
//
// Everything else is copied, not reinvented: same band sets, same cliff gate
// (grid step <= 1.5 m), same shell parameters, same phase-3c forest LOD.

#ifndef THREEPP_EXTRAS_TERRAIN_GEOSCENE_HPP
#define THREEPP_EXTRAS_TERRAIN_GEOSCENE_HPP

#include "threepp/extras/road/RoadNetwork.hpp"
#include "threepp/extras/terrain/CliffShell.hpp"
#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/GeoTerrain.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/extras/terrain/TerrainScatter.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"
#include "threepp/extras/vegetation/CanopyForest.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace threepp::terrain {

    struct GeoSceneOptions {
        std::string packDir;

        // Per-band STRUCTURE sets (grass/rock/scree/snow) resolved at screen
        // density over the macro splat. Off = the legacy single detail layer.
        bool bands = true;

        // Contour-strip shell over the steep faces. AUTO-GATED on the DEM's own
        // resolution (grid step <= 1.5 m): sub-metre benches on a 2 m DEM would
        // be inventing structure below what the data measured.
        bool cliffShell = true;
        float shellLevelStep = 2.f;// contour spacing in height (m)
        float shellExtent = 1200.f;// half-extent of the shell region of interest

        // Canopy-driven forest. AUTO-GATED on the pack carrying a canopy height
        // model (packs fetched with --canopy); without one there is no
        // measurement of where forest stands and we plant nothing.
        bool forest = true;
        float forestExtent = 1100.f;// half-extent of the forest region of interest
        int forestCap = 40000;      // instance budget

        // Centre of the shell + forest regions of interest — the SUBJECT. A 4 km
        // pack is 16 M cells and a shot only ever looks at part of it; the budget
        // is better spent dense near the subject than thin across the square.
        Vector3 focus{0.f, 0.f, 0.f};

        // Near-field instanced stones/tufts in the last ~55 m around the camera.
        // Demo parity default. Turn it off for a scene whose camera lives over
        // water: the cells would be built on the (sunk) seabed.
        bool scatter = true;

        // ── synthetic bathymetry (see the header comment) ────────────────────
        bool bathymetry = true;
        float shoreSlope = 0.35f;// metres of depth per metre from shore
        float maxDepth = 180.f;  // open-water saturation depth
        // Fallback when bathymetry is off: norway_terrain's flat sink, so wave
        // troughs still have water under them.
        float flatSeaDepth = 6.f;

        unsigned int seed = 4242u;
    };

    class GeoScene: public Group {

    public:
        struct Stats {
            int tiles = 0;      // live LOD tiles
            int baking = 0;     // tile bakes in flight
            std::size_t shellTris = 0;
            int forestSites = 0;// CHM sites that survived the gates
            int forestCells = 0;// LOD cells planted
            float loadSeconds = 0.f;
        };

        static std::shared_ptr<GeoScene> create(const GeoSceneOptions& opts) {
            auto s = std::shared_ptr<GeoScene>(new GeoScene());
            s->build(opts);
            return s;
        }

        // Once per frame, with the ACTIVE camera's position: tile LOD refine /
        // merge + near-field scatter. (The forest is a tree of threepp LOD
        // nodes; the renderer updates those itself.)
        void update(const Vector3& camPos) {
            if (tiles_) tiles_->update(camPos);
            if (scatter_) scatter_->update(camPos);
        }

        // Provider height — the surface the tiles actually bake, i.e. DEM +
        // road carve + relief + bathymetry. Thread-safe.
        [[nodiscard]] float heightAt(float x, float z) const {
            return prov_.height ? prov_.height(x, z) : 0.f;
        }

        [[nodiscard]] const GeoTerrainPack& pack() const { return pack_; }
        [[nodiscard]] const TerrainProvider& provider() const { return prov_; }
        [[nodiscard]] float packWorldSize() const { return pack_.region.worldSize; }
        [[nodiscard]] float seaLevel() const { return pack_.region.seaLevel; }

        [[nodiscard]] Stats stats() const {
            Stats s = stats_;
            if (tiles_) {
                s.tiles = tiles_->activeTiles();
                s.baking = tiles_->pendingBakes();
            }
            return s;
        }

        [[nodiscard]] std::string type() const override { return "GeoScene"; }

        ~GeoScene() override = default;

    private:
        GeoScene() = default;

        // ── synthetic bathymetry ────────────────────────────────────────────
        //
        // depth(d) = maxDepth · (1 − exp(−shoreSlope · d · ease(d) / maxDepth))
        //
        // where d is the distance (m) to the nearest non-sea cell. The exponent
        // is the plan's linear ramp `shoreSlope · d`; wrapping it in the
        // saturating exponential is what makes the maxDepth clamp C1 (a plain
        // min() puts a crease along an isoline of the distance field, and the
        // provider's bicubic turns a crease into a visible ridge on the
        // seabed). `ease` is a smoothstep over the first few metres so the
        // shoreline leaves at zero SLOPE as well as zero depth — otherwise the
        // beach starts with a 19° cliff exactly where the bicubic support
        // straddles the mask boundary, and the waterline reads as a torn edge.
        static void makeBathymetry(HeightGrid& grid, float seaLevel, float shoreSlope,
                                   float maxDepth) {
            const int n = grid.dim();
            if (n < 4 || maxDepth <= 0.f || shoreSlope <= 0.f) return;
            const float step = grid.worldSize() / static_cast<float>(n - 1);
            auto& h = grid.data();

            // Exact squared Euclidean distance transform (Felzenszwalb &
            // Huttenlocher 2004): two separable O(N) lower-envelope passes.
            // Exact rather than a chamfer approximation because the depth is a
            // pure function of this field — a 2% chamfer error is a 2% depth
            // ripple aligned with the chamfer's octagonal artefacts, and on a
            // mirror-flat seabed that is exactly the kind of structure that
            // shows. "Infinity" is a large FINITE value: the textbook version
            // divides inf by inf on all-sea rows and walks the envelope stack
            // off its lower bound.
            const double big = 4.0 * static_cast<double>(n) * static_cast<double>(n);
            std::vector<double> f(static_cast<std::size_t>(n) * n);
            for (std::size_t i = 0; i < f.size(); ++i)
                f[i] = (h[i] <= seaLevel + 0.05f) ? big : 0.0;// 0 = land = a source

            std::vector<double> col(n), out(n), z(static_cast<std::size_t>(n) + 1);
            std::vector<int> v(n);
            const auto envelope = [&](std::vector<double>& src) {
                int k = 0;
                v[0] = 0;
                z[0] = -1e30;
                z[1] = 1e30;
                for (int q = 1; q < n; ++q) {
                    double s = ((src[q] + double(q) * q) - (src[v[k]] + double(v[k]) * v[k])) /
                               (2.0 * q - 2.0 * v[k]);
                    while (k > 0 && s <= z[k]) {
                        --k;
                        s = ((src[q] + double(q) * q) - (src[v[k]] + double(v[k]) * v[k])) /
                            (2.0 * q - 2.0 * v[k]);
                    }
                    ++k;
                    v[k] = q;
                    z[k] = s;
                    z[k + 1] = 1e30;
                }
                k = 0;
                for (int q = 0; q < n; ++q) {
                    while (z[k + 1] < q) ++k;
                    const double dq = q - v[k];
                    out[q] = dq * dq + src[v[k]];
                }
            };

            for (int iz = 0; iz < n; ++iz) {// rows
                double* row = f.data() + static_cast<std::size_t>(iz) * n;
                std::copy(row, row + n, col.begin());
                envelope(col);
                std::copy(out.begin(), out.end(), row);
            }
            for (int ix = 0; ix < n; ++ix) {// columns
                for (int iz = 0; iz < n; ++iz) col[iz] = f[static_cast<std::size_t>(iz) * n + ix];
                envelope(col);
                for (int iz = 0; iz < n; ++iz) f[static_cast<std::size_t>(iz) * n + ix] = out[iz];
            }

            const float feather = std::max(4.f * step, 4.f);// C1 ease-in at the shore
            for (std::size_t i = 0; i < f.size(); ++i) {
                if (h[i] > seaLevel + 0.05f) continue;
                const float d = std::sqrt(static_cast<float>(f[i])) * step;
                const float t = std::clamp(d / feather, 0.f, 1.f);
                const float ease = t * t * (3.f - 2.f * t);
                h[i] -= maxDepth * (1.f - std::exp(-shoreSlope * d * ease / maxDepth));
            }
        }

        void build(const GeoSceneOptions& o) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            name = "geo_scene";

            pack_ = GeoTerrainPack::load(o.packDir);// throws on a bad pack
            const GeoRegion& reg = pack_.region;

            // ── roads → unified ground height ────────────────────────────────
            // Even a pack with no roads goes through this: makeGeoProvider takes
            // a RoadNetwork by reference and queries corridorWeight() on every
            // height sample. An empty network answers 0 everywhere.
            std::vector<road::RoadSpec> specs;
            specs.reserve(pack_.roads.size());
            for (const auto& gr : pack_.roads) {
                road::RoadSpec s;
                s.id = gr.id;
                s.category = gr.category;
                s.width = gr.width;
                s.points = gr.points;
                specs.push_back(std::move(s));
            }
            network_ = std::make_unique<road::RoadNetwork>(std::move(specs));
            road::RoadProfileOptions rpo;
            rpo.enabled = true;// classify bridges / tunnels / ferry legs from the
            rpo.seaLevel = reg.seaLevel;// pack's own point heights
            network_->conformTo([this](float x, float z) { return pack_.grid.sampleBicubic(x, z); },
                                14, rpo);
            RoadCarveOptions rco;
            rco.bakeSurface = true;// the terrain IS the road (paint carries the look)
            carveRoads(pack_.grid, *network_, rco);

            // Bathymetry AFTER conform + carve: the road profile classification
            // must see the real DTM water level, and no roadbed cell is left at
            // sea level once bridges and ferry legs are excluded.
            if (o.bathymetry && reg.heightMin < 1.0f) {
                makeBathymetry(pack_.grid, reg.seaLevel, o.shoreSlope, o.maxDepth);
            } else if (o.flatSeaDepth > 0.f && reg.heightMin < 1.0f) {
                for (float& h: pack_.grid.data())
                    if (h <= reg.seaLevel + 0.05f) h -= o.flatSeaDepth;
            }

            // ── provider ─────────────────────────────────────────────────────
            const float gridStep = reg.worldSize / static_cast<float>(reg.dim - 1);
            const bool cliffPack = gridStep <= 1.5f;
            const bool shellOn = cliffPack && o.cliffShell;

            GeoTerrainOptions gopt;
            gopt.snowHeightMin = std::max(reg.heightMax - 350.f, 900.f);// scene-relative snowline
            gopt.grassHeightMax =
                    std::clamp(reg.heightMin + 0.45f * (reg.heightMax - reg.heightMin), 200.f, 900.f);
            gopt.wetlandBand = 6.f;
            gopt.paintRoads = true;
            gopt.roadEdgeFeather = 1.2f;// near-tile splat texels are ~0.6-1.3 m
            gopt.paintUrban = true;
            // The shell OWNS the wall relief once it is on: a positive terrain
            // relief under it would poke through the shell's 0.35 m offset.
            gopt.cliffRelief = cliffPack && !shellOn;
            prov_ = makeGeoProvider(pack_, *network_, gopt);

            // ── tiles ────────────────────────────────────────────────────────
            TileTerrainOptions tileOpts;
            tileOpts.worldSize = reg.worldSize;
            tileOpts.rootGrid = 4;
            tileOpts.maxDepth = 5;
            tileOpts.tileRes = 96;
            tileOpts.splitFactor = 1.2f;
            tileOpts.mergeFactor = 1.7f;
            tileOpts.splatTexelsPerQuad = 2;// 4 is flicker-identical and 4x the bake
            tileOpts.asyncBake = true;
            // Road-aware LOD: corridor tiles refine ~2.2x sooner so the painted
            // roadbed stays crisp at mid distance.
            tileOpts.refineBias = [this](float cx, float cz, float half) {
                return network_->corridorIntersects(cx, cz, half) ? 2.2f : 1.0f;
            };

            bool bandSetBuilt = false;
            if (o.bands) {
                // On a cliff pack the rock slot carries the GNEISS generator
                // (vertical foliation, joint blocks, wet veins).
                bandSet_ = makeTerrainBandSet(o.seed, cliffPack ? BandKind::Cliff : BandKind::Rock);
                bandSetBuilt = true;
                for (std::size_t i = 0; i < 4; ++i) {
                    tileOpts.bandAlbedo[i] = bandSet_.band[i].albedo;
                    tileOpts.bandNormalRough[i] = bandSet_.band[i].normalRough;
                }
                tileOpts.bandRepeat = bandSet_.repeat;
                tileOpts.bandRoughness = bandSet_.roughness;
                // A wall gets the band layer HOT: the macro splat under it is
                // baked in XZ and smears vertically on a near-vertical face, so
                // the triplanar band is the only layer that can put structure
                // there — it has to out-shout the smear.
                tileOpts.bandStrength = cliffPack ? 1.0f : 0.8f;
                tileOpts.bandNormalScale = cliffPack ? 2.2f : 1.4f;
                tileOpts.bandRoughStrength = 0.6f;
            }
            {   // Legacy cm-scale detail layer — the fallback wherever bands are
                // off (and on the forward GL path, which ignores band fields).
                const DetailMaps dm = makeDetailMaps({});
                tileOpts.detailMap = dm.albedo;
                tileOpts.detailNormalMap = dm.normalRough;
                tileOpts.detailRepeat = 0.6f;
                tileOpts.detailStrength = 0.7f;
                tileOpts.detailNormalScale = 1.0f;
                tileOpts.detailRoughStrength = 0.5f;
            }

            tiles_ = TileTerrain::create(prov_, tileOpts);
            tiles_->name = "geo_tiles";
            add(tiles_);

            if (o.scatter) {
                scatter_ = TerrainScatter::create(prov_, {});
                scatter_->name = "geo_ground_cover";
                add(scatter_);
            }

            // ── cliff shell ──────────────────────────────────────────────────
            // The terrain is a heightfield: on a near-vertical wall every baked
            // tile texel is one stretched vertical column, so nothing baked on
            // the tiles can vary ALONG a column. The shell is a free mesh whose
            // parametrisation is u = contour arc length / v = world height —
            // metric on the wall, so ledge rows and seepage streaks can exist.
            if (shellOn && bandSetBuilt) {
                CliffShellOptions so;
                so.seaLevel = reg.seaLevel;
                so.levelStep = o.shellLevelStep;
                so.snowHeightMin = gopt.snowHeightMin;
                so.snowFeather = gopt.snowFeather;
                so.canopyForestMin = gopt.canopyForestMin;
                so.centerX = o.focus.x;
                so.centerZ = o.focus.z;
                so.halfExtent = o.shellExtent;
                auto shellRoot = Group::create();
                shellRoot->name = "geo_cliff_shell";
                const auto st = buildCliffShell(*shellRoot, pack_, bandSet_, so,
                                                cliffPack ? 1.0f : 0.8f,
                                                cliffPack ? 2.2f : 1.4f);
                stats_.shellTris = st.triangles;
                add(shellRoot);
            }

            // ── canopy-driven forest ─────────────────────────────────────────
            // The CHM (DOM − DTM) is a MEASUREMENT of where forest stands and
            // how tall it is; trees go exactly there instead of on a slope /
            // elevation rule that invents a forest. No CHM ⇒ no forest.
            if (o.forest && pack_.hasCanopy()) buildForest(o, reg);

            stats_.loadSeconds = std::chrono::duration<float>(
                                         std::chrono::high_resolution_clock::now() - t0)
                                         .count();
        }

        void buildForest(const GeoSceneOptions& o, const GeoRegion& reg) {
            vegetation::CanopySiteOptions so;
            so.seaLevel = reg.seaLevel;
            so.centerX = o.focus.x;
            so.centerZ = o.focus.z;
            so.halfExtent = o.forestExtent;
            const auto sites = vegetation::detectTreeSites(pack_.canopy, pack_.grid, so);
            if (sites.empty()) return;

            // Two prototypes per species for the near tier (card/frond canopies),
            // three for the far tier (cheap blob puffs) — enough silhouette
            // variety that a hillside does not read as one stamp repeated.
            std::array<vegetation::SpeciesVariants, 3> species;
            for (int s = 0; s < 3; ++s) {
                const auto sp = static_cast<vegetation::TreeSpecies>(s);
                const auto base = static_cast<unsigned int>(100 + s * 37);
                species[s].near = {vegetation::makeForestTreeVariant(sp, base + 1u, false),
                                   vegetation::makeForestTreeVariant(sp, base + 2u, false)};
                species[s].far = {vegetation::makeForestTreeVariant(sp, base + 11u, true, true),
                                  vegetation::makeForestTreeVariant(sp, base + 12u, true, true),
                                  vegetation::makeForestTreeVariant(sp, base + 13u, true, true)};
            }

            // The canopy-surface material. The leaf atlas is a GRAIN map, not
            // the colour: it is generated near-neutral so the per-blob vertex
            // tint survives the multiply. NO alphaTest — at the 3 m lattice a
            // cutout punches the sheet into a bubble-wrap net of square holes;
            // the ragged edge comes from the GEOMETRY instead.
            auto leafTex = vegetation::makeLeafClusterTexture(256, 77u, {0.90f, 0.93f, 0.86f},
                                                              vegetation::LeafShape::Ovate, 8, 2);
            leafTex->wrapS = TextureWrapping::Repeat;
            leafTex->wrapT = TextureWrapping::Repeat;
            auto canopyMat = MeshStandardMaterial::create(
                    MeshStandardMaterial::Params{}.color(Color::white).roughness(0.9f).metalness(0.f));
            canopyMat->map = leafTex;
            canopyMat->side = Side::Double;// the skirt is a curtain, seen from both
            canopyMat->vertexColors = true;
            canopyMat->translucencyColor = Color(0.50f, 0.80f, 0.28f);
            canopyMat->translucency = 0.3f;

            vegetation::ForestLodOptions lo;
            lo.cap = o.forestCap;
            lo.cellSize = 128.f;
            lo.l0Distance = 300.f;
            lo.l1Distance = 800.f;
            lo.l2Keep = 4;// the far tier costs 21% of the frame at full density
            lo.mesh.seaLevel = reg.seaLevel;
            lo.mesh.maxSlopeDeg = so.maxSlopeDeg;      // same gates as the sites,
            lo.mesh.minGroundHeight = so.minGroundHeight;// or the handoff grows new forest

            auto forest = Group::create();
            forest->name = "geo_canopy_forest";
            // Bases come from the PROVIDER (relief + road carve + bathymetry
            // included), not the raw DEM, or every trunk floats or sinks.
            const auto st = vegetation::buildCanopyForestLod(*forest, sites, species, canopyMat,
                                                            pack_.canopy, pack_.grid, prov_.height, lo);
            stats_.forestSites = st.sites;
            stats_.forestCells = st.cells;
            add(forest);
        }

        // Order matters: pack_ and network_ are captured BY REFERENCE inside
        // prov_'s callbacks, and tiles_/scatter_ hold copies of prov_. Declaring
        // them first means they are destroyed last.
        GeoTerrainPack pack_;
        std::unique_ptr<road::RoadNetwork> network_;
        TerrainProvider prov_;
        TerrainBandSet bandSet_;
        std::shared_ptr<TileTerrain> tiles_;
        std::shared_ptr<TerrainScatter> scatter_;
        Stats stats_;
    };

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_GEOSCENE_HPP
