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
// THE URBAN LAYER. A pack fetched with --buildings/--landuse carries a town,
// and until it is wired here a Python consumer gets a bare hill where Ålesund
// is. Buildings, pier decks, parked cars and moored boats are therefore built
// here too, each gated on the pack ACTUALLY carrying the data it needs — a
// Norddal pack (no footprints, no land use) takes none of these branches and
// behaves exactly as it did before they existed.
//
// THE STREETS. For a while the urban layer had a hole in it: GeoScene used the
// RoadNetwork only to conform and carve the terrain and to answer pavedWeight()
// gates, so a consumer got a town with NO STREETS IN IT — no asphalt-coloured
// bed (gopt.roadColor was left at its literal default instead of the network's
// own baked mean) and no near-field ribbon geometry at all. The demo has had
// both since the hybrid landed. Both are here now; see the roadColor line in
// build() and buildRoadRibbon() below.
//
// The urban layer is also what makes the FOREST correct. A CHM is DOM − DTM,
// so over a town every roof is a 10 m "canopy peak"; without the town gates in
// buildForest() the detector plants a spruce on all 8287 roofs in Ålesund.
// The gates are not a refinement, they are the difference between a forest and
// a bug — see the comment at the top of buildForest().
//
// Everything else is copied, not reinvented: same band sets, same cliff gate
// (grid step <= 1.5 m), same shell parameters, same phase-3c forest LOD.

#ifndef THREEPP_EXTRAS_TERRAIN_GEOSCENE_HPP
#define THREEPP_EXTRAS_TERRAIN_GEOSCENE_HPP

#include "threepp/extras/road/RoadNetwork.hpp"
#include "threepp/extras/terrain/CellStreamer.hpp"
#include "threepp/extras/terrain/CliffShell.hpp"
#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/GeoBuildings.hpp"
#include "threepp/extras/terrain/GeoTerrain.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/extras/terrain/TerrainScatter.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"
#include "threepp/extras/terrain/UrbanProps.hpp"
#include "threepp/extras/vegetation/CanopyForest.hpp"
#include "threepp/extras/vegetation/TreeTextures.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
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

        // ── the urban layer ─────────────────────────────────────────────────
        // All four gates below are ALSO gated on the pack: buildings need
        // footprints (buildings.json), props need land use (landuse.json).
        // Asking for them on a pack that has neither is not an error, it is a
        // no-op, exactly like `forest` on a pack with no CHM.

        // Extruded OSM footprints with nDSM-measured heights, batched into
        // 500 m chunk meshes.
        bool buildings = true;
        bool pitchedRoofs = true; // gable heuristic on footprints with no roof block
        bool measuredRoofs = true;// roof shapes read off the pack's 1 m DOM

        // Pier decks, parked cars, moored boats — placed from the survey, never
        // scattered. Cars are STREAMED by camera distance (a pack holds tens of
        // thousands of them); decks and boats are a few hundred objects for the
        // whole pack and are built once, because they are part of the LAND.
        bool urbanProps = true;
        bool cars = true;
        bool boats = true;
        bool decks = true;
        float propsCellSize = 250.f;// one car draw per cell
        float propsExtent = 1500.f; // car streaming radius from the camera
        int streamBudget = 2;       // cell builds per update(), forest and cars alike
        // Steepest ground a parked car is allowed to stand on, in degrees. OSM
        // land use is drawn in plan view, so a `parking` polygon can be draped
        // over a mountainside — the Ålesund pack has one at world XZ (684, -85)
        // whose ground climbs 57.9 m across 50 m, mean slope 47.2°, where the
        // reality is a park and a stepped walkway. Real car parks in that pack
        // reach 21.2° at the 98th percentile over 223 polygons, so 28° sits in
        // the gap: no mapped lot loses a car, the cliff loses all of them. See
        // UrbanPropsOptions::carMaxSlope for the full measurement.
        float carMaxSlope = 28.f;

        // Footprint dilation for the forest's town gate. This is NOT padding:
        // it is the registration offset between the Kartverket DOM and the OSM
        // footprint, measured on the Ålesund pack. At 2 m a full ring of
        // roof-edge "canopy" survives and the town grows a hedge on every eave.
        float forestDilate = 4.f;
        // Instance budget on an urban pack, where the forest streams per cell:
        // this is a PER-CELL cap, so it is deliberately generous. The town's
        // real bound is the streaming radius, not a global count.
        int urbanForestCap = 80000;

        // ── roads ────────────────────────────────────────────────────────────
        // Roads are BAKED into the terrain — carveRoads() puts the bed at the
        // conformed grade and the provider paints asphalt over the paved band —
        // and that painted bed is what a distant road IS: tile texture, so mip
        // and aniso filtering integrate it as it recedes instead of letting a
        // sub-pixel ribbon's raster coverage shimmer.
        //
        // The ribbon CHUNKS below are the near-field half of that hybrid: real
        // geometry a hair above the bed, carrying crisp edges and baked lane
        // markings — "stand-on" quality that ~1 m splat texels cannot hold.
        // They are distance-culled in update(); past the cut the painted bed
        // alone carries the road. BRIDGE DECKS are always ribbon geometry and
        // are never culled: there is no terrain under a span to paint.
        //
        // Gated on the pack actually carrying roads, like every other gate
        // here: a pack with no roads.json takes none of this.
        bool roadRibbon = true;
        float ribbonDistance = 600.f;// the demo's NT_ROAD_RIBBON_DIST; 6 m road ≈ 5 px

        // Surveyed town SURFACING painted into the splat — parking asphalt with
        // bay stripes, grass and pitches that block the urban grey, gravel,
        // quay concrete, breakwater rock. Without it the Fjellstua car park is a
        // meadow with cars standing on it. (GeoTerrainOptions already defaults
        // this on; the knob is here so a consumer can A/B the layer alone the
        // way NT_NO_LANDUSE_PAINT does in the demo.)
        bool landUsePaint = true;

        // ── the quay apron ───────────────────────────────────────────────────
        // A Kartverket DTM reads RECLAIMED LAND as water: the harbour front is
        // stored at exactly seaLevel, with no relief at all. Measured on the
        // Ålesund pack: 93.8% of sampled ground inside its 53 surveyed `pier`
        // polygons reads exactly 0.00 m. That is not a cosmetic error, because
        // everything downstream keys off "is this cell at sea level":
        //   * makeBathymetry() seeds its distance transform with
        //     `h <= seaLevel + 0.05f` = SEA, so the quay is not merely awash,
        //     it is EXCAVATED into the seabed along with the fjord;
        //   * UrbanProps builds a pier deck only where the ground is "wet"
        //     (< sea + 0.5 m), so a deck gets built OVER the quay — see the
        //     comment at UrbanProps.hpp:446, which already assumes this apron
        //     exists ("a pier line that runs up onto the quay is already
        //     ground (the apron raise gave it a surface)");
        //   * and the buildings stand with their walls in the fjord, which is
        //     the symptom norway_terrain names at its own apron block.
        // So the raise happens HERE, in the facade, on the same terms the demo
        // does it. Gated on the pack carrying buildings or land use, like every
        // other urban gate: a bare fjord pack takes none of it.
        bool quayApron = true;

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
            std::uint64_t treeSignature = 0;// TerrainTiles::treeSignature(): which tiles, not how many
            int buildings = 0;              // footprints extruded
            std::size_t buildingTris = 0;
            std::size_t cars = 0;// parked cars PLACED pack-wide (not the live ones)
            int carCellsLive = 0;// car cells currently streamed in around the camera
            int boats = 0;
            int deckRuns = 0;
            int carsRejectedSlope = 0;// bays refused for standing on a cliff
            int roadChunks = 0;       // near-field ribbon chunks BUILT (pack-wide)
            int roadChunksLive = 0;   // ...of those, visible after the last update()
            int apronCells = 0;       // sea-level grid cells raised into reclaimed land
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
            // Streamed content follows the LIVE camera. A ROI fixed at load
            // time is a still-frame trick: fly two kilometres and the town is
            // bare because the ROI never moved.
            if (forestStream_) forestStream_->update(camPos);
            if (carStream_) carStream_->update(camPos);
            cullRoadRibbon(camPos);
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
                s.treeSignature = tiles_->treeSignature();
            }
            if (forestStream_) s.forestCells = forestStream_->stats().active;
            if (carStream_) s.carCellsLive = carStream_->stats().active;
            return s;
        }

        [[nodiscard]] std::string type() const override { return "GeoScene"; }

        ~GeoScene() override = default;

    private:
        GeoScene() = default;

    public:
        // Public so demos (norway_terrain NT_SEA_BATHY) can sink a pack
        // grid with the same profile the facade uses; otherwise internal.
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
    private:

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

            // ── quay apron ───────────────────────────────────────────────────
            // See GeoSceneOptions::quayApron for WHY. Mechanically: every cell
            // the DTM left at sea level that is under a building (footprints
            // grown by 8 m — the apron is the YARD around the shed, not just
            // its outline) or inside a surveyed pier/quay polygon becomes land
            // 0.9 m above the water, and every breakwater cell a 1.5 m rock
            // ridge. The heightfield gives that a short ramp at its edge rather
            // than a vertical quay wall; at the ranges these packs are judged
            // from that is the right trade, and it costs no extra geometry and
            // leaves no seam with the tiles.
            //
            // THE ORDER IS THE POINT, and it is why this cannot be bolted on
            // afterwards. It runs BEFORE the bathymetry / flat sink below, or
            // the sink still classifies these cells as water and drops them
            // into the seabed; and BEFORE makeGeoProvider() below, because the
            // land-use paint takes the mask (gopt.apronMask) and paints the
            // reclaimed land as concrete instead of letting a lawn run into
            // the fjord.
            std::shared_ptr<const FootprintMask> apronMask;
            if (o.quayApron && reg.heightMin < 1.0f &&
                (!pack_.buildings.empty() || pack_.hasLandUse())) {
                const auto builtNear = buildFootprintMask(pack_, 8.f, 2.f);
                const auto quayPoly = buildLandUseMask(pack_, {"pier", "quay"}, 0.f, 2.f);
                const auto breakPoly = buildLandUseMask(pack_, {"breakwater"}, 0.f, 2.f);
                if (builtNear || quayPoly || breakPoly) {
                    const int gdim = pack_.grid.dim();
                    const float gstep = pack_.grid.worldSize() / static_cast<float>(gdim - 1);
                    const float ghalf = pack_.grid.worldSize() * 0.5f;
                    auto apron = detail::geoMakeMask(reg.worldSize, gstep);
                    auto& hh = pack_.grid.data();
                    for (int iz = 0; iz < gdim && iz < apron->dim; ++iz) {
                        const float z = -ghalf + static_cast<float>(iz) * gstep;
                        for (int ix = 0; ix < gdim && ix < apron->dim; ++ix) {
                            float& hv = hh[static_cast<std::size_t>(iz) * gdim + ix];
                            if (hv > reg.seaLevel + 0.05f) continue;
                            const float x = -ghalf + static_cast<float>(ix) * gstep;
                            const bool rock = breakPoly && breakPoly->inside(x, z);
                            if (!rock && !(builtNear && builtNear->inside(x, z)) &&
                                !(quayPoly && quayPoly->inside(x, z)))
                                continue;
                            hv = reg.seaLevel + (rock ? 1.5f : 0.9f);
                            apron->m[static_cast<std::size_t>(iz) * apron->dim + ix] = 1u;
                            ++stats_.apronCells;
                        }
                    }
                    apronMask = apron;
                }
            }

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
            // The far road is a flat tint mixed into the splat albedo. Take it
            // from the SAME bake the near ribbon uses instead of the literal
            // default: that default is a near-black matched to an older
            // near-black ribbon, and any drift between the two makes the
            // ribbonDistance hand-off STEP as a chunk winks out. This has to
            // run after conformTo() — meanSurfaceColor reads the baked surface
            // sets — and before makeGeoProvider() reads gopt.
            gopt.roadColor = network_->meanSurfaceColor(road::SurfaceKind::Asphalt);
            // Surveyed town surfacing (parking / grass / pitch / concrete /
            // gravel / rock), plus the apron raised above: buildLandUsePaint()
            // takes the mask and paints those reclaimed cells as quay concrete.
            // Null on a pack with no harbour, which is the no-op path.
            gopt.landUsePaint = o.landUsePaint;
            gopt.apronMask = apronMask;
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

            // ── road ribbon ──────────────────────────────────────────────────
            // Bridge decks + near-field ground chunks. Gated on the pack having
            // roads: an empty network builds an empty group, which is harmless
            // but still a Group in the graph and a mesh walk every frame.
            if (o.roadRibbon && !pack_.roads.empty()) buildRoadRibbon(o);

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
            // ── buildings ────────────────────────────────────────────────────
            // Packs fetched with --buildings carry OSM footprints plus a per-
            // building roof block measured off the 1 m DOM. No footprints ⇒
            // nothing to extrude, and the branch is skipped entirely.
            if (o.buildings && !pack_.buildings.empty()) {
                GeoBuildingsOptions bo;
                bo.pitchedRoofs = o.pitchedRoofs;
                bo.measuredRoofs = o.measuredRoofs;
                GeoBuildingsStats bs;
                bo.stats = &bs;
                auto buildings = buildGeoBuildingMeshes(pack_, bo);
                buildings->name = "geo_buildings";
                stats_.buildings = bs.buildings;
                stats_.buildingTris = bs.triangles;
                add(buildings);
            }

            if (o.forest && pack_.hasCanopy()) buildForest(o, reg);

            // ── urban props ──────────────────────────────────────────────────
            if (o.urbanProps && pack_.hasLandUse()) buildUrbanLayer(o, reg, gopt);

            stats_.loadSeconds = std::chrono::duration<float>(
                                         std::chrono::high_resolution_clock::now() - t0)
                                         .count();
        }

        // ── road ribbon: bridge decks + distance-culled ground chunks ────────
        // See the roadRibbon comment in GeoSceneOptions for WHY this is only
        // the near field. Mechanically: the decks go in unconditionally (there
        // is no terrain under a span for the paint to live on), the ground
        // chunks are measured once here — bounding sphere per chunk, so the
        // per-frame test is a distance and a subtraction — and switched on and
        // off in cullRoadRibbon().
        void buildRoadRibbon(const GeoSceneOptions& o) {
            auto bridges = network_->buildBridgeMeshes();
            bridges->name = "geo_road_bridges";
            add(bridges);

            roadChunks_ = network_->buildGroundChunkMeshes();
            roadChunks_->name = "geo_road_chunks";
            chunkCenters_.reserve(roadChunks_->children.size());
            chunkRadii_.reserve(roadChunks_->children.size());
            for (auto* child : roadChunks_->children) {
                auto* mesh = child->as<Mesh>();
                if (!mesh) continue;
                auto geo = mesh->geometry();
                geo->computeBoundingSphere();
                chunkCenters_.emplace_back(child, geo->boundingSphere->center);
                chunkRadii_.push_back(geo->boundingSphere->radius);
            }
            add(roadChunks_);
            ribbonDist_ = o.ribbonDistance;
            stats_.roadChunks = static_cast<int>(chunkCenters_.size());
            // The startup cull, for the same reason the streamers prime their
            // rings at o.focus: a headless render or a one-shot capture must not
            // photograph every chunk in the pack at once because update() has
            // not been called yet.
            cullRoadRibbon(o.focus);
        }

        // Visible within ribbonDist_ of the camera, with 10% hysteresis so a
        // chunk sitting on the boundary does not flip every frame. The distance
        // is to the chunk's SURFACE (centre minus radius): a 240 m chunk seen
        // end-on is 120 m nearer than its centre says, and culling on the centre
        // pops the road out from under a camera standing on it.
        void cullRoadRibbon(const Vector3& camPos) {
            if (chunkCenters_.empty()) return;
            int live = 0;
            for (std::size_t i = 0; i < chunkCenters_.size(); ++i) {
                auto* obj = chunkCenters_[i].first;
                const float d = camPos.distanceTo(chunkCenters_[i].second) - chunkRadii_[i];
                if (obj->visible) {
                    if (d > ribbonDist_ * 1.1f) obj->visible = false;
                } else if (d < ribbonDist_) {
                    obj->visible = true;
                }
                if (obj->visible) ++live;
            }
            stats_.roadChunksLive = live;
        }

        void buildForest(const GeoSceneOptions& o, const GeoRegion& reg) {
            // ── TOWN GATES ───────────────────────────────────────────────────
            // A CHM is DOM − DTM, so on an urban pack every BUILDING is a 10 m
            // "canopy peak" and every ship, crane and pier crate is a taller
            // one. Without the gates below the detector plants a spruce on
            // every roof in Ålesund. Measured on that pack: 30.2% of land cells
            // carry canopy >= 2.5 m, and 29.3% of those sit inside a footprint
            // DILATED BY 4 M — the dilation is the DOM-vs-OSM registration
            // offset, not padding.
            const bool urbanPack = !pack_.buildings.empty() || pack_.hasLandUse();

            // Rasters at the pack's own 2 m resolution. Both are null on a pack
            // that carries no footprints / no land use, and a null mask is an
            // open gate — which is why a fjord pack takes the same code path
            // and comes out unchanged.
            const auto fpMask = buildFootprintMask(pack_, o.forestDilate, 2.f);
            // Decks and lots: a pier is not ground, a marina is water, and a
            // parking lot or a football pitch is a surface nobody plants in.
            const auto deckMask = buildLandUseMask(
                    pack_, {"pier", "quay", "breakwater", "marina", "parking", "pitch"}, 1.f, 2.f);

            vegetation::CanopySiteOptions so;
            so.seaLevel = reg.seaLevel;
            so.centerX = o.focus.x;
            so.centerZ = o.focus.z;
            so.halfExtent = o.forestExtent;
            if (urbanPack) {
                // PACK-WIDE detection. A square ROI centred on the focus makes
                // the trees a function of where the camera happened to start;
                // detection is a one-off scan and a site is 16 bytes, so what
                // has to be bounded is the GEOMETRY — the streamer's job below.
                so.centerX = 0.f;
                so.centerZ = 0.f;
                so.halfExtent = 1e9f;
                // 2 m grid: a 3x3 window is a 4 m crown spacing, which is a
                // town tree. The 5x5 default is a plantation rule and it halves
                // the garden crowns.
                so.windowRadius = 1;
                // Cranes, spires, masts and ship superstructure are the tall
                // end of a CHM over a HARBOUR. On a fjord a 30 m peak is a
                // spruce, so this cap belongs to the pack, not to the detector.
                so.maxCanopyHeight = 28.f;
                so.reject = [this, fpMask, deckMask](float x, float z) {
                    if (fpMask && fpMask->inside(x, z)) return true;
                    if (network_->pavedWeight(x, z, 1.0f) > 0.2f) return true;
                    if (deckMask && deckMask->inside(x, z)) return true;
                    return false;
                };
            }
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
            lo.cap = urbanPack ? o.urbanForestCap : o.forestCap;
            lo.cellSize = 128.f;
            lo.l0Distance = 300.f;
            lo.l1Distance = 800.f;
            lo.l2Keep = 4;// the far tier costs 21% of the frame at full density
            lo.mesh.seaLevel = reg.seaLevel;
            lo.mesh.maxSlopeDeg = so.maxSlopeDeg;      // same gates as the sites,
            lo.mesh.minGroundHeight = so.minGroundHeight;// or the handoff grows new forest
            if (urbanPack) {
                // A town's tall trees are limes, maples, chestnuts and rowans,
                // not Norway spruce: spruce needs BOTH height above sea and a
                // stand around it. Aksla's plantation still gets conifers; the
                // 16 m tree in a churchyard at 12 m becomes the broadleaf it
                // is. Both terms are 0 on a fjord pack = the old rule.
                lo.spruceMinElevation = 60.f;
                lo.spruceMinStandHeight = 12.f;
            }

            if (!urbanPack) {
                auto forest = Group::create();
                forest->name = "geo_canopy_forest";
                // Bases come from the PROVIDER (relief + road carve + bathymetry
                // included), not the raw DEM, or every trunk floats or sinks.
                const auto st = vegetation::buildCanopyForestLod(*forest, sites, species, canopyMat,
                                                                 pack_.canopy, pack_.grid,
                                                                 prov_.height, lo);
                stats_.forestSites = st.sites;
                stats_.forestCells = st.cells;
                add(forest);
                return;
            }

            // ── the town's forest is STREAMED ────────────────────────────────
            // Pack-wide sites, but geometry only where the camera is. The cell
            // is the unit of object count and the object count is the frame: a
            // 2.5 km urban ROI built in one go is ~1500 cells and several
            // thousand draws before a triangle is considered. Beyond the fine
            // ring a whole 3x3 block collapses to one coarse cell — 9x fewer
            // objects for crowns that are 2-4 px wide.
            stats_.forestSites = static_cast<int>(sites.size());
            lo.farCellSize = lo.cellSize * 3.f;

            // Sites binned on the fine cell ONCE, so a cell build is a lookup
            // and not a scan over the whole pack.
            auto grid = std::make_shared<std::unordered_map<std::int64_t,
                                                            std::vector<vegetation::TreeSite>>>();
            const float gcs = lo.cellSize;
            const auto gkey = [](int cx, int cz) {
                return (static_cast<std::int64_t>(cx) << 32) ^ static_cast<std::uint32_t>(cz);
            };
            for (const auto& s : sites)
                (*grid)[gkey(static_cast<int>(std::floor(s.x / gcs)),
                             static_cast<int>(std::floor(s.z / gcs)))]
                        .push_back(s);

            auto speciesPtr = std::make_shared<std::array<vegetation::SpeciesVariants, 3>>(species);
            auto heightFn = prov_.height;
            auto builder = [this, grid, gkey, speciesPtr, canopyMat, heightFn, lo](
                                   int level, int cx, int cz, float) -> std::shared_ptr<Object3D> {
                std::vector<vegetation::TreeSite> sub;
                const int span = level ? 3 : 1;
                const int bx = level ? cx * 3 : cx, bz = level ? cz * 3 : cz;
                for (int i = 0; i < span; ++i)
                    for (int j = 0; j < span; ++j) {
                        auto it = grid->find(gkey(bx + j, bz + i));
                        if (it != grid->end())
                            sub.insert(sub.end(), it->second.begin(), it->second.end());
                    }
                if (sub.empty()) return nullptr;
                auto g = Group::create();
                g->name = level ? "forest_coarse" : "forest_fine";
                vegetation::ForestLodOptions co = lo;
                co.coarseOnly = level != 0;
                // Per-cell seed: the yaw stream restarts inside every build, so
                // a shared seed gives every cell the same rotation sequence — a
                // rhythm the eye finds on a hillside.
                co.seed = lo.seed ^ (static_cast<unsigned int>(cx) * 73856093u) ^
                          (static_cast<unsigned int>(cz) * 19349663u) ^
                          (static_cast<unsigned int>(level) * 83492791u);
                vegetation::buildCanopyForestLod(*g, sub, *speciesPtr, canopyMat, pack_.canopy,
                                                 pack_.grid, heightFn, co);
                return g;
            };

            CellStreamerOptions cso;
            cso.fineCellSize = lo.cellSize;
            cso.fineRadius = 800.f;
            cso.coarseRadius = 1600.f;
            cso.maxCellBuildsPerFrame = o.streamBudget;
            forestStream_ = CellStreamer::create(builder, cso);
            forestStream_->name = "geo_forest_stream";
            add(forestStream_);
            // The startup ring, in one go: a headless render or a --shot must
            // not photograph a half-grown world before update() is ever called.
            forestStream_->update(o.focus);
            stats_.forestCells = forestStream_->stats().active;
        }

        // ── urban props: pier decks, parked cars, moored boats ───────────────
        // Everything here is placed from the SURVEY, never scattered, and every
        // placement passes the same gates the trees do (footprint, pavement,
        // sea). The placement pass runs pack-wide — a lot 2 km away is still a
        // lot — and only the car GEOMETRY is streamed: a placement is 32 bytes,
        // the triangles for the same car are ~4 kB, and that ratio is the whole
        // reason the two are separated. Decks and boats are not streamed at
        // all; they are a few hundred objects for the whole pack and they are
        // part of the LAND.
        void buildUrbanLayer(const GeoSceneOptions& o, const GeoRegion& reg,
                             const GeoTerrainOptions& gopt) {
            // 1 m dilation only: a car parked hard against a wall is normal, a
            // car INSIDE the wall is the failure this gate exists for.
            const auto propFp = buildFootprintMask(pack_, 1.f, 2.f);
            const auto propUrban = gopt.paintUrban ? buildUrbanMask(pack_, gopt) : nullptr;
            // Mown ground. Footways, service roads and thin parking ribbons run
            // straight through the town's parks, and a kerb car placed off one
            // of them stands on the grass; 2 m of dilation covers the shoulder.
            const auto propParks =
                    buildLandUseMask(pack_, {"grass", "pitch", "playground", "cemetery"}, 2.f, 2.f);

            UrbanPropsOptions po;
            po.seaLevel = reg.seaLevel;
            po.centerX = 0.f;// pack-wide placement; the streamer bounds the draws
            po.centerZ = 0.f;
            po.halfExtent = 1e9f;
            po.cellSize = o.propsCellSize;
            po.cars = o.cars;
            po.boats = o.boats;
            po.decks = o.decks;
            po.carMaxSlope = o.carMaxSlope;
            po.urban = propUrban.get();
            po.footprints = propFp.get();
            po.parks = propParks.get();
            po.ground = prov_.height;

            auto props = Group::create();
            props->name = "geo_urban_props";
            carField_ = std::make_shared<UrbanCarField>();
            const auto ps = buildUrbanProps(*props, pack_, *network_, po, carField_.get());
            add(props);
            stats_.cars = carField_->count();
            stats_.boats = ps.boats;
            stats_.deckRuns = ps.deckLines;
            stats_.carsRejectedSlope = ps.rejectSlope;

            if (!o.cars) return;
            auto carMat = makeUrbanCarMaterial();
            CellStreamerOptions pso;
            pso.fineCellSize = po.cellSize;
            pso.fineRadius = o.propsExtent;
            pso.coarseRadius = 0.f;// a car has no far tier: past the ring, nothing
            pso.maxCellBuildsPerFrame = o.streamBudget;
            auto carField = carField_;
            carStream_ = CellStreamer::create(
                    [carField, carMat](int, int cx, int cz, float) -> std::shared_ptr<Object3D> {
                        const auto* cars = carField->at(cx, cz);
                        if (!cars) return nullptr;
                        return buildCarCellMesh(*cars, carMat,
                                                "cars_" + std::to_string(cx) + "_" +
                                                        std::to_string(cz));
                    },
                    pso);
            carStream_->name = "geo_car_stream";
            add(carStream_);
            carStream_->update(o.focus);// the startup ring, like the forest's
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
        // Both streamers' builders capture `this` and read pack_ / prov_, so
        // they must be destroyed BEFORE those — declaring them here, after the
        // pack, is what guarantees it.
        std::shared_ptr<CellStreamer> forestStream_;
        std::shared_ptr<CellStreamer> carStream_;
        std::shared_ptr<UrbanCarField> carField_;
        // The ribbon chunk group and its per-chunk cull data. Raw Object3D* into
        // roadChunks_->children is safe because the group is a member and is
        // never rebuilt: the chunks are built once and only toggled.
        std::shared_ptr<Group> roadChunks_;
        std::vector<std::pair<Object3D*, Vector3>> chunkCenters_;
        std::vector<float> chunkRadii_;
        float ribbonDist_ = 600.f;
        Stats stats_;
    };

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_GEOSCENE_HPP
