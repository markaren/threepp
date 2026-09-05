// Real-world geodata "region pack" loader (elevation + roads).
//
// A region pack is a small self-describing directory produced by an external
// fetch/bake tool (e.g. the Norwegian Kartverket DTM + NVDB road pipeline). It
// carries a square, north-up float32 elevation grid plus a set of road
// polylines already resolved into the pack's LOCAL world frame, so the C++ side
// needs no CRS math or reprojection — it just loads and renders.
//
// FROZEN pack format (the contract with the producer):
//   <pack>/region.json  metadata (below)
//   <pack>/heights.f32   raw little-endian float32, dim*dim, row-major
//                        index = iz*dim + ix; ix=0 ↔ x=-worldSize/2 (west),
//                        iz=0 ↔ z=-worldSize/2 (north edge). World mapping is
//                        x=east, z=-north — this layout drops straight into
//                        terrain::HeightGrid(heights, dim, worldSize).
//   <pack>/roads.json    { version, roads:[ { id, category, typeVeg, width,
//                        points:[[x,y,z],...] } ] } — points already in local
//                        world coords, y = road height (metres).
//   <pack>/buildings.json (optional) { version, buildings:[ { id, type,
//                        height, heightSource, groundMin, groundMax, levels?,
//                        outer:[[x,z],...], holes?:[[[x,z],...],...] } ] } —
//                        extruded-footprint buildings. Rings are OPEN (first
//                        point not repeated), outer wound to POSITIVE shoelace
//                        area in (x,z), holes negative. Roof top sits at
//                        groundMin + height (slope relief already folded in).
//                        Each building may carry an optional roof block
//                        { kind, eave, ridge, axis:[dx,dz] } measured from the
//                        1 m DOM (see GeoRoof).
//   <pack>/landuse.json  (optional) { version, polygons:[{class,id,outer,
//                        holes?}], lines:[{class,id,width,points}],
//                        points:[{class,x,z}] } — OSM land use in the same
//                        local frame and winding as buildings.json.
//
//   <pack>/texture.png   (optional) square RGB basemap drape (Kartverket topo
//                        raster WMS). Row 0 = north edge, so
//                        u=(x+worldSize/2)/worldSize, v=(z+worldSize/2)/worldSize
//                        maps local world coords onto the image. Not consumed
//                        by load() — the terrain albedo is the procedural
//                        splat (GeoTerrain.hpp); the file is for external
//                        consumers.
//
// region.json fields: version, name, crs, originEasting, originNorthing,
// worldSize, dim, heightMin, heightMax, seaLevel, heights, roads, attribution,
// buildings (optional), texture + textureLayer (optional).
//
// load() reads all three files, validates that heights.f32 is exactly
// dim*dim*4 bytes, and returns a fully-populated GeoTerrainPack (a HeightGrid
// ready for TileTerrain, the region metadata, and the road list). It throws
// std::runtime_error with a descriptive message on any missing / malformed /
// short file. The JSON parse lives in the .cpp (nlohmann is a PRIVATE threepp
// dependency), so this header stays dependency-free beyond threepp core.

#ifndef THREEPP_EXTRAS_TERRAIN_GEOTERRAINPACK_HPP
#define THREEPP_EXTRAS_TERRAIN_GEOTERRAINPACK_HPP

#include "threepp/extras/terrain/TerrainTiles.hpp"// terrain::HeightGrid
#include "threepp/math/Vector2.hpp"
#include "threepp/math/Vector3.hpp"

#include <string>
#include <vector>

namespace threepp::terrain {

    // One road polyline from the pack. `points` are already in local world
    // coordinates (metres, Y-up); point.y is the road surface elevation.
    struct GeoRoad {
        std::string id;
        std::string category;      // "E" (europavei) | "R" | "F" | "K" | "P" | "S"
        std::string typeVeg;       // NVDB road-type label/code (informational)
        float width = 6.f;         // total carriageway width (m)
        std::vector<Vector3> points;// centerline, local world coords, y = height
    };

    // ── Measured roof block (optional per building) ─────────────────────────
    //
    // The fetcher measures these four numbers off the 1 m surface model (DOM),
    // which resolves a ridge the pack's own 2 m grid smears into the eave:
    //   kind   "flat" | "gabled" | "hipped" (empty = not measured)
    //   eave   p15 of the footprint's interior nDSM, metres above groundMin
    //   ridge  p95 of the same (== eave for "flat"), the rise capped at 50°
    //          over half the footprint's short side; the building's `height`
    //   axis   unit ridge direction in (x, z); (1,0) for "flat"
    //   tower  the uncapped rise was > 1.5x that cap: a spire, tower or mast
    //          stands here, and the roof planes stop at the capped ridge
    // Absent on old packs, on footprints too small to sample (< 12 interior
    // 1 m cells), and wherever the lidar predates the building or the
    // footprint is mislocated (there the pack keeps the old height chain), so
    // every consumer must check hasRoof() and keep its own shape heuristic.
    struct GeoRoof {
        std::string kind;
        float eave = 0.f;
        float ridge = 0.f;
        Vector2 axis{1.f, 0.f};
        bool tower = false;
    };

    // One building footprint from the pack (OSM-sourced). Rings are OPEN
    // (first point not repeated) in local world metres; Vector2 = (x, z).
    // Extrusion contract: walls rise from groundMin (sink slightly below for
    // slope embedding) to the flat roof at groundMin + height.
    struct GeoBuilding {
        std::string id;          // OSM element ("w<id>" way / "r<id>" relation)
        std::string type;        // OSM building=* value ("house", "garage", ...)
        float height = 6.f;      // roof top above groundMin (m)
        float groundMin = 0.f;   // DTM min under the footprint (m)
        float groundMax = 0.f;   // DTM max under the footprint (m)
        float levels = 0.f;      // OSM building:levels (0 = unknown)
        std::string heightSource;// "tag" | "ndsm" | "levels" | "default"
        std::string colour;      // OSM building:colour (rare; empty = none)
        std::string roofColour;  // OSM roof:colour (rare; empty = none)
        std::string roofShape;   // OSM roof:shape (rare; empty = none)
        GeoRoof roof;            // measured off the 1 m DOM; empty kind = absent
        std::vector<Vector2> outer;              // footprint, +shoelace in (x,z)
        std::vector<std::vector<Vector2>> holes; // courtyards, -shoelace

        [[nodiscard]] bool hasRoof() const { return !roof.kind.empty(); }
    };

    // ── OSM land use (optional landuse.json) ────────────────────────────────
    //
    // Everything the town is made of that is neither a road nor a building:
    // parking lots, gardens, pitches, woods, rock, quays and piers as polygons;
    // footpaths, steps, service drives, hedges, fences and tree rows as lines
    // with a width; single trees as points. Same local frame and winding as
    // buildings.json (rings OPEN, outer +shoelace in (x,z), holes negative).
    // `cls` is the producer's class string, deliberately not an enum: a new
    // OSM class added by the fetcher must not require a loader change, and a
    // consumer simply ignores classes it does not paint.
    struct GeoLandUse {
        struct Polygon {
            std::string cls;
            std::string id;
            std::vector<Vector2> outer;
            std::vector<std::vector<Vector2>> holes;
        };
        struct Line {
            std::string cls;
            std::string id;
            float width = 2.f;// metres, full width of the strip
            std::vector<Vector2> points;
        };
        struct Point {
            std::string cls;
            Vector2 pos;
        };
        std::vector<Polygon> polygons;
        std::vector<Line> lines;
        std::vector<Point> points;

        [[nodiscard]] bool empty() const {
            return polygons.empty() && lines.empty() && points.empty();
        }
    };

    // Region metadata mirroring region.json (minus the file references). Kept
    // separate so callers can inspect georeferencing / attribution without the
    // heavy grid.
    struct GeoRegion {
        int version = 0;
        std::string name;
        std::string crs;             // e.g. "EPSG:25833"
        double originEasting = 0.0;  // pack local (0,0) in the CRS
        double originNorthing = 0.0;
        float worldSize = 0.f;       // metres, square extent
        int dim = 0;                 // heightfield samples per side
        float heightMin = 0.f;       // metres (NN2000)
        float heightMax = 0.f;
        float seaLevel = 0.f;
        std::string attribution;     // data licence / credit to print
    };

    // ── Flow accumulation (D8) ──────────────────────────────────────────────
    //
    // "How much land drains through this point": the single field that tells a
    // shader where water runs. Streams, wet streaks on a rock face and the
    // scree fans below a gully are all the SAME structure at different slopes,
    // so one grid drives all three instead of three hand-tuned noise fields.
    //
    // Returns a HeightGrid (same layout/extent as `dem`, coarser) whose value
    // is log1p(drained area) / log1p(total area) — 0 on a ridge crest, →1 in a
    // main channel. Log because drainage area is scale-free: a linear grid is
    // ~0 everywhere except the few main channels, which paints nothing.
    //
    // `targetCell` is the routing resolution in metres (the DEM is box-averaged
    // down to it). Coarsening is deliberate, not just a speed trick: raw 1 m
    // lidar is full of one-cell pits that terminate D8 routing, and a 2 m box
    // mean removes most of them while keeping gullies. Sinks that survive
    // simply stop accumulating (no depression filling) — on a fjord wall the
    // fall lines are monotone, so this is invisible.
    HeightGrid computeFlowAccumulation(const HeightGrid& dem, float targetCell = 2.f);

    // A loaded region pack: elevation grid + roads + buildings + metadata.
    struct GeoTerrainPack {
        GeoRegion region;
        HeightGrid grid;             // dim×dim, centred at origin, worldSize wide
        std::vector<GeoRoad> roads;
        std::vector<GeoBuilding> buildings;// empty if the pack has none
        GeoLandUse landuse;                // empty if the pack has no landuse.json

        // Canopy height model (DOM − DTM), metres of vegetation above ground,
        // on the SAME node grid as `grid`. Invalid (!valid()) when the pack has
        // no canopy.u8 — every consumer must check. This is measured ground
        // truth for where forest stands, replacing hand-tuned altitude rules.
        HeightGrid canopy;

        // D8 flow accumulation, log-normalised 0..1 (see above). Built at load
        // from `grid`; coarser than `grid`, same world extent. Both grids are
        // read-only after load(), so provider callbacks stay pure/thread-safe.
        HeightGrid flow;

        [[nodiscard]] bool hasCanopy() const { return canopy.valid(); }
        [[nodiscard]] bool hasFlow() const { return flow.valid(); }
        [[nodiscard]] bool hasLandUse() const { return !landuse.empty(); }

        [[nodiscard]] bool valid() const { return region.dim >= 4 && grid.valid(); }

        // Load a pack directory. `path` is the pack folder (containing
        // region.json). Throws std::runtime_error on any I/O or format error.
        static GeoTerrainPack load(const std::string& path);
    };

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_GEOTERRAINPACK_HPP
