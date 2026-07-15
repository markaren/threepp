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
//
// region.json fields: version, name, crs, originEasting, originNorthing,
// worldSize, dim, heightMin, heightMax, seaLevel, heights, roads, attribution.
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

    // A loaded region pack: elevation grid + roads + metadata.
    struct GeoTerrainPack {
        GeoRegion region;
        HeightGrid grid;             // dim×dim, centred at origin, worldSize wide
        std::vector<GeoRoad> roads;

        [[nodiscard]] bool valid() const { return region.dim >= 4 && grid.valid(); }

        // Load a pack directory. `path` is the pack folder (containing
        // region.json). Throws std::runtime_error on any I/O or format error.
        static GeoTerrainPack load(const std::string& path);
    };

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_GEOTERRAINPACK_HPP
