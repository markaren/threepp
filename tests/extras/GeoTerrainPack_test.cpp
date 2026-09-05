#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/terrain/GeoTerrainPack.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::terrain;

namespace {

    // Smallest pack the loader accepts: dim >= 4, heights.f32 exactly
    // dim*dim*4 bytes. Everything else here is the new Phase-A payload.
    std::filesystem::path writeTinyPack() {
        namespace fs = std::filesystem;
        const fs::path dir = fs::temp_directory_path() / "threepp_geopack_test";
        fs::remove_all(dir);
        fs::create_directories(dir);

        constexpr int dim = 8;
        std::vector<float> h(static_cast<size_t>(dim) * dim, 10.f);
        {
            std::ofstream f(dir / "heights.f32", std::ios::binary);
            f.write(reinterpret_cast<const char*>(h.data()),
                    static_cast<std::streamsize>(h.size() * sizeof(float)));
        }
        const auto write = [&dir](const char* name, const std::string& text) {
            std::ofstream f(dir / name, std::ios::binary);
            f << text;
        };
        write("region.json",
              R"({"version":1,"name":"tiny","crs":"EPSG:25833","originEasting":0,)"
              R"("originNorthing":0,"worldSize":70.0,"dim":8,"heightMin":10,)"
              R"("heightMax":10,"seaLevel":0,"heights":"heights.f32",)"
              R"("roads":"roads.json","buildings":"buildings.json",)"
              R"("landuse":"landuse.json"})");
        write("roads.json", R"({"version":1,"roads":[]})");
        // One building carrying a measured roof block, one without: the second
        // proves an old-style entry still loads and reports hasRoof() == false.
        write("buildings.json",
              R"({"version":1,"buildings":[)"
              R"({"id":"w1","type":"house","height":8.4,"heightSource":"roof",)"
              R"("groundMin":10,"groundMax":10,)"
              R"("roof":{"kind":"gabled","eave":5.2,"ridge":8.4,"axis":[0.0,2.0]},)"
              R"("outer":[[-5,-4],[5,-4],[5,4],[-5,4]]},)"
              R"({"id":"w2","type":"garage","height":3.0,"heightSource":"default",)"
              R"("groundMin":10,"groundMax":10,)"
              R"("outer":[[10,10],[16,10],[16,16],[10,16]]}]})");
        write("landuse.json",
              R"({"version":1,)"
              R"("polygons":[{"class":"parking","id":"w9",)"
              R"("outer":[[0,0],[10,0],[10,10],[0,10]],)"
              R"("holes":[[[2,2],[2,4],[4,4],[4,2]]]}],)"
              R"("lines":[{"class":"footway","id":"w8","width":2.5,)"
              R"("points":[[0,0],[5,5],[9,9]]}],)"
              R"("points":[{"class":"tree","x":-3.5,"z":7.25}]})");
        return dir;
    }

}// namespace

// The roof block is the whole point of the 1 m DOM pass: it must survive the
// JSON round-trip with the axis normalised (the producer rounds to 4 decimals,
// so the loader cannot assume unit length) and eave clamped under ridge.
TEST_CASE("GeoTerrainPack loads a measured roof block", "[terrain]") {
    const auto dir = writeTinyPack();
    const auto pack = GeoTerrainPack::load(dir.string());

    REQUIRE(pack.buildings.size() == 2);
    const GeoBuilding& b = pack.buildings[0];
    REQUIRE(b.hasRoof());
    CHECK(b.roof.kind == "gabled");
    CHECK(b.roof.ridge == 8.4f);
    CHECK(b.roof.eave == 5.2f);
    // axis [0,2] arrives normalised to (0,1).
    CHECK(std::abs(b.roof.axis.x) < 1e-5f);
    CHECK(std::abs(b.roof.axis.y - 1.f) < 1e-5f);
    // A building with no roof key keeps the consumer's own heuristic.
    CHECK_FALSE(pack.buildings[1].hasRoof());

    std::filesystem::remove_all(dir);
}

// landuse.json is optional and its classes are free-form strings, so the load
// contract is simply "the three lists arrive intact, with holes and widths".
TEST_CASE("GeoTerrainPack loads OSM land use", "[terrain]") {
    const auto dir = writeTinyPack();
    const auto pack = GeoTerrainPack::load(dir.string());

    REQUIRE(pack.hasLandUse());
    REQUIRE(pack.landuse.polygons.size() == 1);
    REQUIRE(pack.landuse.lines.size() == 1);
    REQUIRE(pack.landuse.points.size() == 1);
    CHECK(pack.landuse.polygons[0].cls == "parking");
    CHECK(pack.landuse.polygons[0].outer.size() == 4);
    CHECK(pack.landuse.polygons[0].holes.size() == 1);
    CHECK(pack.landuse.lines[0].cls == "footway");
    CHECK(pack.landuse.lines[0].width == 2.5f);
    CHECK(pack.landuse.lines[0].points.size() == 3);
    CHECK(pack.landuse.points[0].cls == "tree");
    CHECK(pack.landuse.points[0].pos.y == 7.25f);

    std::filesystem::remove_all(dir);
}
