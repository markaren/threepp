#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/FontLoader.hpp"

using namespace threepp;

TEST_CASE("Test FontLoader") {

    FontLoader loader;
    auto data = loader.load(std::string(DATA_FOLDER) + "/fonts/optimer_regular.typeface.json");

    REQUIRE(data);

    auto o = data->glyphs['o'];

    CHECK(data->familyName == "Optimer");
    CHECK(data->boundingBox.xMin == -71);
    CHECK(data->boundingBox.xMax == 1511);
    CHECK(data->boundingBox.yMin == -373.75);
    CHECK(data->boundingBox.yMax == 1267);

    CHECK(o.x_min == 41 );
    CHECK(o.x_max == 710 );
    CHECK(o.ha == 753 );

}
