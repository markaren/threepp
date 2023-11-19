#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/loaders/FontLoader.hpp"

using namespace threepp;

TEST_CASE("Test FontLoader") {

    FontLoader loader;
    auto font = loader.load(std::string(DATA_FOLDER) + "/fonts/optimer_regular.typeface.json");
    auto& data = font->data();

    REQUIRE(font);

    auto o = data.glyphs['o'];

    CHECK(data.familyName == "Optimer");
    CHECK_THAT(data.boundingBox.xMin, Catch::Matchers::WithinRel(-71.));
    CHECK_THAT(data.boundingBox.xMax, Catch::Matchers::WithinRel(1511.));
    CHECK_THAT(data.boundingBox.yMin, Catch::Matchers::WithinRel(-373.75));
    CHECK_THAT(data.boundingBox.yMax, Catch::Matchers::WithinRel(1267.));

    CHECK_THAT(o.x_min, Catch::Matchers::WithinRel(41.));
    CHECK_THAT(o.x_max, Catch::Matchers::WithinRel(710.));
    CHECK(o.ha == 753);
}
