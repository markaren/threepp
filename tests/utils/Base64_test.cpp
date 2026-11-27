#include <catch2/catch_test_macros.hpp>

#include "threepp/utils/Base64.hpp"

#include <string>
#include <vector>

using namespace threepp;

TEST_CASE("base64 round trip") {

    const std::vector<uint8_t> bytes{0, 1, 2, 250, 251, 252, 253, 254, 255, 42};
    const std::string encoded = utils::base64Encode(bytes);
    CHECK(encoded == "AAEC+vv8/f7/Kg==");
    CHECK(utils::base64Decode(encoded) == bytes);
    // Padding is optional on the way in.
    CHECK(utils::base64Decode("AAEC+vv8/f7/Kg") == bytes);
}

TEST_CASE("decodeDataUri") {

    SECTION("a base64 payload") {
        const auto bytes = utils::decodeDataUri("data:application/octet-stream;base64,AAEC");
        REQUIRE(bytes.has_value());
        CHECK(*bytes == std::vector<uint8_t>{0, 1, 2});
    }

    SECTION("no media type, empty payload") {
        const auto bytes = utils::decodeDataUri("data:,");
        REQUIRE(bytes.has_value());
        CHECK(bytes->empty());
    }

    SECTION("no comma: not a payload") {
        // Decoding from the start would have read "data" as base64.
        CHECK_FALSE(utils::decodeDataUri("data:application/octet-stream;base64").has_value());
    }

    SECTION("not a data: URI") {
        CHECK_FALSE(utils::decodeDataUri("textures/albedo.png").has_value());
        CHECK_FALSE(utils::decodeDataUri("").has_value());
    }
}
