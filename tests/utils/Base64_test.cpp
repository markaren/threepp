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
