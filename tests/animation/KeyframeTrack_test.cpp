
#include <catch2/catch_test_macros.hpp>

#include "threepp/animation/tracks/NumberKeyframeTrack.hpp"
#include "threepp/animation/tracks/VectorKeyframeTrack.hpp"

#include <stdexcept>

using namespace threepp;

TEST_CASE("KeyframeTrack rejects empty times") {

    // getValueSize() divides by the time count and interpolation copies a
    // sample at size()-1; an empty track was UB deferred to first playback.
    CHECK_THROWS_AS(NumberKeyframeTrack("t", {}, {}), std::invalid_argument);
    CHECK_THROWS_AS(NumberKeyframeTrack("t", {}, {1.f}), std::invalid_argument);
}

TEST_CASE("KeyframeTrack rejects mismatched value counts") {

    CHECK_THROWS_AS(NumberKeyframeTrack("t", {0.f, 1.f}, {}), std::invalid_argument);
    // 2 times cannot carry 3 values in any integral stride
    CHECK_THROWS_AS(VectorKeyframeTrack("t", {0.f, 1.f}, {1.f, 2.f, 3.f}), std::invalid_argument);
}

TEST_CASE("KeyframeTrack accepts well-formed tracks") {

    CHECK_NOTHROW(NumberKeyframeTrack("t", {0.f, 1.f}, {1.f, 2.f}));
    CHECK_NOTHROW(VectorKeyframeTrack("t", {0.f, 1.f}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f}));
}
