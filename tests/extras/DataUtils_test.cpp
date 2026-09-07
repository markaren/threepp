// threepp/extras/DataUtils.hpp — the r129 float32 -> float16 conversion and
// its (threepp-only) inverse.
//
// The round trip is the claim worth pinning: a value that goes through
// toHalfFloat and back must land within half-float precision of where it
// started, for positives, negatives, zero, denormals and the two saturation
// ends. The bit-exact cases below are the ones that would move if somebody
// "simplified" the rounding away.

#include "threepp/extras/DataUtils.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace threepp;
using Catch::Approx;

namespace {

    float roundTrip(float v) {

        return DataUtils::fromHalfFloat(DataUtils::toHalfFloat(v));
    }

    // Half has 11 bits of significand, so the representable spacing near a
    // value is about 2^-10 of it. Anything tighter than that is not a claim
    // about the conversion, it is a claim about luck.
    bool withinHalfPrecision(float v) {

        const float back = roundTrip(v);
        const float tolerance = std::max(std::abs(v) * (1.f / 1024.f), 6e-8f);
        return std::abs(back - v) <= tolerance;
    }

}// namespace


TEST_CASE("DataUtils: the exact values are exact") {

    // Bit patterns from the IEEE 754 binary16 definition, not from running
    // the code and writing down what came out.
    CHECK(DataUtils::toHalfFloat(0.f) == 0x0000);
    CHECK(DataUtils::toHalfFloat(-0.f) == 0x8000);
    CHECK(DataUtils::toHalfFloat(1.f) == 0x3c00);
    CHECK(DataUtils::toHalfFloat(-1.f) == 0xbc00);
    CHECK(DataUtils::toHalfFloat(2.f) == 0x4000);
    CHECK(DataUtils::toHalfFloat(0.5f) == 0x3800);
    CHECK(DataUtils::toHalfFloat(-2.f) == 0xc000);

    // The largest finite half, and the smallest normal one.
    CHECK(DataUtils::toHalfFloat(65504.f) == 0x7bff);
    CHECK(DataUtils::toHalfFloat(6.103515625e-5f) == 0x0400);

    CHECK(DataUtils::fromHalfFloat(0x3c00) == 1.f);
    CHECK(DataUtils::fromHalfFloat(0xbc00) == -1.f);
    CHECK(DataUtils::fromHalfFloat(0x7bff) == 65504.f);
    CHECK(DataUtils::fromHalfFloat(0x0000) == 0.f);
    CHECK(std::signbit(DataUtils::fromHalfFloat(0x8000)));
}

TEST_CASE("DataUtils: the round trip holds across the representable range") {

    const std::vector<float> values{
            0.f, -0.f, 1.f, -1.f, 0.5f, -0.5f, 0.1f, -0.1f,
            3.14159265f, -3.14159265f, 1e-3f, -1e-3f, 1234.f, -1234.f,
            65504.f, -65504.f, 6.103515625e-5f, -6.103515625e-5f,
            0.28209479f,// SH_C0, which is what put this file here
            -0.4886025f};

    for (float v : values) {

        INFO("value " << v << " -> 0x" << std::hex << DataUtils::toHalfFloat(v)
                      << std::dec << " -> " << roundTrip(v));
        CHECK(withinHalfPrecision(v));
        CHECK(std::signbit(roundTrip(v)) == std::signbit(v));
    }
}

TEST_CASE("DataUtils: a sweep of the ordinary range survives") {

    // Deterministic sweep rather than a handful of constants: every step of a
    // logarithmic ramp, both signs.
    for (int i = -40; i <= 40; ++i) {

        const float v = std::pow(1.7f, static_cast<float>(i));
        if (v > 60000.f || v < 1e-4f) continue;// out of half's comfortable range

        INFO("value " << v);
        CHECK(withinHalfPrecision(v));
        CHECK(withinHalfPrecision(-v));
    }
}

TEST_CASE("DataUtils: denormals and underflow go to the nearest half, then to zero") {

    // Inside the denormal range the answer is coarse but not zero...
    const float denormal = 3e-5f;// below the smallest normal half (6.1e-5)
    CHECK(roundTrip(denormal) > 0.f);
    CHECK(std::abs(roundTrip(denormal) - denormal) < 1e-6f);

    // ...and below the smallest denormal (5.96e-8) it is signed zero, which is
    // r129's explicit "e < 103" branch.
    CHECK(roundTrip(1e-9f) == 0.f);
    CHECK(roundTrip(-1e-9f) == 0.f);
    CHECK(std::signbit(roundTrip(-1e-9f)));
}

TEST_CASE("DataUtils: overflow saturates to the largest finite half") {

    // THE DELIBERATE DEVIATION FROM r129, and the reason it is one. r129's
    // overflow branch ORs the float's 23-bit mantissa into a value about to be
    // stored as 16 bits, so 70000 comes back as NEGATIVE infinity and 1e20 as
    // a NaN. Current three.js sidesteps the branch by clamping the input to
    // +/-65504 before converting; this port does the same — half-float
    // texture data should saturate, not turn infinite.
    CHECK(roundTrip(70000.f) == 65504.f);
    CHECK(roundTrip(-70000.f) == -65504.f);
    CHECK(roundTrip(1e20f) == 65504.f);

    CHECK(DataUtils::toHalfFloat(std::numeric_limits<float>::infinity()) == 0x7bff);
    CHECK(DataUtils::toHalfFloat(-std::numeric_limits<float>::infinity()) == 0xfbff);

    // Just under the top the answer is still finite: 65504 is representable
    // and 65519 rounds down onto it.
    CHECK(roundTrip(65504.f) == 65504.f);
    CHECK(roundTrip(65519.f) == 65504.f);
}

TEST_CASE("DataUtils: NaN stays NaN") {

    // Not true of r129 as written, where NaN converts to infinity.
    CHECK(std::isnan(roundTrip(std::numeric_limits<float>::quiet_NaN())));
}

TEST_CASE("DataUtils: rounding is round-to-nearest, not truncation") {

    // Halfway between two representable halves at this magnitude, so a
    // truncating conversion and a rounding one disagree. 1.0 and the next
    // half above it (1 + 2^-10) are 0x3c00 and 0x3c01.
    CHECK(DataUtils::toHalfFloat(1.f + 1.f / 2048.f) == 0x3c01);// rounds up
    CHECK(DataUtils::toHalfFloat(1.f + 1.f / 4096.f) == 0x3c00);// rounds down

    // A value a hair under 1 must not round up to 1. Below 1.0 the spacing
    // halves, so 0x3bff is 1 - 2^-11 and 0x3bfe is 1 - 2^-10.
    CHECK(DataUtils::toHalfFloat(1.f - 1.f / 2048.f) == 0x3bff);
    CHECK(DataUtils::toHalfFloat(1.f - 1.f / 1024.f) == 0x3bfe);
}
