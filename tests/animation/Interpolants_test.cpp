// The four interpolants underneath every KeyframeTrack.
//
// These are pure math with exact expected values, so they are worth pinning
// directly rather than only through the animation stack: a regression here shows
// up as "animation looks slightly wrong", which is near-impossible to bisect from
// a scene.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/math/Interpolant.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/interpolants/CubicInterpolant.hpp"
#include "threepp/math/interpolants/DiscreteInterpolant.hpp"
#include "threepp/math/interpolants/LinearInterpolant.hpp"
#include "threepp/math/interpolants/QuaternionLinearInterpolant.hpp"

#include <cmath>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;

TEST_CASE("LinearInterpolant interpolates between keys") {

    // Two keys at t=0 and t=1, one scalar each: 0 -> 10.
    const Sample times{0.f, 1.f};
    const Sample values{0.f, 10.f};
    Sample result;
    LinearInterpolant interp(times, values, 1, &result);

    CHECK_THAT(interp.evaluate(0.f)[0], WithinAbs(0.f, 1e-5));
    CHECK_THAT(interp.evaluate(0.25f)[0], WithinAbs(2.5f, 1e-5));
    CHECK_THAT(interp.evaluate(0.5f)[0], WithinAbs(5.f, 1e-5));
    CHECK_THAT(interp.evaluate(1.f)[0], WithinAbs(10.f, 1e-5));
}

TEST_CASE("LinearInterpolant clamps outside the key range") {

    const Sample times{1.f, 2.f};
    const Sample values{5.f, 7.f};
    Sample result;
    LinearInterpolant interp(times, values, 1, &result);

    // Before the first key and after the last, the endpoint value is held.
    CHECK_THAT(interp.evaluate(-10.f)[0], WithinAbs(5.f, 1e-5));
    CHECK_THAT(interp.evaluate(0.f)[0], WithinAbs(5.f, 1e-5));
    CHECK_THAT(interp.evaluate(99.f)[0], WithinAbs(7.f, 1e-5));
}

TEST_CASE("LinearInterpolant handles multi-component values") {

    // Three keys, 3 floats each — the shape a `.position` track has.
    const Sample times{0.f, 1.f, 2.f};
    const Sample values{
            0.f, 0.f, 0.f,
            1.f, 2.f, 3.f,
            2.f, 4.f, 6.f};
    Sample result;
    LinearInterpolant interp(times, values, 3, &result);

    const auto mid = interp.evaluate(0.5f);
    REQUIRE(mid.size() == 3);
    CHECK_THAT(mid[0], WithinAbs(0.5f, 1e-5));
    CHECK_THAT(mid[1], WithinAbs(1.0f, 1e-5));
    CHECK_THAT(mid[2], WithinAbs(1.5f, 1e-5));

    const auto exact = interp.evaluate(2.f);
    CHECK_THAT(exact[0], WithinAbs(2.f, 1e-5));
    CHECK_THAT(exact[2], WithinAbs(6.f, 1e-5));
}

TEST_CASE("DiscreteInterpolant holds the previous key") {

    const Sample times{0.f, 1.f, 2.f};
    const Sample values{10.f, 20.f, 30.f};
    Sample result;
    DiscreteInterpolant interp(times, values, 1, &result);

    // No blending: the value steps at each key and holds until the next.
    CHECK_THAT(interp.evaluate(0.0f)[0], WithinAbs(10.f, 1e-5));
    CHECK_THAT(interp.evaluate(0.9f)[0], WithinAbs(10.f, 1e-5));
    CHECK_THAT(interp.evaluate(1.0f)[0], WithinAbs(20.f, 1e-5));
    CHECK_THAT(interp.evaluate(1.9f)[0], WithinAbs(20.f, 1e-5));
    CHECK_THAT(interp.evaluate(2.0f)[0], WithinAbs(30.f, 1e-5));
}

TEST_CASE("CubicInterpolant passes exactly through its keyframes") {

    const Sample times{0.f, 1.f, 2.f, 3.f};
    const Sample values{0.f, 1.f, 0.f, 1.f};
    Sample result;
    CubicInterpolant interp(times, values, 1, &result);

    // A smooth curve may overshoot between keys, but it must hit the keys.
    for (std::size_t i = 0; i < times.size(); ++i) {
        INFO("key " << i << " at t=" << times[i]);
        CHECK_THAT(interp.evaluate(times[i])[0], WithinAbs(values[i], 1e-4));
    }
}

TEST_CASE("CubicInterpolant stays bounded between keys") {

    const Sample times{0.f, 1.f, 2.f, 3.f};
    const Sample values{0.f, 1.f, 4.f, 9.f};
    Sample result;
    CubicInterpolant interp(times, values, 1, &result);

    // A cubic may legitimately overshoot between keys, so this does not demand
    // monotonicity — only that the curve stays in the neighbourhood of its data.
    // Before the ending-settings fix the boundary intervals read indeterminate
    // memory and this produced values in the thousands from data bounded by 9.
    constexpr float lo = 0.f, hi = 9.f;
    constexpr float slack = 2.f * (hi - lo);

    for (int i = 0; i <= 300; ++i) {
        const float t = 3.f * static_cast<float>(i) / 300.f;
        const float v = interp.evaluate(t)[0];
        INFO("t=" << t << " v=" << v);
        REQUIRE(std::isfinite(v));
        REQUIRE(v > lo - slack);
        REQUIRE(v < hi + slack);
    }
}

TEST_CASE("QuaternionLinearInterpolant slerps and stays unit length") {

    // Identity -> 90 degrees about Y.
    const float s = std::sqrt(0.5f);
    const Sample times{0.f, 1.f};
    const Sample values{
            0.f, 0.f, 0.f, 1.f,
            0.f, s, 0.f, s};
    Sample result;
    QuaternionLinearInterpolant interp(times, values, 4, &result);

    // Endpoints exact.
    const auto start = interp.evaluate(0.f);
    CHECK_THAT(start[3], WithinAbs(1.f, 1e-5));

    const auto end = interp.evaluate(1.f);
    CHECK_THAT(end[1], WithinAbs(s, 1e-5));
    CHECK_THAT(end[3], WithinAbs(s, 1e-5));

    // Halfway is 45 degrees about Y, and every sample stays normalised —
    // a plain lerp would shorten the quaternion in the middle.
    const auto mid = interp.evaluate(0.5f);
    const float len = std::sqrt(mid[0] * mid[0] + mid[1] * mid[1] + mid[2] * mid[2] + mid[3] * mid[3]);
    CHECK_THAT(len, WithinAbs(1.f, 1e-4));
    CHECK_THAT(mid[1], WithinAbs(std::sin(math::PI / 8.f), 1e-3));
    CHECK_THAT(mid[3], WithinAbs(std::cos(math::PI / 8.f), 1e-3));

    for (int i = 0; i <= 20; ++i) {
        const float t = static_cast<float>(i) / 20.f;
        const auto q = interp.evaluate(t);
        const float l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        INFO("t=" << t << " |q|=" << l);
        REQUIRE_THAT(l, WithinAbs(1.f, 1e-4));
    }
}
