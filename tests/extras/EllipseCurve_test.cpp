#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/curves/EllipseCurve.hpp"
#include "threepp/math/MathUtils.hpp"

#include <cmath>

using namespace threepp;
using Catch::Matchers::WithinAbs;

TEST_CASE("A full turn stays a full turn whatever the start angle") {

    // for these start angles, a + 2 PI - a comes out a few ulps above or below 2 PI
    for (float start : {2.2f, 2.3f, 9.94369602f}) {
        for (float turn : {math::TWO_PI, -math::TWO_PI}) {
            for (bool clockwise : {false, true}) {

                const float end = start + turn;
                EllipseCurve curve(0, 0, 10, 10, start, end, clockwise, 0);
                const float direction = clockwise ? -1.f : 1.f;

                for (float t : {0.f, 0.25f, 0.5f, 1.f}) {

                    Vector2 p;
                    curve.getPoint(t, p);

                    INFO("start " << start << ", end " << end << ", clockwise " << clockwise << ", t " << t);
                    CHECK_THAT(p.x, WithinAbs(10 * std::cos(start + direction * t * math::TWO_PI), 1e-3));
                    CHECK_THAT(p.y, WithinAbs(10 * std::sin(start + direction * t * math::TWO_PI), 1e-3));
                }
            }
        }
    }
}

TEST_CASE("Equal angles and small arcs are left alone") {

    Vector2 a;
    Vector2 b;

    EllipseCurve point(0, 0, 10, 10, 1, 1, false, 0);
    point.getPoint(0, a);
    point.getPoint(0.5f, b);
    CHECK(a.distanceTo(b) == 0);

    EllipseCurve arc(0, 0, 10, 10, 1, 1.01f, false, 0);
    arc.getPoint(0, a);
    arc.getPoint(1, b);
    CHECK_THAT(a.distanceTo(b), WithinAbs(0.1, 1e-3));
}
