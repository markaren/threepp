
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Vector2.hpp"

#include <array>
#include <vector>

using namespace threepp;

namespace {

    const float x = 2;
    const float y = 3;

}// namespace

TEST_CASE("add") {

    Vector2 a{x, y};
    Vector2 b{-x, -y};

    a.add(b);

    CHECK(a.x == 0);
    CHECK(a.y == 0);

    auto c = Vector2().addVectors(b, b);
    CHECK(c.x == -2 * x);
    CHECK(c.y == -2 * y);
}

TEST_CASE("sub") {

    Vector2 a{x, y};
    Vector2 b{-x, -y};

    a.sub(b);

    CHECK(a.x == 2 * x);
    CHECK(a.y == 2 * y);

    auto c = Vector2().subVectors(b, b);
    CHECK(c.x == 0);
    CHECK(c.y == 0);
}

TEST_CASE("dot") {

    Vector2 a(x, y);
    Vector2 b(-x, -y);
    Vector2 c;

    float result = a.dot(b);
    CHECK(result == (-x * x - y * y));

    result = a.dot(c);
    CHECK(result == 0);
}

TEST_CASE("angleTo") {

    Vector2 a(-0.18851655680720186f, 0.9820700116639124f);
    Vector2 b(0.18851655680720186f, -0.9820700116639124f);

    CHECK(a.angleTo(a) == 0);
    CHECK_THAT(a.angleTo(b), Catch::Matchers::WithinRel(math::PI));

    Vector2 _x(1, 0);
    Vector2 _y(0, 1);

    CHECK_THAT(_x.angleTo(_y), Catch::Matchers::WithinRel(math::PI / 2));
    CHECK_THAT(_y.angleTo(_x), Catch::Matchers::WithinRel(math::PI / 2));

    CHECK(std::abs(_x.angleTo(Vector2(1, 1)) - (math::PI / 4)) < 0.0000001);
}

TEST_CASE("from arraylike") {

    std::array<float, 2> arr{1, 2};
    std::vector<float> v{arr.begin(), arr.end()};
    for (auto& value : v) { value += 1; }

    Vector2 result;

    result.fromArray(arr);
    REQUIRE(result == Vector2{arr[0], arr[1]});

    result.fromArray(v);
    REQUIRE(result == Vector2{v[0], v[1]});
}

TEST_CASE("equals") {

    Vector2 v1;
    Vector2 v2;

    REQUIRE(v1 == v2);

    v1.set(0.01, 0);

    REQUIRE(v1 != v2);

    v1.set(1, 1);
    v2.copy(v1);

    REQUIRE(v1 == v2);
}
