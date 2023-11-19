
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Vector3.hpp"

#include <array>
#include <vector>

using namespace threepp;

namespace {

    const float x = 2;
    const float y = 3;
    const float z = 4;

}// namespace

TEST_CASE("add") {

    Vector3 a{x, y, z};
    Vector3 b{-x, -y, -z};

    a.add(b);

    CHECK(a.x == 0);
    CHECK(a.y == 0);
    CHECK(a.z == 0);

    auto c = Vector3().addVectors(b, b);
    CHECK(c.x == -2 * x);
    CHECK(c.y == -2 * y);
    CHECK(c.z == -2 * z);
}

TEST_CASE("sub") {

    Vector3 a{x, y, z};
    Vector3 b{-x, -y, -z};

    a.sub(b);

    CHECK(a.x == 2 * x);
    CHECK(a.y == 2 * y);
    CHECK(a.z == 2 * z);

    auto c = Vector3().subVectors(b, b);
    CHECK(c.x == 0);
    CHECK(c.y == 0);
    CHECK(c.z == 0);
}

TEST_CASE("dot") {

    Vector3 a(x, y, z);
    Vector3 b(-x, -y, -z);
    Vector3 c;

    float result = a.dot(b);
    CHECK(result == (-x * x - y * y - z * z));

    result = a.dot(c);
    CHECK(result == 0);
}

TEST_CASE("angleTo") {

    Vector3 a(0, -0.18851655680720186f, 0.9820700116639124f);
    Vector3 b(0, 0.18851655680720186f, -0.9820700116639124f);

    CHECK(a.angleTo(a) == 0);
    CHECK_THAT(a.angleTo(b), Catch::Matchers::WithinRel(math::PI));

    Vector3 _x(1, 0, 0);
    Vector3 _y(0, 1, 0);
    Vector3 _z(0, 0, 1);

    CHECK_THAT(_x.angleTo(_y), Catch::Matchers::WithinRel(math::PI / 2));
    CHECK_THAT(_x.angleTo(_z), Catch::Matchers::WithinRel(math::PI / 2));
    CHECK_THAT(_z.angleTo(_x), Catch::Matchers::WithinRel(math::PI / 2));

    CHECK(std::abs(_x.angleTo(Vector3(1, 1, 0)) - (math::PI / 4)) < 0.0000001);
}

TEST_CASE("from arraylike") {

    std::array<float, 3> arr{1, 2, 3};
    std::vector<float> v{arr.begin(), arr.end()};
    for (auto& value : v) { value += 1; }

    Vector3 result;

    result.fromArray(arr);
    REQUIRE(result == Vector3{arr[0], arr[1], arr[2]});

    result.fromArray(v);
    REQUIRE(result == Vector3{v[0], v[1], v[2]});
}

TEST_CASE("equals") {

    Vector3 v1;
    Vector3 v2;

    REQUIRE(v1 == v2);

    v1.set(0.01, 0, 0);

    REQUIRE(v1 != v2);

    v1.set(1, 1, 1);
    v2.copy(v1);

    REQUIRE(v1 == v2);
}
