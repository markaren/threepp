
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "threepp/math/Vector3.hpp"

#include <array>
#include <vector>

using namespace threepp;

TEST_CASE("add") {

    Vector3 v1{1, 2, 3};
    Vector3 v2{9, 8, 7};

    Vector3 result = v1.add(v2);

    REQUIRE(result == Vector3{10, 10, 10});
}

TEST_CASE("subtract") {

    Vector3 v1{1, 2, 3};
    Vector3 v2{9, 8, 7};

    Vector3 result = v1.sub(v2);

    REQUIRE(result == Vector3{-8, -6, -4});
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
