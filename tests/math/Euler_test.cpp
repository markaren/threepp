
#include <catch2/catch_test_macros.hpp>

#include "threepp/math/Euler.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Quaternion.hpp"

#include "../equals_util.hpp"

using namespace threepp;

namespace {

    const Euler eulerZero(0, 0, 0, Euler::XYZ);
    const Euler eulerAxyz(1, 0, 0, Euler::XYZ);
    const Euler eulerAzyx(0, 1, 0, Euler::ZYX);


}// namespace

TEST_CASE("Instancing") {

    Euler a;
    CHECK(a.equals(eulerZero));
    CHECK(!a.equals(eulerAxyz));
    CHECK(!a.equals(eulerAzyx));
}

TEST_CASE("Quaternion.setFromEuler/Euler.setFromQuaternion") {

    std::vector<Euler> testValues{eulerZero, eulerAxyz, eulerAzyx};
    for (const auto& v : testValues) {

        const auto q = Quaternion().setFromEuler(v);

        const auto v2 = Euler().setFromQuaternion(q, v.getOrder());
        const auto q2 = Quaternion().setFromEuler(v2);
        CHECK(quatEquals(q, q2));
    }
}

TEST_CASE("Matrix4.makeRotationFromEuler/Euler.setFromRotationMatrix") {

    std::vector<Euler> testValues{eulerZero, eulerAxyz, eulerAzyx};
    for (const auto& v : testValues) {

        const auto m = Matrix4().makeRotationFromEuler(v);

        const auto v2 = Euler().setFromRotationMatrix(m, v.getOrder());
        const auto m2 = Matrix4().makeRotationFromEuler(v2);
        CHECK(matrixEquals4(m, m2, 0.0001));
    }
}
