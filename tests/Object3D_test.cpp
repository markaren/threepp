
#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Euler.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

#include "threepp/math/MathUtils.hpp"

using namespace threepp;

namespace {

    constexpr float eps = 0.0001;

    bool matrixEquals4(const Matrix4 &a, const Matrix4 &b, float tolerance = eps) {

        for (unsigned i = 0, il = a.elements.size(); i < il; i++) {

            auto delta = a.elements[i] - b.elements[i];
            if (delta > tolerance) {

                return false;
            }
        }

        return true;
    }

    bool eulerEquals(const Euler &a, const Euler &b, float tolerance = eps) {

        auto diff = std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z);
        return (diff < tolerance);
    }


}// namespace

TEST_CASE("applyMatrix4") {

    float x = 1;
    float y = 2;
    float z = 3;

    auto a = Object3D::create();
    auto m = Matrix4();
    auto expectedPos = Vector3(x, y, z);
    auto expectedQuat = Quaternion(0.5f * static_cast<float>(std::sqrt(2)), 0, 0, 0.5f * static_cast<float>(std::sqrt(2)));

    m.makeRotationX(math::PI / 2);
    m.setPosition(Vector3(x, y, z));

    a->applyMatrix4(m);

    REQUIRE(a->position == expectedPos);
    auto result = std::abs(a->quaternion.x() - expectedQuat.x()) <= eps &&
                  std::abs(a->quaternion.y() - expectedQuat.y()) <= eps &&
                  std::abs(a->quaternion.z() - expectedQuat.z()) <= eps;
    REQUIRE(result);
}

TEST_CASE("applyQuaternion") {

    auto a = Object3D::create();
    auto sqrt = 0.5f * static_cast<float>(std::sqrt(2));
    auto quat = Quaternion(0, sqrt, 0, sqrt);
    auto expected = Quaternion(sqrt / 2, sqrt / 2, 0, 0);

    a->quaternion.set(0.25, 0.25, 0.25, 0.25);
    a->applyQuaternion(quat);

    auto result = std::abs(a->quaternion.x() - expected.x()) <= eps &&
                  std::abs(a->quaternion.y() - expected.y()) <= eps &&
                  std::abs(a->quaternion.z() - expected.z()) <= eps;
    REQUIRE(result);
}

TEST_CASE("localToWorld") {

    auto v = Vector3();
    const auto expectedPosition = Vector3(5, -1, -4);

    const auto parent = Object3D::create();
    const auto child = Object3D::create();

    parent->position.set(1, 0, 0);
    parent->rotation.set(0, math::PI / 2, 0);
    parent->scale.set(2, 1, 1);

    child->position.set(0, 1, 0);
    child->rotation.set(math::PI / 2, 0, 0);
    child->scale.set(1, 2, 1);

    parent->add(child);
    parent->updateMatrixWorld();

    child->localToWorld(v.set(2, 2, 2));

    auto result = std::abs(v.x - expectedPosition.x) <= eps &&
                  std::abs(v.y - expectedPosition.y) <= eps &&
                  std::abs(v.z - expectedPosition.z) <= eps;
    REQUIRE(result);
}

TEST_CASE("worldToLocal") {

    auto v = Vector3();
    const auto expectedPosition = Vector3(-1, 0.5, -1);

    const auto parent = Object3D::create();
    const auto child = Object3D::create();

    parent->position.set(1, 0, 0);
    parent->rotation.set(0, math::PI / 2, 0);
    parent->scale.set(2, 1, 1);

    child->position.set(0, 1, 0);
    child->rotation.set(math::PI / 2, 0, 0);
    child->scale.set(1, 2, 1);

    parent->add(child);
    parent->updateMatrixWorld();

    child->worldToLocal(v.set(2, 2, 2));

    auto result = std::abs(v.x - expectedPosition.x) <= eps &&
                  std::abs(v.y - expectedPosition.y) <= eps &&
                  std::abs(v.z - expectedPosition.z) <= eps;
    REQUIRE(result);
}

TEST_CASE("lookAt") {

    auto obj = Object3D::create();
    obj->lookAt(Vector3(0, -1, 1));

    REQUIRE(obj->rotation.x * math::RAD2DEG == Approx(45));
}
