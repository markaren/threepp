
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/core/Object3D.hpp"
#include "threepp/math/Euler.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Matrix4.hpp"
#include "threepp/math/Vector3.hpp"

#include "../equals_util.hpp"

#include <cmath>

using namespace threepp;


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
    auto result = std::abs(a->quaternion.x - expectedQuat.x) <= eps &&
                  std::abs(a->quaternion.y - expectedQuat.y) <= eps &&
                  std::abs(a->quaternion.z - expectedQuat.z) <= eps;
    REQUIRE(result);
}

TEST_CASE("applyQuaternion") {

    auto a = Object3D::create();
    auto sqrt = 0.5f * static_cast<float>(std::sqrt(2));
    auto quat = Quaternion(0, sqrt, 0, sqrt);
    auto expected = Quaternion(sqrt / 2, sqrt / 2, 0, 0);

    a->quaternion.set(0.25, 0.25, 0.25, 0.25);
    a->applyQuaternion(quat);

    auto result = std::abs(a->quaternion.x - expected.x) <= eps &&
                  std::abs(a->quaternion.y - expected.y) <= eps &&
                  std::abs(a->quaternion.z - expected.z) <= eps;
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

    REQUIRE_THAT(obj->rotation.x * math::RAD2DEG, Catch::Matchers::WithinRel(45.f));
}

TEST_CASE("getWorldPosition") {

    float x = 1;
    float y = 2;
    float z = 3;

    auto a = Object3D::create();
    auto b = Object3D::create();
    auto expectedSingle = Vector3(x, y, z);
    auto expectedParent = Vector3(x, y, 0);
    auto expectedChild = Vector3(x, y, 7);
    auto position = Vector3();

    a->translateX(x);
    a->translateY(y);
    a->translateZ(z);

    a->getWorldPosition(position);
    REQUIRE(position == expectedSingle);

    // translate child and then parent
    b->translateZ(7);
    a->add(b);
    a->translateZ(-z);

    a->getWorldPosition(position);
    REQUIRE(position == expectedParent);
    b->getWorldPosition(position);
    REQUIRE(position == expectedChild);
}

TEST_CASE("getWorldScale") {

    float x = 1;
    float y = 2;
    float z = 3;

    auto a = Object3D::create();
    auto m = Matrix4().makeScale(x, y, z);
    auto expected = Vector3(x, y, z);

    a->applyMatrix4(m);

    Vector3 scale;
    a->getWorldScale(scale);
    REQUIRE(scale == expected);
}

TEST_CASE("updateMatrixWorld") {

    auto parent = Object3D::create();
    auto child = Object3D::create();

    // -- Standard usage test

    parent->position.set(1, 2, 3);
    child->position.set(4, 5, 6);
    parent->add(child);

    parent->updateMatrixWorld();

    REQUIRE(parent->matrix->elements == std::array<float, 16>{
                                                1, 0, 0, 0,
                                                0, 1, 0, 0,
                                                0, 0, 1, 0,
                                                1, 2, 3, 1});

    REQUIRE(parent->matrixWorld->elements == std::array<float, 16>{
                                                     1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0,
                                                     1, 2, 3, 1});

    REQUIRE(child->matrix->elements == std::array<float, 16>{
                                               1, 0, 0, 0,
                                               0, 1, 0, 0,
                                               0, 0, 1, 0,
                                               4, 5, 6, 1});

    REQUIRE(child->matrixWorld->elements == std::array<float, 16>{
                                                    1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 1, 0,
                                                    5, 7, 9, 1});

    REQUIRE((parent->matrixWorldNeedsUpdate || child->matrixWorldNeedsUpdate) == false);

    // -- No sync between local position/quaternion/scale/matrix and world matrix test

    parent->position.set(0, 0, 0);
    parent->updateMatrix();

    REQUIRE(parent->matrixWorld->elements == std::array<float, 16>{
                                                     1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0,
                                                     1, 2, 3, 1});

    // -- matrixAutoUpdate = false test

    // Resetting local and world matrices to the origin
    child->position.set(0, 0, 0);
    parent->updateMatrixWorld();

    parent->position.set(1, 2, 3);
    parent->matrixAutoUpdate = false;
    child->matrixAutoUpdate = false;
    parent->updateMatrixWorld();

    REQUIRE(parent->matrix->elements == std::array<float, 16>{
                                                1, 0, 0, 0,
                                                0, 1, 0, 0,
                                                0, 0, 1, 0,
                                                0, 0, 0, 1});

    REQUIRE(parent->matrixWorld->elements == std::array<float, 16>{
                                                     1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0,
                                                     0, 0, 0, 1});

    REQUIRE(child->matrixWorld->elements == std::array<float, 16>{
                                                    1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 1, 0,
                                                    0, 0, 0, 1});

    // -- matrixWorldAutoUpdate = false test

    parent->position.set(3, 2, 1);
    parent->updateMatrix();
    parent->matrixWorldNeedsUpdate = false;

    parent->updateMatrixWorld();

    REQUIRE(child->matrixWorld->elements == std::array<float, 16>{
                                                    1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 1, 0,
                                                    0, 0, 0, 1});

    // -- Propagation to children world matrices test

    child->position.set(0, 0, 0);
    parent->position.set(1, 2, 3);

    parent->matrixAutoUpdate = true;
    parent->updateMatrixWorld();

    REQUIRE(child->matrixWorld->elements == std::array<float, 16>{
                                                    1, 0, 0, 0,
                                                    0, 1, 0, 0,
                                                    0, 0, 1, 0,
                                                    1, 2, 3, 1});

    // -- force argument test

    // Resetting the local and world matrices to the origin
    child->position.set(0, 0, 0);
    child->matrixAutoUpdate = true;
    parent->updateMatrixWorld();

    parent->position.set(1, 2, 3);
    parent->updateMatrix();
    parent->matrixAutoUpdate = false;
    parent->matrixWorldNeedsUpdate = false;

    parent->updateMatrixWorld(true);

    REQUIRE(parent->matrixWorld->elements == std::array<float, 16>{
                                                     1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0,
                                                     1, 2, 3, 1});

    // -- Restriction test: No effect to parent matrices

    // Resetting the local and world matrices to the origin
    parent->position.set(0, 0, 0);
    child->position.set(0, 0, 0);
    parent->matrixAutoUpdate = true;
    child->matrixAutoUpdate = true;
    parent->updateMatrixWorld();

    parent->position.set(1, 2, 3);
    child->position.set(4, 5, 6);

    child->updateMatrixWorld();

    REQUIRE(parent->matrix->elements == std::array<float, 16>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});

    REQUIRE(parent->matrixWorld->elements == std::array<float, 16>{
                                                     1, 0, 0, 0,
                                                     0, 1, 0, 0,
                                                     0, 0, 1, 0,
                                                     0, 0, 0, 1});

    REQUIRE(child->matrixWorld->elements == std::array<float, 16>{1, 0, 0, 0,
                                                                  0, 1, 0, 0,
                                                                  0, 0, 1, 0,
                                                                  4, 5, 6, 1});
}

TEST_CASE("updateWorldMatrix") {

    auto object = Object3D::create();
    auto parent = Object3D::create();
    auto child = Object3D::create();

    auto m = Matrix4();
    auto v = Vector3();

    parent->add(object);
    object->add(child);

    parent->position.set(1, 2, 3);
    object->position.set(4, 5, 6);
    child->position.set(7, 8, 9);

    // Update the world matrix of an object

    object->updateWorldMatrix();

    REQUIRE(parent->matrix->elements == m.elements);

    REQUIRE(parent->matrixWorld->elements == m.elements);

    REQUIRE(object->matrix->elements == m.setPosition(object->position).elements);

    REQUIRE(object->matrixWorld->elements == m.setPosition(object->position).elements);

    REQUIRE(child->matrix->elements == m.identity().elements);

    REQUIRE(child->matrixWorld->elements == m.elements);

    // Update the world matrices of an object and its parents

    object->matrix->identity();
    object->matrixWorld->identity();

    object->updateWorldMatrix(true, false);

    REQUIRE(parent->matrix->elements == m.setPosition(parent->position).elements);

    REQUIRE(parent->matrixWorld->elements == m.setPosition(parent->position).elements);

    REQUIRE(object->matrix->elements == m.setPosition(object->position).elements);

    REQUIRE(object->matrixWorld->elements == m.setPosition(v.copy(parent->position).add(object->position)).elements);

    REQUIRE(child->matrix->elements == m.identity().elements);

    REQUIRE(child->matrixWorld->elements == m.identity().elements);

    // Update the world matrices of an object and its children

    parent->matrix->identity();
    parent->matrixWorld->identity();
    object->matrix->identity();
    object->matrixWorld->identity();

    object->updateWorldMatrix(false, true);

    REQUIRE(parent->matrix->elements == m.elements);

    REQUIRE(parent->matrixWorld->elements == m.elements);

    REQUIRE(object->matrix->elements == m.setPosition(object->position).elements);

    REQUIRE(object->matrixWorld->elements == m.setPosition(object->position).elements);

    REQUIRE(child->matrix->elements == m.setPosition(child->position).elements);

    REQUIRE(child->matrixWorld->elements == m.setPosition(v.copy(object->position).add(child->position)).elements);

    // Update the world matrices of an object and its parents and children

    object->matrix->identity();
    object->matrixWorld->identity();
    child->matrix->identity();
    child->matrixWorld->identity();

    object->updateWorldMatrix(true, true);

    REQUIRE(parent->matrix->elements == m.setPosition(parent->position).elements);

    REQUIRE(parent->matrixWorld->elements == m.setPosition(parent->position).elements);

    REQUIRE(object->matrix->elements == m.setPosition(object->position).elements);

    REQUIRE(object->matrixWorld->elements == m.setPosition(v.copy(parent->position).add(object->position)).elements);

    REQUIRE(child->matrix->elements == m.setPosition(child->position).elements);

    REQUIRE(child->matrixWorld->elements == m.setPosition(v.copy(parent->position).add(object->position).add(child->position)).elements);

    // object->matrixAutoUpdate = false test

    object->matrix->identity();
    object->matrixWorld->identity();

    object->matrixAutoUpdate = false;
    object->updateWorldMatrix(true, false);

    REQUIRE(object->matrix->elements == m.identity().elements);

    REQUIRE(object->matrixWorld->elements == m.setPosition(parent->position).elements);
}
