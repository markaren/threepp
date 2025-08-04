
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Raycaster.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"

using namespace threepp;

namespace {

    auto getSphere() {

        return Mesh::create(SphereGeometry::create(1, 100, 100));
    }

    auto getObjectsToCheck() {
        std::vector<std::shared_ptr<Object3D>> objects;

        auto sphere1 = getSphere();
        sphere1->position.set(0, 0, -10);
        sphere1->name = "1";
        objects.emplace_back(sphere1);

        auto sphere11 = getSphere();
        sphere11->position.set(0, 0, 1);
        sphere11->name = "11";
        sphere1->add(sphere11);

        auto sphere12 = getSphere();
        sphere12->position.set(0, 0, -1);
        sphere12->name = "12";
        sphere1->add(sphere12);

        auto sphere2 = getSphere();
        sphere2->position.set(-5, 0, -5);
        sphere2->name = "2";
        objects.emplace_back(sphere2);

        for (auto& o : objects) {

            o->updateMatrixWorld();
        }

        return objects;
    }

    Raycaster getRaycaster() {

        return Raycaster(
                Vector3(0, 0, 0),
                Vector3(0, 0, -1),
                1,
                100);
    }

    bool isRayDirectionAgainstReferenceVector(const Vector3& rayDirection, const Vector3& refVector) {

        return refVector.x - rayDirection.x <= std::numeric_limits<float>::epsilon() && refVector.y - rayDirection.y <= std::numeric_limits<float>::epsilon() && refVector.z - rayDirection.z <= std::numeric_limits<float>::epsilon();
    }
}// namespace


TEST_CASE("set") {
    Vector3 origin(0, 0, 0);
    Vector3 direction(0, 0, -1);
    Raycaster a(origin.clone(), direction.clone());

    CHECK(a.ray.origin == origin);
    CHECK(a.ray.direction == direction);

    origin.set(1, 1, 1);
    direction.set(-1, 0, 0);
    a.set(origin, direction);

    CHECK(a.ray.origin == origin);
    CHECK(a.ray.direction == direction);
}

TEST_CASE("setFromCamera (Perspective)") {
    Raycaster raycaster;
    const auto& rayDirection = raycaster.ray.direction;
    PerspectiveCamera camera(90, 1, 1, 1000);

    raycaster.setFromCamera({0,
                             0},
                            camera);
    CHECK_THAT(rayDirection.x, Catch::Matchers::WithinRel(0.));
    CHECK_THAT(rayDirection.y, Catch::Matchers::WithinRel(0.));
    CHECK_THAT(rayDirection.z, Catch::Matchers::WithinRel(-1.));

    const float step = 0.1;

    for (float x = -1; x <= 1; x += step) {

        for (float y = -1; y <= 1; y += step) {

            raycaster.setFromCamera({x,
                                     y},
                                    camera);

            auto refVector = Vector3(x, y, -1).normalize();

            CHECK(isRayDirectionAgainstReferenceVector(rayDirection, refVector));
        }
    }
}

TEST_CASE("intersectObject") {
    auto raycaster = getRaycaster();
    auto objectsToCheck = getObjectsToCheck();

    CHECK(raycaster.intersectObject(*objectsToCheck[0], false).size() == 1);

    CHECK(raycaster.intersectObject(*objectsToCheck[0], true).size() == 3);

    auto intersections = raycaster.intersectObject(*objectsToCheck[0]);
    for (auto i = 0; i < intersections.size() - 1; i++) {

        CHECK(intersections[i].distance <= intersections[i + 1].distance);
    }
}

TEST_CASE("intersectObjects") {
    auto raycaster = getRaycaster();
    auto objectsToCheck = getObjectsToCheck();

    std::vector<Object3D*> refVector;
    for (auto& o : objectsToCheck) {
        refVector.emplace_back(o.get());
    }

    CHECK(raycaster.intersectObjects(refVector, false).size() == 1);

    CHECK(raycaster.intersectObjects(refVector, true).size() == 3);

    auto intersections = raycaster.intersectObjects(refVector);
    for (auto i = 0; i < intersections.size() - 1; i++) {

        CHECK(intersections[i].distance <= intersections[i + 1].distance);
    }
}

TEST_CASE("setFromCamera (Orthographic)") {
    Raycaster raycaster;
    const auto& rayOrigin = raycaster.ray.origin;
    const auto& rayDirection = raycaster.ray.direction;
    OrthographicCamera camera(-1, 1, 1, -1, 0, 1000);
    Vector3 expectedOrigin(0, 0, 0);
    Vector3 expectedDirection(0, 0, -1);

    raycaster.setFromCamera({0,
                             0},
                            camera);
    CHECK(rayOrigin == expectedOrigin);
    CHECK(rayDirection == expectedDirection);
}

TEST_CASE("Line intersection threshold") {
    auto raycaster = getRaycaster();
    std::vector points{Vector3(-2, -10, -5), Vector3(-2, 10, -5)};
    auto geometry = BufferGeometry::create();
    geometry->setFromPoints(points);
    auto line = Line::create(geometry, nullptr);

    raycaster.params.lineThreshold = 1.999f;
    CHECK(raycaster.intersectObject(*line).empty());

    raycaster.params.lineThreshold = 2.001f;
    CHECK(raycaster.intersectObject(*line).size() == 1);
}

TEST_CASE("Points intersection threshold") {
    auto raycaster = getRaycaster();
    std::vector coordinates{Vector3(-2, 0, -5)};
    auto geometry = BufferGeometry::create();
    geometry->setFromPoints(coordinates);
    auto points = Points::create(geometry, nullptr);

    raycaster.params.pointsThreshold = 1.999f;
    CHECK(raycaster.intersectObject(*points).empty());

    raycaster.params.pointsThreshold = 2.001f;
    CHECK(raycaster.intersectObject(*points).size() == 1);
}
