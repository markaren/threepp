
#include <catch2/catch_test_macros.hpp>

#include "threepp/core/BufferAttribute.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/math/Frustum.hpp"
#include "threepp/objects/Mesh.hpp"

using namespace threepp;

namespace {

    inline Vector3 unit3(1, 0, 0);

}

TEST_CASE("Instancing") {

    auto a = Frustum();

    const auto pDefault = Plane();
    for (unsigned i = 0; i < 6; i++) {

        CHECK(a.planes()[i] == (pDefault));
    }

    const auto p0 = Plane(unit3, -1);
    const auto p1 = Plane(unit3, 1);
    const auto p2 = Plane(unit3, 2);
    const auto p3 = Plane(unit3, 3);
    const auto p4 = Plane(unit3, 4);
    const auto p5 = Plane(unit3, 5);

    a = Frustum(p0, p1, p2, p3, p4, p5);
    CHECK(a.planes()[0].equals(p0));
    CHECK(a.planes()[1].equals(p1));
    CHECK(a.planes()[2].equals(p2));
    CHECK(a.planes()[3].equals(p3));
    CHECK(a.planes()[4].equals(p4));
    CHECK(a.planes()[5].equals(p5));
}

TEST_CASE("set") {

    auto a = Frustum();
    const auto p0 = Plane(unit3, -1);
    const auto p1 = Plane(unit3, 1);
    const auto p2 = Plane(unit3, 2);
    const auto p3 = Plane(unit3, 3);
    const auto p4 = Plane(unit3, 4);
    const auto p5 = Plane(unit3, 5);

    a.set(p0, p1, p2, p3, p4, p5);

    CHECK(a.planes()[0].equals(p0));
    CHECK(a.planes()[1].equals(p1));
    CHECK(a.planes()[2].equals(p2));
    CHECK(a.planes()[3].equals(p3));
    CHECK(a.planes()[4].equals(p4));
    CHECK(a.planes()[5].equals(p5));
}

TEST_CASE("setFromProjectionMatrix/makeOrthographic/containsPoint") {

    const auto m = Matrix4().makeOrthographic(-1, 1, -1, 1, 1, 100);
    const auto a = Frustum().setFromProjectionMatrix(m);

    CHECK(!a.containsPoint(Vector3(0, 0, 0)));
    CHECK(a.containsPoint(Vector3(0, 0, -50)));
    CHECK(a.containsPoint(Vector3(0, 0, -1.001f)));
    CHECK(a.containsPoint(Vector3(-1, -1, -1.001f)));
    CHECK(!a.containsPoint(Vector3(-1.1, -1.1, -1.001f)));
    CHECK(a.containsPoint(Vector3(1, 1, -1.001f)));
    CHECK(!a.containsPoint(Vector3(1.1, 1.1, -1.001f)));
    CHECK(a.containsPoint(Vector3(0, 0, -100)));
    CHECK(a.containsPoint(Vector3(-1, -1, -100)));
    CHECK(!a.containsPoint(Vector3(-1.1, -1.1, -100.1f)));
    CHECK(a.containsPoint(Vector3(1, 1, -100)));
    CHECK(!a.containsPoint(Vector3(1.1, 1.1, -100.1f)));
    CHECK(!a.containsPoint(Vector3(0, 0, -101)));
}

TEST_CASE("setFromProjectionMatrix/makePerspective/containsPoint") {

    const auto m = Matrix4().makePerspective(-1, 1, 1, -1, 1, 100);
    const auto a = Frustum().setFromProjectionMatrix(m);

    CHECK(!a.containsPoint(Vector3(0, 0, 0)));
    CHECK(a.containsPoint(Vector3(0, 0, -50)));
    CHECK(a.containsPoint(Vector3(0, 0, -1.001f)));
    CHECK(a.containsPoint(Vector3(-1, -1, -1.001f)));
    CHECK(!a.containsPoint(Vector3(-1.1f, -1.1f, -1.001f)));
    CHECK(a.containsPoint(Vector3(1, 1, -1.001f)));
    CHECK(!a.containsPoint(Vector3(1.1f, 1.1f, -1.001f)));
    CHECK(a.containsPoint(Vector3(0, 0, -99.999f)));
    CHECK(a.containsPoint(Vector3(-99.999f, -99.999f, -99.999f)));
    CHECK(!a.containsPoint(Vector3(-100.1f, -100.1f, -100.1f)));
    CHECK(a.containsPoint(Vector3(99.999f, 99.999f, -99.999f)));
    CHECK(!a.containsPoint(Vector3(100.1f, 100.1f, -100.1f)));
    CHECK(!a.containsPoint(Vector3(0, 0, -101)));
}

TEST_CASE("setFromProjectionMatrix/makePerspective/intersectsSphere") {

    const auto m = Matrix4().makePerspective(-1, 1, 1, -1, 1, 100);
    const auto a = Frustum().setFromProjectionMatrix(m);

    CHECK(!a.intersectsSphere(Sphere(Vector3(0, 0, 0), 0)));
    CHECK(!a.intersectsSphere(Sphere(Vector3(0, 0, 0), 0.9f)));
    CHECK(a.intersectsSphere(Sphere(Vector3(0, 0, 0), 1.1f)));
    CHECK(a.intersectsSphere(Sphere(Vector3(0, 0, -50), 0)));
    CHECK(a.intersectsSphere(Sphere(Vector3(0, 0, -1.001f), 0)));
    CHECK(a.intersectsSphere(Sphere(Vector3(-1, -1, -1.001f), 0)));
    CHECK(!a.intersectsSphere(Sphere(Vector3(-1.1f, -1.1f, -1.001f), 0)));
    CHECK(a.intersectsSphere(Sphere(Vector3(-1.1f, -1.1f, -1.001f), 0.5f)));
    CHECK(a.intersectsSphere(Sphere(Vector3(1, 1, -1.001f), 0)));
    CHECK(!a.intersectsSphere(Sphere(Vector3(1.1f, 1.1f, -1.001f), 0)));
    CHECK(a.intersectsSphere(Sphere(Vector3(1.1f, 1.1f, -1.001f), 0.5f)));
    CHECK(a.intersectsSphere(Sphere(Vector3(0, 0, -99.999f), 0)));
    CHECK(a.intersectsSphere(Sphere(Vector3(-99.999f, -99.999f, -99.999f), 0)));
    CHECK(!a.intersectsSphere(Sphere(Vector3(-100.1f, -100.1f, -100.1f), 0)));
    CHECK(a.intersectsSphere(Sphere(Vector3(-100.1f, -100.1f, -100.1f), 0.5f)));
    CHECK(a.intersectsSphere(Sphere(Vector3(99.999f, 99.999f, -99.999f), 0)));
    CHECK(!a.intersectsSphere(Sphere(Vector3(100.1f, 100.1f, -100.1f), 0)));
    CHECK(a.intersectsSphere(Sphere(Vector3(100.1f, 100.1f, -100.1f), 0.2f)));
    CHECK(!a.intersectsSphere(Sphere(Vector3(0, 0, -101), 0)));
    CHECK(a.intersectsSphere(Sphere(Vector3(0, 0, -101), 1.1f)));
}

TEST_CASE("intersectsObject") {

    const auto m = Matrix4().makePerspective(-1, 1, 1, -1, 1, 100);
    const auto a = Frustum().setFromProjectionMatrix(m);
    const auto object = Mesh::create(BoxGeometry::create(1, 1, 1));

    bool intersects = a.intersectsObject(*object);
    CHECK(!intersects);

    object->position.set(-1, -1, -1);
    object->updateMatrixWorld();

    intersects = a.intersectsObject(*object);
    CHECK(intersects);

    object->position.set(1, 1, 1);
    object->updateMatrixWorld();

    intersects = a.intersectsObject(*object);
    CHECK(!intersects);
}
