
#include "threepp/cameras/StereoCamera.hpp"

#include "../equals_util.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace threepp;

namespace {

    // World point to one eye's normalized device coordinates.
    Vector3 project(const PerspectiveCamera& eye, Vector3 p) {
        return p.applyMatrix4(eye.matrixWorldInverse).applyMatrix4(eye.projectionMatrix);
    }

}// namespace

TEST_CASE("the eyes sit eyeSep apart on the camera's x axis") {

    PerspectiveCamera camera(40, 1.6f, 0.1f, 100);
    camera.position.set(1, 2, 3);
    camera.rotation.y = 0.7f;
    camera.updateMatrixWorld();

    StereoCamera stereo;
    stereo.eyeSep = 0.07f;
    stereo.update(camera);

    Vector3 left, right, x, unused;
    left.setFromMatrixPosition(*stereo.cameraL.matrixWorld);
    right.setFromMatrixPosition(*stereo.cameraR.matrixWorld);
    camera.matrixWorld->extractBasis(x, unused, unused);

    Vector3 between = Vector3().subVectors(right, left);
    CHECK(between.length() == Catch::Approx(0.07f).margin(1e-5));
    CHECK(between.normalize().dot(x) == Catch::Approx(1.f).margin(1e-5));

    const Vector3 middle = Vector3().addVectors(left, right).multiplyScalar(0.5f);
    CHECK(middle.distanceTo(camera.position) < 1e-5f);
}

TEST_CASE("a point at the focus distance has the same place in both pictures") {

    PerspectiveCamera camera(40, 1.6f, 0.1f, 100);
    camera.focus = 4;
    camera.position.set(-2, 1, 0.5f);
    camera.rotation.set(0.2f, -1.1f, 0.05f);
    camera.updateMatrixWorld();

    StereoCamera stereo;
    stereo.update(camera);

    // Off the axis, so that the test does not pass on symmetry alone.
    auto worldPoint = [&](float depth) {
        return Vector3(0.3f, -0.2f, -depth).applyMatrix4(*camera.matrixWorld);
    };

    const Vector3 atFocus = worldPoint(4);
    CHECK(project(stereo.cameraL, atFocus).x == Catch::Approx(project(stereo.cameraR, atFocus).x).margin(1e-5));
    CHECK(project(stereo.cameraL, atFocus).y == Catch::Approx(project(stereo.cameraR, atFocus).y).margin(1e-5));

    // Nearer than the focus distance: further right in the left eye's picture.
    const Vector3 near = worldPoint(1);
    CHECK(project(stereo.cameraL, near).x > project(stereo.cameraR, near).x + 1e-3f);

    // Further away: the other way round.
    const Vector3 far = worldPoint(20);
    CHECK(project(stereo.cameraL, far).x < project(stereo.cameraR, far).x - 1e-3f);
}

TEST_CASE("zero eye separation gives the camera's own projection") {

    PerspectiveCamera camera(55, 1.25f, 0.2f, 50);
    camera.updateMatrixWorld();

    StereoCamera stereo;
    stereo.eyeSep = 0;
    stereo.update(camera);

    CHECK(matrixEquals4(stereo.cameraL.projectionMatrix, camera.projectionMatrix, 1e-6));
    CHECK(matrixEquals4(stereo.cameraR.projectionMatrix, camera.projectionMatrix, 1e-6));
}

TEST_CASE("an eye keeps its place through the update a renderer gives a parentless camera") {

    PerspectiveCamera camera(40, 1.6f, 0.1f, 100);
    camera.position.set(5, 0, -3);
    camera.updateMatrixWorld();

    StereoCamera stereo;
    stereo.update(camera);

    const Matrix4 before = *stereo.cameraL.matrixWorld;
    stereo.cameraL.updateMatrixWorld();

    CHECK(matrixEquals4(*stereo.cameraL.matrixWorld, before, 1e-6));

    Vector3 eye;
    eye.setFromMatrixPosition(*stereo.cameraL.matrixWorld);
    CHECK(eye.x == Catch::Approx(5 - stereo.eyeSep / 2).margin(1e-5));

    // And follows the camera on the next update.
    camera.position.x = 6;
    camera.updateMatrixWorld();
    stereo.update(camera);
    stereo.cameraL.updateMatrixWorld();
    eye.setFromMatrixPosition(*stereo.cameraL.matrixWorld);
    CHECK(eye.x == Catch::Approx(6 - stereo.eyeSep / 2).margin(1e-5));
}
