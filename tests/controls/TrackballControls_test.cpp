
#include "threepp/controls/TrackballControls.hpp"

#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/input/PeripheralsEventSource.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace threepp;

namespace {

    constexpr int WIDTH = 800;
    constexpr int HEIGHT = 600;

    // Drives the controls without a window. onMouse*Event are protected on
    // PeripheralsEventSource, so a subclass is the way to synthesize input.
    class FakeEventSource: public PeripheralsEventSource {

    public:
        [[nodiscard]] WindowSize size() const override {

            return {WIDTH, HEIGHT};
        }

        void press(int button, const Vector2& pos) {

            onMousePressedEvent(button, pos, MouseAction::PRESS);
        }

        void release(int button, const Vector2& pos) {

            onMousePressedEvent(button, pos, MouseAction::RELEASE);
        }

        void move(const Vector2& pos) {

            onMouseMoveEvent(pos);
        }

        void wheel(float amount) {

            onMouseWheelEvent({0.f, amount});
        }
    };

    // One mouse gesture, stepped like a real frame loop so every intermediate
    // delta reaches the controls.
    void drag(FakeEventSource& source, TrackballControls& controls,
              int button, const Vector2& from, const Vector2& to, int steps = 20) {

        source.press(button, from);
        controls.update();

        for (int i = 1; i <= steps; ++i) {

            const auto t = static_cast<float>(i) / static_cast<float>(steps);
            source.move({from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t});
            controls.update();
        }

        source.release(button, to);
        controls.update();
    }

    // World-space right vector of the camera. Its y component is the horizon tilt:
    // zero for anything that keeps a fixed up-vector, non-zero once the view rolls.
    Vector3 cameraRight(Camera& camera) {

        camera.updateMatrixWorld();

        Vector3 right;
        right.setFromMatrixColumn(*camera.matrixWorld, 0);

        return right;
    }

}// namespace

TEST_CASE("rotation preserves the distance to target") {

    FakeEventSource source;
    PerspectiveCamera camera(50, 4 / 3.f);
    camera.position.set(0, 0, 10);

    TrackballControls controls(camera, source);
    controls.staticMoving = true;

    drag(source, controls, 0, {200, 400}, {600, 150});

    REQUIRE(std::abs(controls.getDistance() - 10.f) < 1e-3f);
    REQUIRE(std::isfinite(camera.position.x));
}

TEST_CASE("a closed drag loop rolls the horizon") {

    FakeEventSource source;
    PerspectiveCamera camera(50, 4 / 3.f);
    camera.position.set(0, 0, 10);

    TrackballControls controls(camera, source);
    controls.staticMoving = true;

    // the horizon starts level
    REQUIRE(std::abs(cameraRight(camera).y) < 1e-5f);

    // walk a square on the trackball and come back: the rotations do not commute,
    // so the view returns rolled. OrbitControls cannot reach this state at all.
    drag(source, controls, 0, {200, 300}, {600, 300});
    drag(source, controls, 0, {600, 300}, {600, 100});
    drag(source, controls, 0, {600, 100}, {200, 100});
    drag(source, controls, 0, {200, 100}, {200, 300});

    REQUIRE(std::abs(cameraRight(camera).y) > 0.05f);
    REQUIRE(std::abs(controls.getDistance() - 10.f) < 1e-3f);
}

TEST_CASE("dragging upwards tumbles through the pole") {

    FakeEventSource source;
    PerspectiveCamera camera(50, 4 / 3.f);
    camera.position.set(0, 0, 10);

    TrackballControls controls(camera, source);
    controls.staticMoving = true;

    // ~1.3 rad per full-height drag, so three of them pass straight over the top
    for (int i = 0; i < 3; ++i) {

        drag(source, controls, 0, {400, 550}, {400, 50});
    }

    // past the pole and coming down the far side - no clamp, no gimbal lock
    REQUIRE(camera.position.z < 0.f);
    REQUIRE(std::abs(controls.getDistance() - 10.f) < 1e-3f);
}

TEST_CASE("wheel scrolls up to zoom in") {

    FakeEventSource source;
    PerspectiveCamera camera(50, 4 / 3.f);
    camera.position.set(0, 0, 10);

    TrackballControls controls(camera, source);

    source.wheel(1);// GLFW reports positive y for scroll-up

    for (int i = 0; i < 60; ++i) controls.update();

    REQUIRE(controls.getDistance() < 10.f);

    const auto zoomedIn = controls.getDistance();

    source.wheel(-1);

    for (int i = 0; i < 60; ++i) controls.update();

    REQUIRE(controls.getDistance() > zoomedIn);
}

TEST_CASE("pan drags target and camera together") {

    FakeEventSource source;
    PerspectiveCamera camera(50, 4 / 3.f);
    camera.position.set(0, 0, 10);

    TrackballControls controls(camera, source);
    controls.staticMoving = true;

    drag(source, controls, 1, {400, 300}, {600, 300});

    // the scene follows the cursor, so the camera goes the other way
    REQUIRE(camera.position.x < -0.1f);

    // and the orbit point travels with it
    REQUIRE(std::abs(controls.target.x - camera.position.x) < 1e-3f);
    REQUIRE(std::abs(controls.getDistance() - 10.f) < 1e-3f);
}

TEST_CASE("rotation keeps spinning after release unless staticMoving") {

    FakeEventSource source;
    PerspectiveCamera camera(50, 4 / 3.f);
    camera.position.set(0, 0, 10);

    TrackballControls controls(camera, source);

    drag(source, controls, 0, {300, 300}, {500, 300});

    const auto afterRelease = camera.position.clone();

    for (int i = 0; i < 10; ++i) controls.update();

    REQUIRE(afterRelease.distanceTo(camera.position) > 1e-3f);

    controls.staticMoving = true;

    for (int i = 0; i < 10; ++i) controls.update();

    const auto settled = camera.position.clone();

    for (int i = 0; i < 10; ++i) controls.update();

    REQUIRE(settled.distanceTo(camera.position) < 1e-6f);
}

TEST_CASE("reset restores the initial view") {

    FakeEventSource source;
    PerspectiveCamera camera(50, 4 / 3.f);
    camera.position.set(0, 0, 10);

    TrackballControls controls(camera, source);
    controls.staticMoving = true;

    drag(source, controls, 0, {200, 300}, {600, 120});
    drag(source, controls, 1, {400, 300}, {500, 200});
    source.wheel(2);

    for (int i = 0; i < 30; ++i) controls.update();

    controls.reset();
    controls.update();

    REQUIRE(camera.position.distanceTo({0, 0, 10}) < 1e-4f);
    REQUIRE(controls.target.length() < 1e-4f);
    REQUIRE(std::abs(cameraRight(camera).y) < 1e-5f);
}
