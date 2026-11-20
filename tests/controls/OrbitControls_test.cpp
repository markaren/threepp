
#include "threepp/controls/OrbitControls.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/cameras/PerspectiveCamera.hpp"
#include "threepp/input/PeripheralsEventSource.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Quaternion.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

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

    // One frame: the controls' update followed by the world-matrix refresh a
    // render performs. pan() reads camera.matrix, so without the refresh a
    // headless pan would use a stale orientation.
    bool step(OrbitControls& controls, Camera& camera) {

        const bool changed = controls.update();
        camera.updateMatrixWorld();

        return changed;
    }

    void settle(OrbitControls& controls, Camera& camera, int frames) {

        for (int i = 0; i < frames; ++i) step(controls, camera);
    }

    // One mouse gesture, stepped like a real frame loop. The controls update on
    // every move themselves; the frame update after it is what the app's loop adds.
    void drag(FakeEventSource& source, OrbitControls& controls, Camera& camera,
              int button, const Vector2& from, const Vector2& to, int steps = 20) {

        source.press(button, from);
        step(controls, camera);

        for (int i = 1; i <= steps; ++i) {

            const auto t = static_cast<float>(i) / static_cast<float>(steps);
            source.move({from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t});
            step(controls, camera);
        }

        source.release(button, to);
        step(controls, camera);
    }

    Vector3 cameraAxis(Camera& camera, unsigned int column) {

        camera.updateMatrixWorld();

        Vector3 axis;
        axis.setFromMatrixColumn(*camera.matrixWorld, column);

        return axis.normalize();
    }

    Vector3 cameraForward(Camera& camera) {

        return cameraAxis(camera, 2).negate();
    }

    // The on-screen heading about world +Y, read from the camera's right vector:
    // OrbitControls keeps that vector horizontal, and at the poles it is the only
    // thing that tells one top-down picture from another.
    float heading(Camera& camera) {

        const auto right = cameraAxis(camera, 0);

        return std::atan2(-right.z, right.x);
    }

    float angleBetween(float a, float b) {

        return std::abs(std::remainder(a - b, math::TWO_PI));
    }

    // Spherical placement about +Y, the controls' own convention: phi from +Y,
    // theta from +Z towards +X.
    Vector3 onSphere(const Vector3& target, float radius, float phi, float theta) {

        return {target.x + radius * std::sin(phi) * std::sin(theta),
                target.y + radius * std::cos(phi),
                target.z + radius * std::sin(phi) * std::cos(theta)};
    }

    std::unique_ptr<Camera> makeCamera(bool orthographic) {

        if (orthographic) return std::make_unique<OrthographicCamera>(-4.f / 3.f, 4.f / 3.f, 1.f, -1.f, 0.001f, 10000.f);

        return std::make_unique<PerspectiveCamera>(50.f, 4.f / 3.f, 0.001f, 10000.f);
    }

    // Rotation angle between two orientations, in double and from the chord
    // rather than acos(dot): a float quaternion is unit only to ~1e-7, which
    // acos near 1 turns into a phantom 4e-4 rad.
    double angleBetween(const Quaternion& a, const Quaternion& b) {

        std::array<double, 4> p{}, q{};
        double np = 0, nq = 0, dot = 0;
        for (unsigned int i = 0; i < 4; ++i) {
            p[i] = a[i];
            q[i] = b[i];
            np += p[i] * p[i];
            nq += q[i] * q[i];
            dot += p[i] * q[i];
        }
        const double sign = dot < 0 ? -1.0 : 1.0;

        double difference = 0, sum = 0;
        for (unsigned int i = 0; i < 4; ++i) {
            const auto u = p[i] / std::sqrt(np);
            const auto v = sign * q[i] / std::sqrt(nq);
            difference += (u - v) * (u - v);
            sum += (u + v) * (u + v);
        }

        return 2.0 * std::atan2(std::sqrt(difference), std::sqrt(sum));
    }

    // A pixel drag of this many rows turns the polar angle by one radian.
    constexpr float kPixelsPerRadian = HEIGHT / math::TWO_PI;

    const std::array<Vector3, 4> kTargets{Vector3(0, 0, 0), Vector3(1, 0, 0.5f), Vector3(100, 0, 50), Vector3(1000, 0, 500)};
    const std::array<float, 3> kRadii{0.05f, 1.f, 10.f};

    struct Pose {
        std::array<float, 3> position;
        std::array<float, 4> quaternion;
        float zoom;
    };

    Pose poseOf(const Camera& camera) {

        return {{camera.position.x, camera.position.y, camera.position.z},
                {camera.quaternion[0], camera.quaternion[1], camera.quaternion[2], camera.quaternion[3]},
                camera.zoom};
    }

    // Gestures away from the poles, recorded pose by pose. The reference values
    // in the test below were produced by this function on the implementation
    // that re-derived its spherical from the camera on every update (1da1aba10).
    std::vector<Pose> scenario(int id) {

        FakeEventSource source;
        auto camera = makeCamera(id == 2);
        camera->position.set(3, 4, 5);

        OrbitControls controls(*camera, source);
        controls.target.set(1, 0.5f, -2);
        controls.enableDamping = id == 1;

        const int frames = controls.enableDamping ? 100 : 1;
        std::vector<Pose> poses;

        step(controls, *camera);
        poses.push_back(poseOf(*camera));

        drag(source, controls, *camera, 0, {300, 300}, {420, 250});
        settle(controls, *camera, frames);
        poses.push_back(poseOf(*camera));

        drag(source, controls, *camera, 1, {400, 300}, {350, 340});
        settle(controls, *camera, frames);
        poses.push_back(poseOf(*camera));

        for (int i = 0; i < 3; ++i) {
            source.wheel(1);
            step(controls, *camera);
        }
        settle(controls, *camera, frames);
        poses.push_back(poseOf(*camera));

        drag(source, controls, *camera, 2, {400, 300}, {400, 340});
        settle(controls, *camera, frames);
        poses.push_back(poseOf(*camera));

        drag(source, controls, *camera, 0, {400, 300}, {250, 420});
        settle(controls, *camera, frames);
        poses.push_back(poseOf(*camera));

        if (id == 0) {
            controls.autoRotate = true;
            settle(controls, *camera, 120);
            poses.push_back(poseOf(*camera));
        }

        return poses;
    }

}// namespace

TEST_CASE("rotate, pan and dolly away from the poles match the stateless implementation") {

    // scenario(id) on 1da1aba10: 0 perspective, 1 perspective with damping,
    // 2 orthographic
    const std::array<std::vector<Pose>, 3> reference{{
            {
                    Pose{{3.f, 3.99999976f, 5.00000048f}, {-0.220051527f, 0.135233864f, 0.0308193024f, 0.965577185f}, 1.f},
                    Pose{{-5.68199062f, -0.10896188f, 2.49779844f}, {0.033296898f, -0.469557911f, 0.0177246351f, 0.882095575f}, 1.f},
                    Pose{{-5.36284351f, 0.391837537f, 3.03973341f}, {0.033296898f, -0.469557911f, 0.0177246351f, 0.882095575f}, 1.f},
                    Pose{{-4.40982342f, 0.478690803f, 2.39823484f}, {0.033296898f, -0.469557852f, 0.0177246351f, 0.882095575f}, 1.f},
                    Pose{{-14.6618919f, -0.455629587f, 9.29913425f}, {0.033296898f, -0.469557822f, 0.0177246351f, 0.882095575f}, 1.f},
                    Pose{{5.41676998f, 18.8721199f, 4.62941456f}, {-0.53259629f, 0.242469206f, 0.162553608f, 0.794434488f}, 1.f},
                    Pose{{2.58650541f, 18.8721199f, 5.76977825f}, {-0.554754615f, 0.0719981343f, 0.0482682176f, 0.827486396f}, 1.f},
            },
            {
                    Pose{{3.f, 3.99999976f, 5.00000048f}, {-0.220051527f, 0.135233864f, 0.0308193024f, 0.965577185f}, 1.f},
                    Pose{{-5.66973543f, -0.0994029045f, 2.51723528f}, {0.032798335f, -0.468310356f, 0.0173992775f, 0.882783592f}, 1.f},
                    Pose{{-5.36268902f, 0.390716612f, 3.0379982f}, {0.0332971327f, -0.469556481f, 0.0177246928f, 0.882096291f}, 1.f},
                    Pose{{-4.40895939f, 0.47870034f, 2.39771605f}, {0.0332971253f, -0.469557106f, 0.0177247152f, 0.882095933f}, 1.f},
                    Pose{{-14.6610184f, -0.455622077f, 9.2986412f}, {0.0332971253f, -0.469557106f, 0.0177247152f, 0.882095933f}, 1.f},
                    Pose{{5.42424965f, 18.8511562f, 4.68582582f}, {-0.531750262f, 0.241284937f, 0.161261201f, 0.795624316f}, 1.f},
            },
            {
                    Pose{{3.f, 3.99999976f, 5.00000048f}, {-0.220051527f, 0.135233864f, 0.0308193024f, 0.965577185f}, 1.f},
                    Pose{{-5.68199062f, -0.10896188f, 2.49779844f}, {0.033296898f, -0.469557911f, 0.0177246351f, 0.882095575f}, 1.f},
                    Pose{{-5.59726143f, 0.0239922404f, 2.64167213f}, {0.033296898f, -0.469557911f, 0.0177246351f, 0.882095575f}, 1.f},
                    Pose{{-5.59726143f, 0.0239922404f, 2.64167213f}, {0.033296898f, -0.469557911f, 0.0177246351f, 0.882095575f}, 1.16635084f},
                    Pose{{-5.59726143f, 0.0239922404f, 2.64167213f}, {0.033296898f, -0.469557911f, 0.0177246351f, 0.882095575f}, 0.418120205f},
                    Pose{{2.79802585f, 8.10530853f, 0.689170361f}, {-0.53259635f, 0.242469102f, 0.162553534f, 0.794434488f}, 0.418120205f},
            },
    }};

    for (int id = 0; id < 3; ++id) {

        const auto poses = scenario(id);
        REQUIRE(poses.size() == reference[id].size());

        for (std::size_t i = 0; i < poses.size(); ++i) {

            CAPTURE(id, i);

            const auto& got = poses[i];
            const auto& want = reference[id][i];

            const Vector3 p(got.position[0], got.position[1], got.position[2]);
            const Vector3 q(want.position[0], want.position[1], want.position[2]);
            CHECK(p.distanceTo(q) < 2e-4f);

            const Quaternion a(got.quaternion[0], got.quaternion[1], got.quaternion[2], got.quaternion[3]);
            const Quaternion b(want.quaternion[0], want.quaternion[1], want.quaternion[2], want.quaternion[3]);
            CHECK(angleBetween(a, b) < 1e-5);

            CHECK(std::abs(got.zoom - want.zoom) < 1e-6f);
        }
    }
}

TEST_CASE("the heading survives the poles") {

    for (const bool orthographic : {false, true}) {
        for (const bool damping : {false, true}) {
            for (const bool top : {true, false}) {
                for (const auto& target : kTargets) {
                    for (const auto radius : kRadii) {

                        CAPTURE(orthographic, damping, top, target, radius);

                        FakeEventSource source;
                        auto camera = makeCamera(orthographic);
                        camera->position = onSphere(target, radius, 1.f, 0.7f);

                        OrbitControls controls(*camera, source);
                        controls.target = target;
                        controls.enableDamping = damping;
                        step(controls, *camera);

                        const float set = heading(*camera);

                        // half a turn of polar angle: past the pole either way from phi = 1
                        if (top) {
                            drag(source, controls, *camera, 0, {400, 150}, {400, 450});
                        } else {
                            drag(source, controls, *camera, 0, {400, 450}, {400, 150});
                        }
                        settle(controls, *camera, damping ? 400 : 1);

                        CHECK(cameraForward(*camera).y * (top ? -1.f : 1.f) > 0.99999f);

                        float worst = 0;
                        for (int i = 0; i < 200; ++i) {
                            step(controls, *camera);
                            worst = std::max(worst, angleBetween(heading(*camera), set));
                        }
                        CHECK(worst < 1e-4f);
                    }
                }
            }
        }
    }
}

TEST_CASE("a rotate drag at the pole turns the picture by the dragged amount") {

    for (const bool orthographic : {false, true}) {
        for (const bool damping : {false, true}) {
            for (const auto& target : kTargets) {

                CAPTURE(orthographic, damping, target);

                FakeEventSource source;
                auto camera = makeCamera(orthographic);
                camera->position = onSphere(target, 1.f, 1.f, 0.3f);

                OrbitControls controls(*camera, source);
                controls.target = target;
                controls.enableDamping = damping;
                step(controls, *camera);

                drag(source, controls, *camera, 0, {400, 150}, {400, 450});
                settle(controls, *camera, damping ? 400 : 1);

                const float before = heading(*camera);

                // 100 px to the right is rotateLeft(2 pi 100 / height), theta -= that
                drag(source, controls, *camera, 0, {300, 300}, {400, 300});
                settle(controls, *camera, damping ? 400 : 1);

                const float turned = std::remainder(heading(*camera) - before, math::TWO_PI);

                CHECK(std::abs(turned + 100.f / kPixelsPerRadian) < 1e-3f);
                CHECK(cameraForward(*camera).y < -0.99999f);
            }
        }
    }
}

TEST_CASE("up to the pole and back returns to the starting view") {

    for (const bool damping : {false, true}) {
        for (const auto& target : kTargets) {

            CAPTURE(damping, target);

            FakeEventSource source;
            auto camera = makeCamera(false);
            camera->position = onSphere(target, 1.f, 1.f, 0.7f);

            OrbitControls controls(*camera, source);
            controls.target = target;
            controls.enableDamping = damping;
            step(controls, *camera);

            const auto startPosition = camera->position;
            const auto startQuaternion = camera->quaternion;

            drag(source, controls, *camera, 0, {400, 150}, {400, 450});
            settle(controls, *camera, damping ? 400 : 1);

            // the clamp holds phi at 1e-6, so coming back down is 1 - 1e-6 rad
            const float back = (1.f - 1e-6f) * kPixelsPerRadian;
            drag(source, controls, *camera, 0, {400, 450}, {400.f, 450.f - back});
            settle(controls, *camera, damping ? 400 : 1);

            CHECK(camera->position.distanceTo(startPosition) < 1e-3f);
            CHECK(angleBetween(camera->quaternion, startQuaternion) < 1e-3);
        }
    }
}

TEST_CASE("writes to the camera and the target from outside take effect") {

    const bool damping = GENERATE(false, true);
    {

        CAPTURE(damping);

        FakeEventSource source;
        auto camera = makeCamera(false);
        camera->position.set(3, 4, 5);

        OrbitControls controls(*camera, source);
        controls.target.set(1, 0.5f, -2);
        controls.enableDamping = damping;
        settle(controls, *camera, 2);

        SECTION("camera position") {

            const Vector3 written(-2, 1, 6);
            camera->position = written;
            step(controls, *camera);

            CHECK(camera->position.distanceTo(written) < 1e-4f);
            CHECK(cameraForward(*camera).dot((controls.target - written).normalize()) > 0.99999f);
        }

        SECTION("target") {

            const auto position = camera->position;
            const Vector3 target(0, 2, 1);
            controls.target = target;
            step(controls, *camera);

            CHECK(controls.target.distanceTo(target) < 1e-6f);
            CHECK(camera->position.distanceTo(position) < 1e-4f);
            CHECK(cameraForward(*camera).dot((target - position).normalize()) > 0.99999f);
        }

        SECTION("both, as a rigid translation") {

            const auto quaternion = camera->quaternion;
            const Vector3 delta(250, -3, 40);
            const auto position = camera->position + delta;
            const auto target = controls.target + delta;
            camera->position = position;
            controls.target = target;
            step(controls, *camera);

            CHECK(camera->position.distanceTo(position) < 1e-3f);
            CHECK(controls.target.distanceTo(target) < 1e-6f);
            CHECK(angleBetween(camera->quaternion, quaternion) < 1e-4);
        }

        SECTION("back onto a value an earlier update wrote") {

            const auto positionA = camera->position;
            const auto quaternionA = camera->quaternion;

            drag(source, controls, *camera, 0, {300, 300}, {450, 260});
            settle(controls, *camera, damping ? 400 : 1);
            REQUIRE(camera->position.distanceTo(positionA) > 1.f);

            camera->position = positionA;
            step(controls, *camera);

            CHECK(camera->position.distanceTo(positionA) < 1e-4f);
            CHECK(angleBetween(camera->quaternion, quaternionA) < 1e-4);

            // and the target, after a pan moved it away
            const auto targetA = controls.target;
            drag(source, controls, *camera, 1, {400, 300}, {300, 250});
            settle(controls, *camera, damping ? 400 : 1);
            REQUIRE(controls.target.distanceTo(targetA) > 0.5f);

            const auto position = camera->position;
            controls.target = targetA;
            step(controls, *camera);

            CHECK(controls.target.distanceTo(targetA) < 1e-6f);
            CHECK(camera->position.distanceTo(position) < 1e-4f);
            CHECK(cameraForward(*camera).dot((targetA - position).normalize()) > 0.99999f);
        }

        SECTION("camera up") {

            const auto position = camera->position;
            camera->up.set(0, 0, 1);
            step(controls, *camera);

            CHECK(camera->position.distanceTo(position) < 1e-4f);
            // the horizon is level about the new up
            CHECK(std::abs(cameraAxis(*camera, 0).z) < 1e-5f);

            // and a sideways drag orbits about it
            const auto height = camera->position.z - controls.target.z;
            const auto distance = controls.getDistance();
            drag(source, controls, *camera, 0, {300, 300}, {450, 300});
            settle(controls, *camera, damping ? 400 : 1);

            CHECK(camera->position.distanceTo(position) > 1.f);
            CHECK(std::abs(camera->position.z - controls.target.z - height) < 1e-4f);
            CHECK(std::abs(controls.getDistance() - distance) < 1e-4f);
            CHECK(std::abs(cameraAxis(*camera, 0).z) < 1e-5f);
        }
    }
}

TEST_CASE("update reports whether the camera changed") {

    for (const bool damping : {false, true}) {

        CAPTURE(damping);

        FakeEventSource source;
        auto camera = makeCamera(false);
        camera->position.set(3, 4, 5);

        OrbitControls controls(*camera, source);
        controls.enableDamping = damping;

        // the constructor's own update has already placed the camera
        CHECK_FALSE(step(controls, *camera));
        CHECK_FALSE(step(controls, *camera));

        // the wheel's scale is applied undamped by the handler's own update
        source.wheel(1);
        CHECK_FALSE(step(controls, *camera));

        // a rotate move: undamped, the handler's update applied all of it;
        // damped, the frames after it carry the rest
        source.press(0, {300, 300});
        source.move({340, 310});
        source.release(0, {340, 310});
        CHECK(step(controls, *camera) == damping);

        // once damped motion has decayed nothing more is reported
        settle(controls, *camera, 400);
        int quiet = 0;
        for (int i = 0; i < 50; ++i) quiet += step(controls, *camera) ? 0 : 1;
        CHECK(quiet == 50);

        camera->position.set(-4, 2, 1);
        CHECK(step(controls, *camera));

        quiet = 0;
        for (int i = 0; i < 50; ++i) quiet += step(controls, *camera) ? 0 : 1;
        CHECK(quiet == 50);
    }
}

TEST_CASE("dollying in stops short of the target's float precision and backs out") {

    for (const auto& target : kTargets) {

        CAPTURE(target);

        FakeEventSource source;
        auto camera = makeCamera(false);
        camera->position = onSphere(target, 1.f, 1.f, 0.7f);

        OrbitControls controls(*camera, source);
        controls.target = target;
        step(controls, *camera);

        const auto forward = cameraForward(*camera);

        for (int i = 0; i < 3000; ++i) {
            source.wheel(1);
            step(controls, *camera);
        }

        CHECK(camera->position != controls.target);
        CHECK(cameraForward(*camera).dot(forward) > 0.9999f);

        for (int i = 0; i < 100; ++i) {
            source.wheel(-1);
            step(controls, *camera);
        }

        const auto backedOut = controls.getDistance();
        for (int i = 0; i < 100; ++i) {
            source.wheel(-1);
            step(controls, *camera);
        }

        CHECK(controls.getDistance() > 100.f * backedOut);
        CHECK(cameraForward(*camera).dot(forward) > 0.9999f);
    }
}

TEST_CASE("the picture turns smoothly into, across and out of the polar clamp") {

    for (const auto& target : kTargets) {

        CAPTURE(target);

        FakeEventSource source;
        auto camera = makeCamera(false);
        camera->position = onSphere(target, 1.f, 0.3f, 0.7f);

        OrbitControls controls(*camera, source);
        controls.target = target;
        step(controls, *camera);

        // up into the clamp while turning, along it, and back out
        const std::array<Vector2, 4> path{Vector2(400, 200), Vector2(450, 300), Vector2(550, 300), Vector2(600, 200)};
        constexpr int steps = 100;

        double worst = 0;
        double allowed = 0;
        for (std::size_t leg = 0; leg + 1 < path.size(); ++leg) {

            const auto& from = path[leg];
            const auto& to = path[leg + 1];

            // the most one move can turn the view: all of its theta as roll at the
            // pole plus all of its phi
            const auto perStep = (std::abs(to.x - from.x) + std::abs(to.y - from.y)) / steps / kPixelsPerRadian;
            allowed = std::max(allowed, 1.01 * perStep + 1e-5);

            source.press(0, from);
            auto previous = camera->quaternion;
            for (int i = 1; i <= steps; ++i) {
                const auto t = static_cast<float>(i) / steps;
                source.move({from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t});
                step(controls, *camera);
                worst = std::max(worst, angleBetween(camera->quaternion, previous));
                previous = camera->quaternion;
            }
            source.release(0, to);
        }

        CHECK(worst < allowed);
    }
}

TEST_CASE("a camera under an untransformed parent orbits like one without") {

    const Vector3 target(100, 0, 50);

    FakeEventSource source;
    auto group = Group::create();
    std::shared_ptr<Camera> camera = makeCamera(false);
    group->add(camera);
    camera->position = onSphere(target, 1.f, 1.f, 0.7f);

    OrbitControls controls(*camera, source);
    controls.target = target;
    step(controls, *camera);

    const float set = heading(*camera);

    drag(source, controls, *camera, 0, {400, 150}, {400, 450});

    CHECK(cameraForward(*camera).y < -0.99999f);
    CHECK(angleBetween(heading(*camera), set) < 1e-4f);
}

TEST_CASE("a camera under a transformed parent keeps aiming with lookAt") {

    FakeEventSource source;
    auto group = Group::create();
    group->position.set(2, 0, 0);
    std::shared_ptr<Camera> camera = makeCamera(false);
    group->add(camera);
    camera->position.set(0, 1, 5);

    OrbitControls controls(*camera, source);
    step(controls, *camera);

    drag(source, controls, *camera, 0, {300, 300}, {420, 250});

    // lookAt() aims from the world position at the target as a world point
    Vector3 world;
    camera->getWorldPosition(world);
    CHECK(cameraForward(*camera).dot((controls.target - world).normalize()) > 0.99999f);
}
