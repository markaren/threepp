// Authored vehicles, played.
//
// The model under test (see VehicleConfig): the config sits on the model's
// root, the four wheels are picked by name, geometry is derived from them, and
// Play builds a PhysxVehicle whose chassis the root mirrors. Each case steps a
// fixed 1/60 s clock, the same doctrine as EditorPhysicsPlay_test: nothing
// here may depend on wall time, and anything transient is sampled over the
// run, never read at a single instant.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/JointConfig.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/VehicleConfig.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/geometries/CylinderGeometry.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

namespace {

    constexpr float kDt = 1.f / 60.f;

    // A static slab for the tires to push against, top face at y = 0.
    void addGround(Scene& scene) {

        auto ground = Mesh::create(BoxGeometry::create(400.f, 1.f, 400.f));
        ground->name = "Ground";
        ground->position.set(0.f, -0.5f, 0.f);

        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Static;
        config.shape = PhysicsConfig::Shape::Box;
        config.write(*ground);
        scene.add(ground);
    }

    // A primitives car: box body over four cylinder wheels named FR/FL/RR/RL,
    // hubs at (±0.8, 0.4, ±1.4), radius 0.4 — resting exactly on the ground.
    std::shared_ptr<Group> addCar(Scene& scene, const char* name = "Car",
                                  const Vector3& position = {}) {

        auto car = Group::create();
        car->name = name;
        car->position.copy(position);

        auto body = Mesh::create(BoxGeometry::create(1.6f, 0.8f, 4.2f));
        body->name = "Body";
        body->position.set(0.f, 1.f, 0.f);
        car->add(body);

        const char* names[4] = {"FR", "FL", "RR", "RL"};
        const Vector3 hubs[4] = {{0.8f, 0.4f, 1.4f},
                                 {-0.8f, 0.4f, 1.4f},
                                 {0.8f, 0.4f, -1.4f},
                                 {-0.8f, 0.4f, -1.4f}};
        for (int i = 0; i < 4; ++i) {
            auto wheel = Mesh::create(CylinderGeometry::create(0.4f, 0.4f, 0.3f, 24));
            wheel->name = names[i];
            wheel->rotation.z = math::PI / 2;// cylinder height (Y) onto the axle (X)
            wheel->position.copy(hubs[i]);
            car->add(wheel);
        }

        VehicleConfig config;
        config.wheels = {"FR", "FL", "RR", "RL"};
        config.write(*car);
        scene.add(car);
        return car;
    }

}// namespace


TEST_CASE("throttle drives an authored car forward", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);
    auto car = addCar(scene);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.vehicleCount() == 1);
    // The chassis actor answers for the root, so joints, sensors and scripts
    // can name the car.
    CHECK(session.findActor(car.get()) != nullptr);

    // 4 seconds of full throttle. The car was authored facing +Z (the front
    // axle is at +z), so +Z is where it must go.
    float maxZ = 0.f;
    float maxY = 0.f;
    for (int i = 0; i < 240; ++i) {
        session.driveVehicles(true, false, 0.f, false, kDt);
        session.update(kDt);
        maxZ = std::max(maxZ, car->position.z);
        maxY = std::max(maxY, car->position.y);
    }

    CHECK(maxZ > 3.f);                            // it drove, several metres
    CHECK(std::abs(car->position.x) < 1.f);       // straight
    CHECK(maxY < 1.f);                            // and stayed on the ground

    session.stop();
    CHECK(session.vehicleCount() == 0);
}

TEST_CASE("the car rests at its authored ride height", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);
    auto car = addCar(scene);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.vehicleCount() == 1);

    // No input, 2 seconds: the settle compensation in the derived suspension
    // attachment means the body neither sags nor pops. Sampled over the run —
    // the first frames are where a bad spawn would bounce.
    float minY = 1e30f, maxY = -1e30f;
    for (int i = 0; i < 120; ++i) {
        session.update(kDt);
        minY = std::min(minY, car->position.y);
        maxY = std::max(maxY, car->position.y);
    }
    CHECK(minY > -0.15f);
    CHECK(maxY < 0.15f);

    session.stop();
}

TEST_CASE("steer plus throttle yaws the car", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);
    auto car = addCar(scene);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.vehicleCount() == 1);

    // Full left steer under throttle. Heading is read off the root's world
    // forward; sampled as the largest turn reached, since a circling car's
    // instantaneous heading wraps.
    Vector3 forward0(0.f, 0.f, 1.f);
    forward0.applyQuaternion(car->quaternion);
    float maxTurn = 0.f;
    for (int i = 0; i < 240; ++i) {
        session.driveVehicles(true, false, 1.f, false, kDt);
        session.update(kDt);
        Vector3 forward(0.f, 0.f, 1.f);
        forward.applyQuaternion(car->quaternion);
        maxTurn = std::max(maxTurn, forward.angleTo(forward0));
    }
    CHECK(maxTurn > 0.5f);// well past any straight-line wobble

    session.stop();
}

TEST_CASE("the brake stops a rolling car", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);
    auto car = addCar(scene);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.vehicleCount() == 1);
    const auto* played = session.findVehicle(car.get());
    REQUIRE(played);
    REQUIRE(played->drive);

    // Get it rolling...
    for (int i = 0; i < 120; ++i) {
        session.driveVehicles(true, false, 0.f, false, kDt);
        session.update(kDt);
    }
    REQUIRE(played->drive->forwardSpeed() > 2.f);

    // ...then stand on the brake (SPACE, so the backward input's
    // brake-then-reverse split is not what is being tested here).
    for (int i = 0; i < 180; ++i) {
        session.driveVehicles(false, false, 0.f, true, kDt);
        session.update(kDt);
    }
    CHECK(std::abs(played->drive->forwardSpeed()) < 0.3f);

    session.stop();
}

TEST_CASE("the backward input reverses a standing car", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);
    auto car = addCar(scene);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.vehicleCount() == 1);

    // Held S from rest: the arcade split selects reverse and throttles.
    float minZ = 0.f;
    for (int i = 0; i < 240; ++i) {
        session.driveVehicles(false, true, 0.f, false, kDt);
        session.update(kDt);
        minZ = std::min(minZ, car->position.z);
    }
    CHECK(minZ < -1.f);

    session.stop();
}

TEST_CASE("the wheels visually spin and steer", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);
    auto car = addCar(scene);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.vehicleCount() == 1);
    const auto* played = session.findVehicle(car.get());
    REQUIRE(played);

    auto* wheelFR = played->wheels[0];
    REQUIRE(wheelFR != nullptr);
    const Quaternion before = wheelFR->quaternion;

    // Under throttle the wheel MESH must turn — the mirror writes the spin
    // onto the node, not just the physics state. Sampled as the largest
    // orientation change, since a spinning wheel returns near its start.
    float maxDelta = 0.f;
    float maxSpin = 0.f;
    for (int i = 0; i < 120; ++i) {
        session.driveVehicles(true, false, 0.f, false, kDt);
        session.update(kDt);
        maxDelta = std::max(maxDelta, before.angleTo(wheelFR->quaternion));
        maxSpin = std::max(maxSpin, std::abs(played->drive->wheelAngularSpeed(0)));
    }
    CHECK(maxDelta > 0.5f);// radians of visible roll
    CHECK(maxSpin > 2.f);  // rad/s of physics spin behind it

    session.stop();
}

TEST_CASE("two authored cars play side by side", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);
    // Same wheel names on both on purpose: picks resolve against the MODEL's
    // own subtree, so names only need to be unique within one car. Two
    // vehicles also cross the PhysX vehicle-extension init/close pairing.
    auto left = addCar(scene, "Left", {-4.f, 0.f, 0.f});
    auto right = addCar(scene, "Right", {4.f, 0.f, 0.f});

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.vehicleCount() == 2);

    float leftMaxZ = 0.f, rightMaxZ = 0.f;
    for (int i = 0; i < 240; ++i) {
        session.driveVehicles(true, false, 0.f, false, kDt);
        session.update(kDt);
        leftMaxZ = std::max(leftMaxZ, left->position.z);
        rightMaxZ = std::max(rightMaxZ, right->position.z);
    }
    CHECK(leftMaxZ > 3.f);
    CHECK(rightMaxZ > 3.f);
    // Each stayed in its lane: they simulate independently.
    CHECK_THAT(left->position.x, WithinAbs(-4.f, 1.f));
    CHECK_THAT(right->position.x, WithinAbs(4.f, 1.f));

    session.stop();
}

TEST_CASE("an authored joint hitches a trailer to the car", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);
    auto car = addCar(scene);

    // The trailer: a dynamic crate behind the car, tethered to the chassis by
    // a distance joint — the composition the docs promise (a trailer is a
    // second body plus an authored joint naming the car).
    auto trailer = Mesh::create(BoxGeometry::create(1.f, 1.f, 1.f));
    trailer->name = "Trailer";
    trailer->position.set(0.f, 0.5f, -4.f);
    {
        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Dynamic;
        config.shape = PhysicsConfig::Shape::Box;
        config.write(*trailer);
    }
    scene.add(trailer);

    auto hitch = Object3D::create();
    hitch->name = "Hitch";
    hitch->position.set(0.f, 0.f, 1.f);// the trailer's nose
    {
        JointConfig config;
        config.type = JointConfig::Type::Distance;
        config.body = "Car";// resolves to the chassis actor the session registered
        config.lower = 0.f;
        config.upper = 2.5f;
        config.write(*hitch);
    }
    trailer->add(hitch);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.vehicleCount() == 1);
    REQUIRE(session.jointCount() == 1);

    float trailerMaxZ = -10.f;
    for (int i = 0; i < 300; ++i) {// 5 s of towing
        session.driveVehicles(true, false, 0.f, false, kDt);
        session.update(kDt);
        trailerMaxZ = std::max(trailerMaxZ, trailer->position.z);
    }
    // The crate started 4 m behind; being dragged past the car's start line
    // proves the tether grabbed the chassis, not the world.
    CHECK(trailerMaxZ > 1.f);

    session.stop();
}

TEST_CASE("a missing wheel pick is skipped, play intact", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();
    addGround(scene);
    auto car = addCar(scene);

    auto config = VehicleConfig::read(*car);
    REQUIRE(config);
    config->wheels[2] = "Nowhere";
    config->write(*car);

    PhysicsPlaySession session;
    std::vector<std::string> logged;
    session.setLogger([&](const std::string& line) { logged.push_back(line); });
    session.start(scene);

    // No vehicle, one line saying why, and the ground still plays.
    CHECK(session.vehicleCount() == 0);
    CHECK(session.bodyCount() == 1);
    REQUIRE_FALSE(logged.empty());
    CHECK(logged.front().find("Nowhere") != std::string::npos);

    for (int i = 0; i < 30; ++i) session.update(kDt);

    session.stop();
}
