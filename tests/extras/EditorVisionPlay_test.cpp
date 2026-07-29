// The PhysX-free half of the sensor session.
//
// SensorPlaySession runs the vision sensors and the recording plumbing with no
// physics world at all — that is the point of the base/derived split, and this
// test is the proof that compiles everywhere: it drives the BASE class exactly
// the way a no-PhysX editor build does. No SDK, no renderer (a scan needs a GL
// context, so the clouds themselves are asserted in the editor's --selftest),
// which leaves the seams: authored config in, built sensors and honest statuses
// out, a dt-accumulated sim clock, and everything gone again after Stop.
//
// The PhysX half — live IMUs, contacts, joint sensors, the world clock — is
// EditorSensorPlay_test, which only exists where the SDK was found.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/SensorPlaySession.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <memory>
#include <string>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

namespace {

    constexpr float kFrame = 1.f / 60.f;

    // The base session, wired the way a no-PhysX editor wires it: a rig for the
    // sensor nodes, no physics, no renderer.
    struct Rig {
        Scene scene;
        std::shared_ptr<Group> rig = Group::create();
        SensorPlaySession sensors;

        Rig() {
            scene.addRef(*rig);
            sensors.setRig(rig.get());
        }

        void update(int frames) {
            for (int i = 0; i < frames; ++i) sensors.update(kFrame);
        }
    };

    std::shared_ptr<Mesh> authorLidar(Scene& scene, const char* name) {

        auto post = ObjectFactory::createPrimitive(Primitive::Box, scene);
        post->name = name;

        SensorConfig sensor;
        sensor.enabled = true;
        sensor.type = SensorConfig::Type::Lidar;
        sensor.beams = SensorConfig::Beams::VLP16;
        sensor.faceSize = 32;
        sensor.rateHz = 10.f;
        sensor.write(*post);
        return post;
    }

}// namespace


TEST_CASE("A vision sensor without a renderer is built and says so") {

    Rig rig;
    auto post = authorLidar(rig.scene, "Mast");
    rig.scene.add(post);

    rig.sensors.start(rig.scene);
    REQUIRE(rig.sensors.liveCount() == 1);
    const auto& entry = *rig.sensors.entries().front();
    CHECK(entry.lidar != nullptr);
    CHECK_FALSE(entry.status.empty());// "no renderer"
    // It IS in the rig, so the editor's export/snapshot filter covers it.
    CHECK(entry.lidar->parent == rig.rig.get());

    rig.update(30);
    CHECK(entry.scans == 0);// nothing to scan with
    rig.sensors.stop();
    CHECK(rig.rig->children.empty());
}

TEST_CASE("With no physics world the clock accumulates the frame delta") {

    // The pushed sensors are stamped by the world; without one the session's own
    // dt accumulation is the time base every scan would be stamped with. Sim
    // time, never wall time — 30 updates of 1/60 s is half a second no matter
    // how long they took (to float precision: dt arrives as a float).
    Rig rig;
    rig.scene.add(authorLidar(rig.scene, "Mast"));

    rig.sensors.start(rig.scene);
    CHECK_THAT(rig.sensors.simTime(), WithinAbs(0.0, 1e-12));
    rig.update(30);
    CHECK_THAT(rig.sensors.simTime(), WithinAbs(30.0 / 60.0, 1e-6));

    rig.sensors.stop();
    CHECK_THAT(rig.sensors.simTime(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("A body sensor in a physics-free session authors, reports, and stays inert") {

    // The base class does not pretend: an IMU is counted (the authoring took),
    // not live (nothing simulates it), and the status names the missing build.
    Rig rig;
    auto box = ObjectFactory::createPrimitive(Primitive::Box, rig.scene);
    box->name = "Body";

    SensorConfig sensor;
    sensor.enabled = true;
    sensor.type = SensorConfig::Type::Imu;
    sensor.write(*box);
    rig.scene.add(box);

    rig.sensors.start(rig.scene);
    CHECK(rig.sensors.sensorCount() == 1);
    CHECK(rig.sensors.liveCount() == 0);
    CHECK(rig.sensors.entries().front()->status.find("PhysX") != std::string::npos);

    rig.update(10);
    CHECK(rig.sensors.entries().front()->samples == 0);
    rig.sensors.stop();

    // And the authored config survived untouched.
    const auto config = SensorConfig::read(*box);
    REQUIRE(config.has_value());
    CHECK(config->type == SensorConfig::Type::Imu);
}

TEST_CASE("The all-joints encoder does not fan out without a physics build") {

    // The fan-out lives in the PhysX subclass; here one authored entry stays
    // ONE reported entry, not a per-DOF spray of identical failures.
    Rig rig;
    auto robot = Group::create();
    robot->name = "Arm";

    SensorConfig sensor;
    sensor.enabled = true;
    sensor.type = SensorConfig::Type::Encoder;
    sensor.joint = SensorConfig::allJoints;
    sensor.write(*robot);
    rig.scene.add(robot);

    rig.sensors.start(rig.scene);
    CHECK(rig.sensors.sensorCount() == 1);
    CHECK(rig.sensors.liveCount() == 0);
    CHECK_FALSE(rig.sensors.entries().front()->status.empty());
    rig.sensors.stop();
}

TEST_CASE("Stop clears the vision sensors, and Play/Stop twice is fine") {

    Rig rig;
    rig.scene.add(authorLidar(rig.scene, "Mast"));

    for (int pass = 0; pass < 2; ++pass) {
        rig.sensors.start(rig.scene);
        CHECK(rig.sensors.liveCount() == 1);
        rig.update(5);
        rig.sensors.stop();
        CHECK(rig.sensors.sensorCount() == 0);
        CHECK(rig.sensors.entries().empty());
    }

    // The sensor nodes are gone from the rig too — nothing left parented to
    // editor furniture that outlives the play.
    CHECK(rig.rig->children.empty());
}
