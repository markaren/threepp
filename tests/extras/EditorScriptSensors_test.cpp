// threepp.editor's sensor handles, driven exactly as the editor drives them:
// physics, sensors and scripts are independent PlaySessions stepped in that
// order, and a script reaches its sensors through a free function whose only
// context is "the session that is playing".
//
// The point of these handles is the noise. Ground truth is already reachable
// (articulation_from_object reports exact joint positions), so what is asserted
// here is that a script sees the SENSOR's numbers — rate-gated, seeded,
// quantized — and that reading them costs the Sensors panel nothing.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "Scripting.hpp"

#include "threepp/extras/editor/ArticulationConfig.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/ScriptConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"

#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/scenes/Scene.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

namespace {

    constexpr float kFrame = 1.f / 60.f;// exactly one default substep per update
    constexpr float kG = 9.81f;

    // A dynamic box carrying an IMU. Noiseless by default: a zero NoiseModel is
    // a bit-exact passthrough, which is what lets a test assert the physics
    // truth the sensor is supposed to be reporting.
    std::shared_ptr<Mesh> addInstrumentedBox(Scene& scene, const char* name,
                                             const std::string& script, bool noisy = false) {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->name = name;
        box->position.set(0.f, 3.f, 0.f);

        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Dynamic;
        physics.shape = PhysicsConfig::Shape::Box;
        physics.mass = 2.f;
        physics.write(*box);

        SensorConfig sensor;
        sensor.enabled = true;
        sensor.type = SensorConfig::Type::Imu;
        sensor.rateHz = 0.f;// every substep
        if (!noisy) {
            sensor.gyroNoiseDensity = 0.f;
            sensor.gyroRandomWalk = 0.f;
            sensor.accelNoiseDensity = 0.f;
            sensor.accelRandomWalk = 0.f;
        }
        sensor.write(*box);

        ScriptConfig config;
        config.source = script;
        config.write(*box);

        scene.add(box);
        return box;
    }

    // The editor's own registration order: physics, then sensors, then scripts.
    // A script therefore reads the measurements taken from the step that just
    // happened, and its commands land on the next one.
    struct Rig {

        SceneDocument document;
        PhysicsPlaySession physics;
        PhysxSensorPlaySession sensors;
        ScriptPlaySession scripts;

        Rig() { sensors.setPhysics(&physics); }

        Scene& scene() { return document.scene(); }

        void start() {
            physics.start(scene());
            sensors.start(scene());
            scripts.start(scene());
        }

        void run(int frames) {
            for (int i = 0; i < frames; ++i) {
                physics.update(kFrame);
                sensors.update(kFrame);
                scripts.update(kFrame);
            }
        }

        // Stop order is the controller's — the REVERSE of registration, so a
        // script's stop() still has the sensors and the world under it. The
        // dead-handle window (a teardown that skips the controller) is driven
        // by hand in the test that pins it, not manufactured here.
        void stop() {
            scripts.stop();
            sensors.stop();
            physics.stop();
        }
    };

    // "At rest" in the accelerometer sense: proper acceleration zero while world
    // gravity is unchanged, i.e. a body on a table. Disabling gravity on the
    // actor models that without needing a ground and a settle.
    void holdStill(PhysicsPlaySession& physics, const Object3D& object) {

        using namespace ::physx;
        auto* actor = physics.findActor(&object);
        REQUIRE(actor != nullptr);
        auto* body = actor->is<PxRigidDynamic>();
        REQUIRE(body != nullptr);
        body->setActorFlag(PxActorFlag::eDISABLE_GRAVITY, true);
        body->setLinearVelocity(PxVec3(0.f));
        body->setAngularVelocity(PxVec3(0.f));
    }

    Object3D* marker(Scene& scene, const char* name = "Ledger") {

        return scene.getObjectByName(name);
    }

    // A three-link, two-joint arm: an all-joints encoder must fan out to TWO
    // live encoders, which a one-joint arm could not show.
    const char* kArmUrdf = R"(
        <robot name="arm2">
          <link name="base_link">
            <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
            <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
          </link>
          <link name="upper_link">
            <visual><geometry><box size="0.1 0.4 0.1"/></geometry></visual>
            <collision><geometry><box size="0.1 0.4 0.1"/></geometry></collision>
          </link>
          <link name="fore_link">
            <visual><geometry><box size="0.1 0.3 0.1"/></geometry></visual>
            <collision><geometry><box size="0.1 0.3 0.1"/></geometry></collision>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base_link"/>
            <child link="upper_link"/>
            <origin xyz="0 0 0.3" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-2.0" upper="2.0"/>
          </joint>
          <joint name="elbow" type="revolute">
            <parent link="upper_link"/>
            <child link="fore_link"/>
            <origin xyz="0 0.4 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-2.0" upper="2.0"/>
          </joint>
        </robot>)";

    std::filesystem::path armFixture() {

        const auto dir = std::filesystem::temp_directory_path() / "threepp-script-sensors";
        std::filesystem::create_directories(dir);
        const auto path = dir / "arm2.urdf";
        std::ofstream(path, std::ios::trunc) << kArmUrdf;
        return path;
    }

    // A simulated arm carrying an all-joints encoder and a script.
    std::shared_ptr<Robot> addSimulatedArm(Scene& scene, const std::string& script) {

        const auto path = armFixture();
        URDFLoader loader;
        auto robot = loader.load(path);

        RobotConfig rc;
        rc.urdf = path.string();
        rc.joints = std::vector<float>(robot->numDOF(), 0.f);
        rc.write(*robot);

        ArticulationConfig ac;
        ac.enabled = true;
        ac.fixedBase = true;
        ac.write(*robot);

        // ONE authored entry on the robot's root; the session fans it out to one
        // live encoder per DOF.
        SensorConfig sensor;
        sensor.enabled = true;
        sensor.type = SensorConfig::Type::Encoder;
        sensor.rateHz = 0.f;
        sensor.joint = SensorConfig::allJoints;
        // A coarse encoder, so "the script is reading the sensor and not the
        // simulation" is visible in the numbers: 2048 counts/rev is ~3 mrad.
        sensor.encoderResolution = 0.003f;
        sensor.write(*robot);

        ScriptConfig config;
        config.source = script;
        config.write(*robot);

        scene.add(robot);
        return robot;
    }

}// namespace


TEST_CASE("a script reads the IMU the session is running", "[editor][scripting][sensors][physx]") {

    Rig rig;

    // Records into a marker's transform, the only numeric channel a script has
    // back into a test: position = the newest acceleration, scale = (samples
    // seen, newest timestamp, oldest timestamp).
    auto box = addInstrumentedBox(rig.scene(), "Body", R"(
import threepp

class Reader:
    def start(self, obj):
        self.imu = threepp.editor.imu_from_object(obj)
        self.marker = threepp.Object3D()
        self.marker.name = "Ledger"
        obj.parent.add(self.marker)
        self.seen = 0
        self.first = 0.0

    def update(self, dt):
        if self.imu is None:
            return
        fresh = self.imu.read_new()
        if not fresh:
            return
        if self.seen == 0:
            self.first = fresh[0].time
        self.seen += len(fresh)
        s = self.imu.latest()
        self.marker.position.copy(s.acceleration)
        self.marker.scale.set(float(self.seen), float(s.time), float(self.first))
)");

    rig.start();
    CHECK(rig.scripts.errorFor(box->uuid) == "");
    REQUIRE(rig.sensors.liveCount() == 1);

    holdStill(rig.physics, *box);
    rig.run(30);
    CHECK(rig.scripts.errorFor(box->uuid) == "");

    auto* ledger = marker(rig.scene());
    REQUIRE(ledger != nullptr);

    // The IMU convention, straight through the handle: a level accelerometer at
    // rest reads +g on its up axis (specific force, not proper acceleration).
    CHECK_THAT(ledger->position.y, WithinAbs(kG, 1e-3));
    CHECK_THAT(ledger->position.x, WithinAbs(0.f, 1e-3));
    CHECK_THAT(ledger->position.z, WithinAbs(0.f, 1e-3));

    // Samples arrived, and their timestamps advanced with the simulation rather
    // than with the wall clock.
    CHECK(ledger->scale.x > 20.f);
    CHECK(ledger->scale.y > ledger->scale.z);
    CHECK_THAT(ledger->scale.y, WithinAbs(30.f / 60.f, 1e-3));

    // And the panel was never starved: the session's own counters see every
    // measurement the script did.
    CHECK(rig.sensors.entries().front()->samples >= static_cast<std::size_t>(ledger->scale.x));
    CHECK(rig.sensors.entries().front()->traces[4].count > 0);// accel y still plotted

    rig.stop();
}

TEST_CASE("the readings a script sees carry the noise", "[editor][scripting][sensors][physx]") {

    // The whole reason these handles exist: with a noise model authored, the
    // reading is NOT the physics truth the same script could read off the body.
    Rig rig;
    auto box = addInstrumentedBox(
            rig.scene(), "Body", R"(
import threepp

class Reader:
    def start(self, obj):
        self.imu = threepp.editor.imu_from_object(obj)
        self.marker = threepp.Object3D()
        self.marker.name = "Ledger"
        obj.parent.add(self.marker)

    def update(self, dt):
        s = self.imu.latest() if self.imu else None
        if s is not None:
            self.marker.position.copy(s.acceleration)
)",
            /*noisy*/ true);

    rig.start();
    CHECK(rig.scripts.errorFor(box->uuid) == "");
    holdStill(rig.physics, *box);
    rig.run(30);

    auto* ledger = marker(rig.scene());
    REQUIRE(ledger != nullptr);
    // Near g, but not g: a MEMS-class accelerometer at rest does not report a
    // round number, and a controller tuned against one that did is a controller
    // that has never met hardware. The bound is a few sigma — the authored
    // 0.06 m/s^2/sqrt(Hz) at 60 Hz is a ~0.47 m/s^2 per-sample deviation.
    CHECK(ledger->position.y != kG);
    CHECK_THAT(ledger->position.y, WithinAbs(kG, 2.5));

    rig.stop();
}

TEST_CASE("a sensor is None outside Play and dead after it", "[editor][scripting][sensors][physx]") {

    Rig rig;

    auto box = addInstrumentedBox(rig.scene(), "Ledger", R"(
import threepp

class Ledger:
    def start(self, obj):
        self.obj = obj
        self.imu = threepp.editor.imu_from_object(obj)
        obj.name = "got" if self.imu else "none"

    def update(self, dt):
        pass

    def stop(self):
        # The test tears the sensor session down FIRST, by hand - the hostile
        # order a controller-skipping teardown can still produce - so the handle
        # must say it is gone rather than read a freed entry.
        self.obj.name += "/valid" if self.imu.valid else "/dead"
        try:
            self.imu.latest()
            self.obj.name += "/read"
        except RuntimeError:
            self.obj.name += "/raised"
        try:
            self.imu.read_new()
            self.obj.name += "/read"
        except RuntimeError:
            self.obj.name += "/raised"
)");

    // Scripts alone: no sensor session means no sensor.
    rig.scripts.start(rig.scene());
    CHECK(rig.scripts.errorFor(box->uuid) == "");
    CHECK(box->name == "none");
    rig.scripts.stop();

    box->name = "Ledger";
    rig.start();
    CHECK(box->name == "got");
    rig.run(10);

    // The hostile order, by hand: sensors and world torn down under a script
    // still holding a handle. The controller no longer produces this (it stops
    // sessions in reverse registration order), but a destructor-order teardown
    // still can, and a handle stashed beyond its session always could — THIS is
    // what the lifetime token is for.
    rig.sensors.stop();
    rig.physics.stop();
    rig.scripts.stop();

    CHECK(box->name == "got/dead/raised/raised");
}

TEST_CASE("read_new advances a per-handle cursor", "[editor][scripting][sensors][physx]") {

    Rig rig;

    // Two handles on the same sensor. Each must see every measurement — if the
    // cursor were shared (or if a handle drained the sensor) they would split
    // the stream between them, and the panel would get whatever was left.
    auto box = addInstrumentedBox(rig.scene(), "Body", R"(
import threepp

class Twice:
    def start(self, obj):
        self.a = threepp.editor.imu_from_object(obj)
        self.b = threepp.editor.imu_from_object(obj)
        self.marker = threepp.Object3D()
        self.marker.name = "Ledger"
        obj.parent.add(self.marker)
        self.count_a = 0
        self.count_b = 0
        self.leaked = 0
        # A fresh handle reads what arrives from now on, so before any step
        # there is nothing to catch up on.
        self.backlog = len(self.a.read_new())

    def update(self, dt):
        self.count_a += len(self.a.read_new())
        # A second call in the same frame has nothing new to hand back.
        self.leaked += len(self.a.read_new())
        self.count_b += len(self.b.read_new())
        self.marker.position.set(float(self.count_a), float(self.count_b), float(self.leaked))
        self.marker.scale.x = float(self.backlog)
)");

    rig.start();
    CHECK(rig.scripts.errorFor(box->uuid) == "");
    rig.run(20);

    auto* ledger = marker(rig.scene());
    REQUIRE(ledger != nullptr);
    CHECK(ledger->position.x > 10.f);              // it read something
    CHECK(ledger->position.x == ledger->position.y);// neither handle starved the other
    CHECK(ledger->position.z == 0.f);              // the immediate second read was empty
    CHECK(ledger->scale.x == 0.f);                 // and a fresh handle starts empty

    rig.stop();
}

TEST_CASE("an all-joints encoder resolves by joint name", "[editor][scripting][sensors][physx]") {

    Rig rig;

    // The closed-loop demo, and the fan-out rules with it: one authored entry,
    // two live encoders, so an unqualified ask has to refuse rather than pick.
    auto robot = addSimulatedArm(rig.scene(), R"(
import threepp

class Elbow:
    target = 0.6

    def start(self, obj):
        self.art = threepp.editor.articulation_from_object(obj)
        self.all = threepp.editor.encoders_from_object(obj)
        self.ambiguous = 0
        try:
            threepp.editor.encoder_from_object(obj)
        except RuntimeError:
            self.ambiguous = 1
        self.unknown = 0
        try:
            threepp.editor.encoder_from_object(obj, joint="wrist")
        except RuntimeError:
            self.unknown = 1
        self.enc = threepp.editor.encoder_from_object(obj, joint="elbow")
        self.marker = threepp.Object3D()
        self.marker.name = "Ledger"
        obj.parent.add(self.marker)
        self.reads = 0

    def update(self, dt):
        s = self.enc.latest()
        if s is None:
            return
        self.reads += len(self.enc.read_new())
        # Closed loop on the NOISY, quantized reading — not on the exact joint
        # position the articulation would hand over.
        self.art.set_drive_target(self.enc.joint, s.position + 0.5 * (self.target - s.position))
        self.marker.position.set(float(len(self.all)), float(self.ambiguous), float(self.unknown))
        self.marker.scale.set(float(self.reads), s.position, s.velocity)
)");

    rig.start();
    CHECK(rig.scripts.errorFor(robot->uuid) == "");
    REQUIRE(rig.sensors.liveCount() == 2);// one live encoder per DOF

    rig.run(240);
    CHECK(rig.scripts.errorFor(robot->uuid) == "");

    auto* ledger = marker(rig.scene());
    REQUIRE(ledger != nullptr);
    CHECK(ledger->position.x == 2.f);// encoders_from_object saw both DOFs
    CHECK(ledger->position.y == 1.f);// an unqualified ask refused
    CHECK(ledger->position.z == 1.f);// so did an unknown joint name
    CHECK(ledger->scale.x > 100.f);  // readings kept coming

    // The loop closed: the elbow travelled from 0 toward the setpoint, driven by
    // what the encoder said rather than by what the solver knew.
    CHECK(ledger->scale.y > 0.2f);
    // And the reading is the SENSOR's: quantized to the authored tick, so it is
    // a whole number of ticks rather than the solver's continuous angle.
    const float ticks = ledger->scale.y / 0.003f;
    CHECK_THAT(ticks - std::round(ticks), WithinAbs(0.f, 1e-2));

    rig.stop();
}

// The ElbowHold script from doc/editor.md, verbatim. If this stops working the
// documentation is handing users a script that does not run.
TEST_CASE("the documented encoder script holds its setpoint", "[editor][scripting][sensors][physx]") {

    Rig rig;
    auto robot = addSimulatedArm(rig.scene(), R"(
import threepp


class ElbowHold:
    target = 0.6
    gain = 0.5

    def start(self, obj: threepp.Robot):
        self.art = threepp.editor.articulation_from_object(obj)
        self.enc = threepp.editor.encoder_from_object(obj, joint="elbow")

    def update(self, dt: float):
        if self.enc is None:
            return
        reading = self.enc.latest()
        if reading is None:
            return  # no measurement yet this Play
        # Closed on the MEASURED position - quantized to whole encoder ticks and
        # noise-corrupted - not on the joint angle the solver knows.
        command = reading.position + self.gain * (self.target - reading.position)
        self.art.set_drive_target(self.enc.joint, command)
)");

    rig.start();
    CHECK(rig.scripts.errorFor(robot->uuid) == "");
    rig.run(240);
    CHECK(rig.scripts.errorFor(robot->uuid) == "");

    // Read the answer off the same encoder the script steered by.
    const JointEncoder* elbow = nullptr;
    for (const auto& entry : rig.sensors.entries()) {
        if (entry->config.joint == "elbow") elbow = entry->encoder.get();
    }
    REQUIRE(elbow != nullptr);
    const auto sample = elbow->latest();
    REQUIRE(sample.has_value());
    // Travelled from 0 toward the setpoint. Not all the way: the drive is a PD
    // with the authored stiffness and the forearm has gravity on it, which is
    // exactly the kind of steady-state error a real loop lives with.
    CHECK(sample->position > 0.2f);

    rig.stop();
}
