// Sensor authoring, played.
//
// Drives the real SensorPlaySession against a real PhysicsPlaySession, headless
// (no renderer), so the assertions are about the seam the editor actually uses:
// userData in, live sensors out, measurements stamped with SIM time, and
// everything gone again after Stop.
//
// Needs the PhysX SDK, so this target only exists where the SDK was found (see
// tests/extras/CMakeLists.txt). The format half — the userData round trip and
// the always-emit-all-keys rule — is EditorSensorConfig_test and runs everywhere.
//
// What is deliberately NOT here: the vision sensors' clouds. A scan needs a GL
// context, so that half is asserted in the editor's --selftest, which has one.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"
#include "threepp/extras/editor/SensorPlaySession.hpp"

#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;
using Catch::Matchers::WithinAbs;

namespace {

    constexpr float kFrame = 1.f / 60.f;// exactly one default substep per update
    constexpr float kG = 9.81f;

    // A dynamic box carrying an IMU. Noiseless: a zero NoiseModel is a bit-exact
    // passthrough, which is what makes the physics-truth assertions possible.
    std::shared_ptr<Mesh> makeInstrumentedBox(Scene& scene, const char* name) {

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
        sensor.gyroNoiseDensity = 0.f;
        sensor.gyroRandomWalk = 0.f;
        sensor.accelNoiseDensity = 0.f;
        sensor.accelRandomWalk = 0.f;
        sensor.write(*box);
        return box;
    }

    // Both sessions, wired the way EditorApp wires them (physics first, so the
    // world exists before the sensors register with it).
    struct Rig {
        Scene scene;
        std::shared_ptr<Group> rig = Group::create();
        PhysicsPlaySession physics;
        SensorPlaySession sensors;

        Rig() {
            scene.addRef(*rig);
            sensors.setPhysics(&physics);
            sensors.setRig(rig.get());
        }

        void start() {
            physics.start(scene);
            sensors.start(scene);
        }

        void update(int frames) {
            for (int i = 0; i < frames; ++i) {
                physics.update(kFrame);
                sensors.update(kFrame);
            }
        }

        // Stop order matters and is the controller's: physics first, so the world
        // is already gone when the sensor session shuts down. If the sensors
        // dereferenced it there, this is where it would crash.
        void stop() {
            physics.stop();
            sensors.stop();
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

}// namespace


TEST_CASE("An authored IMU comes up and reads +g at rest") {

    Rig rig;
    auto box = makeInstrumentedBox(rig.scene, "Body");
    rig.scene.add(box);

    rig.start();
    REQUIRE(rig.sensors.sensorCount() == 1);
    REQUIRE(rig.sensors.liveCount() == 1);
    const auto& entry = *rig.sensors.entries().front();
    CHECK(entry.status.empty());
    CHECK(entry.label == "Body");
    CHECK(entry.uuid == box->uuid);

    holdStill(rig.physics, *box);
    rig.update(30);

    CHECK(entry.samples > 0);
    // Sim time, not wall time: it must track the substeps we asked for.
    CHECK(entry.lastTime > 0.4);
    CHECK_THAT(rig.sensors.simTime(), WithinAbs(entry.lastTime, 1e-6));

    // A level accelerometer at rest reads +g on its up axis. The gyro reads zero.
    const auto sample = entry.imu->latest();
    REQUIRE(sample.has_value());
    CHECK_THAT(sample->linearAcceleration.y, WithinAbs(kG, 1e-3));
    CHECK_THAT(sample->linearAcceleration.x, WithinAbs(0.f, 1e-3));
    CHECK_THAT(sample->linearAcceleration.z, WithinAbs(0.f, 1e-3));
    CHECK_THAT(sample->angularVelocity.length(), WithinAbs(0.f, 1e-4));

    rig.stop();
}

TEST_CASE("Sensor timestamps advance monotonically with the simulation") {

    Rig rig;
    auto box = makeInstrumentedBox(rig.scene, "Body");
    rig.scene.add(box);

    rig.start();
    rig.update(2);
    const double early = rig.sensors.entries().front()->lastTime;
    rig.update(20);
    const double late = rig.sensors.entries().front()->lastTime;

    CHECK(early > 0.0);
    CHECK(late > early);
    // Real time is not consulted anywhere: 22 updates of 1/60 s is 22 substeps.
    CHECK_THAT(late, WithinAbs(22.0 / 60.0, 1e-6));

    rig.stop();
}

TEST_CASE("Rebuilding the sensor per Play replays the seed") {

    // The determinism story: sensors are play-time constructs, so the second
    // Play starts from the authored seed rather than from wherever the first
    // one's random walk had drifted to.
    const auto run = [] {
        Rig rig;
        auto box = ObjectFactory::createPrimitive(Primitive::Box, rig.scene);
        box->name = "Body";
        box->position.set(0.f, 3.f, 0.f);

        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Dynamic;
        physics.write(*box);

        SensorConfig sensor;
        sensor.enabled = true;
        sensor.type = SensorConfig::Type::Imu;
        sensor.rateHz = 0.f;
        sensor.seed = 99;// noise left at the MEMS defaults, so the seed matters
        sensor.write(*box);
        rig.scene.add(box);

        rig.start();
        rig.update(24);

        // Copy the trace out before stop() drops the entry.
        const auto& trace = rig.sensors.entries().front()->traces[3];// accel x
        std::vector<float> values(trace.values.begin(), trace.values.begin() + trace.count);
        rig.stop();
        return values;
    };

    const auto first = run();
    const auto second = run();

    REQUIRE(first.size() > 10);
    REQUIRE(first.size() == second.size());
    // Bit-exact. A seeded sensor that only matched statistically would make a
    // recorded dataset unreplayable, which is the whole point of the seed.
    for (std::size_t i = 0; i < first.size(); ++i) {
        CHECK(first[i] == second[i]);
    }
    // And the noise is actually doing something — otherwise the loop above
    // would pass on two streams of zeros.
    bool nonzero = false;
    for (const float v : first) nonzero = nonzero || v != 0.f;
    CHECK(nonzero);
}

TEST_CASE("A zero noise model is a bit-exact passthrough") {

    // Same body, same physics, noise on vs off: the noiseless run must produce
    // exactly the value the physics truth implies, and the noisy one must not.
    const auto run = [](bool noisy) {
        Rig rig;
        auto box = makeInstrumentedBox(rig.scene, "Body");
        if (noisy) {
            auto config = SensorConfig::read(*box).value();
            config.accelNoiseDensity = 0.06f;
            config.write(*box);
        }
        rig.scene.add(box);
        rig.start();
        holdStill(rig.physics, *box);
        rig.update(30);
        const auto sample = rig.sensors.entries().front()->imu->latest();
        REQUIRE(sample.has_value());
        const float y = sample->linearAcceleration.y;
        rig.stop();
        return y;
    };

    CHECK_THAT(run(false), WithinAbs(kG, 1e-4));
    CHECK(run(true) != run(false));
}

TEST_CASE("An IMU on an object with no rigid body reports why, and does not crash") {

    Rig rig;
    auto group = Group::create();
    group->name = "Bare";

    SensorConfig sensor;
    sensor.enabled = true;
    sensor.type = SensorConfig::Type::Imu;
    sensor.write(*group);
    rig.scene.add(group);

    rig.start();
    // Counted, but not live: the mistake surfaces at Play rather than as a
    // sensor quietly reading zeros forever.
    CHECK(rig.sensors.sensorCount() == 1);
    CHECK(rig.sensors.liveCount() == 0);
    CHECK_FALSE(rig.sensors.entries().front()->status.empty());

    rig.update(10);
    CHECK(rig.sensors.entries().front()->samples == 0);
    rig.stop();
}

TEST_CASE("Stop clears the sensors, and Play/Stop twice is fine") {

    Rig rig;
    auto box = makeInstrumentedBox(rig.scene, "Body");
    rig.scene.add(box);

    for (int pass = 0; pass < 2; ++pass) {
        rig.start();
        CHECK(rig.sensors.liveCount() == 1);
        rig.update(5);
        CHECK(rig.sensors.entries().front()->samples > 0);
        rig.stop();
        CHECK(rig.sensors.sensorCount() == 0);
        CHECK(rig.sensors.entries().empty());
        CHECK_THAT(rig.sensors.simTime(), WithinAbs(0.0, 1e-12));
    }

    // The sensor nodes are gone from the rig too — nothing left parented to
    // editor furniture that outlives the play.
    CHECK(rig.rig->children.empty());
}

TEST_CASE("Recording writes a CSV with rows") {

    const auto dir = std::filesystem::temp_directory_path() / "threepp-sensor-record-test";
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    Rig rig;
    auto box = makeInstrumentedBox(rig.scene, "Wheel Hub");
    rig.scene.add(box);

    rig.sensors.setRecordDirectory(dir);
    // Armed while stopped: the file opens on the first measurement of the play
    // that follows, so "Record then Play" captures from t=0.
    rig.sensors.setRecording(true);
    CHECK(rig.sensors.recording());

    rig.start();
    rig.update(20);
    CHECK(rig.sensors.recordedRows() > 0);
    rig.stop();

    REQUIRE(std::filesystem::exists(dir));
    int files = 0;
    int dataRows = 0;
    for (const auto& file : std::filesystem::directory_iterator(dir)) {
        if (file.path().extension() != ".csv") continue;
        ++files;
        std::ifstream in(file.path());
        std::string line;
        int lines = 0;
        while (std::getline(in, line)) ++lines;
        // Header plus at least one measurement.
        CHECK(lines > 1);
        dataRows += lines - 1;
        // The object's name is in the filename, with the space folded away.
        CHECK(file.path().filename().string().rfind("Wheel_Hub_", 0) == 0);
    }
    CHECK(files == 1);
    CHECK(dataRows > 0);

    std::filesystem::remove_all(dir, ec);
}

TEST_CASE("A vision sensor without a renderer is built and says so") {

    Rig rig;
    auto post = ObjectFactory::createPrimitive(Primitive::Box, rig.scene);
    post->name = "Mast";

    SensorConfig sensor;
    sensor.enabled = true;
    sensor.type = SensorConfig::Type::Lidar;
    sensor.beams = SensorConfig::Beams::VLP16;
    sensor.faceSize = 32;
    sensor.rateHz = 10.f;
    sensor.write(*post);
    rig.scene.add(post);

    rig.start();
    REQUIRE(rig.sensors.liveCount() == 1);
    const auto& entry = *rig.sensors.entries().front();
    CHECK(entry.lidar != nullptr);
    CHECK_FALSE(entry.status.empty());// "no renderer"
    // It IS in the rig, so the editor's export/snapshot filter covers it.
    CHECK(entry.lidar->parent == rig.rig.get());

    rig.update(30);
    CHECK(entry.scans == 0);// nothing to scan with
    rig.stop();
    CHECK(rig.rig->children.empty());
}

TEST_CASE("An authored contact sensor reports landing on the ground") {

    Rig rig;

    auto ground = ObjectFactory::createPrimitive(Primitive::Box, rig.scene);
    ground->name = "Ground";
    ground->scale.set(20.f, 0.4f, 20.f);
    ground->position.set(0.f, -0.2f, 0.f);
    {
        PhysicsConfig floor;
        floor.enabled = true;
        floor.body = PhysicsConfig::Body::Static;
        floor.shape = PhysicsConfig::Shape::Box;
        floor.write(*ground);
    }
    rig.scene.add(ground);

    auto foot = ObjectFactory::createPrimitive(Primitive::Box, rig.scene);
    foot->name = "Foot";
    foot->position.set(0.f, 1.f, 0.f);
    {
        PhysicsConfig physics;
        physics.enabled = true;
        physics.body = PhysicsConfig::Body::Dynamic;
        physics.shape = PhysicsConfig::Shape::Box;
        physics.mass = 1.f;
        physics.restitution = 0.f;
        physics.write(*foot);

        SensorConfig sensor;
        sensor.enabled = true;
        sensor.type = SensorConfig::Type::Contact;
        sensor.rateHz = 0.f;
        sensor.write(*foot);
    }
    rig.scene.add(foot);

    rig.start();
    REQUIRE(rig.sensors.liveCount() == 1);
    const auto& entry = *rig.sensors.entries().front();
    CHECK(entry.status.empty());

    // Not touching to begin with: it is a metre up.
    rig.update(1);
    CHECK_FALSE(entry.inContact);

    // Fall and land. Waiting on the LATCH rather than on a frame count keeps
    // this about the sensor instead of about how fast a box falls.
    bool touched = false;
    for (int i = 0; i < 300 && !touched; ++i) {
        rig.update(1);
        touched = entry.inContact;
    }
    CHECK(touched);
    CHECK(entry.samples > 0);

    // A force, not just a boolean. PhysX stops reporting a sleeping pair, so the
    // peak over the settle is the honest thing to assert - a resting contact goes
    // quiet while still physically touching.
    float peak = 0.f;
    for (int i = 0; i < 120; ++i) {
        rig.update(1);
        peak = std::max(peak, entry.contactForce);
    }
    // Magnitude only: the manifold's per-axis split is a solver detail.
    CHECK(peak > 0.f);

    rig.stop();
}

TEST_CASE("Types that need an articulation are authored, not simulated") {

    Rig rig;
    auto arm = ObjectFactory::createPrimitive(Primitive::Box, rig.scene);
    arm->name = "Joint 1";

    SensorConfig sensor;
    sensor.enabled = true;
    sensor.type = SensorConfig::Type::Encoder;
    sensor.write(*arm);
    rig.scene.add(arm);

    rig.start();
    CHECK(rig.sensors.sensorCount() == 1);
    CHECK(rig.sensors.liveCount() == 0);
    CHECK(rig.sensors.entries().front()->status.find("articulated") != std::string::npos);
    // And the authored config survived the round trip untouched.
    const auto config = SensorConfig::read(*arm);
    REQUIRE(config.has_value());
    CHECK(config->type == SensorConfig::Type::Encoder);
    rig.stop();
}
