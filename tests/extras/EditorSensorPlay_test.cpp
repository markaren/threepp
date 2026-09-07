// Sensor authoring, played.
//
// Drives the real PhysxSensorPlaySession against a real PhysicsPlaySession,
// headless (no renderer), so the assertions are about the seam the editor
// actually uses: userData in, live sensors out, measurements stamped with SIM
// time, and everything gone again after Stop.
//
// Needs the PhysX SDK, so this target only exists where the SDK was found (see
// tests/extras/CMakeLists.txt). The format half — the userData round trip and
// the always-emit-all-keys rule — is EditorSensorConfig_test, and the PhysX-free
// session half (vision sensors, the dt clock, the authored-only statuses) is
// EditorVisionPlay_test; both run everywhere.
//
// What is deliberately NOT here: the vision sensors' clouds. A scan needs a GL
// context, so that half is asserted in the editor's --selftest, which has one.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/ArticulationConfig.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"

#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
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
        PhysxSensorPlaySession sensors;

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

    // A two-link arm with one revolute joint, primitives only so the articulation
    // builder can cook it with no external mesh files. Written to a temp file
    // because loadArticulation (and the URDF re-parse) both take a path.
    const char* kArmUrdf = R"(
        <robot name="arm">
          <link name="base_link">
            <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
            <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
          </link>
          <link name="upper_link">
            <visual><geometry><box size="0.1 0.4 0.1"/></geometry></visual>
            <collision><geometry><box size="0.1 0.4 0.1"/></geometry></collision>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base_link"/>
            <child link="upper_link"/>
            <origin xyz="0 0 0.3" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-2.0" upper="2.0"/>
          </joint>
        </robot>)";

    std::filesystem::path armFixture() {
        const auto dir = std::filesystem::temp_directory_path() / "threepp-sensor-articulation";
        std::filesystem::create_directories(dir);
        const auto path = dir / "arm.urdf";
        std::ofstream(path, std::ios::trunc) << kArmUrdf;
        return path;
    }

    // A three-link, two-joint arm for the all-joints fan-out: one encoder
    // entry must become TWO live encoders, which a one-joint arm cannot show.
    const char* kTwoJointUrdf = R"(
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

    std::filesystem::path twoJointFixture() {
        const auto dir = std::filesystem::temp_directory_path() / "threepp-sensor-articulation";
        std::filesystem::create_directories(dir);
        const auto path = dir / "arm2.urdf";
        std::ofstream(path, std::ios::trunc) << kTwoJointUrdf;
        return path;
    }

    Object3D* findByName(Object3D& root, const std::string& name) {
        Object3D* found = nullptr;
        root.traverse([&](Object3D& node) {
            if (!found && node.name == name) found = &node;
        });
        return found;
    }

    // A simulated arm added to the scene, with an authored joint pose.
    std::shared_ptr<Robot> makeSimulatedArm(Scene& scene, const std::filesystem::path& path,
                                            float shoulder = 0.5f) {
        URDFLoader loader;
        auto robot = loader.load(path);
        RobotConfig rc;
        rc.urdf = path.string();
        rc.joints = std::vector<float>(robot->numDOF(), 0.f);
        if (!rc.joints.empty()) rc.joints[0] = shoulder;
        rc.write(*robot);
        for (std::size_t i = 0; i < rc.joints.size() && i < robot->numDOF(); ++i) {
            robot->setJointValue(i, rc.joints[i]);
        }
        ArticulationConfig ac;
        ac.enabled = true;
        ac.fixedBase = true;
        ac.write(*robot);
        scene.add(robot);
        return robot;
    }

    // Author a joint sensor on a link of the robot.
    void authorJointSensor(Object3D& linkNode, SensorConfig::Type type, const std::string& joint) {
        SensorConfig sensor;
        sensor.enabled = true;
        sensor.type = type;
        sensor.rateHz = 0.f;// every substep
        sensor.joint = joint;
        sensor.write(linkNode);
    }

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

// "A vision sensor without a renderer is built and says so" lives in
// EditorVisionPlay_test now — it never needed the SDK, and there it runs on
// every platform.

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

TEST_CASE("A joint sensor off an articulated robot says so, and does not crash") {

    Rig rig;
    auto arm = ObjectFactory::createPrimitive(Primitive::Box, rig.scene);
    arm->name = "Joint 1";

    SensorConfig sensor;
    sensor.enabled = true;
    sensor.type = SensorConfig::Type::Encoder;
    sensor.joint = "shoulder";
    sensor.write(*arm);
    rig.scene.add(arm);

    rig.start();
    CHECK(rig.sensors.sensorCount() == 1);
    CHECK(rig.sensors.liveCount() == 0);
    // The plain box is not an articulated robot, and the status names that.
    CHECK(rig.sensors.entries().front()->status.find("articulated robot") != std::string::npos);
    rig.update(10);
    CHECK(rig.sensors.entries().front()->samples == 0);
    // And the authored config survived the round trip untouched.
    const auto config = SensorConfig::read(*arm);
    REQUIRE(config.has_value());
    CHECK(config->type == SensorConfig::Type::Encoder);
    rig.stop();
}

TEST_CASE("A joint encoder authored on a link produces samples during play") {

    const auto path = armFixture();
    Rig rig;
    auto robot = makeSimulatedArm(rig.scene, path, /*shoulder*/ 0.5f);

    // Author the encoder on the moving link, naming the joint it measures.
    auto* link = findByName(*robot, "upper_link");
    REQUIRE(link != nullptr);
    authorJointSensor(*link, SensorConfig::Type::Encoder, "shoulder");

    rig.start();
    REQUIRE(rig.sensors.sensorCount() == 1);
    const auto& entry = *rig.sensors.entries().front();
    CHECK(entry.status.empty());
    CHECK(entry.encoder != nullptr);

    rig.update(30);
    CHECK(entry.samples > 0);
    const auto sample = entry.encoder->latest();
    REQUIRE(sample.has_value());
    // The encoder reads the shoulder held near its authored 0.5 rad by the drive.
    CHECK(std::abs(sample->position - 0.5f) < 0.2f);

    rig.stop();
}

TEST_CASE("A force/torque sensor authored on a link produces wrenches during play") {

    const auto path = armFixture();
    Rig rig;
    auto robot = makeSimulatedArm(rig.scene, path, /*shoulder*/ 0.4f);

    auto* link = findByName(*robot, "upper_link");
    REQUIRE(link != nullptr);
    authorJointSensor(*link, SensorConfig::Type::ForceTorque, "shoulder");

    rig.start();
    REQUIRE(rig.sensors.liveCount() == 1);
    const auto& entry = *rig.sensors.entries().front();
    CHECK(entry.status.empty());
    CHECK(entry.forceTorque != nullptr);

    rig.update(30);
    CHECK(entry.samples > 0);
    const auto sample = entry.forceTorque->latest();
    REQUIRE(sample.has_value());
    // The link has mass and the drive holds it against gravity, so a non-zero
    // wrench is transmitted through the joint.
    const float magnitude = sample->force.length() + sample->torque.length();
    CHECK(magnitude > 0.f);

    rig.stop();
}

TEST_CASE("An IMU and a contact sensor on robot links ride the articulation") {

    // The visual link nodes are never bound (the joint mirror drives them), so
    // this only works through the world's resolution-only associations. Three
    // sensors, one per object: a base IMU on the ROBOT node (resolves to the
    // root link), an IMU on the moving link, and a contact sensor on the base
    // link. Before the associations existed, all three failed at registration
    // with "no PhysxWorld-managed rigid body".
    const auto path = armFixture();
    Rig rig;
    auto robot = makeSimulatedArm(rig.scene, path, /*shoulder*/ 0.3f);

    SensorConfig imu;
    imu.enabled = true;
    imu.type = SensorConfig::Type::Imu;
    imu.rateHz = 0.f;
    imu.write(*robot);

    auto* upper = findByName(*robot, "upper_link");
    REQUIRE(upper != nullptr);
    imu.write(*upper);

    auto* base = findByName(*robot, "base_link");
    REQUIRE(base != nullptr);
    SensorConfig contact;
    contact.enabled = true;
    contact.type = SensorConfig::Type::Contact;
    contact.rateHz = 0.f;
    contact.write(*base);

    rig.start();
    REQUIRE(rig.sensors.sensorCount() == 3);
    CHECK(rig.sensors.liveCount() == 3);
    for (const auto& entry : rig.sensors.entries()) {
        CHECK(entry->status.empty());
    }

    rig.update(30);
    for (const auto& entry : rig.sensors.entries()) {
        CHECK(entry->samples > 0);
    }

    rig.stop();
    CHECK(rig.sensors.sensorCount() == 0);
}

TEST_CASE("An all-joints encoder fans out to one live encoder per DOF") {

    const auto path = twoJointFixture();
    Rig rig;
    auto robot = makeSimulatedArm(rig.scene, path, /*shoulder*/ 0.4f);
    REQUIRE(robot->numDOF() == 2);

    // ONE authored entry, on the robot's root — the point of the fan-out is
    // that an object carries one sensor, and this covers the whole robot.
    authorJointSensor(*robot, SensorConfig::Type::Encoder, SensorConfig::allJoints);

    rig.start();
    REQUIRE(rig.physics.articulationCount() == 1);
    CHECK(rig.sensors.sensorCount() == 2);
    CHECK(rig.sensors.liveCount() == 2);

    std::set<std::string> joints;
    for (const auto& entry : rig.sensors.entries()) {
        CHECK(entry->encoder != nullptr);
        CHECK(entry->status.empty());
        CHECK(entry->uuid == robot->uuid);
        // The label carries the joint name — it is what keeps the readout rows
        // and the per-sensor CSV files apart when they all share one node.
        CHECK(entry->label.find(entry->config.joint) != std::string::npos);
        joints.insert(entry->config.joint);
    }
    CHECK(joints == std::set<std::string>{"shoulder", "elbow"});

    rig.update(30);
    for (const auto& entry : rig.sensors.entries()) {
        CHECK(entry->samples > 0);
    }

    rig.stop();
    CHECK(rig.sensors.sensorCount() == 0);
}

TEST_CASE("All joints on a Force/Torque sensor is refused with a reason") {

    const auto path = armFixture();
    Rig rig;
    auto robot = makeSimulatedArm(rig.scene, path);
    authorJointSensor(*robot, SensorConfig::Type::ForceTorque, SensorConfig::allJoints);

    rig.start();
    CHECK(rig.sensors.sensorCount() == 1);
    CHECK(rig.sensors.liveCount() == 0);
    CHECK(rig.sensors.entries().front()->status.find("one joint") != std::string::npos);
    rig.update(10);
    CHECK(rig.sensors.entries().front()->samples == 0);
    rig.stop();
}

TEST_CASE("An unknown joint name reports a status and does not crash") {

    const auto path = armFixture();
    Rig rig;
    auto robot = makeSimulatedArm(rig.scene, path);

    auto* link = findByName(*robot, "upper_link");
    REQUIRE(link != nullptr);
    authorJointSensor(*link, SensorConfig::Type::Encoder, "no_such_joint");

    rig.start();
    CHECK(rig.sensors.sensorCount() == 1);
    CHECK(rig.sensors.liveCount() == 0);
    CHECK(rig.sensors.entries().front()->status.find("no_such_joint") != std::string::npos);
    rig.update(10);
    CHECK(rig.sensors.entries().front()->samples == 0);
    rig.stop();
}

TEST_CASE("A joint sensor survives a start/stop/start cycle (teardown order)") {

    // The regression this guards: physics stops first, so the world AND the
    // articulation are gone by the time the sensor session stops. The FT sensor's
    // cache must be released while the world is alive (through unregisterSensor),
    // or the second Stop reads freed memory.
    const auto path = armFixture();
    Rig rig;
    auto robot = makeSimulatedArm(rig.scene, path);

    auto* link = findByName(*robot, "upper_link");
    REQUIRE(link != nullptr);
    authorJointSensor(*link, SensorConfig::Type::ForceTorque, "shoulder");

    for (int pass = 0; pass < 2; ++pass) {
        rig.start();
        CHECK(rig.sensors.liveCount() == 1);
        rig.update(10);
        CHECK(rig.sensors.entries().front()->samples > 0);
        rig.stop();
        CHECK(rig.sensors.sensorCount() == 0);
    }
}
