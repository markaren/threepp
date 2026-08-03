// Authored joints, played.
//
// The model under test (see JointConfig): a joint is its own node, its
// transform is the joint frame (X = hinge/slide axis), its parent chain is
// body A and the other body is named — empty meaning the world. Each case
// steps a fixed 1/60 s clock, the same doctrine as EditorPhysicsPlay_test:
// nothing here may depend on wall time.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/JointConfig.hpp"
#include "threepp/extras/editor/ObjectFactory.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/PhysxSensorPlaySession.hpp"
#include "threepp/extras/editor/SceneDocument.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"

#include "threepp/extras/sensors/ForceTorqueSensor.hpp"
#include "threepp/extras/sensors/JointEncoder.hpp"

#include "threepp/math/MathUtils.hpp"
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

    std::shared_ptr<Mesh> makeDynamicBox(Scene& scene, const char* name,
                                         const Vector3& position) {

        auto box = ObjectFactory::createPrimitive(Primitive::Box, scene);
        box->name = name;
        box->position.copy(position);

        PhysicsConfig config;
        config.enabled = true;
        config.body = PhysicsConfig::Body::Dynamic;
        config.shape = PhysicsConfig::Shape::Box;
        config.write(*box);
        return box;
    }

    // A joint node under `bodyA`, at `localPosition`, hinging about the WORLD
    // Z axis: rotating local X onto world Z is what makes a pendulum swing in
    // the XY plane, where the assertions can read it.
    std::shared_ptr<Object3D> makeJointNode(Object3D& bodyA, const Vector3& localPosition,
                                            const JointConfig& config) {

        auto node = Object3D::create();
        node->name = "Joint";
        node->position.copy(localPosition);
        node->rotation.y = -math::PI / 2;// local X -> world Z
        config.write(*node);
        bodyA.add(node);
        return node;
    }

    void run(PhysicsPlaySession& session, int steps) {

        for (int i = 0; i < steps; ++i) session.update(1.f / 60.f);
    }

}// namespace


TEST_CASE("a fixed joint welds two falling boxes together", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto a = makeDynamicBox(scene, "A", {0.f, 6.f, 0.f});
    auto b = makeDynamicBox(scene, "B", {1.5f, 6.f, 0.f});
    scene.add(a);
    scene.add(b);

    JointConfig config;
    config.type = JointConfig::Type::Fixed;
    config.body = "B";
    auto joint = Object3D::create();
    joint->name = "Weld";
    joint->position.set(0.75f, 0.f, 0.f);// midway between the two
    config.write(*joint);
    a->add(joint);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.jointCount() == 1);

    run(session, 60);// 1 second of free fall

    // Both fell...
    CHECK(a->position.y < 5.f);
    // ...and the weld held: the offset between them is still what was authored.
    CHECK_THAT(b->position.x - a->position.x, WithinAbs(1.5f, 0.05f));
    CHECK_THAT(b->position.y - a->position.y, WithinAbs(0.f, 0.05f));
    CHECK_THAT(b->position.z - a->position.z, WithinAbs(0.f, 0.05f));

    session.stop();
    CHECK(session.jointCount() == 0);
}

TEST_CASE("a revolute joint swings a pendulum about a world anchor", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto bob = makeDynamicBox(scene, "Bob", {1.f, 4.f, 0.f});
    scene.add(bob);

    // Anchor at world (0,4,0): the joint node sits 1 m towards the pivot from
    // the bob. An empty body name means the other side is the world itself.
    JointConfig config;
    config.type = JointConfig::Type::Revolute;
    makeJointNode(*bob, {-1.f, 0.f, 0.f}, config);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.jointCount() == 1);

    // Sample every step: the pendulum keeps swinging, so any single instant
    // may catch it back near the top of its arc — the bottom of the swing is
    // what proves it swung.
    float lowestY = bob->position.y;
    for (int i = 0; i < 90; ++i) {// 1.5 s: well past the bottom of the first swing
        session.update(1.f / 60.f);
        lowestY = std::min(lowestY, bob->position.y);
    }

    // It swung down through (near) the bottom, 1 m below the anchor...
    CHECK(lowestY < 3.2f);
    // ...it stayed on the 1 m arm (the constraint held)...
    const Vector3 anchor(0.f, 4.f, 0.f);
    CHECK_THAT(bob->position.distanceTo(anchor), WithinAbs(1.f, 0.15f));
    // ...and it swung IN PLANE: the hinge axis (world Z) let nothing drift
    // sideways, which is what separates a hinge from a ball socket here.
    CHECK_THAT(bob->position.z, WithinAbs(0.f, 0.05f));

    session.stop();
}

TEST_CASE("revolute limits clamp the swing", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto bob = makeDynamicBox(scene, "Bob", {1.f, 4.f, 0.f});
    scene.add(bob);

    JointConfig config;
    config.type = JointConfig::Type::Revolute;
    config.limited = true;
    config.lower = -math::degToRad(30.f);
    config.upper = math::degToRad(30.f);
    makeJointNode(*bob, {-1.f, 0.f, 0.f}, config);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.jointCount() == 1);

    // Sample every step: the deepest point of the swing is transient, and an
    // unlimited pendulum passes 30 deg on its way to the bottom rather than
    // resting there.
    float lowestY = bob->position.y;
    for (int i = 0; i < 180; ++i) {// 3 seconds
        session.update(1.f / 60.f);
        lowestY = std::min(lowestY, bob->position.y);
    }

    // 30 deg below horizontal on a 1 m arm is y = 4 - sin(30 deg) = 3.5. An
    // unlimited pendulum reaches 3.0 at the bottom, so 3.35 (a little slop for
    // the limit's spring) cleanly separates the two.
    CHECK(lowestY > 3.35f);

    session.stop();
}

TEST_CASE("a position drive holds a pendulum at its target", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto bob = makeDynamicBox(scene, "Bob", {1.f, 4.f, 0.f});
    scene.add(bob);

    // Target 0 is the authored (horizontal) angle — the pose gravity is most
    // eager to pull away from. The target acts through STIFFNESS: this is the
    // gate the inspector's Driven checkbox seeds.
    JointConfig config;
    config.type = JointConfig::Type::Revolute;
    config.stiffness = 2000.f;
    config.damping = 100.f;
    config.target = 0.f;
    makeJointNode(*bob, {-1.f, 0.f, 0.f}, config);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.jointCount() == 1);

    float lowestY = bob->position.y;
    for (int i = 0; i < 120; ++i) {// 2 s
        session.update(1.f / 60.f);
        lowestY = std::min(lowestY, bob->position.y);
    }

    // Gravity torque on the horizontal arm is ~10 Nm; at 2000 Nm/rad the sag
    // is a fraction of a degree. A passive pendulum reaches y = 3.0 — a drive
    // that holds above 3.9 is unmistakably acting.
    CHECK(lowestY > 3.9f);

    session.stop();
}

TEST_CASE("a velocity drive swings a pendulum over the top", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto bob = makeDynamicBox(scene, "Bob", {1.f, 4.f, 0.f});
    scene.add(bob);

    // The velocity acts through DAMPING (stiffness stays 0): a steady 2 rad/s
    // spin, strong enough to carry the bob over the top.
    JointConfig config;
    config.type = JointConfig::Type::Revolute;
    config.damping = 100.f;
    config.velocity = 2.f;
    makeJointNode(*bob, {-1.f, 0.f, 0.f}, config);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.jointCount() == 1);

    float highestY = bob->position.y;
    for (int i = 0; i < 180; ++i) {// 3 s: over a full turn at ~2 rad/s
        session.update(1.f / 60.f);
        highestY = std::max(highestY, bob->position.y);
    }

    // The energy argument: a PASSIVE pendulum released at rest at y = 4 can
    // never rise above y = 4. Climbing well past it proves the drive fed
    // energy in — whichever direction the axis convention spins it.
    CHECK(highestY > 4.3f);
    // And it stayed on the arm while doing so.
    const Vector3 anchor(0.f, 4.f, 0.f);
    CHECK_THAT(bob->position.distanceTo(anchor), WithinAbs(1.f, 0.15f));

    session.stop();
}

TEST_CASE("a distance joint tethers a falling box", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto crate = makeDynamicBox(scene, "Crate", {0.f, 3.f, 0.f});
    scene.add(crate);

    // Anchor at the crate's spawn point; a 1.5 m tether to the world.
    JointConfig config;
    config.type = JointConfig::Type::Distance;
    config.lower = 0.f;
    config.upper = 1.5f;
    auto joint = Object3D::create();
    joint->name = "Tether";
    config.write(*joint);
    crate->add(joint);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.jointCount() == 1);

    run(session, 180);// 3 s: free fall would be ~44 m down by now

    // It fell to the end of the tether and hangs there.
    const Vector3 anchor(0.f, 3.f, 0.f);
    CHECK(crate->position.y < 2.5f);
    CHECK(crate->position.distanceTo(anchor) < 1.7f);

    session.stop();
}

TEST_CASE("a joint encoder reads a plain hinge", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto bob = makeDynamicBox(scene, "Bob", {1.f, 4.f, 0.f});
    scene.add(bob);

    JointConfig config;
    config.type = JointConfig::Type::Revolute;
    auto joint = makeJointNode(*bob, {-1.f, 0.f, 0.f}, config);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.jointCount() == 1);
    const auto* played = session.findJoint(joint.get());
    REQUIRE(played);
    REQUIRE(played->joint);

    JointEncoder encoder(*joint, *played->joint);
    session.world()->registerSensor(&encoder);

    run(session, 60);// 1 s: mid-swing, a decidedly nonzero angle

    const auto sample = encoder.latest();
    REQUIRE(sample.has_value());
    CHECK(std::abs(sample->position) > 0.2f);
    // The encoder agrees with the geometry: the bob sits at
    // anchor + R(angle)·(1 m, 0, 0), so the angle is recoverable from where
    // the bob actually is. Magnitudes, to stay out of sign conventions.
    const float geometric = std::atan2(4.f - bob->position.y, bob->position.x);
    CHECK_THAT(std::abs(sample->position), WithinAbs(std::abs(geometric), 0.1f));

    session.world()->unregisterSensor(&encoder);
    session.stop();
}

TEST_CASE("a force/torque sensor reads a plain joint's reaction", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    // A 1 kg crate WELDED to the world with nothing under it: the weld
    // carries its whole weight, so the load cell across it must read m*g.
    auto crate = makeDynamicBox(scene, "Crate", {0.f, 3.f, 0.f});
    scene.add(crate);

    JointConfig config;
    config.type = JointConfig::Type::Fixed;
    auto joint = Object3D::create();
    joint->name = "Weld";
    config.write(*joint);
    crate->add(joint);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.jointCount() == 1);
    const auto* played = session.findJoint(joint.get());
    REQUIRE(played);
    REQUIRE(played->joint);

    ForceTorqueSensor ft(*joint, *played->joint);
    session.world()->registerSensor(&ft);

    run(session, 120);// settle

    const auto sample = ft.latest();
    REQUIRE(sample.has_value());
    CHECK_THAT(sample->force.length(), WithinAbs(9.81f, 1.f));

    session.world()->unregisterSensor(&ft);
    session.stop();
}

TEST_CASE("a breakable joint reports its break", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    // A weld armed to fail: 1 N against a 1 kg crate's ~10 N of weight, so
    // gravity alone snaps it within the first steps.
    auto crate = makeDynamicBox(scene, "Crate", {0.f, 3.f, 0.f});
    scene.add(crate);

    JointConfig config;
    config.type = JointConfig::Type::Fixed;
    config.breakForce = 1.f;
    config.breakTorque = 1.f;
    auto joint = Object3D::create();
    joint->name = "Fuse";
    config.write(*joint);
    crate->add(joint);

    PhysicsPlaySession session;
    std::vector<std::string> logged;
    session.setLogger([&](const std::string& line) { logged.push_back(line); });
    session.start(scene);
    REQUIRE(session.jointCount() == 1);

    run(session, 30);// 0.5 s: far more than the break needs

    // The constraint is gone...
    const auto* played = session.findJoint(joint.get());
    REQUIRE(played);
    CHECK(played->joint->broken());
    // ...the crate is in free fall...
    CHECK(crate->position.y < 2.5f);
    // ...the break was queued once, against the authored node...
    std::vector<Object3D*> broken;
    session.drainBrokenJoints(broken);
    REQUIRE(broken.size() == 1);
    CHECK(broken.front() == joint.get());
    // ...draining drained it...
    broken.clear();
    session.drainBrokenJoints(broken);
    CHECK(broken.empty());
    // ...and the console heard, listener or not.
    const bool reported = std::any_of(logged.begin(), logged.end(), [](const std::string& line) {
        return line.find("Fuse") != std::string::npos && line.find("broke") != std::string::npos;
    });
    CHECK(reported);

    session.stop();
}

TEST_CASE("a sensor authored on a joint node reads the joint", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto bob = makeDynamicBox(scene, "Bob", {1.f, 4.f, 0.f});
    scene.add(bob);

    JointConfig jointConfig;
    jointConfig.type = JointConfig::Type::Revolute;
    auto joint = makeJointNode(*bob, {-1.f, 0.f, 0.f}, jointConfig);

    // The encoder rides the joint node itself — no joint name needed. The
    // all-joints sentinel is planted on purpose: a leftover from a previous
    // home on a robot, and it must be ignored here rather than fanned out.
    SensorConfig sensorConfig;
    sensorConfig.enabled = true;
    sensorConfig.type = SensorConfig::Type::Encoder;
    sensorConfig.rateHz = 0.f;
    sensorConfig.joint = SensorConfig::allJoints;
    sensorConfig.write(*joint);

    PhysicsPlaySession physics;
    PhysxSensorPlaySession sensors;
    sensors.setPhysics(&physics);

    physics.start(scene);
    sensors.start(scene);

    for (int i = 0; i < 60; ++i) {
        physics.update(1.f / 60.f);
        sensors.update(1.f / 60.f);
    }

    REQUIRE(sensors.entries().size() == 1);
    const auto& entry = *sensors.entries().front();
    INFO("status: " << entry.status);
    CHECK(entry.status.empty());
    REQUIRE(entry.encoder != nullptr);
    CHECK(entry.samples > 0);

    // The controller's stop order: physics first, sensors after, with the
    // world already gone — the seam where a stale world pointer would crash.
    physics.stop();
    sensors.stop();
}

TEST_CASE("an impact break records the failure spike, not the last quiet sample",
          "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    // The Timber Yard shape: a weld carrying a settled ~10 N standing load,
    // rated for 50 N, and a crate dropped on it to spike past the rating. The
    // interesting number — the load at the instant of failure — is far above
    // both the threshold and anything the pre-break stream saw, which is
    // exactly the gap this exists to close.
    auto crate = makeDynamicBox(scene, "Crate", {0.f, 3.f, 0.f});
    scene.add(crate);
    auto hammer = makeDynamicBox(scene, "Hammer", {0.f, 6.f, 0.f});
    scene.add(hammer);

    JointConfig config;
    config.type = JointConfig::Type::Fixed;
    config.breakForce = 50.f;
    config.breakTorque = 1e9f;
    auto joint = Object3D::create();
    joint->name = "Fuse";
    config.write(*joint);
    crate->add(joint);

    PhysicsPlaySession session;
    std::vector<std::string> logged;
    session.setLogger([&](const std::string& line) { logged.push_back(line); });
    session.start(scene);
    REQUIRE(session.jointCount() == 1);
    const auto* played = session.findJoint(joint.get());
    REQUIRE(played);
    REQUIRE(played->joint);

    ForceTorqueSensor ft(*joint, *played->joint);
    session.world()->registerSensor(&ft);

    for (int i = 0; i < 300 && !played->joint->broken(); ++i) session.update(1.f / 60.f);
    REQUIRE(played->joint->broken());
    run(session, 6);// a few steps past the break, so the stream shows the zeros

    // The session's break watch latched the breaking step's wrench onto the
    // joint: the failure load, past the threshold that armed the break and
    // far above the standing load.
    Vector3 breakForce, breakTorque;
    played->joint->breakWrench(breakForce, breakTorque);
    INFO("break |F|=" << breakForce.length() << " N");
    CHECK(breakForce.length() >= config.breakForce);

    // The sensor stream tells the whole story in order: quiet standing load,
    // ONE sample carrying the failure spike, zeros from there on.
    std::vector<WrenchSample> samples;
    ft.drain(samples);
    REQUIRE(samples.size() >= 8);
    std::size_t spikes = 0;
    std::size_t spikeAt = 0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples[i].force.length() >= config.breakForce) {
            ++spikes;
            spikeAt = i;
        }
    }
    REQUIRE(spikes == 1);
    INFO("spike |F|=" << samples[spikeAt].force.length() << " N at sample " << spikeAt);
    // The spike IS the latched failure load...
    CHECK_THAT(samples[spikeAt].force.length(), WithinAbs(breakForce.length(), 1e-3f));
    // ...preceded by the settled standing load (the crate's ~10 N weight)...
    REQUIRE(spikeAt > 0);
    CHECK_THAT(samples[spikeAt - 1].force.length(), WithinAbs(9.81f, 2.f));
    // ...and followed by silence: a broken joint transmits nothing.
    for (std::size_t i = spikeAt + 1; i < samples.size(); ++i) {
        INFO("sample " << i);
        REQUIRE_THAT(samples[i].force.length(), WithinAbs(0.f, 1e-4f));
    }

    // reaction_force (what scripts poll) reads zero now; the console line
    // named the measured load.
    Vector3 liveForce, liveTorque;
    played->joint->reactionForce(liveForce, liveTorque);
    CHECK_THAT(liveForce.length(), WithinAbs(0.f, 1e-4f));
    const bool reported = std::any_of(logged.begin(), logged.end(), [](const std::string& line) {
        return line.find("Fuse") != std::string::npos && line.find("broke at") != std::string::npos &&
               line.find(" N") != std::string::npos;
    });
    CHECK(reported);

    session.world()->unregisterSensor(&ft);
    session.stop();
}

TEST_CASE("a joint naming a missing body is skipped, play intact", "[editor][physx]") {

    SceneDocument document;
    auto& scene = document.scene();

    auto box = makeDynamicBox(scene, "Loner", {0.f, 4.f, 0.f});
    scene.add(box);

    JointConfig config;
    config.type = JointConfig::Type::Fixed;
    config.body = "Nowhere";
    auto joint = Object3D::create();
    config.write(*joint);
    box->add(joint);

    PhysicsPlaySession session;
    std::vector<std::string> logged;
    session.setLogger([&](const std::string& line) { logged.push_back(line); });
    session.start(scene);

    // No joint, one line saying why, and the body still plays.
    CHECK(session.jointCount() == 0);
    CHECK(session.bodyCount() == 1);
    REQUIRE_FALSE(logged.empty());
    CHECK(logged.front().find("Nowhere") != std::string::npos);

    run(session, 60);
    CHECK(box->position.y < 3.9f);// it fell — the broken reference cost nothing

    session.stop();
}
