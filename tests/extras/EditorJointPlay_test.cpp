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
#include "threepp/extras/editor/ArticulationConfig.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/SensorConfig.hpp"

#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/objects/Robot.hpp"

#include "threepp/extras/sensors/ForceTorqueSensor.hpp"
#include "threepp/extras/sensors/JointEncoder.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <algorithm>
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

    // A two-link robot with one prismatic DOF, for the articulation-link case.
    // Deliberately an ARTICULATION and not two rigid bodies: a link is exactly
    // what the session's own actor registry does not contain.
    constexpr const char* kLiftUrdf = R"(<?xml version="1.0"?>
        <robot name="lift">
          <link name="base">
            <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
            <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
          </link>
          <link name="tool">
            <visual><geometry><box size="0.15 0.15 0.15"/></geometry></visual>
            <collision><geometry><box size="0.15 0.15 0.15"/></geometry></collision>
          </link>
          <joint name="lift" type="prismatic">
            <parent link="base"/>
            <child link="tool"/>
            <origin xyz="0 0 0.4" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="0.0" upper="0.6"/>
          </joint>
        </robot>)";

    std::shared_ptr<Robot> makeLiftRobot(Scene& scene) {

        const auto dir = std::filesystem::temp_directory_path() / "threepp-runtime-joint-test";
        std::filesystem::create_directories(dir);
        const auto path = dir / "lift.urdf";
        std::ofstream(path, std::ios::trunc) << kLiftUrdf;

        URDFLoader loader;
        auto robot = loader.load(path);
        if (!robot) return nullptr;
        robot->name = "Lift";
        // The part hangs at y=3; put the tool link up there to weld to.
        robot->position.set(0.f, 2.6f, 0.f);

        RobotConfig rc;
        rc.urdf = path.string();
        rc.write(*robot);

        ArticulationConfig ac;
        ac.enabled = true;
        ac.fixedBase = true;
        ac.write(*robot);
        return robot;
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

TEST_CASE("a joint built at runtime welds, then lets go", "[editor][physx]") {

    // The C++ half of what a script does through editor.create_joint: no
    // authored node, no JointConfig, no document entry - the session is asked
    // for a constraint mid-play and asked to drop it again.
    SceneDocument document;
    auto& scene = document.scene();

    auto a = makeDynamicBox(scene, "A", {0.f, 6.f, 0.f});
    auto b = makeDynamicBox(scene, "B", {1.5f, 6.f, 0.f});
    scene.add(a);
    scene.add(b);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.jointCount() == 0);// nothing authored

    auto* actorA = session.resolveJointBody(a.get());
    auto* actorB = session.resolveJointBody(b.get());
    REQUIRE(actorA != nullptr);
    REQUIRE(actorB != nullptr);

    auto weld = session.createJoint(actorA, actorB, {0.75f, 6.f, 0.f}, Quaternion(),
                                    Joint::Params{});
    REQUIRE(weld != nullptr);
    CHECK(session.jointCount() == 1);

    run(session, 60);
    // Held: the pair fell together, keeping the offset they were welded at.
    CHECK(a->position.y < 5.f);
    CHECK_THAT(b->position.x - a->position.x, WithinAbs(1.5f, 0.05f));
    CHECK_THAT(b->position.y - a->position.y, WithinAbs(0.f, 0.05f));

    // Let go. The session drops its reference, and since it held the only
    // owning one the constraint dies here - while the world is alive, which is
    // the whole reason the session owns it.
    CHECK(session.destroyJoint(weld.get()));
    CHECK(session.jointCount() == 0);
    CHECK_FALSE(session.destroyJoint(weld.get()));// idempotent

    // Released means RELEASED, not "will be released once the last reference
    // goes". This test still holds one, and the constraint is gone anyway.
    CHECK(weld.use_count() == 1);
    CHECK_FALSE(weld->attached());
    CHECK_THROWS(weld->broken());

    session.stop();
    // And the reference outliving stop() is inert rather than fatal - see the
    // ownership case below, which is where that is asserted properly.
}

TEST_CASE("a runtime joint reaches an articulation link", "[editor][physx]") {

    // The failure issue 408 is about. A robot's LINKS are not in the session's
    // own actor registry - they belong to the articulation - so the lookup a
    // rigid-body handle uses answers None for them. resolveJointBody is the
    // rule that does reach them, and this is the assertion that it does.
    SceneDocument document;
    auto& scene = document.scene();

    auto robot = makeLiftRobot(scene);
    REQUIRE(robot != nullptr);
    scene.add(robot);

    auto part = makeDynamicBox(scene, "Part", {0.f, 3.f, 0.f});
    scene.add(part);

    PhysicsPlaySession session;
    session.start(scene);
    REQUIRE(session.articulationCount() == 1);

    auto* link = robot->getObjectByName("tool");
    REQUIRE(link != nullptr);

    // The contrast, stated as an assertion rather than a comment: the
    // rigid-body route stops at the session's registry and finds nothing for a
    // link, while the joint route resolves it through the world's
    // associations.
    CHECK(session.findActor(link) == nullptr);
    auto* linkActor = session.resolveJointBody(link);
    REQUIRE(linkActor != nullptr);

    auto* partActor = session.resolveJointBody(part.get());
    REQUIRE(partActor != nullptr);

    Vector3 anchor;
    part->getWorldPosition(anchor);
    auto grip = session.createJoint(partActor, linkActor, anchor, Quaternion(),
                                    Joint::Params{});
    REQUIRE(grip != nullptr);

    // Welded to a link of a fixed-base robot, the part stops falling.
    run(session, 90);
    CHECK(part->position.y > 2.f);

    session.stop();
    CHECK(session.jointCount() == 0);
}

TEST_CASE("a runtime joint dies with the session that owns it", "[editor][physx]") {

    // The crash this ownership exists to prevent. ~Joint calls release() on
    // its PxJoint, which is only legal while the PxPhysics that made it lives.
    // A joint owned by a script would be collected whenever Python got round
    // to it - after Stop, that is a use-after-free. Owned here, stop() takes
    // it down BEFORE the world, and a reference outliving the session finds an
    // already-dead joint rather than a live pointer into freed memory.
    SceneDocument document;
    auto& scene = document.scene();

    auto a = makeDynamicBox(scene, "A", {0.f, 6.f, 0.f});
    scene.add(a);

    std::weak_ptr<Joint> survivor;
    {
        PhysicsPlaySession session;
        session.start(scene);

        auto* actorA = session.resolveJointBody(a.get());
        REQUIRE(actorA != nullptr);
        // One side the world: a pendulum pinned to nothing.
        auto joint = session.createJoint(actorA, nullptr, {0.f, 6.f, 0.f}, Quaternion(),
                                         Joint::Params{});
        REQUIRE(joint != nullptr);
        survivor = joint;
        CHECK_FALSE(survivor.expired());

        run(session, 30);
        session.stop();

        // stop() dropped the session's reference, but THIS scope still holds
        // one - exactly the shape of a script that stashed its grasp handle
        // somewhere that outlives the Play. The object is still alive...
        CHECK(session.jointCount() == 0);
        CHECK_FALSE(survivor.expired());
        // ...and defused: stop() released the constraint while the world was
        // still up, so the destructor that runs at the end of this scope - by
        // which time the world is gone - has nothing left to release. Without
        // this, that destructor is a use-after-free on a freed PxPhysics, and
        // it fires whenever the last reference happens to die.
        CHECK_FALSE(joint->attached());
        // broken() rather than position(): a FIXED joint's position is 0 by
        // definition and never reaches PhysX, so it would answer happily and
        // prove nothing about whether the constraint is still there.
        CHECK_THROWS(joint->broken());
    }
    // Scope over: the joint went without touching the released world.
    CHECK(survivor.expired());
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

TEST_CASE("a runtime joint's break is watched and reported", "[editor][physx]") {

    // Review coverage for two claims d8818040 makes and no other test runs.
    //
    // First: the break watch exists even when the scene AUTHORED no joints.
    // It used to be gated on "any authored joints at start()?", and under that
    // gate a breakable runtime joint snaps in silence - broken() still flips
    // (it reads the eBROKEN flag straight off PhysX) but nothing latches the
    // failure wrench and nothing reaches the console. The log assertion below
    // is the one that bites if the gate ever comes back.
    //
    // Second: the watch's callback has a node to name for an authored joint
    // and NOTHING for a runtime one (node == nullptr). It must log without
    // dereferencing, and must NOT queue a null node for drainBrokenJoints -
    // that queue feeds on_joint_break dispatch, which looks scripts up BY
    // node.
    SceneDocument document;
    auto& scene = document.scene();

    auto crate = makeDynamicBox(scene, "Crate", {0.f, 3.f, 0.f});
    scene.add(crate);

    PhysicsPlaySession session;
    std::vector<std::string> logged;
    session.setLogger([&](const std::string& line) { logged.push_back(line); });
    session.start(scene);
    REQUIRE(session.jointCount() == 0);// nothing authored, so no watch yet

    auto* actor = session.resolveJointBody(crate.get());
    REQUIRE(actor != nullptr);

    // A weld to the world armed to fail: 1 N against ~10 N of crate weight.
    Joint::Params params;
    params.breakForce = 1.f;
    params.breakTorque = 1.f;
    auto fuse = session.createJoint(actor, nullptr, {0.f, 3.f, 0.f}, Quaternion(), params);
    REQUIRE(fuse != nullptr);

    run(session, 30);

    // The break registered on the handle a script would be holding...
    CHECK(fuse->broken());
    // ...the crate is in free fall...
    CHECK(crate->position.y < 2.5f);
    // ...the console heard, with no node to name...
    const bool reported = std::any_of(logged.begin(), logged.end(), [](const std::string& line) {
        return line.find("runtime joint broke") != std::string::npos;
    });
    CHECK(reported);
    // ...and nothing was queued for a script dispatch that resolves by node.
    std::vector<Object3D*> broken;
    session.drainBrokenJoints(broken);
    CHECK(broken.empty());

    session.stop();
}

TEST_CASE("a session destroyed mid-play takes its runtime joints safely",
          "[editor][physx]") {

    // The editor closing during Play destroys the session WITHOUT a stop() -
    // ~PhysicsPlaySession only clears active(). Runtime joints then die by
    // MEMBER destruction, and the only thing keeping that safe is that
    // joints_ is declared after world_, so it destructs first. This test is
    // what turns that declaration-order coincidence into a contract: reorder
    // the members and it crashes here instead of in a closing editor.
    SceneDocument document;
    auto& scene = document.scene();

    auto crate = makeDynamicBox(scene, "Crate", {0.f, 3.f, 0.f});
    scene.add(crate);

    std::weak_ptr<Joint> survivor;
    {
        PhysicsPlaySession session;
        session.start(scene);

        auto* actor = session.resolveJointBody(crate.get());
        REQUIRE(actor != nullptr);
        auto joint = session.createJoint(actor, nullptr, {0.f, 3.f, 0.f}, Quaternion(),
                                         Joint::Params{});
        survivor = joint;
        run(session, 10);
        // No stop(). The session goes down with the joint still live.
    }
    // ~joints_ ran before ~world_, so the release found a live PxPhysics, and
    // nothing outlived anything it depended on.
    CHECK(survivor.expired());
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
