// URDF robots simulated during Play.
//
// Two things this pins that nothing else does:
//
//   frame consistency  the visual Robot from URDFLoader::load and the PhysX
//                       articulation from loadArticulation must agree on where a
//                       link is. They share URDFLoader's frame handling, so no
//                       Z-up->Y-up correction is applied - this test is what would
//                       catch a silent double-rotation if that ever changed.
//   the play seam       PhysicsPlaySession builds an articulation from a robot's
//                       ArticulationConfig, holds the authored pose with the PD
//                       drive, mirrors the solved joints back onto the visual
//                       robot, skips a robot's links in the rigid-body scan, and
//                       refuses a scaled robot.
//
// Needs the PhysX SDK, so this target only exists where the SDK was found.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/editor/ArticulationConfig.hpp"
#include "threepp/extras/editor/PhysicsConfig.hpp"
#include "threepp/extras/editor/PhysicsPlaySession.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"

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

    constexpr float kFrame = 1.f / 60.f;

    // A three-link arm with one revolute and one prismatic joint, primitives only
    // (box + cylinder collision), so the articulation builder can cook it and no
    // external mesh file is needed. The revolute link starts horizontal (+Y at 0.3
    // above the base), so gravity would swing it down without a drive holding it.
    const char* kUrdf = R"(
        <robot name="arm">
          <link name="base_link">
            <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
            <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
          </link>
          <link name="upper_link">
            <visual><geometry><box size="0.1 0.4 0.1"/></geometry></visual>
            <collision><geometry><box size="0.1 0.4 0.1"/></geometry></collision>
          </link>
          <link name="slider_link">
            <visual><geometry><box size="0.05 0.2 0.05"/></geometry></visual>
            <collision><geometry><box size="0.05 0.2 0.05"/></geometry></collision>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base_link"/>
            <child link="upper_link"/>
            <origin xyz="0 0 0.3" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-2.0" upper="2.0"/>
          </joint>
          <joint name="extend" type="prismatic">
            <parent link="upper_link"/>
            <child link="slider_link"/>
            <origin xyz="0 0.4 0" rpy="0 0 0"/>
            <axis xyz="0 1 0"/>
            <limit lower="0.0" upper="0.5"/>
          </joint>
        </robot>)";

    // loadArticulation and the URDF re-parse both take a path, so the fixture has
    // to be on disk. One temp file, reused by every test.
    std::filesystem::path writeFixture() {
        const auto dir = std::filesystem::temp_directory_path() / "threepp-articulation-test";
        std::filesystem::create_directories(dir);
        const auto path = dir / "arm.urdf";
        std::ofstream(path, std::ios::trunc) << kUrdf;
        return path;
    }

    std::shared_ptr<Robot> loadArm(const std::filesystem::path& path) {
        URDFLoader loader;
        return loader.load(path);
    }

    Object3D* findByName(Object3D& root, const std::string& name) {
        Object3D* found = nullptr;
        root.traverse([&](Object3D& node) {
            if (!found && node.name == name) found = &node;
        });
        return found;
    }

    // Author a robot into a scene: URDF reference + a joint pose + a Simulate opt-in.
    std::shared_ptr<Robot> authorRobot(Scene& scene, const std::filesystem::path& path,
                                       const std::vector<float>& pose, bool fixedBase = true) {
        auto robot = loadArm(path);
        RobotConfig rc;
        rc.urdf = path.string();
        rc.joints = pose;
        rc.write(*robot);
        for (std::size_t i = 0; i < pose.size() && i < robot->numDOF(); ++i) {
            robot->setJointValue(i, pose[i]);
        }
        ArticulationConfig ac;
        ac.enabled = true;
        ac.fixedBase = fixedBase;
        ac.write(*robot);
        scene.add(robot);
        return robot;
    }

}// namespace


TEST_CASE("The articulation and the visual robot agree on where a link is") {

    const auto path = writeFixture();

    // Build the visual robot at a non-trivial pose and read a distal link's world
    // position; then build the articulation at the same pose and read the bound
    // link's position. No Z-up->Y-up correction is applied to either, so they must
    // match - a double-rotation would put them metres apart.
    const std::vector<float> pose{0.6f, 0.2f};// revolute 0.6 rad, prismatic 0.2 m

    auto visual = loadArm(path);
    REQUIRE(visual);
    for (std::size_t i = 0; i < pose.size() && i < visual->numDOF(); ++i) {
        visual->setJointValue(i, pose[i]);
    }
    visual->updateMatrixWorld(true);
    auto* visualLink = findByName(*visual, "slider_link");
    REQUIRE(visualLink != nullptr);
    Vector3 visualPos;
    visualLink->getWorldPosition(visualPos);

    PhysxWorld::Settings settings;
    settings.fixedTimestep = kFrame;
    settings.maxSubSteps = 1;
    PhysxWorld world(settings);

    URDFArticulationOptions opts;
    opts.fixedBase = true;
    opts.renderVisuals = false;
    auto built = loadArticulation(world, path, opts);
    REQUIRE(built.articulation);
    REQUIRE(built.jointNames.size() == 2);
    REQUIRE(built.links.size() == 2);

    // Pose the articulation through the SAME name -> value mapping the play
    // session uses (add order need not equal the visual joint order).
    std::vector<float> dofPose(built.jointNames.size(), 0.f);
    for (std::size_t d = 0; d < built.jointNames.size(); ++d) {
        if (built.jointNames[d] == "shoulder") dofPose[d] = pose[0];
        else if (built.jointNames[d] == "extend") dofPose[d] = pose[1];
    }
    built.articulation->setJointPositions(dofPose.data(), dofPose.size());
    // One step so the bound-link poses reflect the applied cache.
    world.step(kFrame);

    // The slider link is the child of the "extend" joint; links are index-aligned
    // with jointNames, so find that index and read the link's world position.
    std::size_t extendIdx = built.jointNames.size();
    for (std::size_t d = 0; d < built.jointNames.size(); ++d) {
        if (built.jointNames[d] == "extend") extendIdx = d;
    }
    REQUIRE(extendIdx < built.links.size());
    const Vector3 artPos = built.links[extendIdx].position();

    CHECK_THAT(artPos.x, WithinAbs(visualPos.x, 2e-2f));
    CHECK_THAT(artPos.y, WithinAbs(visualPos.y, 2e-2f));
    CHECK_THAT(artPos.z, WithinAbs(visualPos.z, 2e-2f));
}

TEST_CASE("A fixed-base robot holds its authored pose under the drive") {

    const auto path = writeFixture();
    Scene scene;

    // The shoulder is authored well off zero; gravity would pull the horizontal
    // arm down if the drive were not holding it.
    const std::vector<float> pose{0.8f, 0.1f};
    auto robot = authorRobot(scene, path, pose, /*fixedBase*/ true);

    PhysicsPlaySession physics;
    physics.start(scene);
    REQUIRE(physics.articulationCount() == 1);

    for (int i = 0; i < 120; ++i) physics.update(kFrame);

    // The visual robot's joints track the simulated ones, and the PD held the
    // authored pose against gravity (a wider tolerance than the sensor tests, since
    // a stiff-but-finite drive settles a little off target).
    CHECK_THAT(robot->getJointValue(0), WithinAbs(0.8f, 0.15f));
    // And it did not just fall to a limit.
    CHECK(std::abs(robot->getJointValue(0)) > 0.4f);

    physics.stop();
    CHECK(physics.articulationCount() == 0);
}

TEST_CASE("The visual robot's joints track the articulation") {

    const auto path = writeFixture();
    Scene scene;

    // Author both joints off zero; the drive holds them there and the visual
    // robot must mirror whatever the articulation actually settled at, by NAME
    // (the DOF add-order need not equal the visual joint order).
    auto robot = authorRobot(scene, path, {0.7f, 0.3f}, /*fixedBase*/ true);

    PhysicsPlaySession physics;
    physics.start(scene);
    const auto* played = physics.findArticulation(robot.get());
    REQUIRE(played != nullptr);
    REQUIRE(played->articulation != nullptr);

    for (int i = 0; i < 120; ++i) physics.update(kFrame);

    const auto positions = played->articulation->jointPositions();// DOF add-order
    REQUIRE(positions.size() == 2);

    // For each simulated joint, the visual robot's same-named joint mirrors it.
    const auto info = robot->getArticulatedJointInfo();
    bool checkedAny = false;
    for (std::size_t d = 0; d < played->jointNames.size(); ++d) {
        for (std::size_t v = 0; v < info.size(); ++v) {
            if (info[v].name == played->jointNames[d]) {
                CHECK_THAT(robot->getJointValue(v), WithinAbs(positions[d], 1e-3f));
                checkedAny = true;
            }
        }
    }
    CHECK(checkedAny);
    // And the shoulder is near its authored 0.7, so the mirror is of a real pose,
    // not of a collapsed-to-zero one.
    CHECK(std::abs(robot->getJointValue(0)) > 0.4f);

    physics.stop();
}

TEST_CASE("A PhysicsConfig on a robot link does not create a second body") {

    const auto path = writeFixture();
    Scene scene;

    auto robot = authorRobot(scene, path, {0.0f, 0.0f}, /*fixedBase*/ true);

    // Author a rigid body on a LINK of the articulated robot. It must be skipped:
    // the articulation link is already the body there.
    auto* link = findByName(*robot, "upper_link");
    REQUIRE(link != nullptr);
    PhysicsConfig pc;
    pc.enabled = true;
    pc.body = PhysicsConfig::Body::Dynamic;
    pc.write(*link);

    PhysicsPlaySession physics;
    physics.start(scene);

    CHECK(physics.articulationCount() == 1);
    // No rigid body was created for the link (nor for any robot descendant).
    CHECK(physics.bodyCount() == 0);
    CHECK(physics.findActor(link) == nullptr);

    physics.stop();
}

TEST_CASE("A non-unit-scale robot is skipped with a log line") {

    const auto path = writeFixture();
    Scene scene;

    auto robot = authorRobot(scene, path, {0.0f, 0.0f}, /*fixedBase*/ true);
    robot->scale.set(2.f, 2.f, 2.f);// PhysX links cannot scale

    std::string logged;
    PhysicsPlaySession physics;
    physics.setLogger([&](const std::string& m) { logged += m + "\n"; });
    physics.start(scene);

    CHECK(physics.articulationCount() == 0);
    CHECK(logged.find("scale") != std::string::npos);

    physics.stop();
}
