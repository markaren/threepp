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
//                       robot, skips a robot's links in the rigid-body scan, folds
//                       a uniform scale into the build (the millimetre-URDF case)
//                       and refuses a non-uniform one.
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
#include <map>
#include <memory>
#include <string>
#include <utility>
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

    // The same arm drawn in millimetres — every length x1000, every angle left
    // alone. What a CAD export looks like when it lands in a metre scene.
    const char* kUrdfMillimetres = R"(
        <robot name="arm">
          <link name="base_link">
            <visual><geometry><box size="200 200 200"/></geometry></visual>
            <collision><geometry><box size="200 200 200"/></geometry></collision>
          </link>
          <link name="upper_link">
            <visual><geometry><box size="100 400 100"/></geometry></visual>
            <collision><geometry><box size="100 400 100"/></geometry></collision>
          </link>
          <link name="slider_link">
            <visual><geometry><box size="50 200 50"/></geometry></visual>
            <collision><geometry><box size="50 200 50"/></geometry></collision>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base_link"/>
            <child link="upper_link"/>
            <origin xyz="0 0 300" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-2.0" upper="2.0"/>
          </joint>
          <joint name="extend" type="prismatic">
            <parent link="upper_link"/>
            <child link="slider_link"/>
            <origin xyz="0 400 0" rpy="0 0 0"/>
            <axis xyz="0 1 0"/>
            <limit lower="0.0" upper="500.0"/>
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

    std::filesystem::path writeMillimetreFixture() {
        const auto dir = std::filesystem::temp_directory_path() / "threepp-articulation-test";
        std::filesystem::create_directories(dir);
        const auto path = dir / "arm_mm.urdf";
        std::ofstream(path, std::ios::trunc) << kUrdfMillimetres;
        return path;
    }

    std::shared_ptr<Robot> loadArm(const std::filesystem::path& path) {
        URDFLoader loader;
        return loader.load(path);
    }

    // A PARAMETERISED description: the upper link's length, and so where the distal joint sits,
    // comes from a xacro argument. Two links only, revolute, primitives - the point is the arg,
    // not the mechanism.
    //
    // The real case this stands in for is UR's ur.urdf.xacro, which derives its joint-limit and
    // kinematics yaml PATHS from $(arg ur_type); expanded with the file's default the paths do
    // not exist and nothing builds at all. An arg that moves a joint fails more quietly, which
    // makes it the better regression: it is the silent version.
    // NOTE the XML( ) delimiter rather than a bare R"( )": `$(arg reach)"` contains the sequence
    // that would close the default one.
    const char* kUrdfWithArgs = R"XML(
        <robot name="arm" xmlns:xacro="http://www.ros.org/wiki/xacro">
          <xacro:arg name="reach" default="0.1"/>
          <link name="base_link">
            <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
            <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
          </link>
          <link name="upper_link">
            <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
            <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base_link"/>
            <child link="upper_link"/>
            <origin xyz="0 0 $(arg reach)" rpy="0 0 0"/>
            <axis xyz="0 1 0"/>
            <limit lower="-2.0" upper="2.0"/>
          </joint>
        </robot>)XML";

    std::filesystem::path writeArgFixture() {
        const auto dir = std::filesystem::temp_directory_path() / "threepp-articulation-test";
        std::filesystem::create_directories(dir);
        const auto path = dir / "arm_args.urdf.xacro";
        std::ofstream(path, std::ios::trunc) << kUrdfWithArgs;
        return path;
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
                                       const std::vector<float>& pose, bool fixedBase = true,
                                       std::vector<std::pair<std::string, std::string>> args = {}) {
        URDFLoader loader;
        if (!args.empty()) {
            std::map<std::string, std::string> map;
            for (const auto& [name, value] : args) map[name] = value;
            loader.setArgs(map);
        }
        auto robot = loader.load(path);
        RobotConfig rc;
        rc.urdf = path.string();
        rc.joints = pose;
        rc.xacroArgs = std::move(args);
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


TEST_CASE("A failed articulation load says why") {

    // loadArticulation builds its own URDFLoader, uses it, and lets it die -
    // so everything the parser worked out about the file (an xacro line
    // number, an unreadable include, a document with no <robot> root) went
    // with it, and the caller got back an empty result that says only "no".
    // The result now carries the account out. python's load_articulation puts
    // it in the exception, which is where a ROS node reads it.
    const auto dir = std::filesystem::temp_directory_path() / "threepp-articulation-test";
    std::filesystem::create_directories(dir);

    PhysxWorld::Settings settings;
    settings.fixedTimestep = kFrame;
    settings.maxSubSteps = 1;

    SECTION("a file that is not there") {

        PhysxWorld world(settings);
        const auto missing = dir / "no_such_robot.urdf";
        std::error_code ec;
        std::filesystem::remove(missing, ec);

        auto built = loadArticulation(world, missing, URDFArticulationOptions{});
        REQUIRE_FALSE(built.articulation);
        REQUIRE_FALSE(built.error.empty());
        CHECK(built.error.find("no_such_robot") != std::string::npos);
        CHECK_FALSE(built.diagnostics.empty());
    }

    SECTION("a xacro that cannot expand") {

        PhysxWorld world(settings);
        const auto path = dir / "undefined_property.urdf.xacro";
        // The undefined name is on line 4, which is what the assertion below
        // reads back out of the message.
        std::ofstream(path, std::ios::trunc) << R"XML(<?xml version='1.0'?>
<robot name='x' xmlns:xacro='http://www.ros.org/wiki/xacro'>
  <link name='base'><visual><geometry>
    <box size='${nope} 1 1'/>
  </geometry></visual></link>
</robot>
)XML";

        auto built = loadArticulation(world, path, URDFArticulationOptions{});
        REQUIRE_FALSE(built.articulation);
        REQUIRE_FALSE(built.error.empty());
        // The name it could not resolve AND the line it was on - the two
        // things that turn "it failed" into a fix.
        CHECK(built.error.find("nope") != std::string::npos);
        CHECK(built.error.find(":4") != std::string::npos);
    }

    SECTION("a load that succeeds leaves the error empty") {

        PhysxWorld world(settings);
        const auto path = writeFixture();

        URDFArticulationOptions opts;
        opts.fixedBase = true;
        opts.renderVisuals = false;
        auto built = loadArticulation(world, path, opts);
        REQUIRE(built.articulation);
        // The invariant the field is documented with: empty exactly when the
        // articulation came up. Diagnostics may still hold warnings, which is
        // why the two are separate.
        CHECK(built.error.empty());
    }
}

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

TEST_CASE("A non-uniformly scaled robot is skipped with a log line") {

    const auto path = writeFixture();
    Scene scene;

    auto robot = authorRobot(scene, path, {0.0f, 0.0f}, /*fixedBase*/ true);
    // A sphere, a capsule and a joint frame have no way to take a per-axis
    // scale, so this one has to be refused rather than approximated.
    robot->scale.set(2.f, 1.f, 2.f);

    std::string logged;
    PhysicsPlaySession physics;
    physics.setLogger([&](const std::string& m) { logged += m + "\n"; });
    physics.start(scene);

    CHECK(physics.articulationCount() == 0);
    CHECK(logged.find("scale") != std::string::npos);

    physics.stop();
}

TEST_CASE("A millimetre robot scaled into a metre scene simulates in the right place") {

    // The whole point of the units support, end to end: the SAME arm drawn in
    // millimetres, dropped in at 0.001, must simulate where the metre arm does.
    const auto metrePath = writeFixture();
    const auto mmPath = writeMillimetreFixture();

    const std::vector<float> metrePose{0.6f, 0.2f};    // rad, metres
    const std::vector<float> mmPose{0.6f, 200.f};      // rad, millimetres - the
                                                       // robot's own units, which
                                                       // is what the document stores
    constexpr float unit = 0.001f;

    // Where the metre robot puts its distal link, kinematically.
    auto metreVisual = loadArm(metrePath);
    REQUIRE(metreVisual);
    for (std::size_t i = 0; i < metrePose.size() && i < metreVisual->numDOF(); ++i) {
        metreVisual->setJointValue(i, metrePose[i]);
    }
    metreVisual->updateMatrixWorld(true);
    auto* metreLink = findByName(*metreVisual, "slider_link");
    REQUIRE(metreLink != nullptr);
    Vector3 expected;
    metreLink->getWorldPosition(expected);

    // The millimetre robot, scaled into the scene the way a user would.
    Scene scene;
    auto robot = authorRobot(scene, mmPath, mmPose, /*fixedBase*/ true);
    robot->scale.set(unit, unit, unit);

    std::string logged;
    PhysicsPlaySession physics;
    physics.setLogger([&](const std::string& m) { logged += m + "\n"; });
    physics.start(scene);

    INFO(logged);
    REQUIRE(physics.articulationCount() == 1);

    const auto* played = physics.findArticulation(robot.get());
    REQUIRE(played != nullptr);
    const auto* extend = played->linkFor("extend");
    REQUIRE(extend != nullptr);

    // A few frames so the drive settles and the bound poses are current.
    for (int i = 0; i < 60; ++i) physics.update(kFrame);

    // The articulation link sits where the metre robot's link does - the shapes,
    // the joint frames and the prismatic offset were all built at scene size.
    const Vector3 artPos = extend->position();
    CHECK_THAT(artPos.x, WithinAbs(expected.x, 2e-2f));
    CHECK_THAT(artPos.y, WithinAbs(expected.y, 2e-2f));
    CHECK_THAT(artPos.z, WithinAbs(expected.z, 2e-2f));

    // And the mirror came back in the ROBOT's units, not the scene's: a
    // prismatic DOF solves in metres here, and the visual robot slides in
    // millimetres. Without the conversion this reads ~0.2 instead of ~200 and
    // the arm silently collapses to its zero pose on screen.
    const auto info = robot->getArticulatedJointInfo();
    std::size_t prismatic = info.size();
    for (std::size_t v = 0; v < info.size(); ++v) {
        if (info[v].name == "extend") prismatic = v;
    }
    REQUIRE(prismatic < info.size());
    CHECK_THAT(robot->getJointValue(prismatic), WithinAbs(200.f, 40.f));

    // Which is the same thing as saying the visual link agrees with the sim.
    robot->updateMatrixWorld(true);
    auto* visualLink = findByName(*robot, "slider_link");
    REQUIRE(visualLink != nullptr);
    Vector3 visualPos;
    visualLink->getWorldPosition(visualPos);
    CHECK_THAT(visualPos.x, WithinAbs(artPos.x, 3e-2f));
    CHECK_THAT(visualPos.y, WithinAbs(artPos.y, 3e-2f));
    CHECK_THAT(visualPos.z, WithinAbs(artPos.z, 3e-2f));

    physics.stop();
}

TEST_CASE("An articulation is built with the caller's xacro arguments") {

    const auto path = writeArgFixture();

    PhysxWorld::Settings settings;
    settings.fixedTimestep = kFrame;
    settings.maxSubSteps = 1;

    // PhysX allows ONE foundation per process, so the two worlds below are scoped rather than
    // both held - the second cannot be constructed until the first is destroyed.
    float withArgs = 0.f;
    float withoutArgs = 0.f;

    {
        PhysxWorld world(settings);
        // The fixture puts the shoulder joint at z = $(arg reach), default 0.1. Asking for 0.6
        // has to move it: before opts.args existed, loadArticulation built its own URDFLoader and
        // never called setArgs, so every caller silently got the file's defaults.
        URDFArticulationOptions opts;
        opts.fixedBase = true;
        opts.renderVisuals = false;
        opts.args = {{"reach", "0.6"}};

        auto built = loadArticulation(world, path, opts);
        REQUIRE(built.articulation);
        REQUIRE(built.links.size() == 1);
        world.step(kFrame);
        withArgs = built.links[0].position().z;
    }

    {
        // And the default is still the default when nobody asks - this is an override, not a
        // requirement, and a file with no args must keep working.
        PhysxWorld plain(settings);
        URDFArticulationOptions bare;
        bare.fixedBase = true;
        bare.renderVisuals = false;
        auto fallback = loadArticulation(plain, path, bare);
        REQUIRE(fallback.articulation);
        REQUIRE(fallback.links.size() == 1);
        plain.step(kFrame);
        withoutArgs = fallback.links[0].position().z;
    }

    CHECK_THAT(withArgs, WithinAbs(0.6f, 1e-3f));
    CHECK_THAT(withoutArgs, WithinAbs(0.1f, 1e-3f));
}

TEST_CASE("Play simulates the robot the document describes, not the xacro's defaults") {

    // The bug this pins: EditorApp::rearticulateRobots passes RobotConfig::argMap() when it
    // rebuilds the VISUAL robot, so the viewport shows the arguments the document was imported
    // with. PhysicsPlaySession did not, so the articulation was built from the same file with
    // different arguments - one document, two robots, and only the invisible one was wrong.
    const auto path = writeArgFixture();
    Scene scene;

    auto robot = authorRobot(scene, path, {0.0f}, /*fixedBase*/ true, {{"reach", "0.6"}});
    REQUIRE(robot->numDOF() == 1);

    PhysicsPlaySession physics;
    physics.start(scene);
    REQUIRE(physics.articulationCount() == 1);

    for (int i = 0; i < 30; ++i) physics.update(kFrame);

    // Where the visual robot has the joint, from the document's arguments.
    robot->updateMatrixWorld(true);
    auto* visualLink = findByName(*robot, "upper_link");
    REQUIRE(visualLink != nullptr);
    Vector3 visualPos;
    visualLink->getWorldPosition(visualPos);
    CHECK_THAT(visualPos.z, WithinAbs(0.6f, 1e-2f));

    // And where the simulation has it: the link whose INBOUND joint is "shoulder" is upper_link.
    // Same number, or the two are describing different robots.
    const auto* played = physics.findArticulation(robot.get());
    REQUIRE(played != nullptr);
    const auto* link = played->linkFor("shoulder");
    REQUIRE(link != nullptr);
    CHECK_THAT(link->position().z, WithinAbs(visualPos.z, 2e-2f));

    physics.stop();
}
