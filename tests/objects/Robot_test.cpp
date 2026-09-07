// Robot forward kinematics.
//
// Robot answers "where is the end effector" two different ways:
//
//   getEndEffectorTransform()      drives the scene graph — setJointValue()
//                                  writes each joint node's local transform and
//                                  this reads the resulting world matrix.
//   computeEndEffectorTransform()  composes the same chain analytically,
//                                  without touching the scene graph.
//
// They are two implementations of one thing, so they have to agree for every
// joint type and every pose. That agreement is the invariant these tests pin:
// it is exactly what a prismatic joint used to violate, silently, because no
// URDF in the repo has one.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/objects/Robot.hpp"

#include <filesystem>
#include <string>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    std::shared_ptr<Robot> parseUrdf(const std::string& urdf) {
        URDFLoader loader;
        return loader.parse(std::filesystem::temp_directory_path(), urdf);
    }

    // A one-joint slider whose joint frame is rotated 90 degrees about Z, so the
    // joint's local +X axis points along the PARENT's +Y. Sliding it therefore
    // has to move the carriage in +Y — if the axis is used unrotated it moves in
    // +X instead, which is the whole bug.
    std::shared_ptr<Robot> slider(const std::string& limits = R"(<limit lower="-10" upper="10" effort="1" velocity="1"/>)") {
        const std::string urdf = R"(
            <robot name="slider">
              <link name="base">
                <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
              </link>
              <link name="carriage">
                <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
              </link>
              <joint name="slide" type="prismatic">
                <parent link="base"/>
                <child link="carriage"/>
                <origin xyz="0 0 0" rpy="0 0 1.5707963"/>
                <axis xyz="1 0 0"/>
                )" + limits + R"(
              </joint>
            </robot>)";
        return parseUrdf(urdf);
    }

    Vector3 positionOf(const Matrix4& m) {
        Vector3 p, s;
        Quaternion q;
        m.decompose(p, q, s);
        return p;
    }

    // An arm that BRANCHES: two revolute joints up to a palm, then two prismatic
    // fingers hanging off that palm side by side. This is the shape of every
    // real manipulator, and the shape the FK used to get wrong — it multiplied
    // every joint in the file into one product, so closing the gripper moved the
    // tool pose.
    std::shared_ptr<Robot> armWithGripper() {
        const std::string urdf = R"(
            <robot name="arm_with_gripper">
              <link name="base"><visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual></link>
              <link name="upper"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
              <link name="palm"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
              <link name="finger_left"><visual><geometry><box size="0.02 0.05 0.02"/></geometry></visual></link>
              <link name="finger_right"><visual><geometry><box size="0.02 0.05 0.02"/></geometry></visual></link>

              <joint name="shoulder" type="revolute">
                <parent link="base"/><child link="upper"/>
                <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
                <limit lower="-3" upper="3" effort="1" velocity="1"/>
              </joint>
              <joint name="wrist" type="revolute">
                <parent link="upper"/><child link="palm"/>
                <origin xyz="0 1 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
                <limit lower="-3" upper="3" effort="1" velocity="1"/>
              </joint>
              <joint name="finger_l" type="prismatic">
                <parent link="palm"/><child link="finger_left"/>
                <origin xyz="0 0.1 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
                <limit lower="0" upper="0.04" effort="1" velocity="1"/>
              </joint>
              <joint name="finger_r" type="prismatic">
                <parent link="palm"/><child link="finger_right"/>
                <origin xyz="0 0.1 0" rpy="0 0 0"/><axis xyz="-1 0 0"/>
                <limit lower="0" upper="0.04" effort="1" velocity="1"/>
              </joint>
            </robot>)";
        return parseUrdf(urdf);
    }

}// namespace


TEST_CASE("A prismatic joint slides along its own axis, not the parent's") {

    auto robot = slider();
    REQUIRE(robot);
    REQUIRE(robot->numDOF() == 1);

    robot->setJointValue(0, 0.5f);
    robot->updateMatrixWorld(true);

    // The joint frame is yawed 90 degrees, so its local +X is the parent's +Y.
    const Vector3 p = positionOf(robot->getEndEffectorTransform());
    INFO("end effector at " << p.x << ", " << p.y << ", " << p.z);
    CHECK_THAT(p.x, WithinAbs(0.f, 1e-5));
    CHECK_THAT(p.y, WithinAbs(0.5f, 1e-5));
    CHECK_THAT(p.z, WithinAbs(0.f, 1e-5));
}

TEST_CASE("Scene-graph and analytic FK agree for a prismatic joint") {

    // The invariant. Both paths are public API and callers mix them freely —
    // computeEndEffectorTransform() is the one an IK loop calls to evaluate a
    // candidate pose without disturbing the scene.
    auto robot = slider();
    REQUIRE(robot);

    for (const float v: {-1.5f, -0.4f, 0.f, 0.25f, 1.f, 2.75f}) {
        robot->setJointValue(0, v);
        robot->updateMatrixWorld(true);

        const Vector3 fromGraph = positionOf(robot->getEndEffectorTransform());
        const Vector3 fromCompute = positionOf(robot->computeEndEffectorTransform({v}));

        INFO("value " << v << ": graph (" << fromGraph.x << ", " << fromGraph.y << ", "
                      << fromGraph.z << ") vs computed (" << fromCompute.x << ", "
                      << fromCompute.y << ", " << fromCompute.z << ")");
        REQUIRE_THAT(fromGraph.x, WithinAbs(fromCompute.x, 1e-5));
        REQUIRE_THAT(fromGraph.y, WithinAbs(fromCompute.y, 1e-5));
        REQUIRE_THAT(fromGraph.z, WithinAbs(fromCompute.z, 1e-5));
    }
}

TEST_CASE("Rotating the robot does not steer its prismatic joints") {

    // A joint slides along its own axis in its own frame; where the robot as a
    // whole happens to be pointing is irrelevant to that. The old code rotated
    // the slide axis by the ROBOT's rotation, so posing the robot changed which
    // way its own joints travelled.
    //
    // Rotation about Y is chosen deliberately: with the joint yawed about Z, a
    // robot rotation about Z would coincidentally produce the right answer.
    auto robot = slider();
    REQUIRE(robot);

    robot->rotation.y = math::PI / 2.f;
    robot->setJointValue(0, 0.5f);
    robot->updateMatrixWorld(true);

    // Local displacement is +Y, and rotating about Y leaves +Y alone.
    const Vector3 p = positionOf(robot->getEndEffectorTransform());
    INFO("end effector at " << p.x << ", " << p.y << ", " << p.z);
    CHECK_THAT(p.x, WithinAbs(0.f, 1e-5));
    CHECK_THAT(p.y, WithinAbs(0.5f, 1e-5));
    CHECK_THAT(p.z, WithinAbs(0.f, 1e-5));

    // And the two FK paths still agree with the robot posed.
    const Vector3 c = positionOf(robot->computeEndEffectorTransform({0.5f}));
    CHECK_THAT(p.x, WithinAbs(c.x, 1e-5));
    CHECK_THAT(p.y, WithinAbs(c.y, 1e-5));
    CHECK_THAT(p.z, WithinAbs(c.z, 1e-5));
}

TEST_CASE("setJointValue clamps a prismatic joint to its limits") {

    // The revolute branch next to it clamps, and computeEndEffectorTransform
    // clamps prismatic too — so without this the setter drove the joint past a
    // limit the analytic path refused to, and the two disagreed again.
    auto robot = slider(R"(<limit lower="-0.2" upper="0.3" effort="1" velocity="1"/>)");
    REQUIRE(robot);

    const auto range = robot->getJointRange(0);
    REQUIRE_THAT(range.min, WithinAbs(-0.2f, 1e-6));
    REQUIRE_THAT(range.max, WithinAbs(0.3f, 1e-6));

    robot->setJointValue(0, 5.f);
    robot->updateMatrixWorld(true);
    CHECK_THAT(robot->jointValues()[0], WithinAbs(0.3f, 1e-5));
    CHECK_THAT(positionOf(robot->getEndEffectorTransform()).y, WithinAbs(0.3f, 1e-5));

    robot->setJointValue(0, -5.f);
    robot->updateMatrixWorld(true);
    CHECK_THAT(robot->jointValues()[0], WithinAbs(-0.2f, 1e-5));
    CHECK_THAT(positionOf(robot->getEndEffectorTransform()).y, WithinAbs(-0.2f, 1e-5));
}

TEST_CASE("Closing the gripper does not move the tool") {

    // The branch bug, stated as a test. With the end effector at the palm, the
    // finger joints are not on the root-to-tip path, so their values must have
    // no effect whatsoever on the tool pose. The old FK multiplied the flat
    // joint list, so squeezing the fingers translated the tool — which would
    // let an IK solver "reach" a target by closing the hand.
    auto robot = armWithGripper();
    REQUIRE(robot);
    REQUIRE(robot->numDOF() == 4);

    robot->setEndEffector("palm");
    REQUIRE(robot->endEffectorLink() == "palm");

    // Only the two arm joints are solvable; the fingers are excluded.
    const auto dofs = robot->chainDofs();
    REQUIRE(dofs.size() == 2);
    CHECK(dofs[0] == 0);
    CHECK(dofs[1] == 1);

    const Vector3 open = positionOf(robot->computeEndEffectorTransform({0.3f, -0.2f, 0.f, 0.f}));
    const Vector3 shut = positionOf(robot->computeEndEffectorTransform({0.3f, -0.2f, 0.04f, 0.04f}));

    INFO("open (" << open.x << ", " << open.y << ", " << open.z << ") vs "
                  << "shut (" << shut.x << ", " << shut.y << ", " << shut.z << ")");
    CHECK_THAT(open.x, WithinAbs(shut.x, 1e-6));
    CHECK_THAT(open.y, WithinAbs(shut.y, 1e-6));
    CHECK_THAT(open.z, WithinAbs(shut.z, 1e-6));
}

TEST_CASE("Scene-graph and analytic FK agree for a branched robot") {

    // The same invariant the serial tests pin, on a tree. Both paths have to
    // land on the palm, ignoring the fingers, for every arm pose.
    auto robot = armWithGripper();
    REQUIRE(robot);
    robot->setEndEffector("palm");

    for (const float a: {-1.1f, 0.f, 0.85f}) {
        for (const float b: {-0.6f, 0.f, 1.3f}) {

            // Fingers deliberately non-zero: they must not perturb either path.
            const std::vector<float> q{a, b, 0.03f, 0.01f};
            robot->setJointValues(q);
            robot->updateMatrixWorld(true);

            const Vector3 fromGraph = positionOf(robot->getEndEffectorTransform());
            const Vector3 fromCompute = positionOf(robot->computeEndEffectorTransform(q));

            INFO("q = (" << a << ", " << b << "): graph (" << fromGraph.x << ", " << fromGraph.y
                         << ", " << fromGraph.z << ") vs computed (" << fromCompute.x << ", "
                         << fromCompute.y << ", " << fromCompute.z << ")");
            REQUIRE_THAT(fromGraph.x, WithinAbs(fromCompute.x, 1e-5));
            REQUIRE_THAT(fromGraph.y, WithinAbs(fromCompute.y, 1e-5));
            REQUIRE_THAT(fromGraph.z, WithinAbs(fromCompute.z, 1e-5));
        }
    }
}

TEST_CASE("Without an explicit tip, the deepest leaf wins") {

    // Both fingers sit at the same depth, so the tie breaks on declaration
    // order and the left one wins. What matters is that the walk is a real
    // root-to-tip path either way: the OTHER finger is never multiplied in.
    auto robot = armWithGripper();
    REQUIRE(robot);
    REQUIRE(robot->endEffectorLink() == "finger_left");

    const auto dofs = robot->chainDofs();
    REQUIRE(dofs.size() == 3);
    CHECK(dofs[0] == 0);
    CHECK(dofs[1] == 1);
    CHECK(dofs[2] == 2);

    // Moving the right finger cannot move the left one.
    const Vector3 a = positionOf(robot->computeEndEffectorTransform({0.f, 0.f, 0.02f, 0.f}));
    const Vector3 b = positionOf(robot->computeEndEffectorTransform({0.f, 0.f, 0.02f, 0.04f}));
    CHECK_THAT(a.x, WithinAbs(b.x, 1e-6));
    CHECK_THAT(a.y, WithinAbs(b.y, 1e-6));
    CHECK_THAT(a.z, WithinAbs(b.z, 1e-6));
}

TEST_CASE("The KUKA iiwa's end effector is its tool, not its pedestal") {

    // The case that makes the depth rule non-negotiable. lbr_iiwa_14_r820
    // declares a base_link->base plate joint AFTER joint_a7-tool0, so the last
    // joint added names the robot's own BASE. Defaulting the tip to it puts the
    // end effector on the floor, and this is the robot RobotCell drives.
    const std::filesystem::path urdfPath =
            std::filesystem::path(DATA_FOLDER) / "urdf" / "lbr_iiwa_14_r820.urdf";
    if (!std::filesystem::exists(urdfPath)) {
        WARN("iiwa URDF not present at " << urdfPath.string() << ", skipping");
        return;
    }

    URDFLoader loader;
    auto robot = loader.load(urdfPath);
    REQUIRE(robot);
    REQUIRE(robot->numDOF() == 7);

    CHECK(robot->endEffectorLink() == "tool0");
    CHECK(robot->chainDofs().size() == 7);

    // The tool is a metre-ish up and out from the base once posed; the pedestal
    // would sit at the origin no matter what the joints do.
    const std::vector<float> q{0.3f, -0.5f, 0.2f, 0.9f, -0.4f, 0.6f, 0.1f};
    robot->setJointValues(q);
    robot->updateMatrixWorld(true);

    const Vector3 fromGraph = positionOf(robot->getEndEffectorTransform());
    const Vector3 fromCompute = positionOf(robot->computeEndEffectorTransform(q));

    INFO("graph (" << fromGraph.x << ", " << fromGraph.y << ", " << fromGraph.z
                   << ") vs computed (" << fromCompute.x << ", " << fromCompute.y
                   << ", " << fromCompute.z << ")");
    REQUIRE_THAT(fromGraph.x, WithinAbs(fromCompute.x, 1e-5));
    REQUIRE_THAT(fromGraph.y, WithinAbs(fromCompute.y, 1e-5));
    REQUIRE_THAT(fromGraph.z, WithinAbs(fromCompute.z, 1e-5));

    CHECK(fromGraph.length() > 0.3f);
}

TEST_CASE("The root link is found by topology, not by document order") {

    // finalize() used to attach links_.front() and call it the root. Here the
    // first <link> element is the TIP, so that assumption tears the tip out
    // from under its own joint and leaves the real root unattached — the whole
    // robot then floats free of its own transform. xacro-expanded URDFs emit
    // links in macro-expansion order, so this is the ordinary case, not a
    // contrived one.
    const std::string urdf = R"(
        <robot name="root_not_first">
          <link name="tip"><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>
          <link name="base"><visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual></link>
          <joint name="shoulder" type="revolute">
            <parent link="base"/><child link="tip"/>
            <origin xyz="0 1 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
            <limit lower="-3" upper="3" effort="1" velocity="1"/>
          </joint>
        </robot>)";

    auto robot = parseUrdf(urdf);
    REQUIRE(robot);
    REQUIRE(robot->numDOF() == 1);

    // Posing the robot is what exposes it: if the real root never got attached,
    // the joint's world matrix does not carry the robot's own transform.
    robot->position.set(1.f, 0.f, 0.f);
    robot->rotation.z = math::PI / 2.f;
    robot->setJointValue(0, 0.4f);
    robot->updateMatrixWorld(true);

    const Vector3 fromGraph = positionOf(robot->getEndEffectorTransform());
    const Vector3 fromCompute = positionOf(robot->computeEndEffectorTransform({0.4f}));

    INFO("graph (" << fromGraph.x << ", " << fromGraph.y << ", " << fromGraph.z
                   << ") vs computed (" << fromCompute.x << ", " << fromCompute.y
                   << ", " << fromCompute.z << ")");
    REQUIRE_THAT(fromGraph.x, WithinAbs(fromCompute.x, 1e-5));
    REQUIRE_THAT(fromGraph.y, WithinAbs(fromCompute.y, 1e-5));
    REQUIRE_THAT(fromGraph.z, WithinAbs(fromCompute.z, 1e-5));

    // Rotating +90 deg about Z sends the joint's +Y offset to -X, off the base.
    CHECK_THAT(fromGraph.x, WithinAbs(0.f, 1e-5));
    CHECK_THAT(fromGraph.y, WithinAbs(0.f, 1e-5));
}

TEST_CASE("The Franka FR3 resolves to its tool frame, with the fingers off-chain") {

    // A real arm-plus-gripper: 7 revolute arm joints, two prismatic fingers
    // branching off the hand, and an fr3_hand_tcp frame between them. The tool
    // frame and both fingers sit at the same depth, so the declaration-order
    // tie-break is what lands on the tool rather than on a fingertip.
    const std::filesystem::path urdfPath =
            std::filesystem::path(DATA_FOLDER) / "urdf" / "franka" / "fr3.urdf";
    if (!std::filesystem::exists(urdfPath)) {
        WARN("Franka URDF not present at " << urdfPath.string() << ", skipping");
        return;
    }

    URDFLoader loader;
    auto robot = loader.load(urdfPath);
    REQUIRE(robot);

    // 7 arm + 2 fingers. The many fixed joints (accelerometers, flange, hand,
    // tcp) carry transform but no DOF.
    REQUIRE(robot->numDOF() == 9);
    CHECK(robot->endEffectorLink() == "fr3_hand_tcp");

    // The arm solves; the fingers do not.
    const auto dofs = robot->chainDofs();
    REQUIRE(dofs.size() == 7);
    for (size_t i = 0; i < dofs.size(); ++i) CHECK(dofs[i] == i);

    // Squeezing the gripper must not move the tool frame by so much as a micron.
    std::vector<float> q(9, 0.f);
    q[1] = -0.4f;
    q[3] = -1.9f;
    q[5] = 1.6f;
    const Vector3 open = positionOf(robot->computeEndEffectorTransform(q));
    q[7] = 0.04f;
    q[8] = 0.04f;
    const Vector3 shut = positionOf(robot->computeEndEffectorTransform(q));

    INFO("tcp open (" << open.x << ", " << open.y << ", " << open.z << ") vs shut ("
                      << shut.x << ", " << shut.y << ", " << shut.z << ")");
    CHECK_THAT(open.x, WithinAbs(shut.x, 1e-6));
    CHECK_THAT(open.y, WithinAbs(shut.y, 1e-6));
    CHECK_THAT(open.z, WithinAbs(shut.z, 1e-6));

    // And the two FK paths agree on the posed robot.
    robot->setJointValues(q);
    robot->updateMatrixWorld(true);
    const Vector3 fromGraph = positionOf(robot->getEndEffectorTransform());
    INFO("graph (" << fromGraph.x << ", " << fromGraph.y << ", " << fromGraph.z << ")");
    REQUIRE_THAT(fromGraph.x, WithinAbs(shut.x, 1e-5));
    REQUIRE_THAT(fromGraph.y, WithinAbs(shut.y, 1e-5));
    REQUIRE_THAT(fromGraph.z, WithinAbs(shut.z, 1e-5));
}
TEST_CASE("Revolute joints already agreed, and still do") {

    // The control: the revolute path was never broken, so it must be unchanged.
    const std::string urdf = R"(
        <robot name="arm">
          <link name="base">
            <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
          </link>
          <link name="upper">
            <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base"/>
            <child link="upper"/>
            <origin xyz="0 1 0" rpy="0 0 0.7853981"/>
            <axis xyz="0 0 1"/>
            <limit lower="-3" upper="3" effort="1" velocity="1"/>
          </joint>
        </robot>)";

    auto robot = parseUrdf(urdf);
    REQUIRE(robot);
    REQUIRE(robot->numDOF() == 1);

    for (const float v: {-1.2f, 0.f, 0.6f, 1.5f}) {
        robot->setJointValue(0, v);
        robot->updateMatrixWorld(true);

        const Vector3 fromGraph = positionOf(robot->getEndEffectorTransform());
        const Vector3 fromCompute = positionOf(robot->computeEndEffectorTransform({v}));

        INFO("value " << v);
        REQUIRE_THAT(fromGraph.x, WithinAbs(fromCompute.x, 1e-5));
        REQUIRE_THAT(fromGraph.y, WithinAbs(fromCompute.y, 1e-5));
        REQUIRE_THAT(fromGraph.z, WithinAbs(fromCompute.z, 1e-5));
    }
}
