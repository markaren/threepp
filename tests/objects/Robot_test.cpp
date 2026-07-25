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
