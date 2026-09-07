// Inverse kinematics over a Robot.
//
// Reachability is the awkward part of testing an IK solver: a target picked by
// hand is either trivially close or quietly outside the workspace, and either
// way the test says nothing. So most of these pick a joint vector, run FORWARD
// kinematics to get a target, and require the solver to find its way back from
// somewhere else. The target is then reachable by construction, and the only
// thing under test is the solve.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/kinematics/InverseKinematics.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/objects/Robot.hpp"

#include <filesystem>
#include <string>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    std::shared_ptr<Robot> parseUrdf(const std::string& urdf) {
        URDFLoader loader;
        auto robot = loader.parse(std::filesystem::temp_directory_path(), urdf);
        if (robot) robot->updateMatrix();
        return robot;
    }

    std::string revolute(const std::string& name, const std::string& parent,
                         const std::string& child, const std::string& xyz,
                         const std::string& limits = "-3.0\" upper=\"3.0") {
        return R"(<joint name=")" + name + R"(" type="revolute">
                    <parent link=")" + parent + R"("/><child link=")" + child + R"("/>
                    <origin xyz=")" + xyz + R"(" rpy="0 0 0"/><axis xyz="0 0 1"/>
                    <limit lower=")" + limits + R"(" effort="1" velocity="1"/>
                  </joint>)";
    }

    std::string link(const std::string& name) {
        return R"(<link name=")" + name + R"("><visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual></link>)";
    }

    // Three revolute joints, all about Z, links running along +Y. Planar, and
    // redundant for a planar POSITION task — three joints for two constraints,
    // which is what gives the null-space term something to do.
    std::shared_ptr<Robot> planarArm(const std::string& limits = "-3.0\" upper=\"3.0") {
        const std::string urdf =
                R"(<robot name="planar">)" + link("base") + link("l1") + link("l2") + link("l3") +
                revolute("j1", "base", "l1", "0 0 0", limits) +
                revolute("j2", "l1", "l2", "0 1 0", limits) +
                revolute("j3", "l2", "l3", "0 1 0", limits) +
                R"(</robot>)";
        return parseUrdf(urdf);
    }

    // The same arm with a two-finger hand hanging off the tool link.
    std::shared_ptr<Robot> armWithHand() {
        const std::string urdf =
                R"(<robot name="handed">)" + link("base") + link("l1") + link("l2") + link("palm") +
                link("finger_left") + link("finger_right") +
                revolute("j1", "base", "l1", "0 0 0") +
                revolute("j2", "l1", "l2", "0 1 0") +
                revolute("j3", "l2", "palm", "0 1 0") +
                R"(<joint name="finger_l" type="prismatic">
                     <parent link="palm"/><child link="finger_left"/>
                     <origin xyz="0 0.1 0" rpy="0 0 0"/><axis xyz="1 0 0"/>
                     <limit lower="0" upper="0.04" effort="1" velocity="1"/>
                   </joint>
                   <joint name="finger_r" type="prismatic">
                     <parent link="palm"/><child link="finger_right"/>
                     <origin xyz="0 0.1 0" rpy="0 0 0"/><axis xyz="-1 0 0"/>
                     <limit lower="0" upper="0.04" effort="1" velocity="1"/>
                   </joint>)" +
                R"(</robot>)";
        auto robot = parseUrdf(urdf);
        if (robot) robot->setEndEffector("palm");
        return robot;
    }

    Vector3 positionOf(const Matrix4& m) {
        Vector3 p, s;
        Quaternion q;
        m.decompose(p, q, s);
        return p;
    }

}// namespace


TEST_CASE("Position IK finds its way back to a pose the arm can strike") {

    auto robot = planarArm();
    REQUIRE(robot);
    REQUIRE(robot->numDOF() == 3);

    IkOptions opts;
    opts.task = IkTask::Position;
    opts.maxIterations = 200;
    const IkSolver solver(*robot, opts);

    const Vector3 target = positionOf(solver.toolTransform({0.4f, -0.6f, 0.3f}));

    std::vector<float> q{0.f, 0.f, 0.f};
    const IkResult r = solver.solve(q, target);

    INFO("target (" << target.x << ", " << target.y << ", " << target.z << ") after "
                    << r.iterations << " iterations, error " << r.positionError);
    CHECK(r.converged);
    CHECK_THAT(r.positionError, WithinAbs(0.f, 1e-4));
}

TEST_CASE("A tool offset is what the solver actually drives") {

    // The TCP, not the flange. With a 0.5 m tool the flange must end up half a
    // metre short of the target along the tool's own axis — if the offset were
    // ignored the flange would sit ON the target and the error would still read
    // as zero, which is the failure this catches.
    auto robot = planarArm();
    REQUIRE(robot);

    IkOptions opts;
    opts.task = IkTask::Position;
    opts.maxIterations = 200;
    opts.toolOffset.setPosition(0.f, 0.5f, 0.f);
    const IkSolver solver(*robot, opts);

    const Vector3 target{0.7f, 1.2f, 0.f};
    std::vector<float> q{0.2f, 0.2f, 0.2f};
    const IkResult r = solver.solve(q, target);

    const Vector3 flange = positionOf(robot->computeEndEffectorTransform(q));
    const Vector3 tcp = positionOf(solver.toolTransform(q));

    INFO("q = (" << q[0] << ", " << q[1] << ", " << q[2] << ") after " << r.iterations
                 << " iters, pos err " << r.positionError << ", ori err " << r.orientationError
                 << ", flange (" << flange.x << ", " << flange.y << ", " << flange.z
                 << "), tcp (" << tcp.x << ", " << tcp.y << ", " << tcp.z << ")");
    REQUIRE(r.converged);

    INFO("flange (" << flange.x << ", " << flange.y << ") tcp (" << tcp.x << ", " << tcp.y << ")");
    CHECK_THAT(tcp.distanceTo(target), WithinAbs(0.f, 1e-4));
    CHECK_THAT(flange.distanceTo(tcp), WithinAbs(0.5f, 1e-4));
}

TEST_CASE("AxisAlign aims the tool while it reaches") {

    auto robot = planarArm();
    REQUIRE(robot);

    // Steer the tool's +Y (the direction the links run) onto world +X.
    IkOptions opts;
    opts.task = IkTask::AxisAlign;
    opts.toolAxis.set(0.f, 1.f, 0.f);
    opts.targetAxis.set(1.f, 0.f, 0.f);
    opts.orientationWeight = 1.f;
    opts.maxIterations = 400;
    const IkSolver solver(*robot, opts);

    const Vector3 target = positionOf(solver.toolTransform({-0.3f, 0.5f, 0.4f}));

    std::vector<float> q{0.f, 0.2f, 0.2f};
    const IkResult r = solver.solve(q, target);

    INFO("pos err " << r.positionError << ", ori err " << r.orientationError
                    << " after " << r.iterations);
    CHECK_THAT(r.positionError, WithinAbs(0.f, 1e-3));

    // The residual is sin(angle) between the axes, so a small residual is a
    // small angle. Check the direction directly rather than trusting the number.
    Vector3 p, s;
    Quaternion rot;
    solver.toolTransform(q).decompose(p, rot, s);
    Vector3 axis{0.f, 1.f, 0.f};
    axis.applyQuaternion(rot).normalize();
    INFO("tool axis (" << axis.x << ", " << axis.y << ", " << axis.z << ")");
    CHECK_THAT(axis.x, WithinAbs(1.f, 1e-2));
}

TEST_CASE("Pose IK reproduces a full target transform") {

    auto robot = planarArm();
    REQUIRE(robot);

    IkOptions opts;
    opts.task = IkTask::Pose;
    opts.orientationWeight = 1.f;
    opts.maxIterations = 400;
    const IkSolver solver(*robot, opts);

    const Matrix4 target = solver.toolTransform({0.5f, 0.4f, -0.7f});

    std::vector<float> q{0.f, 0.f, 0.f};
    const IkResult r = solver.solve(q, target);

    INFO("pos err " << r.positionError << ", ori err " << r.orientationError
                    << " after " << r.iterations);
    CHECK(r.converged);
    CHECK_THAT(r.positionError, WithinAbs(0.f, 1e-3));
    CHECK_THAT(r.orientationError, WithinAbs(0.f, 1e-2));
}

TEST_CASE("IK never moves the gripper's finger joints") {

    // The reason Robot grew a tip link at all. The fingers are DOFs of the same
    // robot and sit in the same joint vector, but they are not on the path to
    // the tool, so no amount of reaching may disturb them.
    auto robot = armWithHand();
    REQUIRE(robot);
    REQUIRE(robot->numDOF() == 5);

    IkOptions opts;
    opts.task = IkTask::Position;
    opts.maxIterations = 200;
    const IkSolver solver(*robot, opts);

    REQUIRE(solver.solvedDofs().size() == 3);

    const Vector3 target = positionOf(solver.toolTransform({0.3f, 0.5f, -0.4f, 0.f, 0.f}));

    std::vector<float> q{0.f, 0.f, 0.f, 0.021f, 0.033f};
    const IkResult r = solver.solve(q, target);

    INFO("pos err " << r.positionError);
    CHECK(r.converged);
    CHECK_THAT(q[3], WithinAbs(0.021f, 1e-9));
    CHECK_THAT(q[4], WithinAbs(0.033f, 1e-9));
}

TEST_CASE("IK stays inside the joint limits") {

    auto robot = planarArm("-0.35\" upper=\"0.35");
    REQUIRE(robot);

    IkOptions opts;
    opts.task = IkTask::Position;
    opts.maxIterations = 200;
    const IkSolver solver(*robot, opts);

    // Deliberately out of reach for a hobbled arm, so the solve pushes hard
    // against the stops for every one of its iterations.
    const Vector3 target{2.5f, 0.2f, 0.f};
    std::vector<float> q{0.f, 0.f, 0.f};
    solver.solve(q, target);

    for (size_t i = 0; i < q.size(); ++i) {
        INFO("joint " << i << " = " << q[i]);
        CHECK(q[i] <= 0.35f + 1e-6f);
        CHECK(q[i] >= -0.35f - 1e-6f);
    }
}

TEST_CASE("A joint that reaches a limit is not stuck there") {

    // Robot's FK clamps to the joint limits, so a joint sitting exactly on a
    // stop is the one place a forward finite difference reports nothing: probe
    // outward, get clamped back, and the column reads as zero. A joint whose
    // column is zero cannot be moved in EITHER direction, so the arm that
    // pushed into a stop once would never leave it — the whole solve stops
    // working from that moment on, not just the joint.
    auto robot = planarArm("-0.35\" upper=\"0.35");
    REQUIRE(robot);

    IkOptions opts;
    opts.task = IkTask::Position;
    opts.maxIterations = 200;
    const IkSolver solver(*robot, opts);

    // Well out of reach: every joint ends up pinned against a stop.
    std::vector<float> q{0.f, 0.f, 0.f};
    solver.solve(q, {2.5f, 0.2f, 0.f});
    REQUIRE(q[0] > 0.35f - 1e-4f);// pinned against the upper stop

    // Now ask for a pose the arm can strike from well inside its range.
    const std::vector<float> goal{-0.2f, 0.15f, -0.1f};
    const Vector3 target = positionOf(solver.toolTransform(goal));

    const auto result = solver.solve(q, target);
    INFO("left at [" << q[0] << ", " << q[1] << ", " << q[2] << "]");
    CHECK(result.converged);
    CHECK(positionOf(solver.toolTransform(q)).distanceTo(target) < 1e-3f);
}

TEST_CASE("The speed cap bounds what one call may move") {

    auto robot = planarArm();
    REQUIRE(robot);

    IkOptions opts;
    opts.task = IkTask::Position;
    opts.maxIterations = 200;
    opts.maxJointSpeed = 1.f;// rad/s
    const IkSolver solver(*robot, opts);

    const Vector3 target = positionOf(solver.toolTransform({1.2f, -1.1f, 0.9f}));

    const std::vector<float> q0{0.f, 0.f, 0.f};
    std::vector<float> q = q0;
    const float dt = 0.01f;
    solver.solve(q, target, dt);

    for (size_t i = 0; i < q.size(); ++i) {
        INFO("joint " << i << " moved " << (q[i] - q0[i]) << " in " << dt << "s");
        CHECK(std::abs(q[i] - q0[i]) <= 1.f * dt + 1e-6f);
    }
}

TEST_CASE("The rest-posture pull does not cost reach") {

    // Three joints for a two-constraint planar task leaves one redundant DOF.
    // The null-space term is supposed to spend exactly that DOF and nothing
    // else, so switching it on must not degrade the position error.
    auto robot = planarArm();
    REQUIRE(robot);

    const Vector3 target{0.9f, 1.1f, 0.f};

    IkOptions plain;
    plain.task = IkTask::Position;
    plain.maxIterations = 200;
    std::vector<float> qPlain{0.1f, 0.1f, 0.1f};
    const IkResult rPlain = IkSolver(*robot, plain).solve(qPlain, target);

    IkOptions biased = plain;
    biased.restPoseGain = 0.05f;
    biased.restPose = {0.f, 0.6f, 0.6f};
    std::vector<float> qBiased{0.1f, 0.1f, 0.1f};
    const IkResult rBiased = IkSolver(*robot, biased).solve(qBiased, target);

    INFO("plain err " << rPlain.positionError << ", biased err " << rBiased.positionError);
    REQUIRE(rPlain.converged);
    CHECK(rBiased.converged);
    CHECK_THAT(rBiased.positionError, WithinAbs(0.f, 1e-3));

    // And it actually went somewhere else — otherwise the term did nothing and
    // the test above would pass for the wrong reason.
    float moved = 0.f;
    for (size_t i = 0; i < qPlain.size(); ++i) moved += std::abs(qBiased[i] - qPlain[i]);
    INFO("total joint difference " << moved);
    CHECK(moved > 1e-3f);
}
