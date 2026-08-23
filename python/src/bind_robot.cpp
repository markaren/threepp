// URDF robots: URDFLoader parses a URDF/xacro file (or XML string) into a Robot
// — an articulated Object3D with forward kinematics and joint introspection.
// Robot derives non-virtually from Object3D, so it inherits the Object3D base
// bindings (position, traverse, matrix_world, ...) directly.
#include "bindings.hpp"

#include <pybind11/stl.h>

#include "threepp/extras/kinematics/InverseKinematics.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/objects/Robot.hpp"

#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // What to raise when a load/parse came back empty.
    //
    // A free function rather than a local: the lambdas below are handed to
    // pybind11 and capture nothing.
    //
    // The loader's own message already names the file it could not read, so
    // this uses it INSTEAD of the generic sentence rather than in front of it -
    // prefixing produced "could not load 'X' - cannot read 'X': ...", which
    // states the path twice and buries the only new word in the line. The
    // generic form is the fallback for the case that cannot happen by contract
    // (lastError is documented never to be empty after a failure) but would be
    // a silent, messageless exception if it ever did.
    std::string loadFailure(const threepp::URDFLoader& loader, const std::string& what) {

        const auto why = loader.lastError();
        return why.empty() ? "URDFLoader: " + what : "URDFLoader: " + why;
    }

}// namespace

namespace threepp_py {

    void init_robot(py::module_& m) {

        // ---- Joint value types ----------------------------------------------
        py::enum_<Robot::JointType>(m, "JointType")
                .value("Revolute", Robot::JointType::Revolute)
                .value("Prismatic", Robot::JointType::Prismatic)
                .value("Fixed", Robot::JointType::Fixed);

        py::class_<Robot::JointRange>(m, "JointRange")
                .def_readonly("min", &Robot::JointRange::min)
                .def_readonly("max", &Robot::JointRange::max)
                .def("mid", &Robot::JointRange::mid)
                .def("clamp", &Robot::JointRange::clamp, py::arg("value"))
                .def("__repr__", [](const Robot::JointRange& r) {
                    std::ostringstream o;
                    o << "JointRange(min=" << r.min << ", max=" << r.max << ")";
                    return o.str();
                });

        py::class_<Robot::JointInfo>(m, "JointInfo")
                .def_readonly("axis", &Robot::JointInfo::axis)
                .def_readonly("type", &Robot::JointInfo::type)
                .def_readonly("name", &Robot::JointInfo::name)
                .def_readonly("parent", &Robot::JointInfo::parent)
                .def_readonly("child", &Robot::JointInfo::child)
                // std::optional<JointRange> -> JointRange or None
                .def_property_readonly("range", [](const Robot::JointInfo& j) -> py::object {
                    if (j.range) return py::cast(*j.range);
                    return py::none();
                })
                .def("__repr__", [](const Robot::JointInfo& j) {
                    return "<threepp.JointInfo name='" + j.name + "'>";
                });

        // ---- Robot (articulated Object3D) -----------------------------------
        // DOF index addresses the articulated (non-Fixed) joints in order. Angles
        // are radians unless deg=True. set/get joint values, query limits, and run
        // forward kinematics to the end-effector.
        py::class_<Robot, Object3D, std::shared_ptr<Robot>>(m, "Robot")
                .def(py::init<>())
                .def_property_readonly("num_dof", &Robot::numDOF)
                .def("set_joint_value", &Robot::setJointValue,
                     py::arg("index"), py::arg("value"), py::arg("deg") = false)
                .def("set_joint_values", [](Robot& r, const std::vector<float>& values, bool deg) { r.setJointValues(values, deg); },
                     py::arg("values"), py::arg("deg") = false)
                .def("get_joint_value", &Robot::getJointValue,
                     py::arg("index"), py::arg("deg") = false)
                .def("joint_values", &Robot::jointValues, py::arg("deg") = false)
                .def("get_joint_range", &Robot::getJointRange,
                     py::arg("index"), py::arg("deg") = false)
                .def("get_joint_ranges", &Robot::getJointRanges, py::arg("deg") = false)
                .def("get_articulated_joint_info", &Robot::getArticulatedJointInfo)
                // FK: world transform of the end-effector (last joint). The
                // _transform getter reflects the current joint values; compute_*
                // evaluates a candidate joint vector without mutating the robot.
                .def("get_end_effector_transform", &Robot::getEndEffectorTransform)
                .def("compute_end_effector_transform",
                     [](const Robot& r, const std::vector<float>& values, bool deg, bool enforce_limits) {
                         return r.computeEndEffectorTransform(values, deg, enforce_limits);
                     },
                     py::arg("values"), py::arg("deg") = false, py::arg("enforce_limits") = true)
                .def("show_colliders", &Robot::showColliders, py::arg("flag"))
                // ---- Tool frame -------------------------------------------
                // Which link the FK and the IK solver drive. Defaults to the
                // deepest leaf of the articulated tree (fr3_hand_tcp on a
                // Franka); name a different link to solve for a flange, an
                // elbow, or a tool the URDF happens to carry.
                .def("set_end_effector", &Robot::setEndEffector, py::arg("link_name"),
                     "Retarget FK/IK at the named link. Recomputes the root-to-tool "
                     "path and therefore `chain_dofs`. Raises if the link is unknown.")
                .def_property_readonly("end_effector_link", &Robot::endEffectorLink,
                     "Name of the link FK and IK currently drive.")
                .def_property_readonly("chain_dofs", &Robot::chainDofs,
                     "DOF indices on the root-to-end-effector path, ascending — the "
                     "only ones an IkSolver is allowed to move. A gripper's finger "
                     "joints keep their slots in the joint vector but are not here, "
                     "so closing the hand can never be mistaken for extra reach.");

        // ---- Inverse kinematics (damped least squares) ----------------------
        // Header-only C++ solver, see extras/kinematics/InverseKinematics.hpp.
        // The solve is INCREMENTAL: it steps q toward the target, bounded by
        // maxIterations * maxPositionStep, so a real-time caller runs it once
        // per frame and an offline one loops until `converged`.
        py::enum_<IkTask>(m, "IkTask", "How much of the tool pose the solve must reproduce.")
                .value("Position", IkTask::Position, "3-DOF: reach the point, any orientation.")
                .value("AxisAlign", IkTask::AxisAlign,
                       "5-DOF: reach the point AND aim the tool axis; spin about that axis "
                       "is left free — what a drill, a suction cup or a symmetric two-finger "
                       "grasp wants.")
                .value("Pose", IkTask::Pose, "6-DOF: reproduce the full target transform.");

        py::class_<IkOptions>(m, "IkOptions")
                .def(py::init<>())
                .def_readwrite("task", &IkOptions::task)
                .def_readwrite("tool_offset", &IkOptions::toolOffset,
                               "Flange -> tool centre point (Matrix4). The solve drives the TCP, "
                               "so a tool of any length or mounting is described here rather "
                               "than in the URDF.")
                .def_readwrite("tool_axis", &IkOptions::toolAxis,
                               "AxisAlign only, in the TOOL frame (+Z is the URDF convention "
                               "for an approach direction).")
                .def_readwrite("target_axis", &IkOptions::targetAxis,
                               "AxisAlign only, in WORLD space.")
                .def_readwrite("max_iterations", &IkOptions::maxIterations)
                .def_readwrite("position_tolerance", &IkOptions::positionTolerance, "metres")
                .def_readwrite("orientation_tolerance", &IkOptions::orientationTolerance, "radians")
                .def_readwrite("max_position_step", &IkOptions::maxPositionStep,
                               "Largest correction one iteration will attempt, in metres. A "
                               "Gauss-Newton step is a LOCAL statement; clamping keeps it inside "
                               "the trust region. Travel per solve is bounded by "
                               "max_iterations * step. Zero disables the clamp.")
                .def_readwrite("max_orientation_step", &IkOptions::maxOrientationStep, "radians")
                .def_readwrite("damping", &IkOptions::damping,
                               "DLS damping; must be > 0 — it is what lets the solve succeed at a "
                               "singularity instead of flinging the arm.")
                .def_readwrite("orientation_weight", &IkOptions::orientationWeight,
                               "Weight on the orientation rows relative to position. Below 1 the "
                               "solver reaches the point first and straightens up after, which "
                               "reads as natural motion.")
                .def_readwrite("revolute_step", &IkOptions::revoluteStep,
                               "Finite-difference probe for revolute joints (radians).")
                .def_readwrite("prismatic_step", &IkOptions::prismaticStep,
                               "Finite-difference probe for prismatic joints (metres).")
                .def_readwrite("rest_pose_gain", &IkOptions::restPoseGain,
                               "Null-space rest-posture pull per iteration; zero disables it. "
                               "Only does anything on a redundant arm.")
                .def_readwrite("rest_pose", &IkOptions::restPose,
                               "Rest posture, indexed by GLOBAL dof like every other joint vector.")
                .def_readwrite("null_space_damping", &IkOptions::nullSpaceDamping,
                               "Damping for the null-space PROJECTION — much smaller than `damping`, "
                               "or the posture bias leaks into the tool pose and the arm never "
                               "reports convergence.")
                .def_readwrite("max_joint_speed", &IkOptions::maxJointSpeed,
                               "Per-call joint speed cap in rad/s or m/s, applied against the joint "
                               "values as they arrived. Zero disables it; needs a non-zero dt.")
                .def("__repr__", [](const IkOptions& o) {
                    std::ostringstream s;
                    s << "IkOptions(task=" << static_cast<int>(o.task)
                      << ", max_iterations=" << o.maxIterations
                      << ", damping=" << o.damping << ")";
                    return s.str();
                });

        py::class_<IkResult>(m, "IkResult")
                .def_readonly("iterations", &IkResult::iterations)
                .def_readonly("position_error", &IkResult::positionError, "metres")
                .def_readonly("orientation_error", &IkResult::orientationError, "radians")
                .def_readonly("converged", &IkResult::converged,
                              "Both errors are inside tolerance — judged on the TRUE error, so "
                              "the step clamp never fakes it.")
                .def("__repr__", [](const IkResult& r) {
                    std::ostringstream s;
                    s << "IkResult(iterations=" << r.iterations
                      << ", position_error=" << r.positionError
                      << ", orientation_error=" << r.orientationError
                      << ", converged=" << (r.converged ? "True" : "False") << ")";
                    return s.str();
                });

        py::class_<IkSolver>(m, "IkSolver")
                .def(py::init([](const Robot& robot, IkOptions options) {
                         return std::make_unique<IkSolver>(robot, std::move(options));
                     }),
                     py::arg("robot"), py::arg("options") = IkOptions(), py::keep_alive<1, 2>(),
                     "Damped-least-squares IK over the robot's root-to-end-effector chain.\n\n"
                     "Joint ranges and the solvable DOF set are cached at construction, so a "
                     "Robot that is re-parsed or given a new end effector needs a fresh solver.")
                .def("solve", [](const IkSolver& s, std::vector<float> q, const py::object& target, float dt) {
                    // q is copied in and handed back: the C++ solve mutates in
                    // place, which a Python caller passing a list would never
                    // expect, and the copy is 9 floats.
                    IkResult r;
                    if (py::isinstance<Matrix4>(target)) {
                        r = s.solve(q, target.cast<const Matrix4&>(), dt);
                    } else {
                        r = s.solve(q, target.cast<const Vector3&>(), dt);
                    }
                    return py::make_tuple(std::move(q), r);
                },
                     py::arg("q"), py::arg("target"), py::arg("dt") = 0.f,
                     "Step q toward placing the tool at `target` (Vector3 = point, Matrix4 = "
                     "full pose). q is a FULL joint vector indexed by global dof; only "
                     "`solved_dofs` are modified. Returns (new_q, IkResult) — the input list is "
                     "left alone. `dt` is used solely by the max_joint_speed cap.")
                .def("tool_transform", &IkSolver::toolTransform, py::arg("q"),
                     "The tool centre point for a joint vector, in the robot's PARENT frame "
                     "(FK composed with tool_offset). Call robot.update_matrix() first if the "
                     "robot itself has moved.")
                .def_property("options",
                              [](const IkSolver& s) { return s.options(); },
                              [](IkSolver& s, const IkOptions& o) { s.options() = o; },
                              "Solver options. Reading gives a COPY — assign back to change them.")
                .def_property_readonly("solved_dofs", [](const IkSolver& s) { return s.solvedDofs(); },
                                       "The DOF indices this solver is allowed to move (robot.chain_dofs).");

        // ---- URDFLoader -----------------------------------------------------
        py::class_<URDFLoader>(m, "URDFLoader")
                .def(py::init<>())
                .def("set_args", [](URDFLoader& l, std::map<std::string, std::string> args) { l.setArgs(std::move(args)); },
                     py::arg("args"),
                     "xacro arg overrides (equivalent to name:=value on the xacro CLI).")
                .def("load", [](URDFLoader& l, const std::string& path) {
                    auto robot = l.load(path);
                    // The loader spent real effort working out WHY - an xacro
                    // expansion error with a line number, an unreadable
                    // include, a document with no <robot> root - and lastError()
                    // is where it put it. Raising without it throws that away
                    // and leaves a caller with "failed", which is the one thing
                    // they already knew.
                    if (!robot) throw std::runtime_error(loadFailure(l, "could not load '" + path + "'"));
                    return robot;
                }, py::arg("path"),
                   "Load a .urdf/.xacro file into a Robot (meshes via ModelLoader).\n\n"
                   "Raises RuntimeError carrying the parser's own explanation - the same text "
                   "`last_error` holds, which for a xacro failure includes the file and LINE. "
                   "Read `diagnostics` afterwards for the warnings too, which a load that "
                   "SUCCEEDS can also produce.")
                .def("parse", [](URDFLoader& l, const std::string& base_dir, const std::string& xml) {
                    auto robot = l.parse(base_dir, xml);
                    if (!robot) throw std::runtime_error(loadFailure(l, "could not parse the URDF XML"));
                    return robot;
                }, py::arg("base_dir"), py::arg("xml"),
                   "Parse URDF XML from a string; base_dir resolves relative mesh paths.\n\n"
                   "Raises RuntimeError carrying the parser's own explanation, exactly as `load` does. "
                   "This is the call a ROS node makes on /robot_description, where the XML came off a "
                   "topic and there is no file to go and look at.")
                .def_property_readonly("last_error", &URDFLoader::lastError,
                   "Why the most recent load/parse/parse_articulation failed: the errors it "
                   "produced, joined into one message. Empty after a call that succeeded, and "
                   "never empty after one that did not.\n\n"
                   "The raise already carries this. It is here for the caller that wants to "
                   "report rather than propagate - a ROS node logging a bad "
                   "/robot_description and carrying on, say.")
                .def_property_readonly("diagnostics",
                   [](const URDFLoader& l) { return l.diagnostics(); },
                   "Everything the most recent call had to say: warnings first, then errors, "
                   "each group in the order it was produced.\n\n"
                   "Not the same as `last_error`, and the difference is the point: the warnings "
                   "come from the XACRO expansion - a redefined macro, an undeclared attribute "
                   "being ignored, a name resolved as an arg rather than a property - and a "
                   "document that produces them still loads. They only ever went to stderr, "
                   "where a script cannot see them and a GUI has nowhere to show them.\n\n"
                   "Cleared at the start of every call, so an empty list after a success means "
                   "there was genuinely nothing to say.");
    }

}// namespace threepp_py
