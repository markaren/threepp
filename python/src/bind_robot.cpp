// URDF robots: URDFLoader parses a URDF/xacro file (or XML string) into a Robot
// — an articulated Object3D with forward kinematics and joint introspection.
// Robot derives non-virtually from Object3D, so it inherits the Object3D base
// bindings (position, traverse, matrix_world, ...) directly.
#include "bindings.hpp"

#include <pybind11/stl.h>

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
                .def("show_colliders", &Robot::showColliders, py::arg("flag"));

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
