// Corpus gate for the xacro engine: the Universal Robots ROS 2 description is the
// reference workload — nested includes, macros with ^ defaults, load_yaml with custom
// tags, dict subscripting, scope="parent" writes and $(find). The clone is not vendored,
// so these cases skip unless THREEPP_UR_DESCRIPTION points at one.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/loaders/Xacro.hpp"
#include "threepp/loaders/xacro/YamlLite.hpp"
#include "threepp/objects/Robot.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <string>

using namespace threepp;
using namespace threepp::xacro;

namespace {

    constexpr const char* skipReason =
            "set THREEPP_UR_DESCRIPTION to a Universal_Robots_ROS2_Description clone to run this test";

    std::optional<std::filesystem::path> urRoot() {

        const char* env = std::getenv("THREEPP_UR_DESCRIPTION");
        if (!env || !*env) return std::nullopt;

        std::filesystem::path root(env);
        std::error_code ec;
        if (!std::filesystem::exists(root / "urdf" / "ur.urdf.xacro", ec)) return std::nullopt;

        return root;
    }

    ProcessResult expandUr(const std::filesystem::path& root, const std::string& file,
                           const std::map<std::string, std::string>& args) {

        Processor processor;
        processor.addPackagePath("ur_description", root);
        processor.setArgs(args);

        return processor.processFile(root / "urdf" / file);
    }

    std::string requireOk(const ProcessResult& result) {

        const std::string why = result.errors.empty() ? std::string{} : result.errors.front();
        INFO(why);
        REQUIRE(result.ok);
        REQUIRE(result.errors.empty());
        return result.xml;
    }

    bool contains(const std::string& haystack, const std::string& needle) {

        return haystack.find(needle) != std::string::npos;
    }

    std::size_t count(const std::string& haystack, const std::string& needle) {

        std::size_t n = 0;
        for (std::size_t i = haystack.find(needle); i != std::string::npos;
             i = haystack.find(needle, i + needle.size())) {
            ++n;
        }
        return n;
    }

    // The output is flat XML on one line per element, so a named element and its attributes
    // can be picked out without dragging a parser into the test binary.
    std::string blockOf(const std::string& xml, const std::string& tag, const std::string& name) {

        const auto b = xml.find("<" + tag + " name=\"" + name + "\"");
        if (b == std::string::npos) return {};

        const auto e = xml.find("</" + tag + ">", b);
        return xml.substr(b, e == std::string::npos ? std::string::npos : e - b);
    }

    std::string elementIn(const std::string& block, const std::string& tag) {

        const auto b = block.find("<" + tag);
        if (b == std::string::npos) return {};

        const auto e = block.find('>', b);
        return block.substr(b, e == std::string::npos ? std::string::npos : e - b + 1);
    }

    std::optional<std::string> attributeOf(const std::string& element, const std::string& attribute) {

        const auto b = element.find(" " + attribute + "=\"");
        if (b == std::string::npos) return std::nullopt;

        const auto s = b + attribute.size() + 3;
        const auto e = element.find('"', s);
        if (e == std::string::npos) return std::nullopt;

        return element.substr(s, e - s);
    }

    double numberOf(const std::string& element, const std::string& attribute) {

        const auto text = attributeOf(element, attribute);
        INFO("attribute '" << attribute << "' of " << element);
        REQUIRE(text.has_value());
        return std::stod(*text);
    }

}// namespace


TEST_CASE("xacro expands the UR5e description") {

    const auto root = urRoot();
    if (!root) SKIP(skipReason);

    const auto xml = requireOk(expandUr(*root, "ur.urdf.xacro", {{"name", "ur"}, {"ur_type", "ur5e"}}));

    const auto robot = xml.substr(xml.find("<robot"), xml.find('>', xml.find("<robot")) - xml.find("<robot") + 1);
    REQUIRE(attributeOf(robot, "name") == "ur");
    REQUIRE_FALSE(contains(xml, "xacro"));

    for (const std::string link : {"world", "base_link", "base_link_inertia", "shoulder_link",
                                   "upper_arm_link", "forearm_link", "wrist_1_link",
                                   "wrist_2_link", "wrist_3_link", "tool0", "flange", "ft_frame"}) {
        INFO("link " << link);
        REQUIRE(contains(xml, "<link name=\"" + link + "\""));
    }

    for (const std::string joint : {"shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"}) {
        INFO("joint " << joint);
        REQUIRE(contains(xml, "<joint name=\"" + joint + "\""));
    }

    // every ur5e arm joint has position limits, so all six are revolute
    REQUIRE(count(xml, "type=\"revolute\"") == 6);
}

TEST_CASE("xacro carries the UR5e joint limits, !degrees and all") {

    const auto root = urRoot();
    if (!root) SKIP(skipReason);

    const auto xml = requireOk(expandUr(*root, "ur.urdf.xacro", {{"name", "ur"}, {"ur_type", "ur5e"}}));

    const Value config = loadYamlFile(*root / "config" / "ur5e" / "joint_limits.yaml");
    const Value& expected = config.asDict().at("joint_limits").asDict().at("shoulder_pan_joint");

    const std::string limit = elementIn(blockOf(xml, "joint", "shoulder_pan_joint"), "limit");
    REQUIRE_FALSE(limit.empty());

    REQUIRE(numberOf(limit, "lower") == Catch::Approx(expected.asDict().at("min_position").asNumber()).margin(1e-12));
    REQUIRE(numberOf(limit, "upper") == Catch::Approx(expected.asDict().at("max_position").asNumber()).margin(1e-12));
    REQUIRE(numberOf(limit, "effort") == Catch::Approx(expected.asDict().at("max_effort").asNumber()).margin(1e-12));
    REQUIRE(numberOf(limit, "velocity") == Catch::Approx(expected.asDict().at("max_velocity").asNumber()).margin(1e-12));

    // the yaml says `!degrees 360.0`, which is a full turn in radians
    constexpr double pi = 3.14159265358979323846;
    REQUIRE(numberOf(limit, "upper") == Catch::Approx(2.0 * pi).margin(1e-12));
    REQUIRE(numberOf(limit, "lower") == Catch::Approx(-2.0 * pi).margin(1e-12));
    REQUIRE(numberOf(limit, "velocity") == Catch::Approx(pi).margin(1e-12));
    REQUIRE(numberOf(limit, "effort") == Catch::Approx(150.0).margin(1e-12));
}

TEST_CASE("xacro writes UR mesh URIs, package:// by default and file:// on request") {

    const auto root = urRoot();
    if (!root) SKIP(skipReason);

    const auto relative = requireOk(expandUr(*root, "ur.urdf.xacro", {{"name", "ur"}, {"ur_type", "ur5e"}}));

    const std::string mesh = elementIn(blockOf(relative, "link", "base_link_inertia"), "mesh");
    REQUIRE(attributeOf(mesh, "filename") == "package://ur_description/meshes/ur5e/visual/base.dae");

    const auto absolute = requireOk(expandUr(*root, "ur.urdf.xacro",
                                             {{"name", "ur"}, {"ur_type", "ur5e"}, {"force_abs_paths", "true"}}));

    const auto file = attributeOf(elementIn(blockOf(absolute, "link", "base_link_inertia"), "mesh"), "filename");
    REQUIRE(file.has_value());
    REQUIRE(file->rfind("file://", 0) == 0);

    const std::filesystem::path onDisk(file->substr(std::string("file://").size()));
    INFO(onDisk.string());
    REQUIRE(std::filesystem::exists(onDisk));
    REQUIRE(std::filesystem::equivalent(onDisk, *root / "meshes" / "ur5e" / "visual" / "base.dae"));
}

TEST_CASE("xacro emits UR safety controllers when safety_limits is on") {

    const auto root = urRoot();
    if (!root) SKIP(skipReason);

    const auto off = requireOk(expandUr(*root, "ur.urdf.xacro", {{"name", "ur"}, {"ur_type", "ur5e"}}));
    REQUIRE_FALSE(contains(off, "<safety_controller"));

    const auto on = requireOk(expandUr(*root, "ur.urdf.xacro",
                                       {{"name", "ur"}, {"ur_type", "ur5e"}, {"safety_limits", "true"}}));
    REQUIRE(count(on, "<safety_controller") == 6);

    const std::string safety = elementIn(blockOf(on, "joint", "shoulder_pan_joint"), "safety_controller");
    constexpr double pi = 3.14159265358979323846;
    REQUIRE(numberOf(safety, "soft_upper_limit") == Catch::Approx(2.0 * pi - 0.15).margin(1e-9));
    REQUIRE(numberOf(safety, "k_position") == Catch::Approx(20.0).margin(1e-12));
}

TEST_CASE("xacro expands a non-e-series UR") {

    const auto root = urRoot();
    if (!root) SKIP(skipReason);

    const auto xml = requireOk(expandUr(*root, "ur.urdf.xacro", {{"name", "ur"}, {"ur_type", "ur3"}}));

    REQUIRE(contains(xml, "package://ur_description/meshes/ur3/visual/base.dae"));
    REQUIRE(contains(xml, "<link name=\"wrist_3_link\""));

    // ur3's wrist_3 has no position limits, so that joint comes out continuous
    REQUIRE(count(xml, "type=\"revolute\"") == 5);
    REQUIRE(count(xml, "type=\"continuous\"") == 1);
}

TEST_CASE("xacro expands the mocked UR, dict defaults and all") {

    const auto root = urRoot();
    if (!root) SKIP(skipReason);

    const auto xml = requireOk(expandUr(*root, "ur_mocked.urdf.xacro", {{"name", "ur"}, {"ur_type", "ur5e"}}));

    REQUIRE(contains(xml, "<ros2_control name=\"ur\""));
    REQUIRE(contains(xml, "mock_components/GenericSystem"));

    // the initial positions come from config/initial_positions.yaml through load_yaml; the
    // same joint names appear in the kinematic tree first, so look inside <ros2_control>
    const auto control = xml.find("<ros2_control");
    REQUIRE(control != std::string::npos);

    const Value positions = loadYamlFile(*root / "config" / "initial_positions.yaml");
    const std::string joint = blockOf(xml.substr(control), "joint", "shoulder_lift_joint");
    REQUIRE(contains(joint, ">" + positions.asDict().at("shoulder_lift_joint").toString() + "<"));
}

TEST_CASE("URDFLoader loads the UR5e description straight from xacro") {

    const auto root = urRoot();
    if (!root) SKIP(skipReason);

    URDFLoader loader;
    loader.addPackagePath("ur_description", *root);
    loader.setArgs({{"name", "ur"}, {"ur_type", "ur5e"}});

    const auto robot = loader.load(*root / "urdf" / "ur.urdf.xacro");
    REQUIRE(robot);
    REQUIRE(robot->numDOF() == 6);

    const auto info = robot->getArticulatedJointInfo();
    REQUIRE(info.size() == 6);
    REQUIRE(info.front().name == "shoulder_pan_joint");
}
