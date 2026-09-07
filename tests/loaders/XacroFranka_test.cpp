// Second corpus gate for the xacro engine, after the Universal Robots one. franka's
// descriptions are written differently enough to have found five separate gaps in a row:
// substitutions nested inside expressions, YAML anchors, macro defaults spelled with '=',
// quoted defaults, dotted access into what load_yaml returned â€” and, in the two-armed
// robots, list slices and the same utils file included once per arm. The clone is not
// vendored, so these cases skip unless THREEPP_FRANKA_DESCRIPTION points at one.

#include <catch2/catch_test_macros.hpp>

#include "UrdfStructure.hpp"

#include "threepp/loaders/Xacro.hpp"

#include <cstdlib>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>

using namespace threepp;
using namespace threepp::xacro;

namespace {

    constexpr const char* skipReason =
            "set THREEPP_FRANKA_DESCRIPTION to a franka_description clone to run this test";

    std::optional<std::filesystem::path> frankaRoot() {

        const char* env = std::getenv("THREEPP_FRANKA_DESCRIPTION");
        if (!env || !*env) return std::nullopt;

        std::filesystem::path root(env);
        std::error_code ec;
        if (!std::filesystem::exists(root / "robots", ec)) return std::nullopt;

        return root;
    }

    ProcessResult expandFranka(const std::filesystem::path& root, const std::filesystem::path& file,
                               const std::map<std::string, std::string>& args = {}) {

        Processor processor;
        processor.addPackagePath("franka_description", root);
        processor.setArgs(args);

        return processor.processFile(root / file);
    }

    std::string requireOk(const ProcessResult& result) {

        const std::string why = result.errors.empty() ? std::string{} : result.errors.front();
        INFO(why);
        REQUIRE(result.ok);
        return result.xml;
    }

    bool contains(const std::string& haystack, const std::string& needle) {

        return haystack.find(needle) != std::string::npos;
    }

    std::size_t count(const std::string& haystack, const std::string& needle) {

        std::size_t n = 0;
        for (auto i = haystack.find(needle); i != std::string::npos;
             i = haystack.find(needle, i + needle.size())) {
            ++n;
        }
        return n;
    }

}// namespace


TEST_CASE("xacro expands the FR3 arm") {

    const auto root = frankaRoot();
    if (!root) SKIP(skipReason);

    const auto xml = requireOk(expandFranka(*root, "robots/fr3/fr3.urdf.xacro"));

    // Nothing is asserted about the prefix on a name: whether the links come out as
    // `link0` or `fr3_link0` is franka's own argument to have with itself, and the
    // description reads that off `no_prefix` in a ternary. What has to hold is that the
    // arm is all there and nothing was left unexpanded.
    REQUIRE_FALSE(contains(xml, "xacro"));
    REQUIRE_FALSE(contains(xml, "$("));
    REQUIRE_FALSE(contains(xml, "${"));

    REQUIRE(contains(xml, "link0\""));
    REQUIRE(count(xml, "type=\"revolute\"") == 7);// an FR3 is a seven-axis arm

    // Whichever way that ternary goes, both ends of every joint have to name a link this
    // document declares. The arm derives its prefix from `no_prefix` while the hand derives
    // one from `robot_type` alone, so the two only agree when `$(arg no_prefix)` - the text
    // "false" - is read as the boolean it spells. Read as a truthy string it expanded the
    // arm bare, the hand still asked for `fr3_link8`, and every assertion above still
    // passed with the whole gripper hanging off nothing.
    const auto dangling = urdf_structure::danglingJointEndpoints(xml);
    INFO(urdf_structure::joined(dangling));
    REQUIRE(dangling.empty());

    // The inertias come out of inertials.yaml, read through
    // ${xacro.load_yaml('$(find franka_description)/...')} and reached by dotted name.
    REQUIRE(count(xml, "<inertia ") >= 7);
    REQUIRE(contains(xml, "package://franka_description/meshes"));
}

TEST_CASE("xacro expands the franka hand, whose inertias are anchored yaml") {

    const auto root = frankaRoot();
    if (!root) SKIP(skipReason);

    const auto xml = requireOk(expandFranka(*root, "end_effectors/franka_hand/franka_hand.urdf.xacro"));

    REQUIRE(contains(xml, "hand\""));

    // leftfinger carries the anchor in inertials.yaml and rightfinger is the alias `*finger`,
    // so whatever the two are called, they have to come out weighing the same.
    const auto massOf = [&xml](const std::string& finger) {
        const auto at = xml.find(finger + "\"");
        REQUIRE(at != std::string::npos);
        const auto mass = xml.find("<mass", at);
        REQUIRE(mass != std::string::npos);
        return xml.substr(mass, xml.find('>', mass) - mass);
    };

    REQUIRE(massOf("leftfinger") == massOf("rightfinger"));

    // `rpy:='0 0 0'` is a default whose quotes group it; they are not part of the value.
    REQUIRE_FALSE(contains(xml, "rpy=\"'"));
}

TEST_CASE("xacro expands the two-armed franka, which slices and re-includes") {

    const auto root = frankaRoot();
    if (!root) SKIP(skipReason);

    // The .urdf.xacro is the robot; the .xacro beside it only defines the macro it calls,
    // and expanding that on its own is an empty document by design.
    const auto file = std::filesystem::path("robots") / "mobile_fr3_duo_v0_2" / "mobile_fr3_duo_v0_2.urdf.xacro";
    std::error_code ec;
    if (!std::filesystem::exists(*root / file, ec)) SKIP("this clone has no mobile_fr3_duo_v0_2");

    const auto result = expandFranka(*root, file);
    const auto xml = requireOk(result);

    // Two arms on a mobile base: fourteen arm axes and the spine.
    REQUIRE(count(xml, "type=\"revolute\"") == 15);
    REQUIRE(count(xml, "<link ") > 50);
    REQUIRE_FALSE(contains(xml, "${"));
    REQUIRE_FALSE(contains(xml, "$("));

    const auto dangling = urdf_structure::danglingJointEndpoints(xml);
    INFO(urdf_structure::joined(dangling));
    REQUIRE(dangling.empty());

    // A shared utils file lands once per branch of the robot, and two different files here
    // really do define an inertia-cylinder between them. The disagreement is worth saying;
    // saying it again every time the includes take another turn is not.
    std::set<std::string> seen;
    for (const auto& warning : result.warnings) {
        if (!contains(warning, "redefined")) continue;
        INFO(warning);
        REQUIRE(seen.insert(warning).second);
    }
}

