// Re-articulation transplant: descendant userData survives.
//
// A document round trip flattens a Robot into a plain Object3D subtree; the
// editor re-imports the URDF and transplants a live Robot over the frozen
// placeholder. The transplant used to keep only the ROOT's userData, silently
// dropping a sensor (or a physics entry) authored on a LINK. transplantRobot now
// preserves each descendant's userData onto the same-named node of the fresh
// robot, which is what makes "an encoder on a link" a durable authoring choice.
//
// PhysX-free: this is a URDFLoader + RobotConfig test, so it runs everywhere.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/scenes/Scene.hpp"

#include <any>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

using namespace threepp;
using namespace threepp::editor;

namespace {

    // A two-link arm: a base and one link on a revolute joint. Primitive geometry
    // only, so it needs no external mesh files.
    const char* kUrdf = R"(
        <robot name="arm">
          <link name="base_link">
            <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
          </link>
          <link name="upper_link">
            <visual><geometry><box size="0.1 0.5 0.1"/></geometry></visual>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base_link"/>
            <child link="upper_link"/>
            <origin xyz="0 0 0.2" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-1.5" upper="1.5"/>
          </joint>
        </robot>)";

    std::shared_ptr<Robot> loadArm() {
        URDFLoader loader;
        return loader.parse(std::filesystem::temp_directory_path(), kUrdf);
    }

    std::string readString(const Object3D& node, const char* key) {
        const auto it = node.userData.find(key);
        if (it == node.userData.end() || it->second.type() != typeid(std::string)) return {};
        return std::any_cast<const std::string&>(it->second);
    }

    Object3D* findByName(Object3D& root, const std::string& name) {
        Object3D* found = nullptr;
        root.traverse([&](Object3D& node) {
            if (!found && node.name == name) found = &node;
        });
        return found;
    }

}// namespace


TEST_CASE("transplantRobot preserves userData authored on a link, not just the root") {

    Scene scene;

    // The placeholder: the same arm, standing in for the flattened document node.
    // Author a RobotConfig on the root and a sensor entry on a LINK.
    auto placeholder = loadArm();
    REQUIRE(placeholder);
    placeholder->name = "MyArm";

    RobotConfig robotConfig;
    robotConfig.urdf = "arm.urdf";
    robotConfig.joints = std::vector<float>(placeholder->numDOF(), 0.f);
    if (!robotConfig.joints.empty()) robotConfig.joints[0] = 0.4f;
    robotConfig.write(*placeholder);

    // A sensor authored on the link, the case the old transplant dropped.
    auto* link = findByName(*placeholder, "upper_link");
    REQUIRE(link != nullptr);
    link->userData["sensor"] = std::string("type=encoder;joint=shoulder");
    // And something on the root, to prove the root path still works.
    placeholder->userData["marker"] = std::string("root-data");

    scene.add(placeholder);
    const std::string keptUuid = placeholder->uuid;

    // The fresh robot the editor would build from the same URDF.
    auto fresh = loadArm();
    REQUIRE(fresh);

    transplantRobot(*placeholder, fresh);

    // The fresh robot is now in the scene under the placeholder's identity.
    auto* inScene = findByUuid(scene, keptUuid);
    REQUIRE(inScene != nullptr);
    auto* robot = inScene->as<Robot>();
    REQUIRE(robot != nullptr);
    CHECK(robot->name == "MyArm");

    // Root userData survived (both the RobotConfig and the marker).
    CHECK(readString(*robot, "marker") == "root-data");
    CHECK(RobotConfig::read(*robot).has_value());

    // The link's sensor entry survived onto the same-named node — the fix.
    auto* freshLink = findByName(*robot, "upper_link");
    REQUIRE(freshLink != nullptr);
    CHECK(readString(*freshLink, "sensor") == "type=encoder;joint=shoulder");

    // And the authored pose was re-applied.
    if (robot->numDOF() > 0) {
        CHECK(std::abs(robot->getJointValue(0) - 0.4f) < 1e-3f);
    }
}

TEST_CASE("transplantRobot reports a link name that no longer resolves") {

    Scene scene;

    auto placeholder = loadArm();
    REQUIRE(placeholder);
    RobotConfig robotConfig;
    robotConfig.urdf = "arm.urdf";
    robotConfig.write(*placeholder);

    // Author userData on a node whose name the fresh robot will not have.
    auto* link = findByName(*placeholder, "upper_link");
    REQUIRE(link != nullptr);
    link->name = "renamed_link";
    link->userData["sensor"] = std::string("type=forcetorque;joint=shoulder");

    scene.add(placeholder);

    auto fresh = loadArm();
    REQUIRE(fresh);

    bool reported = false;
    transplantRobot(*placeholder, fresh, [&](const std::string& m) {
        if (m.find("renamed_link") != std::string::npos) reported = true;
    });

    // The name did not resolve, so it was reported rather than lost silently.
    CHECK(reported);
}
