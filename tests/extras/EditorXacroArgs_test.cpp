// The xacro arguments a robot was imported with, and the promise that it is
// never rebuilt without them.
//
// A description that takes arguments is not one robot: `ur_type:=ur5e` and
// `ur_type:=ur10e` are different machines out of the same file. The editor asks
// once, at import, and the answer has to survive everything that rebuilds the
// robot afterwards — a play/stop cycle, a scene reload, a linked-asset
// re-import — or a saved UR5e quietly comes back as something else.
//
// PhysX-free: RobotConfig, URDFLoader and ObjectLoader only, so it runs
// everywhere.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/SceneSnapshot.hpp"
#include "threepp/loaders/AssetSource.hpp"
#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/loaders/Xacro.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/scenes/Scene.hpp"

#include <any>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace threepp;
using namespace threepp::editor;

namespace {

    struct TempDir {

        std::filesystem::path path;

        explicit TempDir(const std::string& tag) {

            static int counter = 0;
            path = std::filesystem::temp_directory_path() /
                   ("threepp_xacroargs_" + tag + "_" + std::to_string(++counter));
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
            std::filesystem::create_directories(path);
        }

        ~TempDir() {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
        }

        TempDir(const TempDir&) = delete;
        TempDir& operator=(const TempDir&) = delete;
    };

    // An arm whose JOINT COUNT is an argument. Nothing else in this file would
    // notice a lost argument; a robot with the wrong number of DOF is impossible
    // to miss, and it is the shape of the real failure — a UR5e rebuilt as
    // whatever ur.urdf.xacro defaults to.
    const char* kArmXacro = R"XML(<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="arm">
  <xacro:arg name="dof" default="1"/>
  <xacro:property name="dof" value="$(arg dof)"/>
  <link name="base_link"><visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual></link>
  <link name="link_1"><visual><geometry><box size="0.1 0.1 0.4"/></geometry></visual></link>
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/><child link="link_1"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
  <xacro:if value="${dof >= 2}">
    <link name="link_2"><visual><geometry><box size="0.1 0.1 0.4"/></geometry></visual></link>
    <joint name="joint_2" type="revolute">
      <parent link="link_1"/><child link="link_2"/>
      <origin xyz="0 0 0.4"/><axis xyz="0 1 0"/>
      <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
    </joint>
  </xacro:if>
</robot>
)XML";

    std::filesystem::path writeArm(const TempDir& dir) {

        const auto file = dir.path / "arm.urdf.xacro";
        std::ofstream out(file, std::ios::binary);
        out << kArmXacro;
        return file;
    }

    std::string readString(const Object3D& node, const std::string& key) {

        const auto it = node.userData.find(key);
        if (it == node.userData.end() || it->second.type() != typeid(std::string)) return {};
        return std::any_cast<const std::string&>(it->second);
    }

    bool hasKey(const Object3D& node, const std::string& key) {

        return node.userData.count(key) > 0;
    }

    // What every rebuild path does: the config says which file and which
    // arguments, and the loader is told both.
    std::shared_ptr<Robot> rebuild(const RobotConfig& config) {

        URDFLoader loader;
        loader.setArgs(config.argMap());
        return loader.load(config.urdf);
    }

}// namespace


TEST_CASE("RobotConfig round-trips xacro arguments through userData") {

    Object3D object;

    RobotConfig config;
    config.urdf = "C:\\models\\ur\\ur.urdf.xacro";

    SECTION("values the packed key=value configs could not carry") {

        // Every delimiter the other configs reserve, plus the case that made
        // RobotConfig use plain entries in the first place: a Windows path.
        config.xacroArgs = {
                {"joint_limit_params", "C:\\ros\\ur_description\\config\\ur5e\\joint_limits.yaml"},
                {"initial_positions", "shoulder=0.5;elbow=-1.25"},
                {"description", "a robot, with a comma"},
                {"tf_prefix", "left arm "},
                {"empty", ""},
        };
        config.write(object);

        const auto back = RobotConfig::read(object);
        REQUIRE(back);
        CHECK(back->xacroArgs == config.xacroArgs);
    }

    SECTION("declaration order is the stored order") {

        config.xacroArgs = {{"z_last", "1"}, {"a_first", "2"}, {"m_middle", "3"}};
        config.write(object);

        const auto back = RobotConfig::read(object);
        REQUIRE(back);
        REQUIRE(back->xacroArgs.size() == 3);
        CHECK(back->xacroArgs[0].first == "z_last");
        CHECK(back->xacroArgs[1].first == "a_first");
        CHECK(back->xacroArgs[2].first == "m_middle");
    }

    SECTION("write -> read -> write is stable") {

        config.xacroArgs = {{"ur_type", "ur5e"}, {"name", "ur"}};
        config.write(object);

        Object3D second;
        RobotConfig::read(object)->write(second);

        REQUIRE(object.userData.size() == second.userData.size());
        for (const auto& [key, value] : object.userData) {
            REQUIRE(second.userData.count(key));
            CHECK(readString(object, key) == readString(second, key));
        }
    }

    SECTION("a removed argument leaves no orphan behind") {

        config.xacroArgs = {{"ur_type", "ur5e"}, {"tf_prefix", "left_"}};
        config.write(object);
        REQUIRE(hasKey(object, std::string(xacro::argValueUserDataPrefix) + "tf_prefix"));

        config.xacroArgs = {{"ur_type", "ur5e"}};
        config.write(object);

        // The name list no longer mentions it AND the value is gone — if it were
        // only the former, a later write that re-listed the name would hand back
        // the stale value.
        CHECK_FALSE(hasKey(object, std::string(xacro::argValueUserDataPrefix) + "tf_prefix"));
        CHECK(readString(object, xacro::argsUserDataKey) == "ur_type");

        const auto back = RobotConfig::read(object);
        REQUIRE(back);
        REQUIRE(back->xacroArgs.size() == 1);
        CHECK(back->xacroArgs.front().first == "ur_type");
    }

    SECTION("no arguments, no trace in the document") {

        config.write(object);

        CHECK_FALSE(hasKey(object, xacro::argsUserDataKey));
        for (const auto& [key, value] : object.userData) {
            CHECK(key.rfind(xacro::argValueUserDataPrefix, 0) != 0);
        }
        CHECK(RobotConfig::read(object)->xacroArgs.empty());
    }

    SECTION("erase takes the argument entries with it") {

        config.xacroArgs = {{"ur_type", "ur5e"}};
        config.write(object);

        RobotConfig::erase(object);
        CHECK_FALSE(hasKey(object, xacro::argsUserDataKey));
        CHECK_FALSE(hasKey(object, std::string(xacro::argValueUserDataPrefix) + "ur_type"));
    }

    SECTION("a value key without a name is ignored, not read as empty") {

        config.xacroArgs = {{"ur_type", "ur5e"}};
        config.write(object);
        object.userData.erase(std::string(xacro::argValueUserDataPrefix) + "ur_type");

        CHECK(RobotConfig::read(object)->xacroArgs.empty());
    }
}


TEST_CASE("an argument that changes the robot survives re-articulation") {

    const TempDir dir("rebuild");
    const auto file = writeArm(dir);

    // What the file does when nobody says anything — the thing an import that
    // loses its arguments would silently fall back to.
    URDFLoader plain;
    const auto asDeclared = plain.load(file);
    REQUIRE(asDeclared);
    REQUIRE(asDeclared->numDOF() == 1);

    // The import, with the argument the dialog collected.
    URDFLoader loader;
    loader.setArgs({{"dof", "2"}});
    auto robot = loader.load(file);
    REQUIRE(robot);
    REQUIRE(robot->numDOF() == 2);

    RobotConfig config;
    config.urdf = file.string();
    config.joints = robot->jointValues();
    config.xacroArgs = {{"dof", "2"}};
    config.write(*robot);

    SECTION("straight from the config") {

        const auto rebuilt = rebuild(*RobotConfig::read(*robot));
        REQUIRE(rebuilt);
        CHECK(rebuilt->numDOF() == 2);
    }

    SECTION("through a play/stop round trip, which is how the editor rebuilds") {

        Scene scene;
        robot->name = "arm";
        scene.add(robot);

        SceneSnapshot snapshot;
        std::string error;
        REQUIRE(snapshot.capture(scene, &error));
        const auto restored = snapshot.restore(&error);
        REQUIRE(restored);

        // Stop re-articulates whatever came back carrying a urdf reference.
        Object3D* placeholder = nullptr;
        restored->traverse([&](Object3D& node) {
            if (!placeholder && !node.as<Robot>() && RobotConfig::read(node)) placeholder = &node;
        });
        REQUIRE(placeholder != nullptr);

        const auto config2 = *RobotConfig::read(*placeholder);
        REQUIRE(config2.xacroArgs.size() == 1);
        CHECK(config2.xacroArgs.front().second == "2");

        const auto fresh = rebuild(config2);
        REQUIRE(fresh);
        CHECK(fresh->numDOF() == 2);

        // And after the transplant the arguments are still on the live robot,
        // ready for the NEXT rebuild — a play/stop cycle is not a one-shot.
        transplantRobot(*placeholder, fresh);
        const auto again = RobotConfig::read(*fresh);
        REQUIRE(again);
        CHECK(again->xacroArgs == config2.xacroArgs);
    }
}


TEST_CASE("a linked .xacro asset is re-imported with the arguments it was imported with") {

    // The third rebuild path: the document stores a path instead of the
    // subtree, and ObjectLoader re-runs the loader on open. It knows nothing
    // about RobotConfig — it reads the same two userData entries by the key
    // names Xacro.hpp owns.

    const TempDir dir("linked");
    const auto file = writeArm(dir);

    URDFLoader loader;
    loader.setArgs({{"dof", "2"}});
    auto robot = loader.load(file);
    REQUIRE(robot);
    REQUIRE(robot->numDOF() == 2);

    RobotConfig config;
    config.urdf = file.string();
    config.joints = robot->jointValues();
    config.xacroArgs = {{"dof", "2"}};
    config.write(*robot);

    setAssetSource(*robot, std::filesystem::weakly_canonical(file));

    auto scene = Scene::create();
    scene->add(robot);

    ObjectExporterOptions referenced;
    referenced.models = ModelStorage::Reference;
    referenced.resourcePath = dir.path;

    ObjectExporter exporter;
    const auto text = exporter.toJson(*scene, referenced);
    // The premise: the geometry is NOT in the document, so what comes back is
    // whatever the loader builds from the file.
    REQUIRE(text.find("threeppAsset") != std::string::npos);

    ObjectLoader objects;
    objects.setResourcePath(dir.path);
    const auto parsed = objects.parse(text);

    REQUIRE(parsed);
    REQUIRE(parsed->children.size() == 1);
    auto* restored = parsed->children.front()->as<Robot>();
    REQUIRE(restored != nullptr);
    CHECK(restored->numDOF() == 2);
    CHECK(RobotConfig::read(*restored)->xacroArgs == config.xacroArgs);
}
