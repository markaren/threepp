// Re-articulation: what the editor does to a robot on Stop, and what has to
// survive it.
//
// The rule these cases pin is one sentence: A PLAY/STOP CYCLE MUST NOT REBUILD
// THE ROBOT'S SUBTREE FROM ITS URDF. It used to, and everything authored into
// the robot — a camera bolted to the wrist, a sensor on the unnamed mesh a
// viewport click lands on, a collider deleted — was deleted with it, on every
// Stop.
//
// Two mechanisms, tested here in that order:
//
//   1. Documents carry the joint table (ObjectExporter's "threeppRobot" block),
//      so the snapshot round trip hands back a LIVE Robot and never reads the
//      file at all. This is the path every current document takes.
//   2. transplantRobot, for documents written before that block existed. It
//      re-imports the URDF for its joint table ONLY and moves that onto the
//      subtree the document carries, rather than planting the file's subtree
//      over it.
//
// PhysX-free: this is a URDFLoader + RobotConfig test, so it runs everywhere.

#include <catch2/catch_test_macros.hpp>

#include "threepp/extras/editor/EditorCommands.hpp"
#include "threepp/extras/editor/RobotConfig.hpp"
#include "threepp/extras/editor/SceneSnapshot.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/scenes/Scene.hpp"

#include <any>
#include <cmath>
#include <cstddef>
#include <filesystem>
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


TEST_CASE("a play/stop round trip hands back a live robot, without reading the URDF") {

    // The premise of the whole fix: the document carries the joint table, so
    // Stop restores an articulated robot from the document alone. The urdf path
    // here names a file that does not exist — if anything on this path still
    // reads it, this case says so.
    Scene scene;

    auto robot = loadArm();
    REQUIRE(robot);
    robot->name = "arm";
    REQUIRE(robot->numDOF() == 1);

    robot->setJointValue(0, 0.4f);

    RobotConfig robotConfig;
    robotConfig.urdf = (std::filesystem::temp_directory_path() / "no_such_robot.urdf").string();
    robotConfig.joints = robot->jointValues();
    robotConfig.write(*robot);

    scene.add(robot);
    const auto keptUuid = robot->uuid;
    const auto reference = robot->computeEndEffectorTransform({0.9f});

    SceneSnapshot snapshot;
    std::string error;
    REQUIRE(snapshot.capture(scene, &error));
    const auto restored = snapshot.restore(&error);
    REQUIRE(restored);

    // The joint table really is in the document, not reconstructed from the file.
    CHECK(snapshot.json().find("\"threeppRobot\"") != std::string::npos);

    auto* found = findByUuid(*restored, keptUuid);
    REQUIRE(found != nullptr);

    // Live, not a frozen Object3D subtree.
    auto* live = found->as<Robot>();
    REQUIRE(live != nullptr);
    CHECK(live->numDOF() == 1);

    // The pose it was saved in, and the limits that constrain it.
    CHECK(std::abs(live->getJointValue(0) - 0.4f) < 1e-4f);
    CHECK(std::abs(live->getJointRange(0).min - (-1.5f)) < 1e-4f);
    CHECK(std::abs(live->getJointRange(0).max - 1.5f) < 1e-4f);

    // The joint node came back in its DRIVEN pose, not re-zeroed and not with
    // the saved angle baked into its rest frame and applied twice: analytic FK
    // through the restored table matches the original's, element for element.
    const auto after = live->computeEndEffectorTransform({0.9f});
    for (unsigned i = 0; i < 16; ++i) {
        CHECK(std::abs(after.elements[i] - reference.elements[i]) < 1e-4f);
    }

    // And it drives.
    live->setJointValue(0, -0.6f);
    CHECK(std::abs(live->getJointValue(0) - (-0.6f)) < 1e-4f);
}

TEST_CASE("repeated play/stop cycles neither drift nor accumulate") {

    // A cycle is not a one-shot: the pose is re-derived from the stored rest
    // pose every time, so a rest pose that drank the driven one would compound
    // visibly by the third Stop, and a table appended to rather than rebuilt
    // would double the DOF count.
    auto robot = loadArm();
    REQUIRE(robot);
    robot->name = "arm";
    robot->setJointValue(0, 0.4f);

    RobotConfig robotConfig;
    robotConfig.urdf = "arm.urdf";
    robotConfig.joints = robot->jointValues();
    robotConfig.write(*robot);

    auto* link = findByName(*robot, "upper_link");
    REQUIRE(link != nullptr);
    auto authored = Group::create();
    authored->name = "WristCamera";
    link->add(authored);

    const auto reference = robot->computeEndEffectorTransform({0.4f});

    auto stage = Scene::create();
    stage->add(robot);

    for (int cycle = 0; cycle < 3; ++cycle) {

        SceneSnapshot snapshot;
        std::string error;
        REQUIRE(snapshot.capture(*stage, &error));
        stage = snapshot.restore(&error);
        REQUIRE(stage);

        Robot* live = nullptr;
        stage->traverse([&](Object3D& node) {
            if (!live) live = node.as<Robot>();
        });
        REQUIRE(live != nullptr);

        CHECK(live->numDOF() == 1);
        CHECK(std::abs(live->getJointValue(0) - 0.4f) < 1e-4f);
        CHECK(findByName(*stage, "WristCamera") != nullptr);

        const auto now = live->computeEndEffectorTransform({0.4f});
        for (unsigned i = 0; i < 16; ++i) {
            CHECK(std::abs(now.elements[i] - reference.elements[i]) < 1e-4f);
        }
    }
}

TEST_CASE("a restored robot co-owns its joint nodes") {

    // Robot keeps RAW pointers to its joint nodes (origPose_, articulatedJoints_)
    // and relies on its own shared_ptrs to keep them alive. A URDF-built robot
    // owns its links outright; a restored one has to as well, or deleting a link
    // out of the hierarchy — which the editor lets you do — frees a node the
    // joint table still drives.
    Scene scene;

    auto robot = loadArm();
    REQUIRE(robot);
    robot->name = "arm";
    scene.add(robot);

    SceneSnapshot snapshot;
    std::string error;
    REQUIRE(snapshot.capture(scene, &error));
    const auto restored = snapshot.restore(&error);
    REQUIRE(restored);

    auto* link = findByName(*restored, "upper_link");
    REQUIRE(link != nullptr);
    const std::weak_ptr<Object3D> watch = link->shared_from_this();

    // Drop the editor's own reference to it: the hierarchy no longer owns it.
    link->removeFromParent();
    CHECK_FALSE(watch.expired());

    // And the robot still drives without touching freed memory.
    Robot* live = nullptr;
    restored->traverse([&](Object3D& node) {
        if (!live) live = node.as<Robot>();
    });
    REQUIRE(live != nullptr);
    live->setJointValue(0, 0.5f);
    live->showColliders(true);
    CHECK(std::abs(live->getJointValue(0) - 0.5f) < 1e-4f);
}

TEST_CASE("a sensor on an unnamed node survives a play/stop round trip") {

    // The node a viewport click actually lands on inside a robot is the unnamed
    // Mesh under a link's unnamed visual group, so that is where an authored
    // sensor sits — and it used to be thrown away on every Stop, because the
    // rebuild had no way to name the node it belonged to.
    Scene scene;

    auto robot = loadArm();
    REQUIRE(robot);
    robot->name = "arm";

    RobotConfig robotConfig;
    robotConfig.urdf = (std::filesystem::temp_directory_path() / "threepp_transplant_arm.urdf").string();
    robotConfig.joints = robot->jointValues();
    robotConfig.write(*robot);

    auto* link = findByName(*robot, "upper_link");
    REQUIRE(link != nullptr);

    Object3D* mesh = nullptr;
    link->traverse([&](Object3D& node) {
        if (!mesh && node.type() == "Mesh") mesh = &node;
    });
    REQUIRE(mesh != nullptr);
    // The premise: URDFLoader names links and joints, nothing below them.
    REQUIRE(mesh->name.empty());
    mesh->userData["sensor"] = std::string("type=depth;width=160;height=120");

    scene.add(robot);

    // Play captures, Stop restores.
    SceneSnapshot snapshot;
    std::string error;
    REQUIRE(snapshot.capture(scene, &error));
    const auto restored = snapshot.restore(&error);
    REQUIRE(restored);

    std::size_t sensors = 0;
    Object3D* carrier = nullptr;
    restored->traverse([&](Object3D& node) {
        if (readString(node, "sensor").empty()) return;
        ++sensors;
        carrier = &node;
    });

    CHECK(sensors == 1);
    REQUIRE(carrier != nullptr);
    // On the mesh it was authored on, not smeared onto the named link above it.
    CHECK(carrier->name.empty());
    CHECK(carrier->type() == "Mesh");
    CHECK(readString(*carrier, "sensor") == "type=depth;width=160;height=120");
}

TEST_CASE("a child authored under a robot link survives a play/stop round trip") {

    // Authoring INTO an imported robot — a camera on the wrist, a marker on the
    // base — is the natural way to build a robot cell. The nodes have to still be
    // there after Stop, still where they were put, and the robot has to still
    // drive them.
    Scene scene;

    auto robot = loadArm();
    REQUIRE(robot);
    robot->name = "arm";

    RobotConfig robotConfig;
    robotConfig.urdf = (std::filesystem::temp_directory_path() / "threepp_transplant_child.urdf").string();
    robotConfig.joints = robot->jointValues();
    robotConfig.write(*robot);

    auto* link = findByName(*robot, "upper_link");
    REQUIRE(link != nullptr);

    auto authored = Group::create();
    authored->name = "WristCamera";
    // Off the shoulder's axis (0 0 1), or turning the joint would not move it
    // and the check below would pass for the wrong reason.
    authored->position.set(0.3f, 0.f, 0.25f);
    link->add(authored);

    // And one directly on the robot root, the other place a drop lands.
    auto onRoot = Group::create();
    onRoot->name = "ToolMount";
    robot->add(onRoot);

    scene.add(robot);

    SceneSnapshot snapshot;
    std::string error;
    REQUIRE(snapshot.capture(scene, &error));
    const auto restored = snapshot.restore(&error);
    REQUIRE(restored);

    auto* camera = findByName(*restored, "WristCamera");
    REQUIRE(camera != nullptr);
    REQUIRE(camera->parent != nullptr);
    CHECK(camera->parent->name == "upper_link");
    CHECK(camera->position.z == 0.25f);

    CHECK(findByName(*restored, "ToolMount") != nullptr);

    // Not just present: still carried by the joint. Driving the shoulder moves
    // the node bolted to the link it turns.
    auto* live = findByName(*restored, "arm");
    REQUIRE(live != nullptr);
    auto* liveRobot = live->as<Robot>();
    REQUIRE(liveRobot != nullptr);

    restored->updateMatrixWorld(true);
    Vector3 before;
    camera->getWorldPosition(before);

    liveRobot->setJointValue(0, 1.2f);
    restored->updateMatrixWorld(true);
    Vector3 after;
    camera->getWorldPosition(after);

    CHECK(before.distanceTo(after) > 1e-3f);
}

// --------------------------------------------------------------------------
// Documents written BEFORE the articulation extension: the subtree is there,
// the joint table is not, and transplantRobot re-imports the URDF for it.

namespace {

    // What such a document parses back into: the same hierarchy, as a plain
    // Object3D. Object3D::clone() produces exactly that — Robot does not
    // override createDefault(), so cloning one drops the articulation and keeps
    // the tree, which is precisely what the old format did.
    std::shared_ptr<Object3D> freeze(Robot& robot) {

        auto frozen = robot.clone();
        frozen->uuid = robot.uuid;
        return frozen;
    }

}// namespace

TEST_CASE("transplantRobot articulates the document's subtree, not the file's") {

    Scene scene;

    auto authoredIn = loadArm();
    REQUIRE(authoredIn);
    authoredIn->name = "MyArm";

    RobotConfig robotConfig;
    robotConfig.urdf = "arm.urdf";
    robotConfig.joints = std::vector<float>(authoredIn->numDOF(), 0.f);
    if (!robotConfig.joints.empty()) robotConfig.joints[0] = 0.4f;
    robotConfig.write(*authoredIn);

    // Everything a rebuild from the file would wipe: a child on a link, an entry
    // on a link, an entry on the root, and a node deleted out of the tree.
    auto* link = findByName(*authoredIn, "upper_link");
    REQUIRE(link != nullptr);
    link->userData["sensor"] = std::string("type=encoder;joint=shoulder");
    authoredIn->userData["marker"] = std::string("root-data");

    auto tool = Group::create();
    tool->name = "Gripper";
    link->add(tool);

    // base_link carries its own visual group AND the shoulder joint; delete the
    // visual, which is the kind of thing a rebuild from the file resurrects.
    auto* base = findByName(*authoredIn, "base_link");
    REQUIRE(base != nullptr);
    REQUIRE(base->children.size() == 2);
    base->children.front()->removeFromParent();
    REQUIRE(base->children.size() == 1);

    const auto placeholder = freeze(*authoredIn);
    scene.add(placeholder);
    const std::string keptUuid = placeholder->uuid;

    // The fresh import the editor makes for its joint table.
    auto donor = loadArm();
    REQUIRE(donor);

    const auto live = transplantRobot(*placeholder, donor);
    REQUIRE(live);

    // Same identity, same place in the scene, now articulated.
    auto* inScene = findByUuid(scene, keptUuid);
    REQUIRE(inScene != nullptr);
    CHECK(inScene == live.get());
    CHECK(live->name == "MyArm");
    CHECK(live->numDOF() == 1);
    CHECK(std::abs(live->getJointValue(0) - 0.4f) < 1e-3f);

    // The document's subtree, not the file's.
    CHECK(readString(*live, "marker") == "root-data");
    CHECK(RobotConfig::read(*live).has_value());

    auto* liveLink = findByName(*live, "upper_link");
    REQUIRE(liveLink != nullptr);
    CHECK(readString(*liveLink, "sensor") == "type=encoder;joint=shoulder");

    auto* gripper = findByName(*live, "Gripper");
    REQUIRE(gripper != nullptr);
    CHECK(gripper->parent == liveLink);

    // The deleted node stayed deleted rather than coming back with the file.
    auto* liveBase = findByName(*live, "base_link");
    REQUIRE(liveBase != nullptr);
    CHECK(liveBase->children.size() == 1);

    // And the joint really drives the document's node.
    auto* jointNode = liveLink->parent;
    REQUIRE(jointNode != nullptr);
    const auto beforeW = jointNode->quaternion.w;
    live->setJointValue(0, -1.f);
    CHECK(std::abs(jointNode->quaternion.w - beforeW) > 1e-3f);
}

TEST_CASE("transplantRobot reports a link the saved subtree no longer has") {

    Scene scene;

    auto authoredIn = loadArm();
    REQUIRE(authoredIn);
    RobotConfig robotConfig;
    robotConfig.urdf = "arm.urdf";
    robotConfig.write(*authoredIn);

    // A node the file names but the document no longer has under that name.
    auto* link = findByName(*authoredIn, "upper_link");
    REQUIRE(link != nullptr);
    link->name = "renamed_link";

    const auto placeholder = freeze(*authoredIn);
    scene.add(placeholder);

    auto donor = loadArm();
    REQUIRE(donor);

    bool reported = false;
    const auto live = transplantRobot(*placeholder, donor, [&](const std::string& m) {
        if (m.find("upper_link") != std::string::npos) reported = true;
    });

    // Reported rather than passed over in silence.
    CHECK(reported);
    // The joint still resolved — its node is the one above the renamed link —
    // so the robot is articulated anyway.
    REQUIRE(live);
    CHECK(live->numDOF() == 1);
    CHECK(findByName(*live, "renamed_link") != nullptr);
}

TEST_CASE("transplantRobot says so when it has to fall back to the file") {

    // A subtree that no longer resembles the file at all: every name changed.
    // The articulation wins over the annotations, and the console is told which
    // way the trade went.
    Scene scene;

    auto authoredIn = loadArm();
    REQUIRE(authoredIn);
    RobotConfig robotConfig;
    robotConfig.urdf = "arm.urdf";
    robotConfig.write(*authoredIn);

    authoredIn->traverse([](Object3D& node) {
        if (!node.name.empty()) node.name += "_v2";
    });

    const auto placeholder = freeze(*authoredIn);
    scene.add(placeholder);

    auto donor = loadArm();
    REQUIRE(donor);

    bool complained = false;
    const auto live = transplantRobot(*placeholder, donor, [&](const std::string& m) {
        if (m.find("rebuilt from the file") != std::string::npos) complained = true;
    });

    REQUIRE(live);
    CHECK(complained);
    CHECK(live->numDOF() == 1);
    // The file's names, because the file's subtree is what is standing there.
    CHECK(findByName(*live, "upper_link") != nullptr);
}

TEST_CASE("a joint that fails to resolve does not shift its neighbour's saved value") {

    // The saved values are indexed by the file's DOF layout. When one joint
    // cannot be matched the live table compacts around it — and the surviving
    // joints must still read their OWN values, not slide down into the gap.
    const char* kTwoJoint = R"(
        <robot name="arm2">
          <link name="base_link">
            <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
          </link>
          <link name="link1">
            <visual><geometry><box size="0.1 0.4 0.1"/></geometry></visual>
          </link>
          <link name="link2">
            <visual><geometry><box size="0.1 0.3 0.1"/></geometry></visual>
          </link>
          <joint name="shoulder" type="revolute">
            <parent link="base_link"/>
            <child link="link1"/>
            <origin xyz="0 0 0.2"/>
            <axis xyz="0 0 1"/>
            <limit lower="-1.5" upper="1.5"/>
          </joint>
          <joint name="elbow" type="revolute">
            <parent link="link1"/>
            <child link="link2"/>
            <origin xyz="0 0.4 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-1.5" upper="1.5"/>
          </joint>
        </robot>)";

    const auto load = [&] {
        URDFLoader loader;
        return loader.parse(std::filesystem::temp_directory_path(), kTwoJoint);
    };

    Scene scene;

    auto authoredIn = load();
    REQUIRE(authoredIn);
    REQUIRE(authoredIn->numDOF() == 2);

    RobotConfig robotConfig;
    robotConfig.urdf = "arm2.urdf";
    robotConfig.joints = {0.3f, 0.7f};// shoulder, elbow
    robotConfig.write(*authoredIn);

    // Make the SHOULDER unresolvable: both its child link and its joint node
    // change names, so neither lookup can find it. The elbow still resolves —
    // its joint node is the parent of "link2", which is untouched.
    findByName(*authoredIn, "link1")->name = "link1_v2";
    findByName(*authoredIn, "shoulder")->name = "shoulder_v2";

    const auto placeholder = freeze(*authoredIn);
    scene.add(placeholder);

    auto donor = load();
    REQUIRE(donor);

    const auto live = transplantRobot(*placeholder, donor);
    REQUIRE(live);

    // One joint survived, and it kept the elbow's value — not the shoulder's.
    REQUIRE(live->numDOF() == 1);
    CHECK(std::abs(live->getJointValue(0) - 0.7f) < 1e-4f);
}

TEST_CASE("transplantRobot falls back to the file when the document has no subtree") {

    // A node carrying nothing but a urdf reference — a document whose linked
    // asset could not be resolved at parse time, or one written by hand. There
    // is nothing to adopt, so the file's own subtree is planted. No complaint:
    // nothing was lost, because there was nothing there.
    Scene scene;

    auto placeholder = Object3D::create();
    placeholder->name = "arm";
    RobotConfig robotConfig;
    robotConfig.urdf = "arm.urdf";
    robotConfig.write(*placeholder);
    scene.add(placeholder);

    auto donor = loadArm();
    REQUIRE(donor);

    bool complained = false;
    const auto live = transplantRobot(*placeholder, donor, [&](const std::string& m) {
        if (m.find("rebuilt from the file") != std::string::npos) complained = true;
    });

    REQUIRE(live);
    CHECK_FALSE(complained);
    CHECK(live->numDOF() == 1);
    CHECK(findByName(*live, "upper_link") != nullptr);

    live->setJointValue(0, 0.5f);
    CHECK(std::abs(live->getJointValue(0) - 0.5f) < 1e-4f);
}
