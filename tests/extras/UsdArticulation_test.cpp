// loadUsdArticulation against a hand-written robot, covering what a serial arm from
// a SimReady package does not: a fixed joint between bodies, a slider, a branch, a
// joint frame away from the child's origin, a pose saved away from joint zero,
// several colliders on a link, a bound physics material, and what cannot be imported.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/extras/physx/PhysxWorld.hpp"
#include "threepp/extras/physx/UsdArticulation.hpp"
#include "threepp/geometries/SphereGeometry.hpp"
#include "threepp/materials/MeshBasicMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#include <PxPhysicsAPI.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    const std::string cube = R"(
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0, 1, 3, 2, 2, 3, 7, 6, 6, 7, 5, 4, 4, 5, 1, 0, 2, 6, 4, 0, 7, 3, 1, 5]
            point3f[] points = [(-0.05, -0.05, -0.05), (0.05, -0.05, -0.05), (-0.05, 0.05, -0.05), (0.05, 0.05, -0.05), (-0.05, -0.05, 0.05), (0.05, -0.05, 0.05), (-0.05, 0.05, 0.05), (0.05, 0.05, 0.05)]
)";

    std::string collider(const std::string& name, const std::string& extra = "") {
        return R"(
        def Mesh ")" + name + R"(" (
            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
        ) {
            uniform token physics:approximation = "convexHull"
            uniform token purpose = "guide"
            )" + extra + cube + "        }\n";
    }

    std::string visual() {
        return "        def Mesh \"visual\" {" + cube + "        }\n";
    }

    // Z-up. The hinge sits 0.1 above the base and 0.2 below the upper link's origin,
    // and the file saves it at 30 degrees: 0.1 + 0.2 sin30 along X, 0.1 + 0.2 cos30 up.
    const std::string body = R"(
    def Xform "base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    ) {
        float physics:mass = 5
        point3f physics:centerOfMass = (0.01, 0, 0.05)
        float3 physics:diagonalInertia = (0.1, 0.2, 0.3)
        quatf physics:principalAxes = (0.70710678, 0, 0.70710678, 0)
)" + visual() + collider("collider") + R"(    }

    def Xform "upper" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    ) {
        float physics:mass = 2
        double3 xformOp:translate = (0.1, 0, 0.27320508)
        quatf xformOp:orient = (0.9659258, 0, 0.25881904, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
)" + visual() + collider("collider") + R"(    }

    def Xform "carriage" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    ) {
        float physics:mass = 1
        double3 xformOp:translate = (0.2, 0, 0.44641016)
        quatf xformOp:orient = (0.9659258, 0, 0.25881904, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
)" + visual() + collider("collider") + R"(    }

    def Xform "tool" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    ) {
        float physics:mass = 0.5
        double3 xformOp:translate = (0.25, 0, 0.53301270)
        quatf xformOp:orient = (0.9659258, 0, 0.25881904, 0)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]
)" + visual() + collider("collider") + R"(    }

    def Xform "branch" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    ) {
        double3 xformOp:translate = (0, 0.3, 0.1)
        uniform token[] xformOpOrder = ["xformOp:translate"]
)" + visual() + collider("first", "rel material:binding:physics = </robot/grippy>") + collider("second") + R"(    }

    def Material "grippy" (
        prepend apiSchemas = ["PhysicsMaterialAPI"]
    ) {
        float physics:staticFriction = 0.9
        float physics:dynamicFriction = 0.7
        float physics:restitution = 0.1
    }

    def PhysicsRevoluteJoint "hinge" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    ) {
        rel physics:body0 = </robot/base>
        rel physics:body1 = </robot/upper>
        uniform token physics:axis = "Y"
        point3f physics:localPos0 = (0, 0, 0.1)
        point3f physics:localPos1 = (0, 0, -0.2)
        float physics:lowerLimit = -90
        float physics:upperLimit = 120
        float drive:angular:physics:stiffness = 100
        float drive:angular:physics:damping = 10
        float drive:angular:physics:maxForce = 500
        float drive:angular:physics:targetPosition = 30
        uniform token drive:angular:physics:type = "force"
    }

    def PhysicsPrismaticJoint "slide" (
        prepend apiSchemas = ["PhysicsDriveAPI:linear"]
    ) {
        rel physics:body0 = </robot/upper>
        rel physics:body1 = </robot/carriage>
        uniform token physics:axis = "Z"
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        float physics:lowerLimit = 0
        float physics:upperLimit = 0.4
        float drive:linear:physics:stiffness = 20000
        float drive:linear:physics:damping = 1000
        float drive:linear:physics:targetPosition = 0.2
        float physxMimicJoint:rotZ:gearing = 1
    }

    def PhysicsFixedJoint "weld" {
        rel physics:body0 = </robot/carriage>
        rel physics:body1 = </robot/tool>
        point3f physics:localPos0 = (0, 0, 0.1)
        point3f physics:localPos1 = (0, 0, 0)
    }

)";

    // The passive joint takes extra properties (a mimic coupling), so every
    // other case keeps it free.
    std::string passiveJoint(const std::string& extra) {
        return R"(
    def PhysicsRevoluteJoint "passive" {
        rel physics:body0 = </robot/base>
        rel physics:body1 = </robot/branch>
        uniform token physics:axis = "Y"
        point3f physics:localPos0 = (0, 0.3, 0.1)
        point3f physics:localPos1 = (0, 0, 0)
)" + extra + R"(    }

    def PhysicsSphericalJoint "ball" {
        rel physics:body0 = </robot/tool>
        rel physics:body1 = </robot/branch>
    }
)";
    }

    std::string document(const std::string& anchor, const std::string& passive = "") {
        return R"(#usda 1.0
(
    defaultPrim = "robot"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "robot" (
    variants = {
        string PhysX = "Disabled"
    }
    prepend variantSets = ["PhysX"]
) {
)" + anchor + body + passiveJoint(passive) +
               R"(
    variantSet "PhysX" = {
        "Disabled" {
        }
        "Enabled" {
            over "hinge" {
                float physxJoint:maxJointVelocity = 90
                float physxJoint:armature = 0.25
            }
            over "root_joint" {
                int physxArticulation:solverPositionIterationCount = 40
            }
        }
    }
}
)";
    }

    const std::string rootJoint = R"(
    def PhysicsFixedJoint "root_joint" (
        prepend apiSchemas = ["PhysicsArticulationRootAPI"]
    ) {
        rel physics:body1 = </robot/base>
    }
)";

    std::filesystem::path writeFixture(const std::string& name, const std::string& text) {
        const auto path = std::filesystem::temp_directory_path() / name;
        std::ofstream(path) << text;
        return path;
    }

    PhysxWorld::Settings settings() {
        PhysxWorld::Settings s;
        s.fixedTimestep = 1.f / 240.f;
        s.maxSubSteps = 1;
        return s;
    }

    const ArticulationLink& linkNamed(const USDArticulationResult& robot, const std::string& name) {
        const auto* link = robot.articulation->findLink(name);
        REQUIRE(link != nullptr);
        return *link;
    }

    int jointIndex(const USDArticulationResult& robot, const std::string& name) {
        const auto it = std::ranges::find(robot.jointNames, name);
        REQUIRE(it != robot.jointNames.end());
        return static_cast<int>(it - robot.jointNames.begin());
    }

    void requireNear(const Vector3& got, float x, float y, float z, float tolerance = 1e-4f) {
        REQUIRE_THAT(got.x, WithinAbs(x, tolerance));
        REQUIRE_THAT(got.y, WithinAbs(y, tolerance));
        REQUIRE_THAT(got.z, WithinAbs(z, tolerance));
    }

}// namespace

TEST_CASE("the tree: bodies become links, moving joints become DOFs", "[physx][usd]") {
    PhysxWorld world(settings());
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_tree.usda", document(rootJoint)));
    REQUIRE(robot.error.empty());
    REQUIRE(robot.articulation);

    // A fixed joint welds tool to carriage without a DOF, and stays a link with its own mass.
    CHECK(robot.articulation->rawArt()->getNbLinks() == 5);
    CHECK(robot.numDof() == 3);
    CHECK(robot.jointNames.size() == robot.links.size());
    CHECK(static_cast<bool>(robot.articulation->rawArt()->getArticulationFlags() & ::physx::PxArticulationFlag::eFIX_BASE));
    REQUIRE_THAT(linkNamed(robot, "tool").raw()->getMass(), WithinAbs(0.5f, 1e-5f));
    REQUIRE(robot.articulation->findLink("/robot/tool") != nullptr);
}

TEST_CASE("USD's per-degree drives arrive per radian", "[physx][usd]") {
    PhysxWorld world(settings());
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_drive.usda", document(rootJoint)));
    REQUIRE(robot.articulation);

    const auto index = jointIndex(robot, "hinge");
    auto* joint = robot.links[index].raw()->getInboundJoint();
    const auto drive = joint->getDriveParams(robot.links[index].axis());
    REQUIRE_THAT(drive.stiffness, WithinAbs(100.f * 180.f / math::PI, 1e-2f));
    REQUIRE_THAT(drive.damping, WithinAbs(10.f * 180.f / math::PI, 1e-3f));
    REQUIRE_THAT(drive.maxForce, WithinAbs(500.f, 1e-4f));
    REQUIRE_THAT(joint->getDriveTarget(robot.links[index].axis()), WithinAbs(math::degToRad(30.f), 1e-5f));
    const auto limit = joint->getLimitParams(robot.links[index].axis());
    REQUIRE_THAT(limit.low, WithinAbs(math::degToRad(-90.f), 1e-5f));
    REQUIRE_THAT(limit.high, WithinAbs(math::degToRad(120.f), 1e-5f));

    // Linear quantities pass through unconverted.
    const auto slide = jointIndex(robot, "slide");
    const auto linear = robot.links[slide].raw()->getInboundJoint()->getDriveParams(robot.links[slide].axis());
    REQUIRE_THAT(linear.stiffness, WithinAbs(20000.f, 1e-2f));

    // The passive joint has no drive and no limits, as authored.
    const auto passive = jointIndex(robot, "passive");
    auto* free = robot.links[passive].raw()->getInboundJoint();
    CHECK(free->getMotion(robot.links[passive].axis()) == ::physx::PxArticulationMotion::eFREE);
    CHECK(free->getDriveParams(robot.links[passive].axis()).stiffness == 0.f);
}

TEST_CASE("the file's PhysX variant is selected", "[physx][usd]") {
    PhysxWorld world(settings());
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_variant.usda", document(rootJoint)));
    REQUIRE(robot.articulation);

    ::physx::PxU32 position = 0, velocity = 0;
    robot.articulation->rawArt()->getSolverIterationCounts(position, velocity);
    CHECK(position == 40);
    auto* hinge = robot.links[jointIndex(robot, "hinge")].raw()->getInboundJoint();
    REQUIRE_THAT(hinge->getMaxJointVelocity(), WithinAbs(math::degToRad(90.f), 1e-5f));
    REQUIRE_THAT(hinge->getArmature(robot.links[jointIndex(robot, "hinge")].axis()), WithinAbs(0.25f, 1e-6f));
}

TEST_CASE("a robot saved away from joint zero loads where the file puts it", "[physx][usd]") {
    PhysxWorld world(settings());
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_pose.usda", document(rootJoint)));
    REQUIRE(robot.articulation);

    const auto q = robot.articulation->jointPositions();
    REQUIRE_THAT(q[jointIndex(robot, "hinge")], WithinAbs(math::degToRad(30.f), 1e-4f));
    REQUIRE_THAT(q[jointIndex(robot, "slide")], WithinAbs(0.2f, 1e-4f));

    // USD (x, y, z) is threepp (x, z, -y).
    requireNear(linkNamed(robot, "upper").position(), 0.1f, 0.27320508f, 0.f);
    requireNear(linkNamed(robot, "carriage").position(), 0.2f, 0.44641016f, 0.f);
    requireNear(linkNamed(robot, "tool").position(), 0.25f, 0.53301270f, 0.f);

    // The bound meshes start there too, not at joint zero.
    const auto mesh = std::ranges::find_if(robot.meshes, [&](const auto& m) {
        return m->position.distanceTo(linkNamed(robot, "tool").position()) < 1e-4f;
    });
    CHECK(mesh != robot.meshes.end());
}

TEST_CASE("a joint pivots about its own frame, away from the child's origin", "[physx][usd]") {
    PhysxWorld world(settings());
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_pivot.usda", document(rootJoint)));
    REQUIRE(robot.articulation);

    auto targets = robot.articulation->jointPositions();
    targets[jointIndex(robot, "hinge")] = math::degToRad(75.f);
    targets[jointIndex(robot, "slide")] = 0.2f;
    robot.articulation->setDriveTargets(targets.data(), targets.size());
    for (int i = 0; i < 960; ++i) world.step(1.f / 240.f);

    REQUIRE_THAT(robot.articulation->jointPositions()[jointIndex(robot, "hinge")], WithinAbs(math::degToRad(75.f), 1e-2f));
    // 0.1 above the base, whatever the angle: localPos1 in the upper link stays on it.
    requireNear(linkNamed(robot, "upper").worldPoint({0.f, 0.f, -0.2f}), 0.f, 0.1f, 0.f, 1e-3f);
}

TEST_CASE("mass: what is authored wins, the colliders supply the rest", "[physx][usd]") {
    PhysxWorld world(settings());
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_mass.usda", document(rootJoint)));
    REQUIRE(robot.articulation);

    const auto* base = linkNamed(robot, "base").raw();
    REQUIRE_THAT(base->getMass(), WithinAbs(5.f, 1e-5f));
    const auto pose = base->getCMassLocalPose();
    REQUIRE_THAT(pose.p.x, WithinAbs(0.01f, 1e-6f));
    REQUIRE_THAT(pose.p.z, WithinAbs(0.05f, 1e-6f));
    REQUIRE_THAT(pose.q.y, WithinAbs(0.70710678f, 1e-5f));// wxyz in the file, xyzw in PhysX
    REQUIRE_THAT(pose.q.w, WithinAbs(0.70710678f, 1e-5f));
    REQUIRE_THAT(base->getMassSpaceInertiaTensor().z, WithinAbs(0.3f, 1e-5f));

    // Two 0.1 m cubes overlapping exactly at default density, no MassAPI at all.
    REQUIRE_THAT(linkNamed(robot, "branch").raw()->getMass(), WithinAbs(2.f * 1000.f * 0.001f, 1e-3f));
}

TEST_CASE("every collider of a link is a shape, with its bound material", "[physx][usd]") {
    PhysxWorld world(settings());
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_shapes.usda", document(rootJoint)));
    REQUIRE(robot.articulation);

    const auto* branch = linkNamed(robot, "branch").raw();
    REQUIRE(branch->getNbShapes() == 2);
    ::physx::PxShape* shapes[2];
    branch->getShapes(shapes, 2);
    ::physx::PxMaterial* material = nullptr;
    shapes[0]->getMaterials(&material, 1);
    REQUIRE_THAT(material->getStaticFriction(), WithinAbs(0.9f, 1e-6f));
    REQUIRE_THAT(material->getDynamicFriction(), WithinAbs(0.7f, 1e-6f));
    REQUIRE_THAT(material->getRestitution(), WithinAbs(0.1f, 1e-6f));
    shapes[1]->getMaterials(&material, 1);
    CHECK(material == &world.defaultMaterial());
}

TEST_CASE("a falling ball lands on the collider, in the link's frame", "[physx][usd]") {
    PhysxWorld world(settings());
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_contact.usda", document(rootJoint)));
    REQUIRE(robot.articulation);

    // The base's collider is a 0.1 m cube about the origin: its top is 0.05 up.
    auto ball = Mesh::create(SphereGeometry::create(0.02f), MeshBasicMaterial::create());
    ball->position.set(-0.01f, 0.2f, 0.f);
    auto* body = world.add(*ball, 500.f);
    for (int i = 0; i < 480; ++i) world.step(1.f / 240.f);

    REQUIRE_THAT(body->getGlobalPose().p.y, WithinAbs(0.07f, 2e-3f));
}

TEST_CASE("no world joint means a floating base", "[physx][usd]") {
    PhysxWorld world(settings());
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_float.usda", document("")));
    REQUIRE(robot.articulation);
    CHECK_FALSE(static_cast<bool>(robot.articulation->rawArt()->getArticulationFlags() & ::physx::PxArticulationFlag::eFIX_BASE));

    const float start = linkNamed(robot, "base").position().y;
    for (int i = 0; i < 24; ++i) world.step(1.f / 240.f);
    CHECK(linkNamed(robot, "base").position().y < start - 0.01f);
}

TEST_CASE("options override the file", "[physx][usd]") {
    PhysxWorld world(settings());
    USDArticulationOptions options;
    options.fixedBase = false;
    options.stiffness = 7.f;
    options.basePosition.set(1.f, 2.f, 3.f);
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_options.usda", document(rootJoint)), options);
    REQUIRE(robot.articulation);

    CHECK_FALSE(static_cast<bool>(robot.articulation->rawArt()->getArticulationFlags() & ::physx::PxArticulationFlag::eFIX_BASE));
    const auto index = jointIndex(robot, "hinge");
    CHECK(robot.links[index].raw()->getInboundJoint()->getDriveParams(robot.links[index].axis()).stiffness == 7.f);
    requireNear(linkNamed(robot, "upper").position(), 1.1f, 2.27320508f, 3.f);
}

TEST_CASE("what cannot be imported is said", "[physx][usd]") {
    PhysxWorld world(settings());
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_diagnostics.usda", document(rootJoint)));
    REQUIRE(robot.articulation);

    const auto said = [&](const std::string& text) {
        return std::ranges::any_of(robot.diagnostics, [&](const auto& d) { return d.find(text) != std::string::npos; });
    };
    CHECK(said("/robot/ball: PhysicsSphericalJoint is not represented"));
    CHECK(said("/robot/slide: mimic coupling names the rotZ axis, which a prismatic joint has not"));
}

TEST_CASE("a mimic coupling holds q + gearing * qRef + offset = 0", "[physx][usd]") {
    PhysxWorld world(settings());
    // The passive joint follows the hinge at half its angle, five degrees behind.
    const std::string mimic = R"(        float physxMimicJoint:rotY:gearing = -0.5
        float physxMimicJoint:rotY:offset = 5
        rel physxMimicJoint:rotY:referenceJoint = </robot/hinge>
)";
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_mimic.usda", document(rootJoint, mimic)));
    REQUIRE(robot.articulation);
    REQUIRE(robot.mimics.size() == 1);
    CHECK(robot.mimics[0].joint == "/robot/passive");
    CHECK(robot.mimics[0].referenceJoint == "/robot/hinge");
    const auto& links = robot.articulation->links();
    CHECK(&links[static_cast<std::size_t>(robot.mimics[0].link)] == robot.articulation->findLink("branch"));
    CHECK(&links[static_cast<std::size_t>(robot.mimics[0].referenceLink)] == robot.articulation->findLink("upper"));
    REQUIRE_THAT(robot.mimics[0].gearing, WithinAbs(-0.5f, 1e-6f));
    REQUIRE_THAT(robot.mimics[0].offset, WithinAbs(math::degToRad(5.f), 1e-6f));// per degree in the file
    CHECK(robot.articulation->mimicJoints().size() == 1);
    CHECK(robot.diagnostics.size() == 2);// the ball joint and the slide's axis, nothing about the coupling

    auto targets = robot.articulation->jointPositions();
    targets[jointIndex(robot, "hinge")] = math::degToRad(60.f);
    robot.articulation->setDriveTargets(targets.data(), targets.size());
    for (int i = 0; i < 960; ++i) world.step(1.f / 240.f);

    const auto q = robot.articulation->jointPositions();
    const float hinge = q[jointIndex(robot, "hinge")];
    const float passive = q[jointIndex(robot, "passive")];
    REQUIRE_THAT(hinge, WithinAbs(math::degToRad(60.f), 2e-2f));
    REQUIRE_THAT(passive - 0.5f * hinge + math::degToRad(5.f), WithinAbs(0.f, 1e-3f));
}

TEST_CASE("a mimic coupling that cannot be built is said, and the joint stays free", "[physx][usd]") {
    PhysxWorld world(settings());
    const std::string dangling = R"(        float physxMimicJoint:rotY:gearing = 1
        rel physxMimicJoint:rotY:referenceJoint = </robot/ball>
)";
    auto robot = loadUsdArticulation(world, writeFixture("usd_articulation_mimic_dangling.usda", document(rootJoint, dangling)));
    REQUIRE(robot.articulation);
    CHECK(robot.mimics.empty());
    CHECK(robot.articulation->mimicJoints().empty());
    CHECK(std::ranges::any_of(robot.diagnostics, [](const auto& d) {
        return d == "/robot/passive: mimic coupling follows /robot/ball, which is not a moving joint of this articulation";
    }));
}

TEST_CASE("a file that holds no robot is an error, not an empty result", "[physx][usd]") {
    PhysxWorld world(settings());
    const auto missing = loadUsdArticulation(world, std::filesystem::temp_directory_path() / "usd_articulation_missing.usda");
    CHECK_FALSE(missing.articulation);
    CHECK_FALSE(missing.error.empty());

    const auto rigid = loadUsdArticulation(world, writeFixture("usd_articulation_rigid.usda", R"(#usda 1.0
def Xform "thing" {
    def Mesh "visual" {)" + cube + R"(    }
}
)"));
    CHECK_FALSE(rigid.articulation);
    CHECK(rigid.error.find("no UsdPhysics joints") != std::string::npos);
}
