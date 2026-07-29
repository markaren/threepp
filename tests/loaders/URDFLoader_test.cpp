#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Robot.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace threepp;

namespace {

    std::shared_ptr<Robot> parseUrdf(const std::string& urdf) {
        URDFLoader loader;
        return loader.parse(std::filesystem::temp_directory_path(), urdf);
    }

}// namespace

TEST_CASE("URDFLoader handles material without inline color") {

    // A named material reference has no <color> child. This previously indexed
    // out of bounds into the tokens of an empty rgba attribute.
    const std::string urdf = R"(
        <robot name="test">
          <link name="base">
            <visual>
              <geometry><box size="1 1 1"/></geometry>
              <material name="grey"/>
            </visual>
          </link>
        </robot>)";

    auto robot = parseUrdf(urdf);
    REQUIRE(robot);

    int meshes = 0;
    robot->traverseType<Mesh>([&meshes](Mesh&) { ++meshes; });
    CHECK(meshes > 0);
}

TEST_CASE("URDFLoader handles truncated and padded rgba values") {

    // Fewer than 4 components and runs of whitespace must not crash; the
    // malformed color falls back to the default material.
    const std::string urdf = R"(
        <robot name="test">
          <link name="base">
            <visual>
              <geometry><box size="1 1 1"/></geometry>
              <material name="bad"><color rgba="0.5 0.5"/></material>
            </visual>
            <visual>
              <geometry><box size="1 1 1"/></geometry>
              <material name="padded"><color rgba="1   0  0    1"/></material>
            </visual>
          </link>
        </robot>)";

    auto robot = parseUrdf(urdf);
    REQUIRE(robot);

    // The padded-but-valid color must still be applied.
    bool foundRed = false;
    robot->traverseType<Mesh>([&foundRed](Mesh& mesh) {
        if (const auto* mat = mesh.materialAs<MeshStandardMaterial>()) {
            if (mat->color.equals(Color(1.f, 0.f, 0.f))) foundRed = true;
        }
    });
    CHECK(foundRed);
}

TEST_CASE("URDFLoader loads a real robot with DAE visuals and STL collisions") {

    const auto path = std::filesystem::path(DATA_FOLDER) / "urdf" / "lbr_iiwa_14_r820.urdf";
    if (!std::filesystem::exists(path)) {
        SKIP("lbr_iiwa_14_r820.urdf not available");
    }

    URDFLoader loader;
    auto robot = loader.load(path);
    REQUIRE(robot);

    int meshes = 0;
    robot->traverseType<Mesh>([&meshes](Mesh&) { ++meshes; });
    CHECK(meshes > 0);
}

TEST_CASE("parseArticulation turns a mesh collision into a convex hull") {

    // A <mesh> collision used to become its bounding box (a chair leg collided
    // as a solid slab). It now becomes a Hull carrying the mesh's vertices, so
    // the articulation builder can cook one convex hull instead.
    const auto dir = std::filesystem::temp_directory_path() / "threepp_urdf_hull";
    std::filesystem::create_directories(dir);

    {
        // A tetrahedron: 4 vertices, enough to cook a hull.
        std::ofstream obj(dir / "tetra.obj", std::ios::trunc);
        obj << "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
            << "f 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n";
    }
    {
        std::ofstream urdf(dir / "robot.urdf", std::ios::trunc);
        urdf << R"(
        <robot name="test">
          <link name="base">
            <collision>
              <geometry><mesh filename="tetra.obj"/></geometry>
            </collision>
          </link>
        </robot>)";
    }

    URDFLoader loader;
    const auto desc = loader.parseArticulation(dir / "robot.urdf", false);
    REQUIRE(desc.links.size() == 1);

    using Shape = URDFArticulationDesc::Collision::Shape;
    const auto& coll = desc.links[0].collision;
    CHECK(coll.shape == Shape::Hull);
    // The mesh's vertices survive as flat x,y,z floats. The OBJ loader
    // de-indexes the four triangular faces (12 vertices, 36 floats); whatever
    // the exact count, it must be a non-empty multiple of 3 and enough to cook
    // a hull (>= 4 points).
    CHECK(coll.hullPoints.size() % 3 == 0);
    CHECK(coll.hullPoints.size() >= 12);

    std::filesystem::remove_all(dir);
}

TEST_CASE("URDFLoader shares mesh resources between visual and collision") {

    // The same mesh referenced from <visual> and <collision> should be loaded
    // once and cloned: distinct Mesh instances, shared BufferGeometry.
    const auto dir = std::filesystem::temp_directory_path() / "threepp_urdf_share";
    std::filesystem::create_directories(dir);

    {
        std::ofstream obj(dir / "tri.obj", std::ios::trunc);
        obj << "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    }
    {
        std::ofstream urdf(dir / "robot.urdf", std::ios::trunc);
        urdf << R"(
        <robot name="test">
          <link name="base">
            <visual>
              <geometry><mesh filename="tri.obj"/></geometry>
            </visual>
            <collision>
              <geometry><mesh filename="tri.obj"/></geometry>
            </collision>
          </link>
        </robot>)";
    }

    URDFLoader loader;
    auto robot = loader.load(dir / "robot.urdf");
    REQUIRE(robot);

    std::vector<Mesh*> meshes;
    std::unordered_map<BufferGeometry*, int> geometryUses;
    robot->traverseType<Mesh>([&](Mesh& mesh) {
        meshes.push_back(&mesh);
        if (mesh.geometry()) geometryUses[mesh.geometry().get()] += 1;
    });

    // Two distinct Mesh objects (visual + collision wireframe)...
    REQUIRE(meshes.size() == 2);
    CHECK(meshes[0] != meshes[1]);
    // ...backed by ONE shared BufferGeometry.
    REQUIRE(geometryUses.size() == 1);
    CHECK(geometryUses.begin()->second == 2);

    std::filesystem::remove_all(dir);
}

TEST_CASE("scaleArticulationDesc reads a millimetre URDF as metres") {

    // The claim in one assertion: the SAME robot drawn in millimetres, scaled by
    // 0.001, is the robot drawn in metres. Written as two files rather than as a
    // list of expected numbers so the test states the property instead of
    // restating the implementation.
    const auto dir = std::filesystem::temp_directory_path() / "threepp_urdf_units";
    std::filesystem::create_directories(dir);

    // A shape per collision kind, a revolute and a prismatic joint, an offset
    // origin on both the joint and the collider, and a mass — one of everything
    // the scaling has an opinion about.
    const auto write = [&dir](const char* name, float u) {
        std::ofstream urdf(dir / name, std::ios::trunc);
        urdf << "<robot name=\"units\">\n"
             << "  <link name=\"base\">\n"
             << "    <collision><origin xyz=\"" << 0.5f * u << " 0 " << 0.25f * u << "\"/>\n"
             << "      <geometry><box size=\"" << 0.4f * u << " " << 0.2f * u << " " << 0.1f * u << "\"/></geometry>\n"
             << "    </collision>\n"
             << "    <inertial><mass value=\"2.5\"/></inertial>\n"
             << "  </link>\n"
             << "  <link name=\"arm\">\n"
             << "    <collision><geometry><cylinder radius=\"" << 0.05f * u << "\" length=\"" << 0.6f * u << "\"/></geometry></collision>\n"
             << "  </link>\n"
             << "  <link name=\"slide\">\n"
             << "    <collision><geometry><sphere radius=\"" << 0.03f * u << "\"/></geometry></collision>\n"
             << "  </link>\n"
             << "  <joint name=\"shoulder\" type=\"revolute\">\n"
             << "    <parent link=\"base\"/><child link=\"arm\"/>\n"
             << "    <origin xyz=\"0 0 " << 0.3f * u << "\"/><axis xyz=\"0 0 1\"/>\n"
             << "    <limit lower=\"-1.5\" upper=\"1.5\"/>\n"
             << "  </joint>\n"
             << "  <joint name=\"rail\" type=\"prismatic\">\n"
             << "    <parent link=\"arm\"/><child link=\"slide\"/>\n"
             << "    <origin xyz=\"0 " << 0.1f * u << " 0\"/><axis xyz=\"1 0 0\"/>\n"
             << "    <limit lower=\"0\" upper=\"" << 0.8f * u << "\"/>\n"
             << "  </joint>\n"
             << "</robot>\n";
    };
    write("metres.urdf", 1.f);
    write("millimetres.urdf", 1000.f);

    URDFLoader loader;
    const auto metres = loader.parseArticulation(dir / "metres.urdf", false);
    auto millimetres = loader.parseArticulation(dir / "millimetres.urdf", false);
    REQUIRE(metres.links.size() == 3);
    REQUIRE(millimetres.links.size() == metres.links.size());

    scaleArticulationDesc(millimetres, 0.001f);

    const auto close = [](float a, float b) { return std::abs(a - b) < 1e-4f; };

    for (std::size_t i = 0; i < metres.links.size(); ++i) {

        const auto& m = metres.links[i];
        const auto& s = millimetres.links[i];
        INFO("link " << m.name);

        CHECK(s.name == m.name);
        CHECK(s.collision.shape == m.collision.shape);
        CHECK(close(s.collision.halfExtents.x, m.collision.halfExtents.x));
        CHECK(close(s.collision.halfExtents.y, m.collision.halfExtents.y));
        CHECK(close(s.collision.halfExtents.z, m.collision.halfExtents.z));
        CHECK(close(s.collision.radius, m.collision.radius));
        CHECK(close(s.collision.halfHeight, m.collision.halfHeight));

        for (int e = 0; e < 16; ++e) {
            INFO("jointOrigin element " << e);
            CHECK(close(s.jointOrigin.elements[e], m.jointOrigin.elements[e]));
            INFO("collision origin element " << e);
            CHECK(close(s.collision.origin.elements[e], m.collision.origin.elements[e]));
        }

        // Limits: the prismatic one is a distance and scales, the revolute one is
        // an angle and must not.
        REQUIRE(s.range.has_value() == m.range.has_value());
        if (m.range) {
            CHECK(close(s.range->min, m.range->min));
            CHECK(close(s.range->max, m.range->max));
        }

        // Mass is kilograms in both files, so it is the ONE length-free number
        // here that the scale must leave alone.
        CHECK(s.hasMass == m.hasMass);
        CHECK(close(s.mass, m.mass));

        // A direction, not a length.
        CHECK(close(s.jointAxis.x, m.jointAxis.x));
        CHECK(close(s.jointAxis.y, m.jointAxis.y));
        CHECK(close(s.jointAxis.z, m.jointAxis.z));
    }

    // The revolute limit really was carried through unscaled (it would read
    // -0.0015..0.0015 if the scale had been applied blindly to every range).
    const auto arm = std::find_if(millimetres.links.begin(), millimetres.links.end(),
                                  [](const auto& l) { return l.name == "arm"; });
    REQUIRE(arm != millimetres.links.end());
    REQUIRE(arm->range.has_value());
    CHECK(close(arm->range->max, 1.5f));

    std::filesystem::remove_all(dir);
}

TEST_CASE("scaleArticulationDesc leaves a unit or invalid scale alone") {

    const std::string urdf = R"(
        <robot name="test">
          <link name="base">
            <collision><geometry><box size="1 2 3"/></geometry></collision>
          </link>
        </robot>)";

    const auto dir = std::filesystem::temp_directory_path() / "threepp_urdf_units_noop";
    std::filesystem::create_directories(dir);
    {
        std::ofstream out(dir / "robot.urdf", std::ios::trunc);
        out << urdf;
    }

    URDFLoader loader;
    auto desc = loader.parseArticulation(dir / "robot.urdf", false);
    REQUIRE(desc.links.size() == 1);
    const auto before = desc.links[0].collision.halfExtents;

    scaleArticulationDesc(desc, 1.f);
    CHECK(desc.links[0].collision.halfExtents.x == before.x);

    // A zero or negative scale would collapse or mirror the robot; refuse both
    // rather than build something inside out.
    scaleArticulationDesc(desc, 0.f);
    CHECK(desc.links[0].collision.halfExtents.x == before.x);
    scaleArticulationDesc(desc, -2.f);
    CHECK(desc.links[0].collision.halfExtents.x == before.x);

    std::filesystem::remove_all(dir);
}
