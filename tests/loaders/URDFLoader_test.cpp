#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Robot.hpp"

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
