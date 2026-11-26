#include <catch2/catch_test_macros.hpp>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/loaders/ColladaLoader.hpp"
#include "threepp/materials/interfaces.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/textures/Texture.hpp"

#include "tiny_png.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    std::filesystem::path writeTempFile(const std::string& name, const std::string& contents) {
        auto path = std::filesystem::temp_directory_path() / name;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << contents;
        return path;
    }

    // A unit quad (two triangles sharing an edge) instanced from two nodes.
    std::string quadDae(const std::string& p, const std::string& vcount = "3 3") {
        return R"(<?xml version="1.0"?>
<COLLADA version="1.4.1">
  <asset><up_axis>Y_UP</up_axis></asset>
  <library_geometries>
    <geometry id="quad" name="quad">
      <mesh>
        <source id="quad-pos">
          <float_array id="quad-pos-array" count="12">0 0 0  1 0 0  1 1 0  0 1 0</float_array>
          <technique_common><accessor source="#quad-pos-array" count="4" stride="3"/></technique_common>
        </source>
        <vertices id="quad-verts"><input semantic="POSITION" source="#quad-pos"/></vertices>
        <polylist count="2">
          <input semantic="VERTEX" source="#quad-verts" offset="0"/>
          <vcount>)" + vcount +
               R"(</vcount>
          <p>)" + p + R"(</p>
        </polylist>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="scene">
      <node id="a"><instance_geometry url="#quad"/></node>
      <node id="b"><instance_geometry url="#quad"/></node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>)";
    }

    std::vector<Mesh*> collectMeshes(const std::shared_ptr<Group>& group) {
        std::vector<Mesh*> meshes;
        if (group) {
            group->traverseType<Mesh>([&meshes](Mesh& m) { meshes.push_back(&m); });
        }
        return meshes;
    }

}// namespace

TEST_CASE("ColladaLoader produces indexed geometry with joined vertices") {

    const auto path = writeTempFile("threepp_collada_quad.dae", quadDae("0 1 2 0 2 3"));

    ColladaLoader loader;
    auto scene = loader.load(path);
    REQUIRE(scene);

    const auto meshes = collectMeshes(scene);
    REQUIRE(meshes.size() == 2);

    const auto geometry = meshes.front()->geometry();
    REQUIRE(geometry);

    // 6 face corners referencing 4 unique vertices -> indexed, not expanded.
    const auto* index = geometry->getIndex();
    REQUIRE(index);
    CHECK(index->count() == 6);
    const auto* position = geometry->getAttribute<float>("position");
    REQUIRE(position);
    CHECK(position->count() == 4);

    // Both instances of the same geometry share one BufferGeometry.
    CHECK(meshes[0]->geometry().get() == meshes[1]->geometry().get());

    std::filesystem::remove(path);
}

TEST_CASE("ColladaLoader survives out-of-range source indices") {

    // Vertex index 99 points far outside the 4-entry position source. This must
    // not read out of bounds; the bad corner is zero-filled instead.
    const auto path = writeTempFile("threepp_collada_oob.dae", quadDae("0 1 99 0 2 3"));

    ColladaLoader loader;
    auto scene = loader.load(path);
    REQUIRE(scene);

    const auto meshes = collectMeshes(scene);
    REQUIRE(meshes.size() == 2);
    const auto* position = meshes.front()->geometry()->getAttribute<float>("position");
    REQUIRE(position);
    CHECK(position->count() == 5);// 4 valid + 1 zero-filled

    std::filesystem::remove(path);
}

TEST_CASE("ColladaLoader survives truncated primitive streams") {

    // <p> holds fewer corners than <vcount> promises; the trailing polygon is
    // dropped rather than read past the end.
    const auto path = writeTempFile("threepp_collada_trunc.dae", quadDae("0 1 2 0"));

    ColladaLoader loader;
    auto scene = loader.load(path);
    REQUIRE(scene);

    const auto meshes = collectMeshes(scene);
    REQUIRE(meshes.size() == 2);
    const auto* index = meshes.front()->geometry()->getIndex();
    REQUIRE(index);
    CHECK(index->count() == 3);// only the first, complete triangle

    std::filesystem::remove(path);
}

TEST_CASE("ColladaLoader tags a diffuse texture sRGB") {

    // TextureLoader::load(path) tags NoColorSpace and the renderers decode only
    // sRGB-tagged textures, so an untagged diffuse map rendered too light.
    const auto png = loadertest::writeTinyPng("threepp_collada_diffuse.png");
    const auto path = writeTempFile("threepp_collada_textured.dae", R"(<?xml version="1.0"?>
<COLLADA version="1.4.1">
  <asset><up_axis>Y_UP</up_axis></asset>
  <library_images>
    <image id="diffuse-img"><init_from>threepp_collada_diffuse.png</init_from></image>
  </library_images>
  <library_effects>
    <effect id="fx">
      <profile_COMMON>
        <technique sid="common">
          <phong><diffuse><texture texture="diffuse-img" texcoord="UVMap"/></diffuse></phong>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_materials>
    <material id="mat"><instance_effect url="#fx"/></material>
  </library_materials>
  <library_geometries>
    <geometry id="tri" name="tri">
      <mesh>
        <source id="tri-pos">
          <float_array id="tri-pos-array" count="9">0 0 0  1 0 0  0 1 0</float_array>
          <technique_common><accessor source="#tri-pos-array" count="3" stride="3"/></technique_common>
        </source>
        <vertices id="tri-verts"><input semantic="POSITION" source="#tri-pos"/></vertices>
        <triangles count="1" material="sym">
          <input semantic="VERTEX" source="#tri-verts" offset="0"/>
          <p>0 1 2</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="scene">
      <node id="n">
        <instance_geometry url="#tri">
          <bind_material><technique_common>
            <instance_material symbol="sym" target="#mat"/>
          </technique_common></bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>)");

    ColladaLoader loader;
    auto scene = loader.load(path);
    REQUIRE(scene);

    const auto meshes = collectMeshes(scene);
    REQUIRE(meshes.size() == 1);
    auto* withMap = meshes.front()->material()->as<MaterialWithMap>();
    REQUIRE(withMap);
    REQUIRE(withMap->map);
    CHECK(withMap->map->colorSpace == ColorSpace::sRGB);

    std::filesystem::remove(path);
    std::filesystem::remove(png);
}
