#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/MTLLoader.hpp"
#include "threepp/loaders/OBJLoader.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/textures/Texture.hpp"

#include "tiny_png.hpp"

#include <filesystem>
#include <fstream>
#include <string>

using namespace threepp;

namespace {

    std::filesystem::path writeTempFile(const std::string& name, const std::string& contents) {
        auto path = std::filesystem::temp_directory_path() / name;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << contents;
        out.close();
        return path;
    }

    int positionCount(const std::shared_ptr<Group>& group) {
        REQUIRE(group);
        REQUIRE(group->children.size() == 1);
        auto mesh = group->children.front()->as<Mesh>();
        REQUIRE(mesh);
        auto geometry = mesh->geometry();
        REQUIRE(geometry);
        auto* position = geometry->getAttribute<float>("position");
        REQUIRE(position);
        return position->count();
    }

    // Load a snippet, returning the produced group. A file whose every face was
    // dropped yields a group with no children (the loader skips empty geometry).
    std::shared_ptr<Group> loadObj(const std::string& name, const std::string& contents) {
        auto path = writeTempFile(name, contents);
        OBJLoader loader;
        loader.useCache = false;
        auto group = loader.load(path, false);
        std::filesystem::remove(path);
        return group;
    }

}// namespace

TEST_CASE("OBJLoader handles vertex-only faces without crashing") {

    // `f 1 2 3` has no slashes, so each face-vertex token has no uv/normal
    // component. This previously read out of bounds and crashed.
    const std::string obj =
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "f 1 2 3\n";

    auto path = writeTempFile("threepp_vertex_only.obj", obj);

    OBJLoader loader;
    loader.useCache = false;
    auto group = loader.load(path, false);

    // One triangle -> three position vertices.
    CHECK(positionCount(group) == 3);

    std::filesystem::remove(path);
}

TEST_CASE("OBJLoader handles vertex-only quads (triangulated)") {

    const std::string obj =
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 1.0 1.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "f 1 2 3 4\n";

    auto path = writeTempFile("threepp_vertex_only_quad.obj", obj);

    OBJLoader loader;
    loader.useCache = false;
    auto group = loader.load(path, false);

    // A quad triangulates into two triangles -> six position vertices.
    CHECK(positionCount(group) == 6);

    std::filesystem::remove(path);
}

TEST_CASE("OBJLoader handles all face-vertex token forms") {

    // Exercise `v`, `v/vt`, `v//vn` and `v/vt/vn` in a single file. None should
    // index out of bounds regardless of which components are present.
    const std::string obj =
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "vt 0.0 0.0\n"
            "vt 1.0 0.0\n"
            "vt 0.0 1.0\n"
            "vn 0.0 0.0 1.0\n";

    SECTION("v/vt") {
        auto path = writeTempFile("threepp_v_vt.obj", obj + "f 1/1 2/2 3/3\n");
        OBJLoader loader;
        loader.useCache = false;
        auto group = loader.load(path, false);
        CHECK(positionCount(group) == 3);
        std::filesystem::remove(path);
    }

    SECTION("v//vn") {
        auto path = writeTempFile("threepp_v_vn.obj", obj + "f 1//1 2//1 3//1\n");
        OBJLoader loader;
        loader.useCache = false;
        auto group = loader.load(path, false);
        CHECK(positionCount(group) == 3);
        std::filesystem::remove(path);
    }

    SECTION("v/vt/vn") {
        auto path = writeTempFile("threepp_v_vt_vn.obj", obj + "f 1/1/1 2/2/1 3/3/1\n");
        OBJLoader loader;
        loader.useCache = false;
        auto group = loader.load(path, false);
        CHECK(positionCount(group) == 3);
        std::filesystem::remove(path);
    }
}

// Face indices come straight out of the file and were fed to operator[] with no
// range check, so a truncated or hand-edited .obj could read far past the end of
// the vertex array. Every case below is a heap out-of-bounds read before the
// fix; after it, the offending face is dropped and the rest of the file loads.
TEST_CASE("OBJLoader rejects out-of-range face indices") {

    const std::string threeVerts =
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 0.0 1.0 0.0\n";

    SECTION("index far past the end") {
        auto group = loadObj("threepp_oob_high.obj", threeVerts + "f 1 2 999999\n");
        REQUIRE(group);
        CHECK(group->children.empty());// the only face was dropped
    }

    SECTION("index one past the end") {
        auto group = loadObj("threepp_oob_off_by_one.obj", threeVerts + "f 1 2 4\n");
        REQUIRE(group);
        CHECK(group->children.empty());
    }

    SECTION("index 0 is never valid in OBJ") {
        // The old arithmetic mapped 0 to exactly one-past-the-end.
        auto group = loadObj("threepp_oob_zero.obj", threeVerts + "f 0 1 2\n");
        REQUIRE(group);
        CHECK(group->children.empty());
    }

    SECTION("negative index below the start") {
        auto group = loadObj("threepp_oob_neg.obj", threeVerts + "f -1 -2 -99\n");
        REQUIRE(group);
        CHECK(group->children.empty());
    }

    SECTION("a bad face does not discard the good ones") {
        auto group = loadObj("threepp_oob_mixed.obj",
                             threeVerts + "f 1 2 3\n"
                                          "f 1 2 999999\n"
                                          "f 3 2 1\n");
        // Two surviving triangles -> six vertices.
        CHECK(positionCount(group) == 6);
    }

    SECTION("out-of-range uv drops the whole face, keeping buffers in step") {
        // Emitting the position and then bailing on the uv would leave
        // position/uv permanently misaligned for the rest of the file.
        auto group = loadObj("threepp_oob_uv.obj",
                             threeVerts +
                                     "vt 0.0 0.0\n"
                                     "f 1/1 2/1 3/9\n"
                                     "f 1/1 2/1 3/1\n");
        REQUIRE(group);
        REQUIRE(group->children.size() == 1);
        auto* mesh = group->children.front()->as<Mesh>();
        REQUIRE(mesh);
        auto geometry = mesh->geometry();
        REQUIRE(geometry);
        auto* position = geometry->getAttribute<float>("position");
        auto* uv = geometry->getAttribute<float>("uv");
        REQUIRE(position);
        REQUIRE(uv);
        CHECK(position->count() == 3);// only the second face survived
        CHECK(uv->count() == position->count());
    }
}

TEST_CASE("OBJLoader resolves negative face indices relative to the end") {

    // -1 is the last vertex declared so far, so `f -3 -2 -1` is `f 1 2 3`.
    auto relative = loadObj("threepp_neg_rel.obj",
                            "v 0.0 0.0 0.0\n"
                            "v 1.0 0.0 0.0\n"
                            "v 0.0 1.0 0.0\n"
                            "f -3 -2 -1\n");
    auto absolute = loadObj("threepp_neg_abs.obj",
                            "v 0.0 0.0 0.0\n"
                            "v 1.0 0.0 0.0\n"
                            "v 0.0 1.0 0.0\n"
                            "f 1 2 3\n");

    REQUIRE(positionCount(relative) == 3);
    REQUIRE(positionCount(absolute) == 3);

    auto posOf = [](const std::shared_ptr<Group>& g) {
        return g->children.front()->as<Mesh>()->geometry()->getAttribute<float>("position")->array();
    };
    CHECK(posOf(relative) == posOf(absolute));
}

TEST_CASE("OBJLoader turns l records into line segments") {

    // A four-vertex polyline is three segments; LineSegments wants both
    // endpoints of each, so six positions. `l` records used to be skipped.
    const std::string obj =
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 1.0 1.0 0.0\n"
            "v 0.0 1.0 0.0\n"
            "l 1 2 3 4\n";

    auto group = loadObj("threepp_obj_lines.obj", obj);
    REQUIRE(group);
    REQUIRE(group->children.size() == 1);
    auto* lines = group->children.front()->as<LineSegments>();
    REQUIRE(lines);
    auto* position = lines->geometry()->getAttribute<float>("position");
    REQUIRE(position);
    REQUIRE(position->count() == 6);
    // segment 0: v1 -> v2, segment 1: v2 -> v3, segment 2: v3 -> v4
    CHECK(position->getX(1) == 1.f);
    CHECK(position->getX(2) == 1.f);
    CHECK(position->getY(3) == 1.f);
    CHECK(position->getX(5) == 0.f);
    CHECK(position->getY(5) == 1.f);
    CHECK(lines->geometry()->getAttribute<float>("uv") == nullptr);
}

TEST_CASE("OBJLoader keeps the texture coordinates of an l record") {

    const std::string obj =
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "v 1.0 1.0 0.0\n"
            "vt 0.0 0.0\n"
            "vt 0.5 0.0\n"
            "vt 1.0 0.0\n"
            "l 1/1 2/2 3/3\n";

    auto group = loadObj("threepp_obj_lines_uv.obj", obj);
    REQUIRE(group);
    REQUIRE(group->children.size() == 1);
    auto* lines = group->children.front()->as<LineSegments>();
    REQUIRE(lines);
    auto* uv = lines->geometry()->getAttribute<float>("uv");
    REQUIRE(uv);
    REQUIRE(uv->count() == 4);
    CHECK(uv->getX(1) == 0.5f);
    CHECK(uv->getX(2) == 0.5f);
    CHECK(uv->getX(3) == 1.f);
}

TEST_CASE("OBJLoader drops an l record with an out-of-range index") {

    const std::string obj =
            "v 0.0 0.0 0.0\n"
            "v 1.0 0.0 0.0\n"
            "l 1 2 9\n";

    auto group = loadObj("threepp_obj_lines_bad.obj", obj);
    REQUIRE(group);
    // The whole record is dropped, and an object with no geometry is skipped.
    CHECK(group->children.empty());
}

TEST_CASE("OBJLoader reads vertex colours from v x y z r g b") {

    // Tabs and repeated spaces separate fields as well as single spaces.
    const std::string obj =
            "v 0 0 0 1 0 0\n"
            "v\t1 0 0  0 1 0\n"
            "v 0 1 0 0 0 1\n"
            "f 1 2 3\n";

    auto group = loadObj("threepp_obj_colors.obj", obj);
    REQUIRE(positionCount(group) == 3);
    auto mesh = group->children.front()->as<Mesh>();
    auto* color = mesh->geometry()->getAttribute<float>("color");
    REQUIRE(color);
    REQUIRE(color->count() == 3);
    CHECK(color->getX(0) == 1.f);
    CHECK(color->getY(1) == 1.f);
    CHECK(color->getZ(2) == 1.f);
    CHECK(color->getY(0) == 0.f);
    CHECK(mesh->material()->vertexColors);
}

TEST_CASE("OBJLoader keeps colours aligned when only some vertices have one") {

    // Vertex 1 has no colour, vertex 2 does: vertex 1 reads white and
    // vertex 3 picks up its own colour, not vertex 2's.
    const std::string obj =
            "v 0 0 0\n"
            "v 1 0 0 0 1 0\n"
            "v 0 1 0 0 0 1\n"
            "f 1 2 3\n";

    auto group = loadObj("threepp_obj_colors_partial.obj", obj);
    REQUIRE(positionCount(group) == 3);
    auto mesh = group->children.front()->as<Mesh>();
    auto* color = mesh->geometry()->getAttribute<float>("color");
    REQUIRE(color);
    REQUIRE(color->count() == 3);
    CHECK(color->getX(0) == 1.f);
    CHECK(color->getY(0) == 1.f);
    CHECK(color->getZ(0) == 1.f);
    CHECK(color->getX(1) == 0.f);
    CHECK(color->getY(1) == 1.f);
    CHECK(color->getZ(2) == 1.f);
    CHECK(color->getY(2) == 0.f);
}

TEST_CASE("OBJLoader drops an object's colours when they begin part way through it") {

    // The first face is emitted before any vertex has a colour, the second
    // after: a color attribute for only the second would be shorter than
    // position, so the object loads without one.
    const std::string obj =
            "v 0 0 0\n"
            "v 1 0 0\n"
            "v 0 1 0\n"
            "f 1 2 3\n"
            "v 1 1 0 1 0 0\n"
            "f 2 4 3\n";

    auto group = loadObj("threepp_obj_colors_late.obj", obj);
    REQUIRE(positionCount(group) == 6);
    auto mesh = group->children.front()->as<Mesh>();
    CHECK(mesh->geometry()->getAttribute<float>("color") == nullptr);
    CHECK_FALSE(mesh->material()->vertexColors);
}

TEST_CASE("OBJLoader reads a short v line as zeros instead of past its fields") {

    const std::string obj =
            "v 1\n"
            "v 1 0 0\n"
            "v 0 1 0\n"
            "vt 0.5\n"
            "vt 1 0\n"
            "vt 0 1\n"
            "f 1/1 2/2 3/3\n";

    auto group = loadObj("threepp_obj_short_v.obj", obj);
    REQUIRE(positionCount(group) == 3);
    auto mesh = group->children.front()->as<Mesh>();
    auto* position = mesh->geometry()->getAttribute<float>("position");
    CHECK(position->getX(0) == 1.f);
    CHECK(position->getY(0) == 0.f);
    CHECK(position->getZ(0) == 0.f);
    auto* uv = mesh->geometry()->getAttribute<float>("uv");
    REQUIRE(uv);
    CHECK(uv->getX(0) == 0.5f);
    CHECK(uv->getY(0) == 0.f);
    CHECK(mesh->geometry()->getAttribute<float>("color") == nullptr);
    CHECK_FALSE(mesh->material()->vertexColors);
}

TEST_CASE("MTLLoader leaves a map empty when its file is missing") {

    // map_Kd names a file that is not there: TextureLoader returns null, and
    // the material used to dereference it.
    const std::string mtl =
            "newmtl red\n"
            "Kd 1.0 0.0 0.0\n"
            "map_Kd threepp_no_such_texture.png\n"
            "map_Bump threepp_no_such_normal.png\n";

    auto path = writeTempFile("threepp_missing_map.mtl", mtl);

    MTLLoader loader;
    auto creator = loader.load(path);
    std::filesystem::remove(path);
    REQUIRE(creator);

    auto material = std::dynamic_pointer_cast<MeshPhongMaterial>(creator->create("red"));
    REQUIRE(material);
    CHECK(material->map == nullptr);
    CHECK(material->bumpMap == nullptr);
    CHECK(material->color.r == 1.0f);
    CHECK(material->color.g == 0.0f);
}

TEST_CASE("OBJLoader does not carry one file's .mtl into the next load") {

    // ModelLoader keeps one OBJLoader per thread. The first file's .mtl used to
    // stay on the loader, so a second file without an .mtl took its materials.
    const auto temp = std::filesystem::temp_directory_path();
    const auto firstObj = writeTempFile("threepp_mtl_reuse_first.obj",
                                        "mtllib threepp_mtl_reuse_first.mtl\n"
                                        "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                                        "usemtl red\n"
                                        "f 1 2 3\n");
    const auto firstMtl = writeTempFile("threepp_mtl_reuse_first.mtl", "newmtl red\nKd 1 0 0\n");
    const auto secondObj = writeTempFile("threepp_mtl_reuse_second.obj",
                                         "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                                         "usemtl red\n"
                                         "f 1 2 3\n");
    std::filesystem::remove(temp / "threepp_mtl_reuse_second.mtl");

    OBJLoader loader;
    loader.useCache = false;

    auto first = loader.load(firstObj);
    REQUIRE(first);
    REQUIRE(first->children.size() == 1);
    auto red = std::dynamic_pointer_cast<MeshPhongMaterial>(first->children.front()->as<Mesh>()->material());
    REQUIRE(red);
    CHECK(red->name == "red");
    CHECK(red->color.g == 0.0f);

    auto second = loader.load(secondObj);
    REQUIRE(second);
    REQUIRE(second->children.size() == 1);
    auto material = std::dynamic_pointer_cast<MeshPhongMaterial>(second->children.front()->as<Mesh>()->material());
    REQUIRE(material);
    CHECK(material != red);
    CHECK(material->name == "red");
    CHECK(material->color.g == 1.0f);// the default white, not the first file's red

    std::filesystem::remove(firstObj);
    std::filesystem::remove(firstMtl);
    std::filesystem::remove(secondObj);
}

TEST_CASE("MTLLoader gives a material name its file does not define a default material") {

    // MaterialCreator looked the name up with .at() and threw; three.js makes a
    // default material. An .obj names "" for faces before any usemtl.
    const auto mtl = writeTempFile("threepp_mtl_undefined.mtl", "newmtl red\nKd 1 0 0\n");

    SECTION("through MaterialCreator") {
        auto creator = MTLLoader().load(mtl);
        REQUIRE(creator);

        std::shared_ptr<Material> material;
        REQUIRE_NOTHROW(material = creator->create("not_in_the_file"));
        REQUIRE(material);
        CHECK(material->name == "not_in_the_file");
        CHECK_NOTHROW(creator->create(""));
    }

    SECTION("through OBJLoader, for faces before any usemtl") {
        const auto obj = writeTempFile("threepp_mtl_undefined.obj",
                                       "mtllib threepp_mtl_undefined.mtl\n"
                                       "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                                       "f 1 2 3\n");
        OBJLoader loader;
        loader.useCache = false;
        std::shared_ptr<Group> group;
        REQUIRE_NOTHROW(group = loader.load(obj));
        CHECK(positionCount(group) == 3);
        std::filesystem::remove(obj);
    }

    std::filesystem::remove(mtl);
}

TEST_CASE("MTLLoader's material options convert the colours as three.js does") {

    // convert() restarted its loop at the first property that was not Kd, Ka or
    // Ks (every material has a name), so any options hung the load.
    const auto mtl = writeTempFile("threepp_mtl_options.mtl",
                                   "newmtl orange\nKd 255 128 0\nKs 0 0 0\nNs 10\n");

    MTLLoader loader;
    MaterialOptions options;
    options.normalizeRGB = true;
    options.ignoreZeroRGBs = true;
    options.side = Side::Double;
    loader.materialOptions = options;
    auto creator = loader.load(mtl);
    REQUIRE(creator);

    auto material = std::dynamic_pointer_cast<MeshPhongMaterial>(creator->create("orange"));
    REQUIRE(material);
    CHECK(material->color.r == 1.f);
    CHECK(material->color.g == 128.f / 255.f);
    CHECK(material->color.b == 0.f);
    // Ks 0 0 0 was left out, so the default specular stays.
    CHECK(material->specular.getHex() == MeshPhongMaterial::create()->specular.getHex());
    CHECK(material->shininess == 10.f);
    CHECK(material->side == Side::Double);

    std::filesystem::remove(mtl);
}

TEST_CASE("MTLLoader tags colour maps sRGB and data maps linear") {

    // TextureLoader::load(path) tags NoColorSpace and the renderers decode only
    // sRGB-tagged textures, so an untagged diffuse map rendered too light.
    const auto png = loadertest::writeTinyPng("threepp_mtl_colour_space.png");
    const auto mtl = writeTempFile("threepp_mtl_colour_space.mtl",
                                   "newmtl textured\n"
                                   "map_Kd threepp_mtl_colour_space.png\n"
                                   "map_Ke threepp_mtl_colour_space.png\n"
                                   "map_Bump threepp_mtl_colour_space.png\n"
                                   "map_Ks threepp_mtl_colour_space.png\n");

    auto creator = MTLLoader().load(mtl);
    REQUIRE(creator);
    auto material = std::dynamic_pointer_cast<MeshPhongMaterial>(creator->create("textured"));
    REQUIRE(material);
    REQUIRE(material->map);
    REQUIRE(material->emissiveMap);
    REQUIRE(material->bumpMap);
    REQUIRE(material->specularMap);
    CHECK(material->map->colorSpace == ColorSpace::sRGB);
    CHECK(material->emissiveMap->colorSpace == ColorSpace::sRGB);
    CHECK(material->bumpMap->colorSpace == ColorSpace::Linear);
    CHECK(material->specularMap->colorSpace == ColorSpace::Linear);

    std::filesystem::remove(mtl);
    std::filesystem::remove(png);
}
