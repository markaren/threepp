#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/OBJLoader.hpp"
#include "threepp/objects/Mesh.hpp"

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
