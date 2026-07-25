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
