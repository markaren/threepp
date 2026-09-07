// ModelLoader's dispatch, at the seams the per-format tests cannot see.
//
// Each format loader has its own test; what nothing else covers is the routing
// itself — and the case that motivated this file: .json, the format the
// threepp EDITOR saves, was the one extension the dispatch produced itself and
// could not read back. Author a cell in the editor, and ModelLoader — the
// loader every consumer reaches for first — answered "unsupported extension"
// to the editor's own output.

#include <catch2/catch_test_macros.hpp>

#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/ObjectExporter.hpp"

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

using namespace threepp;

namespace {

    std::filesystem::path testDir() {

        const auto dir = std::filesystem::temp_directory_path() / "threepp-modelloader-test";
        std::filesystem::create_directories(dir);
        return dir;
    }

    // A small cell, exported exactly as the editor exports: a Scene root
    // carrying its own userData, with a named mesh inside.
    std::filesystem::path writeSceneDocument() {

        auto scene = Scene::create();
        scene->name = "Cell";
        scene->userData["editorView"] = std::string("1,2,3@0,0,0");

        auto part = Mesh::create(BoxGeometry::create(0.2f, 0.2f, 0.2f),
                                 MeshStandardMaterial::create());
        part->name = "Part";
        part->position.set(0.5f, 0.1f, 0.f);
        scene->add(part);

        const auto path = testDir() / "cell.json";
        ObjectExporter exporter;
        exporter.save(*scene, path);
        return path;
    }

}// namespace


TEST_CASE("ModelLoader reads back the document the editor saves") {

    const auto path = writeSceneDocument();

    ModelLoader loader;
    const auto group = loader.load(path);
    REQUIRE(group != nullptr);

    // The document's root is a Scene, which is not a Group - so it is ADOPTED
    // as the group's child, never unwrapped: unwrapping would silently drop
    // everything that lives ON the root (environment, fog, userData).
    REQUIRE(group->children.size() == 1);
    auto* root = group->children.front()->as<Scene>();
    REQUIRE(root != nullptr);
    CHECK(root->name == "Cell");
    // The reason adoption is the rule, asserted: the root's own data survived.
    REQUIRE(root->userData.contains("editorView"));
    CHECK(std::any_cast<std::string>(root->userData.at("editorView")) == "1,2,3@0,0,0");

    // And the content is really there, one level down as documented.
    auto* part = group->getObjectByName("Part");
    REQUIRE(part != nullptr);
    auto* mesh = part->as<Mesh>();
    REQUIRE(mesh != nullptr);
    CHECK(mesh->geometry() != nullptr);
    CHECK(mesh->material() != nullptr);
}

TEST_CASE("A Group-rooted document comes back as itself, not double-wrapped") {

    // ObjectExporter exports any subtree; a Group root is what exporting a
    // prefab-shaped selection produces. That one IS the return type already,
    // and wrapping it again would add a layer nobody authored.
    auto group = Group::create();
    group->name = "Prefab";
    auto part = Mesh::create(BoxGeometry::create(0.1f, 0.1f, 0.1f),
                             MeshStandardMaterial::create());
    part->name = "Widget";
    group->add(part);

    const auto path = testDir() / "prefab.json";
    ObjectExporter exporter;
    exporter.save(*group, path);

    ModelLoader loader;
    const auto loaded = loader.load(path);
    REQUIRE(loaded != nullptr);
    CHECK(loaded->name == "Prefab");
    // The child is the widget itself - no adopted intermediate.
    REQUIRE(loaded->children.size() == 1);
    CHECK(loaded->children.front()->name == "Widget");
}

TEST_CASE("A .json that is not a document is a null, not a fall-through") {

    // Malformed JSON must fail as "this document is bad" (nullptr from the
    // ObjectLoader route), not tumble past the branch into the unsupported-
    // extension path as if .json were still foreign.
    const auto path = testDir() / "garbage.json";
    std::ofstream(path, std::ios::trunc) << "{ not json at all";

    ModelLoader loader;
    CHECK(loader.load(path) == nullptr);
}
