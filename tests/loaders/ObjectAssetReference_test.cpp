// Storing references instead of copies: ImageStorage::Reference for textures
// and ModelStorage::Reference for imported subtrees.
//
// The property under test throughout is that a referenced document is smaller
// and cheaper, and still comes back as the same scene — including the edits
// made after the import, which are the whole reason a reference needs an
// override table rather than just a path.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "threepp/loaders/AssetSource.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/loaders/URDFLoader.hpp"
#include "threepp/objects/Robot.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

#include "external/nlohmann/nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace threepp;
using Catch::Matchers::WithinAbs;

namespace {

    const std::filesystem::path dataFolder{DATA_FOLDER};
    const std::filesystem::path texturePath = dataFolder / "textures" / "checker.png";
    const std::filesystem::path modelPath = dataFolder / "models" / "gltf" / "LeePerrySmith" / "LeePerrySmith.glb";

    std::size_t countNodes(Object3D& root) {

        std::size_t n = 0;
        root.traverse([&n](Object3D&) { ++n; });
        return n;
    }

    // Every node below `root`, in the pre-order the override table is numbered
    // against — the test's own copy of the walk, so a change to the exporter's
    // ordering shows up here as a failure rather than silently agreeing.
    std::vector<Object3D*> descendants(Object3D& root) {

        std::vector<Object3D*> flat;
        root.traverse([&](Object3D& node) {
            if (&node != &root) flat.push_back(&node);
        });
        return flat;
    }

}// namespace


TEST_CASE("Textures reference their source file instead of a base64 copy") {

    TextureLoader textures(false);
    auto texture = textures.load(texturePath, true);
    REQUIRE(texture != nullptr);
    REQUIRE(texture->sourceFile == texturePath);

    auto material = MeshStandardMaterial::create();
    material->map = texture;
    auto mesh = Mesh::create(BoxGeometry::create(), material);

    SECTION("embedding is what you get unless you ask otherwise") {

        ObjectExporter exporter;
        const auto text = exporter.toJson(*mesh);

        CHECK(text.find("data:image/png;base64,") != std::string::npos);
        CHECK(exporter.warnings().empty());
    }

    SECTION("a reference is a path relative to the document, and it round-trips") {

        ObjectExporterOptions options;
        options.images = ImageStorage::Reference;
        options.resourcePath = dataFolder;

        ObjectExporter exporter;
        const auto text = exporter.toJson(*mesh, options);

        CHECK(exporter.warnings().empty());
        CHECK(text.find("data:image/png;base64,") == std::string::npos);
        // Small enough that it obviously does not contain the pixels.
        CHECK(text.size() < 4096);

        const auto j = nlohmann::json::parse(text);
        REQUIRE(j.contains("images"));
        REQUIRE(j["images"].size() == 1);
        const auto url = j["images"][0]["url"].get<std::string>();
        CHECK(url == "textures/checker.png");

        ObjectLoader loader;
        loader.setResourcePath(dataFolder);
        auto parsed = loader.parse(text);

        REQUIRE(parsed != nullptr);
        auto* map = parsed->materialAs<MeshStandardMaterial>()->map.get();
        REQUIRE(map != nullptr);
        REQUIRE_FALSE(map->images().empty());
        CHECK(map->images().front().width() == texture->images().front().width());
        CHECK(map->images().front().height() == texture->images().front().height());
    }

    SECTION("a texture with no file of its own is embedded anyway, and says so") {

        // What a .glb's packed textures look like: real pixels, no path.
        auto packed = Texture::create(texture->images().front());
        material->map = packed;

        ObjectExporterOptions options;
        options.images = ImageStorage::Reference;
        options.resourcePath = dataFolder;

        ObjectExporter exporter;
        const auto text = exporter.toJson(*mesh, options);

        CHECK(text.find("data:image/png;base64,") != std::string::npos);
        REQUIRE(exporter.warnings().size() == 1);
        CHECK(exporter.warnings().front().find("cannot be referenced") != std::string::npos);
    }
}


TEST_CASE("An imported subtree can be stored as a reference to its source file") {

    ModelLoader models;
    auto imported = models.load(modelPath);
    REQUIRE(imported != nullptr);
    REQUIRE_FALSE(imported->children.empty());

    auto scene = Scene::create();
    imported->name = "head";
    imported->position.set(1, 2, 3);
    scene->add(imported);

    const auto originalNodes = countNodes(*imported);
    REQUIRE(originalNodes > 1);

    ObjectExporterOptions referenced;
    referenced.models = ModelStorage::Reference;
    referenced.images = ImageStorage::Reference;
    referenced.resourcePath = dataFolder;

    SECTION("an unmarked subtree is written out in full, reference mode or not") {

        // Nothing has called setAssetSource yet: without a file to point at
        // there is nothing to reference, and the geometry must still be there.
        ObjectExporter exporter;
        const auto text = exporter.toJson(*scene, referenced);

        const auto j = nlohmann::json::parse(text);
        CHECK(j.contains("geometries"));
        CHECK(text.find("threeppAsset") == std::string::npos);
    }

    // Normalised the way the editor stamps it on import, so that the value the
    // loader writes back after a round trip is the same string.
    setAssetSource(*imported, std::filesystem::weakly_canonical(modelPath));

    SECTION("a marked subtree collapses to a path, and comes back whole") {

        ObjectExporter embedding;
        const auto inlined = embedding.toJson(*scene);

        ObjectExporter exporter;
        const auto text = exporter.toJson(*scene, referenced);

        // The point of the exercise.
        CHECK(text.size() * 20 < inlined.size());

        const auto j = nlohmann::json::parse(text);
        CHECK_FALSE(j.contains("geometries"));
        REQUIRE(j["object"]["children"][0].contains("threeppAsset"));
        CHECK(j["object"]["children"][0]["threeppAsset"]["path"].get<std::string>() ==
              "models/gltf/LeePerrySmith/LeePerrySmith.glb");

        // Portable means portable: the document must not smuggle this
        // machine's absolute directory layout in through userData. The
        // assetSource mark is re-stamped from the resolved path on load, so
        // the relative threeppAsset.path is the only copy the file needs.
        CHECK(text.find(dataFolder.generic_string()) == std::string::npos);

        ObjectLoader loader;
        loader.setResourcePath(dataFolder);
        auto parsed = loader.parse(text);

        REQUIRE(parsed != nullptr);
        REQUIRE(parsed->children.size() == 1);

        auto* restored = parsed->children.front();
        CHECK(restored->name == "head");
        CHECK(restored->uuid == imported->uuid);
        CHECK_THAT(restored->position.x, WithinAbs(1.f, 1e-5));
        CHECK_THAT(restored->position.z, WithinAbs(3.f, 1e-5));
        CHECK(countNodes(*restored) == originalNodes);
        // Still linked, so a re-save references it again rather than silently
        // baking it back in.
        CHECK(assetSource(*restored) == std::filesystem::weakly_canonical(modelPath));

        // And re-saving reproduces the document byte for byte. The recorded
        // path is the one place a round trip could drift (absolute vs relative,
        // '..' vs normalised), which would make every autosave a whole-file
        // diff.
        ObjectExporter again;
        CHECK(again.toJson(*parsed, referenced) == text);
    }

    SECTION("edits made after the import survive the round trip") {

        auto flat = descendants(*imported);
        REQUIRE_FALSE(flat.empty());

        // First and last node of the subtree — the same one when the asset has
        // only a single mesh under its root, which is fine: what matters is
        // that both groups of fields make the trip.
        auto* moved = flat.front();
        auto* hidden = flat.back();

        moved->position.set(7, 8, 9);
        moved->userData["tag"] = std::string("kept");
        hidden->visible = false;
        hidden->castShadow = true;
        hidden->renderOrder = 4;

        ObjectExporter exporter;
        const auto text = exporter.toJson(*scene, referenced);
        CHECK(exporter.warnings().empty());

        ObjectLoader loader;
        loader.setResourcePath(dataFolder);
        auto parsed = loader.parse(text);
        REQUIRE(parsed != nullptr);

        auto restored = descendants(*parsed->children.front());
        REQUIRE(restored.size() == flat.size());

        CHECK_THAT(restored.front()->position.x, WithinAbs(7.f, 1e-5));
        CHECK_THAT(restored.front()->position.z, WithinAbs(9.f, 1e-5));
        CHECK(std::any_cast<std::string>(restored.front()->userData.at("tag")) == "kept");

        CHECK(restored.back()->visible == false);
        CHECK(restored.back()->castShadow == true);
        CHECK(restored.back()->renderOrder == 4);
    }

    SECTION("a missing asset leaves a placeholder rather than failing the load") {

        setAssetSource(*imported, dataFolder / "models" / "gone.glb");

        ObjectExporter exporter;
        const auto text = exporter.toJson(*scene, referenced);

        ObjectLoader loader;
        loader.setResourcePath(dataFolder);
        auto parsed = loader.parse(text);

        // The rest of the scene still opens, the node keeps its place, and the
        // reason is on the record.
        REQUIRE(parsed != nullptr);
        REQUIRE(parsed->children.size() == 1);
        CHECK(parsed->children.front()->name == "head");
        CHECK(parsed->children.front()->children.empty());
        CHECK_THAT(parsed->children.front()->position.y, WithinAbs(2.f, 1e-5));

        REQUIRE_FALSE(loader.warnings().empty());
        CHECK(loader.warnings().front().find("could not re-import") != std::string::npos);
    }

    SECTION("edits that no longer match the asset are dropped, not misapplied") {

        auto flat = descendants(*imported);
        REQUIRE_FALSE(flat.empty());
        flat.front()->position.set(7, 8, 9);

        ObjectExporter exporter;
        auto j = nlohmann::json::parse(exporter.toJson(*scene, referenced));

        // Stand in for "somebody edited the .glb": the recorded name no longer
        // identifies the node sitting at that index.
        auto& nodes = j["object"]["children"][0]["threeppAsset"]["nodes"];
        REQUIRE_FALSE(nodes.empty());
        nodes[0]["name"] = "a node that is not there any more";

        ObjectLoader loader;
        loader.setResourcePath(dataFolder);
        auto parsed = loader.parse(j.dump());
        REQUIRE(parsed != nullptr);

        auto restored = descendants(*parsed->children.front());
        REQUIRE_FALSE(restored.empty());
        // The moved transform was NOT applied to whatever now sits at index 0.
        CHECK_THAT(restored.front()->position.x, WithinAbs(0.f, 1e-4));

        REQUIRE_FALSE(loader.warnings().empty());
        CHECK(loader.warnings().front().find("has changed since the document was saved") !=
              std::string::npos);
    }
}


TEST_CASE("A referenced URDF comes back as a live Robot") {

    // The .urdf branch of the loader's re-import dispatch. Everything else in
    // this file goes through ModelLoader; a robot goes through URDFLoader and
    // must come back articulated, not as a frozen Group.
    const auto urdfPath = std::filesystem::weakly_canonical(
            dataFolder / "urdf" / "lbr_iiwa_14_r820.urdf");

    URDFLoader urdf;
    auto robot = urdf.load(urdfPath);
    REQUIRE(robot != nullptr);
    REQUIRE(robot->numDOF() > 0);
    const auto dof = robot->numDOF();

    setAssetSource(*robot, urdfPath);

    auto scene = Scene::create();
    scene->add(robot);

    ObjectExporterOptions referenced;
    referenced.models = ModelStorage::Reference;
    referenced.resourcePath = dataFolder;

    ObjectExporter exporter;
    const auto text = exporter.toJson(*scene, referenced);

    ObjectLoader loader;
    loader.setResourcePath(dataFolder);
    auto parsed = loader.parse(text);

    REQUIRE(parsed != nullptr);
    REQUIRE(parsed->children.size() == 1);

    auto* restored = parsed->children.front()->as<Robot>();
    REQUIRE(restored != nullptr);
    CHECK(restored->numDOF() == dof);
    CHECK(restored->uuid == robot->uuid);
    CHECK(countNodes(*restored) == countNodes(*robot));
}


TEST_CASE("save() makes references relative to the file it is writing") {

    // Everything this case writes lives under the OS temp directory — never in
    // the threepp_data checkout, so a run that dies mid-test strands nothing
    // in a git working tree. The asset is copied in so that document and asset
    // genuinely share a root, which is what the relative path asserts on.
    const auto root = std::filesystem::temp_directory_path() / "threepp_asset_reference_test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);// a stranded earlier run
    std::filesystem::create_directories(root / "models", ec);
    REQUIRE_FALSE(ec);

    const auto assetPath = root / "models" / "head.glb";
    std::filesystem::copy_file(modelPath, assetPath,
                               std::filesystem::copy_options::overwrite_existing, ec);
    REQUIRE_FALSE(ec);

    ModelLoader models;
    auto imported = models.load(assetPath);
    REQUIRE(imported != nullptr);
    setAssetSource(*imported, assetPath);

    auto scene = Scene::create();
    scene->add(imported);

    // Written next to the assets, so the reference should come out relative
    // without the caller having to say anything about paths.
    const auto documentPath = root / "scene.json";

    ObjectExporterOptions options;
    options.models = ModelStorage::Reference;

    ObjectExporter exporter;
    exporter.save(*scene, documentPath, options);

    std::ifstream in(documentPath, std::ios::binary);
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    in.close();

    const auto j = nlohmann::json::parse(text);
    CHECK(j["object"]["children"][0]["threeppAsset"]["path"].get<std::string>() ==
          "models/head.glb");

    // And load() infers the same base directory from the document's own path.
    ObjectLoader loader;
    auto parsed = loader.load(documentPath);
    REQUIRE(parsed != nullptr);
    REQUIRE(parsed->children.size() == 1);
    CHECK(countNodes(*parsed->children.front()) == countNodes(*imported));

    std::filesystem::remove_all(root, ec);
}
