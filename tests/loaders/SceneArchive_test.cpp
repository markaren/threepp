
#include <catch2/catch_test_macros.hpp>

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/loaders/AssetSource.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/loaders/TextureLoader.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/utils/ZipReader.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    template<class T = Object3D>
    T* findByUuid(Object3D& root, const std::string& uuid) {

        T* found = nullptr;
        root.traverse([&](Object3D& o) {
            if (!found && o.uuid == uuid) found = dynamic_cast<T*>(&o);
        });
        return found;
    }

    std::vector<unsigned char> fileBytes(const std::filesystem::path& path) {

        std::ifstream in(path, std::ios::binary);
        return std::vector<unsigned char>{std::istreambuf_iterator<char>(in),
                                          std::istreambuf_iterator<char>()};
    }

    std::shared_ptr<Texture> firstMap(Object3D& root) {

        std::shared_ptr<Texture> found;
        root.traverse([&](Object3D& o) {
            if (found) return;
            if (auto* mesh = o.as<Mesh>()) {
                if (auto* material = mesh->materialAs<MeshStandardMaterial>(); material && material->map) {
                    found = material->map;
                }
            }
        });
        return found;
    }

}// namespace


TEST_CASE("A scene saves and loads as one .tpz") {

    const auto dir = std::filesystem::temp_directory_path() / "threepp-scene-archive-test";
    std::filesystem::create_directories(dir);

    // A P6 PPM, written by hand rather than encoded, with two rows of different
    // colours: the archive stores the file's own bytes, so row order is the
    // thing that proves it came back through the same contract Reference mode
    // has (an import, flipY defaulted to true) and not the embedded one.
    const auto imageFile = dir / "rows.ppm";
    {
        std::ofstream out(imageFile, std::ios::binary | std::ios::trunc);
        out << "P6\n2 2\n255\n";
        const unsigned char rows[] = {
                255, 0, 0, /**/ 255, 0, 0,
                0, 0, 255, /**/ 0, 0, 255,
        };
        out.write(reinterpret_cast<const char*>(rows), sizeof(rows));
    }

    TextureLoader textures;
    auto texture = textures.load(imageFile);
    REQUIRE(texture != nullptr);
    const std::vector<unsigned char> pixels = texture->image().data<unsigned char>();

    auto material = MeshStandardMaterial::create();
    material->map = texture;

    // A data geometry, whose numbers are what the archive moves out of the JSON
    // and into a binary section, next to a parametric one that has no numbers to
    // move — the format has to keep carrying both.
    auto geometry = BufferGeometry::create();
    geometry->setAttribute("position", FloatBufferAttribute::create(
                                               std::vector<float>{0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0}, 3));
    geometry->setAttribute("uv", Uint16BufferAttribute::create(
                                         std::vector<std::uint16_t>{0, 0, 65535, 0, 65535, 65535, 0, 65535}, 2, true));
    geometry->setIndex(std::vector<unsigned int>{0, 1, 2, 2, 3, 0});

    auto scene = Scene::create();
    auto mesh = Mesh::create(geometry, material);
    auto box = Mesh::create(BoxGeometry::create(2, 3, 4), MeshStandardMaterial::create());
    scene->add(mesh);
    scene->add(box);

    const auto path = dir / "scene.tpz";

    ObjectExporter exporter;
    exporter.save(*scene, path);

    // The layout is a zipped project folder: the document at the root, the
    // geometry's numbers under buffers/ named by its uuid, the texture's own
    // file bytes under images/.
    {
        ZipReader archive(path);
        const auto names = archive.names();
        CHECK(std::find(names.begin(), names.end(), "buffers/" + geometry->uuid + ".bin") != names.end());

        const auto document = archive.read("scene.json");
        const std::string text{document.begin(), document.end()};
        // The numbers left the JSON. "array" survives only where nothing points
        // at a section (there is none here), so its absence is the property.
        CHECK(text.find("\"array\"") == std::string::npos);
    }

    ObjectLoader loader;
    auto parsed = loader.load(path);
    REQUIRE(parsed != nullptr);

    auto* parsedMesh = findByUuid<Mesh>(*parsed, mesh->uuid);
    REQUIRE(parsedMesh != nullptr);
    REQUIRE(parsedMesh->geometry() != nullptr);

    const auto* position = parsedMesh->geometry()->getAttribute<float>("position");
    const auto* uv = parsedMesh->geometry()->getAttribute<std::uint16_t>("uv");
    REQUIRE(position != nullptr);
    REQUIRE(uv != nullptr);
    REQUIRE(parsedMesh->geometry()->hasIndex());

    // A narrow attribute goes through as its stored integers with its
    // `normalized` flag intact, same as the JSON form: the section carries the
    // bytes, the entry carries the schema.
    CHECK(position->array() == geometry->getAttribute<float>("position")->array());
    CHECK(uv->array() == geometry->getAttribute<std::uint16_t>("uv")->array());
    CHECK(parsedMesh->geometry()->getIndex()->array() == std::vector<unsigned int>{0, 1, 2, 2, 3, 0});

    auto* parsedMaterial = parsedMesh->materialAs<MeshStandardMaterial>();
    REQUIRE(parsedMaterial != nullptr);
    REQUIRE(parsedMaterial->map != nullptr);
    CHECK(parsedMaterial->map->image().data<unsigned char>() == pixels);

    // The parametric geometry has no numbers to move and must survive anyway.
    REQUIRE(findByUuid<Mesh>(*parsed, box->uuid) != nullptr);

    // Same scene, same bytes: the archive is what the autosave diffs.
    const auto again = dir / "again.tpz";
    exporter.save(*scene, again);
    CHECK(fileBytes(path) == fileBytes(again));
}


TEST_CASE("A texture that came out of a .glb keeps the bytes it came in as") {

    // The texture with no file: a .glb carries its images inside itself, so the
    // archive's copy-the-source-file path has nothing to copy and every one of
    // them used to be re-encoded to PNG on every save. The encoded bytes are
    // retained at import instead, and they are just as original as a file's.
    const std::filesystem::path model =
            std::filesystem::path(DATA_FOLDER) / "models" / "gltf" / "Soldier.glb";

    ModelLoader models;
    auto imported = models.load(model);
    REQUIRE(imported != nullptr);

    auto texture = firstMap(*imported);
    REQUIRE(texture != nullptr);
    REQUIRE_FALSE(texture->encodedSource.empty());
    const std::vector<unsigned char> pixels = texture->image().data<unsigned char>();
    REQUIRE(pixels.size() > 4);

    auto scene = Scene::create();
    scene->add(imported);

    const auto dir = std::filesystem::temp_directory_path() / "threepp-scene-archive-glb-test";
    std::filesystem::create_directories(dir);
    const auto path = dir / "glb.tpz";

    ObjectExporter exporter;
    exporter.save(*scene, path);

    // Stored under the format they actually are, and byte for byte what the
    // .glb held — a re-encode would be a different file of a different length.
    {
        ZipReader archive(path);
        const auto name = "images/" + texture->uuid() + "-image" + texture->encodedSource.extension;
        REQUIRE(archive.has(name));
        CHECK(archive.read(name) == *texture->encodedSource.bytes);
    }

    ObjectLoader loader;
    auto parsed = loader.load(path);
    REQUIRE(parsed != nullptr);

    auto reloaded = firstMap(*parsed);
    REQUIRE(reloaded != nullptr);

    // The flipY contract, decided by a pixel rather than by argument: glTF
    // decodes its images top-down, so the archive entry has to say so and the
    // texture has to come back the same way up it went in. The corner is where
    // a flip shows first; the whole array is the proof it is the same image.
    const auto& back = reloaded->image().data<unsigned char>();
    REQUIRE(back.size() == pixels.size());
    CHECK(std::vector<unsigned char>(back.begin(), back.begin() + 4) ==
          std::vector<unsigned char>(pixels.begin(), pixels.begin() + 4));
    CHECK(back == pixels);
}


TEST_CASE("A linked .glb rides inside the archive and comes back linked") {

    const std::filesystem::path model = std::filesystem::path(DATA_FOLDER) /
                                        "models" / "gltf" / "LeePerrySmith" / "LeePerrySmith.glb";

    ModelLoader models;
    auto imported = models.load(model);
    REQUIRE(imported != nullptr);
    setAssetSource(*imported, std::filesystem::weakly_canonical(model));

    imported->name = "head";
    imported->position.set(1, 2, 3);

    // An edit INSIDE the subtree, which is the reason a reference carries a
    // table of overrides and not just a path.
    std::vector<Object3D*> nodes;
    imported->traverse([&](Object3D& o) { if (&o != imported.get()) nodes.push_back(&o); });
    REQUIRE_FALSE(nodes.empty());
    nodes.back()->visible = false;
    const auto nodeCount = nodes.size();

    auto scene = Scene::create();
    scene->add(imported);

    const auto dir = std::filesystem::temp_directory_path() / "threepp-scene-archive-asset-test";
    std::filesystem::create_directories(dir);
    const auto path = dir / "linked.tpz";

    ObjectExporterOptions options;
    options.models = ModelStorage::Reference;

    ObjectExporter exporter;
    exporter.save(*scene, path, options);
    CHECK(exporter.warnings().empty());

    // The asset itself is a member of the archive, and none of its vertices are
    // in the document: a reference inside an archive is a reference to the
    // archive's own copy.
    {
        ZipReader archive(path);
        const auto names = archive.names();
        CHECK(std::find(names.begin(), names.end(), "assets/0_LeePerrySmith.glb") != names.end());

        const auto document = archive.read("scene.json");
        const std::string text{document.begin(), document.end()};
        CHECK(text.find("\"geometries\"") == std::string::npos);
    }

    ObjectLoader loader;
    auto parsed = loader.load(path);
    REQUIRE(parsed != nullptr);
    REQUIRE(parsed->children.size() == 1);

    auto* restored = parsed->children.front();
    std::vector<Object3D*> back;
    restored->traverse([&](Object3D& o) { if (&o != restored) back.push_back(&o); });

    CHECK(restored->name == "head");
    CHECK(back.size() == nodeCount);
    CHECK(!back.empty());
    CHECK((!back.empty() && !back.back()->visible));

    // Re-saving copies the asset straight out of the archive it was loaded from
    // and into the new one — which is what the '|' mark on the restored subtree
    // is for, and which has to land on the same bytes as the first save.
    const auto again = dir / "again.tpz";
    ObjectExporter resave;
    resave.save(*parsed, again, options);
    CHECK(resave.warnings().empty());
    CHECK(fileBytes(path) == fileBytes(again));
}


TEST_CASE("A linked subtree an archive cannot carry is embedded, and says so") {

    // Only a self-contained format may travel in an archive, and only when its
    // bytes can actually be read. Anything else is written out in full — which
    // the binary sections make cheap, and which is the whole point. The caller
    // asked for a reference and did not get one, so it has to be told which.
    const auto dir = std::filesystem::temp_directory_path() / "threepp-scene-archive-linked-test";
    std::filesystem::create_directories(dir);

    auto scene = Scene::create();
    auto imported = Mesh::create(BoxGeometry::create(), MeshStandardMaterial::create());
    setAssetSource(*imported, dir / "somewhere.glb");
    scene->add(imported);

    ObjectExporterOptions options;
    options.models = ModelStorage::Reference;

    ObjectExporter exporter;
    exporter.save(*scene, dir / "linked.tpz", options);

    REQUIRE(!exporter.warnings().empty());
    CHECK(exporter.warnings().front().find("somewhere.glb") != std::string::npos);

    // The file it points at was never written, so a document that referenced it
    // could not come back at all. This one does.
    ObjectLoader loader;
    auto parsed = loader.load(dir / "linked.tpz");
    REQUIRE(parsed != nullptr);
    CHECK(findByUuid<Mesh>(*parsed, imported->uuid) != nullptr);
}
