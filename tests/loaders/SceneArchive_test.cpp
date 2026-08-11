
#include <catch2/catch_test_macros.hpp>

#include "threepp/geometries/BoxGeometry.hpp"
#include "threepp/loaders/AssetSource.hpp"
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


TEST_CASE("A linked subtree is embedded in an archive, and says so") {

    // Re-importing a .glb from inside a zip needs memory-based import plumbing
    // that does not exist, so an archive writes the subtree out in full — which
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
