// What a scene document costs to write and to read back, split by what the
// document is made of.
//
// Not a ctest — run manually. Saving and opening a scene with real imported
// models is dominated by the models: every vertex becomes a JSON number on the
// way out and is parsed back on the way in. This measures that against the two
// reference modes, so the storage setting can be argued about with numbers.
//
// Usage: SceneSerialize_bench <model file> [more model files...]
//
//   SceneSerialize_bench "path/to/Bistro.fbx"
//
// Phases, per storage mode:
//   export     — Object3D graph -> JSON text
//   parse      — JSON text -> Object3D graph (includes re-import when linked)
//   size       — bytes of the document
//
// The archive row is measured against the file rather than against a string,
// because that is what it is: save() writes a .tpz and load() reads it back.
// Its JSON counterpart ("embed all, via file") does the same through a .json,
// so the pair differs in the storage of the numbers and not in the I/O.

#include "threepp/loaders/AssetSource.hpp"
#include "threepp/loaders/ModelLoader.hpp"
#include "threepp/loaders/ObjectExporter.hpp"
#include "threepp/loaders/ObjectLoader.hpp"
#include "threepp/scenes/Scene.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    using Clock = std::chrono::steady_clock;

    double msSince(Clock::time_point start) {

        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    std::size_t countNodes(Object3D& root) {

        std::size_t n = 0;
        root.traverse([&n](Object3D&) { ++n; });
        return n;
    }

    void run(const char* label, Scene& scene, const ObjectExporterOptions& options,
             const std::filesystem::path& resourcePath) {

        ObjectExporter exporter;

        const auto exportStart = Clock::now();
        const auto text = exporter.toJson(scene, options);
        const auto exportMs = msSince(exportStart);

        ObjectLoader loader;
        loader.setResourcePath(resourcePath);

        const auto parseStart = Clock::now();
        auto parsed = loader.parse(text);
        const auto parseMs = msSince(parseStart);

        std::printf("  %-22s export %8.1f ms   parse %8.1f ms   %8.2f MB   %zu nodes\n",
                    label, exportMs, parseMs,
                    static_cast<double>(text.size()) / (1024.0 * 1024.0),
                    parsed ? countNodes(*parsed) : 0);
    }

    // Same phases through a file, which is the only way to measure the archive:
    // its images and its geometry are members of the container, so there is no
    // intermediate string to time.
    void runFile(const char* label, Scene& scene, const ObjectExporterOptions& options,
                 const std::filesystem::path& path) {

        ObjectExporter exporter;

        const auto saveStart = Clock::now();
        exporter.save(scene, path, options);
        const auto saveMs = msSince(saveStart);

        ObjectLoader loader;

        const auto loadStart = Clock::now();
        auto parsed = loader.load(path);
        const auto loadMs = msSince(loadStart);

        std::error_code ec;
        const auto bytes = std::filesystem::file_size(path, ec);

        std::printf("  %-22s save   %8.1f ms   load  %8.1f ms   %8.2f MB   %zu nodes\n",
                    label, saveMs, loadMs,
                    ec ? 0.0 : static_cast<double>(bytes) / (1024.0 * 1024.0),
                    parsed ? countNodes(*parsed) : 0);
    }

}// namespace


int main(int argc, char** argv) {

    if (argc < 2) {
        std::printf("usage: %s <model file> [more model files...]\n", argv[0]);
        return 1;
    }

    auto scene = Scene::create();

    // The assets are the base for relative references, and they may sit in
    // different directories; the first one's directory is as good a stand-in
    // for "where the document lives" as any.
    std::filesystem::path resourcePath;

    for (int i = 1; i < argc; ++i) {

        const std::filesystem::path path{argv[i]};
        if (resourcePath.empty()) resourcePath = path.parent_path();

        const auto start = Clock::now();
        std::shared_ptr<Group> imported;
        try {
            ModelLoader loader;
            imported = loader.load(path);
        } catch (const std::exception& e) {
            std::printf("failed to load %s: %s\n", path.string().c_str(), e.what());
            continue;
        }
        if (!imported) {
            std::printf("failed to load %s\n", path.string().c_str());
            continue;
        }

        std::printf("loaded %s in %.1f ms (%zu nodes)\n",
                    path.filename().string().c_str(), msSince(start), countNodes(*imported));

        setAssetSource(*imported, path);
        scene->add(imported);
    }

    if (scene->children.empty()) return 1;

    std::printf("\nscene: %zu nodes total\n", countNodes(*scene));

    ObjectExporterOptions embedded;
    embedded.resourcePath = resourcePath;
    run("embed all", *scene, embedded, resourcePath);

    ObjectExporterOptions refImages = embedded;
    refImages.images = ImageStorage::Reference;
    run("reference textures", *scene, refImages, resourcePath);

    ObjectExporterOptions refModels = embedded;
    refModels.models = ModelStorage::Reference;
    run("reference models", *scene, refModels, resourcePath);

    ObjectExporterOptions refBoth = embedded;
    refBoth.images = ImageStorage::Reference;
    refBoth.models = ModelStorage::Reference;
    run("reference both", *scene, refBoth, resourcePath);

    // The A and the B of the archive: the same self-contained document, once as
    // JSON with everything inlined and once as the single-file archive.
    const auto dir = std::filesystem::temp_directory_path() / "threepp-scene-bench";
    std::filesystem::create_directories(dir);

    runFile("embed all (file)", *scene, embedded, dir / "scene.json");
    runFile("archive (.tpz)", *scene, embedded, dir / "scene.tpz");

    return 0;
}
