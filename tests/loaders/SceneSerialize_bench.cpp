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

    return 0;
}
