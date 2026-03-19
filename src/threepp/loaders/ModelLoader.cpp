
#include "threepp/loaders/ModelLoader.hpp"

#include "threepp/loaders/ColladaLoader.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/OBJLoader.hpp"
#include "threepp/loaders/STLLoader.hpp"
#include "threepp/materials/MeshPhongMaterial.hpp"
#include "threepp/objects/Mesh.hpp"

#include <iostream>
#include <threepp/utils/StringUtils.hpp>

using namespace threepp;

std::shared_ptr<Group> ModelLoader::load(const std::filesystem::path& path) {
    const auto ext = utils::toLower(path.extension().string());

    if (!exists(path)) {
        std::cerr << "[ModelLoader] File does not exist: " << std::filesystem::absolute(path).string() << std::endl;
        return nullptr;
    }

    if (ext == ".obj") {
        thread_local OBJLoader loader;
        return loader.load(path);
    }

    if (ext == ".dae") {
        thread_local ColladaLoader loader;
        return loader.load(path);
    }

    if (ext == ".gltf" || ext == ".glb") {
        thread_local GLTFLoader loader;
        auto result = loader.load(path);
        if (result) return result->scene;
        std::cerr << "[ModelLoader] GLTFLoader returned no result for '" << path << "'." << std::endl;
        return nullptr;
    }

    if (ext == ".stl") {
        thread_local STLLoader loader;
        auto geometry = loader.load(path);
        if (!geometry) return nullptr;
        auto group = Group::create();
        group->add(Mesh::create(geometry, MeshPhongMaterial::create()));
        return group;
    }

    std::cerr << "[ModelLoader] Unsupported file extension '" << ext << "'." << std::endl;
    return nullptr;
}
