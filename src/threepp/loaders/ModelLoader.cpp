
#include "threepp/loaders/ModelLoader.hpp"

#include "threepp/loaders/ColladaLoader.hpp"
#include "threepp/loaders/GLTFLoader.hpp"
#include "threepp/loaders/OBJLoader.hpp"
#include "threepp/loaders/STLLoader.hpp"
#ifdef THREEPP_WITH_USD
#include "threepp/loaders/USDLoader.hpp"
#endif
#ifdef THREEPP_WITH_FBX
#include "threepp/loaders/FBXLoader.hpp"
#endif
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
        loader.setIgnoreUpDirection(ignoreUpDirection_);
        return loader.load(path);
    }

    if (ext == ".gltf" || ext == ".glb") {
        thread_local GLTFLoader loader;
        auto result = loader.load(path);
        if (result) {
            auto scene = result->scene;
            scene->animations = result->animations;
            return scene;
        }
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

#ifdef THREEPP_WITH_USD
    if (ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz") {
        thread_local USDLoader loader;
        loader.setIgnoreUpDirection(ignoreUpDirection_);
        return loader.load(path);
    }
#endif

#ifdef THREEPP_WITH_FBX
    if (ext == ".fbx") {
        thread_local FBXLoader loader;
        return loader.load(path);
    }
#endif

    std::cerr << "[ModelLoader] Unsupported file extension '" << ext << "'." << std::endl;
    return nullptr;
}

std::shared_ptr<AsyncGroup> ModelLoader::loadAsync(const std::filesystem::path& path) {
    return threepp::loadAsync(*this, path);
}

ModelLoader& ModelLoader::setIgnoreUpDirection(bool ignore) {
    ignoreUpDirection_ = ignore;
    return *this;
}
