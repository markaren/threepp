
#include "threepp/loaders/AssetSource.hpp"

#include "threepp/core/Object3D.hpp"

#include <any>
#include <string>

using namespace threepp;


void threepp::setAssetSource(Object3D& object, const std::filesystem::path& path) {

    if (path.empty()) {
        clearAssetSource(object);
        return;
    }
    // generic_string() so a document written on Windows resolves on Linux —
    // the path is relative to the scene often enough that the separator
    // matters, and every filesystem here accepts forward slashes.
    object.userData[assetSourceKey] = path.generic_string();
}

std::filesystem::path threepp::assetSource(const Object3D& object) {

    const auto it = object.userData.find(assetSourceKey);
    if (it == object.userData.end()) return {};
    if (it->second.type() != typeid(std::string)) return {};

    const auto& text = std::any_cast<const std::string&>(it->second);
    if (text.empty()) return {};

    return {text};
}

void threepp::clearAssetSource(Object3D& object) {

    object.userData.erase(assetSourceKey);
}
