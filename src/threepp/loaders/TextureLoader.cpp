
#include "threepp/loaders/TextureLoader.hpp"

#include <regex>

using namespace threepp;

std::shared_ptr<Texture> TextureLoader::loadTexture(const std::filesystem::path& path, bool flipY) {

    if (useCache && cache_.count(path.string())) {
        auto cached = cache_[path.string()];
        if (!cached.expired()) {
            auto tex = cached.lock();
            return tex;
        } else {
            cache_.erase(path.string());
        }
    }

    if (!std::filesystem::exists(path)) {
        std::cerr << "[TextureLoader] No such file: '" << absolute(path).string() << "'!" << std::endl;
        return nullptr;
    }

    static std::regex reg(".*jpe?g", std::regex::icase);

    bool isJPEG = std::regex_match(path.string(), reg);

    auto image = imageLoader_.load(path, isJPEG ? 3 : 4, flipY);

    auto texture = Texture::create(image);
    texture->name = path.stem().string();

    texture->format = isJPEG ? RGBFormat : RGBAFormat;
    texture->needsUpdate();

    if (useCache) cache_[path.string()] = texture;

    return texture;
}

void TextureLoader::clearCache() {

    cache_.clear();
}
