
#include "threepp/loaders/TextureLoader.hpp"

#include <regex>

using namespace threepp;

std::shared_ptr<Texture> TextureLoader::loadTexture(const std::filesystem::path& path, bool flipY) {

    static std::regex reg(".*jpe?g", std::regex::icase);

    bool isJPEG = std::regex_match(path.string(), reg);

    auto image = imageLoader_.load(path, isJPEG ? 3 : 4, flipY);

    auto texture = Texture::create(image);

    texture->format = isJPEG ? RGBFormat : RGBAFormat;
    texture->needsUpdate();

    return texture;
}
