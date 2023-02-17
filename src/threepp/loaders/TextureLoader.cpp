
#include "threepp/loaders/TextureLoader.hpp"

#include <re2/re2.h>

using namespace threepp;

std::shared_ptr<Texture> TextureLoader::loadTexture(const std::filesystem::path& path, bool flipY) {

    static RE2 r("(?i).*jpe?g");
    bool isJPEG = RE2::PartialMatch(re2::StringPiece(path.string()), r);

    auto image = imageLoader_.load(path, isJPEG ? 3 : 4, flipY);

    auto texture = Texture::create(image);

    texture->format = isJPEG ? RGBFormat : RGBAFormat;
    texture->needsUpdate();

    return texture;
}
