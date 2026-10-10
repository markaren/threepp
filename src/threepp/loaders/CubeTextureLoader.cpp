
#include "threepp/loaders/CubeTextureLoader.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>
#include <vector>

using namespace threepp;

namespace {

    // The rule TextureLoader applies to a 2D image: the extension, case-insensitive.
    bool isJpegPath(const std::filesystem::path& path) {
        auto ext = path.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return ext == ".jpg" || ext == ".jpeg";
    }

}// namespace

std::shared_ptr<CubeTexture> CubeTextureLoader::load(const std::array<std::filesystem::path, 6>& paths,
                                                     ColorSpace colorSpace) {

    // One format for the set. Six JPEGs upload as RGB; anything else, including
    // a mix, as RGBA, with every face decoded to that channel count. The format
    // used to follow whichever face came last while each face was decoded by
    // its own extension, so a mixed set declared one channel count and held
    // another.
    const bool allJpeg = std::all_of(paths.begin(), paths.end(), isJpegPath);
    const int channels = allJpeg ? 3 : 4;

    std::vector<Image> images;
    images.reserve(paths.size());
    for (const auto& path : paths) {
        auto load = loader.load(path, channels, false);
        if (!load) {
            // ImageLoader returns nullopt for a path that does not exist or
            // will not decode. A mistyped face path is the single most likely
            // way to call this wrongly, so fail the way every other loader does.
            std::cerr << "[CubeTextureLoader] Cannot load cube face: " << path << "\n";
            return nullptr;
        }
        images.emplace_back(std::move(*load));
    }

    auto texture = CubeTexture::create(images);
    texture->format = allJpeg ? Format::RGB : Format::RGBA;

    // The same unpack-alignment trap TextureLoader documents at length: a
    // 3-channel face whose row stride is not 4-aligned is misread under GL's
    // default alignment of 4, and the result reads as a greyscale version of
    // the texture rather than as anything obviously broken.
    if (texture->format == Format::RGB && (images[0].width() * 3u) % 4u != 0u) {
        texture->unpackAlignment = 1;
    }

    texture->colorSpace = colorSpace;
    texture->needsUpdate();

    return texture;
}
