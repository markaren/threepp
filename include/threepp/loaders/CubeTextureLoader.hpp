
#ifndef THREEPP_CUBETEXTURELOADER_HPP
#define THREEPP_CUBETEXTURELOADER_HPP

#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include <array>
#include <iostream>
#include <filesystem>

namespace threepp {

    class CubeTextureLoader {

    public:
        std::shared_ptr<CubeTexture> load(const std::array<std::filesystem::path, 6>& paths,
                                          ColorSpace colorSpace = ColorSpace::sRGB) {

            auto checkIsJPEG = [](const std::string& path) {
                return path.find(".jpg") != std::string::npos || path.find(".jpeg") != std::string::npos;
            };

            bool isJPEG{};
            std::vector<Image> images;
            images.reserve(paths.size());
            for (const auto& path : paths) {
                isJPEG = checkIsJPEG(path.string());
                auto load = loader.load(path, isJPEG ? 3 : 4, false);
                if (!load) {
                    // ImageLoader returns nullopt for a path that does not exist
                    // or will not decode. Dereferencing that is undefined, and a
                    // mistyped face path is the single most likely way to call
                    // this wrongly, so fail the way every other loader does.
                    std::cerr << "[CubeTextureLoader] Cannot load cube face: " << path << "\n";
                    return nullptr;
                }
                images.emplace_back(std::move(*load));
            }

            auto texture = CubeTexture::create(images);
            texture->format = isJPEG ? Format::RGB : Format::RGBA;

            // The same unpack-alignment trap TextureLoader documents at length: a
            // 3-channel face whose row stride is not 4-aligned is misread under
            // GL's default alignment of 4, and the result reads as a greyscale
            // version of the texture rather than as anything obviously broken.
            // TextureLoader sets this for 2D JPEGs; cube faces need it just as
            // much, and are more likely to be JPEGs.
            if (texture->format == Format::RGB && (images[0].width() * 3u) % 4u != 0u) {
                texture->unpackAlignment = 1;
            }

            texture->colorSpace = colorSpace;
            texture->needsUpdate();

            return texture;
        }

    private:
        ImageLoader loader;
    };

}// namespace threepp

#endif//THREEPP_CUBETEXTURELOADER_HPP
