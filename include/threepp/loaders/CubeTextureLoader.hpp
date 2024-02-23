
#ifndef THREEPP_CUBETEXTURELOADER_HPP
#define THREEPP_CUBETEXTURELOADER_HPP

#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/textures/CubeTexture.hpp"

#include <array>
#include <filesystem>

namespace threepp {

    class CubeTextureLoader {

    public:
        std::shared_ptr<CubeTexture> load(const std::array<std::filesystem::path, 6>& paths) {

            auto checkIsJPEG = [](const std::string& path) {
                return path.find(".jpg") != std::string::npos || path.find(".jpeg") != std::string::npos;
            };

            std::vector<Image> images;
            for (const auto& path : paths) {
                bool isJPEG = checkIsJPEG(path.string());
                const auto load = loader.load(path, isJPEG ? 3 : 4, false);
                images.emplace_back(*load);
            }

            auto texture = CubeTexture::create(images);
            texture->needsUpdate();

            return texture;
        }

    private:
        ImageLoader loader;
    };

}// namespace threepp

#endif//THREEPP_CUBETEXTURELOADER_HPP
