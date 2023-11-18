
#ifndef THREEPP_CUBETEXTURELOADER_HPP
#define THREEPP_CUBETEXTURELOADER_HPP

#include "threepp/textures/CubeTexture.hpp"
#include "threepp/loaders/ImageLoader.hpp"

#include <array>
#include <filesystem>

namespace threepp {

    class CubeTextureLoader {

    public:
        std::shared_ptr<CubeTexture> load(const std::array<std::filesystem::path, 6>& paths) {

            std::vector<Image> images;
            for (const auto& path: paths) {
                const auto load = loader.load(path, 3, false);
                images.emplace_back(*load);
            }

            auto texture = CubeTexture::create(images);
            texture->needsUpdate();

            return texture;
        }

    private:
        ImageLoader loader;

    };

}

#endif//THREEPP_CUBETEXTURELOADER_HPP
