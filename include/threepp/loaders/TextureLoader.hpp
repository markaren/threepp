
#ifndef THREEPP_TEXTURELOADER_HPP
#define THREEPP_TEXTURELOADER_HPP

#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/textures/Texture.hpp"

#include <filesystem>

namespace threepp {

    class TextureLoader {

    public:
        std::shared_ptr<Texture> loadTexture(const std::filesystem::path& path, bool flipY = true);

    private:
        ImageLoader imageLoader_{};
    };

}// namespace threepp

#endif//THREEPP_TEXTURELOADER_HPP
