
#ifndef THREEPP_TEXTURELOADER_HPP
#define THREEPP_TEXTURELOADER_HPP

#include "ImageLoader.hpp"

#include "threepp/textures/Texture.hpp"

#include <regex>

namespace threepp {

    class TextureLoader {

    public:
        Texture loadTexture(const char *path) {

            bool isJPEG = std::regex_match(path, std::regex(".*jpe?g", std::regex::icase));

            auto image = imageLoader_.load(path, isJPEG ? 3 : 4);

            Texture texture(image);

            texture.format = isJPEG ? RGBFormat : RGBAFormat;
            texture.needsUpdate();

            return texture;
        }

    private:
        ImageLoader imageLoader_{};
    };

}// namespace threepp

#endif//THREEPP_TEXTURELOADER_HPP
