
#ifndef THREEPP_TEXTURELOADER_HPP
#define THREEPP_TEXTURELOADER_HPP

#include "ImageLoader.hpp"

#include "threepp/textures/Texture.hpp"


namespace threepp {

    class TextureLoader {

    public:
        std::shared_ptr<Texture> loadTexture(const char *path);

    private:
        ImageLoader imageLoader_{};
    };

}// namespace threepp

#endif//THREEPP_TEXTURELOADER_HPP
