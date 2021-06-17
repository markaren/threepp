// https://github.com/mrdoob/three.js/blob/r129/src/textures/Texture.js

#ifndef THREEPP_TEXTURE_HPP
#define THREEPP_TEXTURE_HPP

#include "threepp/constants.hpp"
#include "threepp/textures/Image.hpp"
#include "threepp/math/MathUtils.hpp"

#include <optional>

namespace threepp {

    class Texture {

    public:

        unsigned int id = textureId++;

        std::string uuid = generateUUID();

        std::string name = "";

        unsigned int version = 0;

        explicit Texture(
                std::optional<Image> image = std::nullopt,
                int mapping = Texture::DEFAULT_MAPPING,
                int wrapS = ClampToEdgeWrapping,
                int wrapT = ClampToEdgeWrapping,
                int magFilter = LinearFilter,
                int minFilter = LinearMipmapLinearFilter,
                int format = RGBAFormat,
                int type = UnsignedByteType,
                int anisotropy = 1,
                int encoding = LinearEncoding) {}

    private:

        inline static unsigned int textureId = 0;

        inline static int DEFAULT_MAPPING = UVMapping;

    };

}

#endif//THREEPP_TEXTURE_HPP
