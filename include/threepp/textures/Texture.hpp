// https://github.com/mrdoob/three.js/blob/r129/src/textures/Texture.js

#ifndef THREEPP_TEXTURE_HPP
#define THREEPP_TEXTURE_HPP

#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Vector2.hpp"

#include "threepp/core/EventDispatcher.hpp"

#include "threepp/constants.hpp"
#include "threepp/math/MathUtils.hpp"
#include "threepp/textures/Image.hpp"

#include <memory>
#include <optional>
#include <utility>

namespace threepp {

    class Texture : private EventDispatcher {

    public:
        unsigned int id = textureId++;

        std::string uuid = generateUUID();

        std::optional<Image> image;
        std::vector<Image> mipmaps;

        int mapping;

        int wrapS;
        int wrapT;

        int magFilter;
        int minFilter;

        int anisotropy;

        int format;
        std::optional<int> internalFormat;
        int type;

        Vector2 offset = Vector2(0, 0);
        Vector2 repeat = Vector2(1, 1);
        Vector2 center = Vector2(0, 0);
        float rotation = 0;

        bool matrixAutoUpdate = true;
        Matrix3 matrix;

        bool generateMipmaps = true;
        bool premultiplyAlpha = false;
        bool flipY = true;
        unsigned int unpackAligment = 4;// valid values: 1, 2, 4, 8 (see http://www.khronos.org/opengles/sdk/docs/man/xhtml/glPixelStorei.xml)

        // Values of encoding !== THREE.LinearEncoding only supported on map, envMap and emissiveMap.
        //
        // Also changing the encoding after already used by a Material will not automatically make the Material
        // update. You need to explicitly call Material.needsUpdate to trigger it to recompile.
        int encoding;

        void updateMatrix();

        void dispose();

        void transformUv(Vector2 &uv) const;

        void needsUpdate();

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
                int encoding = LinearEncoding)
            : image(std::move(image)),
              mapping(mapping),
              wrapS(wrapS), wrapT(wrapT),
              magFilter(magFilter), minFilter(minFilter),
              format(format), type(type),
              anisotropy(anisotropy), encoding(encoding) {}

    private:
        unsigned int version_ = 0;

        std::function<void()> onUpdate_;

        inline static unsigned int textureId = 0;

        inline static int DEFAULT_MAPPING = UVMapping;
    };

    //    typedef std::shared_ptr<Texture> TexturePtr;

}// namespace threepp

#endif//THREEPP_TEXTURE_HPP
