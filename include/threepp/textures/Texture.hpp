// https://github.com/mrdoob/three.js/blob/r129/src/textures/Texture.js

#ifndef THREEPP_TEXTURE_HPP
#define THREEPP_TEXTURE_HPP

#include "threepp/constants.hpp"

#include "threepp/core/EventDispatcher.hpp"

#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Vector2.hpp"

#include "threepp/textures/Image.hpp"

#include <memory>
#include <optional>
#include <utility>

namespace threepp {

    class Texture : public EventDispatcher {

    public:
        inline static int DEFAULT_MAPPING = UVMapping;

        unsigned int id = textureId++;

        std::string uuid = utils::generateUUID();

        std::string name;

        std::optional<Image> image;
        std::vector<Image> mipmaps;

        std::optional<int> mapping = Texture::DEFAULT_MAPPING;

        int wrapS = ClampToEdgeWrapping;
        int wrapT = ClampToEdgeWrapping;

        int magFilter = LinearFilter;
        int minFilter = LinearMipmapLinearFilter;

        int anisotropy = 1;

        int format = RGBAFormat;
        std::optional<std::string> internalFormat;
        int type = UnsignedByteType;

        Vector2 offset = Vector2(0, 0);
        Vector2 repeat = Vector2(1, 1);
        Vector2 center = Vector2(0, 0);
        float rotation = 0;

        bool matrixAutoUpdate = true;
        Matrix3 matrix{};

        bool generateMipmaps = true;
        bool premultiplyAlpha = false;
        int unpackAlignment = 4;// valid values: 1, 2, 4, 8 (see http://www.khronos.org/opengles/sdk/docs/man/xhtml/glPixelStorei.xml)

        // Values of encoding !== THREE.LinearEncoding only supported on map, envMap and emissiveMap.
        //
        // Also changing the encoding after already used by a Material will not automatically make the Material
        // update. You need to explicitly call Material.needsUpdate to trigger it to recompile.
        int encoding = LinearEncoding;

        std::optional<std::function<void(Texture &)>> onUpdate;

        void updateMatrix();

        void dispose();

        void transformUv(Vector2 &uv) const;

        void needsUpdate();

        [[nodiscard]] unsigned int version() const;

        Texture &copy(const Texture &source);

        static std::shared_ptr<Texture> create(std::optional<Image> image = std::nullopt) {

            return std::shared_ptr<Texture>(new Texture(std::move(image)));
        }

    protected:
        explicit Texture(std::optional<Image> image = std::nullopt)
            : image(std::move(image)) {}

    private:
        unsigned int version_ = 0;

        inline static unsigned int textureId = 0;
    };

}// namespace threepp

#endif//THREEPP_TEXTURE_HPP
