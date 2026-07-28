// https://github.com/mrdoob/three.js/blob/r129/src/textures/Texture.js

#ifndef THREEPP_TEXTURE_HPP
#define THREEPP_TEXTURE_HPP

#include "threepp/constants.hpp"

#include "threepp/core/EventDispatcher.hpp"

#include "threepp/math/Matrix3.hpp"
#include "threepp/math/Vector2.hpp"

#include "threepp/textures/Image.hpp"

#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>

namespace threepp {

    class Texture: public EventDispatcher {

    public:
        inline static Mapping DEFAULT_MAPPING = Mapping::UV;

        // Atomic — see Object3D::id. Textures are the most exposed of the four:
        // a model load creates them on the worker thread via TextureLoader, and
        // the Vulkan backend skips re-uploading the environment map when
        // `tex->id` matches the one already uploaded.
        unsigned int id = textureId.fetch_add(1, std::memory_order_relaxed);

        std::string name;

        // Where the pixels came from, when they came from a file on disk. Set by
        // TextureLoader; empty for textures built in memory (procedural ones, and
        // the ones a .glb/.fbx carries inside itself).
        //
        // Purely informational to the renderer — it exists so ObjectExporter can
        // write a path reference instead of a base64 data-URI. A texture without
        // one can only ever be embedded.
        std::filesystem::path sourceFile;

        Mapping mapping = DEFAULT_MAPPING;

        TextureWrapping wrapS{TextureWrapping::ClampToEdge};
        TextureWrapping wrapT{TextureWrapping::ClampToEdge};

        Filter magFilter{Filter::Linear};
        Filter minFilter{Filter::LinearMipmapLinear};

        int anisotropy = 1;

        Format format{Format::RGBA};
        std::optional<std::string> internalFormat;
        Type type{Type::UnsignedByte};

        Vector2 offset{0, 0};
        Vector2 repeat{1, 1};
        Vector2 center{0, 0};
        float rotation = 0;

        int texCoord = 0;  // UV set index (0 = TEXCOORD_0, 1 = TEXCOORD_1)

        bool matrixAutoUpdate = true;
        Matrix3 matrix{};

        bool generateMipmaps = true;
        bool premultiplyAlpha = false;
        int unpackAlignment = 4;// valid values: 1, 2, 4, 8 (see http://www.khronos.org/opengles/sdk/docs/man/xhtml/glPixelStorei.xml)

        // Color space tag for this texture's pixel data. Default is
        // NoColorSpace (raw data, no transform). Loaders set the appropriate
        // tag — color/albedo and emissive maps → SRGBColorSpace; normal maps,
        // metallic/roughness, occlusion, and other data textures stay
        // NoColorSpace; HDR loaders set RGBEColorSpace.
        //
        // Changing this after the texture is already used by a material does
        // not automatically rebuild the material; call Material::needsUpdate
        // to trigger a recompile.
        ColorSpace colorSpace{ColorSpace::NoColorSpace};

        Texture(const Texture&) = delete;
        Texture& operator=(const Texture&) = delete;
        Texture(Texture&&) = delete;
        Texture& operator=(Texture&&) = delete;

        [[nodiscard]] const std::string& uuid() const;

        // Only serialization round-trips (ObjectLoader) have a reason to call this.
        void setUuid(const std::string& uuid);

        Image& image();

        [[nodiscard]] const Image& image() const;

        [[nodiscard]] std::vector<Image>& images();

        [[nodiscard]] const std::vector<Image>& images() const;

        [[nodiscard]] std::vector<Image>& mipmaps();

        [[nodiscard]] const std::vector<Image>& mipmaps() const;

        void updateMatrix();

        void dispose();

        // void transformUv(Vector2& uv) const;

        void needsUpdate();

        [[nodiscard]] unsigned int version() const;

        Texture& copy(const Texture& source);

        [[nodiscard]] std::shared_ptr<Texture> clone() const;

        ~Texture() override;

        static std::shared_ptr<Texture> create();

        static std::shared_ptr<Texture> create(const Image& image);

        static std::shared_ptr<Texture> create(std::vector<Image> image);

    protected:
        explicit Texture(std::vector<Image> image);

    private:
        std::string uuid_;
        std::vector<Image> images_;
        std::vector<Image> mipmaps_;

        bool disposed_{false};
        unsigned int version_{0};

        inline static std::atomic<unsigned int> textureId{0};
    };

}// namespace threepp

#endif//THREEPP_TEXTURE_HPP
