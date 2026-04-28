
#ifndef THREEPP_TEXTURELOADER_HPP
#define THREEPP_TEXTURELOADER_HPP

#include "threepp/textures/Texture.hpp"

#include <filesystem>
#include <memory>

namespace threepp {

    class TextureLoader {

    public:
        explicit TextureLoader(bool useCache = true);

        // Load a texture from disk. The default overload tags the result
        // `NoColorSpace` (matches three.js's TextureLoader). For diffuse/albedo/
        // emissive maps pass `ColorSpace::sRGB` so the sampler decodes from sRGB
        // to linear. Data textures (normal, metallic/roughness, occlusion,
        // displacement) should stay `NoColorSpace`.
        std::shared_ptr<Texture> load(const std::filesystem::path& path,
                                      ColorSpace colorSpace,
                                      bool flipY = true);

        std::shared_ptr<Texture> load(const std::filesystem::path& path, bool flipY = true);

        std::shared_ptr<Texture> loadFromMemory(const std::string& name, const std::vector<unsigned char>& data, bool flipY = true);

        std::shared_ptr<Texture> loadFromMemory(const std::string& name, const std::vector<unsigned char>& data,
                                                ColorSpace colorSpace, bool flipY = true);

        void clearCache();

        ~TextureLoader();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_TEXTURELOADER_HPP
