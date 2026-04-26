
#ifndef THREEPP_TEXTURELOADER_HPP
#define THREEPP_TEXTURELOADER_HPP

#include "threepp/textures/Texture.hpp"

#include <filesystem>
#include <memory>

namespace threepp {

    class TextureLoader {

    public:
        explicit TextureLoader(bool useCache = true);

        // Load a texture from disk. Tags the result `SRGBColorSpace` by default —
        // suitable for diffuse/albedo/emissive maps. For data textures (normal,
        // metallic/roughness, occlusion, displacement) pass `LinearSRGBColorSpace`
        // (or `NoColorSpace` after Phase 4) so the GPU does not apply sRGB decode.
        std::shared_ptr<Texture> load(const std::filesystem::path& path,
                                      ColorSpace colorSpace,
                                      bool flipY = true);

        // Backward-compatible default — same as load(path, SRGBColorSpace, flipY).
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
