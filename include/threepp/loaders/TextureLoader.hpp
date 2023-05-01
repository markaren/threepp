
#ifndef THREEPP_TEXTURELOADER_HPP
#define THREEPP_TEXTURELOADER_HPP

#include "threepp/textures/Texture.hpp"

#include <filesystem>
#include <memory>

namespace threepp {

    class TextureLoader {

    public:
        explicit TextureLoader(bool useCache = true);

        std::shared_ptr<Texture> load(const std::filesystem::path& path, bool flipY = true);

        // clang-format off
        [[deprecated("Function 'loadTexture' deprecated. Use 'load'")]]
        std::shared_ptr<Texture> loadTexture(const std::filesystem::path& path, bool flipY = true);
        // clang-format on

        std::shared_ptr<Texture> loadFromMemory(const std::string& name, const std::vector<unsigned char>& data, bool flipY = true);

        std::shared_ptr<Texture> loadFromUrl(const std::string& url, bool flipY = true);

        void clearCache();

        ~TextureLoader();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_TEXTURELOADER_HPP
