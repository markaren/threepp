
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
        std::shared_ptr<Texture> loadTexture(const std::filesystem::path& path, bool flipY = true);

#ifdef THREEPP_WITH_CURL
        std::shared_ptr<Texture> loadFromUrl(const std::string& url, bool flipY = true);
#endif
        void clearCache();

        ~TextureLoader();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}// namespace threepp

#endif//THREEPP_TEXTURELOADER_HPP
