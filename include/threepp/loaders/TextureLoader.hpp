
#ifndef THREEPP_TEXTURELOADER_HPP
#define THREEPP_TEXTURELOADER_HPP

#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/textures/Texture.hpp"

#include <filesystem>
#include <memory>
#include <unordered_map>

namespace threepp {

    class TextureLoader {

    public:
        bool useCache = true;

        std::shared_ptr<Texture> load(const std::filesystem::path& path, bool flipY = true);
        std::shared_ptr<Texture> loadTexture(const std::filesystem::path& path, bool flipY = true);
        std::shared_ptr<Texture> loadFromUrl(const std::string& url, bool flipY = true);

        void clearCache();

    private:
        ImageLoader imageLoader_;
        std::unordered_map<std::string, std::weak_ptr<Texture>> cache_;
    };

}// namespace threepp

#endif//THREEPP_TEXTURELOADER_HPP
