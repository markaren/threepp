
#include "threepp/loaders/TextureLoader.hpp"

#include "threepp/loaders/DDSLoader.hpp"
#include "threepp/loaders/ImageLoader.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <regex>
#include <vector>

using namespace threepp;

namespace {

    bool checkIsJPEG(const std::string& path) {

        static std::regex reg(".*jpe?g", std::regex::icase);

        return std::regex_match(path, reg);
    }

}// namespace

struct TextureLoader::Impl {

    bool useCache_;
    ImageLoader imageLoader_;
    std::unordered_map<std::string, std::weak_ptr<Texture>> cache_;

    explicit Impl(bool useCache): useCache_(useCache) {}

    std::shared_ptr<Texture> checkCache(const std::string& name) {

        std::shared_ptr<Texture> tex;

        if (useCache_ && cache_.contains(name)) {
            auto cached = cache_.at(name);
            if (!cached.expired()) {
                tex = cached.lock();
            } else {
                cache_.erase(name);
            }
        }

        return tex;
    }

    std::shared_ptr<Texture> load(const std::filesystem::path& path, ColorSpace colorSpace, bool flipY) {

        if (auto cachedTexture = checkCache(path.string())) {

            return cachedTexture;
        }

        if (!std::filesystem::exists(path)) {
            std::cerr << "[TextureLoader] No such file: '" << absolute(path).string() << "'!" << std::endl;
            return nullptr;
        }

        {
            auto ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
            if (ext == ".dds") {
                DDSLoader ddsLoader;
                auto texture = ddsLoader.load(path);
                if (texture) {
                    texture->colorSpace = colorSpace;
                    if (useCache_) cache_[path.string()] = texture;
                }
                return texture;
            }
        }

        bool isJPEG = checkIsJPEG(path.string());

        auto image = imageLoader_.load(path, isJPEG ? 3 : 4, flipY);

        auto texture = Texture::create(*image);
        texture->name = path.stem().string();

        texture->format = isJPEG ? Format::RGB : Format::RGBA;
        texture->colorSpace = colorSpace;
        texture->needsUpdate();

        if (useCache_) cache_[path.string()] = texture;

        return texture;
    }

    std::shared_ptr<Texture> loadFromMemory(const std::string& name, const std::vector<unsigned char>& data,
                                            ColorSpace colorSpace, bool flipY) {

        if (auto cachedTexture = checkCache(name)) {

            return cachedTexture;
        }

        bool isJPEG = checkIsJPEG(name);

        auto image = imageLoader_.load(data, isJPEG ? 3 : 4, flipY);

        auto texture = Texture::create(*image);
        texture->name = name;

        texture->format = isJPEG ? Format::RGB : Format::RGBA;
        texture->colorSpace = colorSpace;
        texture->needsUpdate();

        if (useCache_) cache_[name] = texture;

        return texture;
    }
};

TextureLoader::TextureLoader(bool useCache)
    : pimpl_(std::make_unique<Impl>(useCache)) {}

std::shared_ptr<Texture> TextureLoader::load(const std::filesystem::path& path, ColorSpace colorSpace, bool flipY) {

    return pimpl_->load(path, colorSpace, flipY);
}

std::shared_ptr<Texture> TextureLoader::load(const std::filesystem::path& path, bool flipY) {

    return pimpl_->load(path, ColorSpace::sRGB, flipY);
}

std::shared_ptr<Texture> TextureLoader::loadFromMemory(const std::string& name, const std::vector<unsigned char>& data, bool flipY) {

    return pimpl_->loadFromMemory(name, data, ColorSpace::sRGB, flipY);
}

std::shared_ptr<Texture> TextureLoader::loadFromMemory(const std::string& name, const std::vector<unsigned char>& data,
                                                       ColorSpace colorSpace, bool flipY) {

    return pimpl_->loadFromMemory(name, data, colorSpace, flipY);
}

void TextureLoader::clearCache() {

    pimpl_->cache_.clear();
}

TextureLoader::~TextureLoader() = default;
