
#include "threepp/loaders/TextureLoader.hpp"

#include "threepp/loaders/ImageLoader.hpp"
#include "threepp/utils/URLFetcher.hpp"

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

        if (useCache_ && cache_.count(name)) {
            auto cached = cache_[name];
            if (!cached.expired()) {
                tex = cached.lock();
            } else {
                cache_.erase(name);
            }
        }

        return tex;
    }

    std::shared_ptr<Texture> load(const std::filesystem::path& path, bool flipY) {

        if (auto cachedTexture = checkCache(path.string())) {

            return cachedTexture;
        }

        if (!std::filesystem::exists(path)) {
            std::cerr << "[TextureLoader] No such file: '" << absolute(path).string() << "'!" << std::endl;
            return nullptr;
        }

        bool isJPEG = checkIsJPEG(path.string());

        auto image = imageLoader_.load(path, isJPEG ? 3 : 4, flipY);

        auto texture = Texture::create(image);
        texture->name = path.stem().string();

        texture->format = isJPEG ? RGBFormat : RGBAFormat;
        texture->needsUpdate();

        if (useCache_) cache_[path.string()] = texture;

        return texture;
    }

    std::shared_ptr<Texture> loadFromMemory(const std::string& name, const std::vector<unsigned char>& data, bool flipY) {

        if (auto cachedTexture = checkCache(name)) {

            return cachedTexture;
        }

        bool isJPEG = checkIsJPEG(name);

        auto image = imageLoader_.load(data, isJPEG ? 3 : 4, flipY);

        auto texture = Texture::create(image);
        texture->name = name;

        texture->format = isJPEG ? RGBFormat : RGBAFormat;
        texture->needsUpdate();

        if (useCache_) cache_[name] = texture;

        return texture;
    }

    std::shared_ptr<Texture> loadFromUrl(const std::string& url, bool flipY) {

        if (auto cachedTexture = checkCache(url)) {

            return cachedTexture;
        }

        bool isJPEG = checkIsJPEG(url);

        utils::UrlFetcher urlFetcher;
        std::vector<unsigned char> stream;
        bool res = urlFetcher.fetch(url, stream);

        if (res && !stream.empty()) {

            auto image = imageLoader_.load(stream, isJPEG ? 3 : 4, flipY);
            auto texture = Texture::create(image);

            texture->format = isJPEG ? RGBFormat : RGBAFormat;
            texture->needsUpdate();

            if (useCache_) cache_[url] = texture;

            return texture;

        } else {

            std::cerr << "[TextureLoader] Failed loading texture from URL: " << url << std::endl;

            return nullptr;
        }
    }
};

TextureLoader::TextureLoader(bool useCache)
    : pimpl_(std::make_unique<Impl>(useCache)) {}


std::shared_ptr<Texture> TextureLoader::loadTexture(const std::filesystem::path& path, bool flipY) {

    return pimpl_->load(path, flipY);
}

std::shared_ptr<Texture> TextureLoader::load(const std::filesystem::path& path, bool flipY) {

    return pimpl_->load(path, flipY);
}

std::shared_ptr<Texture> TextureLoader::loadFromMemory(const std::string& name, const std::vector<unsigned char>& data, bool flipY) {

    return pimpl_->loadFromMemory(name, data, flipY);
}

std::shared_ptr<Texture> TextureLoader::loadFromUrl(const std::string& url, bool flipY) {

    return pimpl_->loadFromUrl(url, flipY);
}

void TextureLoader::clearCache() {

    pimpl_->cache_.clear();
}

TextureLoader::~TextureLoader() = default;
