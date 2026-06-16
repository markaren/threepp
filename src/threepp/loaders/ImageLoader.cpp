
#include "threepp/loaders/ImageLoader.hpp"

#ifndef STB_IMAGE_IMPLEMENTATION
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <cstring>

using namespace threepp;

namespace {

    struct ImageStruct {

        int width{};
        int height{};
        int channels;
        unsigned char* pixels = nullptr;

        ImageStruct(const std::vector<unsigned char>& data, int channels): channels(channels) {
            pixels = stbi_load_from_memory(data.data(), static_cast<int>(data.size()),
                                           &width, &height, nullptr, channels);
        }

        ImageStruct(const std::filesystem::path& imagePath, int channels): channels(channels) {
            pixels = stbi_load(imagePath.string().c_str(), &width, &height, nullptr, channels);
        }

        [[nodiscard]] bool ok() const noexcept { return pixels != nullptr; }

        [[nodiscard]] std::vector<unsigned char> get(bool flipY) const {
            const size_t rowBytes = static_cast<size_t>(channels) * width;
            const size_t total = rowBytes * height;
            std::vector<unsigned char> result(total);
            if (flipY) {
                for (int y = 0; y < height; ++y) {
                    const int srcY = height - 1 - y;
                    std::memcpy(result.data() + static_cast<size_t>(y) * rowBytes,
                                pixels + static_cast<size_t>(srcY) * rowBytes,
                                rowBytes);
                }
            } else {
                std::memcpy(result.data(), pixels, total);
            }
            return result;
        }

        ~ImageStruct() {
            stbi_image_free(pixels);
        }
    };

}// namespace

std::optional<Image> ImageLoader::load(const std::filesystem::path& imagePath, int channels, bool flipY) {

    if (!std::filesystem::exists(imagePath)) {
        return std::nullopt;
    }

    ImageStruct image{imagePath, channels};
    if (!image.ok()) return std::nullopt;

    return Image{
            image.get(flipY),
            static_cast<unsigned int>(image.width),
            static_cast<unsigned int>(image.height)};
}

std::optional<Image> ImageLoader::load(const std::vector<unsigned char>& data, int channels, bool flipY) {

    ImageStruct image{data, channels};
    if (!image.ok()) return std::nullopt;

    return Image{
            image.get(flipY),
            static_cast<unsigned int>(image.width),
            static_cast<unsigned int>(image.height)};
}
