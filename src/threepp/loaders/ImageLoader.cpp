
#include "threepp/loaders/ImageLoader.hpp"

#ifndef STB_IMAGE_IMPLEMENTATION
#if defined(__GNUC__) || defined(__clang__)
// Temporarily disable specific warnings
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"    // Disables all warnings
#pragma GCC diagnostic ignored "-Wextra"  // Disables extra warnings
#pragma GCC diagnostic ignored "-Wpedantic"  // Disables pedantic warnings
#endif
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb_image.h"

#if defined(__GNUC__) || defined(__clang__)
// Re-enable the warnings
#pragma GCC diagnostic pop
#endif

using namespace threepp;

namespace {

    struct ImageStruct {

        int width{};
        int height{};
        int channels;
        unsigned char* pixels;

        ImageStruct(const std::vector<unsigned char>& data, int channels, bool flipY): channels(channels) {
            stbi_set_flip_vertically_on_load(flipY);
            pixels = stbi_load_from_memory(data.data(), static_cast<int>(data.size()), &width, &height, nullptr, channels);
        }

        ImageStruct(const std::filesystem::path& imagePath, int channels, bool flipY): channels(channels) {
            stbi_set_flip_vertically_on_load(flipY);
            pixels = stbi_load(imagePath.string().c_str(), &width, &height, nullptr, channels);
        }

        [[nodiscard]] std::vector<unsigned char> get() const {
            return {pixels, pixels + (channels * width * height)};
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

    ImageStruct image{imagePath, channels, flipY};

    return Image{
            image.get(),
            static_cast<unsigned int>(image.width),
            static_cast<unsigned int>(image.height),
            0};
}

std::optional<Image> ImageLoader::load(const std::vector<unsigned char>& data, int channels, bool flipY) {

    ImageStruct image{data, channels, flipY};

    return Image{
            image.get(),
            static_cast<unsigned int>(image.width),
            static_cast<unsigned int>(image.height),
            0};
}
