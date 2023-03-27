
#include "threepp/loaders/ImageLoader.hpp"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#include <stb_image.h>

using namespace threepp;

namespace {

    struct ImageStruct {
        int width;
        int height;
        unsigned char* pixels;
    };

}// namespace

std::optional<Image> ImageLoader::load(const std::filesystem::path& imagePath, int channels, bool flipY) {

    if (!std::filesystem::exists(imagePath)) {
        return std::nullopt;
    }

    ImageStruct image{};
    stbi_set_flip_vertically_on_load(flipY);
    image.pixels = stbi_load(imagePath.string().c_str(), &image.width, &image.height, nullptr, channels);

    return Image{
            std::shared_ptr<unsigned char>(image.pixels, free),
            static_cast<unsigned int>(image.width),
            static_cast<unsigned int>(image.height),
            flipY};
}

std::optional<Image> ImageLoader::load(const std::vector<unsigned char>& data, int channels, bool flipY) {

    ImageStruct image{};
    stbi_set_flip_vertically_on_load(flipY);
    image.pixels = stbi_load_from_memory(data.data(), static_cast<int>(data.size()), &image.width, &image.height, nullptr, channels);

    return Image{
            std::shared_ptr<unsigned char>(image.pixels, free),
            static_cast<unsigned int>(image.width),
            static_cast<unsigned int>(image.height),
            flipY};
}
