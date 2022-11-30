
#include "threepp/loaders/ImageLoader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using namespace threepp;

namespace {

    struct ImageStruct {
        int width;
        int height;
        unsigned char *pixels;
    };

}// namespace

Image ImageLoader::load(const std::filesystem::path &imagePath, int channels, bool flipY) {

    ImageStruct image{};
    stbi_set_flip_vertically_on_load(flipY);
    image.pixels = stbi_load(imagePath.string().c_str(), &image.width, &image.height, nullptr, channels);

    return {static_cast<unsigned int>(image.width),
            static_cast<unsigned int>(image.height),
            std::shared_ptr<unsigned char>(image.pixels),
            flipY};
}
