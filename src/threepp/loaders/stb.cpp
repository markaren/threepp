
#include "stb.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

threepp::ImageStruct threepp::stb_load(const std::string &path, int channels) {

    threepp::ImageStruct image{};
    image.pixels = stbi_load(path.c_str(), &image.width, &image.height, nullptr, channels);

    return image;
}
