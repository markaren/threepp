
#include "threepp/loaders/ImageLoader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

using namespace threepp;

Image ImageLoader::load(const std::string &imagePath, int channels) {

    int width, height, nrChannels;
    unsigned char *img_data = stbi_load(imagePath.c_str(), &width, &height, &nrChannels, channels);

    return Image(width, height, std::shared_ptr<unsigned char>(img_data));
}

