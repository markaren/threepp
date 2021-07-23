
#include "threepp/loaders/ImageLoader.hpp"

#include "stb.hpp"

using namespace threepp;

Image ImageLoader::load(const std::string &imagePath, int channels) {

    auto image = stb_load(imagePath, channels);

    return Image(image.width, image.height, std::shared_ptr<unsigned char>(image.pixels));
}
