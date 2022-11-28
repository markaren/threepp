
#include "threepp/loaders/ImageLoader.hpp"

#include "stb.hpp"

using namespace threepp;

Image ImageLoader::load(const std::filesystem::path &imagePath, int channels) {

    auto image = stb_load(imagePath.string(), channels);

    return Image(image.width, image.height, std::shared_ptr<unsigned char>(image.pixels));
}
