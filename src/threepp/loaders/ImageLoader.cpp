
#include "threepp/loaders/ImageLoader.hpp"

#include <CImg.h>

using namespace threepp;
using namespace cimg_library;

Image ImageLoader::load(const char *imagePath) {

    CImg<unsigned char> image(imagePath);

    const size_t size = image.size() * sizeof (unsigned char);

    std::vector<unsigned char> data(size);
    for (int i = 0; i < size; i++) {
        data.emplace_back(image.data()[i]);
    }

    return Image(image.width(), image.height(), data);
}
