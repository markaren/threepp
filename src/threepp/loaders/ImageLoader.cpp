
#include "threepp/loaders/ImageLoader.hpp"

#include <CImg.h>

using namespace threepp;
using namespace cimg_library;

Image ImageLoader::load(const char *imagePath) {

    CImg<unsigned char> image(imagePath);
    image._is_shared = true;

    return Image(image.width(), image.height(), image.data());
}
