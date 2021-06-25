
#include "threepp/loaders/ImageLoader.hpp"

#include <CImg.h>

using namespace threepp;
using namespace cimg_library;

Image ImageLoader::load(const char *imagePath) {

    CImg<unsigned char> image(imagePath);

    return Image(0, 0, image.data());
}
