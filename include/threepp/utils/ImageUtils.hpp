
#ifndef THREEPP_IMAGEUTILS_HPP
#define THREEPP_IMAGEUTILS_HPP

#include <vector>

#include "threepp/textures/Image.hpp"

namespace threepp {

    void convertBGRtoRGB(std::vector<unsigned char>& pixels);

    void flipImage(std::vector<unsigned char>& pixels, int channels, int w, int h);
    void flipImage(Image& image);

}// namespace threepp

#endif //THREEPP_IMAGEUTILS_HPP
