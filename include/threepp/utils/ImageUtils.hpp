
#ifndef THREEPP_IMAGEUTILS_HPP
#define THREEPP_IMAGEUTILS_HPP

#include <vector>

namespace threepp {

    void convertBGRtoRGB(std::vector<unsigned char>& pixels);

    void flipImage(std::vector<unsigned char>& pixels, int channels, int w, int h);

}// namespace threepp

#endif //THREEPP_IMAGEUTILS_HPP
