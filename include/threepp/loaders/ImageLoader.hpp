
#ifndef THREEPP_IMAGELOADER_HPP
#define THREEPP_IMAGELOADER_HPP

#include <threepp/textures/Image.hpp>

namespace threepp {

    class ImageLoader {

    public:

        Image load(const std::string &imagePath, int channels = 4);
    };

}// namespace threepp

#endif//THREEPP_IMAGELOADER_HPP
