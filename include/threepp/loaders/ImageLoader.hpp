
#ifndef THREEPP_IMAGELOADER_HPP
#define THREEPP_IMAGELOADER_HPP

#include <filesystem>

#include <threepp/textures/Image.hpp>

namespace threepp {

    class ImageLoader {

        Image load(const char *imagePath);
    };

}// namespace threepp

#endif//THREEPP_IMAGELOADER_HPP
