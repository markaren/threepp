
#ifndef THREEPP_IMAGELOADER_HPP
#define THREEPP_IMAGELOADER_HPP

#include <string>
#include <filesystem>

#include <threepp/textures/Image.hpp>

namespace threepp {

    class ImageLoader {

    public:

        Image load(const std::filesystem::path &imagePath, int channels = 4, bool flipY = true);
    };

}// namespace threepp

#endif//THREEPP_IMAGELOADER_HPP
