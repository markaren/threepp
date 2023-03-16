
#ifndef THREEPP_IMAGELOADER_HPP
#define THREEPP_IMAGELOADER_HPP

#include <filesystem>
#include <optional>

#include <threepp/textures/Image.hpp>

namespace threepp {

    class ImageLoader {

    public:
        std::optional<Image> load(const std::filesystem::path& imagePath, int channels = 4, bool flipY = true);
        std::optional<Image> load(const std::vector<unsigned char>& data, int channels = 4, bool flipY = true);
    };

}// namespace threepp

#endif//THREEPP_IMAGELOADER_HPP
