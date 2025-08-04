
#include "threepp/utils/ImageUtils.hpp"

#include <algorithm>

using namespace threepp;

void threepp::convertBGRtoRGB(std::vector<unsigned char>& pixels) {
    for (size_t i = 0; i < pixels.size(); i += 3) {
        std::swap(pixels[i], pixels[i + 2]);
    }
}

void threepp::flipImage(std::vector<unsigned char>& pixels, int channels, int w, int h) {
    for (int line = 0; line != h / 2; ++line) {
        std::swap_ranges(pixels.begin() + channels * w * line,
                         pixels.begin() + channels * w * (line + 1),
                         pixels.begin() + channels * w * (h - line - 1));
    }
}
