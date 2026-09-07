// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/math/Lut.js

#ifndef THREEPP_LUT_HPP
#define THREEPP_LUT_HPP

#include "threepp/math/Color.hpp"

#include <unordered_map>
#include <vector>
#include <utility>

namespace threepp {

    class Lut {

    public:
        explicit Lut(const std::string& colormap, int numberofcolors = 32);

        Lut& setMin(float min);

        Lut& setMax(float max);

        Lut& setColorMap(const std::string& colormap, int numberofcolors = 32);

        void copy(const Lut& lut);

        Color getColor(float alpha) const;

        static void addColorMap(const std::string& name, const std::vector<std::pair<float, Color>>& arrayOfColors);

    private:
        std::vector<Color> lut;
        std::vector<std::pair<float, Color>> map;
        int n = 256;
        float minV = 0;
        float maxV = 1;

        inline static std::unordered_map<std::string, std::vector<std::pair<float, Color>>> ColorMapKeywords{
                {"rainbow", {{0.f, 0x0000ff}, {0.2f, 0x00ffff}, {0.5f, 0x00ff00}, {0.8f, 0xffff00}, {1.f, 0xff0000}}},
                {"cooltowarm", {{0.f, 0x3C4EC2}, {0.2f, 0x9BBCFF}, {0.5f, 0xDCDCDC}, {0.8f, 0xF6A385}, {1.f, 0xB40426}}},
                {"blackbody", {{0.f, 0x000000}, {0.2f, 0x9BBCFF}, {0.5f, 0x780000}, {0.8f, 0xFFFF00}, {1.f, 0xFFFFFF}}},
                {"grayscale", {{0.f, 0x000000}, {0.2f, 0x404040}, {0.5f, 0x7F7F80}, {0.8f, 0xBFBFBF}, {1.f, 0xFFFFFF}}},
        };
    };

}// namespace threepp

#endif//THREEPP_LUT_HPP
