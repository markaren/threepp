// https://github.com/mrdoob/three.js/blob/r129/examples/jsm/math/Lut.js

#ifndef THREEPP_LUT_HPP
#define THREEPP_LUT_HPP

#include "threepp/math/Color.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace threepp {

    class Lut {

    public:
        explicit Lut(const std::string& colormap, int numberofcolors = 32) {

            this->setColorMap(colormap, numberofcolors);
        }

        Lut& setMin(float min) {

            this->minV = min;

            return *this;
        }

        Lut& setMax(float max) {

            this->maxV = max;

            return *this;
        }

        Lut& setColorMap(const std::string& colormap, int numberofcolors = 32) {
            this->map = ColorMapKeywords[colormap];
            this->n = numberofcolors;

            float step = 1.f / static_cast<float>(this->n);


            this->lut.clear();

            // sample at 0
            this->lut.emplace_back(this->map[0].second);

            // sample at 1/n, ..., (n-1)/n
            for (int i = 1; i < numberofcolors; i++) {
                float alpha = static_cast<float>(i) * step;

                for (int j = 0; j < this->map.size() - 1; j++) {
                    if (alpha > this->map[j].first && alpha <= this->map[j + 1].first) {
                        float min = this->map[j].first;
                        float max = this->map[j + 1].first;

                        Color minColor(this->map[j].second);
                        Color maxColor(this->map[j + 1].second);

                        Color color = minColor.lerp(maxColor, (alpha - min) / (max - min));

                        this->lut.emplace_back(color);
                    }
                }
            }

            // sample at 1
            this->lut.emplace_back(this->map[this->map.size() - 1].second);

            return *this;
        }

        void copy(const Lut& lut) {
            this->lut = lut.lut;
            this->map = lut.map;
            this->n = lut.n;
            this->minV = lut.minV;
            this->maxV = lut.maxV;
        }

        Color getColor(float alpha) {
            alpha = std::clamp(alpha, this->minV, this->maxV);
            alpha = (alpha - this->minV) / (this->maxV - this->minV);
            int colorPosition = static_cast<int>(std::round(alpha * this->n));

            return this->lut[colorPosition];
        }

        static void addColorMap(const std::string& name, const std::vector<std::pair<float, Color>>& arrayOfColors) {

            ColorMapKeywords[name] = arrayOfColors;
        }

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
