
#include "threepp/math/Lut.hpp"

#include <algorithm>
#include <cmath>

using namespace threepp;

Lut::Lut(const std::string& colormap, int numberofcolors) {

    this->setColorMap(colormap, numberofcolors);
}

Lut& Lut::setMin(float min) {

    this->minV = min;

    return *this;
}

Lut& Lut::setMax(float max) {

    this->maxV = max;

    return *this;
}

Lut& Lut::setColorMap(const std::string& colormap, int numberofcolors) {
    this->map = ColorMapKeywords[colormap];
    this->n = numberofcolors;

    const float step = 1.f / static_cast<float>(this->n);


    this->lut.clear();

    // sample at 0
    this->lut.emplace_back(this->map[0].second);

    // sample at 1/n, ..., (n-1)/n
    for (int i = 1; i < numberofcolors; i++) {
        const float alpha = static_cast<float>(i) * step;

        for (unsigned j = 0; j < this->map.size() - 1; j++) {
            if (alpha > this->map[j].first && alpha <= this->map[j + 1].first) {
                const float min = this->map[j].first;
                const float max = this->map[j + 1].first;

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

void Lut::copy(const Lut& lut) {
    this->lut = lut.lut;
    this->map = lut.map;
    this->n = lut.n;
    this->minV = lut.minV;
    this->maxV = lut.maxV;
}

Color Lut::getColor(float alpha) const {
    alpha = std::clamp(alpha, this->minV, this->maxV);
    alpha = (alpha - this->minV) / (this->maxV - this->minV);
    const int colorPosition = static_cast<int>(std::round(alpha * static_cast<float>(this->n)));

    return this->lut[colorPosition];
}

void Lut::addColorMap(const std::string& name, const std::vector<std::pair<float, Color>>& arrayOfColors) {

    ColorMapKeywords[name] = arrayOfColors;
}