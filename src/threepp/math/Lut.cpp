
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

    // Validate input: need at least 1 color. Clamp to 1 to prevent division by zero.
    // Using silent correction to maintain backward compatibility with existing code.
    if (this->n <= 0) {
        this->n = 1;
    }

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

                // Avoid division by zero if min == max
                float t = (max - min > 0.0f) ? (alpha - min) / (max - min) : 0.0f;
                Color color = minColor.lerp(maxColor, t);

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
    // Avoid division by zero if minV == maxV
    if (this->maxV - this->minV > 0.0f) {
        alpha = (alpha - this->minV) / (this->maxV - this->minV);
    } else {
        alpha = 0.0f;
    }
    const int colorPosition = static_cast<int>(std::round(alpha * static_cast<float>(this->n)));
    
    // Clamp colorPosition to valid range [0, n-1] to prevent out-of-bounds access
    const int clampedPosition = std::min(colorPosition, static_cast<int>(this->n - 1));

    return this->lut[clampedPosition];
}

void Lut::addColorMap(const std::string& name, const std::vector<std::pair<float, Color>>& arrayOfColors) {

    ColorMapKeywords[name] = arrayOfColors;
}