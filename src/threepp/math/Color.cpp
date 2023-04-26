
#include "threepp/math/Color.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/utils/RegexUtil.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <regex>

using namespace threepp;

namespace {

    float hue2rgb(float p, float q, float t) {

        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1.f / 6) return p + (q - p) * 6 * t;
        if (t < 1.f / 2) return q;
        if (t < 2.f / 3) return p + (q - p) * 6 * (2.f / 3 - t);
        return p;
    }

}// namespace

Color::Color(float r, float g, float b)
    : r(r), g(g), b(b) {}

Color::Color(unsigned int hex) {
    setHex(hex);
}

float& Color::operator[](unsigned int index) {
    switch (index) {
        case 0:
            return r;
        case 1:
            return g;
        case 2:
            return b;
        default:
            throw std::runtime_error("index out of bound: " + std::to_string(index));
    }
}

Color& Color::setScalar(float scalar) {

    this->r = scalar;
    this->g = scalar;
    this->b = scalar;

    return *this;
}

Color& Color::setHex(unsigned int hex) {

    this->r = static_cast<float>(hex >> 16 & 255) / 255.f;
    this->g = static_cast<float>(hex >> 8 & 255) / 255.f;
    this->b = static_cast<float>(hex & 255) / 255.f;

    return *this;
}

Color& Color::setRGB(float r, float g, float b) {

    this->r = r;
    this->g = g;
    this->b = b;

    return *this;
}

Color& Color::setHSL(float h, float s, float l) {

    // h,s,l ranges are in 0.0 - 1.0
    h = math::euclideanModulo(h, 1.f);
    s = std::clamp(s, 0.f, 1.f);
    l = std::clamp(l, 0.f, 1.f);

    if (s == 0) {

        this->r = this->g = this->b = l;

    } else {

        const auto p = l <= 0.5f ? l * (1 + s) : l + s - (l * s);
        const auto q = (2 * l) - p;

        this->r = hue2rgb(q, p, h + 1.f / 3);
        this->g = hue2rgb(q, p, h);
        this->b = hue2rgb(q, p, h - 1.f / 3);
    }

    return *this;
}

Color& Color::copy(const Color& color) {

    this->r = color.r;
    this->g = color.g;
    this->b = color.b;

    return *this;
}

Color& Color::add(const Color& color) {

    this->r += color.r;
    this->g += color.g;
    this->b += color.b;

    return *this;
}

Color& Color::addColors(const Color& color1, const Color& color2) {

    this->r = color1.r + color2.r;
    this->g = color1.g + color2.g;
    this->b = color1.b + color2.b;

    return *this;
}

Color& Color::addScalar(float s) {

    this->r += s;
    this->g += s;
    this->b += s;

    return *this;
}

Color& Color::sub(const Color& color) {

    this->r = std::max(0.f, this->r - color.r);
    this->g = std::max(0.f, this->g - color.g);
    this->b = std::max(0.f, this->b - color.b);

    return *this;
}

Color& Color::multiply(const Color& color) {

    this->r *= color.r;
    this->g *= color.g;
    this->b *= color.b;

    return *this;
}

Color& Color::operator*=(const Color& color) {

    return multiply(color);
}

Color& Color::multiplyScalar(float s) {

    this->r *= s;
    this->g *= s;
    this->b *= s;

    return *this;
}

Color& Color::operator*=(float s) {

    return multiplyScalar(s);
}

Color& Color::lerp(const Color& color, float alpha) {

    this->r += (color.r - this->r) * alpha;
    this->g += (color.g - this->g) * alpha;
    this->b += (color.b - this->b) * alpha;

    return *this;
}

Color& Color::lerpColors(const Color& color1, const Color& color2, float alpha) {

    this->r = color1.r + (color2.r - color1.r) * alpha;
    this->g = color1.g + (color2.g - color1.g) * alpha;
    this->b = color1.b + (color2.b - color1.b) * alpha;

    return *this;
}

Color& Color::randomize() {

    this->r = math::random();
    this->g = math::random();
    this->b = math::random();

    return *this;
}

bool Color::equals(const Color& c) const {

    return (c.r == this->r) && (c.g == this->g) && (c.b == this->b);
}

bool Color::operator==(const Color& c) const {

    return equals(c);
}

bool Color::operator!=(const Color& c) const {

    return !equals(c);
}

unsigned int Color::getHex() const {

    return static_cast<int>(this->r * 255) << 16 ^ static_cast<int>(this->g * 255) << 8 ^ static_cast<int>(this->b * 255) << 0;
}

HSL& Color::getHSL(HSL& target) const {
    // h,s,l ranges are in 0.0 - 1.0

    const float r = this->r, g = this->g, b = this->b;

    const auto max = std::max(r, std::max(g, b));
    const auto min = std::min(r, std::min(g, b));

    float hue = 0, saturation;
    const auto lightness = (min + max) / 2.0f;

    if (min == max) {

        hue = 0;
        saturation = 0;

    } else {

        const auto delta = max - min;

        saturation = lightness <= 0.5f ? delta / (max + min) : delta / (2 - max - min);

        if (max == r) {
            hue = (g - b) / delta + (g < b ? 6.f : 0.f);
        } else if (max == g) {
            hue = (b - r) / delta + 2;
        } else if (max == b) {
            hue = (r - g) / delta + 4;
        }

        hue /= 6.f;
    }

    target.h = hue;
    target.s = saturation;
    target.l = lightness;

    return target;
}

std::string Color::getHexString() const {

    std::stringstream ss;
    ss << std::hex << getHex();

    return ss.str();
}

std::string Color::getStyle() const {

    std::stringstream ss;
    ss << "rgb(" << (static_cast<int>(this->r * 255) | 0) << "," << (static_cast<int>(this->g * 255) | 0) << "," << (static_cast<int>(this->b * 255) | 0) << ")";

    return ss.str();
}

Color& Color::setStyle(const std::string& style) {

    static std::regex r1(R"(/^((?:rgb|hsl)a?)\(([^\)]*)\)/)", std::regex::icase);
    static std::regex r2("/^\\s*(\\d+)\\%\\s*,\\s*(\\d+)\\%\\s*,\\s*(\\d+)\\%\\s*(?:,\\s*(\\d*\\.?\\d+)\\s*)?$/", std::regex::icase);


    if (std::regex_match(style, r1)) {

    } else if ()

    return *this;
}
