
#include "threepp/math/Color.hpp"

#include "threepp/math/MathUtils.hpp"
#include "threepp/utils/StringUtils.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <unordered_map>

using namespace threepp;

namespace {

    // sRGB <-> linear transfer functions. Must match the GLSL sRGBToLinear /
    // LinearTosRGB in ShaderChunk/encodings_pars_fragment.glsl so a color
    // linearized on the CPU round-trips exactly through the renderer's output encode.
    float SRGBToLinear(float c) {
        return (c < 0.04045f) ? c * 0.0773993808f : std::pow(c * 0.9478672986f + 0.0521327014f, 2.4f);
    }

    float LinearToSRGB(float c) {
        return (c < 0.0031308f) ? c * 12.92f : 1.055f * std::pow(c, 0.41666f) - 0.055f;
    }

    float hue2rgb(float p, float q, float t) {

        if (t < 0) t += 1;
        if (t > 1) t -= 1;
        if (t < 1.f / 6) return p + (q - p) * 6 * t;
        if (t < 1.f / 2) return q;
        if (t < 2.f / 3) return p + (q - p) * 6 * (2.f / 3 - t);
        return p;
    }

    //clang-format off
    std::unordered_map<std::string, int> colorKeywords{
            {"aliceBlue", Color::aliceblue},
            {"antiquewhite", Color::antiquewhite},
            {"aqua", Color::aqua},
            {"aquamarine", Color::aquamarine},
            {"azure", Color::azure},

            {"beige", Color::beige},
            {"bisque", Color::bisque},
            {"black", Color::black},
            {"blanchedalmond", Color::blanchedalmond},
            {"blue", Color::blue},
            {"blueviolet", Color::blueviolet},

            {"brown", Color::brown},
            {"burlywood", Color::burlywood},
            {"cadetblue", Color::cadetblue},
            {"chartreuse", Color::chartreuse},
            {"chocolate", Color::chocolate},
            {"coral", Color::coral},

            {"cornflowerblue", Color::cornflowerblue},
            {"cornsilk", Color::cornsilk},
            {"crimson", Color::crimson},
            {"cyan", Color::cyan},
            {"darkblue", Color::darkblue},
            {"darkcyan", Color::darkcyan},

            {"darkgoldenrod", Color::darkgoldenrod},
            {"darkgray", Color::darkgray},
            {"darkgreen", Color::darkgreen},
            {"darkgrey", Color::darkgrey},
            {"darkkhaki", Color::darkkhaki},
            {"darkmagenta", Color::darkmagenta},

            {"darkolivegreen", Color::darkolivegreen},
            {"darkorange", Color::darkorange},
            {"darkorchid", Color::darkorchid},
            {"darkred", Color::darkred},
            {"darksalmon", Color::darksalmon},
            {"darkseagreen", Color::darkseagreen},

            {"darkslateblue", Color::darkslateblue},
            {"darkslategray", Color::darkslategray},
            {"darkslategrey", Color::darkslategrey},
            {"darkturquoise", Color::darkturquoise},
            {"darkviolet", Color::darkviolet},

            {"deeppink", Color::deeppink},
            {"deepskyblue", Color::deepskyblue},
            {"dimgray", Color::dimgray},
            {"dimgrey", Color::dimgrey},
            {"dodgerblue", Color::dodgerblue},
            {"firebrick", Color::firebrick},

            {"floralwhite", Color::floralwhite},
            {"forestgreen", Color::forestgreen},
            {"fuchsia", Color::fuchsia},
            {"gainsboro", Color::gainsboro},
            {"ghostwhite", Color::ghostwhite},
            {"gold", Color::gold},

            {"goldenrod", Color::goldenrod},
            {"gray", Color::gray},
            {"green", Color::green},
            {"greenyellow", Color::greenyellow},
            {"grey", Color::grey},
            {"honeydew", Color::honeydew},
            {"hotpink", Color::hotpink},

            {"indianred", Color::indianred},
            {"indigo", Color::indigo},
            {"ivory", Color::ivory},
            {"khaki", Color::khaki},
            {"lavender", Color::lavender},
            {"lavenderblush", Color::lavenderblush},
            {"lawngreen", Color::lawngreen},

            {"lemonchiffon", Color::lemonchiffon},
            {"lightblue", Color::lightblue},
            {"ivory", Color::ivory},
            {"lightcoral", Color::lightcoral},
            {"lightcyan", Color::lightcyan},
            {"lightgoldenrodyellow", Color::lightgoldenrodyellow},
            {"lightgray", Color::lightgray},

            {"lightgreen", Color::lightgreen},
            {"lightgrey", Color::lightgrey},
            {"lightpink", Color::lightpink},
            {"lightsalmon", Color::lightsalmon},
            {"lightseagreen", Color::lightseagreen},
            {"lightskyblue", Color::lightskyblue},

            {"lightslategray", Color::lightslategray},
            {"lightslategrey", Color::lightslategrey},
            {"lightsteelblue", Color::lightsteelblue},
            {"lightyellow", Color::lightyellow},
            {"lime", Color::lime},
            {"limegreen", Color::limegreen},

            {"linen", Color::linen},
            {"magenta", Color::magenta},
            {"maroon", Color::maroon},
            {"mediumaquamarine", Color::mediumaquamarine},
            {"mediumblue", Color::mediumblue},
            {"mediumorchid", Color::mediumorchid},

            {"mediumpurple", Color::mediumpurple},
            {"mediumseagreen", Color::mediumseagreen},
            {"mediumslateblue", Color::mediumslateblue},
            {"mediumspringgreen", Color::mediumspringgreen},
            {"mediumturquoise", Color::mediumturquoise},

            {"mediumvioletred", Color::mediumvioletred},
            {"midnightblue", Color::midnightblue},
            {"mintcream", Color::mintcream},
            {"mistyrose", Color::mistyrose},
            {"moccasin", Color::moccasin},
            {"navajowhite", Color::navajowhite},

            {"navy", Color::navy},
            {"oldlace", Color::oldlace},
            {"olive", Color::olive},
            {"olivedrab", Color::olivedrab},
            {"orange", Color::orange},
            {"orangered", Color::orangered},
            {"orchid", Color::orchid},

            {"palegoldenrod", Color::palegoldenrod},
            {"palegreen", Color::palegreen},
            {"paleturquoise", Color::paleturquoise},
            {"palevioletred", Color::palevioletred},
            {"papayawhip", Color::papayawhip},
            {"peachpuff", Color::peachpuff},

            {"peru", Color::peru},
            {"pink", Color::pink},
            {"plum", Color::plum},
            {"powderblue", Color::powderblue},
            {"purple", Color::purple},
            {"rebeccapurple", Color::rebeccapurple},
            {"rosybrown", Color::rosybrown},

            {"royalblue", Color::royalblue},
            {"saddlebrown", Color::saddlebrown},
            {"salmon", Color::salmon},
            {"sandybrown", Color::sandybrown},
            {"seagreen", Color::seagreen},
            {"seashell", Color::seashell},

            {"sienna", Color::sienna},
            {"silver", Color::silver},
            {"skyblue", Color::skyblue},
            {"slateblue", Color::slateblue},
            {"slategray", Color::slategray},
            {"slategrey", Color::slategrey},
            {"snow", Color::snow},

            {"springgreen", Color::springgreen},
            {"steelblue", Color::steelblue},
            {"tan", Color::tan},
            {"teal", Color::teal},
            {"thistle", Color::thistle},
            {"tomato", Color::tomato},
            {"turquoise", Color::turquoise},

            {"violet", Color::violet},
            {"wheat", Color::wheat},
            {"white", Color::white},
            {"whitesmoke", Color::whitesmoke},
            {"yellow", Color::yellow},
            {"yellowgreen", Color::yellowgreen}

    };
    //clang-format on

}// namespace

bool ColorManagement::enabled = true;
ColorSpace ColorManagement::workingColorSpace = ColorSpace::Linear;

Color& ColorManagement::convert(Color& color, ColorSpace source, ColorSpace target) {

    if (!enabled || source == target ||
        source == ColorSpace::NoColorSpace || target == ColorSpace::NoColorSpace) {

        return color;
    }

    // threepp only deals with the sRGB and linear-sRGB working spaces, which share
    // the same (Rec. 709) primaries, so only the transfer function differs.
    if (source == ColorSpace::sRGB) {

        color.r = SRGBToLinear(color.r);
        color.g = SRGBToLinear(color.g);
        color.b = SRGBToLinear(color.b);
    }

    if (target == ColorSpace::sRGB) {

        color.r = LinearToSRGB(color.r);
        color.g = LinearToSRGB(color.g);
        color.b = LinearToSRGB(color.b);
    }

    return color;
}

Color& ColorManagement::colorSpaceToWorking(Color& color, ColorSpace source) {

    return convert(color, source, workingColorSpace);
}

Color& ColorManagement::workingToColorSpace(Color& color, ColorSpace target) {

    return convert(color, workingColorSpace, target);
}

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

Color& Color::setHex(unsigned int hex, ColorSpace colorSpace) {

    this->r = static_cast<float>(hex >> 16 & 255) / 255.f;
    this->g = static_cast<float>(hex >> 8 & 255) / 255.f;
    this->b = static_cast<float>(hex & 255) / 255.f;

    ColorManagement::colorSpaceToWorking(*this, colorSpace);

    return *this;
}

Color& Color::setRGB(float r, float g, float b, ColorSpace colorSpace) {

    this->r = r;
    this->g = g;
    this->b = b;

    ColorManagement::colorSpaceToWorking(*this, colorSpace);

    return *this;
}

Color& Color::setHSL(float h, float s, float l, ColorSpace colorSpace) {

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

    ColorManagement::colorSpaceToWorking(*this, colorSpace);

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

    this->r = math::randFloat();
    this->g = math::randFloat();
    this->b = math::randFloat();

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

unsigned int Color::getHex(ColorSpace colorSpace) const {

    Color c;
    c.copy(*this);
    ColorManagement::workingToColorSpace(c, colorSpace);

    const auto r = static_cast<unsigned int>(std::round(std::clamp(c.r * 255.f, 0.f, 255.f)));
    const auto g = static_cast<unsigned int>(std::round(std::clamp(c.g * 255.f, 0.f, 255.f)));
    const auto b = static_cast<unsigned int>(std::round(std::clamp(c.b * 255.f, 0.f, 255.f)));

    return (r << 16) | (g << 8) | b;
}

HSL& Color::getHSL(HSL& target, ColorSpace colorSpace) const {
    // h,s,l ranges are in 0.0 - 1.0

    Color c;
    c.copy(*this);
    ColorManagement::workingToColorSpace(c, colorSpace);

    const float r = c.r, g = c.g, b = c.b;

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

std::string Color::getHexString(ColorSpace colorSpace) const {

    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << std::hex << getHex(colorSpace);

    return ss.str();
}

std::string Color::getStyle(ColorSpace colorSpace) const {

    Color c;
    c.copy(*this);
    ColorManagement::workingToColorSpace(c, colorSpace);

    std::stringstream ss;
    ss << "rgb(" << static_cast<int>(std::round(c.r * 255)) << ","
       << static_cast<int>(std::round(c.g * 255)) << ","
       << static_cast<int>(std::round(c.b * 255)) << ")";

    return ss.str();
}

Color& Color::setStyle(const std::string& style, ColorSpace colorSpace) {

    // Each terminal setter (setRGB/setHSL/setHex/setColorName) applies the
    // colorSpace -> working conversion, so the parsing here just routes to them.

    static std::regex r1(R"(((?:rgb|hsl)a?)\(([^\)]*)\))", std::regex::icase);
    static std::regex r2("\\#([A-Fa-f\\d]+)$", std::regex::icase);

    std::smatch match;
    if (std::regex_match(style, match, r1)) {

        // rgb / hsl

        std::string name = match[1];
        std::string components = match[2];

        if (name == "rgb" || name == "rgba") {

            static std::regex ra(R"(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d*\.?\d+)\s*)?$)", std::regex::icase);
            static std::regex rb(R"(\s*(\d+)\%\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$)", std::regex::icase);

            if (std::regex_match(components, match, ra)) {

                // rgb(255,0,0) rgba(255,0,0,0.5)
                return this->setRGB(
                        std::min(255.f, static_cast<float>(utils::parseInt(match[1].str()))) / 255,
                        std::min(255.f, static_cast<float>(utils::parseInt(match[2].str()))) / 255,
                        std::min(255.f, static_cast<float>(utils::parseInt(match[3].str()))) / 255,
                        colorSpace);

            } else if (std::regex_match(components, match, rb)) {

                // rgb(100%,0%,0%) rgba(100%,0%,0%,0.5)
                return this->setRGB(
                        std::min(100.f, static_cast<float>(utils::parseInt(match[1].str()))) / 100,
                        std::min(100.f, static_cast<float>(utils::parseInt(match[2].str()))) / 100,
                        std::min(100.f, static_cast<float>(utils::parseInt(match[3].str()))) / 100,
                        colorSpace);
            }

        } else if (name == "hsl" || name == "hsla") {

            static std::regex r(R"(\s*(\d*\.?\d+)\s*,\s*(\d+)\%\s*,\s*(\d+)\%\s*(?:,\s*(\d*\.?\d+)\s*)?$)", std::regex::icase);

            if (std::regex_match(components, match, r)) {

                // hsl(120,50%,50%) hsla(120,50%,50%,0.5)
                const auto h = utils::parseFloat(match[1].str()) / 360;
                const auto s = static_cast<float>(utils::parseInt(match[2].str())) / 100;
                const auto l = static_cast<float>(utils::parseInt(match[3].str())) / 100;

                return this->setHSL(h, s, l, colorSpace);
            }
        }

    } else if (std::regex_match(style, match, r2)) {

        // hex color

        std::string hex = match[1];
        auto size = hex.length();

        if (size == 3) {

            // #ff0
            return this->setRGB(
                    static_cast<float>(std::stoi(std::string{hex[0]} + std::string{hex[0]}, nullptr, 16)) / 255,
                    static_cast<float>(std::stoi(std::string{hex[1]} + std::string{hex[1]}, nullptr, 16)) / 255,
                    static_cast<float>(std::stoi(std::string{hex[2]} + std::string{hex[2]}, nullptr, 16)) / 255,
                    colorSpace);

        } else if (size == 6) {

            // #ff0000
            return this->setHex(static_cast<unsigned int>(std::stoul(hex, nullptr, 16)), colorSpace);
        }
    }

    if (style.length() > 0) {

        return setColorName(style, colorSpace);
    }

    return *this;
}

Color& Color::copySRGBToLinear(const Color& color) {

    this->r = SRGBToLinear(color.r);
    this->g = SRGBToLinear(color.g);
    this->b = SRGBToLinear(color.b);

    return *this;
}

Color& Color::copyLinearToSRGB(const Color& color) {

    this->r = LinearToSRGB(color.r);
    this->g = LinearToSRGB(color.g);
    this->b = LinearToSRGB(color.b);

    return *this;
}

Color& Color::convertSRGBToLinear() {

    return copySRGBToLinear(*this);
}

Color& Color::convertLinearToSRGB() {

    return copyLinearToSRGB(*this);
}

Color& Color::setColorName(const std::string& style, ColorSpace colorSpace) {

    if (!colorKeywords.contains(style)) {
        std::cerr << "THREE.Color: Unknown color '" + style + "'" << std::endl;
        return *this;
    }

    return setHex(colorKeywords.at(style), colorSpace);
}

Color& Color::lerpHSL(const Color& color, float alpha) {

    static HSL _hslA;
    static HSL _hslB;

    this->getHSL(_hslA);
    color.getHSL(_hslB);

    const auto h = math::lerp(_hslA.h, _hslB.h, alpha);
    const auto s = math::lerp(_hslA.s, _hslB.s, alpha);
    const auto l = math::lerp(_hslA.l, _hslB.l, alpha);

    this->setHSL(h, s, l);

    return *this;
}
