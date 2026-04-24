// Linearly Transformed Cosines (LTC) lookup tables.
// Originally from Eric Heitz et al.; this file embeds the exact data shipped
// by three.js (examples/jsm/lights/RectAreaLightUniformsLib.js) so that
// rectangular area lights produce identical results across renderers.

#ifndef THREEPP_LIGHTS_LTC_DATA_HPP
#define THREEPP_LIGHTS_LTC_DATA_HPP

#include <array>

namespace threepp::ltc {

    constexpr int LUT_SIZE = 64;
    constexpr int LUT_ELEMENTS = LUT_SIZE * LUT_SIZE * 4;

    extern const std::array<float, LUT_ELEMENTS> LTC_MAT_1;
    extern const std::array<float, LUT_ELEMENTS> LTC_MAT_2;

}// namespace threepp::ltc

#endif//THREEPP_LIGHTS_LTC_DATA_HPP
