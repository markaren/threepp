// Scene-illuminant white balance as a linear-RGB matrix: the Bradford
// chromatic adaptation from a Planckian illuminant (Kelvin, tint) to D65,
// the math of the Vulkan renderer's post composite (PostComposite.cpp), so
// the GL OutputPass and the Vulkan composite agree on what a temperature does.

#ifndef THREEPP_WHITEBALANCE_HPP
#define THREEPP_WHITEBALANCE_HPP

#include "threepp/math/Matrix3.hpp"

namespace threepp {

    // CIE 1931 xy of a Planckian illuminant at `kelvin` (1667..25000 K, Kim
    // et al.'s cubic fit); `tint` offsets y, negative toward green, positive
    // toward magenta, one unit about 0.05 in y.
    void cctToXy(float kelvin, float tint, float& x, float& y);

    // The matrix that takes linear sRGB under the illuminant at (kelvin,
    // tint) to linear sRGB under D65: sRGB<-XYZ . Bradford CAT . XYZ<-sRGB.
    // Apply as matrix * colour.
    Matrix3 whiteBalanceToD65(float kelvin, float tint = 0.f);

}// namespace threepp

#endif//THREEPP_WHITEBALANCE_HPP
