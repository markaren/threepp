
#include "threepp/renderers/common/WhiteBalance.hpp"

#include <algorithm>
#include <array>

using namespace threepp;

namespace {

    // Row-major 3x3, m[row * 3 + col].
    using Mat3 = std::array<float, 9>;

    Mat3 mul(const Mat3& a, const Mat3& b) {

        Mat3 r{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                r[i * 3 + j] = a[i * 3 + 0] * b[0 * 3 + j] +
                               a[i * 3 + 1] * b[1 * 3 + j] +
                               a[i * 3 + 2] * b[2 * 3 + j];
        return r;
    }

}// namespace

void threepp::cctToXy(float kelvin, float tint, float& x, float& y) {

    const double T = std::clamp(kelvin, 1667.f, 25000.f);
    const double T2 = T * T, T3 = T2 * T;
    double xd;
    if (T <= 4000.0)
        xd = -0.2661239e9 / T3 - 0.2343589e6 / T2 + 0.8776956e3 / T + 0.179910;
    else
        xd = -3.0258469e9 / T3 + 2.1070379e6 / T2 + 0.2226347e3 / T + 0.240390;
    const double x2 = xd * xd, x3 = x2 * xd;
    double yd;
    if (T <= 2222.0)
        yd = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * xd - 0.20219683;
    else if (T <= 4000.0)
        yd = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * xd - 0.16748867;
    else
        yd = 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * xd - 0.37001483;
    x = static_cast<float>(xd);
    y = static_cast<float>(yd) - tint * 0.05f;
}

Matrix3 threepp::whiteBalanceToD65(float kelvin, float tint) {

    float srcX, srcY;
    cctToXy(kelvin, tint, srcX, srcY);

    const Mat3 bradford = {0.8951f, 0.2664f, -0.1614f,
                           -0.7502f, 1.7135f, 0.0367f,
                           0.0389f, -0.0685f, 1.0296f};
    const Mat3 bradfordInv = {0.9869929f, -0.1470543f, 0.1599627f,
                              0.4323053f, 0.5183603f, 0.0492912f,
                              -0.0085287f, 0.0400428f, 0.9684867f};
    const Mat3 srgbToXyz = {0.4124564f, 0.3575761f, 0.1804375f,
                            0.2126729f, 0.7151522f, 0.0721750f,
                            0.0193339f, 0.1191920f, 0.9503041f};
    const Mat3 xyzToSrgb = {3.2404542f, -1.5371385f, -0.4985314f,
                            -0.9692660f, 1.8760108f, 0.0415560f,
                            0.0556434f, -0.2040259f, 1.0572252f};

    auto whiteXyz = [](float x, float y, float* out) {
        out[0] = x / y;
        out[1] = 1.f;
        out[2] = (1.f - x - y) / y;
    };
    float src[3], dst[3];
    whiteXyz(srcX, srcY, src);
    whiteXyz(0.3127f, 0.3290f, dst);// D65

    auto cone = [&](const float* w, float* lms) {
        for (int i = 0; i < 3; ++i)
            lms[i] = bradford[i * 3 + 0] * w[0] +
                     bradford[i * 3 + 1] * w[1] +
                     bradford[i * 3 + 2] * w[2];
    };
    float lmsSrc[3], lmsDst[3];
    cone(src, lmsSrc);
    cone(dst, lmsDst);

    Mat3 scale{};
    for (int i = 0; i < 3; ++i)
        scale[i * 3 + i] = lmsDst[i] / lmsSrc[i];

    const Mat3 cat = mul(bradfordInv, mul(scale, bradford));
    const Mat3 m = mul(xyzToSrgb, mul(cat, srgbToXyz));

    // Matrix3::set takes the rows; it stores column-major for GL.
    Matrix3 result;
    result.set(m[0], m[1], m[2],
               m[3], m[4], m[5],
               m[6], m[7], m[8]);
    return result;
}
