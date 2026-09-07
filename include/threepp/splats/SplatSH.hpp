// Real spherical-harmonic basis, bands 0-3 — the radiance basis used by 3D
// Gaussian Splatting (arXiv 2308.04079), and the ONE place its constants live.
//
// Provenance: clean-room. The constants below are the orthonormalised real
// solid harmonics, derived from their closed forms (see the comment on each
// line); no third-party source was consulted or copied. The GLSL half of this
// basis lives in SplatCloud.cpp and is written with the *same decimal literals*
// as the C++ table here — SplatSH_test asserts the shader text still contains
// every one of them, so the two cannot drift apart silently.
//
// Sign convention. Trained 3DGS coefficients assume the Condon-Shortley-style
// basis whose odd-|m| terms carry a minus sign, i.e. Y_l^m is scaled by
// (-1)^|m|. threepp's own SphericalHarmonis3 (math/SphericalHarmonics3.cpp)
// uses the same constants with ALL signs positive — the magnitudes agree to the
// six decimals it writes them with, the signs do not. That is deliberate on
// both sides: SphericalHarmonis3 is an irradiance probe evaluated against
// coefficients threepp itself produced, while these are evaluated against
// coefficients someone else's optimiser produced. Flipping either one to match
// the other would corrupt its own data. SplatSH_test pins the relationship.
//
// Ordering is the usual l-major, m-ascending one, matching how the INRIA PLY
// stores coefficients:
//   0            : l=0
//   1,2,3        : l=1, m=-1,0,+1
//   4..8         : l=2, m=-2..+2
//   9..15        : l=3, m=-3..+3

#ifndef THREEPP_SPLATSH_HPP
#define THREEPP_SPLATSH_HPP

#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <array>

namespace threepp::splats {

    // Highest SH degree the loader, the data model and the shader support.
    constexpr int MAX_SH_DEGREE = 3;

    // Coefficients per colour channel for a degree: 1, 4, 9, 16.
    [[nodiscard]] constexpr int shCoeffCount(int degree) {

        return (degree + 1) * (degree + 1);
    }

    // 1 / (2 * sqrt(pi))
    inline constexpr float SH_C0 = 0.28209479177387814f;

    // sqrt(3 / (4*pi))
    inline constexpr float SH_C1 = 0.4886025119029199f;

    // m = -2..+2; signs are the (-1)^|m| convention described above.
    //   |m|=2: 1/2 * sqrt(15/pi) and 1/4 * sqrt(15/pi)
    //   |m|=1: 1/2 * sqrt(15/pi)
    //   m = 0: 1/4 * sqrt(5/pi)
    inline constexpr std::array<float, 5> SH_C2{
            1.0925484305920792f,
            -1.0925484305920792f,
            0.31539156525252005f,
            -1.0925484305920792f,
            0.5462742152960396f};

    // m = -3..+3.
    //   |m|=3: 1/4 * sqrt(35/(2*pi))
    //   |m|=2: 1/2 * sqrt(105/pi)  and  1/4 * sqrt(105/pi)
    //   |m|=1: 1/4 * sqrt(21/(2*pi))
    //   m = 0: 1/4 * sqrt(7/pi)
    inline constexpr std::array<float, 7> SH_C3{
            -0.5900435899266435f,
            2.890611442640554f,
            -0.4570457994644658f,
            0.3731763325901154f,
            -0.4570457994644658f,
            1.445305721320277f,
            -0.5900435899266435f};

    // Evaluates the basis functions for a *unit* direction into `out`, which
    // must hold shCoeffCount(MAX_SH_DEGREE) == 16 floats. Entries above the
    // requested degree are zeroed, so a caller may always read all 16.
    //
    // The l=2 m=0 and l=3 terms are written in the (2z^2-x^2-y^2) form rather
    // than (3z^2-1): identical on the unit sphere, and the form the shader uses.
    inline void shBasis(const Vector3& dir, int degree, float* out) {

        const float x = dir.x, y = dir.y, z = dir.z;

        for (int i = 0; i < shCoeffCount(MAX_SH_DEGREE); ++i) out[i] = 0.f;

        out[0] = SH_C0;
        if (degree < 1) return;

        out[1] = -SH_C1 * y;
        out[2] = SH_C1 * z;
        out[3] = -SH_C1 * x;
        if (degree < 2) return;

        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, yz = y * z, xz = x * z;

        out[4] = SH_C2[0] * xy;
        out[5] = SH_C2[1] * yz;
        out[6] = SH_C2[2] * (2.f * zz - xx - yy);
        out[7] = SH_C2[3] * xz;
        out[8] = SH_C2[4] * (xx - yy);
        if (degree < 3) return;

        out[9] = SH_C3[0] * y * (3.f * xx - yy);
        out[10] = SH_C3[1] * xy * z;
        out[11] = SH_C3[2] * y * (4.f * zz - xx - yy);
        out[12] = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
        out[13] = SH_C3[4] * x * (4.f * zz - xx - yy);
        out[14] = SH_C3[5] * z * (xx - yy);
        out[15] = SH_C3[6] * x * (xx - 3.f * yy);
    }

    // The +0.5 that turns a 3DGS radiance sum into a display-referred colour.
    // The optimiser initialises the DC coefficient as (rgb - 0.5) / SH_C0, so
    // the constant belongs to the *evaluation*, not to the stored data — which
    // is why the PLY loader never bakes it in.
    inline constexpr float SH_COLOR_OFFSET = 0.5f;

    // colour = sum_i coeff_i * basis_i(dir) + 0.5, clamped at 0.
    // `coeffs` is coefficient-major: coeffs[c * 3 + channel].
    //
    // The GLSL twin of this clamps in the fragment stage rather than here, so
    // that it can test for non-finite values first — identical for finite
    // input, which is all this CPU-side query is ever asked about.
    [[nodiscard]] inline Vector3 evalSh(const float* coeffs, int degree, const Vector3& dir) {

        float basis[16];
        shBasis(dir, degree, basis);

        Vector3 result{SH_COLOR_OFFSET, SH_COLOR_OFFSET, SH_COLOR_OFFSET};
        for (int c = 0, n = shCoeffCount(degree); c < n; ++c) {

            result.x += coeffs[c * 3 + 0] * basis[c];
            result.y += coeffs[c * 3 + 1] * basis[c];
            result.z += coeffs[c * 3 + 2] * basis[c];
        }

        result.x = std::max(0.f, result.x);
        result.y = std::max(0.f, result.y);
        result.z = std::max(0.f, result.z);

        return result;
    }

    // Inverse of the DC half of evalSh: the coefficient that renders as `rgb`
    // when every higher band is zero. Used by the generator and by anyone
    // hand-authoring a splat.
    [[nodiscard]] inline Vector3 dcFromColor(const Vector3& rgb) {

        return {(rgb.x - SH_COLOR_OFFSET) / SH_C0,
                (rgb.y - SH_COLOR_OFFSET) / SH_C0,
                (rgb.z - SH_COLOR_OFFSET) / SH_C0};
    }

}// namespace threepp::splats

#endif//THREEPP_SPLATSH_HPP
