#ifndef THREEPP_WGPUPATHTRACERENVCDF_HPP
#define THREEPP_WGPUPATHTRACERENVCDF_HPP

// Private header — environment-map importance-sampling CDF builder used by the
// path tracer's IBL setup. Header-only because it's a function template.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace threepp::wgpu_pt {

    /// Result of buildEnvCdf — 2D CDF for importance-sampling an equirectangular
    /// environment map. Both CDFs are normalized to [0,1]. The sin(theta)
    /// Jacobian is baked in so that pixels near the equator (more solid angle)
    /// are sampled proportionally more.
    struct EnvCdfResult {
        std::vector<float> conditional; // width × height
        std::vector<float> marginal;    // height
        float totalSum = 0.f;
        int   width = 0;
        int   height = 0;
    };

    /// Build a 2D CDF for an equirectangular environment map.
    /// `pixels` is RGBA (4 components per pixel); only RGB is used.
    /// T is float for HDR sources or std::uint8_t for LDR sources.
    template<typename T>
    EnvCdfResult buildEnvCdf(const std::vector<T>& pixels, int w, int h) {
        EnvCdfResult r;
        r.width = w;
        r.height = h;
        r.conditional.resize(static_cast<std::size_t>(w) * h);
        r.marginal.resize(h);

        // Row luminance sums, weighted by the equirect solid-angle Jacobian.
        // theta here is LATITUDE (0 at equator, ±π/2 at poles), so the Jacobian
        // is cos(theta), not sin(theta). Using sin here was a bug that swapped
        // equator/pole weights and caused ~15% furnace deviation on diffuse
        // dielectrics (plus visible banding on glossy reflections).
        for (int y = 0; y < h; ++y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(h);
            const float theta = (0.5f - v) * 3.14159265359f;
            const float cosTheta = std::max(std::abs(std::cos(theta)), 1e-6f);

            float rowSum = 0.f;
            for (int x = 0; x < w; ++x) {
                const std::size_t idx = (static_cast<std::size_t>(y) * w + x) * 4;
                float rf, gf, bf;
                if constexpr (std::is_same_v<T, float>) {
                    rf = pixels[idx]; gf = pixels[idx + 1]; bf = pixels[idx + 2];
                } else {
                    rf = pixels[idx] / 255.f; gf = pixels[idx + 1] / 255.f; bf = pixels[idx + 2] / 255.f;
                }
                const float lum = (0.2126f * rf + 0.7152f * gf + 0.0722f * bf) * cosTheta;
                rowSum += lum;
                r.conditional[static_cast<std::size_t>(y) * w + x] = rowSum;
            }
            // Normalize conditional CDF for this row
            if (rowSum > 1e-10f) {
                for (int x = 0; x < w; ++x)
                    r.conditional[static_cast<std::size_t>(y) * w + x] /= rowSum;
            } else {
                // Uniform fallback for black rows
                for (int x = 0; x < w; ++x)
                    r.conditional[static_cast<std::size_t>(y) * w + x] =
                        static_cast<float>(x + 1) / static_cast<float>(w);
            }
            r.totalSum += rowSum;
            r.marginal[y] = r.totalSum;
        }
        // Normalize marginal CDF
        if (r.totalSum > 1e-10f) {
            for (int y = 0; y < h; ++y)
                r.marginal[y] /= r.totalSum;
        } else {
            for (int y = 0; y < h; ++y)
                r.marginal[y] = static_cast<float>(y + 1) / static_cast<float>(h);
        }
        return r;
    }

}// namespace threepp::wgpu_pt

#endif//THREEPP_WGPUPATHTRACERENVCDF_HPP
