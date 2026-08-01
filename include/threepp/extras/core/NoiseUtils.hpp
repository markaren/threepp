// Small deterministic noise/texture helpers shared by the procedural content
// generators under extras/ (vegetation, architecture, terrain dressing, ...).
//
// Everything here is pure, header-only and dependency-free. The lattice-based
// generators take an explicit `period` so a tile can be made to WRAP: sampling
// `valueNoise(u * period, v * period, period, seed)` over u,v in [0,1) yields a
// field that is continuous across the u=0/u=1 and v=0/v=1 seams, which is what
// lets a generated DataTexture be used with TextureWrapping::Repeat.

#ifndef THREEPP_EXTRAS_CORE_NOISEUTILS_HPP
#define THREEPP_EXTRAS_CORE_NOISEUTILS_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace threepp::noise {

    // Float [0,1] → byte, with rounding and clamping.
    inline unsigned char toByte(float v) {
        return static_cast<unsigned char>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
    }

    // Cheap integer hash → [0,1).
    inline float hash2(int x, int y, unsigned int seed) {
        uint32_t n = static_cast<uint32_t>(x) * 374761393u +
                     static_cast<uint32_t>(y) * 668265263u + seed * 362437u;
        n = (n ^ (n >> 13)) * 1274126177u;
        n = n ^ (n >> 16);
        return static_cast<float>(n & 0xffffffu) / static_cast<float>(0xffffff);
    }

    inline float hash1(int x, unsigned int seed) { return hash2(x, 0, seed); }

    // Smoothstep-style interpolant.
    inline float smooth(float t) { return t * t * (3.f - 2.f * t); }

    inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

    // Tileable 2D value noise over a [0,period) lattice.
    inline float valueNoise(float x, float y, int period, unsigned int seed) {
        const int xi = static_cast<int>(std::floor(x));
        const int yi = static_cast<int>(std::floor(y));
        const float fx = smooth(x - static_cast<float>(xi));
        const float fy = smooth(y - static_cast<float>(yi));
        auto wrap = [period](int v) { return ((v % period) + period) % period; };
        const float a = hash2(wrap(xi), wrap(yi), seed);
        const float b = hash2(wrap(xi + 1), wrap(yi), seed);
        const float c = hash2(wrap(xi), wrap(yi + 1), seed);
        const float d = hash2(wrap(xi + 1), wrap(yi + 1), seed);
        return (a * (1.f - fx) + b * fx) * (1.f - fy) +
               (c * (1.f - fx) + d * fx) * fy;
    }

    // Multi-octave value noise. `baseFreq` is in lattice cells per unit input;
    // each octave doubles both the frequency and the lattice period so every
    // octave keeps wrapping on the same [0,1) tile.
    inline float fbm(float x, float y, int basePeriod, unsigned int seed,
                     int octaves = 4, float gain = 0.5f) {
        float sum = 0.f, amp = 1.f, norm = 0.f;
        int period = basePeriod;
        float fx = x, fy = y;
        for (int o = 0; o < octaves; ++o) {
            sum += amp * valueNoise(fx, fy, period, seed + static_cast<unsigned int>(o) * 131u);
            norm += amp;
            amp *= gain;
            fx *= 2.f;
            fy *= 2.f;
            period *= 2;
        }
        return norm > 0.f ? sum / norm : 0.f;
    }

    // Wrap a coordinate back into [0,1).
    inline float wrap01(float a) { return a - std::floor(a); }

}// namespace threepp::noise

#endif//THREEPP_EXTRAS_CORE_NOISEUTILS_HPP
