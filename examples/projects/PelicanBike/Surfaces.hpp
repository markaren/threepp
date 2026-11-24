// Painted surfaces for the bicycle: leather for the saddle, the grips' knurled leather, and wicker. Every texture is
// made here per pixel with the watermill's workshop (TextureLab.hpp); nothing is loaded.

#ifndef THREEPP_EXAMPLE_PELICANBIKE_SURFACES_HPP
#define THREEPP_EXAMPLE_PELICANBIKE_SURFACES_HPP

#include "TextureLab.hpp"

namespace pb {

    using mill::clamp01;
    using mill::fbm2;
    using mill::Maps;
    using mill::mix;
    using mill::smooth;
    using mill::srgb;
    using mill::TexSet;
    using mill::V3;
    using mill::worley2;

    // brown saddle leather: a pebbled grain, creases, a polished-through sheen toward the middle of the tile
    inline TexSet leatherTexture(uint32_t seed) {
        const int N = 512;
        Maps m(N, N, 0.12f, 0.12f);
        mill::forRows(N, [&](int y) {
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N), v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
                const auto c = worley2(u * 38.f, v * 38.f, seed, 38, 38, 0.9f);
                const auto c2 = worley2(u * 14.f, v * 14.f, seed + 1, 14, 14, 0.9f);
                const float pebble = smooth(0.62f, 0.12f, c.f1);
                const float gap = smooth(0.06f, 0.0f, c.f2 - c.f1);
                const float crease = smooth(0.05f, 0.0f, c2.f2 - c2.f1);
                const float tone = 0.5f + 0.5f * fbm2(u * 6.f, v * 6.f, 4, seed + 2, 6, 6);
                const V3 base = mix(srgb(96, 52, 24), srgb(140, 82, 40), tone);
                const size_t i = m.at(x, y);
                m.albedo[i] = mix(base, base * 0.62f, std::max(gap * 0.9f, crease * 0.7f));
                m.height[i] = 0.00045f * pebble - 0.0004f * gap - 0.0007f * crease;
                m.rough[i] = 0.46f + 0.2f * (1.f - pebble) + 0.1f * tone;
                m.cavity[i] = 1.f - 0.5f * gap - 0.35f * crease;
            }
        });
        return m.finish(0.9f);
    }

    // the grip's leather: darker, with fine cross-hatching and a worn tan streak
    inline TexSet gripTexture(uint32_t seed) {
        const int N = 512;
        Maps m(N, N, 0.06f, 0.06f);
        mill::forRows(N, [&](int y) {
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N), v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
                const float hatch = 0.5f + 0.5f * std::sin((u + v) * 6.2831853f * 24.f) * std::sin((u - v) * 6.2831853f * 24.f);
                const auto c = worley2(u * 30.f, v * 30.f, seed, 30, 30, 0.8f);
                const float wear = smooth(0.2f, 0.8f, 0.5f + 0.5f * fbm2(u * 4.f, v * 4.f, 4, seed + 4, 4, 4));
                const V3 base = mix(srgb(58, 33, 20), srgb(118, 76, 42), wear * 0.7f);
                const size_t i = m.at(x, y);
                m.albedo[i] = base * (0.82f + 0.2f * hatch);
                m.height[i] = 0.0005f * hatch + 0.0003f * smooth(0.6f, 0.1f, c.f1);
                m.rough[i] = 0.62f - 0.25f * wear;
                m.cavity[i] = 0.7f + 0.3f * hatch;
            }
        });
        return m.finish(0.7f);
    }

    // a plain weave of cane strips, over and under, 0.24 m a tile
    inline TexSet wickerTexture(uint32_t seed) {
        const int N = 512;
        Maps m(N, N, 0.24f, 0.24f);
        const int strands = 12;
        mill::forRows(N, [&](int y) {
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N) * static_cast<float>(strands);
                const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N) * static_cast<float>(strands);
                const int iu = static_cast<int>(std::floor(u)), iv = static_cast<int>(std::floor(v));
                const float fu = u - std::floor(u), fv = v - std::floor(v);
                // which direction lies on top here
                const bool horizontalOver = ((iu + iv) & 1) == 0;
                const float across = horizontalOver ? fv : fu;
                const float along = horizontalOver ? fu : fv;
                const float round = std::sin(across * 3.14159265f);
                const float dip = std::sin(along * 3.14159265f);
                const float h = round * (0.4f + 0.6f * dip);
                const float shade = mill::hash2f(iu + (horizontalOver ? 100 : 0), iv, seed);
                const float fibre = 0.5f + 0.5f * fbm2(horizontalOver ? u * 0.7f : u * 18.f, horizontalOver ? v * 18.f : v * 0.7f, 3, seed + 1);
                const V3 base = mix(srgb(150, 108, 58), srgb(206, 164, 98), shade);
                const size_t i = m.at(x, y);
                m.albedo[i] = base * (0.82f + 0.25f * fibre) * (0.55f + 0.45f * round);
                m.height[i] = 0.0038f * h;
                m.rough[i] = 0.78f;
                m.cavity[i] = 0.35f + 0.65f * std::pow(round, 0.6f) * (0.6f + 0.4f * dip);
            }
        });
        return m.finish(1.3f);
    }

    struct SurfaceSet {
        TexSet leather, grip, wicker;
        SurfaceSet(): leather(leatherTexture(3)), grip(gripTexture(5)), wicker(wickerTexture(8)) {}
    };

}// namespace pb

#endif
