// Procedural detail maps for terrain (albedo + normal + roughness).
//
// Large surfaces (terrain) bake their unique-texel macro texture at an
// inevitably coarse per-metre density; MaterialWithDetailMap layers a small,
// world-XZ-anchored, distance-faded repeating field over it. This helper builds
// a coherent SET of detail maps from ONE shared tileable noise heightfield:
//
//   • albedo      — RGBA LINEAR, 0.5-neutral (the ×2 overlay keeps the macro
//                   colour's mean); luminance breakup + a subtle chroma wobble.
//   • normalRough — RGBA LINEAR: RGB = tangent-space normal from the height
//                   gradient (0.5 = flat), A = roughness modulation (0.5 =
//                   neutral) from inverted height (bump tops read slightly
//                   glossier than the pits).
//
// Because both derive from the same heightfield, the albedo grain, the relief
// lighting and the roughness breakup all line up. The field is periodic
// (wrapping value noise) so the texture tiles seamlessly; the shader distance-
// fades every term so nothing patterns or shimmers far away.
//
// Header-only, extras. Renderer-agnostic to build (plain DataTextures); only the
// Vulkan deferred G-buffer consumes the normal/roughness maps.

#ifndef THREEPP_EXTRAS_TERRAIN_DETAILTEXTURE_HPP
#define THREEPP_EXTRAS_TERRAIN_DETAILTEXTURE_HPP

#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace threepp::terrain {

    struct DetailMaps {
        std::shared_ptr<Texture> albedo;     // LINEAR, 0.5-neutral
        std::shared_ptr<Texture> normalRough;// LINEAR, rgb = normal, a = roughness
    };

    struct DetailMapOptions {
        int dim = 256;
        unsigned int seed = 4242u;

        // Albedo: luminance swing around 0.5 and a (deliberately subtle) chroma
        // wobble — the same field lands on snow/rock, where big hue swings read
        // as dye. Luminance carries the breakup; hue barely.
        float albedoContrast = 0.42f;
        float chroma = 0.06f;

        // Normal: metres of relief per unit heightfield → tangent slope. Larger
        // = craggier. The renderer additionally scales by detailNormalScale.
        float normalStrength = 2.2f;

        // Roughness: swing around the 0.5 neutral, from inverted height.
        float roughContrast = 0.5f;
    };

    inline DetailMaps makeDetailMaps(const DetailMapOptions& o = {}) {
        const int D = std::max(o.dim, 8);
        std::mt19937 rng(o.seed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);

        auto lattice = [&](int cells) {
            std::vector<float> v(static_cast<size_t>(cells) * cells);
            for (auto& x : v) x = u01(rng);
            return v;
        };
        // Periodic (wrapping) bilinear value noise — guarantees the map tiles.
        auto sampleLat = [](const std::vector<float>& lat, int cells, float u, float v) {
            const float fu = u * static_cast<float>(cells), fv = v * static_cast<float>(cells);
            const int iu = static_cast<int>(fu) % cells, iv = static_cast<int>(fv) % cells;
            const int ju = (iu + 1) % cells, jv = (iv + 1) % cells;
            const float tu = fu - std::floor(fu), tv = fv - std::floor(fv);
            const float a = lat[static_cast<size_t>(iv) * cells + iu], b = lat[static_cast<size_t>(iv) * cells + ju];
            const float c = lat[static_cast<size_t>(jv) * cells + iu], d = lat[static_cast<size_t>(jv) * cells + ju];
            return (a + (b - a) * tu) * (1.f - tv) + (c + (d - c) * tu) * tv;
        };
        const auto l8 = lattice(8), l32 = lattice(32), l8g = lattice(8);

        // Shared heightfield in [0,1], periodic. Two octaves + fine speckle.
        std::vector<float> hf(static_cast<size_t>(D) * D);
        std::mt19937 srng(o.seed ^ 0x9e3779b9u);
        std::uniform_real_distribution<float> su(0.f, 1.f);
        for (int j = 0; j < D; ++j)
            for (int i = 0; i < D; ++i) {
                const float u = (static_cast<float>(i) + 0.5f) / D, v = (static_cast<float>(j) + 0.5f) / D;
                const float n = 0.55f * sampleLat(l8, 8, u, v) + 0.30f * sampleLat(l32, 32, u, v) + 0.15f * su(srng);
                hf[static_cast<size_t>(j) * D + i] = n;
            }

        const auto at = [D](int x, int y) {
            x = (x % D + D) % D;// wrap → seamless normal at the border
            y = (y % D + D) % D;
            return static_cast<size_t>(y) * D + x;
        };

        std::vector<unsigned char> apx(static_cast<size_t>(D) * D * 4, 255u);
        std::vector<unsigned char> npx(static_cast<size_t>(D) * D * 4, 255u);
        for (int j = 0; j < D; ++j)
            for (int i = 0; i < D; ++i) {
                const float u = (static_cast<float>(i) + 0.5f) / D, v = (static_cast<float>(j) + 0.5f) / D;
                const float n = hf[at(i, j)];

                // ── albedo (LINEAR, 0.5-neutral)
                const float g = o.chroma * (sampleLat(l8g, 8, u, v) - 0.5f);
                const float base = 0.5f + o.albedoContrast * (n - 0.5f);
                const size_t ao = (static_cast<size_t>(j) * D + i) * 4;
                apx[ao + 0] = static_cast<unsigned char>(std::clamp(base - 0.5f * g, 0.f, 1.f) * 255.f + 0.5f);
                apx[ao + 1] = static_cast<unsigned char>(std::clamp(base + g, 0.f, 1.f) * 255.f + 0.5f);
                apx[ao + 2] = static_cast<unsigned char>(std::clamp(base - g, 0.f, 1.f) * 255.f + 0.5f);

                // ── normal from the wrapped height gradient (central diff)
                const float dhdx = (hf[at(i + 1, j)] - hf[at(i - 1, j)]) * 0.5f * o.normalStrength;
                const float dhdz = (hf[at(i, j + 1)] - hf[at(i, j - 1)]) * 0.5f * o.normalStrength;
                float nx = -dhdx, ny = -dhdz, nz = 1.f;
                const float il = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz);
                nx *= il;
                ny *= il;
                nz *= il;
                // ── roughness from inverted height (bump tops slightly glossier)
                const float rough = std::clamp(0.5f + o.roughContrast * (0.5f - n), 0.f, 1.f);
                const size_t no = (static_cast<size_t>(j) * D + i) * 4;
                npx[no + 0] = static_cast<unsigned char>((nx * 0.5f + 0.5f) * 255.f + 0.5f);
                npx[no + 1] = static_cast<unsigned char>((ny * 0.5f + 0.5f) * 255.f + 0.5f);
                npx[no + 2] = static_cast<unsigned char>((nz * 0.5f + 0.5f) * 255.f + 0.5f);
                npx[no + 3] = static_cast<unsigned char>(rough * 255.f + 0.5f);
            }

        auto mk = [D](std::vector<unsigned char> px) {
            auto t = DataTexture::create(ImageData{std::move(px)}, static_cast<unsigned int>(D), static_cast<unsigned int>(D));
            t->colorSpace = ColorSpace::Linear;// 0.5-neutral / normal data — must NOT sRGB-decode
            t->wrapS = t->wrapT = TextureWrapping::Repeat;
            t->magFilter = Filter::Linear;
            t->minFilter = Filter::LinearMipmapLinear;
            return t;
        };

        DetailMaps out;
        out.albedo = mk(std::move(apx));
        out.normalRough = mk(std::move(npx));
        return out;
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_DETAILTEXTURE_HPP
