// Procedural facade / roof detail maps for geodata buildings.
//
// Same philosophy as DetailTexture.hpp (owned code, no assets): bake small
// tileable DataTextures at load. A facade tile spans exactly ONE window bay ×
// ONE floor — GeoBuildings.hpp emits wall UVs in (bay, floor) units snapped
// per building/edge, so window rows never cut at the roofline and columns
// never slice at corners; texture repeat is 1:1.
//
// Three variants, each an {albedo, normal, roughMetal} set:
//   • windowed — clapboard wall + recessed window (frame, cross mullion,
//     sill). Glass roughness ≈0.09 so the deferred renderer's env/sun
//     reflections light the windows — that glitter is most of the realism.
//   • plain    — clapboard + bay-boundary plank seams, no window (garages,
//     sheds, warehouses, sub-bay wall stubs).
//   • roof     — standing-seam field with down-slope weathering streaks.
//
// Map conventions (match the Vulkan deferred G-buffer + three.js):
//   albedo     RGBA LINEAR (ColorSpace::Linear → UNORM, no sRGB decode);
//              multiplies material colour × vertex colour, so it stays
//              near-white and lets the per-building tint carry the palette.
//   normal     tangent-space, +Z out, from the tile heightfield.
//   roughMetal G = roughness, B = metalness (three.js channel convention;
//              use as BOTH roughnessMap and metalnessMap).
//
// Header-only, extras.

#ifndef THREEPP_EXTRAS_TERRAIN_FACADETEXTURE_HPP
#define THREEPP_EXTRAS_TERRAIN_FACADETEXTURE_HPP

#include "threepp/extras/core/TextureBake.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

namespace threepp::terrain {

    struct FacadeSet {
        std::shared_ptr<Texture> albedo;    // LINEAR near-white, tint-friendly
        std::shared_ptr<Texture> normal;    // tangent-space
        std::shared_ptr<Texture> roughMetal;// g = roughness, b = metalness
    };

    struct FacadeMaps {
        FacadeSet windowed;
        FacadeSet plain;
        FacadeSet roof;
    };

    struct FacadeMapOptions {
        int dim = 256;
        unsigned int seed = 1337u;
        float normalStrength = 5.0f;// heightfield gradient → tangent slope
        float glassRoughness = 0.09f;
        float wallRoughness = 0.85f;
    };

    inline FacadeMaps makeFacadeMaps(const FacadeMapOptions& o = {}) {
        const int D = std::max(o.dim, 32);

        // Periodic value noise (same trick as DetailTexture) so tiles wrap.
        std::mt19937 rng(o.seed);
        const auto l16 = texgen::noiseLattice(rng, 16), l48 = texgen::noiseLattice(rng, 48);
        auto noise = [&](float u, float v) {// [0,1], periodic, two octaves
            return 0.65f * texgen::sampleLattice(l16, 16, u, v) +
                   0.35f * texgen::sampleLattice(l48, 48, u, v);
        };

        // Per-pixel surface description a variant fills in.
        struct Px {
            float h = 0.f;            // heightfield (arbitrary units)
            float r = 0.9f, g = 0.9f, b = 0.9f;// LINEAR albedo
            float rough = 0.85f;
            float metal = 0.f;
        };

        // Bake a {albedo, normal, roughMetal} set from a shade(u, v) function.
        auto bake = [&](auto&& shade) {
            std::vector<float> hf(static_cast<size_t>(D) * D);
            std::vector<unsigned char> apx(static_cast<size_t>(D) * D * 4, 255u);
            std::vector<unsigned char> rpx(static_cast<size_t>(D) * D * 4, 255u);
            for (int j = 0; j < D; ++j)
                for (int i = 0; i < D; ++i) {
                    const float u = (static_cast<float>(i) + 0.5f) / D;
                    const float v = 1.f - (static_cast<float>(j) + 0.5f) / D;// v up
                    const Px p = shade(u, v);
                    hf[static_cast<size_t>(j) * D + i] = p.h;
                    const size_t k = (static_cast<size_t>(j) * D + i) * 4;
                    apx[k + 0] = static_cast<unsigned char>(std::clamp(p.r, 0.f, 1.f) * 255.f + 0.5f);
                    apx[k + 1] = static_cast<unsigned char>(std::clamp(p.g, 0.f, 1.f) * 255.f + 0.5f);
                    apx[k + 2] = static_cast<unsigned char>(std::clamp(p.b, 0.f, 1.f) * 255.f + 0.5f);
                    rpx[k + 0] = 0u;
                    rpx[k + 1] = static_cast<unsigned char>(std::clamp(p.rough, 0.f, 1.f) * 255.f + 0.5f);
                    rpx[k + 2] = static_cast<unsigned char>(std::clamp(p.metal, 0.f, 1.f) * 255.f + 0.5f);
                }
            // Normals from the wrapped height gradient. NOTE v runs UP the
            // tile while rows run down the image: dh/dv = -(row gradient),
            // and three.js normal maps expect +G = +v.
            std::vector<unsigned char> npx(static_cast<size_t>(D) * D * 4, 255u);
            const auto at = [&](int x, int y) {
                x = (x % D + D) % D;
                y = (y % D + D) % D;
                return hf[static_cast<size_t>(y) * D + x];
            };
            for (int j = 0; j < D; ++j)
                for (int i = 0; i < D; ++i) {
                    const float dhdu = (at(i + 1, j) - at(i - 1, j)) * 0.5f * o.normalStrength;
                    const float dhdv = -(at(i, j + 1) - at(i, j - 1)) * 0.5f * o.normalStrength;
                    float nx = -dhdu, ny = -dhdv, nz = 1.f;
                    const float il = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz);
                    nx *= il;
                    ny *= il;
                    nz *= il;
                    const size_t k = (static_cast<size_t>(j) * D + i) * 4;
                    npx[k + 0] = static_cast<unsigned char>((nx * 0.5f + 0.5f) * 255.f + 0.5f);
                    npx[k + 1] = static_cast<unsigned char>((ny * 0.5f + 0.5f) * 255.f + 0.5f);
                    npx[k + 2] = static_cast<unsigned char>((nz * 0.5f + 0.5f) * 255.f + 0.5f);
                }
            auto mk = [&](std::vector<unsigned char> px) {
                return texgen::makeLinearRepeatTexture(std::move(px), static_cast<unsigned int>(D));
            };
            FacadeSet s;
            s.albedo = mk(std::move(apx));
            s.normal = mk(std::move(npx));
            s.roughMetal = mk(std::move(rpx));
            return s;
        };

        auto inRect = [](float u, float v, float u0, float u1, float v0, float v1) {
            return u >= u0 && u <= u1 && v >= v0 && v <= v1;
        };

        // Clapboard base shared by the wall variants: horizontal boards with a
        // shadowed under-edge, plus fine grain breakup.
        auto clapboard = [&](float u, float v) {
            Px p;
            const float board = v * 12.f - std::floor(v * 12.f);// 12 boards/floor
            p.h = 0.10f * board;                                // sawtooth relief
            const float grain = noise(u, v);
            const float shade = 1.f - 0.05f * (1.f - board);// darker under each lap
            const float lum = (0.90f + 0.06f * (grain - 0.5f)) * shade;
            p.r = p.g = p.b = lum;
            p.rough = o.wallRoughness + 0.08f * (grain - 0.5f);
            return p;
        };

        // ── windowed bay ────────────────────────────────────────────────────
        // Window band sits high in the tile (v ≥ ~0.3): the sunk foundation
        // strip and ground contact always land on plain wall.
        const FacadeSet windowed = bake([&](float u, float v) {
            Px p = clapboard(u, v);
            constexpr float FU0 = 0.28f, FU1 = 0.72f, FV0 = 0.35f, FV1 = 0.83f;// frame outer
            constexpr float GU0 = 0.315f, GU1 = 0.685f, GV0 = 0.385f, GV1 = 0.795f;// glass
            if (inRect(u, v, FU0, FU1, FV0, FV1)) {
                // frame (recessed from the wall, proud of the glass)
                p.h = -0.30f;
                p.r = p.g = p.b = 0.88f;
                p.rough = 0.55f;
                if (inRect(u, v, GU0, GU1, GV0, GV1)) {
                    // cross mullion splits the glass into four panes
                    const float um = 0.5f * (GU0 + GU1), vm = 0.5f * (GV0 + GV1);
                    if (std::abs(u - um) < 0.013f || std::abs(v - vm) < 0.011f) {
                        p.h = -0.36f;
                        p.r = p.g = p.b = 0.82f;
                        p.rough = 0.55f;
                    } else {
                        // glass: near-black, faint sky gradient, GLOSSY
                        p.h = -0.55f;
                        const float grad = (v - GV0) / (GV1 - GV0);
                        p.r = 0.028f + 0.012f * grad;
                        p.g = 0.034f + 0.016f * grad;
                        p.b = 0.046f + 0.022f * grad;
                        p.rough = o.glassRoughness;
                    }
                }
            } else if (inRect(u, v, 0.26f, 0.74f, 0.315f, 0.35f)) {
                // sill: proud ledge, slightly light
                p.h = 0.45f;
                p.r = p.g = p.b = 0.84f;
                p.rough = 0.6f;
            } else if (inRect(u, v, 0.26f, 0.74f, 0.295f, 0.315f)) {
                // shadow line under the sill
                p.r *= 0.62f;
                p.g *= 0.62f;
                p.b *= 0.62f;
            } else if (v < 0.295f) {
                // weathering streaks running down from the sill corners
                for (const float uc : {0.27f, 0.73f}) {
                    const float du = (u - uc) / 0.018f;
                    const float w = std::exp(-du * du) * (0.295f - v) / 0.295f;
                    const float f = 1.f - 0.14f * w;
                    p.r *= f;
                    p.g *= f;
                    p.b *= f;
                }
            }
            return p;
        });

        // ── plain cladding ──────────────────────────────────────────────────
        // Clapboard + a vertical plank seam at the bay boundary (u = 0/1 wraps
        // to one groove per bay).
        const FacadeSet plain = bake([&](float u, float v) {
            Px p = clapboard(u, v);
            const float du = std::min(u, 1.f - u);
            if (du < 0.012f) {
                p.h -= 0.25f * (1.f - du / 0.012f);
                const float f = 1.f - 0.15f * (1.f - du / 0.012f);
                p.r *= f;
                p.g *= f;
                p.b *= f;
            }
            return p;
        });

        // ── roof: standing seams + down-slope weathering ────────────────────
        // MATTE (rough 0.9): the palette is slate/felt/tile. Anything glossier
        // lets grazing-angle Fresnel bounce the whole sky off dark roofs —
        // from a distance every town roof flared near-WHITE at 0.72.
        const FacadeSet roof = bake([&](float u, float v) {
            Px p;
            const float streak = noise(u * 4.f, v * 0.5f);// elongated down-slope
            const float grain = noise(u, v);
            const float lum = 0.86f + 0.05f * (streak - 0.5f) + 0.03f * (grain - 0.5f);
            p.r = p.g = p.b = lum;
            p.rough = 0.90f + 0.06f * (grain - 0.5f);
            p.h = 0.f;
            const float su = u * 6.f - std::floor(u * 6.f);// 6 seams / tile
            const float ds = std::min(su, 1.f - su);
            if (ds < 0.035f) {
                const float t = 1.f - ds / 0.035f;
                p.h = 0.5f * t;// raised seam rib
                const float f = 1.f - 0.08f * t;
                p.r *= f;
                p.g *= f;
                p.b *= f;
            }
            return p;
        });

        return FacadeMaps{windowed, plain, roof};
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_FACADETEXTURE_HPP
