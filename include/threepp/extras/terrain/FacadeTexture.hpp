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
#include "threepp/math/Rng.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

namespace threepp::terrain {

    struct FacadeSet {
        std::shared_ptr<Texture> albedo;    // LINEAR near-white, tint-friendly
        std::shared_ptr<Texture> normal;    // tangent-space
        std::shared_ptr<Texture> roughMetal;// g = roughness, b = metalness
    };

    // Nine sets: three per facade CLASS (wood / masonry / industrial) plus
    // three roof coverings. GeoBuildings.hpp picks by building type — a town
    // centre of rendered masonry does not read like a suburb of painted wood
    // when both wear the same clapboard.
    struct FacadeMaps {
        FacadeSet windowed;     // wood, upper + ground (2 bays wide)
        FacadeSet plain;        // wood cladding, no window (1 bay)
        FacadeSet roof;         // standing-seam metal / felt — flat + industrial
        FacadeSet masonryUpper; // rendered masonry, tall windows (2 bays)
        FacadeSet masonryGround;// shopfront glazing over a rusticated base
        FacadeSet masonryPlain; // rendered masonry, blind wall
        FacadeSet industrial;   // corrugated sheet with a high window band
        FacadeSet roofSlate;    // fine slate courses (masonry pitched roofs)
        FacadeSet roofTile;     // concrete/clay tile courses (wood pitched)
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
        math::Rng rng(o.seed);
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
        // 0.2 m boards over a 3 m storey, and the lap gets a crisp SHADOW LINE,
        // not the soft 5% gradient it had: at eye level a painted clapboard wall
        // is read entirely off those lines, and 0.05 washed out to plaster.
        auto clapboard = [&](float u, float v) {
            Px p;
            const float boards = 15.f;// 0.2 m boards over a 3 m storey
            const float board = v * boards - std::floor(v * boards);
            p.h = 0.30f * board - 0.30f;// each board laps over the one below
            const float grain = noise(u, v);
            float shade = 1.f - 0.045f * (1.f - board);
            if (board < 0.14f) {// the lap's own shadow
                const float t = 1.f - board / 0.14f;
                shade *= 1.f - 0.085f * t;
                p.h -= 0.34f * t;
            }
            const float lum = (0.90f + 0.06f * (grain - 0.5f)) * shade;
            p.r = p.g = p.b = lum;
            p.rough = o.wallRoughness + 0.08f * (grain - 0.5f);
            return p;
        };

        // Smooth rendered masonry: no board relief, a fine stucco grain.
        auto render = [&](float u, float v) {
            Px p;
            const float grain = noise(u * 2.f, v * 2.f);
            const float lum = 0.90f + 0.045f * (grain - 0.5f);
            p.r = p.g = p.b = lum;
            p.h = 0.02f * grain;
            p.rough = 0.80f + 0.10f * (grain - 0.5f);
            return p;
        };

        // Glass pane shared by every windowed variant: near-black with a faint
        // sky gradient and roughness ≈0.09, so the deferred renderer's env/sun
        // reflections do the glittering.
        auto glass = [&](Px& p, float v, float v0, float v1, float bright) {
            p.h = -0.55f;
            const float grad = (v - v0) / std::max(1e-3f, v1 - v0);
            p.r = bright * (0.42f + 0.18f * grad);
            p.g = bright * (0.50f + 0.24f * grad);
            p.b = bright * (0.68f + 0.33f * grad);
            p.rough = o.glassRoughness;
        };

        // One window: outer frame [u0,u1]x[v0,v1], glass inset by `in`, with a
        // cross mullion and (optionally) a transom two thirds up.
        auto window = [&](Px& p, float u, float v, float u0, float u1, float v0, float v1,
                          float in, bool transom, float frameLum, float bright) {
            if (!inRect(u, v, u0, u1, v0, v1)) return false;
            p.h = -0.30f;
            p.r = p.g = p.b = frameLum;
            p.rough = 0.55f;
            const float gu0 = u0 + in, gu1 = u1 - in, gv0 = v0 + in * 0.85f, gv1 = v1 - in * 0.85f;
            if (!inRect(u, v, gu0, gu1, gv0, gv1)) return true;
            const float um = 0.5f * (gu0 + gu1);
            const float vt = transom ? gv0 + 0.68f * (gv1 - gv0) : 0.5f * (gv0 + gv1);
            if (std::abs(u - um) < 0.012f || std::abs(v - vt) < 0.011f) {
                p.h = -0.36f;
                p.r = p.g = p.b = frameLum * 0.93f;
                p.rough = 0.55f;
            } else {
                glass(p, v, gv0, gv1, bright);
            }
            return true;
        };

        // ── wood: windowed bay pair ─────────────────────────────────────────
        // TWO bays per tile with DIFFERENT windows (left plain, right with a
        // transom): the eye stops reading a stamped repeat. GeoBuildings gives
        // every wall edge a random integer bay phase on top of that. The window
        // band sits high (v ≥ ~0.3) so the sunk foundation strip and the ground
        // contact always land on plain cladding.
        const FacadeSet windowed = bake([&](float u, float v) {
            Px p = clapboard(u, v);
            bool inWin = false;
            for (int k = 0; k < 2; ++k) {
                const float uc = 0.25f + 0.5f * static_cast<float>(k);
                // frameLum 0.52, not 0.88: the frame is the TRIM, and a trim a
                // hair off the wall colour is invisible by 20 m. Against a white
                // painted wall this is the dark-framed window of the photo.
                if (window(p, u, v, uc - 0.120f, uc + 0.120f, 0.35f, 0.83f, 0.042f,
                           k == 1, 0.52f, 0.068f)) {
                    inWin = true;
                    break;
                }
                if (inRect(u, v, uc - 0.140f, uc + 0.140f, 0.310f, 0.35f)) {
                    p.h = 0.45f;// sill
                    p.r = p.g = p.b = 0.58f;
                    p.rough = 0.6f;
                    inWin = true;
                    break;
                }
                if (inRect(u, v, uc - 0.135f, uc + 0.135f, 0.295f, 0.315f)) {
                    p.r *= 0.62f;// shadow under the sill
                    p.g *= 0.62f;
                    p.b *= 0.62f;
                    inWin = true;
                    break;
                }
            }
            if (!inWin && v < 0.295f) {
                for (const float uc : {0.115f, 0.385f, 0.615f, 0.885f}) {
                    const float du = (u - uc) / 0.018f;
                    const float w = std::exp(-du * du) * (0.295f - v) / 0.295f;
                    const float f = 1.f - 0.07f * w;
                    p.r *= f;
                    p.g *= f;
                    p.b *= f;
                }
            }
            return p;
        });

        // ── plain cladding ──────────────────────────────────────────────────
        // Plain clapboard, nothing else. The board that used to sit at u = 0/1
        // wrapped once per BAY, not once per building corner, so a wooden wall
        // came out as a run of panels with vertical seams — the flat plaster
        // read of the suburb shot. Real corner boards are geometry now
        // (GeoBuildings emits them in the trim colour at the ring vertices).
        const FacadeSet plain = bake([&](float u, float v) { return clapboard(u, v); });

        // ── masonry: upper floors ───────────────────────────────────────────
        // Two tall windows per tile with a stone surround and a proud sill,
        // and a string course along the top of the tile.
        const FacadeSet masonryUpper = bake([&](float u, float v) {
            Px p = render(u, v);
            for (int k = 0; k < 2; ++k) {
                const float uc = 0.25f + 0.5f * static_cast<float>(k);
                if (inRect(u, v, uc - 0.155f, uc + 0.155f, 0.235f, 0.895f) &&
                    !inRect(u, v, uc - 0.12f, uc + 0.12f, 0.26f, 0.87f)) {
                    p.h = 0.30f;// stone surround, proud of the render
                    p.r = p.g = p.b = 0.95f;
                    p.rough = 0.72f;
                }
                if (window(p, u, v, uc - 0.12f, uc + 0.12f, 0.26f, 0.87f, 0.030f,
                           k == 0, 0.90f, 0.100f))
                    break;
                if (inRect(u, v, uc - 0.17f, uc + 0.17f, 0.205f, 0.24f)) {
                    p.h = 0.50f;// sill
                    p.r = p.g = p.b = 0.95f;
                    p.rough = 0.70f;
                } else if (inRect(u, v, uc - 0.17f, uc + 0.17f, 0.185f, 0.205f)) {
                    p.r *= 0.60f;
                    p.g *= 0.60f;
                    p.b *= 0.60f;
                }
            }
            if (v > 0.955f) {
                p.h = 0.35f;// string course
                p.r *= 1.02f;
                p.g *= 1.02f;
                p.b *= 1.02f;
            } else if (v > 0.935f) {
                p.r *= 0.72f;
                p.g *= 0.72f;
                p.b *= 0.72f;
            }
            return p;
        });

        // ── masonry: ground floor ───────────────────────────────────────────
        // Shopfront glazing with a transom over a rusticated dark base band —
        // the single strongest "town centre, not suburb" cue at street level.
        const FacadeSet masonryGround = bake([&](float u, float v) {
            Px p = render(u, v);
            if (v < 0.20f) {// rusticated plinth: deep horizontal joints
                const float band = v * 8.f - std::floor(v * 8.f);
                p.h = -0.10f + 0.18f * band;
                const float f = 0.70f - 0.10f * (1.f - band);
                p.r *= f;
                p.g *= f;
                p.b *= f;
                p.rough = 0.88f;
                return p;
            }
            for (int k = 0; k < 2; ++k) {
                const float uc = 0.25f + 0.5f * static_cast<float>(k);
                if (window(p, u, v, uc - 0.18f, uc + 0.18f, 0.235f, 0.90f, 0.028f,
                           true, 0.55f, 0.105f))
                    break;
            }
            if (v > 0.945f) {
                p.h = 0.35f;// fascia band over the shopfronts
                p.r *= 0.82f;
                p.g *= 0.82f;
                p.b *= 0.82f;
            }
            return p;
        });

        // ── masonry: blind wall ─────────────────────────────────────────────
        const FacadeSet masonryPlain = bake([&](float u, float v) {
            Px p = render(u, v);
            if (v < 0.20f) {
                const float band = v * 8.f - std::floor(v * 8.f);
                p.h = -0.10f + 0.18f * band;
                const float f = 0.74f - 0.08f * (1.f - band);
                p.r *= f;
                p.g *= f;
                p.b *= f;
            } else if (v > 0.955f) {
                p.h = 0.35f;
            }
            const float du = std::min(u, 1.f - u);
            if (du < 0.010f) {// quoin joint at the tile edge
                p.r *= 0.92f;
                p.g *= 0.92f;
                p.b *= 0.92f;
            }
            return p;
        });

        // ── industrial: profiled sheet ──────────────────────────────────────
        const FacadeSet industrial = bake([&](float u, float v) {
            Px p;
            const float grain = noise(u * 3.f, v * 3.f);
            const float rib = std::sin(v * 44.f * 3.14159265f);
            p.h = 0.22f * rib;
            const float lum = 0.88f + 0.06f * rib + 0.04f * (grain - 0.5f);
            p.r = p.g = p.b = lum;
            p.rough = 0.62f + 0.10f * (grain - 0.5f);
            p.metal = 0.25f;
            // one high strip window band, only over part of the tile
            if (inRect(u, v, 0.10f, 0.44f, 0.70f, 0.84f) ||
                inRect(u, v, 0.56f, 0.90f, 0.70f, 0.84f)) {
                const bool frame = v < 0.715f || v > 0.825f ||
                                   std::abs(u - 0.27f) < 0.010f || std::abs(u - 0.73f) < 0.010f;
                if (frame) {
                    p.h = -0.20f;
                    p.r = p.g = p.b = 0.80f;
                    p.rough = 0.55f;
                    p.metal = 0.f;
                } else {
                    glass(p, v, 0.715f, 0.825f, 0.090f);
                    p.metal = 0.f;
                }
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

        // ── roof: slate courses ─────────────────────────────────────────────
        // Fine dark courses with staggered vertical joints. MATTE for the same
        // reason as the metal roof: a glossy dark roof flares white at grazing
        // angles and every town roof reads as snow from the air.
        // Base luminance 0.70, not 0.87: this map is a MULTIPLIER on the roof's
        // vertex colour, and a near-white multiplier over a near-white wall tile
        // is why a slate roof and a white wall came out the same brightness.
        // 8 courses over the 3 m repeat = 0.375 m slates, wide enough to survive
        // the mip chain at 400 m where 12 rows averaged to a flat field.
        const FacadeSet roofSlate = bake([&](float u, float v) {
            Px p;
            const float rows = 8.f;
            const float row = v * rows;
            const float ri = std::floor(row);
            const float rf = row - ri;
            const float stagger = (static_cast<int>(ri) & 1) ? 0.5f : 0.f;
            const float col = u * 6.f + stagger;
            const float cf = col - std::floor(col);
            const float grain = noise(u * 2.f, v * 2.f);
            const float slate = noise(std::floor(col) * 0.13f, ri * 0.29f);
            float lum = 0.70f + 0.085f * (slate - 0.5f) + 0.025f * (grain - 0.5f);
            p.h = 0.14f * rf;// each course laps over the one below
            if (rf < 0.13f) {// course shadow line
                lum *= 0.70f;
                p.h -= 0.26f;
            }
            if (std::min(cf, 1.f - cf) < 0.026f) {// vertical joint
                lum *= 0.84f;
                p.h -= 0.12f;
            }
            p.r = p.g = p.b = lum;
            p.rough = 0.90f + 0.05f * (grain - 0.5f);
            return p;
        });

        // ── roof: tile courses ──────────────────────────────────────────────
        const FacadeSet roofTile = bake([&](float u, float v) {
            Px p;
            const float rows = 7.f;
            const float row = v * rows;
            const float ri = std::floor(row);
            const float rf = row - ri;
            const float col = u * 6.f + ((static_cast<int>(ri) & 1) ? 0.5f : 0.f);
            const float cf = col - std::floor(col);
            const float barrel = std::sin(cf * 3.14159265f);// rounded pantile
            const float grain = noise(u * 2.f, v * 2.f);
            float lum = 0.74f + 0.10f * (barrel - 0.5f) + 0.035f * (grain - 0.5f);
            p.h = 0.30f * barrel + 0.20f * rf;
            if (rf < 0.13f) {// deep shadow under each lap
                lum *= 0.66f;
                p.h -= 0.35f;
            }
            p.r = p.g = p.b = lum;
            p.rough = 0.88f + 0.06f * (grain - 0.5f);
            return p;
        });

        return FacadeMaps{windowed, plain, roof, masonryUpper, masonryGround,
                          masonryPlain, industrial, roofSlate, roofTile};
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_FACADETEXTURE_HPP
