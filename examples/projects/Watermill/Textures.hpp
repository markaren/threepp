// The watermill's surfaces, each a recipe painted per pixel (TextureLab.hpp): weathered boards, rough-cut stone,
// water-worn sandstone, the forest floor's litter, furrowed bark, rusted iron, a leaf atlas, grimy glass and the
// water's ripples. Every recipe tiles (its noise is periodic over the tile) unless it says otherwise.

#ifndef THREEPP_EXAMPLE_WATERMILL_TEXTURES_HPP
#define THREEPP_EXAMPLE_WATERMILL_TEXTURES_HPP

#include "TextureLab.hpp"

#include <array>
#include <random>
#include <vector>

namespace mill {

    // wrapped difference of two tile coordinates, in [-0.5, 0.5)
    inline float wrapDelta(float a, float b) {
        float d = a - b;
        d -= std::floor(d + 0.5f);
        return d;
    }

    // ---- weathered wood: boards, posts, beams ---------------------------------------------------------------------
    // The grain runs along v. Sun and rain have eaten the soft early wood of each growth ring, so the late wood stands
    // proud in ridges; checks split along the grain; knots, water streaks, dark fungal speckle and the odd lichen.
    // The tile is 0.32 m across the grain and 1.28 m along it. `grey` 0..1: how silvered the surface is.
    inline TexSet woodTexture(uint32_t seed, float grey = 0.65f, float lichen = 1.f) {
        const int W = 1024, H = 2048;
        const float Wm = 0.32f, Hm = 1.28f;
        Maps m(W, H, Wm, Hm);
        struct Knot {
            float u, v, r;
        };
        std::vector<Knot> knots;
        Rng rng(seed);
        const int knotCount = 2 + static_cast<int>(rng() * 2.f);
        for (int k = 0; k < knotCount; ++k) knots.push_back({rng(), rng(), rng.range(0.006f, 0.016f)});
        const int rings = 96;// across the tile
        const V3 silver = srgb(146, 140, 129), brown = srgb(116, 96, 74), dark = srgb(58, 52, 46);
        const V3 knotDark = srgb(56, 40, 28), crackCol = srgb(28, 24, 20), lichenCol = srgb(168, 172, 142);
        forRows(H, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(H);
            for (int x = 0; x < W; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(W);
                const size_t c = m.at(x, y);
                // the knots: rings swell round them; inside, rings of their own
                float deflect = 0, inKnot = 0, knotPhase = 0;
                for (const auto& k : knots) {
                    const float du = wrapDelta(u, k.u) * Wm, dv = wrapDelta(v, k.v) * Hm;
                    const float d = std::sqrt((du / k.r) * (du / k.r) + (dv / (1.7f * k.r)) * (dv / (1.7f * k.r)));
                    deflect += (du >= 0 ? 1.f : -1.f) * 1.6f * k.r * std::exp(-0.35f * d * d) * smooth(0.f, 1.2f, d);
                    const float in = smooth(1.05f, 0.85f, d);
                    if (in > inKnot) {
                        inKnot = in;
                        knotPhase = d * k.r / 0.0016f;
                    }
                }
                const float wobble = 0.010f * fbm2(u * 4, v * 2, 4, seed + 1, 4, 2) + 0.003f * fbm2(u * 16, v * 8, 3, seed + 2, 16, 8);
                const float phase = static_cast<float>(rings) * (u + (wobble + deflect) / Wm);
                float f = phase - std::floor(phase);
                if (inKnot > 0) f = mix(V3(f, 0, 0), V3(knotPhase - std::floor(knotPhase), 0, 0), inKnot).r;
                const float late = smooth(0.66f, 0.82f, f) * (1.f - smooth(0.93f, 1.f, f));
                const float fibre = fbm2(u * 256, v * 8, 3, seed + 3, 256, 8);

                // checks: long thin splits along the grain, each with ends
                const float cn = perlin2(u * 26, v * 1.f, seed + 4, 26, 1);
                const float widthHere = 0.022f * clamp01(1.8f * fbm2(u * 7, v * 3, 2, seed + 5, 7, 3) - 0.15f);
                const float crack = widthHere > 0 ? smooth(widthHere, widthHere * 0.25f, std::abs(cn)) : 0.f;

                float hgt = 0.00035f * late + 0.00007f * fibre + 0.0012f * fbm2(u * 2, v * 1, 3, seed + 6, 2, 1);
                hgt -= 0.0018f * crack;
                hgt += 0.0002f * inKnot;

                // colour
                const float weather = clamp01(grey + 0.45f * fbm2(u * 2, v * 0.5f + 0.3f, 3, seed + 7, 2, 1));
                V3 col = mix(brown, silver, weather);
                col = col * (0.78f + 0.32f * late + 0.06f * fibre);
                const float streak = fbm2(u * 9, v * 0.6f, 3, seed + 8, 9, 1);
                col = mix(col, dark, 0.5f * smooth(0.2f, 0.75f, streak));
                // fungal speckle in patches
                const Cell sp = worley2(u * 80, v * 320, seed + 9, 80, 320);
                const float patch = smooth(0.15f, 0.55f, fbm2(u * 3, v * 1.5f, 2, seed + 10, 3, 2));
                col = mix(col, dark * 0.6f, 0.8f * patch * smooth(0.3f, 0.12f, sp.f1) * (unit(sp.id) < 0.5f ? 1.f : 0.f));
                // lichen: faint pale patches a hand across, here and there, never dots
                const float lichenHere = lichen * smooth(0.45f, 0.75f, fbm2(u * 3, v * 3, 4, seed + 11, 3, 3)) * smooth(0.f, 0.5f, fbm2(u * 24, v * 96, 3, seed + 12, 24, 96));
                col = mix(col, lichenCol, 0.25f * lichenHere);
                hgt += 0.0004f * lichenHere;
                col = mix(col, knotDark, 0.85f * inKnot * (0.6f + 0.4f * late));
                col = mix(col, crackCol, crack);
                col = col * (0.94f + 0.12f * hash2f(x, y, seed + 13));

                m.albedo[c] = col;
                m.height[c] = hgt;
                m.rough[c] = 0.8f + 0.1f * (1.f - late) + 0.08f * crack - 0.1f * inKnot;
            }
        });
        m.bakeCavity(5, 0.0007f, 0.45f);
        return m.finish(0.8f);
    }

    // ---- rough-cut building stone ---------------------------------------------------------------------------------
    // A hammer-dressed sandstone face: broad undulation, a granular surface with pits, iron staining, black crust
    // lichen in clusters and pale rosettes. 0.9 m a tile. The per-stone colour comes from the mesh.
    inline TexSet stoneTexture(uint32_t seed) {
        const int N = 1024;
        const float S = 0.9f;
        Maps m(N, N, S, S);
        const V3 grey = srgb(150, 146, 138), buff = srgb(160, 146, 122), iron = srgb(140, 110, 82);
        const V3 crust = srgb(52, 52, 48), rosette = srgb(200, 202, 186);
        forRows(N, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N);
                const size_t c = m.at(x, y);
                // dressing: shallow scallops left by the hammer, then the grain
                const Cell sc = worley2(u * 14, v * 14, seed, 14, 14);
                float hgt = 0.0035f * smooth(0.f, 0.7f, sc.f1) * -1.f;
                hgt += 0.006f * fbm2(u * 3, v * 3, 5, seed + 1, 3, 3);
                hgt += 0.0012f * fbm2(u * 48, v * 48, 4, seed + 2, 48, 48);
                const Cell pit = worley2(u * 70, v * 70, seed + 3, 70, 70);
                const float pitHere = (unit(pit.id) < 0.35f ? 1.f : 0.f) * smooth(0.28f, 0.08f, pit.f1);
                hgt -= 0.0016f * pitHere;

                V3 col = mix(grey, buff, clamp01(0.5f + 0.9f * fbm2(u * 2, v * 2, 3, seed + 4, 2, 2)));
                col = mix(col, iron, 0.55f * smooth(0.25f, 0.8f, fbm2(u * 3 + 0.4f, v * 3, 3, seed + 5, 3, 3)));
                // crust lichen: black dots gathered in clouds
                const float cloud = smooth(0.1f, 0.6f, fbm2(u * 4, v * 4, 3, seed + 6, 4, 4));
                const Cell dot = worley2(u * 110, v * 110, seed + 7, 110, 110);
                const float crustHere = cloud * (unit(dot.id) < 0.4f ? 1.f : 0.f) * smooth(0.3f, 0.1f, dot.f1);
                col = mix(col, crust, 0.45f * crustHere);
                hgt += 0.0004f * crustHere;
                // pale rosettes, a few cm across, edges ragged
                const Cell ro = worley2(u * 9, v * 9, seed + 8, 9, 9);
                const float ragged = 0.07f * perlin2(u * 120, v * 120, seed + 9, 120, 120);
                const float rosetteHere = (unit(ro.id) < 0.05f ? 1.f : 0.f) * smooth(0.34f, 0.18f, ro.f1 + 2.5f * ragged);
                col = mix(col, rosette, 0.35f * rosetteHere);
                hgt += 0.0005f * rosetteHere;
                col = col * (0.9f + 0.2f * hash2f(x, y, seed + 10)) * (1.f - 0.25f * pitHere);
                m.albedo[c] = col;
                m.height[c] = hgt;
                m.rough[c] = 0.88f + 0.08f * hash2f(x, y, seed + 11) - 0.1f * rosetteHere;
            }
        });
        m.bakeCavity(6, 0.0015f, 0.5f);
        return m.finish(0.9f);
    }

    // ---- water-worn sandstone: boulders, ledges, the cliff ----------------------------------------------------------
    // Rounder and finer than the dressed stone: a sand grain speckle, faint bedding, weathering pits and lichen. Near
    // neutral, since each rock's colour (and its wet foot and moss) comes from the mesh. 1.6 m a tile.
    inline TexSet rockTexture(uint32_t seed) {
        const int N = 1024;
        const float S = 1.6f;
        Maps m(N, N, S, S);
        const V3 light = srgb(196, 188, 172), mid = srgb(160, 150, 134), lichen = srgb(212, 214, 198), crust = srgb(70, 68, 62);
        forRows(N, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N);
                const size_t c = m.at(x, y);
                const float broad = fbm2(u * 2, v * 2, 5, seed, 2, 2);
                const float bedding = std::sin(6.2831853f * (v * 22.f + 0.8f * fbm2(u * 2, v * 2, 3, seed + 1, 2, 2)));
                float hgt = 0.012f * broad + 0.0008f * bedding + 0.0012f * fbm2(u * 40, v * 40, 4, seed + 2, 40, 40);
                const Cell pit = worley2(u * 36, v * 36, seed + 3, 36, 36);
                const float pitHere = (unit(pit.id) < 0.1f ? 1.f : 0.f) * smooth(0.3f, 0.12f, pit.f1) * smooth(0.f, 0.4f, fbm2(u * 5, v * 5, 2, seed + 11, 5, 5));
                hgt -= 0.0012f * pitHere;
                V3 col = mix(mid, light, clamp01(0.5f + 0.8f * broad + 0.1f * bedding));
                const Cell ro = worley2(u * 7, v * 7, seed + 4, 7, 7);
                const float rag = 0.06f * perlin2(u * 100, v * 100, seed + 5, 100, 100);
                const float lichenHere = (unit(ro.id) < 0.12f ? 1.f : 0.f) * smooth(0.36f, 0.22f, ro.f1 + 2.f * rag) * smooth(-0.2f, 0.3f, fbm2(u * 30, v * 30, 3, seed + 10, 30, 30));
                col = mix(col, lichen, 0.45f * lichenHere);
                const Cell dot = worley2(u * 90, v * 90, seed + 6, 90, 90);
                const float crustHere = smooth(0.35f, 0.8f, fbm2(u * 3, v * 3, 2, seed + 7, 3, 3)) * smooth(0.3f, 0.12f, dot.f1) * (unit(dot.id) < 0.3f ? 1.f : 0.f);
                col = mix(col, crust, 0.2f * crustHere);
                col = col * (0.88f + 0.24f * hash2f(x, y, seed + 8)) * (1.f - 0.1f * pitHere);
                m.albedo[c] = col;
                m.height[c] = hgt;
                m.rough[c] = 0.82f + 0.12f * hash2f(x, y, seed + 9);
            }
        });
        m.bakeCavity(8, 0.002f, 0.5f);
        return m.finish(0.8f);
    }

    // ---- lime mortar and render ------------------------------------------------------------------------------------
    // the lime between the stones and over the inside of the foundation: a sandy, uneven render, grimy, damp-stained
    // low down, the grit standing out; no lichen. 1.2 m a tile.
    inline TexSet limeTexture(uint32_t seed) {
        const int N = 1024;
        const float S = 1.2f;
        Maps m(N, N, S, S);
        const V3 lime = srgb(176, 168, 152), grime = srgb(120, 112, 96);
        forRows(N, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N);
                const size_t c = m.at(x, y);
                const float broad = fbm2(u * 3, v * 3, 4, seed, 3, 3);
                const float grit = fbm2(u * 160, v * 160, 2, seed + 1, 160, 160);
                const float streak = fbm2(u * 14, v * 1.5f, 3, seed + 2, 14, 2);
                V3 col = mix(lime, grime, clamp01(0.35f + 0.6f * broad + 0.3f * smooth(0.f, 0.8f, streak)));
                col = col * (0.92f + 0.1f * grit + 0.06f * hash2f(x, y, seed + 3));
                m.albedo[c] = col;
                m.height[c] = 0.004f * broad + 0.0006f * grit;
                m.rough[c] = 0.95f;
            }
        });
        m.bakeCavity(6, 0.0015f, 0.6f);
        return m.finish(0.8f);
    }

    // ---- the forest floor ------------------------------------------------------------------------------------------
    // Dark soil under last year's leaves, a few twigs. 3 m a tile, near neutral brown: the mesh's colour turns it
    // greener (moss, grass) or wetter.
    inline TexSet groundTexture(uint32_t seed) {
        const int N = 2048;
        const float S = 3.f;
        Maps m(N, N, S, S);
        const V3 soil = srgb(64, 50, 38);
        forRows(N, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N);
                const size_t c = m.at(x, y);
                const float n = fbm2(u * 6, v * 6, 5, seed, 6, 6);
                m.albedo[c] = soil * (0.75f + 0.4f * clamp01(0.5f + n));
                m.height[c] = 0.004f * n + 0.001f * fbm2(u * 80, v * 80, 3, seed + 1, 80, 80);
                m.rough[c] = 0.95f;
            }
        });
        // leaves stamped one over another; each lies on what is under it
        Rng rng(seed + 2);
        const std::array<V3, 6> leafCols{srgb(104, 72, 44), srgb(134, 102, 66), srgb(78, 58, 40), srgb(112, 98, 80), srgb(150, 118, 64), srgb(92, 62, 38)};
        const float px = S / static_cast<float>(N);
        for (int k = 0; k < 2600; ++k) {
            const float cx = rng() * N, cy = rng() * N;
            const float len = rng.range(0.035f, 0.095f) / px, wid = len * rng.range(0.3f, 0.55f);
            const float ang = rng() * 6.2831853f, ca = std::cos(ang), sa = std::sin(ang);
            const V3 col = leafCols[rng.next() % leafCols.size()] * rng.range(0.8f, 1.15f);
            const float curl = rng.range(0.0005f, 0.0025f);
            const int r = static_cast<int>(len * 0.6f) + 2;
            const int x0 = static_cast<int>(cx), y0 = static_cast<int>(cy);
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    const float lx = (static_cast<float>(dx) + static_cast<float>(x0) - cx) * ca + (static_cast<float>(dy) + static_cast<float>(y0) - cy) * sa;
                    const float ly = -(static_cast<float>(dx) + static_cast<float>(x0) - cx) * sa + (static_cast<float>(dy) + static_cast<float>(y0) - cy) * ca;
                    const float t = lx / len + 0.5f;// 0 at the stem, 1 at the tip
                    if (t < 0 || t > 1) continue;
                    const float half = 0.5f * wid * std::pow(std::sin(3.14159f * std::pow(t, 0.8f)), 0.9f);
                    const float edge = half - std::abs(ly);
                    if (edge < 0) continue;
                    const size_t c = m.at(wrapi(x0 + dx, N), wrapi(y0 + dy, N));
                    const float lift = curl * (std::abs(ly) / std::max(half, 1.f));
                    const float top = m.height[c] + 0.0008f + lift;
                    m.height[c] = std::max(m.height[c], top);
                    const float rib = smooth(1.2f, 0.f, std::abs(ly)) * 0.25f;
                    m.albedo[c] = col * (0.9f + rib + 0.1f * hash2f(x0 + dx, y0 + dy, seed + 3));
                    m.rough[c] = 0.85f;
                }
            }
        }
        // twigs
        for (int k = 0; k < 70; ++k) {
            const float cx = rng() * N, cy = rng() * N, len = rng.range(0.06f, 0.3f) / px, ang = rng() * 6.2831853f;
            const float rad = rng.range(0.0015f, 0.004f) / px;
            const V3 col = srgb(88, 74, 60) * rng.range(0.8f, 1.2f);
            for (float s = 0; s < len; s += 0.5f) {
                const float bx = cx + std::cos(ang) * s + 3.f * std::sin(s * 0.05f), by = cy + std::sin(ang) * s;
                const int ir = static_cast<int>(rad) + 1;
                for (int dy = -ir; dy <= ir; ++dy) {
                    for (int dx = -ir; dx <= ir; ++dx) {
                        const float d = std::hypot(static_cast<float>(dx), static_cast<float>(dy));
                        if (d > rad) continue;
                        const size_t c = m.at(wrapi(static_cast<int>(bx) + dx, N), wrapi(static_cast<int>(by) + dy, N));
                        m.height[c] = std::max(m.height[c], 0.006f + rad * px * std::sqrt(1.f - (d / rad) * (d / rad)));
                        m.albedo[c] = col;
                    }
                }
            }
        }
        m.bakeCavity(6, 0.003f, 0.35f);
        return m.finish(0.7f);
    }

    // ---- bark -----------------------------------------------------------------------------------------------------
    // Furrowed bark of an old hardwood: long plates between deep furrows, lichen and the green of algae on the plates.
    // 0.4 m round by 0.8 m along (u round the stem, v along it).
    inline TexSet barkTexture(uint32_t seed) {
        const int W = 512, H = 1024;
        const float Wm = 0.4f, Hm = 0.8f;
        Maps m(W, H, Wm, Hm);
        const V3 plate = srgb(116, 108, 98), furrow = srgb(44, 38, 32), lichen = srgb(150, 160, 138), algae = srgb(92, 108, 62);
        forRows(H, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(H);
            for (int x = 0; x < W; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(W);
                const size_t c = m.at(x, y);
                const float warp = 0.25f * fbm2(u * 3, v * 2, 3, seed, 3, 2);
                const Cell cl = worley2(u * 9 + warp, v * 4, seed + 1, 9, 4, 0.9f);
                const float edge = cl.f2 - cl.f1;
                const float ridge = smooth(0.f, 0.35f, edge);
                const float fine = fbm2(u * 40, v * 20, 4, seed + 2, 40, 20);
                float hgt = 0.014f * ridge + 0.0015f * fine;
                V3 col = mix(furrow, plate * (0.85f + 0.3f * unit(cl.id)), smooth(0.f, 0.6f, ridge));
                const float lichenHere = smooth(0.2f, 0.6f, fbm2(u * 6, v * 3, 3, seed + 3, 6, 3)) * ridge;
                col = mix(col, lichen, 0.55f * lichenHere);
                const float algaeHere = smooth(0.0f, 0.7f, fbm2(u * 2, v * 1, 3, seed + 4, 2, 1)) * ridge;
                col = mix(col, algae, 0.35f * algaeHere);
                col = col * (0.9f + 0.2f * hash2f(x, y, seed + 5));
                m.albedo[c] = col;
                m.height[c] = hgt;
                m.rough[c] = 0.92f;
            }
        });
        m.bakeCavity(8, 0.006f, 0.3f);
        return m.finish(0.6f);
    }

    // ---- rusted iron ------------------------------------------------------------------------------------------------
    // The wheel's iron: red oxide paint gone to rust, flaking, pitted. 0.6 m a tile.
    inline TexSet rustTexture(uint32_t seed) {
        const int N = 512;
        const float S = 0.6f;
        Maps m(N, N, S, S);
        const V3 rust = srgb(104, 56, 36), orange = srgb(136, 76, 44), deep = srgb(58, 34, 26), paint = srgb(110, 42, 32);
        forRows(N, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N);
                const size_t c = m.at(x, y);
                const float n = fbm2(u * 4, v * 4, 5, seed, 4, 4);
                const float paintHere = smooth(0.05f, 0.15f, fbm2(u * 6, v * 6, 4, seed + 1, 6, 6));
                const Cell pit = worley2(u * 60, v * 60, seed + 2, 60, 60);
                const float pitHere = smooth(0.3f, 0.1f, pit.f1) * (unit(pit.id) < 0.5f ? 1.f : 0.f) * (1.f - paintHere);
                V3 col = mix(rust, orange, clamp01(0.5f + n));
                col = mix(col, deep, 0.35f * pitHere + 0.2f * smooth(0.2f, 0.7f, -n));
                col = mix(col, paint * (0.92f + 0.16f * clamp01(0.5f + n)), 0.55f * paintHere);
                // rust streaks running down from the paint's edges
                const float run = fbm2(u * 24, v * 2, 3, seed + 3, 24, 2);
                col = mix(col, orange * 0.8f, 0.35f * smooth(0.2f, 0.6f, run) * (1.f - paintHere));
                m.albedo[c] = col * (0.92f + 0.16f * hash2f(x, y, seed + 4));
                m.height[c] = 0.0006f * paintHere + 0.0004f * n - 0.0006f * pitHere;
                m.rough[c] = mix(V3(0.92f, 0, 0), V3(0.7f, 0, 0), paintHere).r;
                m.metal[c] = 0.15f * (1.f - paintHere) * (1.f - pitHere);
            }
        });
        m.bakeCavity(4, 0.0005f, 0.5f);
        return m.finish(1.f);
    }

    // ---- leaves -----------------------------------------------------------------------------------------------------
    // A 2 x 2 atlas of summer leaves, stem at the bottom of each cell: 0 an ovate serrate leaf (beech, birch), 1 a
    // five-lobed maple, 2 a tulip poplar's square-cut leaf, 3 a long smooth rhododendron leaf. Alpha cuts the outline.
    inline float leafOutline(int kind, float x, float y, float& t) {
        // x in [-0.5, 0.5], y in [0, 1] over the cell; returns the signed distance-ish (>0 inside) in cell units
        constexpr float pi = 3.14159265f;
        if (kind == 0 || kind == 3) {
            const float y0 = 0.1f, y1 = 0.97f;
            t = (y - y0) / (y1 - y0);
            if (t < 0 || t > 1) return -1;
            float half = kind == 0 ? 0.30f * std::pow(std::sin(pi * std::pow(t, 0.72f)), 0.9f)
                                   : 0.17f * std::pow(std::sin(pi * std::pow(t, 0.9f)), 0.65f);
            if (kind == 0) half *= 1.f - 0.05f * (0.5f + 0.5f * std::sin(t * 80.f));
            return half - std::abs(x);
        }
        const float cx = 0.f, cy = kind == 1 ? 0.48f : 0.5f;
        const float dx = x - cx, dy = y - cy;
        const float r = std::hypot(dx, dy);
        float th = std::atan2(dy, dx);// pi/2 points to the tip
        t = clamp01((y - 0.08f) / 0.9f);
        float R;
        if (kind == 1) {
            // five lobes, deep sinuses, pointed teeth
            const float a = th - pi / 2;
            const float lobes = std::pow(0.5f + 0.5f * std::cos(5.f * a), 1.6f);
            R = 0.25f + 0.21f * lobes;
            R *= 1.f - 0.05f * std::pow(0.5f + 0.5f * std::cos(25.f * a), 4.f);
            if (std::cos(a) < -0.75f) R *= 0.55f;// the notch at the stem
        } else {
            // tulip poplar: four broad lobes and a flat, notched tip
            const float a = th - pi / 2;
            R = 0.36f + 0.06f * std::cos(4.f * a + pi);
            R *= 1.f - 0.28f * std::exp(-(a * a) / 0.06f);
            if (std::cos(a) < -0.8f) R *= 0.6f;
        }
        return R - r;
    }

    inline TexSet leafAtlas(uint32_t seed) {
        const int N = 1024, cell = N / 2;
        Maps m(N, N, 0.2f, 0.2f);// a leaf cell is about 10 cm
        const std::array<V3, 4> base{srgb(70, 104, 36), srgb(78, 112, 40), srgb(88, 118, 42), srgb(46, 72, 30)};
        const std::array<V3, 4> pale{srgb(124, 150, 64), srgb(130, 152, 66), srgb(140, 160, 70), srgb(90, 116, 58)};
        forRows(N, [&](int py) {
            for (int px = 0; px < N; ++px) {
                const int kind = (py / cell) * 2 + (px / cell);
                const float x = (static_cast<float>(px % cell) + 0.5f) / static_cast<float>(cell) - 0.5f;
                const float y = (static_cast<float>(py % cell) + 0.5f) / static_cast<float>(cell);
                const size_t c = m.at(px, py);
                float t = 0;
                const float inside = leafOutline(kind, x, y, t);
                const float aa = 1.5f / static_cast<float>(cell);
                float alpha = clamp01(0.5f + inside / aa);
                // the stalk below the blade
                const float stalk = (y > 0.0f && y < 0.14f && std::abs(x) < 0.008f + 0.004f * (0.14f - y) / 0.14f) ? 1.f : 0.f;
                alpha = std::max(alpha, stalk);
                // veins: the midrib and pairs of secondaries running up toward the tip
                const float mid = smooth(0.012f, 0.0f, std::abs(x));
                const float secPhase = (y - 1.3f * std::abs(x)) * (kind == 3 ? 14.f : 9.f);
                const float sec = smooth(0.08f, 0.f, std::abs(secPhase - std::round(secPhase))) * smooth(0.f, 0.03f, std::abs(x));
                const float fine = ridged2((x + 0.5f) * 30.f, y * 30.f, 2, seed + static_cast<uint32_t>(kind));
                V3 col = mix(base[kind], pale[kind], 0.55f * mid + 0.3f * sec + 0.1f * fine);
                const float blotch = fbm2(x * 6 + static_cast<float>(kind) * 3.1f, y * 6, 3, seed + 10);
                col = col * (0.85f + 0.3f * clamp01(0.5f + blotch));
                // a little yellowing and browning at the margins of some
                const float margin = smooth(0.05f, 0.f, inside);
                col = mix(col, srgb(120, 110, 50), 0.35f * margin * clamp01(blotch + 0.3f));
                const Cell spot = worley2((x + 0.5f) * 14, y * 14, seed + 20 + static_cast<uint32_t>(kind));
                col = mix(col, srgb(70, 60, 30), 0.6f * smooth(0.12f, 0.06f, spot.f1) * (unit(spot.id) < 0.15f ? 1.f : 0.f));
                if (stalk > 0 && inside < 0) col = srgb(96, 104, 50);
                m.albedo[c] = alpha > 0 ? col : base[kind];
                m.alpha[c] = alpha;
                // a cupped blade with a sunken midrib and raised secondaries
                m.height[c] = -0.0006f * x * x * 4.f - 0.00012f * mid + 0.00005f * sec + 0.00002f * fine;
                m.rough[c] = kind == 3 ? 0.42f : 0.58f;
            }
        });
        return m.finish(1.f, false, false);
    }

    // ---- window glass ----------------------------------------------------------------------------------------------
    // old glass under decades of grime: mostly clear in the middle, dirtier toward the bottom and the corners, with
    // streaks where rain ran down. Alpha is the dirt.
    inline TexSet glassTexture(uint32_t seed) {
        const int N = 256;
        Maps m(N, N, 0.3f, 0.3f);
        forRows(N, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N);
                const size_t c = m.at(x, y);
                const float edge = std::min(std::min(u, 1.f - u), std::min(v, 1.f - v));
                const float streak = fbm2(u * 20, v * 1.5f, 3, seed, 20, 2);
                float dirt = 0.18f + 0.45f * smooth(0.12f, 0.f, edge) + 0.35f * smooth(0.45f, 0.f, v) * smooth(-0.2f, 0.4f, streak);
                dirt += 0.12f * fbm2(u * 6, v * 6, 3, seed + 1, 6, 6);
                m.albedo[c] = mix(srgb(150, 146, 132), srgb(96, 88, 70), smooth(0.3f, 0.9f, dirt));
                m.alpha[c] = clamp01(dirt);
                m.height[c] = 0.0003f * fbm2(u * 3, v * 3, 2, seed + 2, 3, 3);// old glass is wavy
                m.rough[c] = 0.1f + 0.7f * clamp01(dirt);
            }
        });
        return m.finish(1.f, true, false);
    }

    // the ripple map: a tileable field of small ripples (RGB, a tangent-space normal over world XZ) and a foam pattern
    // carried with them (A: a lace of veins over soft blobs; the simulation's foam decides how much of it is white)
    inline std::shared_ptr<threepp::DataTexture> waterRipples(int size = 256) {
        using namespace threepp;
        struct Wave {
            float m, n, a, phase;
        };
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> u(0, 1);
        std::vector<Wave> ripples, lace;
        while (ripples.size() < 48) {
            const float m = std::round(u(rng) * 32 - 16), n = std::round(u(rng) * 32 - 16), k = std::hypot(m, n);
            if (k < 3 || k > 16) continue;
            ripples.push_back({m, n, std::pow(k, -1.4f), 6.283f * u(rng)});
        }
        while (lace.size() < 10) {
            const float m = std::round(u(rng) * 14 - 7), n = std::round(u(rng) * 14 - 7), k = std::hypot(m, n);
            if (k < 2 || k > 7) continue;
            lace.push_back({m, n, 1, 6.283f * u(rng)});
        }
        std::vector<float> gx(size * size), gy(size * size), veins(size * size);
        float maxSlope = 0, laceNorm = 0;
        for (const auto& w : lace) laceNorm += w.a;
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                const float s = static_cast<float>(x) / static_cast<float>(size), t = static_cast<float>(y) / static_cast<float>(size);
                float dx = 0, dy = 0, field = 0;
                for (const auto& w : ripples) {
                    const float arg = 6.283f * (w.m * s + w.n * t) + w.phase;
                    dx -= w.a * 6.283f * w.m * std::sin(arg);
                    dy -= w.a * 6.283f * w.n * std::sin(arg);
                }
                for (const auto& w : lace) field += w.a * std::cos(6.283f * (w.m * s + w.n * t) + w.phase);
                const int c = y * size + x;
                gx[c] = dx;
                gy[c] = dy;
                veins[c] = field / laceNorm;
                maxSlope = std::max(maxSlope, std::hypot(dx, dy));
            }
        }
        std::vector<unsigned char> pixels(size * size * 4);
        const float scale = 0.55f / maxSlope;
        for (int c = 0; c < size * size; ++c) {
            const float nx = -gx[c] * scale, ny = -gy[c] * scale;
            const float nz = std::sqrt(std::max(0.05f, 1 - nx * nx - ny * ny));
            const float foam = std::clamp(0.8f * std::exp(-std::abs(veins[c]) / 0.09f) + 0.25f * (0.5f + 0.5f * std::sin(9.f * veins[c])), 0.f, 1.f);
            pixels[4 * c] = static_cast<unsigned char>(std::clamp(nx * 0.5f + 0.5f, 0.f, 1.f) * 255);
            pixels[4 * c + 1] = static_cast<unsigned char>(std::clamp(ny * 0.5f + 0.5f, 0.f, 1.f) * 255);
            pixels[4 * c + 2] = static_cast<unsigned char>(std::clamp(nz * 0.5f + 0.5f, 0.f, 1.f) * 255);
            pixels[4 * c + 3] = static_cast<unsigned char>(foam * 255);
        }
        auto texture = DataTexture::create(ImageData{std::move(pixels)}, size, size);
        texture->wrapS = texture->wrapT = TextureWrapping::Repeat;
        texture->magFilter = Filter::Linear;
        texture->minFilter = Filter::LinearMipmapLinear;
        texture->generateMipmaps = true;
        texture->needsUpdate();
        return texture;
    }

    // ---- falling water ----------------------------------------------------------------------------------------------
    // a sheet of water falling: streaks along v (the fall), threads of white air in clear water, frayed at the edges
    // of the sheet. Alpha is how much water there is. Tiles in both directions; scroll it along v to make it fall.
    inline TexSet fallingWaterTexture(uint32_t seed) {
        const int W = 256, H = 512;
        Maps m(W, H, 0.5f, 1.f);
        forRows(H, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(H);
            for (int x = 0; x < W; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(W);
                const size_t c = m.at(x, y);
                const float streak = 0.5f + 0.5f * fbm2(u * 24, v * 2, 4, seed, 24, 2);
                const float fine = 0.5f + 0.5f * fbm2(u * 90, v * 4, 3, seed + 1, 90, 4);
                const float air = smooth(0.55f, 0.85f, 0.7f * streak + 0.3f * fine);
                m.albedo[c] = mix(V3(0.5f, 0.48f, 0.42f), V3(0.94f, 0.95f, 0.96f), air);
                // the sheet's edges (u near 0 and 1) fray: ragged tongues along the fall
                const float rag = 0.1f + 0.08f * fbm2(u * 3, v * 6, 3, seed + 2, 3, 6);
                const float edge = smooth(0.f, rag, u) * smooth(1.f, 1.f - rag, u);
                m.alpha[c] = clamp01(0.3f + 0.65f * air + 0.25f * (streak - 0.5f)) * edge;
                m.height[c] = 0.002f * streak + 0.0006f * fine;
                m.rough[c] = 0.15f + 0.5f * air;
            }
        });
        return m.finish(1.f, true, false);
    }

    // ---- foam lace ----------------------------------------------------------------------------------------------------
    // white water seen close: clusters of bubbles and the veins between them, holes of dark water through it. Alpha is
    // the foam (for an alpha-tested skin). 1.2 m a tile.
    inline std::shared_ptr<threepp::DataTexture> foamLaceTexture(uint32_t seed) {
        const int N = 512;
        std::vector<unsigned char> px(static_cast<size_t>(N) * N * 4);
        forRows(N, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(N);
            for (int x = 0; x < N; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(N);
                const Cell big = worley2(u * 9, v * 9, seed, 9, 9);
                const Cell small = worley2(u * 34, v * 34, seed + 1, 34, 34);
                const float veins = smooth(0.16f, 0.02f, big.f2 - big.f1);
                const float bubbles = smooth(0.42f, 0.18f, small.f1) * smooth(-0.3f, 0.3f, fbm2(u * 5, v * 5, 3, seed + 2, 5, 5));
                const float body = smooth(-0.15f, 0.45f, fbm2(u * 3, v * 3, 4, seed + 3, 3, 3));
                const float a = clamp01(std::max({veins * 0.9f, bubbles, body}) + 0.1f * fbm2(u * 60, v * 60, 2, seed + 4, 60, 60));
                const size_t i = (static_cast<size_t>(y) * N + x) * 4;
                const unsigned char w = toSrgbByte(0.82f + 0.12f * a);
                px[i] = px[i + 1] = px[i + 2] = w;
                px[i + 3] = toByte(a);
            }
        });
        return makeTexture(std::move(px), N, N, true);
    }

}// namespace mill

#endif
