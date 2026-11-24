// Surfaces only the mill's inside wears, painted per pixel like the rest (TextureLab.hpp):
//   - the French burr millstone's face: a disc of quartz pieces set in plaster, full of small cells, dressed in ten
//     harps of furrows (a master furrow and three shorter ones parallel to it, each harp ending at the next master),
//     the lands between worn smooth and the furrows still holding old meal; and the same stone undressed, for the
//     skirts, the backs and the beds;
//   - sacking (a coarse plain weave of jute, floury in the weave);
//   - old cobwebs (a few radial threads and a broken spiral, alpha only where thread is);
//   - leaves blown in long ago: the outlines of the woods' leaves, dry and brown.

#ifndef THREEPP_EXAMPLE_WATERMILL_INTERIORTEXTURES_HPP
#define THREEPP_EXAMPLE_WATERMILL_INTERIORTEXTURES_HPP

#include "Textures.hpp"

namespace mill {

    // the burr stone's pieces: three rings of quartz blocks round the eye; returns the piece's id and the distance
    // (m) to the nearest joint. x, z in metres from the stone's centre.
    inline float burrJoint(float x, float z, float radius, uint32_t seed, uint32_t& piece) {
        const float r = std::hypot(x, z);
        float a = std::atan2(z, x);
        if (a < 0) a += 6.2831853f;
        // ring boundaries wander a little
        const float wob = 0.012f * fbm2(a * 3.f, 0.5f, 2, seed + 1, 0, 0);
        const std::array<float, 3> rb{0.36f * radius + wob, 0.68f * radius - wob, 2.f * radius};
        const std::array<int, 3> count{6, 9, 13};
        int ring = 0;
        while (ring < 2 && r > rb[ring]) ++ring;
        const float rIn = ring == 0 ? 0.f : rb[ring - 1];
        const float rOut = ring == 2 ? 10.f : rb[ring];
        float dRing = std::min(r - rIn, rOut - r);
        if (ring == 0) dRing = rOut - r;
        const int n = count[ring];
        const float rot = unit(hash32(seed + 7u * static_cast<uint32_t>(ring))) * 6.2831853f / static_cast<float>(n);
        // the joints between pieces: radial-ish, each bent a little by the noise
        const float bend = 0.06f * fbm2(r * 6.f, static_cast<float>(ring) * 3.f, 2, seed + 3, 0, 0);
        const float s = (a - rot + bend) / (6.2831853f / static_cast<float>(n));
        const float fs = s - std::floor(s);
        const int k = static_cast<int>(std::floor(s)) % n;
        const float dRad = std::min(fs, 1.f - fs) * (6.2831853f / static_cast<float>(n)) * std::max(r, 0.05f);
        piece = hash32(seed * 131u + static_cast<uint32_t>(ring * 97 + (k + n) % n));
        return std::min(dRing, dRad);
    }

    // `dressed`: the furrowed grinding face; otherwise the stone's plain skirt and back. The tile covers the disc's
    // square, 2 * radius across, centred.
    inline TexSet burrStoneTexture(uint32_t seed, float radius, bool dressed) {
        const int N = 1024;
        const float S = 2.f * radius;
        Maps m(N, N, S, S);
        constexpr int harps = 10;
        const float draught = 0.2f * radius;// the master furrows run tangent to this circle
        const float furrowW = 0.034f, spacing = 0.095f;
        const std::array<V3, 5> quartz{srgb(196, 186, 166), srgb(176, 170, 158), srgb(204, 192, 172), srgb(186, 168, 150), srgb(160, 156, 148)};
        const V3 plaster = srgb(210, 204, 190), meal = srgb(222, 214, 196);
        forRows(N, [&](int py) {
            const float z = ((static_cast<float>(py) + 0.5f) / static_cast<float>(N) - 0.5f) * S;
            for (int px = 0; px < N; ++px) {
                const float x = ((static_cast<float>(px) + 0.5f) / static_cast<float>(N) - 0.5f) * S;
                const size_t c = m.at(px, py);
                uint32_t piece = 0;
                const float joint = burrJoint(x, z, radius, seed, piece);
                const float jointLine = smooth(0.006f, 0.0025f, joint);
                // the quartz: a cellular stone, small cavities in clusters, a fine crystalline grain
                const float u = x / S + 0.5f, v = z / S + 0.5f;
                // cavities: a few larger ones and many small, gathered in clouds, each its own size
                const float cloud = smooth(-0.1f, 0.6f, fbm2(u * 9.f + static_cast<float>(piece % 7u), v * 9.f, 3, seed + 10, 0, 0));
                const float density = (0.03f + 0.22f * cloud) * (0.5f + unit(hash32(piece + 3u)));
                const Cell big = worley2(u * 70.f, v * 70.f, seed + 11 + piece % 5u, 0, 0);
                const Cell small = worley2(u * 210.f, v * 210.f, seed + 17 + piece % 3u, 0, 0);
                const float bigR = 0.12f + 0.25f * unit(hash32(big.id + 1u)), smallR = 0.15f + 0.25f * unit(hash32(small.id + 1u));
                const float cavity = std::max((unit(big.id) < 0.5f * density ? 1.f : 0.f) * smooth(bigR, bigR * 0.5f, big.f1 + 0.06f * perlin2(u * 400.f, v * 400.f, seed + 18, 0, 0)),
                                              0.8f * (unit(small.id) < density ? 1.f : 0.f) * smooth(smallR, smallR * 0.4f, small.f1));
                const float grain = fbm2(u * 90.f, v * 90.f, 3, seed + 12, 0, 0);
                const float blotch = fbm2(u * 7.f + static_cast<float>(piece % 13u), v * 7.f, 3, seed + 13, 0, 0);
                V3 col = quartz[piece % quartz.size()] * (0.9f + 0.12f * unit(hash32(piece)));
                col = col * (0.9f + 0.1f * grain + 0.08f * blotch);
                col = mix(col, srgb(112, 102, 88), 0.45f * cavity);
                float hgt = 0.0015f * grain - 0.0025f * cavity + 0.002f * blotch;
                float rough = 0.78f - 0.1f * smooth(0.2f, 0.7f, grain);
                // the plaster in the joints, set back from the quartz
                col = mix(col, plaster * 0.9f, jointLine);
                hgt -= 0.003f * jointLine;
                rough = std::max(rough, 0.9f * jointLine);
                if (dressed) {
                    const float r = std::hypot(x, z);
                    float a = std::atan2(z, x);
                    // which harp: the master line k is tangent to the draught circle at angle a_k and runs outward
                    // counter-clockwise; at radius r it passes angle a_k + acos(draught / r)
                    const float rr = std::max(r, draught * 1.001f);
                    float ap = a - std::acos(draught / rr);
                    const float sector = 6.2831853f / harps;
                    ap -= std::floor(ap / 6.2831853f) * 6.2831853f;
                    const int k = static_cast<int>(ap / sector) % harps;
                    const float ak = sector * static_cast<float>(k);
                    // distance into the harp from its master line
                    const float s = draught - (x * std::cos(ak) + z * std::sin(ak));
                    float furrow = 0.f, profile = 0.f;
                    for (int f = 0; f < 4; ++f) {
                        const float d = s - spacing * static_cast<float>(f);
                        if (d < 0.f || d > furrowW) continue;
                        // the back of the furrow is steep, the feather edge rises gently to the land
                        const float t = d / furrowW;
                        furrow = 1.f;
                        profile = t < 0.25f ? smooth(0.f, 0.25f, t) : 1.f - smooth(0.25f, 1.f, t);
                    }
                    // the furrows fade to nothing in the skirt's last few cm and deepen into the bosom near the eye
                    const float depth = 0.008f * (0.7f + 0.6f * smooth(0.5f * radius, 0.15f, r));
                    hgt -= depth * profile;
                    // old meal caked in the furrows, the lands worn glassy, fine cracking (stitching) on the lands
                    const float caked = furrow * smooth(-0.2f, 0.5f, fbm2(u * 30.f, v * 30.f, 3, seed + 14, 0, 0));
                    col = mix(col, col * 0.45f, furrow * profile);
                    col = mix(col, meal * 0.85f, 0.3f * caked * profile);
                    col = col * (1.f + 0.05f * (1.f - furrow));
                    const float stitchPhase = s / 0.0035f;
                    const float stitch = (1.f - furrow) * smooth(0.25f, 0.f, std::abs(stitchPhase - std::round(stitchPhase))) *
                                         smooth(0.f, 0.4f, fbm2(u * 12.f, v * 12.f, 2, seed + 15, 0, 0));
                    hgt -= 0.0004f * stitch;
                    rough -= 0.12f * (1.f - furrow) * smooth(0.35f * radius, 0.9f * radius, r);
                    // meal dust over everything, thickest toward the eye
                    col = mix(col, meal, 0.18f * smooth(0.6f * radius, 0.1f, r));
                }
                m.albedo[c] = col * (0.94f + 0.1f * hash2f(px, py, seed + 16));
                m.height[c] = hgt;
                m.rough[c] = std::clamp(rough, 0.35f, 0.95f);
            }
        });
        m.bakeCavity(4, 0.002f, 0.55f, false);
        return m.finish(1.f, false);
    }

    // jute sacking: a plain weave of fat, hairy threads, about 3 mm apart. 0.2 m a tile.
    inline TexSet sackingTexture(uint32_t seed) {
        const int N = 512;
        const float S = 0.2f;
        Maps m(N, N, S, S);
        const int threads = 64;
        const V3 jute = srgb(150, 122, 86), dark = srgb(96, 76, 52), flour = srgb(210, 204, 190);
        forRows(N, [&](int py) {
            const float v = (static_cast<float>(py) + 0.5f) / static_cast<float>(N);
            for (int px = 0; px < N; ++px) {
                const float u = (static_cast<float>(px) + 0.5f) / static_cast<float>(N);
                const size_t c = m.at(px, py);
                // each thread wanders a little
                const float wu = u * threads + 0.25f * perlin2(v * 8.f, u * 64.f, seed, 8, 64);
                const float wv = v * threads + 0.25f * perlin2(u * 8.f, v * 64.f, seed + 1, 8, 64);
                const int iu = static_cast<int>(std::floor(wu)), iv = static_cast<int>(std::floor(wv));
                const float fu = wu - static_cast<float>(iu), fv = wv - static_cast<float>(iv);
                const float warp = std::sin(3.14159265f * fu), weft = std::sin(3.14159265f * fv);
                // over and under
                const bool warpOver = ((iu + iv) & 1) == 0;
                const float warpH = warp * (0.6f + 0.4f * (warpOver ? std::sin(3.14159265f * fv) : 1.f - std::sin(3.14159265f * fv)));
                const float weftH = weft * (0.6f + 0.4f * (!warpOver ? std::sin(3.14159265f * fu) : 1.f - std::sin(3.14159265f * fu)));
                const float h = std::max(warpH * (warpOver ? 1.f : 0.75f), weftH * (warpOver ? 0.75f : 1.f));
                const float hair = fbm2(u * 128.f, v * 128.f, 2, seed + 2, 128, 128);
                V3 col = mix(dark, jute, smooth(0.1f, 0.9f, h) * (0.85f + 0.15f * hair));
                col = col * (0.88f + 0.24f * hash2f(iu & (threads - 1), iv & (threads - 1), seed + 3));
                // meal worked into the weave, patchy
                const float dust = smooth(0.f, 0.6f, fbm2(u * 4.f, v * 4.f, 3, seed + 4, 4, 4));
                col = mix(col, flour, 0.35f * dust * (1.f - 0.6f * h));
                m.albedo[c] = col;
                m.height[c] = 0.0012f * h + 0.0002f * hair;
                m.rough[c] = 0.95f;
            }
        });
        m.bakeCavity(3, 0.0006f, 0.6f);
        return m.finish(1.f);
    }

    // an old cobweb: radial threads from a corner and a broken, dust-laden spiral; alpha is where thread is.
    // The corner is the texture's (0, 0); the web fans out over a quarter disc.
    inline TexSet cobwebTexture(uint32_t seed) {
        const int N = 512;
        Maps m(N, N, 0.5f, 0.5f);
        Rng rng(seed);
        std::array<float, 9> spokes{};
        for (size_t k = 0; k < spokes.size(); ++k) spokes[k] = (static_cast<float>(k) + rng.range(0.15f, 0.85f)) / static_cast<float>(spokes.size()) * 1.5707963f;
        forRows(N, [&](int py) {
            const float y = (static_cast<float>(py) + 0.5f) / static_cast<float>(N);
            for (int px = 0; px < N; ++px) {
                const float x = (static_cast<float>(px) + 0.5f) / static_cast<float>(N);
                const size_t c = m.at(px, py);
                const float r = std::hypot(x, y);
                const float a = std::atan2(y, x);
                const float px1 = 1.f / static_cast<float>(N);
                float thread = 0.f;
                // radial threads, sagging a little
                for (float s : spokes) {
                    const float d = std::abs(a - s - 0.03f * std::sin(r * 9.f)) * r;
                    thread = std::max(thread, smooth(1.6f * px1, 0.3f * px1, d) * smooth(0.98f, 0.8f, r));
                }
                // the spiral: rings that sag between the spokes, broken in places
                const float ring = (r + 0.012f * std::sin(a * 18.f)) / 0.045f;
                const float dr = std::abs(ring - std::round(ring)) * 0.045f;
                const float broken = smooth(-0.1f, 0.2f, fbm2(x * 6.f, y * 6.f, 2, seed + 1, 0, 0));
                thread = std::max(thread, smooth(1.4f * px1, 0.2f * px1, dr) * broken * smooth(0.95f, 0.75f, r) * smooth(0.05f, 0.12f, r));
                // dust caught in it, in tatters
                const float tatter = smooth(0.35f, 0.75f, fbm2(x * 9.f, y * 9.f, 3, seed + 2, 0, 0)) * smooth(0.9f, 0.3f, r);
                const float alpha = clamp01(0.85f * thread + 0.07f * tatter);
                m.albedo[c] = mix(srgb(200, 196, 186), srgb(150, 144, 132), tatter);
                m.alpha[c] = alpha;
                m.rough[c] = 0.9f;
            }
        });
        return m.finish(0.f, false, false);
    }

    // leaves blown in through a broken pane years ago: dry, brown and curled (the woods' outlines, 2 x 2 atlas)
    inline TexSet deadLeafAtlas(uint32_t seed) {
        const int N = 512, cell = N / 2;
        Maps m(N, N, 0.2f, 0.2f);
        const std::array<V3, 4> base{srgb(118, 86, 52), srgb(128, 84, 46), srgb(110, 90, 60), srgb(84, 66, 46)};
        forRows(N, [&](int py) {
            for (int px = 0; px < N; ++px) {
                const int kind = (py / cell) * 2 + (px / cell);
                const float x = (static_cast<float>(px % cell) + 0.5f) / static_cast<float>(cell) - 0.5f;
                const float y = (static_cast<float>(py % cell) + 0.5f) / static_cast<float>(cell);
                const size_t c = m.at(px, py);
                float t = 0;
                const float inside = leafOutline(kind, x, y, t);
                // ragged, nibbled edges on dry leaves
                const float nibble = 0.03f * smooth(0.1f, 0.6f, fbm2(x * 14.f + static_cast<float>(kind), y * 14.f, 2, seed + 1, 0, 0));
                float alpha = clamp01(0.5f + (inside - nibble) / (1.5f / static_cast<float>(cell)));
                const float stalk = (y > 0.0f && y < 0.14f && std::abs(x) < 0.008f) ? 1.f : 0.f;
                alpha = std::max(alpha, stalk);
                const float mid = smooth(0.012f, 0.0f, std::abs(x));
                const float blotch = fbm2(x * 7.f + static_cast<float>(kind) * 2.3f, y * 7.f, 3, seed + 2, 0, 0);
                V3 col = base[kind] * (0.75f + 0.4f * clamp01(0.5f + blotch));
                col = mix(col, srgb(150, 120, 80), 0.3f * mid);
                col = mix(col, srgb(60, 46, 34), 0.4f * smooth(0.04f, 0.f, inside));
                m.albedo[c] = col;
                m.alpha[c] = alpha;
                m.height[c] = -0.0008f * x * x * 4.f - 0.0001f * mid + 0.0001f * blotch;
                m.rough[c] = 0.85f;
            }
        });
        return m.finish(1.f, false, false);
    }

}// namespace mill

#endif
