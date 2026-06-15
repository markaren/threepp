// Procedural textures for the vegetation system.
//
// Everything here is generated in-engine as DataTextures — no external image
// files.  Two generators are provided:
//
//   makeLeafClusterTexture()  — an RGBA alpha-cutout atlas of a small cluster
//                               of leaves, for use as `map` on leaf cards
//                               (sampled with alphaTest to punch out the leaf
//                               silhouette).
//   makeBarkTextures()        — a tiling bark albedo + matching tangent-space
//                               normal map for the trunk/branch tubes.
//
// All output is deterministic for a given seed.

#ifndef THREEPP_EXTRAS_VEGETATION_TREETEXTURES_HPP
#define THREEPP_EXTRAS_VEGETATION_TREETEXTURES_HPP

#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>

namespace threepp::vegetation {

    namespace detail {

        inline unsigned char toByte(float v) {
            return static_cast<unsigned char>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
        }

        // Cheap hash → [0,1).
        inline float hash2(int x, int y, unsigned int seed) {
            uint32_t n = static_cast<uint32_t>(x) * 374761393u +
                         static_cast<uint32_t>(y) * 668265263u + seed * 362437u;
            n = (n ^ (n >> 13)) * 1274126177u;
            n = n ^ (n >> 16);
            return static_cast<float>(n & 0xffffffu) / static_cast<float>(0xffffff);
        }

        inline float smooth(float t) { return t * t * (3.f - 2.f * t); }

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

    }// namespace detail

    // ── Leaf-cluster alpha cutout ────────────────────────────────────────
    //
    // Draws several overlapping leaf silhouettes on a transparent background.
    // `baseColor` is the mid-tone leaf albedo (sRGB 0..1); individual leaves
    // are tinted around it for variety.  Use on a material as:
    //   mat.map = tex;  mat.alphaTest = 0.5f;  mat.side = Side::Double;
    inline std::shared_ptr<DataTexture> makeLeafClusterTexture(
            unsigned int size = 256,
            unsigned int seed = 1337,
            const std::array<float, 3>& baseColor = {0.26f, 0.45f, 0.14f}) {

        auto tex = DataTexture::create(4, size, size);
        auto& px = tex->image().data<unsigned char>();// zero-initialised → transparent

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> u01(0.f, 1.f);

        struct Leaf {
            float cx, cy;     // centre (px)
            float ca, sa;     // axis direction
            float length;     // px
            float halfWidth;  // px
            float tint;       // brightness multiplier
            float hue;        // -1 yellower .. +1 bluer
        };

        const auto S = static_cast<float>(size);
        const int leafCount = 16;
        std::vector<Leaf> leaves;
        leaves.reserve(static_cast<size_t>(leafCount));
        for (int i = 0; i < leafCount; ++i) {
            Leaf lf;
            // Cluster the centres loosely around the middle.
            lf.cx = S * (0.5f + (u01(rng) - 0.5f) * 0.7f);
            lf.cy = S * (0.5f + (u01(rng) - 0.5f) * 0.7f);
            const float ang = u01(rng) * 6.28318530718f;
            lf.ca = std::cos(ang);
            lf.sa = std::sin(ang);
            lf.length = S * (0.28f + u01(rng) * 0.18f);
            lf.halfWidth = lf.length * (0.20f + u01(rng) * 0.12f);
            lf.tint = 0.75f + u01(rng) * 0.45f;
            lf.hue = (u01(rng) - 0.5f) * 2.f;
            leaves.push_back(lf);
        }

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                float bestCov = 0.f;
                int bestLeaf = -1;
                float bestShade = 1.f, bestVein = 0.f;

                const float fx = static_cast<float>(x) + 0.5f;
                const float fy = static_cast<float>(y) + 0.5f;

                for (int i = 0; i < leafCount; ++i) {
                    const Leaf& lf = leaves[static_cast<size_t>(i)];
                    const float dx = fx - lf.cx;
                    const float dy = fy - lf.cy;
                    // Project into leaf-local (u along axis 0..length, v perpendicular).
                    const float u = dx * lf.ca + dy * lf.sa;
                    const float v = -dx * lf.sa + dy * lf.ca;
                    if (u < 0.f || u > lf.length) continue;
                    const float s = u / lf.length;
                    // Teardrop profile: pointed at tip and petiole, widest mid.
                    const float prof = std::pow(std::sin(s * 3.14159265f), 0.7f);
                    const float hw = lf.halfWidth * prof;
                    if (hw < 0.5f) continue;
                    const float av = std::abs(v);
                    if (av > hw) continue;

                    // 1px antialiased edge.
                    const float cov = std::clamp(hw - av, 0.f, 1.f);
                    // Later leaves paint over earlier ones (painter's order).
                    if (i >= bestLeaf) {
                        bestLeaf = i;
                        bestCov = cov;
                        // Shade: darker toward the edge, slightly darker at base.
                        const float edge = av / hw;            // 0 centre .. 1 edge
                        bestShade = (1.f - 0.35f * edge * edge) * (0.8f + 0.2f * s);
                        // Midrib vein + lateral veins.
                        const float vein = std::exp(-av * av * 0.5f);// central rib
                        bestVein = vein;
                    }
                }

                if (bestLeaf < 0) continue;
                const Leaf& lf = leaves[static_cast<size_t>(bestLeaf)];

                float r = baseColor[0] * lf.tint;
                float g = baseColor[1] * lf.tint;
                float b = baseColor[2] * lf.tint;
                // Hue shift: yellower (more r/g) vs cooler.
                r += lf.hue < 0 ? -lf.hue * 0.10f : 0.f;
                g += lf.hue < 0 ? -lf.hue * 0.06f : -lf.hue * 0.02f;
                b += lf.hue > 0 ? lf.hue * 0.05f : 0.f;

                float shade = bestShade;
                r *= shade;
                g *= shade;
                b *= shade;
                // Vein slightly lighter/yellower along the rib.
                const float veinAmt = bestVein * 0.25f;
                r += veinAmt * 0.10f;
                g += veinAmt * 0.12f;

                const size_t idx = (static_cast<size_t>(y) * size + x) * 4;
                px[idx + 0] = detail::toByte(r);
                px[idx + 1] = detail::toByte(g);
                px[idx + 2] = detail::toByte(b);
                px[idx + 3] = detail::toByte(bestCov);
            }
        }

        tex->colorSpace = ColorSpace::sRGB;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::LinearMipmapLinear;
        tex->generateMipmaps = true;
        tex->wrapS = TextureWrapping::ClampToEdge;
        tex->wrapT = TextureWrapping::ClampToEdge;
        tex->needsUpdate();
        return tex;
    }

    // ── Bark albedo + normal map (tiling) ────────────────────────────────
    //
    // Vertically-furrowed bark.  Returns {albedo, normal}.  Both tile
    // seamlessly; set the same `repeat` on each when assigning to a material.
    inline std::pair<std::shared_ptr<DataTexture>, std::shared_ptr<DataTexture>>
    makeBarkTextures(unsigned int size = 256,
                     unsigned int seed = 1337,
                     const std::array<float, 3>& baseColor = {0.34f, 0.24f, 0.16f}) {

        auto albedo = DataTexture::create(4, size, size);
        auto normal = DataTexture::create(4, size, size);
        auto& ca = albedo->image().data<unsigned char>();
        auto& cn = normal->image().data<unsigned char>();

        const auto S = static_cast<float>(size);
        const int period = 8;// noise lattice period (in tiles) → seamless wrap

        // Height field: vertical furrows (stretched in y) + fbm detail.
        auto heightAt = [&](float u, float v) -> float {
            // u,v in [0,1).  Furrows run vertically → high horizontal freq,
            // low vertical freq.
            float furrow = detail::valueNoise(u * period, v * (period / 4.f),
                                              period, seed);
            // Ridged: sharpen into furrows.
            furrow = 1.f - std::abs(2.f * furrow - 1.f);
            float detailN =
                    detail::valueNoise(u * period * 3.f, v * period * 3.f,
                                       period * 3, seed + 11u) *
                            0.5f +
                    detail::valueNoise(u * period * 6.f, v * period * 6.f,
                                       period * 6, seed + 29u) *
                            0.25f;
            return std::clamp(furrow * 0.7f + detailN * 0.5f, 0.f, 1.f);
        };

        const float texel = 1.f / S;
        const float bumpScale = 2.5f;

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = static_cast<float>(x) / S;
                const float v = static_cast<float>(y) / S;
                const float h = heightAt(u, v);

                // Albedo: darker in furrows (low h), lighter on ridges.
                const float t = 0.45f + h * 0.55f;
                float r = baseColor[0] * t;
                float g = baseColor[1] * t;
                float b = baseColor[2] * t;
                // A little grey de-saturation on the ridges.
                const float grey = (r + g + b) / 3.f;
                const float desat = h * 0.25f;
                r = r * (1.f - desat) + grey * desat;
                g = g * (1.f - desat) + grey * desat;
                b = b * (1.f - desat) + grey * desat;

                const size_t idx = (static_cast<size_t>(y) * size + x) * 4;
                ca[idx + 0] = detail::toByte(r);
                ca[idx + 1] = detail::toByte(g);
                ca[idx + 2] = detail::toByte(b);
                ca[idx + 3] = 255;

                // Normal from height gradient (central difference, wrapping).
                auto wrap01 = [](float a) { return a - std::floor(a); };
                const float hL = heightAt(wrap01(u - texel), v);
                const float hR = heightAt(wrap01(u + texel), v);
                const float hD = heightAt(u, wrap01(v - texel));
                const float hU = heightAt(u, wrap01(v + texel));
                float nx = (hL - hR) * bumpScale;
                float ny = (hD - hU) * bumpScale;
                float nz = 1.f;
                const float inv = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz);
                nx *= inv;
                ny *= inv;
                nz *= inv;
                cn[idx + 0] = detail::toByte(nx * 0.5f + 0.5f);
                cn[idx + 1] = detail::toByte(ny * 0.5f + 0.5f);
                cn[idx + 2] = detail::toByte(nz * 0.5f + 0.5f);
                cn[idx + 3] = 255;
            }
        }

        for (auto* tex : {&albedo, &normal}) {
            (*tex)->magFilter = Filter::Linear;
            (*tex)->minFilter = Filter::LinearMipmapLinear;
            (*tex)->generateMipmaps = true;
            (*tex)->wrapS = TextureWrapping::Repeat;
            (*tex)->wrapT = TextureWrapping::Repeat;
            (*tex)->needsUpdate();
        }
        albedo->colorSpace = ColorSpace::sRGB;
        normal->colorSpace = ColorSpace::NoColorSpace;

        return {albedo, normal};
    }

}// namespace threepp::vegetation

#endif// THREEPP_EXTRAS_VEGETATION_TREETEXTURES_HPP
