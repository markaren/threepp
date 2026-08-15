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

#include "threepp/extras/core/NoiseUtils.hpp"
#include "threepp/extras/core/TextureBake.hpp"
#include "threepp/math/Rng.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace threepp::vegetation {

    namespace detail {

        // These lived here first and were re-extracted verbatim into the shared
        // threepp::noise helpers (extras/core/NoiseUtils.hpp) — same bits, so
        // every generated texture is unchanged.
        using noise::hash2;
        using noise::smooth;
        using noise::toByte;

        // sin(PI*x)^e — the standard "widest in the middle, zero at both ends"
        // blade/taper profile.
        //
        // MUST go through here rather than pow(sin(PI*x), e) written inline.
        // PI in float is 3.14159274f, i.e. slightly ABOVE pi, so sin(PI*1.0f)
        // evaluates to -8.7e-8 — negative. A fractional exponent on a negative
        // base is NaN, and x lands exactly on 1.0 whenever the caller clamps
        // with min(..., 1.f). The NaN then propagates into a blade length or
        // half-width, where it is far worse than a bad pixel: every subsequent
        // `if (v > limit) continue` bounds check silently fails (all comparisons
        // against NaN are false), so the shape escapes its own extent and paints
        // NaN — which quantises to BLACK — across the rest of the tile.
        inline float lobe(float x, float e) {
            return std::pow(std::max(std::sin(3.14159265358979f * std::clamp(x, 0.f, 1.f)), 0.f), e);
        }

        // ── Alpha-edge RGB dilation ──────────────────────────────────────
        //
        // Cutout foliage is drawn onto a ZERO-initialised (transparent BLACK)
        // buffer, so every texel outside a leaf holds RGB 0. Bilinear filtering
        // and mip generation both blend colour ACROSS the alpha edge, which
        // drags that black into the outermost leaf texels — a dark fringe that
        // survives alphaTest and reads as grime around every leaf, and gets
        // worse with distance as coarser mips mix in more empty texels.
        //
        // Fix: push opaque colour outward into the transparent region before
        // upload, so the RGB channel is continuous across the silhouette and
        // there is no dark colour left to bleed in. Alpha is untouched — the
        // cutout shape is unchanged, only the colour under it.
        // `rgb` is size*size*3 colour, `filled` a parallel size*size mask that is
        // true wherever the colour is meaningful (i.e. inside the cutout). Each
        // pass grows the filled region by one texel, averaging the already-filled
        // neighbours. Alpha lives elsewhere and is never touched.
        inline void dilateRGB(std::vector<float>& rgb, std::vector<char>& filled,
                              unsigned int size, int passes) {
            const auto S = static_cast<int>(size);
            for (int p = 0; p < passes; ++p) {
                const std::vector<float> src = rgb;
                const std::vector<char> was = filled;
                bool grew = false;
                for (int y = 0; y < S; ++y) {
                    for (int x = 0; x < S; ++x) {
                        const auto px = static_cast<size_t>(y) * static_cast<size_t>(S) + static_cast<size_t>(x);
                        if (was[px]) continue;
                        float r = 0.f, g = 0.f, b = 0.f, w = 0.f;
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                const int nx = x + dx, ny = y + dy;
                                if (nx < 0 || ny < 0 || nx >= S || ny >= S) continue;
                                const auto q = static_cast<size_t>(ny) * static_cast<size_t>(S) + static_cast<size_t>(nx);
                                if (!was[q]) continue;
                                r += src[q * 3 + 0];
                                g += src[q * 3 + 1];
                                b += src[q * 3 + 2];
                                w += 1.f;
                            }
                        }
                        if (w <= 0.f) continue;
                        rgb[px * 3 + 0] = r / w;
                        rgb[px * 3 + 1] = g / w;
                        rgb[px * 3 + 2] = b / w;
                        filled[px] = 1;
                        grew = true;
                    }
                }
                if (!grew) break;
            }
        }

        using noise::valueNoise;

    }// namespace detail

    // ── Leaf blade outline, per species ──────────────────────────────────
    // ── Bark surface character, per species ──────────────────────────────
    enum class BarkStyle {
        Furrowed = 0,// deep vertical ridges — oak, most broadleaves
        Plated = 1,  // irregular scale plates split by fissures — pine, spruce
        Papery = 2,  // smooth pale bark with horizontal lenticels — birch
    };

    enum class LeafShape {
        Ovate = 0,     // broad, rounded, pointed tip — the generic broadleaf
        Lobed = 1,     // deeply scalloped margin — oak
        Serrate = 2,   // fine-toothed margin, small blade — birch
        Lanceolate = 3,// long and narrow — willow
    };

    // ── Leaf-sprig alpha cutout ──────────────────────────────────────────
    //
    // ONE foliage card = one SPRIG: a petiole running up the tile with a dozen
    // small leaflets alternating down both sides, thinning toward the tip.
    //
    // This is the difference between foliage that reads as a tree and foliage
    // that reads as a pile of green shapes. A card is ~0.5-0.8 world units
    // across; if the texture holds three or four leaf silhouettes then each
    // painted leaf is ~25cm and the canopy looks like cabbage. A real broadleaf
    // is 5-10cm, so the card has to carry 10-16 of them — the density has to
    // come from the TEXTURE, not from stacking more quads (which costs
    // triangles, overdraw and depth-fighting for the same visual result).
    //
    // The sprig is oriented with the petiole along +v so it lines up with the
    // card's growth axis (makeLeafGeometry spans v along the branch/up axis),
    // i.e. the leaflets fan out sideways exactly as they would on a real twig.
    //
    // Use on a material as:
    //   mat.map = tex;  mat.alphaTest = 0.4f;  mat.side = Side::Double;
    inline std::shared_ptr<DataTexture> makeLeafClusterTexture(
            unsigned int size = 256,
            unsigned int seed = 1337,
            const std::array<float, 3>& baseColor = {0.26f, 0.45f, 0.14f},
            LeafShape shape = LeafShape::Ovate,
            int leafletsPerTwig = 8) {

        auto tex = DataTexture::create(4, size, size);
        auto& px = tex->image().data<unsigned char>();// zero → transparent

        math::Rng rng(seed);
        auto u01 = [](math::Rng& r) { return r.nextFloat(); };

        const auto S = static_cast<float>(size);
        constexpr float PI = 3.14159265358979f;

        struct Twig {
            float bx, by;// base (uv)
            float dx, dy;// unit axis
            float length;
            float curve; // lateral bow, uv units at the tip
        };
        struct Leaflet {
            float bx, by;   // base on its twig (uv)
            float dx, dy;   // unit axis, petiole → tip
            float length;   // uv units
            float halfWidth;// uv units at the widest point
            float tint;     // brightness multiplier
            float hue;      // <0 yellower (new growth) .. >0 cooler/darker
        };

        // Species blade proportions. `lenScale` is deliberately small: a leaflet
        // must be SHORTER than the gap between successive leaflets on the same
        // side of a twig, or neighbouring blades merge into one solid lens and
        // the sprig reads as a single fern frond instead of a spray of leaves.
        // halfWidth = length * widthRatio, so widthRatio 0.35 gives a 1 : 0.7
        // blade — leaf-shaped. Much above that and the leaflets render as peas.
        float lenScale = 1.f, widthRatio = 0.35f;
        switch (shape) {
            case LeafShape::Lobed: lenScale = 1.12f; widthRatio = 0.40f; break;
            case LeafShape::Serrate: lenScale = 0.88f; widthRatio = 0.37f; break;
            case LeafShape::Lanceolate: lenScale = 1.45f; widthRatio = 0.17f; break;
            case LeafShape::Ovate: break;
        }

        // The card is a BRANCHLET, not a single sprig: a short stem near the
        // bottom that forks into a fan of twigs. One narrow sprig up the middle
        // would leave most of the tile empty (a thin card = a sparse canopy);
        // fanning several twigs fills the tile while keeping every individual
        // leaf small — which is the whole point.
        const int twigCount = 4;
        const float stemTop = 0.13f;
        std::vector<Twig> twigs;
        twigs.reserve(static_cast<size_t>(twigCount));
        for (int i = 0; i < twigCount; ++i) {
            const float f = (twigCount == 1) ? 0.5f
                                             : static_cast<float>(i) / static_cast<float>(twigCount - 1);
            const float spread = (f - 0.5f) * 1.45f + (u01(rng) - 0.5f) * 0.28f;// radians off +v
            Twig tw;
            tw.bx = 0.5f + (f - 0.5f) * 0.05f;
            tw.by = stemTop;
            tw.dx = std::sin(spread);
            tw.dy = std::cos(spread);
            // Ragged lengths: a fan of equal-length twigs gives the card a
            // manicured circular outline, and a canopy of those reads as topiary.
            tw.length = (0.60f + 0.28f * u01(rng)) * (1.f - 0.12f * std::abs(f - 0.5f) * 2.f);
            tw.curve = (u01(rng) - 0.5f) * 0.14f;
            twigs.push_back(tw);
        }

        const int perTwig = std::max(3, leafletsPerTwig);
        std::vector<Leaflet> leaflets;
        leaflets.reserve(static_cast<size_t>(twigCount) * (static_cast<size_t>(perTwig) + 1));

        // Point on a twig at parameter t (0 base .. 1 tip), including its bow.
        auto twigPoint = [](const Twig& tw, float t, float& ox, float& oy) {
            const float px = -tw.dy, py = tw.dx;// left normal
            const float bow = tw.curve * std::sin(t * PI);
            ox = tw.bx + tw.dx * tw.length * t + px * bow;
            oy = tw.by + tw.dy * tw.length * t + py * bow;
        };

        for (const auto& tw : twigs) {
            for (int i = 0; i < perTwig; ++i) {
                const float t = (static_cast<float>(i) + 0.6f) / static_cast<float>(perTwig);
                Leaflet lf;
                twigPoint(tw, t, lf.bx, lf.by);
                // Alternate sides; angle up-and-out from the twig. The angle
                // opens toward the base (older, more spread leaflets).
                const float side = (i & 1) ? 1.f : -1.f;
                const float a = (0.78f + 0.30f * (1.f - t)) + (u01(rng) - 0.5f) * 0.26f;
                const float ca = std::cos(a), sa = std::sin(a);
                // Rotate the twig axis by ±a to get the leaflet axis.
                lf.dx = tw.dx * ca - side * tw.dy * sa;
                lf.dy = tw.dy * ca + side * tw.dx * sa;
                // Largest around the lower-middle of the twig, shrinking to the tip.
                const float prof = 0.55f + 0.45f * detail::lobe(t * 1.10f, 0.6f);
                lf.length = 0.150f * lenScale * prof * (0.80f + 0.40f * u01(rng));
                lf.halfWidth = lf.length * widthRatio * (0.88f + 0.24f * u01(rng));
                lf.tint = 0.84f + 0.32f * u01(rng);
                // New growth at the twig tips runs yellower; shaded basal
                // leaflets run cooler and darker. This within-card hue spread is
                // what stops a canopy reading as one flat sheet of the same green.
                lf.hue = (0.55f - t) * 1.3f + (u01(rng) - 0.5f) * 0.5f;
                leaflets.push_back(lf);
            }
            {// terminal leaflet, on the twig axis
                Leaflet lf;
                twigPoint(tw, 1.f, lf.bx, lf.by);
                lf.dx = tw.dx;
                lf.dy = tw.dy;
                lf.length = 0.13f * lenScale * (0.9f + 0.2f * u01(rng));
                lf.halfWidth = lf.length * widthRatio;
                lf.tint = 1.00f + 0.16f * u01(rng);
                lf.hue = -0.6f;
                leaflets.push_back(lf);
            }
        }

        // Blade half-width profile along the leaflet (s = 0 petiole .. 1 tip).
        // `s^0.85` skews the widest point below halfway and draws the tip out to
        // a point — a symmetric sin() lobe reads as an almond, not a leaf.
        auto bladeProfile = [&](float s) {
            const float ovate = detail::lobe(std::pow(std::clamp(s, 0.f, 1.f), 0.85f), 0.70f);
            switch (shape) {
                case LeafShape::Lobed:
                    // Deep rounded lobes down both margins — oak.
                    return ovate * (0.62f + 0.38f * std::cos(6.f * PI * s + 0.6f));
                case LeafShape::Serrate:
                    // Fine marginal teeth — birch.
                    return ovate * (0.90f + 0.10f * std::sin(26.f * s));
                case LeafShape::Lanceolate:
                    // Long, narrow, drawn out to a point — willow.
                    return detail::lobe(std::pow(std::clamp(s, 0.f, 1.f), 0.72f), 1.00f);
                case LeafShape::Ovate:
                default:
                    return ovate;
            }
        };

        // Colour is accumulated in float and dilated across the cutout edge
        // before it is quantised, so no black background bleeds into the leaves.
        std::vector<float> rgb(static_cast<size_t>(size) * size * 3, 0.f);
        std::vector<float> alpha(static_cast<size_t>(size) * size, 0.f);
        std::vector<char> filled(static_cast<size_t>(size) * size, 0);

        const float aa = 1.2f / S;// ~1px antialiased margin, in uv units

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / S;
                const float v = (static_cast<float>(y) + 0.5f) / S;

                float outR = 0.f, outG = 0.f, outB = 0.f, outA = 0.f;
                bool hit = false;

                // Woody structure, drawn first so leaflets paint over it: the
                // stem plus every twig. Bare twig visible through the canopy
                // gaps is a strong realism cue — a card of pure leaves with no
                // wood in it reads as moss draped over the branches.
                {
                    auto stroke = [&](float ax, float ay, float bx, float by,
                                      float hw0, float hw1) {
                        const float ex = bx - ax, ey = by - ay;
                        const float len2 = ex * ex + ey * ey;
                        if (len2 < 1e-9f) return;
                        float t = ((u - ax) * ex + (v - ay) * ey) / len2;
                        t = std::clamp(t, 0.f, 1.f);
                        const float px2 = ax + ex * t, py2 = ay + ey * t;
                        const float d = std::sqrt((u - px2) * (u - px2) + (v - py2) * (v - py2));
                        const float hw = hw0 + (hw1 - hw0) * t;
                        if (d > hw + aa) return;
                        const float cov = std::clamp((hw - d) / aa + 0.5f, 0.f, 1.f);
                        if (cov <= 0.f) return;
                        outR = 0.25f; outG = 0.18f; outB = 0.11f;
                        outA = std::max(outA, cov);
                        hit = true;
                    };
                    stroke(0.5f, 0.f, twigs[0].bx, stemTop, 0.014f, 0.011f);
                    for (const auto& tw : twigs) {
                        float px0 = tw.bx, py0 = tw.by;
                        constexpr int steps = 6;
                        for (int s = 1; s <= steps; ++s) {
                            const float t = static_cast<float>(s) / static_cast<float>(steps);
                            float px1, py1;
                            twigPoint(tw, t, px1, py1);
                            stroke(px0, py0, px1, py1,
                                   0.010f * (1.f - 0.8f * (t - 1.f / steps)),
                                   0.010f * (1.f - 0.8f * t));
                            px0 = px1;
                            py0 = py1;
                        }
                    }
                }

                for (size_t li = 0; li < leaflets.size(); ++li) {
                    const Leaflet& lf = leaflets[li];
                    const float ddx = u - lf.bx, ddy = v - lf.by;
                    const float along = ddx * lf.dx + ddy * lf.dy;
                    if (along < 0.f || along > lf.length) continue;
                    const float perp = -ddx * lf.dy + ddy * lf.dx;
                    const float s = along / lf.length;
                    const float hw = lf.halfWidth * bladeProfile(s);
                    if (hw <= 0.f) continue;
                    const float ap = std::abs(perp);
                    if (ap > hw + aa) continue;

                    const float cov = std::clamp((hw - ap) / aa + 0.5f, 0.f, 1.f);
                    if (cov <= 0.f) continue;

                    float r = baseColor[0] * lf.tint;
                    float g = baseColor[1] * lf.tint;
                    float b = baseColor[2] * lf.tint;
                    // Hue: negative = new growth (yellow-green), positive = older
                    // shade leaf (deeper, bluer).
                    if (lf.hue < 0.f) {
                        r += -lf.hue * 0.11f;
                        g += -lf.hue * 0.08f;
                        b -= -lf.hue * 0.015f;
                    } else {
                        r -= lf.hue * 0.045f;
                        g -= lf.hue * 0.030f;
                        b += lf.hue * 0.030f;
                    }

                    // Blade shading: darker toward the margin and the petiole,
                    // a lighter midrib, and faint lateral veins angled to the tip.
                    const float edge = ap / std::max(hw, 1e-4f);
                    float shade = (1.f - 0.20f * edge * edge) * (0.86f + 0.14f * s);
                    const float midrib = std::exp(-(ap * ap) / (0.00004f + hw * hw * 0.006f));
                    const float lateral = 0.5f + 0.5f * std::sin((s * 26.f) + ap * 40.f);
                    shade *= (0.94f + 0.06f * lateral);
                    r *= shade; g *= shade; b *= shade;
                    r += midrib * 0.045f;
                    g += midrib * 0.055f;
                    b += midrib * 0.020f;

                    // Painter's order: later leaflets sit on top.
                    outR = r; outG = g; outB = b;
                    outA = std::max(outA, cov);
                    hit = true;
                }

                if (!hit) continue;
                const auto pi = static_cast<size_t>(y) * size + x;
                rgb[pi * 3 + 0] = outR;
                rgb[pi * 3 + 1] = outG;
                rgb[pi * 3 + 2] = outB;
                alpha[pi] = outA;
                filled[pi] = 1;
            }
        }

        detail::dilateRGB(rgb, filled, size, 6);

        for (size_t pi = 0; pi < alpha.size(); ++pi) {
            px[pi * 4 + 0] = detail::toByte(rgb[pi * 3 + 0]);
            px[pi * 4 + 1] = detail::toByte(rgb[pi * 3 + 1]);
            px[pi * 4 + 2] = detail::toByte(rgb[pi * 3 + 2]);
            px[pi * 4 + 3] = detail::toByte(alpha[pi]);
        }

        texgen::finishTexture(tex, true, false);
        return tex;
    }

    // ── Needle frond alpha cutout (conifers) ─────────────────────────────
    //
    // An elongated feather/pinnate frond: a central rachis running up the tile
    // (v axis) with rows of angled needles down both sides, thinning toward the
    // tip, on a transparent background. Dark blue-green, alpha-cutout. Use as
    // `map` on the LeafStyle::Frond cards (alphaTest + Side::Double), where the
    // card's long (v) axis is the branch direction and u spreads sideways — so
    // the drawn rachis lands along the branch and the needles fan out in-plane.
    inline std::shared_ptr<DataTexture> makeNeedleFrondTexture(
            unsigned int size = 256,
            unsigned int seed = 1337,
            const std::array<float, 3>& baseColor = {0.11f, 0.29f, 0.10f}) {

        auto tex = DataTexture::create(4, size, size);
        auto& px = tex->image().data<unsigned char>();// zero → transparent

        math::Rng rng(seed);
        auto u01 = [](math::Rng& r) { return r.nextFloat(); };

        const auto S = static_cast<float>(size);
        const float vBase = 0.03f, vTip = 0.97f;

        // INDIVIDUAL needles, not a blade with a serrated margin.
        //
        // A solid lanceolate blade whose edge is nibbled by a ripple is a
        // fundamentally different silhouette from a conifer spray: it is opaque
        // through the middle, so no sky ever shows between the needles, and its
        // outline reads as a saw or a fern at any distance where the ripple
        // resolves. A real spruce spray is mostly EMPTY — a twig with a few
        // hundred separate needles, and the gaps are most of what you see.
        struct Needle {
            float bx, by;// root on the twig (uv)
            float dx, dy;// unit axis
            float length;
            float halfWidth;
            float tint;
        };

        const float twigCurve = (u01(rng) - 0.5f) * 0.05f;
        auto twigU = [&](float t) { return 0.5f + twigCurve * std::sin(t * 2.6f); };

        constexpr int rows = 46;// needle pairs down the twig
        std::vector<Needle> needles;
        needles.reserve(static_cast<size_t>(rows) * 2 + 2);
        for (int i = 0; i < rows; ++i) {
            const float t = (static_cast<float>(i) + 0.5f) / static_cast<float>(rows);
            const float by = vBase + t * (vTip - vBase);
            // Lanceolate envelope: longest needles a third of the way up.
            const float env = 0.42f + 0.58f * detail::lobe(std::min(t * 1.18f, 1.f), 0.5f);
            for (int s = 0; s < 2; ++s) {
                Needle n;
                n.bx = twigU(t);
                n.by = by;
                // Sweep toward the tip: ~55-70° off the twig, so needles rake
                // forward the way they do on a real shoot.
                const float a = 0.95f + 0.28f * u01(rng);
                const float side = (s == 0) ? -1.f : 1.f;
                n.dx = side * std::sin(a);
                n.dy = std::cos(a);
                n.length = 0.34f * env * (0.78f + 0.44f * u01(rng));
                n.halfWidth = 0.0085f * (0.75f + 0.5f * u01(rng));
                n.tint = 0.72f + 0.52f * u01(rng);
                needles.push_back(n);
            }
        }
        {// leader needle off the tip
            Needle n;
            n.bx = twigU(1.f);
            n.by = vTip - 0.06f;
            n.dx = 0.f;
            n.dy = 1.f;
            n.length = 0.10f;
            n.halfWidth = 0.009f;
            n.tint = 1.05f;
            needles.push_back(n);
        }

        std::vector<float> rgb(static_cast<size_t>(size) * size * 3, 0.f);
        std::vector<float> alpha(static_cast<size_t>(size) * size, 0.f);
        std::vector<char> filled(static_cast<size_t>(size) * size, 0);
        const float aa = 1.1f / S;

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / S;
                const float v = (static_cast<float>(y) + 0.5f) / S;

                float outR = 0.f, outG = 0.f, outB = 0.f, outA = 0.f;
                bool hit = false;

                // Woody twig up the middle.
                {
                    const float t = std::clamp((v - vBase) / (vTip - vBase), 0.f, 1.f);
                    const float du = std::abs(u - twigU(t));
                    const float hw = 0.0075f * (1.f - 0.5f * t);
                    if (v >= vBase && v <= vTip && du < hw + aa) {
                        const float cov = std::clamp((hw - du) / aa + 0.5f, 0.f, 1.f);
                        if (cov > 0.f) {
                            outR = 0.22f; outG = 0.15f; outB = 0.09f;
                            outA = cov;
                            hit = true;
                        }
                    }
                }

                for (const auto& n : needles) {
                    const float ddx = u - n.bx, ddy = v - n.by;
                    const float along = ddx * n.dx + ddy * n.dy;
                    if (along < 0.f || along > n.length) continue;
                    const float perp = -ddx * n.dy + ddy * n.dx;
                    const float s = along / n.length;
                    // Near-parallel sides tapering to a point in the last third —
                    // a needle, not a lens.
                    const float taper = (s < 0.7f) ? 1.f : (1.f - (s - 0.7f) / 0.3f);
                    const float hw = n.halfWidth * (0.72f + 0.28f * taper) * taper;
                    if (hw <= 0.f) continue;
                    const float ap = std::abs(perp);
                    if (ap > hw + aa) continue;
                    const float cov = std::clamp((hw - ap) / aa + 0.5f, 0.f, 1.f);
                    if (cov <= 0.f) continue;

                    // Bias the needle mass GREEN: red pulled down, green pushed
                    // up, so shadowed fronds stay green-dark rather than drifting
                    // bark-brown (near-silhouette conifers read as rock otherwise).
                    const float shade = n.tint * (0.86f + 0.14f * s);
                    outR = baseColor[0] * shade * 0.85f;
                    outG = baseColor[1] * shade * 1.12f;
                    outB = baseColor[2] * shade * 0.95f;
                    outA = std::max(outA, cov);
                    hit = true;
                }

                if (!hit) continue;
                const auto pi = static_cast<size_t>(y) * size + x;
                rgb[pi * 3 + 0] = outR;
                rgb[pi * 3 + 1] = outG;
                rgb[pi * 3 + 2] = outB;
                alpha[pi] = outA;
                filled[pi] = 1;
            }
        }

        // Needles are only a couple of texels wide, so almost every needle texel
        // is an edge texel — without this the whole spray filters toward the
        // black background and a conifer goes charcoal at any distance.
        detail::dilateRGB(rgb, filled, size, 6);

        for (size_t pi = 0; pi < alpha.size(); ++pi) {
            px[pi * 4 + 0] = detail::toByte(rgb[pi * 3 + 0]);
            px[pi * 4 + 1] = detail::toByte(rgb[pi * 3 + 1]);
            px[pi * 4 + 2] = detail::toByte(rgb[pi * 3 + 2]);
            px[pi * 4 + 3] = detail::toByte(alpha[pi]);
        }

        texgen::finishTexture(tex, true, false);
        return tex;
    }

    // ── Bark albedo + normal map (tiling) ────────────────────────────────
    //
    // Vertically-furrowed bark.  Returns {albedo, normal}.  Both tile
    // seamlessly; set the same `repeat` on each when assigning to a material.
    inline std::pair<std::shared_ptr<DataTexture>, std::shared_ptr<DataTexture>>
    makeBarkTextures(unsigned int size = 256,
                     unsigned int seed = 1337,
                     const std::array<float, 3>& baseColor = {0.34f, 0.24f, 0.16f},
                     BarkStyle style = BarkStyle::Furrowed) {

        auto albedo = DataTexture::create(4, size, size);
        auto normal = DataTexture::create(4, size, size);
        auto& ca = albedo->image().data<unsigned char>();

        const auto S = static_cast<float>(size);
        const int period = 8;// noise lattice period (in tiles) → seamless wrap

        // Height field: vertical furrows (stretched in y) + fbm detail.
        auto furrowedHeight = [&](float u, float v) -> float {
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

        // Irregular scale plates split by deep fissures — pine and spruce.
        // Warping the lattice coordinate before quantising it breaks the grid
        // up into uneven polygons; a plain quantised grid reads as brickwork.
        auto platedHeight = [&](float u, float v) -> float {
            const float wu = u + 0.06f * (detail::valueNoise(u * period, v * period, period, seed + 3u) - 0.5f);
            const float wv = v + 0.06f * (detail::valueNoise(u * period, v * period, period, seed + 5u) - 0.5f);
            const float cu = wu * static_cast<float>(period) * 1.5f;
            const float cv = wv * static_cast<float>(period) * 2.2f;
            const float fu = cu - std::floor(cu);
            const float fv = cv - std::floor(cv);
            // Distance to the nearest cell border → 0 in the fissure, 1 mid-plate.
            const float edge = std::min(std::min(fu, 1.f - fu), std::min(fv, 1.f - fv));
            const float plate = detail::smooth(std::clamp(edge * 5.f, 0.f, 1.f));
            const float grain = detail::valueNoise(u * period * 5.f, v * period * 5.f,
                                                   period * 5, seed + 17u);
            return std::clamp(plate * 0.82f + grain * 0.30f, 0.f, 1.f);
        };

        // Smooth papery bark: almost flat, with horizontal LENTICELS — the
        // short dark dashes that make birch instantly recognisable. Without
        // them a pale trunk is just a white pole, which is exactly how the
        // birch preset has been reading at any distance.
        auto paperyHeight = [&](float u, float v) -> float {
            // valueNoise only tiles when the coordinate SPAN is a multiple of
            // the lattice period. The band term wants a low u-frequency (2
            // lobes around the tile), so it gets its own period-2 lattice —
            // sampling u*2 against the shared period-8 lattice leaves the u=1
            // edge mid-cell, and with the bark repeating 3x around a trunk that
            // seam renders as three lit vertical creases on every birch.
            const float band = detail::valueNoise(u * 2.f, v * 12.f, 2, seed + 7u);
            const float fine = detail::valueNoise(u * static_cast<float>(period) * 4.f,
                                                  v * static_cast<float>(period) * 4.f,
                                                  period * 4, seed + 23u);
            return std::clamp(0.55f + band * 0.28f + fine * 0.17f, 0.f, 1.f);
        };

        // Lenticel mask, 1 inside a dash. Rows are spaced in v and each cell
        // either holds a dash or does not, so they scatter rather than stripe.
        auto lenticelAt = [&](float u, float v) -> float {
            constexpr int rowsPerTile = 14;
            constexpr int colsPerTile = 6;
            const float cv = v * rowsPerTile;
            const int row = static_cast<int>(std::floor(cv));
            const float fv = cv - static_cast<float>(row);
            const float cu = u * colsPerTile + detail::hash2(row, 0, seed + 41u) * 3.f;
            const int col = static_cast<int>(std::floor(cu));
            const float fu = cu - static_cast<float>(col);
            const int colW = ((col % colsPerTile) + colsPerTile) % colsPerTile;
            if (detail::hash2(colW, row, seed + 53u) > 0.55f) return 0.f;// empty cell
            // Dash: wide in u, thin in v.
            const float halfU = 0.14f + 0.26f * detail::hash2(colW, row, seed + 67u);
            const float halfV = 0.055f + 0.045f * detail::hash2(colW, row, seed + 71u);
            const float du = std::abs(fu - 0.5f);
            const float dv = std::abs(fv - 0.5f);
            if (du > halfU || dv > halfV) return 0.f;
            const float e = std::min((halfU - du) / 0.05f, (halfV - dv) / 0.02f);
            return std::clamp(e, 0.f, 1.f);
        };

        auto heightAt = [&](float u, float v) -> float {
            switch (style) {
                case BarkStyle::Plated: return platedHeight(u, v);
                case BarkStyle::Papery: return paperyHeight(u, v);
                case BarkStyle::Furrowed:
                default: return furrowedHeight(u, v);
            }
        };

        const float bumpScale = 2.5f;

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = static_cast<float>(x) / S;
                const float v = static_cast<float>(y) / S;
                const float h = heightAt(u, v);

                // Albedo: darker in furrows (low h), lighter on ridges. Papery
                // bark barely varies with height — its character is the
                // lenticels, not relief — so it gets a much flatter ramp.
                const float t = (style == BarkStyle::Papery) ? (0.86f + h * 0.14f)
                                                             : (0.45f + h * 0.55f);
                float r = baseColor[0] * t;
                float g = baseColor[1] * t;
                float b = baseColor[2] * t;
                // A little grey de-saturation on the ridges.
                const float grey = (r + g + b) / 3.f;
                const float desat = h * 0.25f;
                r = r * (1.f - desat) + grey * desat;
                g = g * (1.f - desat) + grey * desat;
                b = b * (1.f - desat) + grey * desat;

                if (style == BarkStyle::Papery) {
                    const float len = lenticelAt(u, v);
                    if (len > 0.f) {
                        // Dark grey-brown scar, blended over the pale ground.
                        r = r * (1.f - len) + 0.16f * len;
                        g = g * (1.f - len) + 0.14f * len;
                        b = b * (1.f - len) + 0.12f * len;
                    }
                }

                const size_t idx = (static_cast<size_t>(y) * size + x) * 4;
                ca[idx + 0] = detail::toByte(r);
                ca[idx + 1] = detail::toByte(g);
                ca[idx + 2] = detail::toByte(b);
                ca[idx + 3] = 255;
            }
        }

        // Matching normal map from the same (pure) height field, then finish
        // both for tiling wrap — shared helpers, same arithmetic as the old
        // inline block.
        texgen::writeNormalFromHeight(normal, size, bumpScale, heightAt, true);
        texgen::finishTexture(albedo, true, true);
        texgen::finishTexture(normal, false, true);

        return {albedo, normal};
    }

    // ── Wildflower alpha cutout ──────────────────────────────────────────
    //
    // A single bloom (petals + centre disc) on a short green stem, on a
    // transparent background. Use on a card material with alphaTest. The petal
    // colour is chosen from a small palette by `seed`, so passing different
    // seeds yields different-coloured flowers.
    inline std::shared_ptr<DataTexture> makeFlowerTexture(
            unsigned int size = 128,
            unsigned int seed = 1337) {

        auto tex = DataTexture::create(4, size, size);
        auto& px = tex->image().data<unsigned char>();// zero → transparent

        static const std::array<std::array<float, 3>, 5> palette = {{
                {0.96f, 0.96f, 0.94f},// white
                {0.97f, 0.85f, 0.25f},// yellow
                {0.93f, 0.42f, 0.62f},// pink
                {0.62f, 0.42f, 0.88f},// violet
                {0.88f, 0.30f, 0.26f},// red
        }};
        const auto& petal = palette[seed % palette.size()];
        const std::array<float, 3> centre = {0.98f, 0.80f, 0.20f};
        const std::array<float, 3> stem = {0.20f, 0.40f, 0.13f};

        const auto S = static_cast<float>(size);
        const float cx = 0.5f, cy = 0.64f;// bloom centre in uv space (v up)
        const float bloomR = 0.30f;
        const int nPetals = 6;
        const float centreR = 0.085f;
        const float stemHalf = 0.022f;

        std::vector<float> rgb(static_cast<size_t>(size) * size * 3, 0.f);
        std::vector<float> alpha(static_cast<size_t>(size) * size, 0.f);
        std::vector<char> filled(static_cast<size_t>(size) * size, 0);

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / S;
                const float v = (static_cast<float>(y) + 0.5f) / S;// 0 at base, 1 at top
                float r = 0.f, g = 0.f, b = 0.f, a = 0.f;

                // Stem.
                if (v < 0.6f && std::abs(u - 0.5f) < stemHalf) {
                    r = stem[0]; g = stem[1]; b = stem[2]; a = 1.f;
                }
                // Petals: rose-curve radius gives a flower outline.
                const float dx = u - cx, dy = v - cy;
                const float rr = std::sqrt(dx * dx + dy * dy);
                const float theta = std::atan2(dy, dx);
                const float petalEdge = bloomR * (0.55f + 0.45f * std::cos(static_cast<float>(nPetals) * theta));
                if (rr < petalEdge) {
                    const float shade = 0.8f + 0.2f * (1.f - rr / std::max(petalEdge, 1e-3f));
                    r = petal[0] * shade; g = petal[1] * shade; b = petal[2] * shade; a = 1.f;
                }
                if (rr < centreR) {
                    r = centre[0]; g = centre[1]; b = centre[2]; a = 1.f;
                }

                const auto pi = static_cast<size_t>(y) * size + x;
                if (a <= 0.f) continue;
                rgb[pi * 3 + 0] = r;
                rgb[pi * 3 + 1] = g;
                rgb[pi * 3 + 2] = b;
                alpha[pi] = a;
                filled[pi] = 1;
            }
        }

        // Petals are small and bright against a transparent BLACK background;
        // without dilation the mip chain drags that black into every bloom and
        // a meadow of white flowers turns to grey specks with distance.
        detail::dilateRGB(rgb, filled, size, 5);

        for (size_t pi = 0; pi < alpha.size(); ++pi) {
            px[pi * 4 + 0] = detail::toByte(rgb[pi * 3 + 0]);
            px[pi * 4 + 1] = detail::toByte(rgb[pi * 3 + 1]);
            px[pi * 4 + 2] = detail::toByte(rgb[pi * 3 + 2]);
            px[pi * 4 + 3] = detail::toByte(alpha[pi]);
        }

        texgen::finishTexture(tex, true, false);
        return tex;
    }

}// namespace threepp::vegetation

#endif// THREEPP_EXTRAS_VEGETATION_TREETEXTURES_HPP
