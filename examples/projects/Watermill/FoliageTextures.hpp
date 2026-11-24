// The woods' surfaces, painted per pixel like the rest of the mill's (TextureLab.hpp): one atlas for every leaf and
// frond, and three barks.
//
// A leaf card is a diamond folded along its midrib (two triangles: stem, one edge, tip, other edge), so each leaf in
// the atlas is drawn to fit the diamond between its cell's mid-edges, stem at the bottom. The atlas builds its own
// mip chain: each level keeps the share of texels that pass the alpha test that the full-size cell has, or distant
// crowns would thin out to nothing as the mips average the leaves into their transparent surroundings.

#ifndef THREEPP_EXAMPLE_WATERMILL_FOLIAGE_TEXTURES_HPP
#define THREEPP_EXAMPLE_WATERMILL_FOLIAGE_TEXTURES_HPP

#include "threepp/textures/DataTexture.hpp"

#include "TextureLab.hpp"

#include <array>
#include <vector>

namespace mill::woods {

    // ---- the atlas's layout -------------------------------------------------------------------------------------
    // 2048 x 2048. Rows 0 and 1 (v 0..0.5): eight leaves in 512 x 512 cells. Rows 2-3 (v 0.5..1), columns 0 and 1:
    // two fern fronds in tall 512 x 1024 cells; columns 2 and 3: four leafy shoots (a twig with its leaves, drawn
    // leaf by leaf), the cards of the trees' middle and far detail, so a distant crown keeps leaves of the size its
    // near twin has rather than one big leaf for six small ones.
    enum LeafKind : int {
        Tulip = 0,// tulip poplar: four lobes and a square-cut tip
        Maple,    // sugar maple: five pointed lobes
        Oak,      // white oak: rounded lobes
        Beech,    // ovate, serrate, straight parallel veins
        Basswood, // heart-shaped, serrate
        Rhodo,    // rhododendron: long, smooth, leathery
        Birch,    // small, pointed, doubly serrate
        Dogwood,  // elliptic with veins curving toward the tip
        FernLady, // a lacy frond, twice divided
        FernXmas, // a dark, once-divided frond (Christmas fern)
        ShootTulip,// leafy shoots
        ShootMaple,
        ShootBeech,
        ShootOak,
    };

    // the leafy shoot that stands in for a kind of leaf in the distance
    inline int shootKindFor(int leaf) {
        switch (leaf) {
            case Tulip: return ShootTulip;
            case Maple: return ShootMaple;
            case Oak: return ShootOak;
            default: return ShootBeech;
        }
    }
    inline int leafOfShoot(int shoot) {
        static constexpr std::array<int, 4> k{Tulip, Maple, Beech, Oak};
        return k[static_cast<size_t>(shoot - ShootTulip)];
    }

    // width / length of a leaf's card
    inline float leafAspect(int kind) {
        static constexpr std::array<float, 14> a{1.0f, 1.0f, 0.64f, 0.6f, 0.86f, 0.36f, 0.68f, 0.62f, 0.5f, 0.5f, 0.85f, 0.85f, 0.8f, 0.8f};
        return a[static_cast<size_t>(kind)];
    }

    struct AtlasCell {
        float u0, v0, du, dv;
    };

    inline AtlasCell atlasCell(int kind) {
        if (kind < 8) return {static_cast<float>(kind % 4) * 0.25f, static_cast<float>(kind / 4) * 0.25f, 0.25f, 0.25f};
        if (kind < ShootTulip) return {static_cast<float>(kind - 8) * 0.25f, 0.5f, 0.25f, 0.5f};
        const int c = kind - ShootTulip;
        return {0.5f + static_cast<float>(c % 2) * 0.25f, 0.5f + static_cast<float>(c / 2) * 0.25f, 0.25f, 0.25f};
    }

    // ---- leaf outlines --------------------------------------------------------------------------------------------
    // Physical coordinates over a card of length 1: X across (|X| <= aspect / 2), y from the stem (0) to the tip (1).
    struct LeafSample {
        float inside = -1;// > 0 inside the blade or the stalk, in card lengths
        float vein = 0;   // 0..1, the midrib and the main veins
        float fine = 0;   // the finer veins
        float t = 0;      // 0 at the blade's base, 1 at its tip
        bool stalk = false;
    };

    inline float distToSegment(float px, float py, float ax, float ay, float bx, float by) {
        const float vx = bx - ax, vy = by - ay;
        const float t = clamp01(((px - ax) * vx + (py - ay) * vy) / std::max(vx * vx + vy * vy, 1e-9f));
        return std::hypot(px - ax - t * vx, py - ay - t * vy);
    }

    inline float fract(float v) { return v - std::floor(v); }

    // piecewise-linear radius over the angle from the tip (degrees, 0..180)
    template<size_t N>
    float polarTable(const std::array<std::array<float, 2>, N>& table, float deg) {
        for (size_t i = 0; i + 1 < N; ++i) {
            if (deg <= table[i + 1][0]) {
                const float t = (deg - table[i][0]) / (table[i + 1][0] - table[i][0]);
                const float s = t * t * (3.f - 2.f * t);
                return table[i][1] + (table[i + 1][1] - table[i][1]) * s;
            }
        }
        return table[N - 1][1];
    }

    inline LeafSample leafShape(int kind, float X, float y) {
        constexpr float pi = 3.14159265f;
        LeafSample s;
        const float ax = std::abs(X);

        // a pinnate blade between yb and yt
        const auto pinnate = [&](float yb, float yt, float halfW, float basePow, float tipPow, float teeth, float toothDepth,
                                 float veins, float veinSlope, float veinCurve) {
            const float t = (y - yb) / (yt - yb);
            s.t = t;
            if (t < 0.f || t > 1.f) return;
            float w = halfW * std::pow(std::sin(pi * std::pow(t, basePow)), tipPow);
            if (teeth > 0) {
                const float ph = fract(t * teeth + 0.13f * X);
                w *= 1.f - toothDepth * ph * std::sin(pi * std::min(t * 1.4f, 1.f));
            }
            s.inside = w - ax;
            const float rel = ax / std::max(halfW, 1e-4f);
            const float mid = smooth(0.0065f, 0.001f, ax) * smooth(1.02f, 0.85f, t);
            const float phase = (t - veinSlope * rel - veinCurve * rel * rel) * veins;
            const float d = std::abs(phase - std::round(phase));
            const float sec = smooth(0.1f, 0.f, d) * smooth(0.f, 0.05f, rel) * smooth(0.f, 0.03f, s.inside);
            s.vein = std::max(mid, 0.55f * sec);
        };

        switch (kind) {
            case Tulip: {
                // four lobes, a broad truncate tip with a shallow notch; the blade sits round (0, 0.53)
                static const std::array<std::array<float, 2>, 9> table{{{0, 0.245f}, {22, 0.27f}, {46, 0.335f}, {66, 0.25f}, {84, 0.275f}, {112, 0.33f}, {140, 0.25f}, {165, 0.14f}, {180, 0.1f}}};
                const float cy = 0.53f;
                const float dx = X, dy = y - cy;
                const float r = std::hypot(dx, dy);
                const float deg = std::atan2(std::abs(dx), dy) * 180.f / pi;
                float R = polarTable(table, deg);
                R *= 1.f - 0.16f * std::exp(-deg * deg / 60.f);
                s.inside = R - r;
                s.t = clamp01((y - 0.2f) / 0.62f);
                // palmate-pinnate veins: the midrib and four main veins from the base
                const float jy = cy - 0.12f;
                float dv = distToSegment(X, y, 0, jy, 0, cy + 0.24f);
                dv = std::min(dv, distToSegment(ax, y, 0, jy + 0.06f, 0.24f, 0.75f));
                dv = std::min(dv, distToSegment(ax, y, 0, jy, 0.28f, 0.38f));
                s.vein = smooth(0.007f, 0.001f, dv);
                const float ph = (y - 0.5f * ax) * 16.f;
                s.fine = smooth(0.1f, 0.f, std::abs(ph - std::round(ph))) * 0.5f;
                if (y < 0.3f && ax < 0.008f) s.stalk = true;
                break;
            }
            case Maple: {
                // five pointed lobes with a few big teeth, deep round sinuses
                const float cy = 0.5f;
                const float dx = X, dy = y - cy;
                const float r = std::hypot(dx, dy);
                const float th = std::atan2(std::abs(dx), dy);// 0 at the tip
                const std::array<float, 3> at{0.f, 0.92f, 1.9f};
                const std::array<float, 3> amp{0.27f, 0.19f, 0.14f};
                float R = 0.15f;
                for (int i = 0; i < 3; ++i) {
                    const float d = (th - at[i]) * 3.6f;
                    if (std::abs(d) < pi / 2) R += amp[i] * std::pow(std::cos(d), 2.2f);
                }
                // teeth: a pair of side points on each lobe
                R *= 1.f + 0.07f * std::pow(std::max(0.f, std::cos(th * 13.f)), 6.f) * smooth(0.18f, 0.3f, R);
                if (th > 2.75f) R *= 0.55f;// the notch at the stem
                s.inside = R - r;
                s.t = clamp01((y - 0.15f) / 0.8f);
                float dv = 1e9f;
                for (int i = 0; i < 3; ++i) {
                    const float rr = 0.13f + amp[i];
                    dv = std::min(dv, distToSegment(ax, y, 0, cy - 0.06f, rr * std::sin(at[i]), cy + rr * std::cos(at[i])));
                }
                s.vein = smooth(0.0065f, 0.001f, dv);
                const Cell c = worley2(X * 40.f, y * 40.f, 77u);
                s.fine = smooth(0.06f, 0.f, c.f2 - c.f1) * 0.5f;
                if (y < 0.24f && ax < 0.007f) s.stalk = true;
                break;
            }
            case Oak: {
                // white oak: seven to nine rounded lobes, deep sinuses, a wedge base
                const float yb = 0.13f, yt = 0.97f;
                const float t = (y - yb) / (yt - yb);
                s.t = t;
                if (t >= 0.f && t <= 1.f) {
                    const float profile = 0.3f * std::pow(std::sin(pi * std::pow(t, 0.85f)), 0.75f);
                    const float lobes = 0.4f + 0.6f * std::pow(std::abs(std::sin(pi * (t * 3.6f + 0.15f))), 0.55f);
                    const float w = profile * lobes * (t > 0.88f ? 1.f : 1.f);
                    s.inside = w - ax;
                    const float mid = smooth(0.007f, 0.001f, ax);
                    // one vein into each lobe
                    const float ph = t * 3.6f + 0.15f - 0.5f - 0.6f * ax;
                    const float d = std::abs(ph - std::round(ph));
                    s.vein = std::max(mid, 0.6f * smooth(0.06f, 0.f, d) * smooth(0.f, 0.02f, ax));
                }
                if (y < yb + 0.02f && ax < 0.008f) s.stalk = true;
                break;
            }
            case Beech:
                pinnate(0.09f, 0.98f, 0.26f, 0.82f, 0.85f, 11.f, 0.07f, 11.f, 0.32f, 0.f);
                if (y < 0.11f && ax < 0.007f) s.stalk = true;
                break;
            case Basswood: {
                // heart-shaped: wide and lobed at the base, an abrupt drawn-out tip
                const float yb = 0.16f, yt = 0.98f;
                const float t = (y - yb) / (yt - yb);
                s.t = t;
                if (t >= 0.f && t <= 1.f) {
                    float w = 0.36f * std::pow(std::sin(pi * std::pow(t, 0.62f)), 1.1f);
                    w *= 1.f + 0.35f * smooth(0.2f, 0.f, t);// the cordate base
                    w *= 1.f - 0.06f * fract(t * 26.f);
                    s.inside = w - ax;
                    // the heart's notch at the stalk
                    if (t < 0.12f && ax < 0.06f * (0.12f - t) / 0.12f) s.inside = -1;
                    const float rel = ax / 0.36f;
                    const float ph = (t - 0.35f * rel) * 8.f;
                    s.vein = std::max(smooth(0.007f, 0.001f, ax), 0.55f * smooth(0.1f, 0.f, std::abs(ph - std::round(ph))) * smooth(0.f, 0.05f, rel));
                }
                if (y < yb + 0.05f && ax < 0.007f) s.stalk = true;
                break;
            }
            case Rhodo:
                pinnate(0.08f, 0.99f, 0.15f, 0.92f, 0.7f, 0.f, 0.f, 16.f, 0.25f, 0.f);
                if (y < 0.1f && ax < 0.009f) s.stalk = true;
                break;
            case Birch: {
                pinnate(0.16f, 0.99f, 0.3f, 0.55f, 1.05f, 18.f, 0.1f, 8.f, 0.4f, 0.f);
                if (y < 0.18f && ax < 0.006f) s.stalk = true;
                break;
            }
            case Dogwood:
            default:
                pinnate(0.1f, 0.98f, 0.27f, 0.85f, 0.8f, 0.f, 0.f, 6.f, 0.15f, 0.55f);
                if (y < 0.12f && ax < 0.007f) s.stalk = true;
                break;
        }
        if (s.stalk) {
            s.inside = std::max(s.inside, 0.0015f);
        }
        return s;
    }

    // ---- fern fronds ------------------------------------------------------------------------------------------------
    // Physical coordinates over a frond's card: X across (|X| <= 0.25), y from the stipe's foot (0) to the tip (1).
    inline LeafSample frondShape(int kind, float X, float y) {
        constexpr float pi = 3.14159265f;
        LeafSample s;
        const float ax = std::abs(X);
        const float y0 = kind == FernLady ? 0.12f : 0.1f;// the bare stipe below the blade
        const float t = (y - y0) / (0.99f - y0);
        s.t = t;
        // the rachis
        const float rachisW = 0.0045f * (1.f - 0.7f * clamp01(t));
        if (y > 0.f && y < 0.99f && ax < rachisW) {
            s.inside = rachisW - ax + 0.001f;
            s.stalk = true;
            s.vein = 1.f;
        }
        if (t < 0.f || t > 1.f) return s;
        // the blade's outline: the pinnae's lengths
        const float halfW = kind == FernLady ? 0.235f * std::pow(std::sin(pi * std::pow(t, 0.62f)), 0.9f)
                                             : 0.19f * std::pow(std::sin(pi * std::pow(t, 0.5f)), 0.7f) * (0.75f + 0.25f * smooth(0.f, 0.15f, t));
        if (halfW < 0.004f) return s;
        // the pinnae rise toward the tip; they alternate from side to side
        const float slope = kind == FernLady ? 0.38f : 0.22f;
        const float count = kind == FernLady ? 19.f : 24.f;
        const float side = X < 0 ? 0.5f : 0.f;
        const float ph = (t - slope * ax) * count + side;
        const float k = std::floor(ph);
        const float across = ph - k - 0.5f;// -0.5..0.5 across one pinna's band
        // where this pinna sits along the frond, and its length
        const float tk = (k + 0.5f - side) / count + slope * ax;
        const float lenK = kind == FernLady ? 0.235f * std::pow(std::sin(pi * std::pow(clamp01(tk), 0.62f)), 0.9f)
                                            : 0.19f * std::pow(std::sin(pi * std::pow(clamp01(tk), 0.5f)), 0.7f);
        const float along = ax / std::max(lenK, 1e-3f);// 0 at the rachis, 1 at the pinna's tip
        if (along > 1.f) return s;
        float width;
        if (kind == FernLady) {
            // pinnules: each pinna cut into rounded lobes
            const float lobes = 0.55f + 0.45f * std::pow(std::abs(std::cos(pi * along * 7.f)), 0.6f);
            width = 0.4f * std::pow(1.f - along, 0.55f) * lobes;
        } else {
            // whole pinnae with a small ear at the base on the upper side
            width = 0.42f * std::sqrt(std::max(0.f, 1.f - along * along));
            if (across > 0 && along < 0.2f) width += 0.12f * smooth(0.2f, 0.05f, along);
            width *= 1.f - 0.08f * fract(along * 12.f);
        }
        const float inPinna = (width - std::abs(across)) / count;// in card lengths
        if (inPinna > s.inside) {
            s.inside = inPinna;
            s.stalk = false;
            const float costa = smooth(0.06f, 0.f, std::abs(across));
            s.vein = 0.7f * costa;
            s.fine = 0.3f * smooth(0.08f, 0.f, std::abs(fract(along * 9.f) - 0.5f));
        }
        return s;
    }

    // ---- leafy shoots ---------------------------------------------------------------------------------------------
    // A twig up the middle of the card with leaves alternating off it and one at its tip, each leaf the outline of
    // its kind. Physical coordinates as for a leaf (card length 1). `shade` is the covering leaf's own brightness.
    inline LeafSample shootShape(int shootKind, float X, float y, float& shade) {
        const int kind = leafOfShoot(shootKind);
        LeafSample best;
        shade = 1.f;
        // the twig
        const float twigTop = 0.55f;
        if (y > 0.f && y < twigTop && std::abs(X) < 0.009f * (1.f - 0.5f * y)) {
            best.inside = 0.009f * (1.f - 0.5f * y) - std::abs(X);
            best.stalk = true;
        }
        const int n = 5;
        for (int k = 0; k <= n; ++k) {
            const bool terminal = k == n;
            const float by = terminal ? twigTop : 0.07f + 0.44f * static_cast<float>(k) / static_cast<float>(n);
            const float side = (k % 2 == 0) ? 1.f : -1.f;
            const float ang = terminal ? 0.f : side * (0.95f - 0.4f * static_cast<float>(k) / static_cast<float>(n));
            const float len = terminal ? 0.45f : 0.32f + 0.08f * static_cast<float>(k) / static_cast<float>(n);
            // into the leaf's own frame: its stem at the origin, its tip along +y
            const float dx = X - 0.f, dy = y - by;
            const float c = std::cos(ang), s = std::sin(ang);
            const float lx = (dx * c - dy * s) / len, ly = (dx * s + dy * c) / len;
            if (ly < -0.05f || ly > 1.05f) continue;
            const LeafSample l = leafShape(kind, lx * (kind == Tulip || kind == Maple ? 1.f : 1.f), ly);
            if (l.inside > 0.f) {
                LeafSample r = l;
                r.inside = l.inside * len;
                best = r;// the later leaves lie over the earlier ones
                shade = 0.84f + 0.3f * hash2f(k, shootKind, 5u);
            }
        }
        return best;
    }

    // ---- the atlas ----------------------------------------------------------------------------------------------------
    struct LeafColours {
        V3 base, vein, edge;
        float rough;
    };

    inline LeafColours leafColours(int kind) {
        switch (kind) {
            case Tulip: return {srgb(112, 150, 52), srgb(168, 186, 92), srgb(132, 152, 54), 0.55f};
            case Maple: return {srgb(88, 128, 44), srgb(140, 166, 80), srgb(104, 130, 46), 0.68f};
            case Oak: return {srgb(78, 114, 40), srgb(128, 152, 76), srgb(96, 118, 44), 0.66f};
            case Beech: return {srgb(102, 142, 52), srgb(150, 172, 86), srgb(116, 140, 52), 0.64f};
            case Basswood: return {srgb(96, 134, 46), srgb(146, 170, 84), srgb(110, 134, 48), 0.68f};
            case Rhodo: return {srgb(46, 76, 36), srgb(104, 124, 66), srgb(62, 80, 38), 0.42f};
            case Birch: return {srgb(110, 148, 54), srgb(156, 176, 90), srgb(124, 148, 56), 0.62f};
            case Dogwood: return {srgb(92, 130, 50), srgb(142, 166, 88), srgb(106, 130, 50), 0.68f};
            case FernLady: return {srgb(116, 156, 58), srgb(146, 176, 78), srgb(130, 158, 60), 0.7f};
            case FernXmas:
            default: return {srgb(54, 90, 38), srgb(96, 122, 60), srgb(68, 94, 40), 0.5f};
        }
    }

    // the colour texture with a mip chain that keeps each block's alpha-test coverage
    inline std::shared_ptr<threepp::DataTexture> coverageMippedAlbedo(const Maps& m, float alphaRef, int block) {
        using namespace threepp;
        int w = m.w, h = m.h;
        std::vector<V3> col = m.albedo;
        std::vector<float> a = m.alpha;
        const int bx = w / block, by = h / block;
        std::vector<float> cover0(static_cast<size_t>(bx) * by, 0.f);
        for (int j = 0; j < by; ++j) {
            for (int i = 0; i < bx; ++i) {
                int n = 0;
                for (int y = j * block; y < (j + 1) * block; ++y)
                    for (int x = i * block; x < (i + 1) * block; ++x) n += a[static_cast<size_t>(y) * w + x] >= alphaRef ? 1 : 0;
                cover0[static_cast<size_t>(j) * bx + i] = static_cast<float>(n) / static_cast<float>(block * block);
            }
        }
        const auto pack = [](const std::vector<V3>& c, const std::vector<float>& alpha, const std::vector<float>& scale, int lw, int lh, int bs, int nbx) {
            std::vector<unsigned char> px(static_cast<size_t>(lw) * lh * 4);
            for (int y = 0; y < lh; ++y) {
                for (int x = 0; x < lw; ++x) {
                    const size_t k = static_cast<size_t>(y) * lw + x;
                    const float s = scale.empty() ? 1.f : scale[static_cast<size_t>(y / bs) * nbx + x / bs];
                    px[4 * k] = toSrgbByte(c[k].r);
                    px[4 * k + 1] = toSrgbByte(c[k].g);
                    px[4 * k + 2] = toSrgbByte(c[k].b);
                    px[4 * k + 3] = toByte(alpha[k] * s);
                }
            }
            return px;
        };
        auto tex = DataTexture::create(ImageData(pack(col, a, {}, w, h, 1, 1)), w, h);
        tex->mipmaps().emplace_back(ImageData(pack(col, a, {}, w, h, 1, 1)), w, h);
        int level = 0;
        while (w > 1 || h > 1) {
            ++level;
            const int w2 = std::max(1, w / 2), h2 = std::max(1, h / 2);
            std::vector<V3> c2(static_cast<size_t>(w2) * h2);
            std::vector<float> a2(c2.size());
            for (int y = 0; y < h2; ++y) {
                for (int x = 0; x < w2; ++x) {
                    V3 sum;
                    float as = 0, wsum = 0;
                    V3 plain;
                    for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                            const int sx = std::min(2 * x + dx, w - 1), sy = std::min(2 * y + dy, h - 1);
                            const size_t k = static_cast<size_t>(sy) * w + sx;
                            sum = sum + col[k] * (a[k] + 1e-4f);
                            wsum += a[k] + 1e-4f;
                            as += a[k];
                            plain = plain + col[k];
                        }
                    }
                    const size_t k2 = static_cast<size_t>(y) * w2 + x;
                    c2[k2] = sum * (1.f / wsum);
                    a2[k2] = as * 0.25f;
                }
            }
            // per block of this level, the alpha scale that restores the coverage
            const int bs = std::max(4, block >> level);
            const int nbx = (w2 + bs - 1) / bs, nby = (h2 + bs - 1) / bs;
            std::vector<float> scale(static_cast<size_t>(nbx) * nby, 1.f);
            for (int j = 0; j < nby; ++j) {
                for (int i = 0; i < nbx; ++i) {
                    // the target: the mean coverage of the level-0 blocks under this one
                    const float f = static_cast<float>(1 << level);
                    const int i0 = static_cast<int>(static_cast<float>(i * bs) * f) / block, i1 = std::min(bx, std::max(i0 + 1, static_cast<int>(static_cast<float>((i + 1) * bs) * f) / block));
                    const int j0 = static_cast<int>(static_cast<float>(j * bs) * f) / block, j1 = std::min(by, std::max(j0 + 1, static_cast<int>(static_cast<float>((j + 1) * bs) * f) / block));
                    float target = 0;
                    int nt = 0;
                    for (int jj = j0; jj < j1; ++jj)
                        for (int ii = i0; ii < i1; ++ii) target += cover0[static_cast<size_t>(jj) * bx + ii], ++nt;
                    target /= static_cast<float>(std::max(nt, 1));
                    if (target <= 0.f) continue;
                    const auto coverage = [&](float s) {
                        int n = 0, all = 0;
                        for (int y = j * bs; y < std::min(h2, (j + 1) * bs); ++y)
                            for (int x = i * bs; x < std::min(w2, (i + 1) * bs); ++x) {
                                n += a2[static_cast<size_t>(y) * w2 + x] * s >= alphaRef ? 1 : 0;
                                ++all;
                            }
                        return static_cast<float>(n) / static_cast<float>(std::max(all, 1));
                    };
                    float lo = 0.5f, hi = 12.f;
                    for (int it = 0; it < 14; ++it) {
                        const float mid = 0.5f * (lo + hi);
                        if (coverage(mid) < target) lo = mid;
                        else hi = mid;
                    }
                    scale[static_cast<size_t>(j) * nbx + i] = std::max(1.f, 0.5f * (lo + hi));
                }
            }
            tex->mipmaps().emplace_back(ImageData(pack(c2, a2, scale, w2, h2, bs, nbx)), w2, h2);
            col = std::move(c2);
            a = std::move(a2);
            w = w2;
            h = h2;
        }
        tex->format = Format::RGBA;
        tex->type = Type::UnsignedByte;
        tex->wrapS = tex->wrapT = TextureWrapping::ClampToEdge;
        tex->magFilter = Filter::Linear;
        tex->minFilter = Filter::LinearMipmapLinear;
        tex->generateMipmaps = false;
        tex->anisotropy = 8;
        tex->colorSpace = ColorSpace::sRGB;
        tex->needsUpdate();
        return tex;
    }

    inline TexSet foliageAtlas(uint32_t seed) {
        const int N = 2048, cell = 512;
        Maps m(N, N, 0.6f, 0.6f);
        forRows(N, [&](int py) {
            for (int px = 0; px < N; ++px) {
                const size_t c = m.at(px, py);
                const int col = px / cell;
                int kind;
                float x, y;// cell coordinates: x -0.5..0.5, y 0..1
                if (py < 2 * cell) {
                    kind = (py / cell) * 4 + col;
                    x = (static_cast<float>(px % cell) + 0.5f) / static_cast<float>(cell) - 0.5f;
                    y = (static_cast<float>(py % cell) + 0.5f) / static_cast<float>(cell);
                } else if (col < 2) {
                    kind = FernLady + col;
                    x = (static_cast<float>(px % cell) + 0.5f) / static_cast<float>(cell) - 0.5f;
                    y = (static_cast<float>(py - 2 * cell) + 0.5f) / static_cast<float>(2 * cell);
                } else {
                    kind = ShootTulip + (col - 2) + 2 * ((py - 2 * cell) / cell);
                    x = (static_cast<float>(px % cell) + 0.5f) / static_cast<float>(cell) - 0.5f;
                    y = (static_cast<float>(py % cell) + 0.5f) / static_cast<float>(cell);
                }
                const float aspect = leafAspect(kind);
                const float X = x * aspect;
                const bool frond = kind == FernLady || kind == FernXmas;
                const bool shoot = kind >= ShootTulip;
                float shootShade = 1.f;
                const LeafSample s = frond ? frondShape(kind, X, y) : (shoot ? shootShape(kind, X, y, shootShade) : leafShape(kind, X, y));
                const float texel = frond ? 1.f / static_cast<float>(2 * cell) : 1.f / static_cast<float>(cell);
                float alpha = clamp01(0.5f + s.inside / (1.2f * texel));
                if (!frond) {
                    // the card is the diamond between the cell's mid-edges
                    const float diamond = 0.5f - (std::abs(x) + std::abs(y - 0.5f));
                    alpha *= clamp01(0.5f + (diamond - 0.004f) / texel);
                }
                const LeafColours lc = leafColours(shoot ? leafOfShoot(kind) : kind);
                const uint32_t ks = seed + static_cast<uint32_t>(kind) * 31u;
                const float blotch = fbm2(X * 7.f + static_cast<float>(kind) * 3.7f, y * 7.f, 3, ks);
                const float grain = fbm2(X * 60.f, y * 60.f, 2, ks + 1);
                V3 colr = lc.base * (0.9f + 0.18f * clamp01(0.5f + blotch) + 0.06f * grain);
                colr = mix(colr, lc.vein, 0.75f * s.vein + 0.25f * s.fine);
                // lighter toward the tip and the margin, as a leaf held up to the light is
                const float margin = smooth(0.03f, 0.f, s.inside);
                colr = mix(colr, lc.edge, 0.4f * margin);
                colr = colr * (0.92f + 0.12f * clamp01(s.t));
                colr = colr * shootShade;
                if (s.stalk) colr = shoot ? srgb(96, 84, 62) : mix(lc.vein, srgb(110, 104, 54), 0.4f);
                m.albedo[c] = colr;
                m.alpha[c] = alpha;
                // veins sunken in a cupped blade
                const float rel = std::abs(X) / (0.5f * aspect);
                m.height[c] = -0.00045f * rel * rel - 0.00012f * s.vein - 0.00004f * s.fine + 0.00002f * grain;
                m.rough[c] = lc.rough + 0.08f * s.vein;
            }
        });
        // the normal and ORM maps as usual; the colour with its own coverage-keeping mips
        TexSet set = m.finish(0.5f, false, false);
        set.albedo = coverageMippedAlbedo(m, 0.5f, 256);
        set.normal->wrapS = set.normal->wrapT = threepp::TextureWrapping::ClampToEdge;
        set.orm->wrapS = set.orm->wrapT = threepp::TextureWrapping::ClampToEdge;
        return set;
    }

    // ---- bark ----------------------------------------------------------------------------------------------------
    // 0.4 m round by 0.8 m along (u round the stem, v along it).
    enum BarkKind : int { Furrowed = 0, SmoothGrey = 1, BirchBark = 2 };

    inline TexSet woodsBark(int kind, uint32_t seed) {
        const int W = 512, H = 1024;
        const float Wm = 0.4f, Hm = 0.8f;
        Maps m(W, H, Wm, Hm);
        forRows(H, [&](int y) {
            const float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(H);
            for (int x = 0; x < W; ++x) {
                const float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(W);
                const size_t c = m.at(x, y);
                V3 col;
                float hgt = 0, rough = 0.9f;
                if (kind == Furrowed) {
                    // long interlacing ridges between deep furrows (tulip poplar, ash, oak): two families of furrows
                    // running up the stem, each wandering its own way round, so they cross and part again in
                    // long diamonds
                    const float w1 = fbm2(u * 2.f, v * 3.f, 3, seed, 2, 3);
                    const float w2 = fbm2(u * 2.f + 3.7f, v * 3.f, 3, seed + 1, 2, 3);
                    const auto lineDist = [](float t) {
                        const float f = t - std::floor(t);
                        return std::min(f, 1.f - f);
                    };
                    const float N = 5.f;
                    const float d = std::min(lineDist(u * N + 0.9f * w1), lineDist(u * N + 0.9f * w2 + 0.5f));
                    const float plates = fbm2(u * 10, v * 6, 3, seed + 2, 10, 6);
                    const float width = 0.075f + 0.035f * plates;
                    const float furrow = smooth(width, 0.012f, d);
                    // cross-checks: short splits across the ridges
                    const float checks = smooth(0.1f, 0.f, std::abs(fbm2(u * 12.f, v * 9.f, 2, seed + 6, 12, 9))) * smooth(0.5f, 0.8f, plates + 0.5f);
                    const float ridge = (1.f - furrow) * (1.f - 0.6f * checks);
                    hgt = 0.012f * ridge + 0.002f * fbm2(u * 40, v * 20, 3, seed + 3, 40, 20);
                    const V3 furrowCol = srgb(50, 44, 38), plate = srgb(132, 126, 116);
                    col = mix(furrowCol, plate * (0.85f + 0.3f * clamp01(0.5f + plates)), smooth(0.f, 0.8f, ridge));
                    const float lichen = smooth(0.25f, 0.6f, fbm2(u * 5, v * 3, 3, seed + 4, 5, 3)) * ridge;
                    col = mix(col, srgb(150, 158, 138), 0.4f * lichen);
                    const float algae = smooth(0.1f, 0.6f, fbm2(u * 2, v * 1, 3, seed + 5, 2, 1));
                    col = mix(col, srgb(82, 98, 58), 0.3f * algae * ridge);
                } else if (kind == SmoothGrey) {
                    // smooth grey skin (beech, young maple): mottled, a few horizontal lenticels, lichen dots
                    const float mott = fbm2(u * 6, v * 3, 4, seed, 6, 3);
                    col = mix(srgb(122, 120, 112), srgb(150, 148, 138), clamp01(0.5f + mott));
                    const float patch = smooth(0.15f, 0.5f, fbm2(u * 3 + 0.3f, v * 1.5f, 3, seed + 1, 3, 2));
                    col = mix(col, srgb(92, 96, 86), 0.4f * patch);
                    const Cell le = worley2(u * 18, v * 70, seed + 2, 18, 70);
                    const float lent = smooth(0.25f, 0.1f, le.f1) * (unit(le.id) < 0.3f ? 1.f : 0.f);
                    col = mix(col, srgb(80, 76, 70), 0.5f * lent);
                    const Cell li = worley2(u * 10, v * 20, seed + 3, 10, 20);
                    const float lic = smooth(0.32f, 0.22f, li.f1 + 0.05f * perlin2(u * 80, v * 160, seed + 4, 80, 160)) * (unit(li.id) < 0.18f ? 1.f : 0.f);
                    col = mix(col, srgb(178, 182, 164), 0.6f * lic);
                    hgt = 0.0015f * mott - 0.0006f * lent + 0.0004f * lic;
                    rough = 0.75f;
                } else {
                    // birch: chalky white with dark horizontal lenticels and black scars, peeling strips
                    const float n = fbm2(u * 3, v * 6, 4, seed, 3, 6);
                    col = mix(srgb(206, 202, 190), srgb(226, 222, 212), clamp01(0.5f + n));
                    const Cell le = worley2(u * 5, v * 60, seed + 1, 5, 60, 0.8f);
                    const float lent = smooth(0.35f, 0.15f, std::abs(le.f1 - 0.15f) + 0.25f * std::abs(perlin2(u * 30, v * 300, seed + 2, 30, 300))) * (unit(le.id) < 0.45f ? 1.f : 0.f);
                    col = mix(col, srgb(52, 46, 40), 0.8f * lent);
                    const float scar = smooth(0.42f, 0.55f, fbm2(u * 4, v * 3, 4, seed + 3, 4, 3));
                    col = mix(col, srgb(40, 36, 32), 0.85f * scar);
                    const float peel = smooth(0.3f, 0.5f, fbm2(u * 8, v * 30, 3, seed + 4, 8, 30));
                    col = mix(col, srgb(196, 160, 130), 0.25f * peel);
                    hgt = 0.0008f * n - 0.0012f * lent - 0.002f * scar + 0.0006f * peel;
                    rough = 0.7f + 0.2f * scar;
                }
                m.albedo[c] = col * (0.92f + 0.16f * hash2f(x, y, seed + 9));
                m.height[c] = hgt;
                m.rough[c] = rough;
            }
        });
        m.bakeCavity(8, kind == Furrowed ? 0.005f : 0.001f, 0.35f);
        return m.finish(kind == Furrowed ? 0.7f : 0.5f);
    }

}// namespace mill::woods

#endif
