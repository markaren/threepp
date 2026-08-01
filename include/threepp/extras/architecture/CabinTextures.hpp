// Procedural material textures for the log-cabin generator.
//
// Everything is generated in-engine as DataTextures — no external image files,
// deterministic for a given seed — so a cabin can be dropped into any demo
// (GL or Vulkan) without shipping assets alongside it.
//
//   makeLogTextures()       — peeled/stained round-log side grain (albedo+normal).
//                             u runs ALONG the log axis, v wraps once around it.
//   makeLogEndTexture()     — end grain: growth rings + radial drying checks,
//                             for the sawn log ends that stick out at corners.
//   makeShingleTextures()   — asphalt-shingle roof: offset courses of tabs with
//                             a shadow line under each butt edge (albedo+normal).
//   makeSawnWoodTextures()  — planed timber for the deck, posts, rails, beams.
//   makeStoneTextures()     — rubble-stone foundation course.
//
// Every tiling texture wraps seamlessly: the lattice noise is sampled with an
// explicit period, and any cellular layout uses an integer cell count per tile.

#ifndef THREEPP_EXTRAS_ARCHITECTURE_CABINTEXTURES_HPP
#define THREEPP_EXTRAS_ARCHITECTURE_CABINTEXTURES_HPP

#include "threepp/extras/core/NoiseUtils.hpp"
#include "threepp/extras/core/TextureBake.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace threepp::architecture {

    namespace detail {

        using namespace threepp::noise;

        constexpr float PI_F = 3.14159265358979f;

        // Texture finishing + height→normal conversion live in the shared
        // extras/core/TextureBake.hpp (extracted verbatim from here).
        using texgen::finishTexture;
        using texgen::writeNormalFromHeight;

        inline void writeAlbedo(const std::shared_ptr<DataTexture>& tex, unsigned int size,
                                unsigned int x, unsigned int y,
                                float r, float g, float b, float a = 1.f) {
            auto& c = tex->image().data<unsigned char>();
            const size_t idx = (static_cast<size_t>(y) * size + x) * 4;
            c[idx + 0] = toByte(r);
            c[idx + 1] = toByte(g);
            c[idx + 2] = toByte(b);
            c[idx + 3] = toByte(a);
        }

    }// namespace detail

    // ── Round-log side grain ─────────────────────────────────────────────
    //
    // A peeled, stained log: long axial grain streaks, a handful of DRYING
    // CHECKS (the deep splits that open along the length as a log seasons —
    // the single most recognisable feature of a log wall up close) and the odd
    // branch knot.
    //
    // UV convention: u runs along the log axis and tiles; v wraps exactly once
    // around the circumference. Both are seamless.
    inline std::pair<std::shared_ptr<DataTexture>, std::shared_ptr<DataTexture>>
    makeLogTextures(unsigned int size = 512,
                    unsigned int seed = 7,
                    const std::array<float, 3>& baseColor = {0.58f, 0.36f, 0.19f}) {

        using namespace detail;

        auto albedo = DataTexture::create(4, size, size);
        auto normal = DataTexture::create(4, size, size);

        constexpr int P = 8;// noise lattice period

        // Grain: streaks running along u → low frequency in u, high in v.
        auto grainAt = [&](float u, float v) {
            const float g1 = valueNoise(u * 2.f, v * static_cast<float>(P) * 3.f, P * 3, seed + 1u);
            const float g2 = valueNoise(u * 4.f, v * static_cast<float>(P) * 8.f, P * 8, seed + 2u);
            const float g3 = valueNoise(u * static_cast<float>(P), v * static_cast<float>(P) * 2.f, P * 2, seed + 3u);
            return g1 * 0.55f + g2 * 0.25f + g3 * 0.20f;
        };

        // Drying checks: `nChecks` axial splits at hashed v positions. Each one
        // narrows/fades along u so it reads as a split that opens and closes,
        // not a painted-on stripe. Returns 1 deep in the split, 0 outside.
        constexpr int nChecks = 5;
        auto checkAt = [&](float u, float v) {
            float deepest = 0.f;
            for (int i = 0; i < nChecks; ++i) {
                const float vc = hash1(i, seed + 41u);
                // Wrapped distance in v (the circumference is periodic).
                float dv = std::abs(v - vc);
                dv = std::min(dv, 1.f - dv);
                // Half-width breathes along u; the modulation itself must tile.
                const float m = valueNoise(u * 3.f + static_cast<float>(i) * 7.f, 0.5f, 3, seed + 53u + static_cast<unsigned>(i));
                const float halfW = (0.004f + 0.012f * m) * (0.4f + hash1(i, seed + 59u));
                if (dv > halfW) continue;
                const float profile = 1.f - dv / halfW;
                deepest = std::max(deepest, profile * profile * (0.35f + 0.65f * m));
            }
            return deepest;
        };

        // Knots: a couple of oval scars with concentric rings. Deliberately
        // few and low-contrast — the tile repeats several times along every
        // log, and a bold knot turns that repeat into a visible grid of
        // identical dark eyes marching along the wall.
        constexpr int nKnots = 2;
        auto knotAt = [&](float u, float v, float& ring) {
            ring = 0.f;
            float best = 0.f;
            for (int i = 0; i < nKnots; ++i) {
                const float ku = hash2(i, 1, seed + 71u);
                const float kv = hash2(i, 2, seed + 71u);
                const float kr = 0.026f + 0.030f * hash2(i, 3, seed + 71u);
                float du = std::abs(u - ku);
                du = std::min(du, 1.f - du);
                float dv = std::abs(v - kv);
                dv = std::min(dv, 1.f - dv);
                // Knots are stretched along the grain (u).
                const float d = std::sqrt((du / 1.7f) * (du / 1.7f) + dv * dv) / kr;
                if (d >= 1.f) continue;
                const float w = 1.f - d;
                if (w > best) {
                    best = w;
                    ring = 0.5f + 0.5f * std::sin(d * 26.f);
                }
            }
            return best;
        };

        auto heightAt = [&](float u, float v) {
            const float g = grainAt(u, v);
            const float chk = checkAt(u, v);
            float ring = 0.f;
            const float k = knotAt(u, v, ring);
            float h = 0.35f + g * 0.55f;
            h += k * (0.10f + 0.12f * ring);// knots stand slightly proud
            h -= chk * 0.85f;               // checks cut in deep
            return std::clamp(h, 0.f, 1.f);
        };

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(size);
                const float v = static_cast<float>(y) / static_cast<float>(size);

                const float g = grainAt(u, v);
                const float chk = checkAt(u, v);
                float ring = 0.f;
                const float k = knotAt(u, v, ring);

                // Stain tone: grain modulates lightness; the late-wood streaks
                // are both darker AND more saturated, which is what makes a
                // stained softwood read as wood rather than as brown noise.
                const float t = 0.72f + g * 0.52f;
                float r = baseColor[0] * t;
                float gr = baseColor[1] * (t * 0.98f);
                float b = baseColor[2] * (t * 0.94f);

                // Knot: dark heart with rings around it.
                if (k > 0.f) {
                    const float kk = std::clamp(k * 1.4f, 0.f, 1.f);
                    const float shade = 0.62f + 0.26f * ring;
                    r = lerp(r, baseColor[0] * shade, kk);
                    gr = lerp(gr, baseColor[1] * shade * 0.92f, kk);
                    b = lerp(b, baseColor[2] * shade * 0.85f, kk);
                }

                // Check: near-black at the bottom of the split.
                if (chk > 0.f) {
                    const float c = std::clamp(chk, 0.f, 1.f);
                    r = lerp(r, 0.055f, c);
                    gr = lerp(gr, 0.035f, c);
                    b = lerp(b, 0.022f, c);
                }

                writeAlbedo(albedo, size, x, y, r, gr, b);
            }
        }

        writeNormalFromHeight(normal, size, 3.4f, heightAt, true);
        finishTexture(albedo, true, true);
        finishTexture(normal, false, true);
        return {albedo, normal};
    }

    // ── Sawn log end grain ───────────────────────────────────────────────
    //
    // Mapped onto the disc that caps a protruding log end at a corner notch.
    // UV convention: the disc fills the unit square (centre at 0.5,0.5,
    // radius 0.5), so the square's corners are never sampled.
    inline std::pair<std::shared_ptr<DataTexture>, std::shared_ptr<DataTexture>>
    makeLogEndTextures(unsigned int size = 256,
                       unsigned int seed = 7,
                       const std::array<float, 3>& baseColor = {0.72f, 0.55f, 0.33f}) {

        using namespace detail;

        auto albedo = DataTexture::create(4, size, size);
        auto normal = DataTexture::create(4, size, size);

        // Pith (ring centre) is offset from the geometric centre — a log is
        // never sawn dead through its own heart.
        const float px = 0.5f + (hash1(0, seed + 91u) - 0.5f) * 0.16f;
        const float py = 0.5f + (hash1(1, seed + 91u) - 0.5f) * 0.16f;
        const float ringCount = 16.f + 12.f * hash1(2, seed + 91u);

        auto fields = [&](float u, float v, float& rings, float& check, float& radial) {
            const float dx = u - px, dy = v - py;
            const float d = std::sqrt(dx * dx + dy * dy);
            const float a = std::atan2(dy, dx);
            // Rings wobble: growth rings are not perfect circles.
            const float wob = 0.012f * std::sin(a * 3.f + hash1(3, seed) * 6.28f) +
                              0.008f * std::sin(a * 7.f + hash1(4, seed) * 6.28f);
            rings = 0.5f + 0.5f * std::sin((d + wob) * ringCount * 6.28318f);
            // Radial drying checks running out from the pith.
            //
            // The angular difference MUST be reduced mod 2pi before the
            // shortest-arc fold. atan2 returns [-pi,pi] while `ac` is [0,2pi],
            // so |a - ac| reaches 3pi; `2pi - |a-ac|` is then NEGATIVE, the
            // `da > halfWidth` reject silently passes, and (1 - da/halfWidth)
            // evaluates to a large positive weight over a whole angular
            // sector. That weight goes straight into a lerp toward near-black,
            // overshoots past 0, and clamps — painting a solid black wedge
            // across every sawn log end in the building. Same shape of defect
            // as an out-of-domain pow(): the bad value defeats the guard that
            // was supposed to catch it, so clamp the weight as well.
            radial = 0.f;
            for (int i = 0; i < 3; ++i) {
                const float ac = hash1(i, seed + 101u) * 6.28318f;
                float da = std::fmod(std::abs(a - ac), 6.28318f);
                da = std::min(da, 6.28318f - da);
                constexpr float halfW = 0.05f;
                const float reach = 0.22f + 0.24f * hash1(i, seed + 103u);
                if (d > reach || da > halfW) continue;
                radial = std::max(radial, std::clamp((1.f - da / halfW) * (1.f - d / reach), 0.f, 1.f));
            }
            check = std::clamp(radial, 0.f, 1.f);
            return d;
        };

        auto heightAt = [&](float u, float v) {
            float rings, check, radial;
            const float d = fields(u, v, rings, check, radial);
            // Saw marks: faint parallel scoring across the whole face.
            const float saw = valueNoise(u * 40.f, v * 3.f, 40, seed + 111u);
            float h = 0.55f + rings * 0.22f + saw * 0.10f;
            h -= check * 0.9f;
            if (d > 0.47f) h -= (d - 0.47f) * 6.f;// slight chamfer at the rim
            return std::clamp(h, 0.f, 1.f);
        };

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(size);
                const float v = static_cast<float>(y) / static_cast<float>(size);
                float rings, check, radial;
                const float d = fields(u, v, rings, check, radial);

                // Sapwood (outer band) is paler than the heartwood core.
                const float sap = smooth(std::clamp((d - 0.30f) / 0.16f, 0.f, 1.f));
                // Softer ring contrast — sawn ends face outward and catch a lot
                // of sky, so a strong ramp turns every notch into a bullseye.
                const float tone = (0.86f + rings * 0.20f) * (0.92f + sap * 0.16f);
                float r = baseColor[0] * tone;
                float g = baseColor[1] * tone * (1.f - 0.05f * (1.f - sap));
                float b = baseColor[2] * tone * (1.f - 0.10f * (1.f - sap));

                if (check > 0.f) {
                    r = lerp(r, 0.06f, check);
                    g = lerp(g, 0.04f, check);
                    b = lerp(b, 0.03f, check);
                }
                writeAlbedo(albedo, size, x, y, r, g, b);
            }
        }

        writeNormalFromHeight(normal, size, 2.6f, heightAt, false);
        finishTexture(albedo, true, false);
        finishTexture(normal, false, false);
        return {albedo, normal};
    }

    // ── Asphalt shingle roof ─────────────────────────────────────────────
    //
    // UV convention: u runs horizontally across the roof, v runs UP the slope.
    // Courses are horizontal bands; each course is offset half a tab from the
    // one below, so the pattern repeats over TWO courses — `courses` must stay
    // even for the tile to wrap.
    inline std::pair<std::shared_ptr<DataTexture>, std::shared_ptr<DataTexture>>
    makeShingleTextures(unsigned int size = 512,
                        unsigned int seed = 7,
                        const std::array<float, 3>& baseColor = {0.31f, 0.22f, 0.155f}) {

        using namespace detail;

        auto albedo = DataTexture::create(4, size, size);
        auto normal = DataTexture::create(4, size, size);

        constexpr int courses = 6; // must be even (half-tab offset per course)
        constexpr int tabs = 4;    // tabs per tile, per course
        constexpr float slotHalf = 0.012f;// half-width of the vertical keyway, in u

        // Which tab a texel belongs to, plus its local coordinates.
        auto tabAt = [&](float u, float v, int& row, int& col, float& fu, float& fv) {
            const float cv = v * static_cast<float>(courses);
            row = static_cast<int>(std::floor(cv));
            fv = cv - static_cast<float>(row);
            const float offset = (row % 2 == 0) ? 0.f : 0.5f;
            const float cu = (u + offset / static_cast<float>(tabs)) * static_cast<float>(tabs);
            col = static_cast<int>(std::floor(cu));
            fu = cu - static_cast<float>(col);
        };

        auto heightAt = [&](float u, float v) {
            int row, col;
            float fu, fv;
            tabAt(u, v, row, col, fu, fv);
            // Butt edge: the bottom of a course sits proud of the course below,
            // so it casts a hard shadow line. Height ramps up quickly from the
            // butt then stays flat to the top of the exposure.
            float h = 0.32f + 0.55f * smooth(std::clamp(fv / 0.14f, 0.f, 1.f));
            // Keyway slot between tabs cuts right through to the course below.
            const float dslot = std::min(fu, 1.f - fu);
            if (dslot < slotHalf) h -= 0.55f * (1.f - dslot / slotHalf);
            // Granule tooth.
            h += (valueNoise(u * 96.f, v * 96.f, 96, seed + 5u) - 0.5f) * 0.22f;
            // Gentle per-tab cupping so the roof is not a dead flat plane.
            h += (hash2(col, row, seed + 9u) - 0.5f) * 0.10f;
            return std::clamp(h, 0.f, 1.f);
        };

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(size);
                const float v = static_cast<float>(y) / static_cast<float>(size);
                int row, col;
                float fu, fv;
                tabAt(u, v, row, col, fu, fv);

                // Per-tab tonal blend — architectural shingles are deliberately
                // multi-toned so a roof never reads as one flat colour. Keep
                // the spread NARROW: at ±25% adjacent tabs read as a
                // black-and-tan checkerboard from any distance, which is the
                // single loudest tell of a procedural roof.
                const float tabTone = 0.93f + 0.13f * hash2(col, row, seed + 13u);
                // Granule speckle at texel scale.
                const float grit = valueNoise(u * 96.f, v * 96.f, 96, seed + 5u);
                const float blotch = fbm(u * 6.f, v * 6.f, 6, seed + 17u, 3);
                float tone = tabTone * (0.86f + 0.28f * grit) * (0.86f + 0.28f * blotch);

                // Shadow under the butt edge of each course.
                tone *= 0.35f + 0.65f * smooth(std::clamp(fv / 0.12f, 0.f, 1.f));
                // Darkened keyway.
                const float dslot = std::min(fu, 1.f - fu);
                if (dslot < slotHalf) tone *= 0.35f + 0.65f * (dslot / slotHalf);

                writeAlbedo(albedo, size, x, y,
                            baseColor[0] * tone,
                            baseColor[1] * tone,
                            baseColor[2] * tone);
            }
        }

        writeNormalFromHeight(normal, size, 2.2f, heightAt, true);
        finishTexture(albedo, true, true);
        finishTexture(normal, false, true);
        return {albedo, normal};
    }

    // ── Planed / sawn timber ─────────────────────────────────────────────
    //
    // Deck boards, posts, rails, beams. u runs along the board.
    inline std::pair<std::shared_ptr<DataTexture>, std::shared_ptr<DataTexture>>
    makeSawnWoodTextures(unsigned int size = 512,
                         unsigned int seed = 7,
                         const std::array<float, 3>& baseColor = {0.60f, 0.42f, 0.25f}) {

        using namespace detail;

        auto albedo = DataTexture::create(4, size, size);
        auto normal = DataTexture::create(4, size, size);

        constexpr int P = 8;

        // Flat-sawn "cathedral" figure: bands whose spacing varies across the
        // width, warped slightly along the length.
        auto figureAt = [&](float u, float v) {
            const float warp = (valueNoise(u * 2.f, v * 2.f, 2, seed + 3u) - 0.5f) * 0.10f;
            const float vv = v + warp;
            const float bands = 0.5f + 0.5f * std::sin(vv * 34.f + std::sin(u * 4.f) * 1.4f);
            const float fine = valueNoise(u * 3.f, vv * static_cast<float>(P) * 9.f, P * 9, seed + 7u);
            const float grit = valueNoise(u * static_cast<float>(P) * 6.f, vv * static_cast<float>(P) * 6.f, P * 6, seed + 11u);
            return std::clamp(bands * 0.45f + fine * 0.35f + grit * 0.20f, 0.f, 1.f);
        };

        auto heightAt = [&](float u, float v) {
            // Weathered decking is slightly raised-grain: the hard late-wood
            // bands stand proud of the soft early wood.
            return std::clamp(0.35f + figureAt(u, v) * 0.55f, 0.f, 1.f);
        };

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(size);
                const float v = static_cast<float>(y) / static_cast<float>(size);
                const float f = figureAt(u, v);
                const float tone = 0.70f + f * 0.55f;
                writeAlbedo(albedo, size, x, y,
                            baseColor[0] * tone,
                            baseColor[1] * tone * 0.99f,
                            baseColor[2] * tone * 0.96f);
            }
        }

        writeNormalFromHeight(normal, size, 1.6f, heightAt, true);
        finishTexture(albedo, true, true);
        finishTexture(normal, false, true);
        return {albedo, normal};
    }

    // ── Old cabin glass ──────────────────────────────────────────────────
    //
    // Returns {normal, roughness}. Cabin glazing is not optical flat: it is
    // drawn or cylinder glass with slow ripples and faint draw lines, plus a
    // film of dust and rain spotting. A mirror-smooth pane samples the sky as
    // one clean gradient and reads as painted plastic; a little surface break-
    // up is what makes it read as glass at all.
    //
    // Amplitudes are deliberately tiny — this must perturb a reflection, not
    // look like frosted bathroom glass.
    inline std::pair<std::shared_ptr<DataTexture>, std::shared_ptr<DataTexture>>
    makeGlassTextures(unsigned int size = 256, unsigned int seed = 7) {

        using namespace detail;

        auto normal = DataTexture::create(4, size, size);
        auto rough = DataTexture::create(4, size, size);

        constexpr int P = 4;

        // Slow rolling waviness + faint vertical draw lines from the float bath.
        auto heightAt = [&](float u, float v) {
            const float roll = fbm(u * static_cast<float>(P), v * static_cast<float>(P), P, seed + 61u, 3);
            const float draw = valueNoise(u * static_cast<float>(P) * 6.f, v * 1.f, P * 6, seed + 67u);
            return std::clamp(roll * 0.72f + draw * 0.28f, 0.f, 1.f);
        };

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(size);
                const float v = static_cast<float>(y) / static_cast<float>(size);
                // Grime: heavier toward the pane edges, plus scattered spotting.
                const float grime = fbm(u * 5.f, v * 5.f, 5, seed + 71u, 3);
                const float spots = valueNoise(u * 26.f, v * 26.f, 26, seed + 73u);
                // Bounded well short of matte: this is weathered glazing, not
                // frosted bathroom glass. Above ~0.45 the pane stops returning
                // a recognisable reflection and reads as grey card.
                const float g = std::clamp(0.11f + 0.20f * grime + 0.16f * std::max(0.f, spots - 0.62f) * 3.f,
                                           0.06f, 0.44f);
                // MeshStandardMaterial reads roughness from the GREEN channel.
                writeAlbedo(rough, size, x, y, g, g, g);
            }
        }

        writeNormalFromHeight(normal, size, 0.55f, heightAt, true);
        finishTexture(normal, false, true);
        finishTexture(rough, false, true);
        return {normal, rough};
    }

    // ── Rubble-stone foundation ──────────────────────────────────────────
    inline std::pair<std::shared_ptr<DataTexture>, std::shared_ptr<DataTexture>>
    makeStoneTextures(unsigned int size = 512,
                      unsigned int seed = 7,
                      const std::array<float, 3>& baseColor = {0.42f, 0.40f, 0.37f}) {

        using namespace detail;

        auto albedo = DataTexture::create(4, size, size);
        auto normal = DataTexture::create(4, size, size);

        constexpr int cellsU = 6;
        constexpr int cellsV = 4;

        // Jittered-lattice cellular field: distance to the nearest site and to
        // the second nearest. (d2 - d1) is small in the joint between stones
        // and large at a stone's centre, which is exactly the mortar mask.
        auto cellular = [&](float u, float v, float& d1, float& d2, int& id) {
            d1 = d2 = 1e9f;
            id = 0;
            const float cu = u * cellsU;
            const float cv = v * cellsV;
            const int bu = static_cast<int>(std::floor(cu));
            const int bv = static_cast<int>(std::floor(cv));
            for (int dv = -1; dv <= 1; ++dv) {
                for (int du = -1; du <= 1; ++du) {
                    const int gu = bu + du;
                    const int gv = bv + dv;
                    const int wu = ((gu % cellsU) + cellsU) % cellsU;
                    const int wv = ((gv % cellsV) + cellsV) % cellsV;
                    const float su = static_cast<float>(gu) + 0.18f + 0.64f * hash2(wu, wv, seed + 21u);
                    const float sv = static_cast<float>(gv) + 0.18f + 0.64f * hash2(wu, wv, seed + 23u);
                    // Warp the metric per stone so they are not all round.
                    const float ddu = (cu - su) * (0.8f + 0.5f * hash2(wu, wv, seed + 25u));
                    const float ddv = (cv - sv) * 1.25f;
                    const float d = std::sqrt(ddu * ddu + ddv * ddv);
                    if (d < d1) {
                        d2 = d1;
                        d1 = d;
                        id = wv * cellsU + wu;
                    } else if (d < d2) {
                        d2 = d;
                    }
                }
            }
        };

        auto heightAt = [&](float u, float v) {
            float d1, d2;
            int id;
            cellular(u, v, d1, d2, id);
            const float joint = smooth(std::clamp((d2 - d1) / 0.16f, 0.f, 1.f));
            const float rough = fbm(u * 24.f, v * 24.f, 24, seed + 27u, 3);
            return std::clamp(joint * 0.80f + rough * 0.22f, 0.f, 1.f);
        };

        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                const float u = static_cast<float>(x) / static_cast<float>(size);
                const float v = static_cast<float>(y) / static_cast<float>(size);
                float d1, d2;
                int id;
                cellular(u, v, d1, d2, id);
                const float joint = smooth(std::clamp((d2 - d1) / 0.16f, 0.f, 1.f));
                const float rough = fbm(u * 24.f, v * 24.f, 24, seed + 27u, 3);

                // Per-stone colour drift (granite/gneiss greys, some warm).
                // Narrow: a wide spread turns a foundation into a chequerboard
                // of light and dark pebbles visible from across the valley.
                const float sh = hash1(id, seed + 31u);
                float r = baseColor[0] * (0.86f + 0.27f * sh);
                float g = baseColor[1] * (0.87f + 0.25f * sh);
                float b = baseColor[2] * (0.88f + 0.23f * sh);
                const float warm = hash1(id, seed + 33u);
                r *= 0.92f + 0.16f * warm;
                b *= 1.06f - 0.14f * warm;

                const float speck = 0.88f + 0.22f * rough;
                r *= speck;
                g *= speck;
                b *= speck;

                // Mortar: recessed and grey-brown, NOT bright. A pale joint
                // against dark stone turns a foundation into a cartoon
                // cobble pattern that reads from a hundred metres away.
                const float mortar = 1.f - joint;
                const float m = smooth(std::clamp(mortar * 1.35f - 0.30f, 0.f, 1.f));
                r = lerp(r, 0.255f, m);
                g = lerp(g, 0.248f, m);
                b = lerp(b, 0.236f, m);
                const float shade = 0.78f + 0.22f * joint;
                writeAlbedo(albedo, size, x, y, r * shade, g * shade, b * shade);
            }
        }

        writeNormalFromHeight(normal, size, 1.7f, heightAt, true);
        finishTexture(albedo, true, true);
        finishTexture(normal, false, true);
        return {albedo, normal};
    }

}// namespace threepp::architecture

#endif//THREEPP_EXTRAS_ARCHITECTURE_CABINTEXTURES_HPP
