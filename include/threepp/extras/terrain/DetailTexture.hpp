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

#include "threepp/math/MathUtils.hpp"
#include "threepp/math/Rng.hpp"
#include "threepp/extras/core/TextureBake.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
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

    // ── Structured per-band material sets ────────────────────────────────────
    // makeDetailMaps' value-noise field breaks up flat colour, but it has no
    // STRUCTURE — the same isotropic mush lands on snow, rock and grass, which
    // is precisely what reads as "90s terrain". These generators build a
    // heightfield with material-specific structure (Voronoi plates + cracks,
    // packed pebbles, grass clumps, wind-rippled snow) and derive a coherent
    // albedo-overlay + normal + roughness set from it, exactly like
    // makeDetailMaps. Differences from the generic set:
    //   • albedo ALPHA carries the material HEIGHT (0..1) — the terrain shader
    //     blends bands by height (tufts interleave into gravel at a boundary
    //     instead of cross-fading), so A is data, not opacity;
    //   • roughness is shaped per material (crack dust rough, pebble tops
    //     smoother, snow near-uniform with sparkle glints).
    // Everything is periodic (wrapping lattices), LINEAR, 0.5-neutral in RGB.
    // Generator kinds. 0..3 double as the standard band-slot order; Cliff is an
    // ALTERNATIVE generator for slot 1 (rock) — a gneiss wall rather than the
    // generic fractured-plate rock — so it takes an index past the four slots.
    enum class BandKind { Grass = 0,
                          Rock = 1,
                          Scree = 2,
                          Snow = 3,
                          Cliff = 4 };

    namespace banddetail {

        // Wrapped-grid Voronoi: one jittered feature point per cell, distances
        // computed with wrapped deltas so the pattern tiles. Returns F1, F2 and
        // the winning cell's hash (for per-feature tint/size variation).
        struct VoronoiSample {
            float f1, f2;
            float id;// [0,1) hash of the nearest feature's cell
        };

        inline float bandHash(int x, int y, unsigned int seed) {
            unsigned int n = static_cast<unsigned int>(x) * 374761393u +
                             static_cast<unsigned int>(y) * 668265263u + seed * 2654435761u;
            n = (n ^ (n >> 13)) * 1274126177u;
            return static_cast<float>((n ^ (n >> 16)) & 0xffffffu) / static_cast<float>(0xffffff);
        }

        inline VoronoiSample voronoi(float u, float v, int cells, unsigned int seed) {
            const float fu = u * static_cast<float>(cells);
            const float fv = v * static_cast<float>(cells);
            const int iu = static_cast<int>(std::floor(fu));
            const int iv = static_cast<int>(std::floor(fv));
            VoronoiSample s{1e30f, 1e30f, 0.f};
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    const int cx = iu + dx, cy = iv + dy;
                    const int wx = ((cx % cells) + cells) % cells;// wrap → tiles
                    const int wy = ((cy % cells) + cells) % cells;
                    const float px = static_cast<float>(cx) + bandHash(wx, wy, seed);
                    const float py = static_cast<float>(cy) + bandHash(wx, wy, seed + 1u);
                    const float ddx = fu - px, ddy = fv - py;
                    const float d = std::sqrt(ddx * ddx + ddy * ddy);
                    if (d < s.f1) {
                        s.f2 = s.f1;
                        s.f1 = d;
                        s.id = bandHash(wx, wy, seed + 2u);
                    } else if (d < s.f2) {
                        s.f2 = d;
                    }
                }
            return s;
        }

    }// namespace banddetail

    // One structured band set. `o.dim`/`o.seed` are honoured; the contrast /
    // normal / roughness knobs default per kind when left at their generic
    // defaults (pass explicit values to override).
    inline DetailMaps makeBandMaps(BandKind kind, DetailMapOptions o = {}) {
        const int D = std::max(o.dim, 8);
        const unsigned int seed = o.seed + static_cast<unsigned int>(kind) * 7919u;

        // Generic periodic value noise (same as makeDetailMaps, local seed).
        // Draw ORDER matters: the l8/l24/l8g fills consume the rng stream in
        // sequence, so reordering them reshuffles every texel.
        math::Rng rng(seed);
        auto lattice = [&](int cells) { return texgen::noiseLattice(rng, cells); };
        const auto& sampleLat = texgen::sampleLattice;
        const auto l8 = lattice(8), l24 = lattice(24), l8g = lattice(8);

        // Per-kind defaults (only where the caller left the generic ones).
        // Chroma is deliberately TINY for rock/scree/snow: the wobble is
        // applied per Voronoi FEATURE, and at generic-detail chroma (0.06) the
        // per-plate green/magenta tints turn a mid-distance rock face into
        // purple "reptile scales" (observed on the aalesund hillside). Mineral
        // surfaces vary in luminance, barely in hue; grass keeps the hue drift.
        float albedoContrast = o.albedoContrast, normalStrength = o.normalStrength,
              roughContrast = o.roughContrast, chroma = o.chroma;
        const DetailMapOptions defaults{};
        const bool defC = albedoContrast == defaults.albedoContrast;
        const bool defN = normalStrength == defaults.normalStrength;
        const bool defR = roughContrast == defaults.roughContrast;
        const bool defH = chroma == defaults.chroma;
        // Normal strengths run HOT on purpose: perceived depth at these
        // metre-scale periods comes almost entirely from the relief lighting +
        // roughness breakup in the crevices — the albedo overlay alone reads
        // flat. (The shader's detailNormalScale multiplies on top.)
        switch (kind) {
            case BandKind::Grass:
                if (defC) albedoContrast = 0.40f;
                if (defN) normalStrength = 4.5f;
                if (defR) roughContrast = 0.35f;
                break;// keeps the generic chroma — hue drift is what grass wants
            case BandKind::Rock:
                if (defC) albedoContrast = 0.30f;
                if (defN) normalStrength = 6.0f;
                if (defR) roughContrast = 0.55f;
                if (defH) chroma = 0.015f;
                break;
            case BandKind::Scree:
                if (defC) albedoContrast = 0.42f;
                if (defN) normalStrength = 6.0f;
                if (defR) roughContrast = 0.45f;
                if (defH) chroma = 0.03f;
                break;
            case BandKind::Cliff:
                // Gneiss reads through LUMINANCE structure, not hue: chroma
                // stays below the 0.02 "reptile scale" threshold that turned
                // the generic rock band purple at mid distance. The extra
                // roughness contrast is what separates wet veins from dry face.
                if (defC) albedoContrast = 0.55f;// wall-scale bands must read at 1 km, not whisper
                if (defN) normalStrength = 6.5f;
                if (defR) roughContrast = 0.60f;
                if (defH) chroma = 0.012f;
                break;
            case BandKind::Snow:
                // NOT near-flat: a flat snow band reads as untextured plaster
                // wherever the snowline transition assigns it partial weight
                // (observed on the fjord's mid-flank — the "flat texture"
                // report). Real snow carries sastrugi ripples and grain.
                if (defC) albedoContrast = 0.20f;
                if (defN) normalStrength = 2.8f;
                if (defR) roughContrast = 0.30f;
                if (defH) chroma = 0.01f;
                break;
        }

        // ── height + per-texel tint/rough shaping ────────────────────────────
        std::vector<float> hf(static_cast<size_t>(D) * D);
        std::vector<float> tint(hf.size());  // per-texel id/tint driver [0,1]
        std::vector<float> roughF(hf.size());// roughness before contrast map
        for (int j = 0; j < D; ++j)
            for (int i = 0; i < D; ++i) {
                const float u = (static_cast<float>(i) + 0.5f) / D;
                const float v = (static_cast<float>(j) + 0.5f) / D;
                const size_t idx = static_cast<size_t>(j) * D + i;
                float h = 0.5f, id = 0.5f, r = 0.5f;
                switch (kind) {
                    case BandKind::Rock: {
                        // Fractured plates: F2-F1 pinches to 0 along crack lines.
                        const auto s = banddetail::voronoi(u, v, 6, seed);
                        const float crack = 1.f - math::smoothstep(0.f, 0.16f, s.f2 - s.f1);
                        const float plateH = 0.55f + 0.35f * s.id;// per-plate level
                        const float micro = (sampleLat(l24, 24, u, v) - 0.5f) * 0.18f;
                        // A second, finer fracture set breaks big plates up.
                        const auto s2 = banddetail::voronoi(u, v, 13, seed + 31u);
                        const float crack2 = 1.f - math::smoothstep(0.f, 0.10f, s2.f2 - s2.f1);
                        h = std::clamp(plateH + micro - crack * 0.34f - crack2 * 0.12f, 0.f, 1.f);
                        id = s.id;
                        r = 0.5f + 0.5f * std::max(crack, crack2 * 0.6f)// crack dust = rough
                            - 0.15f * (h - 0.5f);                      // worn tops slightly smoother
                        break;
                    }
                    case BandKind::Cliff: {
                        // Gneiss wall. Three structures, in the order they read
                        // from 600 m out:
                        //  1. FOLIATION — the metamorphic banding. Stretched
                        //     along texture V (high frequency in u, ~flat in
                        //     v). The Vulkan terrain path projects side faces
                        //     with V along world Y, so these bands hang
                        //     VERTICALLY down a wall. That single cue is what
                        //     separates "cliff" from "grey noise".
                        //  2. JOINT BLOCKS — a coarse Voronoi whose F2−F1
                        //     pinch cuts the fracture planes between columns.
                        //  3. WET VEINS — narrow, near-vertical dark streaks
                        //     with a big roughness DROP: seepage lines. The
                        //     flow-accumulation paint places wet regions; this
                        //     gives them their sub-metre structure and gloss.
                        // Pale weathering + rare lichen specks lift the mean so
                        // the face is not uniformly dark.
                        // The band's world period is 20 m (repeat 0.05/m, see
                        // makeTerrainBandSet), so the u-multipliers below ARE the
                        // world scales: ×2 → 10 m banding, ×5 → 4 m foliation,
                        // ×11 → 1.8 m grain, Voronoi 5 → 4 m joint blocks. A
                        // wall pixel is ~0.5 m from the shot camera, so every
                        // one of these survives the mip chain; the old 7 m
                        // period put the same features at 1.4 m and below,
                        // where the footprint averaged them into flat grey.
                        const auto s = banddetail::voronoi(u, v, 5, seed);
                        const float crack = 1.f - math::smoothstep(0.f, 0.13f, s.f2 - s.f1);
                        const float macro = sampleLat(l8, 8, u * 2.f, v * 0.10f);
                        const float foliation = sampleLat(l24, 24, u * 5.f, v * 0.30f);
                        const float fine = sampleLat(l24, 24, u * 11.f, v * 0.60f);
                        const float weather = sampleLat(l8, 8, u, v);
                        const float vein = math::smoothstep(
                                0.70f, 0.95f, sampleLat(l8g, 8, u * 3.f, v * 0.12f));
                        const float lichen = banddetail::bandHash(i, j, seed + 7u) > 0.988f ? 1.f : 0.f;
                        // Crack weight is the wall's whole legibility at 1 km:
                        // dropped to 0.18 the face went back to a smooth
                        // fabric-like smear, raised past ~0.4 the 4 m Voronoi
                        // cells read as a uniform craquelure net (both looked
                        // at, 1:1, 2026-09-04). 0.30 punctuates the banding.
                        h = std::clamp(0.52f + 0.34f * (macro - 0.5f) + 0.22f * (foliation - 0.5f) +
                                               0.07f * (fine - 0.5f) + 0.12f * (s.id - 0.5f) -
                                               0.30f * crack - 0.16f * vein + 0.16f * lichen,
                                       0.f, 1.f);
                        id = 0.30f * weather + 0.24f * foliation + 0.24f * macro + 0.22f * s.id;
                        r = 0.5f + 0.20f * crack + 0.18f * (weather - 0.5f) - 0.42f * vein -
                            0.20f * lichen;
                        break;
                    }
                    case BandKind::Grass: {
                        // Rounded tufts: inverted F1 of a dense point set. The
                        // clump field alone reads as smooth blur under FLAT
                        // light (no raking sun = no relief shading — observed
                        // on the fjord's fell slopes, a world-anchored pattern
                        // identical every run): the high-frequency terms below
                        // are what keep the band legible in pure luminance.
                        const auto s = banddetail::voronoi(u, v, 22, seed);
                        const float tuft = 1.f - math::smoothstep(0.20f, 0.75f, s.f1);
                        const float undul = sampleLat(l8, 8, u, v);
                        const float blade = sampleLat(l24, 24, u * 2.f, v * 6.f);// stretched → blade streaks
                        const float speck = banddetail::bandHash(i, j, seed + 13u);// per-texel litter/soil grain
                        h = std::clamp(0.28f + 0.42f * tuft + 0.16f * undul +
                                               0.22f * (blade - 0.5f) + 0.10f * (speck - 0.5f),
                                       0.f, 1.f);
                        id = s.id * 0.6f + 0.4f * undul;// patchy greener/strawier drift
                        r = 0.5f + 0.12f * (0.5f - tuft) + 0.08f * (speck - 0.5f);
                        break;
                    }
                    case BandKind::Scree: {
                        // Packed stones: per-cell radius bump, nearest-top wins.
                        const auto s = banddetail::voronoi(u, v, 15, seed);
                        const float stoneR = 0.45f + 0.35f * banddetail::bandHash(
                                                                  static_cast<int>(s.id * 8191.f), 17, seed + 5u);
                        const float stone = math::smoothstep(stoneR, stoneR * 0.15f, s.f1);// 1 at centre → 0 at rim
                        const float gapDust = sampleLat(l24, 24, u, v) * 0.12f;
                        h = std::clamp(0.18f + 0.72f * stone + gapDust, 0.f, 1.f);
                        id = s.id;
                        r = 0.5f + 0.35f * (1.f - stone);// crevice fines rougher than stone faces
                        break;
                    }
                    case BandKind::Snow: {
                        // Wind ripples (stretched noise) over a soft undulation,
                        // plus rare sparkle glints (facet crystals) and a fine
                        // surface grain so partial-coverage transition zones
                        // never collapse to plaster.
                        const float undul = sampleLat(l8, 8, u, v);
                        const float ripple = sampleLat(l24, 24, u * 3.f, v * 1.f);// wind-aligned
                        const float grain = banddetail::bandHash(i, j, seed + 11u);
                        const float glint = banddetail::bandHash(i, j, seed + 9u) > 0.995f ? 1.f : 0.f;
                        h = std::clamp(0.40f + 0.16f * undul + 0.16f * (ripple - 0.5f) +
                                               0.08f * (grain - 0.5f) + 0.25f * glint,
                                       0.f, 1.f);
                        id = undul;
                        r = 0.5f - 0.45f * glint + 0.06f * (ripple - 0.5f);// glints read glossy
                        break;
                    }
                }
                hf[idx] = h;
                tint[idx] = id;
                roughF[idx] = std::clamp(r, 0.02f, 0.98f);
            }

        const auto at = [D](int x, int y) {
            x = (x % D + D) % D;
            y = (y % D + D) % D;
            return static_cast<size_t>(y) * D + x;
        };

        std::vector<unsigned char> apx(static_cast<size_t>(D) * D * 4, 255u);
        std::vector<unsigned char> npx(static_cast<size_t>(D) * D * 4, 255u);
        for (int j = 0; j < D; ++j)
            for (int i = 0; i < D; ++i) {
                const size_t idx = at(i, j);
                const float n = hf[idx];

                // Albedo overlay: luminance from structure height, chroma from
                // the per-feature id (plate/stone/patch variation), A = height.
                const float g = chroma * (tint[idx] - 0.5f) * 2.f;
                const float base = 0.5f + albedoContrast * (n - 0.5f);
                const size_t ao = idx * 4;
                apx[ao + 0] = static_cast<unsigned char>(std::clamp(base - 0.5f * g, 0.f, 1.f) * 255.f + 0.5f);
                apx[ao + 1] = static_cast<unsigned char>(std::clamp(base + g, 0.f, 1.f) * 255.f + 0.5f);
                apx[ao + 2] = static_cast<unsigned char>(std::clamp(base - g, 0.f, 1.f) * 255.f + 0.5f);
                apx[ao + 3] = static_cast<unsigned char>(std::clamp(n, 0.f, 1.f) * 255.f + 0.5f);

                const float dhdx = (hf[at(i + 1, j)] - hf[at(i - 1, j)]) * 0.5f * normalStrength;
                const float dhdz = (hf[at(i, j + 1)] - hf[at(i, j - 1)]) * 0.5f * normalStrength;
                float nx = -dhdx, ny = -dhdz, nz = 1.f;
                const float il = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz);
                nx *= il;
                ny *= il;
                nz *= il;
                const float rough = std::clamp(0.5f + roughContrast * (roughF[idx] - 0.5f) * 2.f, 0.f, 1.f);
                const size_t no = idx * 4;
                npx[no + 0] = static_cast<unsigned char>((nx * 0.5f + 0.5f) * 255.f + 0.5f);
                npx[no + 1] = static_cast<unsigned char>((ny * 0.5f + 0.5f) * 255.f + 0.5f);
                npx[no + 2] = static_cast<unsigned char>((nz * 0.5f + 0.5f) * 255.f + 0.5f);
                npx[no + 3] = static_cast<unsigned char>(rough * 255.f + 0.5f);
            }

        auto mk = [D](std::vector<unsigned char> px) {
            return texgen::makeLinearRepeatTexture(std::move(px), static_cast<unsigned int>(D));
        };

        DetailMaps out;
        out.albedo = mk(std::move(apx));
        out.normalRough = mk(std::move(npx));
        return out;
    }

    // The standard 4-band terrain set, index order matching the terrain shader
    // convention (and SplatLayer::structureBand): 0 grass, 1 rock, 2 scree,
    // 3 snow.
    struct TerrainBandSet {
        std::array<DetailMaps, 4> band;
        // Suggested per-band world repeat (1/m) and base roughness — starting
        // points; demos override per look. Periods are METRES-scale on purpose:
        // grass ~2.9 m (13 cm tufts), rock ~7 m (1.2 m plates), scree ~2.5 m
        // (17 cm stones), snow ~4 m ripples. Finer repeats collapse the
        // structure into speckle noise (measured on the trollstigen close-up);
        // the stochastic tiling is what makes big periods safe from visible
        // repetition.
        std::array<float, 4> repeat{0.3f, 0.14f, 0.4f, 0.25f};
        std::array<float, 4> roughness{0.92f, 0.82f, 0.9f, 0.45f};
    };

    // `rockKind` swaps what fills band slot 1: Rock (generic fractured plates,
    // the default) or Cliff (gneiss foliation + joint blocks + wet veins). The
    // slot index, the repeat and the weight-map channel are unchanged, so a
    // caller flips the look of every steep face with one argument.
    inline TerrainBandSet makeTerrainBandSet(unsigned int seed = 4242u,
                                             BandKind rockKind = BandKind::Rock) {
        TerrainBandSet s;
        DetailMapOptions o;
        o.seed = seed;
        o.dim = 512;// metre-scale periods need cm-scale texels (cracks, blades)
        s.band[0] = makeBandMaps(BandKind::Grass, o);
        s.band[2] = makeBandMaps(BandKind::Scree, o);
        s.band[3] = makeBandMaps(BandKind::Snow, o);
        if (rockKind != BandKind::Cliff) {
            s.band[1] = makeBandMaps(rockKind, o);
        } else {
            // A WALL is the one surface whose structure is metres, not
            // centimetres: foliation banding, joint blocks and seepage veins
            // read at 1-20 m. The generic 7 m rock period puts all of that
            // below a distant wall pixel's footprint, which is exactly why the
            // band layer contributed nothing to a 1 km cliff crop (measured
            // 2026-09-04). 20 m period at 1024 px keeps 2 cm texels, so the
            // close-up is not softer than the generic rock band; stochastic
            // tiling is what makes the long period safe from visible repeats.
            DetailMapOptions co = o;
            co.dim = 1024;
            s.band[1] = makeBandMaps(BandKind::Cliff, co);
            s.repeat[1] = 0.05f;
        }
        return s;
    }

    inline DetailMaps makeDetailMaps(const DetailMapOptions& o = {}) {
        const int D = std::max(o.dim, 8);
        // Periodic (wrapping) bilinear value noise — guarantees the map tiles.
        // Draw ORDER matters (see makeBandMaps).
        math::Rng rng(o.seed);
        auto lattice = [&](int cells) { return texgen::noiseLattice(rng, cells); };
        const auto& sampleLat = texgen::sampleLattice;
        const auto l8 = lattice(8), l32 = lattice(32), l8g = lattice(8);

        // Shared heightfield in [0,1], periodic. Two octaves + fine speckle.
        std::vector<float> hf(static_cast<size_t>(D) * D);
        // Speckle stream forked off the seed so it cannot shift the lattice
        // draws above (draw ORDER matters — see the comment on rng).
        math::Rng srng = math::Rng(o.seed).fork(1);
        for (int j = 0; j < D; ++j)
            for (int i = 0; i < D; ++i) {
                const float u = (static_cast<float>(i) + 0.5f) / D, v = (static_cast<float>(j) + 0.5f) / D;
                const float n = 0.55f * sampleLat(l8, 8, u, v) + 0.30f * sampleLat(l32, 32, u, v) + 0.15f * srng.nextFloat();
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
            return texgen::makeLinearRepeatTexture(std::move(px), static_cast<unsigned int>(D));
        };

        DetailMaps out;
        out.albedo = mk(std::move(apx));
        out.normalRough = mk(std::move(npx));
        return out;
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_DETAILTEXTURE_HPP
