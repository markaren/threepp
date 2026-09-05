// Assemble a terrain::TerrainProvider from a real-world region pack + its road
// network. This is the glue that turns GeoTerrainPack (elevation grid + roads)
// into the two std::functions TileTerrain bakes from:
//
//   height(x,z) = grid.sampleBicubic(x,z)
//                 + detailRelief(x,z) · (1 − corridorWeight(x,z))
//
//     • the grid is CARVED at load (carveRoads below): every cell near a road
//       is clamped to sit below the conformed ribbon surface, so the road cut
//       is baked into the DEM itself — no runtime corridor warp. The carved
//       grid is band-limited at DEM resolution, so bicubic sampling is C1
//       everywhere and tile-LOD splits can never reveal sub-quad road features
//       ("humps popping in" as tiles refine);
//     • bicubic keeps the DEM C1-continuous (no per-cell creases up close);
//     • a small world-anchored fBm adds sub-grid relief the DEM's samples can't
//       hold, FADED OUT inside road corridors so the ribbon stays smooth.
//
//   albedo — a Norwegian-tuned terrain::SplatRules (wetland-dark near sea level,
//     valley grass/heath on gentle low ground, exposed rock/scree on steep
//     slopes, snow up high with a feathered line; macro variation + gentle AO).
//     No roadside tint: a corridor-wide swath reads as a phantom road wherever
//     roads run close (hairpins, dual carriageways) — the ribbon is the road.
//     Where the pack carries buildings, an URBAN ground paint blends the splat
//     toward asphalt/gravel town fabric under building-DENSE areas (see
//     UrbanMask below) — towns stop reading as houses scattered on a lawn.
//
// Both callbacks are pure and thread-safe (HeightGrid + RoadNetwork queries are
// read-only; the SplatRules is captured by value): safe for TileTerrain's async
// bake. The pack and the RoadNetwork must OUTLIVE the returned provider (the
// callbacks reference them); conformTo() must have run on the network first.
//
// Header-only, extras.

#ifndef THREEPP_EXTRAS_TERRAIN_GEOTERRAIN_HPP
#define THREEPP_EXTRAS_TERRAIN_GEOTERRAIN_HPP

#include "threepp/math/MathUtils.hpp"
#include "threepp/extras/road/RoadNetwork.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/extras/terrain/TerrainSplat.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace threepp::terrain {

    // Tunables for the geodata provider. Heights are absolute metres (the pack's
    // NN2000 datum); defaults suit a Norwegian fjord/mountain region.
    struct GeoTerrainOptions {
        // Sub-grid detail relief.
        float detailAmplitude = 0.4f;// metres of extra fBm relief on near tiles
        float detailFreq = 0.05f;    // 1/m base frequency (~20 m wavelength)

        // Splat band boundaries (absolute metres).
        float wetlandBand = 6.f;   // wetland-dark reaches this far above sea level
        float grassHeightMax = 700.f;
        float snowHeightMin = 1200.f;
        float snowFeather = 90.f;

        // BAKED-ROAD paint (the "terrain IS the road" pipeline — pair with
        // carveRoads bakeSurface + RoadNetwork::buildBridgeMeshes): mix asphalt
        // into the splat over the PAVED band only (RoadNetwork::pavedWeight —
        // narrow feather, so stacked hairpins never merge into a phantom swath).
        // A painted road is tile TEXTURE: mip/aniso filtering integrates it as
        // it recedes, so distant roads fade smoothly instead of shimmering the
        // way a sub-pixel ribbon's raster coverage does. Off by default —
        // legacy callers keep the ribbon-is-the-road look unchanged.
        bool paintRoads = false;
        float roadEdgeFeather = 0.8f;                       // paint feather past the pavement edge (m)
        std::array<float, 3> roadColor = {0.075f, 0.075f, 0.08f};// sRGB asphalt

        // URBAN ground paint (packs with buildings): where built COVERAGE is
        // dense, blend the splat toward two-tone asphalt/gravel town fabric,
        // suppress band structure (no grass clumps between houses) and flatten
        // the detail relief (town ground is graded). Density-gated — a lone
        // mountain cabin never earns a grey halo, a town block does — and
        // applied UNDER the road paint, so streets stay visible through town.
        bool paintUrban = true;      // no-op when the pack has no buildings
        float urbanCell = 4.f;       // mask raster cell (m)
        float urbanBlurRadius = 28.f;// coverage smoothing radius (m)
        float urbanCoverLo = 0.05f;  // built fraction where the paint starts
        float urbanCoverHi = 0.20f;  // built fraction of full paint
        float urbanMax = 0.85f;      // paint ceiling — gardens keep some ground tone
        std::array<float, 3> urbanAsphalt = {0.085f, 0.085f, 0.09f};// sRGB street/lot
        std::array<float, 3> urbanGravel = {0.185f, 0.175f, 0.155f};// sRGB yard/gravel

        // ── CLIFF relief (benches + joint sets) ─────────────────────────────
        // A smooth fBm gives ROUNDED slopes; real gneiss walls are stratified
        // (near-horizontal benches, because the rock is layered) and jointed
        // (two oblique fracture sets). Both are added in the height domain and
        // gated on DEM slope, so gentle ground is untouched.
        //
        // OFF by default, and callers should only enable it on ~1 m packs: on a
        // 2 m DEM a 0.9 m riser is finer than the data's own resolution, so it
        // would be inventing structure rather than sharpening measured
        // structure. Existing 2 m packs (trollstigen, aalesund) are therefore
        // bit-identical unless a caller opts in.
        bool cliffRelief = false;
        float cliffBenchAmp = 0.9f;    // riser height (m), peak-to-peak
        float cliffBenchPeriod = 4.5f; // vertical bench spacing (m), fBm-modulated
        float cliffJointAmp = 0.30f;   // joint groove depth (m)
        float cliffJointPeriod = 5.5f; // joint spacing (m)
        float cliffSlopeLo = 0.45f;    // relief fades in over this DEM-slope window
        float cliffSlopeHi = 0.62f;    // (slope = 1 − Ny, so 0.62 ≈ 68°)

        // ── LAND COVER from measured data (canopy CHM + flow accumulation) ──
        // No-op on packs that carry neither. Paints forest, wet seepage streaks,
        // scree fans and bench moss; road/urban paint is applied on top,
        // unchanged.
        bool landCover = true;
        float canopyForestMin = 2.f;// m of canopy where forest paint starts
        float canopyForestFull = 6.f;
        float wetFlowLo = 0.52f;    // log-normalised accumulation window for wet rock
        float wetFlowHi = 0.72f;
        float screeFlowLo = 0.48f;  // fans want drainage AND a relaxed slope
    };

    // ── FootprintMask: a hard yes/no raster of polygon coverage ─────────────
    //
    // `UrbanMask` below answers "how built is the land here" (blurred, soft);
    // this answers "is this exact point inside one of these polygons", which is
    // what a SITE gate needs. Trees, bushes and (phase D) cars all ask the same
    // question of different polygon sets: building footprints, piers and quays,
    // parking lots, pitches.
    //
    // Two extras over a plain point-in-polygon loop:
    //   * it is a raster, so a gate costs one lookup instead of a scan over
    //     8000 rings — the difference between a 200 ms site pass and a 3 minute
    //     one;
    //   * it DILATES. A DOM-derived canopy model and an OSM footprint disagree
    //     by metres (different sources, different epochs): on this pack 23.5% of
    //     canopy cells fall inside a raw footprint and 29.3% inside footprints
    //     grown by 4 m, and the difference is exactly the ring of "canopy" that
    //     is really a roof edge. Dilate by the registration error, not by taste.
    struct FootprintMask {
        int dim = 0;
        float cell = 2.f;
        float half = 0.f;
        std::vector<std::uint8_t> m;

        [[nodiscard]] bool valid() const { return dim > 1 && !m.empty(); }

        // Nearest cell — the raster IS the answer, no interpolation.
        [[nodiscard]] bool inside(float x, float z) const {
            if (dim < 2) return false;
            const int ix = static_cast<int>(std::lround((x + half) / cell));
            const int iz = static_cast<int>(std::lround((z + half) / cell));
            if (ix < 0 || iz < 0 || ix >= dim || iz >= dim) return false;
            return m[static_cast<size_t>(iz) * dim + ix] != 0u;
        }

        // Bilinear coverage in 0..1, for PAINT (a hard raster edge at 4 m reads
        // as a staircase in the albedo).
        [[nodiscard]] float coverage(float x, float z) const {
            if (dim < 2) return 0.f;
            const float fx = (x + half) / cell, fz = (z + half) / cell;
            const int ix = static_cast<int>(std::floor(fx)), iz = static_cast<int>(std::floor(fz));
            if (ix < 0 || iz < 0 || ix >= dim - 1 || iz >= dim - 1) return 0.f;
            const float tx = fx - static_cast<float>(ix), tz = fz - static_cast<float>(iz);
            const auto v = [&](int a, int b) {
                return static_cast<float>(m[static_cast<size_t>(b) * dim + a]);
            };
            const float a = v(ix, iz) + (v(ix + 1, iz) - v(ix, iz)) * tx;
            const float b = v(ix, iz + 1) + (v(ix + 1, iz + 1) - v(ix, iz + 1)) * tx;
            return a + (b - a) * tz;
        }
    };

    namespace detail {

        // Crossing-number rasterisation of ONE open ring into a uint8 grid.
        // Same test the UrbanMask occupancy pass uses; kept separate because
        // that one accumulates into a float field it then blurs.
        inline void geoRasterRing(FootprintMask& mask, const std::vector<Vector2>& ring,
                                  std::uint8_t value = 1u) {
            if (ring.size() < 3) return;
            const int dim = mask.dim;
            const float cell = mask.cell, half = mask.half;
            float minX = ring[0].x, maxX = ring[0].x, minZ = ring[0].y, maxZ = ring[0].y;
            for (const auto& p : ring) {
                minX = std::min(minX, p.x);
                maxX = std::max(maxX, p.x);
                minZ = std::min(minZ, p.y);
                maxZ = std::max(maxZ, p.y);
            }
            const int ix0 = std::max(0, static_cast<int>(std::floor((minX + half) / cell)));
            const int ix1 = std::min(dim - 1, static_cast<int>(std::ceil((maxX + half) / cell)));
            const int iz0 = std::max(0, static_cast<int>(std::floor((minZ + half) / cell)));
            const int iz1 = std::min(dim - 1, static_cast<int>(std::ceil((maxZ + half) / cell)));
            for (int iz = iz0; iz <= iz1; ++iz) {
                const float pz = -half + static_cast<float>(iz) * cell;
                for (int ix = ix0; ix <= ix1; ++ix) {
                    const float px = -half + static_cast<float>(ix) * cell;
                    bool in = false;
                    for (size_t i = 0, j = ring.size() - 1; i < ring.size(); j = i++) {
                        const Vector2& a = ring[i];
                        const Vector2& c = ring[j];
                        if ((a.y > pz) != (c.y > pz) &&
                            px < (c.x - a.x) * (pz - a.y) / (c.y - a.y) + a.x)
                            in = !in;
                    }
                    if (in) mask.m[static_cast<size_t>(iz) * dim + ix] = value;
                }
            }
        }

        // Chebyshev dilation by `r` cells, separable (two max-filter passes).
        inline void geoDilateMask(FootprintMask& mask, int r) {
            if (r <= 0 || mask.dim < 2) return;
            const int dim = mask.dim;
            std::vector<std::uint8_t> tmp(mask.m.size(), 0u);
            for (int b = 0; b < dim; ++b)
                for (int a = 0; a < dim; ++a) {
                    std::uint8_t v = 0u;
                    for (int k = std::max(0, a - r); k <= std::min(dim - 1, a + r); ++k)
                        v = std::max(v, mask.m[static_cast<size_t>(b) * dim + k]);
                    tmp[static_cast<size_t>(b) * dim + a] = v;
                }
            for (int a = 0; a < dim; ++a)
                for (int b = 0; b < dim; ++b) {
                    std::uint8_t v = 0u;
                    for (int k = std::max(0, b - r); k <= std::min(dim - 1, b + r); ++k)
                        v = std::max(v, tmp[static_cast<size_t>(k) * dim + a]);
                    mask.m[static_cast<size_t>(b) * dim + a] = v;
                }
        }

        inline std::shared_ptr<FootprintMask> geoMakeMask(float worldSize, float cell) {
            auto mask = std::make_shared<FootprintMask>();
            mask->cell = std::max(0.5f, cell);
            mask->half = worldSize * 0.5f;
            mask->dim = static_cast<int>(std::ceil(worldSize / mask->cell)) + 1;
            mask->m.assign(static_cast<size_t>(mask->dim) * mask->dim, 0u);
            return mask;
        }

    }// namespace detail

    // Every building footprint, rasterised at `cell` and grown by
    // `dilateMetres`. Courtyard holes are NOT punched out: a courtyard is still
    // building fabric, and a 4 m dilation would swallow a small one anyway.
    inline std::shared_ptr<const FootprintMask> buildFootprintMask(const GeoTerrainPack& pack,
                                                                   float dilateMetres = 4.f,
                                                                   float cell = 2.f) {
        if (pack.buildings.empty() || pack.region.worldSize <= 0.f) return nullptr;
        auto mask = detail::geoMakeMask(pack.region.worldSize, cell);
        for (const auto& b : pack.buildings) detail::geoRasterRing(*mask, b.outer);
        detail::geoDilateMask(*mask, static_cast<int>(std::lround(dilateMetres / mask->cell)));
        return mask;
    }

    // The same, for a set of land-use polygon classes (pier, quay, parking...).
    // `classes` is a null-terminated list of class strings; a pack without
    // landuse.json yields nullptr.
    inline std::shared_ptr<const FootprintMask> buildLandUseMask(
            const GeoTerrainPack& pack, const std::vector<std::string>& classes,
            float dilateMetres = 0.f, float cell = 2.f) {
        if (pack.landuse.polygons.empty() || pack.region.worldSize <= 0.f) return nullptr;
        auto mask = detail::geoMakeMask(pack.region.worldSize, cell);
        int n = 0;
        for (const auto& p : pack.landuse.polygons) {
            if (std::find(classes.begin(), classes.end(), p.cls) == classes.end()) continue;
            detail::geoRasterRing(*mask, p.outer);
            ++n;
        }
        if (n == 0) return nullptr;
        detail::geoDilateMask(*mask, static_cast<int>(std::lround(dilateMetres / mask->cell)));
        return mask;
    }

    namespace detail {

        // Self-contained hash value-noise / fBm (world-anchored, deterministic),
        // used only for the sub-grid detail relief.
        inline float geoHash01(int x, int y) {
            unsigned int n = static_cast<unsigned int>(x) * 374761393u +
                             static_cast<unsigned int>(y) * 668265263u + 0x9E3779B9u;
            n = (n ^ (n >> 13)) * 1274126177u;
            return static_cast<float>((n ^ (n >> 16)) & 0xffffffu) / static_cast<float>(0xffffff);
        }
        inline float geoVNoise(float x, float y) {
            const int xi = static_cast<int>(std::floor(x)), yi = static_cast<int>(std::floor(y));
            auto sm = [](float t) { return t * t * (3.f - 2.f * t); };
            const float fx = sm(x - std::floor(x)), fy = sm(y - std::floor(y));
            const float a = geoHash01(xi, yi), b = geoHash01(xi + 1, yi);
            const float c = geoHash01(xi, yi + 1), d = geoHash01(xi + 1, yi + 1);
            return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
        }
        inline float geoFbm(float x, float y) {
            return 0.55f * geoVNoise(x, y) + 0.30f * geoVNoise(x * 2.13f + 7.3f, y * 2.13f) +
                   0.15f * geoVNoise(x * 4.7f, y * 4.7f + 3.1f);
        }

        // ── Cliff relief ────────────────────────────────────────────────────
        //
        // Value-by-value POD of the cliff knobs, so the height callback captures
        // one small struct instead of eight floats (and can be shared with the
        // splat side without re-reading GeoTerrainOptions).
        struct CliffParams {
            float benchAmp = 0.f, benchPeriod = 4.5f;
            float jointAmp = 0.f, jointPeriod = 5.5f;
            float slopeLo = 0.45f, slopeHi = 0.62f;
            [[nodiscard]] bool active() const { return benchAmp > 0.f || jointAmp > 0.f; }
        };

        inline float smoothstep01(float e0, float e1, float x) {
            const float t = std::clamp((x - e0) / (e1 - e0 + 1e-6f), 0.f, 1.f);
            return t * t * (3.f - 2.f * t);
        }

        // DEM slope (0 flat → 1 vertical, the TileTerrain convention) measured
        // BICUBICALLY. Bilinear would be cheaper, but its derivative jumps at
        // every DEM cell edge; feeding that through the steep gate below and
        // into the HEIGHT field would emboss the 1 m sample grid onto the wall.
        // Bicubic is C1, so the gate — and therefore the relief — is C1 too.
        inline float geoDemSlope(const HeightGrid& g, float x, float z, float e = 6.f) {
            const float hx = g.sampleBicubic(x + e, z) - g.sampleBicubic(x - e, z);
            const float hz = g.sampleBicubic(x, z + e) - g.sampleBicubic(x, z - e);
            return 1.f - (2.f * e) / std::sqrt(hx * hx + hz * hz + 4.f * e * e);
        }

        // Benched + jointed displacement for a steep face, in metres.
        //
        // BENCHES are a smooth staircase in the HEIGHT domain, not the plan
        // domain: a function of h alone follows the wall's own contours, so the
        // treads come out horizontal on a wall of any orientation for free (a
        // plan-space pattern would smear into diagonal stripes as the face
        // turns). The staircase uses smoothstep, whose derivative is zero at
        // both ends, so it is C1 across the period wrap. Riser is kept ≤ ~0.25×
        // period: the vertical Jacobian is 1 + riser·5/period > 0, i.e. the
        // step steepens the face but can never fold it into an overhang.
        //
        // JOINTS are two oblique cosine groove sets — C∞, so the baked normal
        // map never carries a crease the geometry does not have — with an fBm
        // phase wobble so they read as fractures rather than corduroy.
        inline float geoCliffRelief(float x, float z, float base, float demSlope,
                                    const CliffParams& c) {
            const float gate = smoothstep01(c.slopeLo, c.slopeHi, demSlope);
            if (gate <= 0.f) return 0.f;

            float d = 0.f;
            if (c.benchAmp > 0.f) {
                // 50 m drift in the bedding spacing + a separate drift in riser
                // height: real strata thicken and thin along a wall.
                const float period = c.benchPeriod * (0.65f + 0.70f * geoFbm(x * 0.02f, z * 0.02f));
                const float u = base / period;
                const float t = u - std::floor(u);
                const float riser = c.benchAmp * (0.55f + 0.90f * geoFbm(x * 0.05f + 13.f, z * 0.05f));
                d += (smoothstep01(0.30f, 0.70f, t) - 0.5f) * riser;
            }
            if (c.jointAmp > 0.f) {
                constexpr float kTwoPi = 6.28318531f;
                // ±34° from the X axis: an oblique conjugate pair, the usual
                // fracture geometry, and neither set lines up with the DEM grid.
                constexpr float c1 = 0.829f, s1 = 0.559f; // +34°
                constexpr float c2 = 0.829f, s2 = -0.559f;// −34°
                const float p = c.jointPeriod;
                const float w1 = (x * c1 + z * s1) / p + 1.1f * geoFbm(x * 0.03f, z * 0.03f);
                const float w2 = (x * c2 + z * s2) / (p * 0.73f) + 1.1f * geoFbm(x * 0.041f + 5.f, z * 0.041f);
                const float g1 = 0.5f - 0.5f * std::cos(kTwoPi * w1);
                const float g2 = 0.5f - 0.5f * std::cos(kTwoPi * w2);
                d -= c.jointAmp * (g1 * g1 * 0.65f + g2 * g2 * 0.45f);
            }
            return d * gate;
        }

        // ── Land cover from measured grids ──────────────────────────────────
        //
        // Four masks, all derived from data rather than tuned noise:
        //   forest — the canopy height model says where trees ARE.
        //   wet    — high drainage on a steep face = a seepage streak.
        //   scree  — high drainage where the slope has RELAXED = a talus fan.
        //   bench  — DEM steep but the DETAILED surface flat: that is exactly a
        //            bench tread cut by geoCliffRelief, and moss/heath collects
        //            on treads. Free by construction: the geometry places the
        //            paint, so the two can never disagree.
        // All pointers are into the pack (read-only after load), so the masks
        // are pure and safe on the tile-bake threads.
        struct LandCover {
            const HeightGrid* canopy = nullptr;
            const HeightGrid* flow = nullptr;
            const HeightGrid* dem = nullptr;// for the DEM-vs-detail slope split
            // OSM land cover, rasterised once (4 m): the surveyor drew where the
            // rock is bare and where the heath starts. The splat's slope/height
            // rules guess it; these polygons KNOW it.
            std::shared_ptr<const FootprintMask> rock;
            std::shared_ptr<const FootprintMask> heath;
            float forestMin = 2.f, forestFull = 6.f;
            float wetLo = 0.52f, wetHi = 0.72f, screeLo = 0.48f;

            [[nodiscard]] bool active() const { return canopy || flow || rock || heath; }

            struct Mix {
                float forest = 0.f;// 0..1 canopy coverage
                float conifer = 0.f;// 0 broadleaf → 1 spruce/pine (altitude)
                float wet = 0.f;
                float scree = 0.f;
                float bench = 0.f;
                float rock = 0.f; // OSM natural=bare_rock
                float heath = 0.f;// OSM scrub | heath
            };

            [[nodiscard]] Mix eval(float x, float z, float h, float slope) const {
                Mix m;
                // DEM slope is only needed by the bench/scree split; skip the
                // four extra bicubic samples when no mask uses it.
                const float ds = dem ? geoDemSlope(*dem, x, z) : slope;
                if (canopy) {
                    const float ch = canopy->sampleBilinear(x, z);
                    // A near-vertical face has no trees: a CHM reading there is
                    // lidar returning off the wall itself, not a canopy.
                    m.forest = smoothstep01(forestMin, forestFull, ch) *
                               (1.f - smoothstep01(0.62f, 0.80f, ds));
                    m.conifer = smoothstep01(220.f, 620.f, h);
                }
                if (flow) {
                    const float f = flow->sampleBilinear(x, z);
                    m.wet = smoothstep01(wetLo, wetHi, f) * smoothstep01(0.50f, 0.66f, slope) *
                            (1.f - m.forest);
                    // Fan: the drainage line has arrived somewhere the ground
                    // eased off, and it is not the shoreline.
                    m.scree = smoothstep01(screeLo, screeLo + 0.20f, f) *
                              (1.f - smoothstep01(0.36f, 0.54f, slope)) *
                              smoothstep01(8.f, 30.f, h) * (1.f - m.forest);
                }
                if (dem) {
                    m.bench = smoothstep01(0.58f, 0.74f, ds) *
                              (1.f - smoothstep01(0.26f, 0.46f, slope)) * (1.f - m.forest);
                }
                // Surveyed cover wins over the guessed cover: where OSM says bare
                // rock there is no grass tint and no bench moss, and where it says
                // scrub/heath the ground is brown, not lawn.
                if (rock) m.rock = std::clamp(rock->coverage(x, z), 0.f, 1.f) * (1.f - m.forest);
                if (heath)
                    m.heath = std::clamp(heath->coverage(x, z), 0.f, 1.f) *
                              (1.f - m.forest) * (1.f - m.rock);
                if (m.rock > 0.f) m.bench *= 1.f - m.rock;
                return m;
            }
        };

    }// namespace detail

    // ── UrbanMask: smoothed built-coverage over the pack footprint ──────────
    //
    // A coarse world-space raster of "how built is the land here", 0..1:
    // building footprints are rasterized as occupancy, box-blurred to a local
    // COVERAGE fraction (radius ~ a town block), then soft-thresholded between
    // urbanCoverLo and urbanCoverHi. Coverage — not proximity — is the gate:
    // a lone cabin blurs to ~3% and stays natural ground, a dense block hits
    // 30%+ and paints fully; suburbs land in between and read as the half
    // garden / half street fabric they are. Built once at load (the mask is
    // immutable after), sampled bilinearly by the provider callbacks.
    struct UrbanMask {
        int dim = 0;
        float cell = 4.f;
        float half = 0.f;
        std::vector<float> w;// paint weight per cell, 0..1

        float sample(float x, float z) const {
            if (dim < 2) return 0.f;
            const float fx = (x + half) / cell;
            const float fz = (z + half) / cell;
            const int ix = static_cast<int>(std::floor(fx));
            const int iz = static_cast<int>(std::floor(fz));
            if (ix < 0 || iz < 0 || ix >= dim - 1 || iz >= dim - 1) return 0.f;
            const float tx = fx - static_cast<float>(ix);
            const float tz = fz - static_cast<float>(iz);
            const float* r0 = &w[static_cast<size_t>(iz) * dim + ix];
            const float* r1 = r0 + dim;
            const float a = r0[0] + (r0[1] - r0[0]) * tx;
            const float b = r1[0] + (r1[1] - r1[0]) * tx;
            return a + (b - a) * tz;
        }
    };

    inline std::shared_ptr<const UrbanMask> buildUrbanMask(const GeoTerrainPack& pack,
                                                           const GeoTerrainOptions& o = {}) {
        if (pack.buildings.empty() || pack.region.worldSize <= 0.f) return nullptr;
        auto mask = std::make_shared<UrbanMask>();
        mask->cell = std::max(1.f, o.urbanCell);
        mask->half = pack.region.worldSize * 0.5f;
        mask->dim = static_cast<int>(std::ceil(pack.region.worldSize / mask->cell)) + 1;
        const int dim = mask->dim;
        const float cell = mask->cell;
        const float half = mask->half;
        mask->w.assign(static_cast<size_t>(dim) * dim, 0.f);
        auto& w = mask->w;

        // Occupancy: cell centres inside a footprint's outer ring (crossing
        // number; courtyard holes count as built — they ARE town fabric).
        for (const auto& b : pack.buildings) {
            const auto& ring = b.outer;
            if (ring.size() < 3) continue;
            float minX = ring[0].x, maxX = ring[0].x, minZ = ring[0].y, maxZ = ring[0].y;
            for (const auto& p : ring) {
                minX = std::min(minX, p.x);
                maxX = std::max(maxX, p.x);
                minZ = std::min(minZ, p.y);
                maxZ = std::max(maxZ, p.y);
            }
            const int ix0 = std::max(0, static_cast<int>(std::floor((minX + half) / cell)));
            const int ix1 = std::min(dim - 1, static_cast<int>(std::ceil((maxX + half) / cell)));
            const int iz0 = std::max(0, static_cast<int>(std::floor((minZ + half) / cell)));
            const int iz1 = std::min(dim - 1, static_cast<int>(std::ceil((maxZ + half) / cell)));
            for (int iz = iz0; iz <= iz1; ++iz) {
                const float pz = -half + static_cast<float>(iz) * cell;
                for (int ix = ix0; ix <= ix1; ++ix) {
                    const float px = -half + static_cast<float>(ix) * cell;
                    bool inside = false;
                    for (size_t i = 0, j = ring.size() - 1; i < ring.size(); j = i++) {
                        const Vector2& a = ring[i];
                        const Vector2& c = ring[j];
                        if ((a.y > pz) != (c.y > pz) &&
                            px < (c.x - a.x) * (pz - a.y) / (c.y - a.y) + a.x)
                            inside = !inside;
                    }
                    if (inside) w[static_cast<size_t>(iz) * dim + ix] = 1.f;
                }
            }
        }

        // Occupancy → local coverage fraction: two separable box-blur passes
        // (≈ triangular kernel) at the block-scale radius.
        const int r = std::max(1, static_cast<int>(std::lround(o.urbanBlurRadius / cell)));
        std::vector<float> tmp(w.size());
        const auto blurAxis = [&](const std::vector<float>& src, std::vector<float>& dst,
                                  bool alongX) {
            const float norm = 1.f / static_cast<float>(2 * r + 1);
            for (int b = 0; b < dim; ++b) {
                // Sliding-window mean along one row/column (clamped edges).
                const auto at = [&](int a) -> const float& {
                    const int c = std::clamp(a, 0, dim - 1);
                    return alongX ? src[static_cast<size_t>(b) * dim + c]
                                  : src[static_cast<size_t>(c) * dim + b];
                };
                float acc = 0.f;
                for (int a = -r; a <= r; ++a) acc += at(a);
                for (int a = 0; a < dim; ++a) {
                    (alongX ? dst[static_cast<size_t>(b) * dim + a]
                            : dst[static_cast<size_t>(a) * dim + b]) = acc * norm;
                    acc += at(a + r + 1) - at(a - r);
                }
            }
        };
        blurAxis(w, tmp, true);
        blurAxis(tmp, w, false);
        blurAxis(w, tmp, true);
        blurAxis(tmp, w, false);

        // Coverage → paint weight (soft threshold).
        const float lo = o.urbanCoverLo, hi = std::max(o.urbanCoverHi, o.urbanCoverLo + 1e-3f);
        for (auto& v : w) {
            const float t = std::clamp((v - lo) / (hi - lo), 0.f, 1.f);
            v = t * t * (3.f - 2.f * t);
        }
        return mask;
    }

    // ── carveRoads: bake the road cut into the DEM grid (call ONCE at load) ──
    //
    // Rather than warping the height field at runtime (corridor flatten +
    // trench — whose nearest-corridor composition had metre-scale seams that
    // fine tiles faithfully reproduced as humps), edit the GRID so no terrain
    // sample near a road sits above road level.
    //
    // For every grid cell within `pavedHalf + inflate` lateral metres of any
    // conformed road centerline: cell = min(cell, ribbonSurface − clearance),
    // feathered back to natural ground over `feather` metres beyond that.
    // min() only CUTS (the uphill side); cells already below road level are
    // untouched — no fake embankments (the ribbon's shoulders skirt the
    // downhill side). Per cell the NEAREST segment wins, so stacked hairpin
    // switchbacks each carve their own bench and never gouge each other.
    //
    // Sizing `inflate`: a coarse tile quad that straddles the road can lift an
    // interpolated EDGE above road level even when no vertex is inside the
    // corridor, so the full-cut band must cover at least one worst-case quad
    // beyond the pavement edge. With the demos' tile setup (worldSize 8000,
    // rootGrid 4, maxDepth 5, tileRes 96, splitFactor 1.2) and the road-tile
    // refineBias 2.2, a road tile within ~660 m of the camera is ≤250 m wide
    // (2.6 m quads) and within ~1450 m is ≤500 m (5.2 m quads). inflate = 6 m
    // covers that 5.2 m worst case; beyond ~1.5 km a residual poke subtends
    // well under a pixel.
    //
    // Call AFTER RoadNetwork::conformTo(rawGrid) — roads must conform to the
    // REAL ground first; the carve then uses the conformed surface heights.
    // Assumes the grid is centred on the origin (the region-pack contract).
    struct RoadCarveOptions {
        float inflate = 6.f;    // full-cut band beyond the pavement edge (m)
        float feather = 6.f;    // cut→natural blend beyond the full-cut band (m)
        float clearance = 0.40f;// terrain held this far below the ribbon surface (m).
                                // Sized to swallow worst-case bicubic (Catmull-Rom)
                                // overshoot next to carve walls (~0.1 m measured at
                                // Trollstigen's stacked hairpins) with margin.
        // BAKE mode (the "terrain IS the road" pipeline, used with
        // GeoTerrainOptions::paintRoads + RoadNetwork::buildBridgeMeshes):
        // instead of benching the terrain BELOW a ribbon, SET the paved band to
        // the exact road surface — cut AND fill, dead flat across the pavement —
        // feathered back to natural ground (embankments where the road runs
        // above grade, exactly like a real roadbed). No clearance drop: with no
        // ribbon to z-fight, the surface itself is the road. Bridge-classified
        // segments never carve (the deck spans; ground below stays natural) and
        // excluded segments (tunnels/ferries) never carve at all.
        bool bakeSurface = false;
        // Bake-mode FILL is ASYMMETRIC from the cut. Cutting (terrain above road
        // level) must keep the full inflate+feather band — that is the tile-quad
        // coverage guarantee above, and narrowing it lets uphill quads interpolate
        // ABOVE the road and slice through the pavement (torn-asphalt artifacts).
        // Filling (terrain below road level) has no such constraint: a quad edge
        // sagging below the road is just the hillside falling away. So fill is
        // confined to a NARROW shoulder band + short embankment taper — on steep
        // sidehills the road hugs the slope on a tight fill wall instead of the
        // huge flat berm the full band would build (Trollstigen: terrain drops
        // ~27 m across the 24 m corridor; a 12 m-half-width flat shelf there reads
        // as a dark raised bench). Only meaningful with bakeSurface.
        float fillInflate = 2.f;// full-fill band beyond the pavement edge (m)
        float fillFeather = 3.f;// fill→natural embankment taper beyond that (m)
        float bakeClearance = 0.f;// bake mode: hold the baked bed this far below the
                                // road grade (the driving ribbon rides kSurfaceRaise
                                // above grade). The gap stops cut-wall bicubic
                                // overshoot from poking the terrain up through the
                                // ribbon; a driving demo wants ~0.25 m, a pure viewer
                                // 0 (terrain == painted road, no shoulder step).
    };

    inline void carveRoads(HeightGrid& grid, const road::RoadNetwork& net,
                           const RoadCarveOptions& o = {}) {
        if (!grid.valid()) return;
        const int dim = grid.dim();
        const float step = grid.worldSize() / static_cast<float>(dim - 1);
        const float half = grid.worldSize() * 0.5f;
        auto& h = grid.data();

        // Per-cell candidates (load-time transients; freed on return). Full-size
        // planes keep the inner loop branch-cheap.
        //   • bench/feather: NEAREST segment wins (stacked switchbacks never
        //     gouge each other's slopes);
        //   • hard cap: within pavedHalf + hardGuard of ANY segment the cell is
        //     unconditionally clamped to that segment's ceiling (min over all) —
        //     the bicubic support is ±2 cells, so a cell whose nearest segment
        //     is the OTHER leg of a stacked hairpin could otherwise keep its
        //     natural height and get pulled up under this road's pavement.
        const float hardGuard = 2.f * step;// bicubic support radius. Deliberately
        // NOT wider: at stacked hairpins a wider band reaches cells under the
        // OTHER leg's pavement, carving deeper pits whose walls only increase
        // Catmull-Rom overshoot (measured: 3·step tripled the residual pokes).
        // The remaining ≤ ~0.1 m overshoot is absorbed by `clearance` instead.
        const size_t n = static_cast<size_t>(dim) * dim;
        std::vector<float> bestD(n, std::numeric_limits<float>::max());
        std::vector<float> allowed(n, 0.f);// nearest winner's ribbonSurface − clearance
        std::vector<float> innerR(n, 0.f); // nearest winner's full-cut lateral reach
        std::vector<float> fillR(n, 0.f);  // nearest winner's full-FILL reach (bake mode)
        std::vector<float> hardCap(n, std::numeric_limits<float>::max());

        net.forEachSegmentFlagged([&](float ax, float az, float ha, float bx, float bz, float hb,
                                      float pavedHalf, float /*corridorHalf*/, std::uint8_t flags) {
            // Bridge decks span the ground (never carve it); excluded roads
            // (tunnels/ferries) don't exist on the surface. Flags are always 0
            // in legacy (non-profile) mode — behaviour unchanged there.
            if (flags != 0) return;
            const float reach = pavedHalf + o.inflate + o.feather;
            const float hardReach = pavedHalf + hardGuard;
            const int ix0 = std::max(0, static_cast<int>(std::floor((std::min(ax, bx) - reach + half) / step)));
            const int ix1 = std::min(dim - 1, static_cast<int>(std::ceil((std::max(ax, bx) + reach + half) / step)));
            const int iz0 = std::max(0, static_cast<int>(std::floor((std::min(az, bz) - reach + half) / step)));
            const int iz1 = std::min(dim - 1, static_cast<int>(std::ceil((std::max(az, bz) + reach + half) / step)));
            const float abx = bx - ax, abz = bz - az;
            const float abLenSq = abx * abx + abz * abz;
            for (int iz = iz0; iz <= iz1; ++iz) {
                const float z = -half + static_cast<float>(iz) * step;
                for (int ix = ix0; ix <= ix1; ++ix) {
                    const float x = -half + static_cast<float>(ix) * step;
                    float t = (abLenSq > 1e-12f) ? ((x - ax) * abx + (z - az) * abz) / abLenSq : 0.f;
                    t = std::clamp(t, 0.f, 1.f);
                    const float dx = x - (ax + t * abx), dz = z - (az + t * abz);
                    const float d = std::sqrt(dx * dx + dz * dz);
                    if (d >= reach) continue;
                    const size_t idx = static_cast<size_t>(iz) * dim + ix;
                    // Bake mode: the terrain roadbed sits bakeClearance BELOW the
                    // conformed grade. The driving ribbon floats kSurfaceRaise
                    // ABOVE the grade, so the total gap (kSurfaceRaise+bakeClearance)
                    // keeps bicubic overshoot at steep cut walls from poking the
                    // terrain up THROUGH the ribbon and launching the car — while
                    // the ribbon and painted bed still line up to well under a curb.
                    const float ceil = o.bakeSurface
                                               ? ha + (hb - ha) * t - o.bakeClearance
                                               : ha + (hb - ha) * t +
                                                         road::RoadNetwork::kSurfaceRaise - o.clearance;
                    if (d < hardReach) hardCap[idx] = std::min(hardCap[idx], ceil);
                    if (d >= bestD[idx]) continue;
                    bestD[idx] = d;
                    allowed[idx] = ceil;
                    innerR[idx] = pavedHalf + o.inflate;
                    fillR[idx] = pavedHalf + o.fillInflate;
                }
            }
        });

        for (size_t i = 0; i < n; ++i) {
            if (bestD[i] != std::numeric_limits<float>::max()) {
                // Legacy: bench cut only (h > allowed), feathered to natural.
                // Bake: cut AND fill, but ASYMMETRIC — the cut keeps the full
                // inflate band (the tile-quad anti-poke guarantee), while the
                // fill is confined to fillInflate + a short fillFeather taper so
                // steep sidehills get a tight embankment, not a wide berm shelf.
                if (o.bakeSurface && h[i] < allowed[i]) {
                    const float w = (bestD[i] <= fillR[i])
                                            ? 1.f
                                            : 1.f - math::smoothstep(fillR[i], fillR[i] + o.fillFeather, bestD[i]);
                    h[i] += (allowed[i] - h[i]) * w;
                } else if (o.bakeSurface || h[i] > allowed[i]) {
                    const float w = (bestD[i] <= innerR[i])
                                            ? 1.f
                                            : 1.f - math::smoothstep(innerR[i], innerR[i] + o.feather, bestD[i]);
                    h[i] += (allowed[i] - h[i]) * w;
                }
            }
            // Under-pavement guarantee (stacked hairpins). In bake mode a cell
            // INSIDE its winning road's paved band must keep that exact surface
            // — the other leg's cap would gouge a pit into this leg's roadway.
            if (h[i] > hardCap[i] && !(o.bakeSurface && bestD[i] <= innerR[i]))
                h[i] = hardCap[i];
        }
    }

    // Norwegian slope/altitude splat rules over the pack's DEM. Curvature reads
    // the raw grid (fixed eps → LOD-agnostic). Colours are sRGB (baked into an
    // sRGB tile texture).
    inline SplatRules makeNorwegianSplat(const GeoTerrainPack& pack, const GeoTerrainOptions& o = {}) {
        const HeightGrid& grid = pack.grid;
        SplatRules r;
        r.height = [&grid](float x, float z) { return grid.sampleBilinear(x, z); };
        r.curvEps = 3.5f;
        r.curvScale = 55.f;
        r.aoStrength = 0.15f;// gentle occlusion in the folds
        r.aoMax = 0.28f;
        r.macroEnabled = true;

        const float sea = pack.region.seaLevel;

        SplatLayer wetland;// boggy shore / valley-floor darkening near sea level
        wetland.structureBand = 0;// grass-family structure (tufty bog)
        wetland.color = {0.16f, 0.17f, 0.12f};
        wetland.slopeLo = 0.f; wetland.slopeHi = 0.30f; wetland.slopeFeather = 0.06f;
        wetland.heightLo = sea - 0.5f;// the bog stops at the waterline: the seabed has its own layers below
        wetland.heightHi = sea + o.wetlandBand;
        wetland.heightFeather = 4.f;
        wetland.concaveBias = 0.4f;
        wetland.noiseAmpHeight = 2.5f;

        SplatLayer grass;// valley grass/heath on gentle low ground
        grass.structureBand = 0;
        grass.color = {0.22f, 0.28f, 0.14f};
        grass.slopeLo = 0.f; grass.slopeHi = 0.34f; grass.slopeFeather = 0.06f;
        grass.heightHi = o.grassHeightMax;
        grass.heightFeather = 120.f;
        grass.concaveBias = 0.25f;// soil catches in hollows
        grass.noiseAmpSlope = 0.03f;
        grass.noiseAmpHeight = 40.f;

        SplatLayer heath;// brown heath/fell above the grass, still gentle
        heath.structureBand = 0;// still grass-family structure, browner colour
        heath.color = {0.30f, 0.27f, 0.17f};
        heath.slopeLo = 0.f; heath.slopeHi = 0.40f; heath.slopeFeather = 0.07f;
        heath.heightLo = o.grassHeightMax - 120.f;
        heath.heightHi = o.snowHeightMin;
        heath.heightFeather = 150.f;
        heath.noiseAmpHeight = 50.f;

        SplatLayer scree;// talus on medium slopes
        scree.structureBand = 2;
        scree.color = {0.42f, 0.40f, 0.37f};
        scree.slopeLo = 0.34f; scree.slopeHi = 0.58f; scree.slopeFeather = 0.07f;
        scree.concaveBias = 0.9f;// collects in gullies
        scree.noiseAmpSlope = 0.03f;

        SplatLayer rock;// bare rock on the steep faces (fallback)
        rock.structureBand = 1;
        rock.color = {0.34f, 0.33f, 0.31f};
        rock.slopeLo = 0.55f; rock.slopeHi = 1.f; rock.slopeFeather = 0.08f;
        rock.convexBias = 0.8f;   // bare on convex crests
        rock.weightFloor = 0.02f; // no texel resolves to grey

        SplatLayer snow;// snow up high, low slope, feathered line
        snow.structureBand = 3;
        snow.color = {0.88f, 0.90f, 0.94f};
        snow.slopeLo = 0.f; snow.slopeHi = 0.60f; snow.slopeFeather = 0.10f;
        snow.heightLo = o.snowHeightMin;
        snow.heightFeather = o.snowFeather;
        snow.noiseAmpHeight = 70.f;// ragged snowline
        snow.noiseFreq = 0.03f;

        // ── seabed: what the water reveals when it is clear enough ────────
        // Every layer above is a LAND rule; below the surface the slope windows
        // admitted grass (gentle bottoms) and rock (steep ones), so the visible
        // shallows read as a lawn under teal water and switched to bare rock
        // along a slope contour (netpen --terrain aerials, 2026-09-05). Two
        // layers own the underwater band by weight (weightScale): pale gravel
        // and sand in the first metres, grading over ~2-5 m of depth into the
        // dark silt and rock the light no longer reaches. Feathers are in
        // metres of DEPTH; with a ~0.35 m/m shore profile that is a 10-15 m wide
        // shoreline grade, soft at any sensible altitude. Bands: gravel takes
        // the scree structure (stones), the deep bottom the rock structure.
        SplatLayer shallow;
        shallow.structureBand = 2;
        shallow.color = {0.40f, 0.38f, 0.30f};
        shallow.heightLo = sea - 3.5f;
        shallow.heightHi = sea + 0.1f;
        shallow.heightFeather = 1.2f;
        shallow.noiseAmpHeight = 0.8f;// ragged, not an iso-contour
        shallow.noiseFreq = 0.08f;
        shallow.weightScale = 6.f;

        SplatLayer seabed;
        seabed.structureBand = 1;
        seabed.color = {0.09f, 0.11f, 0.10f};
        seabed.heightHi = sea - 2.5f;
        seabed.heightFeather = 2.0f;
        seabed.weightScale = 6.f;

        r.layers = {wetland, grass, heath, scree, rock, snow, shallow, seabed};
        return r;
    }

    // Build the full TerrainProvider. `pack` and `network` must outlive it,
    // network.conformTo() must already have run, and the grid should have been
    // carved (carveRoads) so terrain near roads sits below the ribbons.
    inline TerrainProvider makeGeoProvider(const GeoTerrainPack& pack,
                                           const road::RoadNetwork& network,
                                           const GeoTerrainOptions& o = {}) {
        const HeightGrid& grid = pack.grid;
        const SplatRules rules = makeNorwegianSplat(pack, o);
        const float amp = o.detailAmplitude;
        const float freq = o.detailFreq;

        // Built-coverage mask for the urban paint (null when the pack has no
        // buildings or the paint is off). shared_ptr: the provider's callbacks
        // co-own it, so the mask lives exactly as long as the provider.
        const std::shared_ptr<const UrbanMask> urban =
                o.paintUrban ? buildUrbanMask(pack, o) : nullptr;

        TerrainProvider prov;

        // Height: pure bicubic of the (carved) DEM + corridor-faded sub-grid
        // relief. NO runtime road warp: the road cut is baked into the grid by
        // carveRoads, so the field is band-limited at DEM resolution and C1
        // everywhere — nothing sub-quad-scale for a tile split to reveal. The
        // relief also fades to zero approaching sea level: the DTM stores water
        // as a flat seaLevel sheet, and noise there would dither the seabed up
        // through the sea plane (patchy shoreline). Urban ground is graded
        // flat too — the relief fades with built coverage.
        //
        // CLIFF relief (opt-in) rides on the same shore fade but uses a WIDER
        // one (3 m, the plan's figure): a bench riser cutting the waterline
        // would chop the shoreline into a sawtooth where the DTM's flat sea
        // sheet meets the wall.
        const float sea = pack.region.seaLevel;
        detail::CliffParams cliff;
        if (o.cliffRelief) {
            cliff.benchAmp = o.cliffBenchAmp;
            cliff.benchPeriod = o.cliffBenchPeriod;
            cliff.jointAmp = o.cliffJointAmp;
            cliff.jointPeriod = o.cliffJointPeriod;
            cliff.slopeLo = o.cliffSlopeLo;
            cliff.slopeHi = o.cliffSlopeHi;
        }
        prov.height = [&grid, &network, urban, amp, freq, sea, cliff](float x, float z) {
            const float base = grid.sampleBicubic(x, z);
            const float cw = network.corridorWeight(x, z);
            if (cw >= 0.999f) return base;// fully paved — keep it dead smooth
            const float shore = std::clamp((base - (sea + 0.2f)) / 1.8f, 0.f, 1.f);
            if (shore <= 0.f) return base;
            float relief = (detail::geoFbm(x * freq, z * freq) - 0.5f) * 2.f * amp;
            if (urban) relief *= 1.f - urban->sample(x, z);
            float h = base + relief * (1.f - cw) * shore * shore * (3.f - 2.f * shore);
            if (cliff.active()) {
                const float seaFade = detail::smoothstep01(sea + 0.5f, sea + 3.5f, base);
                if (seaFade > 0.f)
                    h += detail::geoCliffRelief(x, z, base, detail::geoDemSlope(grid, x, z), cliff) *
                         seaFade * (1.f - cw);
            }
            return h;
        };

        // Albedo: Norwegian splat, then (buildings) the urban town-fabric
        // blend, then (paintRoads) asphalt over the paved band ON TOP — the
        // paint order keeps streets readable through town. The road paint uses
        // pavedWeight — pavement + a NARROW edge feather, never the shoulder
        // corridor: a corridor-wide darkened swath reads as a phantom second
        // road wherever roads run close or stack (hairpins, dual carriageways).
        detail::LandCover cover;
        if (o.landCover) {
            if (pack.hasCanopy()) cover.canopy = &pack.canopy;
            if (pack.hasFlow()) cover.flow = &pack.flow;
            // The bench mask only means something when benches exist.
            if (o.cliffRelief) cover.dem = &grid;
            if (pack.hasLandUse()) {
                cover.rock = buildLandUseMask(pack, {"bare_rock", "scree"}, 0.f, 4.f);
                cover.heath = buildLandUseMask(pack, {"scrub", "heath"}, 0.f, 4.f);
            }
            cover.forestMin = o.canopyForestMin;
            cover.forestFull = o.canopyForestFull;
            cover.wetLo = o.wetFlowLo;
            cover.wetHi = o.wetFlowHi;
            cover.screeLo = o.screeFlowLo;
        }

        {
            const bool paint = o.paintRoads;
            const float edgeFeather = o.roadEdgeFeather;
            const std::array<float, 3> roadCol = o.roadColor;
            const std::array<float, 3> urbA = o.urbanAsphalt;
            const std::array<float, 3> urbG = o.urbanGravel;
            const float urbMax = o.urbanMax;
            prov.albedo = [rules, &network, urban, cover, paint, edgeFeather, roadCol, urbA, urbG,
                           urbMax](float x, float z, float h, float slope, float* rgb) {
                const Rgb c = rules.evaluate(x, z, h, slope);
                rgb[0] = c[0];
                rgb[1] = c[1];
                rgb[2] = c[2];
                // ── measured land cover, UNDER the urban/road paint ─────────
                if (cover.active()) {
                    const auto m = cover.eval(x, z, h, slope);
                    // Canopy: birch/alder green low down drifting to spruce
                    // dark with altitude, plus a 40 m stand-scale patchiness so
                    // the mass is not one flat green.
                    if (m.forest > 0.001f) {
                        const float pat = 0.85f + 0.30f * detail::geoFbm(x * 0.025f, z * 0.025f);
                        const std::array<float, 3> birch{0.125f, 0.190f, 0.070f};
                        const std::array<float, 3> spruce{0.042f, 0.072f, 0.045f};
                        for (int i = 0; i < 3; ++i) {
                            const float tree = (birch[i] + (spruce[i] - birch[i]) * m.conifer) * pat;
                            rgb[i] += (tree - rgb[i]) * m.forest;
                        }
                    }
                    // Bench moss/lichen: greener and lighter than the wall.
                    if (m.bench > 0.001f) {
                        const std::array<float, 3> moss{0.145f, 0.170f, 0.095f};
                        for (int i = 0; i < 3; ++i) rgb[i] += (moss[i] - rgb[i]) * m.bench * 0.75f;
                    }
                    // Scree fan below the gullies.
                    if (m.scree > 0.001f) {
                        const std::array<float, 3> talus{0.400f, 0.378f, 0.345f};
                        for (int i = 0; i < 3; ++i) rgb[i] += (talus[i] - rgb[i]) * m.scree * 0.8f;
                    }
                    // OSM bare rock: the splat's slope rule keeps grass on any
                    // gentle ground, and Aksla's crown and the skerries are flat
                    // bare gneiss. Pull to the rock layer's own colour so the
                    // painted patch matches the steep faces around it.
                    if (m.rock > 0.001f) {
                        const float pat = 0.90f + 0.20f * detail::geoFbm(x * 0.06f, z * 0.06f);
                        const std::array<float, 3> stone{0.340f, 0.330f, 0.310f};
                        for (int i = 0; i < 3; ++i)
                            rgb[i] += (stone[i] * pat - rgb[i]) * m.rock * 0.9f;
                    }
                    // OSM scrub / heath: brown-olive, the band between the
                    // gardens and the rock on Aksla's flanks.
                    if (m.heath > 0.001f) {
                        const float pat = 0.88f + 0.24f * detail::geoFbm(x * 0.04f, z * 0.04f);
                        const std::array<float, 3> hth{0.160f, 0.140f, 0.070f};
                        for (int i = 0; i < 3; ++i)
                            rgb[i] += (hth[i] * pat - rgb[i]) * m.heath * 0.85f;
                    }
                    // Wet seepage: wet rock is DARKER and slightly bluer than
                    // dry rock (the water film kills the diffuse bounce).
                    if (m.wet > 0.001f) {
                        const float k = m.wet * 0.62f;
                        rgb[0] *= 1.f - k;
                        rgb[1] *= 1.f - k * 0.94f;
                        rgb[2] *= 1.f - k * 0.86f;
                    }
                }
                if (urban) {
                    const float uw = urban->sample(x, z) * urbMax;
                    if (uw > 0.f) {
                        // Two-tone fabric: ~30 m fBm patches alternate between
                        // asphalt lots and gravel/paved yards, so towns get
                        // block-scale variation instead of one flat grey.
                        const float t = std::clamp(
                                (detail::geoFbm(x * 0.033f, z * 0.033f) - 0.35f) / 0.3f, 0.f, 1.f);
                        for (int i = 0; i < 3; ++i) {
                            const float urb = urbG[i] + (urbA[i] - urbG[i]) * t;
                            rgb[i] += (urb - rgb[i]) * uw;
                        }
                    }
                }
                if (paint) {
                    const float w = network.pavedWeight(x, z, edgeFeather);
                    rgb[0] += (roadCol[0] - rgb[0]) * w;
                    rgb[1] += (roadCol[1] - rgb[1]) * w;
                    rgb[2] += (roadCol[2] - rgb[2]) * w;
                }
            };
        }

        // Structure-band weights for the terrain shader's per-band texture
        // sets. Suppressed over painted pavement (asphalt is smooth — grass
        // clump / rock plate relief crawling across a road reads as damage)
        // and over urban ground (same reasoning; this also keeps TerrainScatter
        // tufts/stones out of town, since scatter samples these weights);
        // the same masks the paint uses, so they can never disagree.
        {
            const float edgeFeather = o.roadEdgeFeather;
            const bool paint = o.paintRoads;
            prov.weights = [rules, &network, urban, cover, edgeFeather, paint](float x, float z, float h,
                                                                               float slope, float* w4) {
                rules.evaluateWeights(x, z, h, slope, w4);
                // Land cover moves STRUCTURE too, not just colour: forest floor
                // and bench moss are soft (grass band 0), a talus fan is loose
                // stone (scree band 2). Colour-only would leave rock plates
                // crawling under the canopy paint.
                if (cover.active()) {
                    const auto m = cover.eval(x, z, h, slope);
                    const auto pushTo = [w4](int band, float t) {
                        if (t <= 0.001f) return;
                        for (int i = 0; i < 4; ++i) w4[i] *= 1.f - t;
                        w4[band] += t;
                    };
                    pushTo(0, std::min(0.9f, m.forest * 0.9f + m.bench * 0.6f));
                    pushTo(2, m.scree * 0.7f);
                    // Surveyed bare rock is the ROCK band (1): plates, not grass
                    // clumps — and TerrainScatter reads these weights, so this is
                    // also what keeps tufts off a rock slab.
                    pushTo(1, m.rock * 0.9f);
                    pushTo(0, m.heath * 0.6f);
                }
                float keep = 1.f;
                if (paint) keep *= 1.f - network.pavedWeight(x, z, edgeFeather);
                if (urban) keep *= 1.f - urban->sample(x, z);
                if (keep < 1.f) {
                    w4[0] *= keep;
                    w4[1] *= keep;
                    w4[2] *= keep;
                    w4[3] *= keep;
                }
            };
        }

        return prov;
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_GEOTERRAIN_HPP
