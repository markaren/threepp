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
//     slopes, snow up high with a feathered line; macro variation + gentle AO),
//     then tinted toward a gravel verge tone by corridorWeight so the ground
//     under/beside the ribbon reads as roadside.
//
// Both callbacks are pure and thread-safe (HeightGrid + RoadNetwork queries are
// read-only; the SplatRules is captured by value): safe for TileTerrain's async
// bake. The pack and the RoadNetwork must OUTLIVE the returned provider (the
// callbacks reference them); conformTo() must have run on the network first.
//
// Header-only, extras.

#ifndef THREEPP_EXTRAS_TERRAIN_GEOTERRAIN_HPP
#define THREEPP_EXTRAS_TERRAIN_GEOTERRAIN_HPP

#include "threepp/extras/road/RoadNetwork.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/extras/terrain/TerrainSplat.hpp"
#include "threepp/extras/terrain/TerrainTiles.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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

        // Roadside: tint the ground toward this gravel/verge tone by corridorWeight,
        // then (at high corridorWeight) push the tint target from gravel toward
        // `asphaltTone` so any far-LOD terrain that peeks through the pavement reads
        // as road rather than green. asphaltTone matches the ribbon's baked asphalt
        // (RoadGenerator asphaltColor, brightened for the ~1-spp grain average).
        std::array<float, 3> gravelTone = {0.30f, 0.28f, 0.24f};
        std::array<float, 3> asphaltTone = {0.11f, 0.11f, 0.115f};
        float roadsideTint = 0.85f;// max blend toward the roadside target at corridor centre
    };

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

    }// namespace detail

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
        std::vector<float> hardCap(n, std::numeric_limits<float>::max());

        net.forEachSegment([&](float ax, float az, float ha, float bx, float bz, float hb,
                               float pavedHalf, float /*corridorHalf*/) {
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
                    const float ceil = ha + (hb - ha) * t + road::RoadNetwork::kSurfaceRaise - o.clearance;
                    if (d < hardReach) hardCap[idx] = std::min(hardCap[idx], ceil);
                    if (d >= bestD[idx]) continue;
                    bestD[idx] = d;
                    allowed[idx] = ceil;
                    innerR[idx] = pavedHalf + o.inflate;
                }
            }
        });

        auto sstep = [](float e0, float e1, float x) {
            const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
            return t * t * (3.f - 2.f * t);
        };
        for (size_t i = 0; i < n; ++i) {
            if (bestD[i] != std::numeric_limits<float>::max() && h[i] > allowed[i]) {
                const float w = (bestD[i] <= innerR[i])
                                        ? 1.f
                                        : 1.f - sstep(innerR[i], innerR[i] + o.feather, bestD[i]);
                h[i] += (allowed[i] - h[i]) * w;// bench cut, feathered to natural
            }
            if (h[i] > hardCap[i]) h[i] = hardCap[i];// under-pavement guarantee
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
        wetland.color = {0.16f, 0.17f, 0.12f};
        wetland.slopeLo = 0.f; wetland.slopeHi = 0.30f; wetland.slopeFeather = 0.06f;
        wetland.heightLo = sea - 5.f;
        wetland.heightHi = sea + o.wetlandBand;
        wetland.heightFeather = 4.f;
        wetland.concaveBias = 0.4f;
        wetland.noiseAmpHeight = 2.5f;

        SplatLayer grass;// valley grass/heath on gentle low ground
        grass.color = {0.22f, 0.28f, 0.14f};
        grass.slopeLo = 0.f; grass.slopeHi = 0.34f; grass.slopeFeather = 0.06f;
        grass.heightHi = o.grassHeightMax;
        grass.heightFeather = 120.f;
        grass.concaveBias = 0.25f;// soil catches in hollows
        grass.noiseAmpSlope = 0.03f;
        grass.noiseAmpHeight = 40.f;

        SplatLayer heath;// brown heath/fell above the grass, still gentle
        heath.color = {0.30f, 0.27f, 0.17f};
        heath.slopeLo = 0.f; heath.slopeHi = 0.40f; heath.slopeFeather = 0.07f;
        heath.heightLo = o.grassHeightMax - 120.f;
        heath.heightHi = o.snowHeightMin;
        heath.heightFeather = 150.f;
        heath.noiseAmpHeight = 50.f;

        SplatLayer scree;// talus on medium slopes
        scree.color = {0.42f, 0.40f, 0.37f};
        scree.slopeLo = 0.34f; scree.slopeHi = 0.58f; scree.slopeFeather = 0.07f;
        scree.concaveBias = 0.9f;// collects in gullies
        scree.noiseAmpSlope = 0.03f;

        SplatLayer rock;// bare rock on the steep faces (fallback)
        rock.color = {0.34f, 0.33f, 0.31f};
        rock.slopeLo = 0.55f; rock.slopeHi = 1.f; rock.slopeFeather = 0.08f;
        rock.convexBias = 0.8f;   // bare on convex crests
        rock.weightFloor = 0.02f; // no texel resolves to grey

        SplatLayer snow;// snow up high, low slope, feathered line
        snow.color = {0.88f, 0.90f, 0.94f};
        snow.slopeLo = 0.f; snow.slopeHi = 0.60f; snow.slopeFeather = 0.10f;
        snow.heightLo = o.snowHeightMin;
        snow.heightFeather = o.snowFeather;
        snow.noiseAmpHeight = 70.f;// ragged snowline
        snow.noiseFreq = 0.03f;

        r.layers = {wetland, grass, heath, scree, rock, snow};
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
        const std::array<float, 3> gravel = o.gravelTone;
        const std::array<float, 3> asphalt = o.asphaltTone;
        const float tintMax = o.roadsideTint;

        TerrainProvider prov;

        // Height: pure bicubic of the (carved) DEM + corridor-faded sub-grid
        // relief. NO runtime road warp: the road cut is baked into the grid by
        // carveRoads, so the field is band-limited at DEM resolution and C1
        // everywhere — nothing sub-quad-scale for a tile split to reveal. The
        // relief also fades to zero approaching sea level: the DTM stores water
        // as a flat seaLevel sheet, and noise there would dither the seabed up
        // through the sea plane (patchy shoreline).
        const float sea = pack.region.seaLevel;
        prov.height = [&grid, &network, amp, freq, sea](float x, float z) {
            const float base = grid.sampleBicubic(x, z);
            const float cw = network.corridorWeight(x, z);
            if (cw >= 0.999f) return base;// fully paved — keep it dead smooth
            const float shore = std::clamp((base - (sea + 0.2f)) / 1.8f, 0.f, 1.f);
            if (shore <= 0.f) return base;
            const float relief = (detail::geoFbm(x * freq, z * freq) - 0.5f) * 2.f * amp;
            return base + relief * (1.f - cw) * shore * shore * (3.f - 2.f * shore);
        };

        // Albedo: Norwegian splat, tinted toward a roadside target inside the
        // corridor. The target ramps from gravel (verge, cw≈0.3) to true asphalt
        // (pavement, cw≈0.9) so residual far-LOD peek-through under the ribbon
        // reads as road, not green.
        prov.albedo = [rules, &network, gravel, asphalt, tintMax](float x, float z, float h, float slope, float* rgb) {
            const Rgb c = rules.evaluate(x, z, h, slope);
            const float cw = network.corridorWeight(x, z);
            // Roadside target: gravel → asphalt as we approach the paved band.
            const float sA = std::clamp((cw - 0.3f) / 0.6f, 0.f, 1.f);
            const float asphaltMix = sA * sA * (3.f - 2.f * sA);// smoothstep(0.3,0.9,cw)
            const float tx = gravel[0] + (asphalt[0] - gravel[0]) * asphaltMix;
            const float ty = gravel[1] + (asphalt[1] - gravel[1]) * asphaltMix;
            const float tz = gravel[2] + (asphalt[2] - gravel[2]) * asphaltMix;
            const float t = std::clamp(cw * tintMax, 0.f, 1.f);
            rgb[0] = c[0] + (tx - c[0]) * t;
            rgb[1] = c[1] + (ty - c[1]) * t;
            rgb[2] = c[2] + (tz - c[2]) * t;
        };

        return prov;
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_GEOTERRAIN_HPP
