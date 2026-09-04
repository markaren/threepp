// CONTOUR-STRIP CLIFF SHELL — a free-parametrised skin over the steep faces.
//
// WHY this exists at all. The terrain is a HEIGHTFIELD: every per-tile map the
// baker produces (macro albedo, band weights, world normal) is a function of
// (x, z). On a near-vertical wall a whole column of the surface projects onto
// ONE texel, so the tile maps stretch into vertical smears and — worse — no
// baked content can VARY along a column at all. That is the single defect the
// fjord shots kept hitting (plans/fjord-cliff-realism.md, phases 1/1b/2): the
// wall is dressed in stripes of stretched paint, and the ledge rows / seepage
// streaks a real gneiss wall shows can never appear, because the parametrisation
// has no room for them.
//
// The fix is a SEPARATE mesh over the steep faces with its OWN parametrisation:
//
//     u = arc length along a DEM contour (metres)
//     v = world height (metres)
//
// Both axes are metric on the wall, so texel density is uniform however vertical
// the face is, and content is free to vary along v. The shell is built by
// tracing contours at every `levelStep` of height inside a slope mask
// (marching squares), then stitching each contour to the one above it into quad
// strips. It sits `offset` metres proud of the terrain, so it simply hides the
// tile surface underneath — which means it does NOT have to be watertight:
// wherever the stitch cannot find a partner (a gap wider than `gapMax`, a
// contour that ends), the strip stops and the ordinary terrain shows through.
// Overlaps where contour lengths disagree are invisible (opaque, coincident
// shading).
//
// The bake is the point of the whole exercise: macro albedo + RGBA band weights
// are rasterized in (u, v) from the SHELL's own geometry — ledge rows keyed on
// the displaced normal's Y, wet streaks accumulated DOWN v from the top of each
// column, forest floor only where the pack's CHM says trees stand AND the row is
// a ledge. Fine structure still comes from the triplanar band layer, which needs
// no UVs (MaterialWithTerrainMaps; the Vulkan G-buffer band block applies to any
// mesh carrying those maps, not just tiles).
//
// Atlas layout, and why it is 1-D. A strip is only `levelStep` metres tall, and
// the target texel size is ~2 m, so a strip is ONE content texel row: the
// v-variation lives in the fact that the next strip is a different row of the
// atlas, at 2 m of height resolution. Each strip gets a 3-row band
// (gutter / content / gutter, the gutters being copies) so vertical bilinear
// anywhere inside the band returns the content row exactly and mip 1 still
// averages identical data. Horizontal neighbours in a band ARE arc neighbours,
// so horizontal filtering is correct by construction.
//
// Everything is deterministic (world-anchored hash noise + a seed) and the whole
// build is a pure function of the pack: nothing here mutates the pack or the
// provider.

#ifndef THREEPP_EXTRAS_TERRAIN_CLIFFSHELL_HPP
#define THREEPP_EXTRAS_TERRAIN_CLIFFSHELL_HPP

#include "threepp/extras/terrain/DetailTexture.hpp"
#include "threepp/extras/terrain/GeoTerrain.hpp"
#include "threepp/extras/terrain/GeoTerrainPack.hpp"
#include "threepp/materials/MeshStandardMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/textures/DataTexture.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace threepp::terrain {

    struct CliffShellOptions {
        float seaLevel = 0.f;

        // ── mask ────────────────────────────────────────────────────────────
        // Gradient MAGNITUDE (dh/dhorizontal), not the 0..1 "slope" the splat
        // rules use: tan(55°) = 1.43, tan(45°) = 1.00. Cells above `slopeOn`
        // seed the mask; the mask then grows through cells above `slopeOff`
        // (bounded — an unbounded hysteresis flood would swallow every
        // connected 45° cell in a 4 km pack) and is finally dilated a few cells
        // so the shell laps a little onto gentler ground instead of ending on
        // exactly the surface where it is most visible.
        float slopeOn = 1.43f;
        float slopeOff = 1.00f;
        int hysteresisCells = 12;
        int dilateCells = 3;

        // ── tracing ─────────────────────────────────────────────────────────
        float levelStep = 2.f;    // contour spacing in HEIGHT (m) = strip height
        float arcStep = 2.f;      // resample spacing along a contour (m)
        float smoothWindow = 4.f; // arc-window box filter (m) — also kills 1 m lidar fluting
        float minLength = 6.f;    // drop shorter contours (they are lidar noise)

        // ── stitching ───────────────────────────────────────────────────────
        // 0 → 2.5 × levelStep, i.e. 2.5× the horizontal run one level step takes
        // at slopeOff. Wider gaps are left OPEN (the terrain renders underneath).
        float gapMax = 0.f;

        // ── relief ──────────────────────────────────────────────────────────
        float offset = 0.35f;   // metres proud of the terrain along the DEM normal
        float benchAmp = 0.55f; // ledge riser, peak-to-peak (m)
        // Bench period in v (m), fBm-modulated between these. The floor is TWO
        // level steps at the 2 m default: a period sampled once per strip row
        // aliases into a random offset per row instead of reading as a ledge.
        float benchLo = 4.f;
        float benchHi = 9.f;
        float fbmAmp = 0.45f;   // 4-12 m fBm on top (m)
        float edgeFade = 4.f;   // displacement fades to 0 this near the mask boundary

        // ── bake ────────────────────────────────────────────────────────────
        float texelSize = 2.f;  // metres per macro/weight texel along the arc
        int atlasWidth = 2048;
        int atlasMaxRows = 2048;
        int regionMaxVerts = 400000;
        float snowHeightMin = 1200.f;
        float snowFeather = 90.f;
        float canopyForestMin = 2.f;
        float streakDecayLength = 70.f;// metres of fall for a wet streak to fade 1/e

        unsigned int seed = 7717u;

        // ── region of interest (0 half-extent = the whole pack) ─────────────
        // A 4 km pack is 16 M cells and a shot looks at part of it; the shell is
        // static geometry submitted whole, so the ROI box IS the cull.
        float centerX = 0.f, centerZ = 0.f, halfExtent = 0.f;
    };

    struct CliffShellStats {
        int regions = 0;
        int levels = 0;
        std::size_t maskCells = 0;
        std::size_t polylines = 0;
        std::size_t strips = 0;
        std::size_t vertices = 0;
        std::size_t triangles = 0;
        std::size_t atlasTexels = 0;
        float traceSeconds = 0.f;
        float stitchSeconds = 0.f;
        float bakeSeconds = 0.f;
    };

    namespace detail {

        // One stitched quad strip: a contiguous run of (base, partner) sample
        // pairs. `s` is arc length along the BASE contour (metres) — the u axis.
        struct ShellRun {
            int level = 0;// base level index (height = level * levelStep)
            std::vector<float> s;
            std::vector<std::array<float, 3>> a;// base row, displaced
            std::vector<std::array<float, 3>> b;// partner row, displaced
            bool flip = false;                  // triangle winding
            // atlas slot, filled by the packer
            int ax = 0, ay = 0, at = 0;// content x0, band y0, content texel count
        };

        // Smoothed DEM normal. `e` deliberately spans several cells: raw 1 m
        // lidar carries ±10 cm fluting whose gradient is large enough to tilt a
        // per-cell normal by tens of degrees, which would make the shell's
        // offset direction — and therefore its silhouette — flute too.
        inline std::array<float, 3> demNormal(const HeightGrid& g, float x, float z, float e = 3.f) {
            const float hx = g.sampleBilinear(x + e, z) - g.sampleBilinear(x - e, z);
            const float hz = g.sampleBilinear(x, z + e) - g.sampleBilinear(x, z - e);
            const float nx = -hx, ny = 2.f * e, nz = -hz;
            const float il = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz);
            return {nx * il, ny * il, nz * il};
        }

    }// namespace detail

    // Build the shell and add one Mesh per region under `parent`.
    //
    // `bands` MUST be the same band set the tiles use — the shell and the tiles
    // share the wall, and two different rock generators meeting at the shell
    // boundary would draw the boundary for the viewer.
    inline CliffShellStats buildCliffShell(Object3D& parent,
                                           const GeoTerrainPack& pack,
                                           const TerrainBandSet& bands,
                                           const CliffShellOptions& opt = {},
                                           float bandStrength = 1.f,
                                           float bandNormalScale = 2.2f,
                                           float heightBlend = 6.f) {
        using clock = std::chrono::high_resolution_clock;
        CliffShellStats st;
        const HeightGrid& grid = pack.grid;
        if (!grid.valid()) return st;

        const int N = grid.dim();
        const float world = grid.worldSize();
        const float cell = world / static_cast<float>(N - 1);
        const float half = world * 0.5f;
        const std::vector<float>& H = grid.data();
        const float lstep = std::max(0.25f, opt.levelStep);
        const float gapMax = opt.gapMax > 0.f ? opt.gapMax : 2.5f * lstep;

        auto WX = [&](int ix) { return -half + static_cast<float>(ix) * cell; };
        auto WZ = [&](int iz) { return -half + static_cast<float>(iz) * cell; };
        auto at = [&](int ix, int iz) {
            return H[static_cast<std::size_t>(iz) * N + ix];
        };

        // ── ROI window in grid indices ──────────────────────────────────────
        int i0 = 0, i1 = N - 1, j0 = 0, j1 = N - 1;
        if (opt.halfExtent > 0.f) {
            auto toIx = [&](float w) {
                return std::clamp(static_cast<int>(std::floor((w + half) / cell)), 0, N - 1);
            };
            i0 = toIx(opt.centerX - opt.halfExtent);
            i1 = toIx(opt.centerX + opt.halfExtent);
            j0 = toIx(opt.centerZ - opt.halfExtent);
            j1 = toIx(opt.centerZ + opt.halfExtent);
        }
        const int rw = i1 - i0 + 1, rh = j1 - j0 + 1;
        if (rw < 8 || rh < 8) return st;

        const auto t0 = clock::now();

        // ── slope mask: seed → bounded hysteresis → dilate → drop the sea ────
        std::vector<std::uint8_t> mask(static_cast<std::size_t>(rw) * rh, 0);
        std::vector<std::uint8_t> weak(mask.size(), 0);
        {
            const float inv2c = 1.f / (2.f * cell);
            for (int jz = 0; jz < rh; ++jz) {
                const int iz = j0 + jz;
                for (int jx = 0; jx < rw; ++jx) {
                    const int ix = i0 + jx;
                    const int xm = std::max(ix - 1, 0), xp = std::min(ix + 1, N - 1);
                    const int zm = std::max(iz - 1, 0), zp = std::min(iz + 1, N - 1);
                    const float gx = (at(xp, iz) - at(xm, iz)) * inv2c;
                    const float gz = (at(ix, zp) - at(ix, zm)) * inv2c;
                    const float g = std::sqrt(gx * gx + gz * gz);
                    const std::size_t k = static_cast<std::size_t>(jz) * rw + jx;
                    if (g > opt.slopeOff) weak[k] = 1;
                    if (g > opt.slopeOn) mask[k] = 1;
                }
            }
            // Bounded multi-source BFS: `dist` cells of growth through `weak`,
            // then `dilateCells` more through anything.
            std::vector<std::int16_t> dist(mask.size(), -1);
            std::deque<int> q;
            for (std::size_t k = 0; k < mask.size(); ++k)
                if (mask[k]) { dist[k] = 0; q.push_back(static_cast<int>(k)); }
            const int maxD = opt.hysteresisCells + opt.dilateCells;
            while (!q.empty()) {
                const int k = q.front();
                q.pop_front();
                const int d = dist[k];
                if (d >= maxD) continue;
                const int jx = k % rw, jz = k / rw;
                for (int e = 0; e < 4; ++e) {
                    const int nx = jx + (e == 0) - (e == 1);
                    const int nz = jz + (e == 2) - (e == 3);
                    if (nx < 0 || nz < 0 || nx >= rw || nz >= rh) continue;
                    const std::size_t nk = static_cast<std::size_t>(nz) * rw + nx;
                    if (dist[nk] >= 0) continue;
                    // Inside the hysteresis budget only steep-ish ground grows;
                    // the last few cells are a plain dilation so the shell laps
                    // over the break of slope.
                    if (d + 1 <= opt.hysteresisCells && !weak[nk]) continue;
                    dist[nk] = static_cast<std::int16_t>(d + 1);
                    mask[nk] = 1;
                    q.push_back(static_cast<int>(nk));
                }
            }
            for (int jz = 0; jz < rh; ++jz)
                for (int jx = 0; jx < rw; ++jx) {
                    const std::size_t k = static_cast<std::size_t>(jz) * rw + jx;
                    if (mask[k] && at(i0 + jx, j0 + jz) <= opt.seaLevel + 0.5f) mask[k] = 0;
                }
        }
        std::vector<std::uint8_t>().swap(weak);

        // Distance (in cells) from the mask boundary — the displacement fade, so
        // the shell meets the terrain flush instead of with a lip.
        std::vector<std::uint8_t> edgeDist(mask.size(), 0);
        {
            constexpr int kCap = 40;
            std::deque<int> q;
            for (std::size_t k = 0; k < mask.size(); ++k)
                if (!mask[k]) q.push_back(static_cast<int>(k));
            // Border cells of the ROI count as outside too.
            std::vector<std::uint8_t> seen(mask.size(), 0);
            for (int k : q) seen[k] = 1;
            int d = 0;
            while (!q.empty() && d < kCap) {
                const std::size_t n = q.size();
                for (std::size_t s = 0; s < n; ++s) {
                    const int k = q.front();
                    q.pop_front();
                    const int jx = k % rw, jz = k / rw;
                    for (int e = 0; e < 4; ++e) {
                        const int nx = jx + (e == 0) - (e == 1);
                        const int nz = jz + (e == 2) - (e == 3);
                        if (nx < 0 || nz < 0 || nx >= rw || nz >= rh) continue;
                        const std::size_t nk = static_cast<std::size_t>(nz) * rw + nx;
                        if (seen[nk]) continue;
                        seen[nk] = 1;
                        edgeDist[nk] = static_cast<std::uint8_t>(std::min(d + 1, 255));
                        q.push_back(static_cast<int>(nk));
                    }
                }
                ++d;
            }
            for (std::size_t k = 0; k < mask.size(); ++k)
                if (mask[k] && !seen[k]) edgeDist[k] = kCap;
        }
        for (std::size_t k = 0; k < mask.size(); ++k) st.maskCells += mask[k];
        if (st.maskCells < 64) return st;

        auto maskAt = [&](int jx, int jz) {
            if (jx < 0 || jz < 0 || jx >= rw || jz >= rh) return false;
            return mask[static_cast<std::size_t>(jz) * rw + jx] != 0;
        };
        auto edgeMetres = [&](float x, float z) {
            const int jx = static_cast<int>(std::lround((x + half) / cell)) - i0;
            const int jz = static_cast<int>(std::lround((z + half) / cell)) - j0;
            if (jx < 0 || jz < 0 || jx >= rw || jz >= rh) return 0.f;
            return static_cast<float>(edgeDist[static_cast<std::size_t>(jz) * rw + jx]) * cell;
        };

        // ── level buckets: visit a cell only for the levels that cross it ────
        // Sweeping all ROI cells per level would be 16 M × 700 tests.
        int kMin = INT_MAX, kMax = INT_MIN;
        std::vector<std::vector<std::uint32_t>> bucket;
        {
            std::vector<std::uint32_t> flat;
            std::vector<std::pair<int, int>> range;// per active cell
            range.reserve(st.maskCells);
            flat.reserve(st.maskCells);
            for (int jz = 0; jz + 1 < rh; ++jz)
                for (int jx = 0; jx + 1 < rw; ++jx) {
                    if (!(maskAt(jx, jz) && maskAt(jx + 1, jz) && maskAt(jx, jz + 1) && maskAt(jx + 1, jz + 1)))
                        continue;
                    const float h00 = at(i0 + jx, j0 + jz), h10 = at(i0 + jx + 1, j0 + jz);
                    const float h01 = at(i0 + jx, j0 + jz + 1), h11 = at(i0 + jx + 1, j0 + jz + 1);
                    const float lo = std::min(std::min(h00, h10), std::min(h01, h11));
                    const float hi = std::max(std::max(h00, h10), std::max(h01, h11));
                    const int a = static_cast<int>(std::floor(lo / lstep)) + 1;
                    const int b = static_cast<int>(std::floor(hi / lstep));
                    if (b < a) continue;
                    flat.push_back(static_cast<std::uint32_t>(jz) * static_cast<std::uint32_t>(rw) + static_cast<std::uint32_t>(jx));
                    range.emplace_back(a, b);
                    kMin = std::min(kMin, a);
                    kMax = std::max(kMax, b);
                }
            if (flat.empty()) return st;
            bucket.assign(static_cast<std::size_t>(kMax - kMin + 1), {});
            for (std::size_t c = 0; c < flat.size(); ++c)
                for (int k = range[c].first; k <= range[c].second; ++k)
                    bucket[static_cast<std::size_t>(k - kMin)].push_back(flat[c]);
        }
        st.levels = kMax - kMin + 1;

        // ── one level's contours ────────────────────────────────────────────
        struct Poly {
            std::vector<float> x, z, s;// s = cumulative arc length
        };
        std::unordered_map<std::uint64_t, int> edgePt;
        std::vector<float> px, pz;
        std::vector<std::array<int, 2>> nbr;

        auto traceLevel = [&](int k, std::vector<Poly>& out) {
            const float lev = static_cast<float>(k) * lstep;
            edgePt.clear();
            px.clear();
            pz.clear();
            nbr.clear();
            auto edge = [&](std::uint64_t key, float ax, float az, float bx, float bz,
                            float ha, float hb) {
                auto it = edgePt.find(key);
                if (it != edgePt.end()) return it->second;
                const float d = hb - ha;
                const float t = std::abs(d) < 1e-6f ? 0.5f : std::clamp((lev - ha) / d, 0.f, 1.f);
                const int id = static_cast<int>(px.size());
                px.push_back(ax + (bx - ax) * t);
                pz.push_back(az + (bz - az) * t);
                nbr.push_back({-1, -1});
                edgePt.emplace(key, id);
                return id;
            };
            auto link = [&](int a, int b) {
                if (a < 0 || b < 0 || a == b) return;
                if (nbr[a][0] < 0) nbr[a][0] = b;
                else if (nbr[a][1] < 0 && nbr[a][0] != b) nbr[a][1] = b;
                if (nbr[b][0] < 0) nbr[b][0] = a;
                else if (nbr[b][1] < 0 && nbr[b][0] != a) nbr[b][1] = a;
            };

            for (std::uint32_t cid : bucket[static_cast<std::size_t>(k - kMin)]) {
                const int jx = static_cast<int>(cid % static_cast<std::uint32_t>(rw));
                const int jz = static_cast<int>(cid / static_cast<std::uint32_t>(rw));
                const int ix = i0 + jx, iz = j0 + jz;
                const float h0 = at(ix, iz), h1 = at(ix + 1, iz);
                const float h2 = at(ix + 1, iz + 1), h3 = at(ix, iz + 1);
                const int b = (h0 > lev ? 1 : 0) | (h1 > lev ? 2 : 0) | (h2 > lev ? 4 : 0) | (h3 > lev ? 8 : 0);
                if (b == 0 || b == 15) continue;
                const float x0 = WX(ix), x1 = WX(ix + 1), z0 = WZ(iz), z1 = WZ(iz + 1);
                const std::uint64_t base = (static_cast<std::uint64_t>(iz) * N + ix) * 2;
                auto E0 = [&] { return edge(base + 0, x0, z0, x1, z0, h0, h1); };
                auto E1 = [&] { return edge((static_cast<std::uint64_t>(iz) * N + ix + 1) * 2 + 1, x1, z0, x1, z1, h1, h2); };
                auto E2 = [&] { return edge((static_cast<std::uint64_t>(iz + 1) * N + ix) * 2 + 0, x0, z1, x1, z1, h3, h2); };
                auto E3 = [&] { return edge(base + 1, x0, z0, x0, z1, h0, h3); };
                switch (b) {
                    case 1: case 14: link(E3(), E0()); break;
                    case 2: case 13: link(E0(), E1()); break;
                    case 3: case 12: link(E3(), E1()); break;
                    case 4: case 11: link(E1(), E2()); break;
                    case 6: case 9: link(E0(), E2()); break;
                    case 7: case 8: link(E3(), E2()); break;
                    // Saddles: resolve by the bilinear centre value, so the two
                    // segments never cross and a ridge stays a ridge.
                    case 5: {
                        const float c = 0.25f * (h0 + h1 + h2 + h3);
                        if (c > lev) { link(E3(), E2()); link(E0(), E1()); }
                        else { link(E3(), E0()); link(E1(), E2()); }
                        break;
                    }
                    case 10: {
                        const float c = 0.25f * (h0 + h1 + h2 + h3);
                        if (c > lev) { link(E0(), E3()); link(E1(), E2()); }
                        else { link(E0(), E1()); link(E2(), E3()); }
                        break;
                    }
                    default: break;
                }
            }
            if (px.empty()) return;

            std::vector<std::uint8_t> used(px.size(), 0);
            std::vector<int> chain;
            auto walk = [&](int start) {
                chain.clear();
                int prev = -1, cur = start;
                while (cur >= 0 && !used[cur]) {
                    used[cur] = 1;
                    chain.push_back(cur);
                    const int n0 = nbr[cur][0], n1 = nbr[cur][1];
                    const int nx = (n0 != prev && n0 >= 0) ? n0 : n1;
                    prev = cur;
                    cur = nx;
                }
            };
            auto emit = [&] {
                if (chain.size() < 3) return;
                // Arc-window box smoothing (also removes the 1 m lidar fluting
                // the contour inherits from the raw DEM), then uniform resample.
                std::vector<float> cx(chain.size()), cz(chain.size()), cs(chain.size(), 0.f);
                for (std::size_t i = 0; i < chain.size(); ++i) { cx[i] = px[chain[i]]; cz[i] = pz[chain[i]]; }
                for (std::size_t i = 1; i < cx.size(); ++i)
                    cs[i] = cs[i - 1] + std::hypot(cx[i] - cx[i - 1], cz[i] - cz[i - 1]);
                const float L = cs.back();
                if (L < opt.minLength) return;
                std::vector<float> sx(cx.size()), sz(cz.size());
                const float hw = 0.5f * opt.smoothWindow;
                std::size_t lo = 0, hi = 0;
                for (std::size_t i = 0; i < cx.size(); ++i) {
                    while (cs[lo] < cs[i] - hw) ++lo;
                    while (hi + 1 < cx.size() && cs[hi + 1] <= cs[i] + hw) ++hi;
                    // The window must always contain i itself — with a spacing
                    // wider than the window the two pointers would cross.
                    lo = std::min(lo, i);
                    hi = std::max(hi, i);
                    float ax = 0.f, az = 0.f;
                    for (std::size_t j = lo; j <= hi; ++j) { ax += cx[j]; az += cz[j]; }
                    const float inv = 1.f / static_cast<float>(hi - lo + 1);
                    sx[i] = ax * inv;
                    sz[i] = az * inv;
                }
                Poly p;
                const int n = std::max(2, static_cast<int>(std::lround(L / opt.arcStep)) + 1);
                p.x.reserve(n); p.z.reserve(n); p.s.reserve(n);
                std::size_t seg = 0;
                for (int i = 0; i < n; ++i) {
                    const float target = L * static_cast<float>(i) / static_cast<float>(n - 1);
                    while (seg + 2 < cs.size() && cs[seg + 1] < target) ++seg;
                    const float d = cs[seg + 1] - cs[seg];
                    const float t = d > 1e-6f ? std::clamp((target - cs[seg]) / d, 0.f, 1.f) : 0.f;
                    p.x.push_back(sx[seg] + (sx[seg + 1] - sx[seg]) * t);
                    p.z.push_back(sz[seg] + (sz[seg + 1] - sz[seg]) * t);
                    p.s.push_back(target);
                }
                out.push_back(std::move(p));
            };
            // Open chains first (their ends must not be swallowed mid-walk),
            // then whatever is left is a closed loop.
            for (std::size_t i = 0; i < px.size(); ++i)
                if (!used[i] && (nbr[i][0] < 0 || nbr[i][1] < 0)) { walk(static_cast<int>(i)); emit(); }
            for (std::size_t i = 0; i < px.size(); ++i)
                if (!used[i]) { walk(static_cast<int>(i)); emit(); }
        };

        // ── trace + stitch, descending so the partner level is already in hand ──
        std::vector<detail::ShellRun> runs;
        std::vector<Poly> upper, lower;
        // Flat (x, z, polyline, index) view of the partner level + a spatial
        // hash over it. Flat, not (poly<<bits | vertex): a resampled contour on
        // a 4 km pack can easily carry tens of thousands of vertices, which no
        // bit-packed vertex field survives.
        std::vector<float> uhx, uhz;
        std::vector<std::uint32_t> uhp, uhi;
        std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> upperHash;
        float stitchSec = 0.f;

        auto hashKey = [&](float x, float z) {
            const auto cx = static_cast<std::int64_t>(std::floor(x / gapMax));
            const auto cz = static_cast<std::int64_t>(std::floor(z / gapMax));
            return (static_cast<std::uint64_t>(cx + 0x40000000) << 32) ^ static_cast<std::uint64_t>(cz + 0x40000000);
        };
        auto rebuildHash = [&] {
            upperHash.clear();
            uhx.clear(); uhz.clear(); uhp.clear(); uhi.clear();
            for (std::size_t p = 0; p < upper.size(); ++p)
                for (std::size_t i = 0; i < upper[p].x.size(); ++i) {
                    upperHash[hashKey(upper[p].x[i], upper[p].z[i])].push_back(static_cast<std::uint32_t>(uhx.size()));
                    uhx.push_back(upper[p].x[i]);
                    uhz.push_back(upper[p].z[i]);
                    uhp.push_back(static_cast<std::uint32_t>(p));
                    uhi.push_back(static_cast<std::uint32_t>(i));
                }
        };

        traceLevel(kMax, upper);
        st.polylines += upper.size();
        rebuildHash();

        for (int k = kMax - 1; k >= kMin; --k) {
            lower.clear();
            traceLevel(k, lower);
            st.polylines += lower.size();

            const auto ts = clock::now();
            const float yA = static_cast<float>(k) * lstep;
            const float yB = static_cast<float>(k + 1) * lstep;
            for (const Poly& p : lower) {
                const std::size_t n = p.x.size();
                std::vector<std::array<float, 2>> partner(n);
                std::vector<std::uint8_t> ok(n, 0);
                for (std::size_t i = 0; i < n; ++i) {
                    const float qx = p.x[i], qz = p.z[i];
                    float best = gapMax * gapMax;
                    int bp = -1, bi = -1;
                    const auto cx = static_cast<std::int64_t>(std::floor(qx / gapMax));
                    const auto cz = static_cast<std::int64_t>(std::floor(qz / gapMax));
                    for (int dz = -1; dz <= 1; ++dz)
                        for (int dx = -1; dx <= 1; ++dx) {
                            const std::uint64_t key =
                                    (static_cast<std::uint64_t>(cx + dx + 0x40000000) << 32) ^
                                    static_cast<std::uint64_t>(cz + dz + 0x40000000);
                            auto it = upperHash.find(key);
                            if (it == upperHash.end()) continue;
                            for (std::uint32_t v : it->second) {
                                const float ddx = uhx[v] - qx, ddz = uhz[v] - qz;
                                const float d2 = ddx * ddx + ddz * ddz;
                                if (d2 < best) { best = d2; bp = static_cast<int>(uhp[v]); bi = static_cast<int>(uhi[v]); }
                            }
                        }
                    if (bp < 0) continue;
                    // Refine onto the two adjacent segments — nearest VERTEX
                    // alone makes the partner row jitter at the resample rate.
                    const auto& up = upper[bp];
                    float bx = up.x[bi], bz = up.z[bi];
                    for (int e = -1; e <= 0; ++e) {
                        const int a = bi + e, b = bi + e + 1;
                        if (a < 0 || b >= static_cast<int>(up.x.size())) continue;
                        const float ex = up.x[b] - up.x[a], ez = up.z[b] - up.z[a];
                        const float len2 = ex * ex + ez * ez;
                        if (len2 < 1e-8f) continue;
                        const float t = std::clamp(((qx - up.x[a]) * ex + (qz - up.z[a]) * ez) / len2, 0.f, 1.f);
                        const float cxp = up.x[a] + ex * t, czp = up.z[a] + ez * t;
                        const float d2 = (cxp - qx) * (cxp - qx) + (czp - qz) * (czp - qz);
                        if (d2 < best) { best = d2; bx = cxp; bz = czp; }
                    }
                    partner[i] = {bx, bz};
                    ok[i] = 1;
                }
                // Cut the polyline into contiguous runs of valid pairs. A gap is
                // fine: the terrain still renders under the shell.
                std::size_t i = 0;
                while (i < n) {
                    if (!ok[i]) { ++i; continue; }
                    std::size_t j = i;
                    while (j + 1 < n && ok[j + 1] &&
                           std::hypot(partner[j + 1][0] - partner[j][0], partner[j + 1][1] - partner[j][1]) < 4.f * gapMax)
                        ++j;
                    if (j > i && p.s[j] - p.s[i] >= opt.minLength) {
                        detail::ShellRun r;
                        r.level = k;
                        const std::size_t m = j - i + 1;
                        r.s.reserve(m); r.a.reserve(m); r.b.reserve(m);
                        for (std::size_t q = i; q <= j; ++q) {
                            r.s.push_back(p.s[q] - p.s[i]);
                            r.a.push_back({p.x[q], yA, p.z[q]});
                            r.b.push_back({partner[q][0], yB, partner[q][1]});
                        }
                        runs.push_back(std::move(r));
                    }
                    i = j + 1;
                }
            }
            stitchSec += std::chrono::duration<float>(clock::now() - ts).count();
            upper.swap(lower);
            rebuildHash();
        }
        st.strips = runs.size();
        st.traceSeconds = std::chrono::duration<float>(clock::now() - t0).count() - stitchSec;
        st.stitchSeconds = stitchSec;
        if (runs.empty()) return st;

        // ── displacement (world-anchored, deterministic) ─────────────────────
        // A bench term whose period is 2.5-8 m in v with an asymmetric ramp (a
        // long outward tread, a short pull-in riser) — that shape is what makes
        // the ledges read as HORIZONTAL rows rather than as noise — plus 4-12 m
        // fBm. Amplitude fades at the mask boundary so the shell meets the
        // terrain flush.
        const float sph = static_cast<float>(opt.seed & 0xffu) * 0.0173f;
        auto displace = [&](float x, float y, float z) {
            const float period = opt.benchLo + (opt.benchHi - opt.benchLo) *
                                                       detail::geoFbm(x * 0.018f + sph, z * 0.018f);
            float t = (y + 3.f * detail::geoFbm(x * 0.03f, z * 0.03f)) / period;
            t -= std::floor(t);
            const float ramp = t < 0.75f ? t / 0.75f : (1.f - t) / 0.25f;
            const float bench = opt.benchAmp * (ramp - 0.5f);
            const float f = 0.5f * (detail::geoFbm(x * 0.10f, y * 0.11f) +
                                    detail::geoFbm(z * 0.10f + 37.f, y * 0.09f));
            const float fbm = opt.fbmAmp * (f - 0.5f) * 2.f;
            return (bench + fbm) * detail::smoothstep01(0.f, opt.edgeFade, edgeMetres(x, z));
        };
        auto place = [&](std::array<float, 3>& p) {
            const auto n = detail::demNormal(grid, p[0], p[2]);
            const float d = opt.offset + displace(p[0], p[1], p[2]);
            p[0] += n[0] * d;
            p[1] += n[1] * d;
            p[2] += n[2] * d;
        };
        for (auto& r : runs) {
            for (auto& v : r.a) place(v);
            for (auto& v : r.b) place(v);
            // Winding: the shell must face AWAY from the hill. One test per run
            // (the contour orientation is consistent along it) beats per-quad —
            // a per-quad test flips faces wherever the displacement noise wins.
            const std::size_t m = r.a.size() / 2;
            if (m + 1 < r.a.size()) {
                const auto& A = r.a[m];
                const auto& A1 = r.a[m + 1];
                const auto& B1 = r.b[m + 1];
                const float ux = A1[0] - A[0], uy = A1[1] - A[1], uz = A1[2] - A[2];
                const float vx = B1[0] - A[0], vy = B1[1] - A[1], vz = B1[2] - A[2];
                const float fx = uy * vz - uz * vy, fy = uz * vx - ux * vz, fz = ux * vy - uy * vx;
                const auto nd = detail::demNormal(grid, A[0], A[2]);
                r.flip = (fx * nd[0] + fy * nd[1] + fz * nd[2]) < 0.f;
            }
        }

        // ── atlas packing: one 3-row band per run (gutter/content/gutter) ────
        const float texel = std::max(0.25f, opt.texelSize);
        const int AW = std::max(64, opt.atlasWidth);
        constexpr int kBandH = 3;
        struct Region {
            std::vector<std::size_t> runIdx;
            int rows = 0;
        };
        std::vector<Region> regions;
        {
            Region cur;
            int cx = 0, cy = 0, used = 0;
            std::size_t verts = 0;
            for (std::size_t ri = 0; ri < runs.size(); ++ri) {
                auto& r = runs[ri];
                const float L = r.s.back();
                const int T = std::clamp(static_cast<int>(std::lround(L / texel)) + 1, 2, AW - 2);
                const int need = T + 2;// 1 gutter texel each side
                int nx = cx, ny = cy;
                if (nx + need > AW) { nx = 0; ny += kBandH; }
                const bool full = (ny + kBandH > opt.atlasMaxRows) ||
                                  (verts + 2 * r.a.size() > static_cast<std::size_t>(opt.regionMaxVerts));
                if (full && !cur.runIdx.empty()) {
                    cur.rows = used;
                    regions.push_back(std::move(cur));
                    cur = Region{};
                    nx = 0; ny = 0; used = 0; verts = 0;
                }
                cx = nx; cy = ny;
                r.ax = cx + 1;
                r.ay = cy;
                r.at = T;
                cx += need;
                used = cy + kBandH;
                verts += 2 * r.a.size();
                cur.runIdx.push_back(ri);
            }
            if (!cur.runIdx.empty()) { cur.rows = used; regions.push_back(std::move(cur)); }
        }

        // ── bake + mesh, per region ─────────────────────────────────────────
        const auto tb0 = clock::now();
        // Wet-streak carrier: a coarse (x,z) column grid walked from the TOP of
        // the shell downwards. On a wall the fall line IS a near-constant column,
        // so a column cell is the right carrier; the streak is SEEDED once (the
        // first, i.e. highest, level that touches the column) from the pack's
        // flow accumulation plus a per-column noise, then decays as it runs down.
        const float colCell = 4.f;
        const int cw = static_cast<int>(std::ceil(world / colCell)) + 1;
        std::vector<float> colVal(static_cast<std::size_t>(cw) * cw, 0.f);
        std::vector<std::int32_t> colLevel(colVal.size(), std::numeric_limits<std::int32_t>::min());
        const float decay = std::exp(-lstep / std::max(1.f, opt.streakDecayLength));
        const std::array<float, 3> rockCol{0.34f, 0.33f, 0.31f};
        const std::array<float, 3> mossCol{0.145f, 0.170f, 0.095f};
        const std::array<float, 3> floorCol{0.070f, 0.095f, 0.055f};
        const std::array<float, 3> snowCol{0.88f, 0.90f, 0.94f};

        auto shell = Group::create();
        shell->name = "cliff_shell";
        for (const Region& reg : regions) {
            const int AH = std::max(kBandH, reg.rows);
            std::vector<float> pos, nrm, uv;
            std::vector<unsigned int> idx;
            std::vector<std::uint8_t> macro(static_cast<std::size_t>(AW) * AH * 4, 0);
            std::vector<std::uint8_t> wts(macro.size(), 0);
            std::size_t base = 0;
            std::vector<std::size_t> runBase(reg.runIdx.size());

            for (std::size_t q = 0; q < reg.runIdx.size(); ++q) {
                const auto& r = runs[reg.runIdx[q]];
                runBase[q] = base;
                const float L = r.s.back();
                const std::size_t n = r.a.size();
                const float vRow = (static_cast<float>(r.ay) + 1.5f) / static_cast<float>(AH);
                for (std::size_t i = 0; i < n; ++i) {
                    // u: arc length, mapped onto this run's atlas texel run.
                    const float t = L > 1e-5f ? r.s[i] / L : 0.f;
                    const float ux = (static_cast<float>(r.ax) + t * static_cast<float>(r.at - 1) + 0.5f) /
                                     static_cast<float>(AW);
                    pos.insert(pos.end(), {r.a[i][0], r.a[i][1], r.a[i][2]});
                    uv.insert(uv.end(), {ux, vRow});
                    pos.insert(pos.end(), {r.b[i][0], r.b[i][1], r.b[i][2]});
                    uv.insert(uv.end(), {ux, vRow});
                }
                for (std::size_t i = 0; i + 1 < n; ++i) {
                    const auto a0 = static_cast<unsigned int>(base + 2 * i);
                    const auto b0 = a0 + 1, a1 = a0 + 2, b1 = a0 + 3;
                    if (!r.flip) idx.insert(idx.end(), {a0, a1, b1, a0, b1, b0});
                    else idx.insert(idx.end(), {a0, b1, a1, a0, b0, b1});
                }
                base += 2 * n;
            }
            nrm.assign(pos.size(), 0.f);
            for (std::size_t t = 0; t + 2 < idx.size(); t += 3) {
                const unsigned int ia = idx[t] * 3, ib = idx[t + 1] * 3, ic = idx[t + 2] * 3;
                const float ux = pos[ib] - pos[ia], uy = pos[ib + 1] - pos[ia + 1], uz = pos[ib + 2] - pos[ia + 2];
                const float vx = pos[ic] - pos[ia], vy = pos[ic + 1] - pos[ia + 1], vz = pos[ic + 2] - pos[ia + 2];
                const float fx = uy * vz - uz * vy, fy = uz * vx - ux * vz, fz = ux * vy - uy * vx;
                for (unsigned int o : {ia, ib, ic}) { nrm[o] += fx; nrm[o + 1] += fy; nrm[o + 2] += fz; }
            }
            for (std::size_t i = 0; i + 2 < nrm.size(); i += 3) {
                const float l = std::sqrt(nrm[i] * nrm[i] + nrm[i + 1] * nrm[i + 1] + nrm[i + 2] * nrm[i + 2]);
                if (l > 1e-8f) { nrm[i] /= l; nrm[i + 1] /= l; nrm[i + 2] /= l; }
                else nrm[i + 1] = 1.f;
            }

            // Rasterize the content row of every run. Runs come in DESCENDING
            // level order (that is how they were traced), which is exactly the
            // order the streak carrier needs.
            for (std::size_t q = 0; q < reg.runIdx.size(); ++q) {
                const auto& r = runs[reg.runIdx[q]];
                const std::size_t vb = runBase[q];
                const float L = r.s.back();
                const std::size_t n = r.a.size();
                std::size_t seg = 0;
                for (int t = 0; t < r.at; ++t) {
                    const float target = L * static_cast<float>(t) / static_cast<float>(std::max(1, r.at - 1));
                    while (seg + 2 < n && r.s[seg + 1] < target) ++seg;
                    const float d = r.s[seg + 1] - r.s[seg];
                    const float f = d > 1e-6f ? std::clamp((target - r.s[seg]) / d, 0.f, 1.f) : 0.f;
                    auto lerpV = [&](std::size_t comp, std::size_t o) {
                        const std::size_t i0v = (vb + 2 * seg + o) * 3 + comp;
                        const std::size_t i1v = (vb + 2 * (seg + 1) + o) * 3 + comp;
                        return pos[i0v] + (pos[i1v] - pos[i0v]) * f;
                    };
                    auto lerpN = [&](std::size_t comp) {
                        const std::size_t i0v = (vb + 2 * seg) * 3 + comp;
                        const std::size_t i1v = (vb + 2 * (seg + 1)) * 3 + comp;
                        return nrm[i0v] + (nrm[i1v] - nrm[i0v]) * f;
                    };
                    const float wx = 0.5f * (lerpV(0, 0) + lerpV(0, 1));
                    const float wy = 0.5f * (lerpV(1, 0) + lerpV(1, 1));
                    const float wz = 0.5f * (lerpV(2, 0) + lerpV(2, 1));
                    float nx = lerpN(0), ny = lerpN(1), nz = lerpN(2);
                    const float nl = std::sqrt(nx * nx + ny * ny + nz * nz);
                    if (nl > 1e-6f) { nx /= nl; ny /= nl; nz /= nl; }

                    // ── content, all of it from the SHELL, none of it from (x,z) maps ──
                    // "Ledge" = the tread of a bench, and on a 70° wall a tread
                    // that a bench 0.7 m tall cuts into a 2 m strip only reaches
                    // Ny ≈ 0.25-0.5. Gating at 0.45 (the flat-ground intuition)
                    // measured out as a wall with NO moss anywhere — the whole
                    // land-cover layer went missing, which was worse than the
                    // smear it replaced.
                    const float ledge = detail::smoothstep01(0.20f, 0.52f, ny);
                    const float canopyH = pack.hasCanopy() ? pack.canopy.sampleBilinear(wx, wz) : 0.f;
                    const float forest = detail::smoothstep01(opt.canopyForestMin,
                                                              opt.canopyForestMin + 4.f, canopyH);
                    const float snowW = detail::smoothstep01(opt.snowHeightMin,
                                                             opt.snowHeightMin + opt.snowFeather, wy) *
                                        detail::smoothstep01(0.25f, 0.55f, ny);

                    // Wet streak: seed once at the top of the column, then run down.
                    const float wanderX = (detail::geoVNoise(wx * 0.03f, wy * 0.07f) - 0.5f) * 5.f;
                    const float wanderZ = (detail::geoVNoise(wz * 0.03f + 11.f, wy * 0.07f) - 0.5f) * 5.f;
                    const int ccx = std::clamp(static_cast<int>((wx + wanderX + half) / colCell), 0, cw - 1);
                    const int ccz = std::clamp(static_cast<int>((wz + wanderZ + half) / colCell), 0, cw - 1);
                    const std::size_t ck = static_cast<std::size_t>(ccz) * cw + ccx;
                    float wet;
                    if (colLevel[ck] == r.level) {
                        wet = colVal[ck];
                    } else if (colLevel[ck] > r.level) {
                        const int drop = colLevel[ck] - r.level;
                        wet = colVal[ck] * std::pow(decay, static_cast<float>(drop));
                        colVal[ck] = wet;
                        colLevel[ck] = r.level;
                    } else {
                        const float fl = pack.hasFlow() ? pack.flow.sampleBilinear(wx, wz) : 0.f;
                        const float seepage = detail::smoothstep01(0.45f, 0.75f, fl);
                        const float streak = detail::smoothstep01(0.58f, 0.80f,
                                                                  detail::geoFbm(wx * 0.10f, wz * 0.10f)) * 0.75f;
                        wet = std::max(seepage, streak);
                        colVal[ck] = wet;
                        colLevel[ck] = r.level;
                    }
                    wet *= (1.f - ledge) * (1.f - snowW);

                    // Gneiss macro: foliation/weathering varying in ALL THREE
                    // axes — this is the whole point of the reparametrisation,
                    // a heightfield bake can only vary in x and z.
                    const float var = 0.80f + 0.42f * (0.5f * (detail::geoFbm(wx * 0.045f, wy * 0.05f) +
                                                               detail::geoFbm(wz * 0.045f + 61.f, wy * 0.04f)));
                    const float pale = detail::smoothstep01(0.56f, 0.78f,
                                                            detail::geoFbm((wx + wz) * 0.020f, wy * 0.085f));
                    std::array<float, 3> c{};
                    for (int i2 = 0; i2 < 3; ++i2) c[i2] = rockCol[i2] * var + pale * 0.10f;
                    // Moss/heath on the treads, forest floor where the pack's CHM
                    // says trees actually stand on that tread.
                    for (int i2 = 0; i2 < 3; ++i2) c[i2] += (mossCol[i2] - c[i2]) * ledge * 0.85f;
                    if (forest > 0.001f)
                        for (int i2 = 0; i2 < 3; ++i2)
                            c[i2] += (floorCol[i2] - c[i2]) * ledge * forest * 0.9f;
                    if (wet > 0.001f) {
                        const float kw = wet * 0.62f;
                        c[0] *= 1.f - kw; c[1] *= 1.f - kw * 0.94f; c[2] *= 1.f - kw * 0.86f;
                    }
                    for (int i2 = 0; i2 < 3; ++i2) c[i2] += (snowCol[i2] - c[i2]) * snowW;

                    float w4[4];
                    w4[0] = ledge * (1.f - snowW) * (0.55f + 0.35f * forest);// grass family on treads
                    w4[2] = ledge * (1.f - snowW) * (1.f - forest) * 0.30f;  // rubble on bare treads
                    w4[3] = snowW;
                    w4[1] = std::max(0.05f, 1.f - w4[0] - w4[2] - w4[3]);    // gneiss everywhere else
                    const float ws = 1.f / (w4[0] + w4[1] + w4[2] + w4[3]);
                    for (float& w : w4) w *= ws;

                    const int ax = r.ax + t;
                    for (int row = 0; row < kBandH; ++row) {// content + its two gutter copies
                        const std::size_t o = (static_cast<std::size_t>(r.ay + row) * AW + ax) * 4;
                        for (int i2 = 0; i2 < 3; ++i2)
                            macro[o + i2] = static_cast<std::uint8_t>(std::clamp(c[i2], 0.f, 1.f) * 255.f + 0.5f);
                        macro[o + 3] = 255;
                        for (int i2 = 0; i2 < 4; ++i2)
                            wts[o + i2] = static_cast<std::uint8_t>(std::clamp(w4[i2], 0.f, 1.f) * 255.f + 0.5f);
                    }
                    // Horizontal gutters at the run's ends.
                    if (t == 0 || t == r.at - 1) {
                        const int gx = t == 0 ? r.ax - 1 : r.ax + r.at;
                        if (gx >= 0 && gx < AW)
                            for (int row = 0; row < kBandH; ++row) {
                                const std::size_t src = (static_cast<std::size_t>(r.ay + row) * AW + ax) * 4;
                                const std::size_t dst = (static_cast<std::size_t>(r.ay + row) * AW + gx) * 4;
                                for (int i2 = 0; i2 < 4; ++i2) { macro[dst + i2] = macro[src + i2]; wts[dst + i2] = wts[src + i2]; }
                            }
                    }
                }
            }

            auto geo = BufferGeometry::create();
            geo->setIndex(idx);
            geo->setAttribute("position", FloatBufferAttribute::create(pos, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(nrm, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(uv, 2));
            geo->computeBoundingSphere();

            auto mat = MeshStandardMaterial::create();
            mat->roughness = 0.92f;
            mat->metalness = 0.f;
            {
                auto tex = DataTexture::create(ImageData{std::move(macro)},
                                               static_cast<unsigned int>(AW), static_cast<unsigned int>(AH));
                tex->colorSpace = ColorSpace::sRGB;
                tex->magFilter = Filter::Linear;
                tex->minFilter = Filter::LinearMipmapLinear;
                tex->generateMipmaps = true;
                tex->anisotropy = 8;
                mat->map = tex;
            }
            {
                auto tex = DataTexture::create(ImageData{std::move(wts)},
                                               static_cast<unsigned int>(AW), static_cast<unsigned int>(AH));
                tex->colorSpace = ColorSpace::Linear;// coverage data, not colour
                tex->magFilter = Filter::Linear;
                tex->minFilter = Filter::LinearMipmapLinear;
                tex->generateMipmaps = true;
                mat->terrainWeightMap = tex;
            }
            for (std::size_t i2 = 0; i2 < 4; ++i2) {
                mat->terrainBandAlbedo[i2] = bands.band[i2].albedo;
                mat->terrainBandNormalRough[i2] = bands.band[i2].normalRough;
            }
            mat->terrainBandRepeat = bands.repeat;
            mat->terrainBandRoughness = bands.roughness;
            mat->terrainBandStrength = bandStrength;
            mat->terrainBandNormalScale = bandNormalScale;
            mat->terrainBandRoughStrength = 0.6f;
            mat->terrainHeightBlend = heightBlend;
            // terrainNormalMap stays null on purpose: the shell has REAL vertex
            // normals carrying the displacement, so a baked world-normal map
            // would only band-limit them.

            auto mesh = Mesh::create(geo, mat);
            mesh->name = "cliff_shell_region";
            mesh->castShadow = true;
            mesh->receiveShadow = true;
            shell->add(mesh);

            st.vertices += pos.size() / 3;
            st.triangles += idx.size() / 3;
            st.atlasTexels += static_cast<std::size_t>(AW) * AH;
            ++st.regions;
        }
        parent.add(shell);
        st.bakeSeconds = std::chrono::duration<float>(clock::now() - tb0).count();
        return st;
    }

}// namespace threepp::terrain

#endif//THREEPP_EXTRAS_TERRAIN_CLIFFSHELL_HPP
