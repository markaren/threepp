// Procedural road ribbon generator (CPU).
//
// Pipeline:
//   1. ctor          — sample a centripetal CatmullRomCurve3 through the supplied
//                      control points into a dense centerline polyline (XZ
//                      positions + cumulative arc length), and build a uniform
//                      spatial hash so lateral "how far from the road?" queries
//                      only scan the local 3×3 cells.
//   2. conformTo()   — evaluate a ground-height function along the centerline,
//                      then LONGITUDINALLY smooth it (1-2-1 passes) so the road
//                      grade is gentle rather than tracing terrain noise. After
//                      this the per-sample frame (smoothed height + unit tangent
//                      + right vector) is complete; centerHeightAt/corridorWeight
//                      /buildSurface all depend on it.
//   3. buildSurface()/bakeSurfaceTexture() — bake the paved ribbon plus graded
//                      shoulder strips into one indexed BufferGeometry, and bake a
//                      matching sRGB albedo (asphalt + verge gravel + lane
//                      markings) for the surface.
//
// Header-only, dependency-free beyond threepp core, so it can be reused from
// examples and as the CPU reference for a later GPU port.
//
// Coordinate convention (matches TerrainGenerator): the geometry lies in the XZ
// plane, +Y up, metres. Control-point Y is ignored — elevations come from the
// ground function supplied to conformTo(). The road conforms to whatever terrain
// it is draped over; corridorWeight()/groundHeight() let a demo flatten the
// terrain into the road so the two meet seamlessly.

#ifndef THREEPP_EXTRAS_ROAD_ROADGENERATOR_HPP
#define THREEPP_EXTRAS_ROAD_ROADGENERATOR_HPP

#include "threepp/math/MathUtils.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/extras/curves/CatmullRomCurve3.hpp"
#include "threepp/math/Vector3.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace threepp::road {

    // Configurator-facing knobs. Plain data, copyable, with sane defaults that
    // produce a two-lane country road with gravel verges.
    struct RoadParams {
        float laneWidth = 3.5f;             // metres per lane
        int   laneCount = 2;                // paved width = laneWidth * laneCount
        float shoulderWidth = 2.5f;         // graded verge each side, beyond paved edge
        float surfaceRaise = 0.05f;         // ribbon sits this far above conformed ground (anti z-fight)
        float maxBanking = 0.10f;           // radians, max camber roll on tight curves
        float bankingCurvatureScale = 30.f; // larger => need tighter curve for same banking
        int   samplesPerSegment = 20;       // centerline tessellation between consecutive control points
        float markingWidth = 0.16f;         // painted line width (metres)
        bool  dashedCenter = true;          // dashed centre line vs solid
        float textureTileLength = 12.f;     // metres of road per vertical texture tile (UV v scale)
        std::array<float, 3> asphaltColor = {0.055f, 0.055f, 0.06f};
        std::array<float, 3> markingColor = {0.82f, 0.78f, 0.5f};  // warm off-white
        std::array<float, 3> shoulderColor = {0.32f, 0.30f, 0.24f};// gravel verge
    };

    // ── N302 marking classes ─────────────────────────────────────────────────
    // Statens vegvesen N302 / Rapport 452, in the terms this bake needs:
    //   • the centre marking is YELLOW and exists only where the asphalt is
    //     >= 6.0 m wide; below that the road is marked as a ONE-lane road;
    //   • the edge line (kantlinje) is WHITE — solid normally, dashed 3+3 m on
    //     the narrow roads that carry no centre line;
    //   • line width 0.10 m standard, 0.15 m on wide fast roads;
    //   • centre patterns: skillelinje 3+9 at 70-90 km/h, varsellinje 9+3,
    //     sperrelinje solid.
    enum class SurfaceKind {
        Asphalt,
        Gravel
    };
    enum class LinePattern {
        None,
        Solid,
        Dashed
    };

    // Everything the surface bake needs, in METRES — no texel counts leak into
    // the look, so the same style bakes correctly at any resolution.
    struct RoadSurfaceStyle {
        SurfaceKind kind = SurfaceKind::Asphalt;
        LinePattern centre = LinePattern::None;
        float centreOn = 3.f, centreOff = 9.f;// skillelinje 3+9
        LinePattern edge = LinePattern::Solid;
        float edgeOn = 3.f, edgeOff = 3.f;// stiplet kantlinje 3+3
        float lineWidth = 0.10f;          // N302 standard width
        float shoulderInset = 0.25f;      // paved edge -> outer side of the edge line
        std::array<float, 3> whiteColor = {0.86f, 0.86f, 0.84f};
        std::array<float, 3> yellowColor = {0.88f, 0.70f, 0.12f};
        float pavedWidth = 6.f;  // asphalt width (m)
        float fullWidth = 7.f;   // the texture spans this much road + sealed edge (m)
        float tileLength = 48.f; // metres of road per v tile (4 x 3+9, 8 x 3+3)
        float laneCount = 2.f;   // wheel-track placement (Phase B)
        float wear = 0.f;        // 0 fresh, 1 studded-tyre ruin
        unsigned int seed = 0u;  // variant selector
        bool gravelEdge = true;  // narrow grusskulder outside the pavement
    };

    // Three channel-packed maps, three.js convention (as FacadeTexture bakes):
    // albedo sRGB, tangent-space normal (+Z out), roughMetal G = roughness and
    // B = metalness.
    struct RoadSurfaceMaps {
        std::vector<unsigned char> albedo;
        std::vector<unsigned char> normal;
        std::vector<unsigned char> roughMetal;
        int width = 0, height = 0;
        std::array<float, 3> meanPaved = {0.f, 0.f, 0.f};// mean sRGB over the paved band
    };

    class RoadGenerator {

    public:
        // controlPoints: world-space path. Their Y is ignored — elevations come
        // from conformTo(). Needs at least two points to form a curve; with fewer
        // the centerline degenerates to the points themselves.
        explicit RoadGenerator(std::vector<Vector3> controlPoints, RoadParams params = {})
            : params_(params), controlPoints_(std::move(controlPoints)) {
            buildCenterline();
            buildSpatialHash();
        }

        // Sample the supplied ground-height function along the centerline, store a
        // longitudinally-SMOOTHED elevation per centerline sample (gentle grades,
        // not terrain noise), and complete the per-sample ribbon frame. MUST be
        // called once before buildSurface()/centerHeightAt()/corridorWeight().
        void conformTo(const std::function<float(float, float)>& groundHeight, int smoothingPasses = 12) {
            const size_t n = samples_.size();
            if (n == 0) return;

            // Raw ground sample at each centerline point.
            for (size_t i = 0; i < n; ++i) {
                samples_[i].height = groundHeight(samples_[i].pos.x, samples_[i].pos.z);
            }

            // Longitudinal 1-2-1 smoothing so the road grade is gentle. Endpoints
            // are pinned to their sampled ground height (the road must still meet
            // the terrain where it starts/ends); interior samples relax toward the
            // running average of their neighbours.
            std::vector<float> tmp(n);
            for (int pass = 0; pass < std::max(smoothingPasses, 0); ++pass) {
                tmp[0] = samples_[0].height;
                tmp[n - 1] = samples_[n - 1].height;
                for (size_t i = 1; i + 1 < n; ++i) {
                    tmp[i] = 0.25f * samples_[i - 1].height +
                             0.5f * samples_[i].height +
                             0.25f * samples_[i + 1].height;
                }
                for (size_t i = 0; i < n; ++i) samples_[i].height = tmp[i];
            }

            conformed_ = true;
        }

        // Conform to EXPLICIT per-centerline-sample elevations (same 1-2-1 grade
        // smoothing as conformTo). Caller supplies exactly one target height per
        // centerlineSamples() entry — RoadNetwork's elevation-profile mode uses
        // this so bridge decks / ground spans get their classified heights
        // directly instead of a (x,z)-pure ground function having to re-derive
        // which regime each sample is in.
        void conformToHeights(const std::vector<float>& heights, int smoothingPasses = 12) {
            const size_t n = samples_.size();
            if (n == 0 || heights.size() != n) return;
            for (size_t i = 0; i < n; ++i) samples_[i].height = heights[i];

            std::vector<float> tmp(n);
            for (int pass = 0; pass < std::max(smoothingPasses, 0); ++pass) {
                tmp[0] = samples_[0].height;
                tmp[n - 1] = samples_[n - 1].height;
                for (size_t i = 1; i + 1 < n; ++i) {
                    tmp[i] = 0.25f * samples_[i - 1].height +
                             0.5f * samples_[i].height +
                             0.25f * samples_[i + 1].height;
                }
                for (size_t i = 0; i < n; ++i) samples_[i].height = tmp[i];
            }

            conformed_ = true;
        }

        // ── geometry of the corridor ─────────────────────────────────────────
        [[nodiscard]] float pavedHalfWidth() const {
            return params_.laneWidth * static_cast<float>(std::max(params_.laneCount, 1)) * 0.5f;
        }
        [[nodiscard]] float corridorHalfWidth() const {
            return pavedHalfWidth() + std::max(params_.shoulderWidth, 0.f);
        }

        // Lateral distance (m) from world (x,z) to the nearest point on the
        // centerline polyline (point-to-SEGMENT, so the band is smooth).
        [[nodiscard]] float distanceToCenter(float x, float z) const {
            int nearest = nearestSampleIndex(x, z);
            if (nearest < 0) return std::numeric_limits<float>::max();

            // Test the segments incident to the nearest candidate sample (and its
            // hash neighbours, gathered by nearestSampleIndex) for the true
            // point-to-segment distance, not just point-to-point.
            float best = std::numeric_limits<float>::max();
            for (int idx : scratchCandidates_) {
                // segment (idx-1, idx) and (idx, idx+1)
                if (idx > 0) best = std::min(best, distPointToSegmentSq(x, z, idx - 1, idx));
                if (idx + 1 < static_cast<int>(samples_.size()))
                    best = std::min(best, distPointToSegmentSq(x, z, idx, idx + 1));
            }
            if (best == std::numeric_limits<float>::max()) {
                // Degenerate: single-sample centerline — fall back to point dist.
                const auto& p = samples_[nearest].pos;
                const float dx = x - p.x, dz = z - p.z;
                best = dx * dx + dz * dz;
            }
            return std::sqrt(best);
        }

        // 1.0 on the paved road, smoothstep down to 0.0 across the shoulder band,
        // 0 beyond. Argument order of smoothstep gives 1 near the road, 0 at the
        // corridor edge (so it can weight a terrain→road height blend).
        [[nodiscard]] float corridorWeight(float x, float z) const {
            const float d = distanceToCenter(x, z);
            const float paved = pavedHalfWidth();
            const float corridor = corridorHalfWidth();
            if (d <= paved) return 1.f;
            if (d >= corridor) return 0.f;
            return math::smoothstep(corridor, paved, d);
        }

        // Smoothed centerline elevation of the nearest centerline sample to (x,z).
        [[nodiscard]] float centerHeightAt(float x, float z) const {
            const int nearest = nearestSampleIndex(x, z);
            if (nearest < 0) return 0.f;
            return samples_[nearest].height;
        }

        // Convenience blend the demo uses for its unified ground height:
        //   mix(terrainHeight, centerHeightAt(x,z), corridorWeight(x,z))
        [[nodiscard]] float groundHeight(float terrainHeight, float x, float z) const {
            const float w = corridorWeight(x, z);
            return lerp(terrainHeight, centerHeightAt(x, z), w);
        }

        // Build the paved ribbon plus graded shoulder strips as one indexed
        // BufferGeometry. Each centerline sample emits a 4-vertex cross-section
        //   [-corridorHalf, -pavedHalf, +pavedHalf, +corridorHalf]
        // giving three quads per segment (left shoulder, paved, right shoulder).
        // Paved vertices sit at height+surfaceRaise; shoulder-outer vertices sit
        // flush at the conformed ground height. Curves bank INTO the turn (outer
        // edge higher) by camber roll about the tangent.
        // UV: u = 0..1 across the FULL geometry width; v = arcLength/textureTileLength.
        //
        // vOffsetMeters shifts the v coordinate by that many metres of road, so a
        // ribbon PIECE sliced out of a longer centreline can carry the piece's
        // GLOBAL station: with a shared, tiling marking texture the dash phase is
        // a function of v alone, and without the offset every chunk would restart
        // its dashes at its own seam. Default 0 = the whole-road case, unchanged.
        //
        // mirrorAlternateTiles flips u on ODD texture tiles (even tiles sample u,
        // odd tiles 1-u), which doubles the shared tile's along-road repeat again
        // without a second bake: the aggregate grain, the wheel-track amplitude
        // and the crack field all come back MIRRORED rather than identical.
        // SAFE ONLY BECAUSE EVERY BAKED CLASS LAYOUT IS LEFT-RIGHT SYMMETRIC —
        // edge lines, centre line, gravel strips and wheel tracks all sit at
        // ±the same offsets, so the mirrored tile carries the markings in the
        // same places. A future asymmetric layout (a climbing lane, a one-sided
        // kerb, right-hand-drive-only paint) would swap sides at every odd tile
        // and this flag would have to be gated off for that class.
        // The flip lands EXACTLY on the tile boundary (a duplicated
        // cross-section there), so no triangle ever interpolates u across it.
        // Default false = byte-identical to the pre-mirror geometry.
        [[nodiscard]] std::shared_ptr<BufferGeometry> buildSurface(float vOffsetMeters = 0.f,
                                                                    bool mirrorAlternateTiles = false) const {
            auto geo = std::make_shared<BufferGeometry>();
            const size_t n = samples_.size();
            if (n < 2) return geo;

            const float pavedHalf = pavedHalfWidth();
            const float corridorHalf = corridorHalfWidth();
            const float raise = params_.surfaceRaise;
            const float tile = std::max(params_.textureTileLength, 1e-3f);

            // Lateral offsets of the four cross-section vertices, and which are
            // paved (raised). u runs 0..1 across the full width.
            const std::array<float, 4> off = {-corridorHalf, -pavedHalf, pavedHalf, corridorHalf};
            const std::array<bool, 4> paved = {false, true, true, false};
            const float fullWidth = 2.f * corridorHalf;
            std::array<float, 4> uCoord{};
            for (int k = 0; k < 4; ++k)
                uCoord[k] = (fullWidth > 1e-6f) ? (off[k] + corridorHalf) / fullWidth : 0.f;

            // ── cross-section node list ──────────────────────────────────────
            // One node per centerline sample in the default case. With mirroring
            // on, a DUPLICATE PAIR is inserted at every tile boundary the ribbon
            // crosses: the first node closes the previous tile with its own u
            // convention, the second opens the next tile with the flipped one,
            // and no quad spans the two (join == false), so the flip is a hard
            // edge exactly at v = integer rather than a smear across a segment.
            struct Node {
                float x = 0.f, z = 0.f, h = 0.f;// centre position + conformed height
                float rx = 1.f, rz = 0.f;       // unit right (XZ)
                float cb = 1.f, sb = 0.f;       // cos/sin of the camber roll
                float v = 0.f;                  // texture v (tiles of road)
                bool flip = false;              // sample 1-u instead of u
                bool join = true;               // quad back to the previous node
            };
            std::vector<Node> nodes;
            nodes.reserve(n + 8);
            // t == 0 takes the sample verbatim (no lerp, no renormalise) so the
            // default path is bit-for-bit what it was before mirroring existed.
            const auto emit = [&](size_t i, float t, bool flip, bool join) {
                Node nd;
                float bank;
                if (t <= 0.f) {
                    const Sample& a = samples_[i];
                    nd.x = a.pos.x;
                    nd.z = a.pos.z;
                    nd.h = a.height;
                    nd.rx = a.right.x;
                    nd.rz = a.right.z;
                    bank = bankingAngle(i);
                    nd.v = (a.arcLength + vOffsetMeters) / tile;
                } else {
                    const size_t j = std::min(i + 1, n - 1);
                    const Sample& a = samples_[i];
                    const Sample& b = samples_[j];
                    nd.x = lerp(a.pos.x, b.pos.x, t);
                    nd.z = lerp(a.pos.z, b.pos.z, t);
                    nd.h = lerp(a.height, b.height, t);
                    nd.rx = lerp(a.right.x, b.right.x, t);
                    nd.rz = lerp(a.right.z, b.right.z, t);
                    const float rl = std::sqrt(nd.rx * nd.rx + nd.rz * nd.rz);
                    if (rl > 1e-6f) {
                        nd.rx /= rl;
                        nd.rz /= rl;
                    }
                    bank = lerp(bankingAngle(i), bankingAngle(j), t);
                    nd.v = (lerp(a.arcLength, b.arcLength, t) + vOffsetMeters) / tile;
                }
                nd.cb = std::cos(bank);
                nd.sb = std::sin(bank);
                nd.flip = flip;
                nd.join = join;
                nodes.push_back(nd);
            };
            const auto tileOf = [&](float arc) {
                return static_cast<int>(std::floor((arc + vOffsetMeters) / tile));
            };
            for (size_t i = 0; i < n; ++i) {
                const int ti = tileOf(samples_[i].arcLength);
                const bool flip = mirrorAlternateTiles && ((ti & 1) != 0);
                emit(i, 0.f, flip, i != 0);
                if (!mirrorAlternateTiles || i + 1 >= n) continue;
                // Cross every tile boundary between this sample and the next
                // (sample spacing is sub-metre against a ~96 m tile, so this is
                // normally zero or one crossing — the loop is for safety).
                const int tj = tileOf(samples_[i + 1].arcLength);
                const float a0 = samples_[i].arcLength, a1 = samples_[i + 1].arcLength;
                const float span = a1 - a0;
                for (int tb = ti + 1; tb <= tj; ++tb) {
                    if (span <= 1e-6f) break;
                    const float sB = static_cast<float>(tb) * tile - vOffsetMeters;
                    const float t = std::clamp((sB - a0) / span, 0.f, 1.f);
                    if (t <= 1e-6f || t >= 1.f - 1e-6f) continue;
                    emit(i, t, flip, true);                        // closes tile tb-1
                    emit(i, t, ((tb & 1) != 0), false);            // opens tile tb
                }
            }
            const size_t nNodes = nodes.size();

            std::vector<float> positions;
            std::vector<float> normals;
            std::vector<float> uvs;
            positions.reserve(nNodes * 4 * 3);
            normals.reserve(nNodes * 4 * 3);
            uvs.reserve(nNodes * 4 * 2);

            for (const Node& nd : nodes) {
                for (int k = 0; k < 4; ++k) {
                    const float lat = off[k];
                    // Lateral displacement: right rotated about tangent by `bank`.
                    // right' = right*cos(bank) + up*sin(bank)  (a roll in the cross
                    // plane). The vertex sits at center + lat * right'.
                    const float px = nd.x + lat * (nd.rx * nd.cb);
                    const float pz = nd.z + lat * (nd.rz * nd.cb);
                    // Camber raises the outer edge: height contribution = lat*sin(bank)
                    // (lat sign distinguishes left/right of centerline).
                    float py = nd.h + lat * nd.sb;
                    if (paved[k]) py += raise;

                    positions.push_back(px);
                    positions.push_back(py);
                    positions.push_back(pz);

                    // Provisional normal (recomputed below); seed with up.
                    normals.push_back(0.f);
                    normals.push_back(1.f);
                    normals.push_back(0.f);

                    uvs.push_back(nd.flip ? 1.f - uCoord[k] : uCoord[k]);
                    uvs.push_back(nd.v);
                }
            }

            // Index: 3 quads per segment, 2 triangles per quad. Cross-section of
            // node i occupies vertices [i*4 .. i*4+3]; a node with join == false
            // starts a new strip (the far half of a mirror-boundary pair).
            std::vector<unsigned int> indices;
            indices.reserve((nNodes - 1) * 3 * 6);
            for (size_t i = 0; i + 1 < nNodes; ++i) {
                if (!nodes[i + 1].join) continue;
                const unsigned int a = static_cast<unsigned int>(i * 4);
                const unsigned int b = static_cast<unsigned int>((i + 1) * 4);
                for (int q = 0; q < 3; ++q) {
                    const unsigned int v0 = a + q;       // this section, lateral k=q
                    const unsigned int v1 = a + q + 1;   // this section, lateral k=q+1
                    const unsigned int v2 = b + q;       // next section, lateral k=q
                    const unsigned int v3 = b + q + 1;   // next section, lateral k=q+1
                    // Two triangles wound so the surface normal points UP (+Y).
                    // The cross-section `right` vector is tangent×up (the left side
                    // for a +Z heading), so this order — not its reverse — yields the
                    // upward face; the reverse leaves the road back-face-culled
                    // (invisible from above) on rasterisers and lit-from-below on PT.
                    indices.push_back(v0);
                    indices.push_back(v1);
                    indices.push_back(v2);
                    indices.push_back(v1);
                    indices.push_back(v3);
                    indices.push_back(v2);
                }
            }

            geo->setIndex(indices);
            geo->setAttribute("position", FloatBufferAttribute::create(positions, 3));
            geo->setAttribute("normal", FloatBufferAttribute::create(normals, 3));
            geo->setAttribute("uv", FloatBufferAttribute::create(uvs, 2));
            geo->computeVertexNormals();// recompute smooth normals from the final mesh
            geo->computeBoundingBox();
            geo->computeBoundingSphere();
            return geo;
        }

        // Total XZ arc length of the centerline (m).
        [[nodiscard]] float totalLength() const {
            return samples_.empty() ? 0.f : samples_.back().arcLength;
        }

        // A point ON the paved surface, at arc-length `station` (m from the start)
        // and lateral offset `lat` (m, + toward `right`). This is the SAME
        // cross-section formula buildSurface bakes — the banking roll about the
        // tangent, plus surfaceRaise — so anything placed with it (a repair-patch
        // decal, a manhole, a stop line) lies exactly on the ribbon rather than
        // on a flat plane the ribbon curves away from. `extraRaise` lifts it
        // clear of the ribbon by that many metres.
        [[nodiscard]] Vector3 surfacePointAt(float station, float lat, float extraRaise = 0.f) const {
            const size_t n = samples_.size();
            if (n == 0) return Vector3::ZEROS();
            if (n == 1) return Vector3(samples_[0].pos.x, samples_[0].height + params_.surfaceRaise + extraRaise,
                                       samples_[0].pos.z);
            station = std::clamp(station, 0.f, samples_[n - 1].arcLength);
            // Binary search the arc-length table for the bracketing samples.
            size_t lo = 0, hi = n - 1;
            while (lo + 1 < hi) {
                const size_t mid = (lo + hi) / 2;
                if (samples_[mid].arcLength <= station) lo = mid; else hi = mid;
            }
            const Sample& a = samples_[lo];
            const Sample& b = samples_[hi];
            const float span = std::max(b.arcLength - a.arcLength, 1e-6f);
            const float t = std::clamp((station - a.arcLength) / span, 0.f, 1.f);
            const float bank = lerp(bankingAngle(lo), bankingAngle(hi), t);
            const float cb = std::cos(bank), sb = std::sin(bank);
            float rx = lerp(a.right.x, b.right.x, t);
            float rz = lerp(a.right.z, b.right.z, t);
            const float rl = std::sqrt(rx * rx + rz * rz);
            if (rl > 1e-6f) { rx /= rl; rz /= rl; }
            const float cx = lerp(a.pos.x, b.pos.x, t);
            const float cz = lerp(a.pos.z, b.pos.z, t);
            const float ch = lerp(a.height, b.height, t);
            return Vector3(cx + lat * (rx * cb),
                           ch + lat * sb + params_.surfaceRaise + extraRaise,
                           cz + lat * (rz * cb));
        }

        // Bake an sRGB RGBA8 albedo (row-major, `width` across road = U, `height`
        // along road = V). Layers: gravel shoulders at the U edges, asphalt in the
        // middle, solid edge lines just inside the paved edges, and a centre line
        // (dashed if params.dashedCenter) down U=0.5. Subtle multi-scale noise on
        // the asphalt keeps it from reading as flat plastic. Returns the byte buffer.
        [[nodiscard]] std::vector<unsigned char> bakeSurfaceTexture(int width, int height) const {
            const int W = std::max(width, 1);
            const int H = std::max(height, 1);
            std::vector<unsigned char> out(static_cast<size_t>(W) * H * 4, 255u);

            const float corridorHalf = corridorHalfWidth();
            const float pavedHalf = pavedHalfWidth();
            const float fullWidth = 2.f * corridorHalf;
            // U fractions where the paved band starts/ends (gravel outside these).
            const float pavedFracLo = (fullWidth > 1e-6f) ? (corridorHalf - pavedHalf) / fullWidth : 0.f;
            const float pavedFracHi = 1.f - pavedFracLo;
            // Marking widths expressed as U fractions.
            const float markFrac = (fullWidth > 1e-6f) ? params_.markingWidth / fullWidth : 0.f;
            // Centres of the two solid edge lines (just inside the paved edges).
            const float edgeLineLo = pavedFracLo + markFrac;
            const float edgeLineHi = pavedFracHi - markFrac;
            // Dash pattern along V (in texel rows): ~3 m on, 6 m of gap per tile.
            const float dashPeriod = std::max(static_cast<float>(H) * 0.18f, 2.f);

            const auto setPx = [&](int x, int y, float r, float g, float b) {
                const size_t o = (static_cast<size_t>(y) * W + x) * 4;
                out[o + 0] = toByte(r);
                out[o + 1] = toByte(g);
                out[o + 2] = toByte(b);
                out[o + 3] = 255u;
            };

            for (int y = 0; y < H; ++y) {
                const float v = static_cast<float>(y);
                for (int x = 0; x < W; ++x) {
                    const float u = (W > 1) ? static_cast<float>(x) / static_cast<float>(W - 1) : 0.5f;

                    float r, g, b;
                    if (u < pavedFracLo || u > pavedFracHi) {
                        // Gravel shoulder — noisy verge colour.
                        const float n = fbm(static_cast<float>(x) * 0.18f, v * 0.18f, 4, 2.f, 0.5f);
                        const float k = std::clamp(1.f + 0.22f * n, 0.6f, 1.35f);
                        r = params_.shoulderColor[0] * k;
                        g = params_.shoulderColor[1] * k;
                        b = params_.shoulderColor[2] * k;
                    } else {
                        // Asphalt base + multi-scale grain (de-plastic idiom).
                        const float n1 = fbm(static_cast<float>(x) * 0.22f, v * 0.22f, 4, 2.f, 0.5f);
                        const float n2 = noise2(static_cast<float>(x) * 0.9f, v * 0.9f);
                        const float k = std::clamp(1.f + 0.18f * n1 + 0.10f * n2, 0.7f, 1.3f);
                        r = params_.asphaltColor[0] * k;
                        g = params_.asphaltColor[1] * k;
                        b = params_.asphaltColor[2] * k;

                        // Solid edge lines.
                        const bool onEdgeLo = std::abs(u - edgeLineLo) <= markFrac * 0.5f;
                        const bool onEdgeHi = std::abs(u - edgeLineHi) <= markFrac * 0.5f;
                        // Centre line at U=0.5 (dashed if requested).
                        bool onCenter = std::abs(u - 0.5f) <= markFrac * 0.5f;
                        if (onCenter && params_.dashedCenter) {
                            const float phase = std::fmod(v, dashPeriod) / dashPeriod;
                            onCenter = phase < 0.4f;// ~40% duty cycle => dash
                        }
                        if (onEdgeLo || onEdgeHi || onCenter) {
                            // Slightly noisy paint so it isn't a flat stripe.
                            const float pn = noise2(static_cast<float>(x) * 0.6f, v * 0.6f);
                            const float pk = std::clamp(1.f + 0.06f * pn, 0.85f, 1.1f);
                            r = params_.markingColor[0] * pk;
                            g = params_.markingColor[1] * pk;
                            b = params_.markingColor[2] * pk;
                        }
                    }
                    setPx(x, y, r, g, b);
                }
            }
            return out;
        }

        // Bake the THREE maps of a marking class (see RoadSurfaceStyle): sRGB
        // albedo, tangent-space normal, and roughness/metalness. This is the road
        // look; bakeSurfaceTexture above is the legacy single-albedo path and is
        // left alone on purpose (buildMeshes and the Drive demo still call it).
        //
        // Two things it does that the legacy bake does not:
        //   • lines are anti-aliased by COVERAGE — a texel that a 0.10 m line
        //     half-covers gets half the paint. Hard per-texel thresholding is why
        //     the old markings staircase at the grazing angles a road is always
        //     seen at, no matter how much anisotropy the sampler has.
        //   • every field varying along the road uses a lattice that WRAPS at the
        //     tile, so the shared, repeating tile has no seam across the road.
        //
        // Everything is expressed in metres: `width` texels span style.fullWidth
        // and `height` texels span style.tileLength.
        [[nodiscard]] static RoadSurfaceMaps bakeSurfaceMaps(const RoadSurfaceStyle& st,
                                                             int width, int height) {
            RoadSurfaceMaps m;
            const int W = std::max(width, 4);
            const int H = std::max(height, 4);
            m.width = W;
            m.height = H;
            const size_t nPx = static_cast<size_t>(W) * static_cast<size_t>(H);
            m.albedo.assign(nPx * 4, 255u);
            m.normal.assign(nPx * 4, 255u);
            m.roughMetal.assign(nPx * 4, 255u);

            const float full = std::max(st.fullWidth, 0.5f);
            const float pavedHalf = std::max(st.pavedWidth, 0.5f) * 0.5f;
            const float tile = std::max(st.tileLength, 1.f);
            const float du = full / static_cast<float>(W); // metres per texel across
            const float dv = tile / static_cast<float>(H); // metres per texel along
            const float halfLine = std::max(st.lineWidth, 0.02f) * 0.5f;
            // Edge line: its OUTER side sits `shoulderInset` in from the asphalt
            // edge (N302 keeps >= 0.25 m of asphalt outside the kantlinje).
            const float edgeCentre = std::max(pavedHalf - st.shoulderInset - halfLine, 0.05f);
            const float w = std::clamp(st.wear, 0.f, 1.f);
            const float seedX = static_cast<float>(st.seed % 977u) * 13.37f;
            // Every along-road field wraps with the TILE, so the wrap period of a
            // lattice whose cell is `cellM` metres must be derived from
            // st.tileLength — hard-coded periods silently pin the longest visible
            // feature to whatever tile length they were written for (they were
            // written for 48 m; the tile is 96 m now, and a hard-coded 48 m wrap
            // would have handed back the very 48 m repeat this bake is trying to
            // get rid of).
            const auto periodFor = [tile](float cellM) {
                return std::max(1, static_cast<int>(std::lround(tile / std::max(cellM, 1e-3f))));
            };

            // Coverage of a band of half-width `hw` centred on 0, sampled at
            // signed distance d with texel footprint `texel`.
            const auto cover = [](float d, float hw, float texel) {
                return std::clamp((hw - std::abs(d)) / std::max(texel, 1e-6f) + 0.5f, 0.f, 1.f);
            };
            // Coverage of the ON part of an on/off dash pattern at station s.
            const auto dashCover = [&cover](float s, float on, float off, float texel) {
                const float period = std::max(on + off, 1e-3f);
                float ph = std::fmod(s, period);
                if (ph < 0.f) ph += period;
                return cover(ph - on * 0.5f, on * 0.5f, texel);
            };

            std::vector<float> hgt(nPx, 0.f);// relief for the normal map (metres)
            double sumR = 0.0, sumG = 0.0, sumB = 0.0;
            size_t pavedCount = 0;

            const bool gravel = (st.kind == SurfaceKind::Gravel);
            // Wheel tracks sit where the traffic is: a pair per lane on a road
            // wide enough for two, a single pair down the middle where there is
            // only room for one vehicle. Studded tyres polish them lighter and
            // smoother than the aggregate beside them — on an overcast day the
            // tracks are the part of the road that catches the sky.
            const bool twoLane = (st.centre != LinePattern::None);
            std::array<float, 4> trackC{};
            int nTracks = 0;
            if (twoLane) {
                const float laneC = pavedHalf * 0.5f;
                trackC[nTracks++] = laneC - 0.75f;
                trackC[nTracks++] = laneC + 0.75f;
                trackC[nTracks++] = -laneC + 0.75f;
                trackC[nTracks++] = -laneC - 0.75f;
            } else {
                trackC[nTracks++] = 0.75f;
                trackC[nTracks++] = -0.75f;
            }
            // 1 in the core of a band, smoothly out over `soft` metres.
            const auto softBand = [](float d, float halfW, float soft) {
                const float a = std::abs(d);
                if (a <= halfW) return 1.f;
                if (a >= halfW + soft) return 0.f;
                return 1.f - math::smoothstep(halfW, halfW + soft, a);
            };
            const int sd = static_cast<int>(st.seed & 0x7FFFFFFFu);

            // NOTHING DISCRETE IS BAKED INTO THIS TILE ANY MORE. The tile is
            // SHARED by every chunk of every road of its class, so a recognisable
            // object in it — a repair patch, a pothole plug, a manhole — comes
            // back at the identical lateral position every tileLength metres
            // along every road in the scene, which is exactly what "the patches
            // are just repeated the same way at an interval" describes. Repair
            // patches now live per CHUNK as decal quads (RoadNetwork's
            // buildPatchDecals) where their positions are a function of the road
            // id and the piece index and therefore never line up.
            // What is still allowed here: fields with no recognisable SHAPE —
            // grain, wheel-track polish, paint wear, cracks.
            //
            // The one thing that survives is the old EDGE REPAIR — a 0.6 m strip
            // of a different mix along one side — because it has no shape of its
            // own, only an extent; it is gated by a run fbm so it fades in and
            // out over tens of metres instead of starting and stopping with the
            // tile.
            const float repairSide = (hash2(sd, 42) > 0.f) ? 1.f : -1.f;

            // ── transverse cracks ────────────────────────────────────────────
            // One CANDIDATE per 8 m cell, but a third of the cells carry no
            // crack and every survivor is jittered by up to half a cell, so
            // consecutive cracks land anywhere from ~5 to ~25 m apart and the
            // eye never finds the beat. What was there before — one crack per
            // cell, jittered ±2.4 m, drawn as a single station for the whole row
            // — is precisely a set of evenly spaced straight lines spanning the
            // carriageway, i.e. concrete slab joints, which is what the road
            // read as.
            //
            // Everything that makes a crack look like a crack is a function of
            // LAT (how far across the road you are), evaluated per texel:
            // the station wanders, the width breathes, pieces are missing, and
            // most cracks do not reach both edges.
            struct XCrack {
                float at = 0.f;      // nominal station of the crack (m)
                float halfW = 0.f;   // base half width (m)
                float sealed = 0.f;  // 1 = tar-sealed, 0 = open
                float sx = 0.f;      // per-crack noise offset
                float latA = 0.f, latB = 0.f;// piece 1 lateral extent (m)
                float latC = 0.f, latD = 0.f;// piece 2 (empty when latC >= latD)
                float jog = 0.f;     // station offset of piece 2 (m)
            };
            const float crackCell = 8.f;
            const int crackCells = periodFor(crackCell);
            const auto crackAt = [&](int c, XCrack& out) {
                const int cw = ((c % crackCells) + crackCells) % crackCells;
                const int cs = sd + cw * 37;
                if (hash2(cs, 5) < -0.30f) return false;// ~35% of cells: no crack
                const auto u01 = [cs](int k) { return hash2(cs, k) * 0.5f + 0.5f; };
                out.at = (static_cast<float>(c) + 0.5f + hash2(cs, 6) * 0.5f) * crackCell;
                const bool sealed = hash2(cs, 9) > -0.2f;
                out.sealed = sealed ? 1.f : 0.f;
                // A texel is dv (~4.7 cm) long, so a literal 2 cm crack can never
                // cover half of one and mips erase it by the second car length.
                // Real tar sealing is a smeared 5-7 cm band anyway, and this is
                // the signature of a Norwegian road — it has to survive to eye
                // level. An UNSEALED crack is a hairline and stays thin.
                out.halfW = sealed ? 0.035f : 0.012f;
                out.sx = static_cast<float>(cw) * 7.31f + seedX * 0.017f;
                out.latA = -pavedHalf - 0.2f;
                out.latB = pavedHalf + 0.2f;
                out.latC = 1.f;
                out.latD = 0.f;// piece 2 empty by default
                out.jog = 0.f;
                const float span = 2.f * pavedHalf;
                if (hash2(cs, 11) <= 0.20f) {// ~40% are PARTIAL
                    if (hash2(cs, 12) > 0.5f) {
                        // Two offset pieces with a jog between them: the crack
                        // steps sideways along the road where it crossed a joint.
                        const float f1 = 0.25f + 0.25f * u01(13);
                        const float f2 = std::min(f1 + 0.05f + 0.15f * u01(14), 0.95f);
                        out.latB = -pavedHalf + f1 * span;
                        out.latC = -pavedHalf + f2 * span;
                        out.latD = pavedHalf + 0.2f;
                        out.jog = (hash2(cs, 15) > 0.f ? 1.f : -1.f) * (0.10f + 0.25f * u01(16));
                    } else {
                        const float f = 0.20f + 0.40f * u01(13);
                        if (hash2(cs, 14) > 0.f) out.latA = -pavedHalf + f * span;// starts late
                        else out.latB = -pavedHalf + f * span;                    // stops early
                    }
                }
                return true;
            };

            for (int y = 0; y < H; ++y) {
                // Station along the tile, in metres, and in lattice cells for the
                // wrapping noise (tile / cellSize must be an integer — see below).
                const float s = (static_cast<float>(y) + 0.5f) * dv;

                // ── fields that depend only on the station ──────────────────
                // A real pavement edge is chipped and frayed, not sawn.
                const float ragL = 0.13f * w * fbmP(seedX + 3.f, s / 3.f, periodFor(3.f), 3, 2.f, 0.5f);
                const float ragR = 0.13f * w * fbmP(seedX + 91.f, s / 3.f, periodFor(3.f), 3, 2.f, 0.5f);
                // Paint is renewed in stretches, so how faded a line is depends
                // on WHERE along the road you are, in ~12 m segments.
                const float segWear =
                        std::clamp(0.55f + 0.60f * fbmP(seedX + 17.f, s / 12.f, periodFor(12.f), 3, 2.f, 0.5f), 0.f, 1.f) * w;
                // A ghost of an older line, offset a few centimetres, on some
                // segments: the road was re-marked and the old one still shows.
                const float ghostSeg = (fbmP(seedX + 57.f, s / 12.f, periodFor(12.f), 2, 2.f, 0.5f) > 0.28f) ? 1.f : 0.f;
                const float ghostOff = 0.03f + 0.03f * (hash2(sd, 61) * 0.5f + 0.5f);
                // The edge repair runs in and out over tens of metres.
                const float repairRun = !gravel && w > 0.05f
                        ? std::clamp(fbmP(seedX + 167.f, s / 24.f, periodFor(24.f), 3, 2.f, 0.5f) * 3.f - 0.55f, 0.f, 1.f)
                        : 0.f;
                // Which transverse cracks can possibly reach THIS row: a crack of
                // cell c sits inside cell c after jitter, and wanders at most
                // ~0.3 m in station, so only the three cells around s matter and
                // in practice none of them do. Rows with no crack nearby pay one
                // hash per cell and nothing in the x loop.
                std::array<XCrack, 3> xc{};
                int nXc = 0;
                if (!gravel && w > 0.05f) {
                    const int c0 = static_cast<int>(std::floor(s / crackCell));
                    for (int c = c0 - 1; c <= c0 + 1; ++c) {
                        XCrack cr;
                        if (!crackAt(c, cr)) continue;
                        // 0.29 m of wander + 0.35 m of jog + the widest band.
                        if (std::abs(s - cr.at) > 0.75f + cr.halfW * 1.5f) continue;
                        xc[nXc++] = cr;
                        if (nXc == 3) break;
                    }
                }

                // How hard a wheel track is polished varies ALONG the road: a
                // lane's two tracks are driven differently and each fades in
                // and out over 10-40 m. Without this the four tracks of a
                // two-lane road read from the air as four painted stripes.
                std::array<float, 4> trackAmp{};
                for (int k = 0; k < nTracks; ++k) {
                    const float kf = static_cast<float>(k);
                    const float lo = fbmP(seedX + 211.f + 37.f * kf, s / 24.f, periodFor(24.f), 2, 2.f, 0.5f);
                    const float hi = fbmP(seedX + 409.f + 53.f * kf, s / 12.f, periodFor(12.f), 4, 2.f, 0.5f);
                    trackAmp[k] = std::clamp(0.55f + 0.85f * lo + 0.35f * hi, 0.05f, 1.f);
                }

                for (int x = 0; x < W; ++x) {
                    // Lateral offset from the centreline, matching buildSurface's
                    // u = 0 at -corridorHalf .. u = 1 at +corridorHalf.
                    const float lat = ((static_cast<float>(x) + 0.5f) / static_cast<float>(W) - 0.5f) * full;
                    // Where the asphalt actually ends on THIS row.
                    const float edgeHere = pavedHalf + ((lat < 0.f) ? ragL : ragR);
                    const bool onPaved = std::abs(lat) <= edgeHere;

                    float r, g, b, rough;
                    float relief = 0.f;

                    // ── track bands (shared by both surface kinds) ───────────
                    // `trackW` is the band SHAPE (polish and ruts are always
                    // there); `trackLight` is the same band with the along-road
                    // amplitude folded in, and only the LIGHTENING uses it.
                    float trackW = 0.f, trackLight = 0.f;
                    for (int k = 0; k < nTracks; ++k) {
                        const float bandK = softBand(lat - trackC[k], 0.18f, 0.27f);
                        trackW = std::max(trackW, bandK);
                        trackLight = std::max(trackLight, bandK * trackAmp[k]);
                    }
                    // The strip between a lane's two tracks stays dirtier.
                    float betweenW = 0.f;
                    if (twoLane) {
                        const float laneC = pavedHalf * 0.5f;
                        betweenW = std::max(softBand(lat - laneC, 0.35f, 0.2f),
                                            softBand(lat + laneC, 0.35f, 0.2f));
                    } else {
                        betweenW = softBand(lat, 0.35f, 0.2f);
                    }
                    betweenW = std::max(betweenW - trackW, 0.f);

                    if (gravel || !onPaved) {
                        // Loose gravel: warm grey stones, no bitumen. Outside a
                        // frayed asphalt edge this is the narrow grusskulder the
                        // terrain's grass then takes over from - it must stay a
                        // FRAYED EDGE, never a dirt band beside the road.
                        const float stone = noise2p(lat / 0.055f + seedX, s / 0.055f, periodFor(0.055f));
                        const float patch = fbmP(lat / 6.f + seedX, s / 6.f, periodFor(6.f), 3, 2.f, 0.5f);
                        const float base = gravel ? 0.55f : 0.50f;
                        const float k = base + 0.105f * stone + 0.05f * patch +
                                        (gravel ? 0.05f * trackW : 0.f);
                        r = k * 1.05f;
                        g = k * 0.99f;
                        b = k * 0.90f;
                        rough = gravel ? (0.95f - 0.10f * trackW) : 0.95f;
                        relief = 0.004f * stone - (gravel ? 0.010f * trackW : 0.f);
                        if (!st.gravelEdge && !gravel) {
                            // A/B: no gravel strip, the sealed edge runs to the rim.
                            r = 0.40f; g = 0.39f; b = 0.38f; rough = 0.9f;
                        }
                        // A grass stripe grows down the middle of a forest track.
                        if (gravel && !twoLane && st.edge == LinePattern::None) {
                            const float gr = softBand(lat, 0.15f, 0.12f) * 0.4f * w *
                                             std::clamp(0.5f + fbmP(seedX + 5.f, s / 4.f, periodFor(4.f), 3, 2.f, 0.5f), 0.f, 1.f);
                            r += (0.20f - r) * gr;
                            g += (0.28f - g) * gr;
                            b += (0.12f - b) * gr;
                        }
                    } else {
                        // Weathered Norwegian asphalt: light gneiss aggregate
                        // exposed by studded tyres, not the fresh-bitumen black a
                        // 0.055 albedo bakes (which decodes to linear 0.004).
                        // 2 cm stones, not 3: at driving height the coarser
                        // grain read as gravel. Half the amplitude too, and a
                        // darker base - 0.42 was reading as concrete.
                        const float aggregate = noise2p(lat / 0.02f + seedX, s / 0.02f, periodFor(0.02f));
                        const float patchN = fbmP(lat / 8.f + seedX, s / 8.f, periodFor(8.f), 4, 2.f, 0.5f);
                        float k = 0.38f + 0.040f * aggregate + 0.050f * patchN;
                        // Polished wheel tracks, dirtier strip between them. The
                        // lightening is deliberately small: what sells a track
                        // is the ROUGHNESS step catching the sky, not albedo.
                        k += 0.025f * trackLight * w;
                        k -= 0.020f * betweenW * w;
                        rough = 0.88f - 0.26f * trackW * w;
                        relief = 0.002f * aggregate - 0.012f * trackW * w;

                        // The old edge repair — an extent, not an object.
                        if (repairRun > 0.f) {
                            const float a = softBand(lat - repairSide * (pavedHalf - 0.3f), 0.30f, 0.08f) * repairRun;
                            k -= 0.06f * a * w;
                            relief -= 0.002f * a * w;
                        }
                        r = k * 1.02f;
                        g = k;
                        b = k * 0.97f;

                        // ── cracks ──────────────────────────────────────────
                        // TRANSVERSE. Everything that varies along the crack is
                        // a function of lat and is evaluated here, per texel:
                        // the station wanders (a 1.5 m meander plus 15 cm
                        // kinks), the width breathes between half and one and a
                        // half of its nominal, and about a sixth of the length
                        // is simply missing in 10-30 cm gaps. That is what stops
                        // a crack being a ruled line across the carriageway.
                        float crackX = 0.f, crackXSealed = 0.f;
                        for (int c = 0; c < nXc; ++c) {
                            const XCrack& cr = xc[c];
                            // Which piece of the crack is this texel in?
                            float jog = 0.f;
                            if (lat >= cr.latA && lat <= cr.latB) jog = 0.f;
                            else if (lat >= cr.latC && lat <= cr.latD) jog = cr.jog;
                            else continue;
                            const float wander = 0.25f * noise2(lat / 1.5f + cr.sx * 3.1f, cr.sx) +
                                                 0.04f * noise2(lat / 0.15f + cr.sx * 5.7f, cr.sx + 11.f);
                            const float wmul = std::clamp(1.f + 0.5f * noise2(lat / 2.f + cr.sx * 2.3f, cr.sx + 3.f),
                                                          0.4f, 1.6f);
                            // 10-30 cm gaps: a 0.2 m lattice thresholded high.
                            const float gapN = noise2(lat / 0.20f + cr.sx * 9.7f, cr.sx + 7.f);
                            const float present = std::clamp((0.62f - gapN) * 8.f, 0.f, 1.f);
                            if (present <= 0.f) continue;
                            // Soft 5 cm ends so a partial crack tapers out
                            // instead of being clipped square.
                            const float endA = (jog == 0.f) ? cr.latA : cr.latC;
                            const float endB = (jog == 0.f) ? cr.latB : cr.latD;
                            const float ends = std::clamp((lat - endA) / 0.05f, 0.f, 1.f) *
                                               std::clamp((endB - lat) / 0.05f, 0.f, 1.f);
                            const float cv = cover(s - cr.at - jog + wander, cr.halfW * wmul, dv) *
                                             present * ends * w;
                            if (cr.sealed > 0.5f) crackXSealed = std::max(crackXSealed, cv);
                            else crackX = std::max(crackX, cv);
                        }
                        // Longitudinal hairlines wander along the OUTER track
                        // edges, where the pavement flexes most.
                        float crackL = 0.f;
                        {
                            const float wander = 0.05f * fbmP(seedX + 71.f, s / 12.f, periodFor(12.f), 3, 2.f, 0.5f);
                            const float run = std::clamp(fbmP(seedX + 83.f, s / 12.f, periodFor(12.f), 2, 2.f, 0.5f) * 2.f, 0.f, 1.f);
                            for (int kk = 0; kk < nTracks; ++kk) {
                                const float at = trackC[kk] + (trackC[kk] > 0.f ? 0.24f : -0.24f) + wander;
                                crackL = std::max(crackL, cover(lat - at, 0.012f, du) * run);
                            }
                            crackL *= w;
                        }
                        // LONGITUDINAL tar seams along the outer track edges.
                        // These are what carries the crack-sealing look to eye
                        // level: a transverse crack is one texel row and mips
                        // erase it two car lengths ahead, while a seam running
                        // away from the camera stays a continuous dark line all
                        // the way to the vanishing point.
                        //
                        // PER EDGE, not one field for all four: a single shared
                        // wander and a single shared run made both sides of the
                        // road start, stop and bend at the same station, which
                        // from above is a pair of ruled lines. Each edge now
                        // gets its own wander (±0.10 m at a 3 m scale, so it
                        // visibly snakes), its own width, and its own run hash.
                        float sealL = 0.f;
                        for (int kk = 0; kk < nTracks; ++kk) {
                            const float kf = static_cast<float>(kk);
                            const float wander = 0.10f * fbmP(seedX + 131.f + 61.f * kf, s / 3.f,
                                                              periodFor(3.f), 3, 2.f, 0.5f);
                            const float run = std::clamp(fbmP(seedX + 149.f + 83.f * kf, s / 24.f,
                                                              periodFor(24.f), 2, 2.f, 0.5f) * 2.6f + 0.35f, 0.f, 1.f);
                            const float wmul = std::clamp(1.f + 0.55f * fbmP(seedX + 311.f + 47.f * kf, s / 6.f,
                                                                             periodFor(6.f), 2, 2.f, 0.5f), 0.4f, 1.6f);
                            const float at = trackC[kk] + (trackC[kk] > 0.f ? 0.30f : -0.30f) + wander;
                            sealL = std::max(sealL, cover(lat - at, 0.026f * wmul, du) * run);
                        }
                        sealL *= w;
                        // Sealed cracks are the signature of a Norwegian road:
                        // black bitumen wiggles standing slightly proud.
                        const float sealAll = std::max(crackXSealed, sealL);
                        const float openAll = std::max(crackX, crackL);
                        if (openAll > 0.f) {
                            const float t = openAll;
                            r += (0.20f * r - r) * t;
                            g += (0.20f * g - g) * t;
                            b += (0.20f * b - b) * t;
                            relief -= 0.002f * t;
                        }
                        if (sealAll > 0.f) {
                            r += (0.085f - r) * sealAll;
                            g += (0.082f - g) * sealAll;
                            b += (0.080f - b) * sealAll;
                            rough += (0.50f - rough) * sealAll;
                            relief += 0.001f * sealAll;
                        }

                        // ── markings, composited by coverage ─────────────────
                        // Ragged edges: the paint's own boundary jitters by about
                        // a texel, so a worn line never reads as a vector stroke.
                        const float jit = du * 0.9f * noise2p(0.f, s / 0.25f, periodFor(0.25f)) * w;
                        // Some dash cells have a 0.3-1.5 m bite eaten out.
                        const auto dashWorn = [&](float on, float off) {
                            const float period = std::max(on + off, 1e-3f);
                            float ph = std::fmod(s, period);
                            if (ph < 0.f) ph += period;
                            float c = cover(ph - on * 0.5f, on * 0.5f, dv);
                            const int di = ((static_cast<int>(std::floor(s / period))) % 16 + 16) % 16;
                            if (hash2(sd + di * 7, 77) > 0.70f) {
                                const float bite = 0.15f + 0.60f * (hash2(sd + di * 7, 78) * 0.5f + 0.5f);
                                const float at = on * (0.25f + 0.5f * (hash2(sd + di * 7, 79) * 0.5f + 0.5f));
                                c *= 1.f - cover(ph - at, bite, dv) * w;
                            }
                            return c;
                        };
                        float covWhite = 0.f, covYellow = 0.f, ghostWhite = 0.f, ghostYellow = 0.f;
                        if (st.edge != LinePattern::None) {
                            float c = std::max(cover(lat + edgeCentre + jit, halfLine, du),
                                               cover(lat - edgeCentre + jit, halfLine, du));
                            float gc = std::max(cover(lat + edgeCentre + ghostOff, halfLine, du),
                                                cover(lat - edgeCentre - ghostOff, halfLine, du));
                            if (st.edge == LinePattern::Dashed) {
                                const float d = dashWorn(st.edgeOn, st.edgeOff);
                                c *= d;
                                gc *= d;
                            }
                            covWhite = c;
                            ghostWhite = gc * 0.25f * ghostSeg * w;
                        }
                        if (st.centre != LinePattern::None) {
                            float c = cover(lat + jit, halfLine, du);
                            float gc = cover(lat - ghostOff, halfLine, du);
                            if (st.centre == LinePattern::Dashed) {
                                const float d = dashWorn(st.centreOn, st.centreOff);
                                c *= d;
                                gc *= d;
                            }
                            covYellow = c;
                            ghostYellow = gc * 0.25f * ghostSeg * w;
                        }
                        // Fresh paint fades toward a grey-beige ghost of itself.
                        const std::array<float, 3> wornWhite = {0.62f, 0.60f, 0.55f};
                        const std::array<float, 3> wornYellow = {0.72f, 0.60f, 0.24f};
                        const auto faded = [&](const std::array<float, 3>& fresh,
                                               const std::array<float, 3>& worn) {
                            return std::array<float, 3>{fresh[0] + (worn[0] - fresh[0]) * segWear,
                                                        fresh[1] + (worn[1] - fresh[1]) * segWear,
                                                        fresh[2] + (worn[2] - fresh[2]) * segWear};
                        };
                        const auto paint = [&](const std::array<float, 3>& col, float c) {
                            if (c <= 0.f) return;
                            r += (col[0] - r) * c;
                            g += (col[1] - g) * c;
                            b += (col[2] - b) * c;
                            // Glass beads in the paint: markings are markedly
                            // smoother than the aggregate around them.
                            rough += (0.55f - rough) * c;
                        };
                        const auto fw = faded(st.whiteColor, wornWhite);
                        const auto fy = faded(st.yellowColor, wornYellow);
                        paint(fw, ghostWhite);
                        paint(fy, ghostYellow);
                        paint(fw, covWhite);
                        paint(fy, covYellow);
                    }

                    // Mean over the NOMINAL paved band, whatever surface won it:
                    // this is what the far painted road is tinted with, and a
                    // gravel road has to report its own gravel colour.
                    if (std::abs(lat) <= pavedHalf) {
                        sumR += r;
                        sumG += g;
                        sumB += b;
                        ++pavedCount;
                    }

                    const size_t o = (static_cast<size_t>(y) * W + x) * 4;
                    m.albedo[o + 0] = toByte(r);
                    m.albedo[o + 1] = toByte(g);
                    m.albedo[o + 2] = toByte(b);
                    m.roughMetal[o + 0] = 0u;
                    m.roughMetal[o + 1] = toByte(rough);
                    m.roughMetal[o + 2] = 0u;// asphalt is a dielectric
                    hgt[static_cast<size_t>(y) * W + x] = relief;
                }
            }

            // Normal map from the relief field: x clamps at the ribbon edge, y
            // wraps with the tile. Same convention as FacadeTexture (+G = +v).
            const float invDu = 1.f / std::max(du, 1e-4f);
            const float invDv = 1.f / std::max(dv, 1e-4f);
            const auto hAt = [&](int x, int y) {
                x = std::clamp(x, 0, W - 1);
                y = ((y % H) + H) % H;
                return hgt[static_cast<size_t>(y) * W + x];
            };
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x) {
                    const float dhdu = (hAt(x + 1, y) - hAt(x - 1, y)) * 0.5f * invDu;
                    const float dhdv = -(hAt(x, y + 1) - hAt(x, y - 1)) * 0.5f * invDv;
                    float nx = -dhdu, ny = -dhdv, nz = 1.f;
                    const float il = 1.f / std::sqrt(nx * nx + ny * ny + nz * nz);
                    const size_t o = (static_cast<size_t>(y) * W + x) * 4;
                    m.normal[o + 0] = toByte(nx * il * 0.5f + 0.5f);
                    m.normal[o + 1] = toByte(ny * il * 0.5f + 0.5f);
                    m.normal[o + 2] = toByte(nz * il * 0.5f + 0.5f);
                }

            if (pavedCount > 0) {
                m.meanPaved[0] = static_cast<float>(sumR / static_cast<double>(pavedCount));
                m.meanPaved[1] = static_cast<float>(sumG / static_cast<double>(pavedCount));
                m.meanPaved[2] = static_cast<float>(sumB / static_cast<double>(pavedCount));
            }
            return m;
        }

        // Bake the REPAIR-PATCH ATLAS: four variants side by side, each `cellRes`
        // square, one shared albedo + roughMetal pair. A patch is a resurfacing:
        // a rectangle sawn out of the carriageway and filled with a newer, finer,
        // DARKER mix (about 0.08 sRGB below the 0.38 road base), its own grain,
        // an uneven tone across it, and a bitumen seam right at the saw cut.
        //
        // The quad IS the patch — the decal is opaque (the Vulkan deferred path
        // has blend gotchas, and a repair has hard sawn edges anyway), so the
        // cell border must carry that edge itself: the outer few texels are the
        // dark seam, with a ~1.5-texel ramp inward so the cut is crisp without
        // aliasing. That also makes the atlas safe to mip and to sample near a
        // variant boundary — every cell's border is the same dark seam.
        [[nodiscard]] static RoadSurfaceMaps bakePatchAtlas(int cellRes = 256,
                                                            unsigned int seed = 4711u) {
            RoadSurfaceMaps m;
            const int C = std::max(cellRes, 16);
            constexpr int kVariants = 4;
            const int W = C * kVariants, H = C;
            m.width = W;
            m.height = H;
            const size_t nPx = static_cast<size_t>(W) * static_cast<size_t>(H);
            m.albedo.assign(nPx * 4, 255u);
            m.normal.assign(nPx * 4, 255u);
            m.roughMetal.assign(nPx * 4, 255u);
            // Flat tangent-space normal everywhere (the patch's own relief is far
            // below what a 1.5 cm lift already reads as).
            for (size_t p = 0; p < nPx; ++p) {
                m.normal[p * 4 + 0] = toByte(0.5f);
                m.normal[p * 4 + 1] = toByte(0.5f);
                m.normal[p * 4 + 2] = 255u;
            }
            const float fC = static_cast<float>(C);
            for (int v = 0; v < kVariants; ++v) {
                const int sv = static_cast<int>(seed) + v * 9173;
                const float sx = static_cast<float>(v) * 137.7f + static_cast<float>(seed % 601u) * 0.31f;
                // Each variant is its own mix: how much darker than the road, how
                // coarse the aggregate, how blotchy the tone.
                // MEASURED, not guessed: at 0.275-0.30 minus its own grain the
                // patch bottomed out around 0.21 and rendered at 0.45x the road
                // (34-52 against 88 in the road-top frame) — a black hole, not a
                // repair. The road base is 0.38; a resurfacing is about 0.08
                // below it, so the mix is centred at 0.307 and its noise is
                // small enough that it never dips below ~0.26.
                const float dark = 0.315f - 0.015f * (hash2(sv, 3) * 0.5f + 0.5f);// 0.30-0.315
                const float grainScale = 1.6f + 1.6f * (hash2(sv, 4) * 0.5f + 0.5f);
                const float seamW = 2.f + 1.5f * (hash2(sv, 5) * 0.5f + 0.5f);// 2-3.5 texels
                for (int y = 0; y < C; ++y) {
                    const float fy = static_cast<float>(y) + 0.5f;
                    for (int x = 0; x < C; ++x) {
                        const float fx = static_cast<float>(x) + 0.5f;
                        const float grain = noise2(fx / grainScale + sx, fy / grainScale + sx);
                        const float tone = fbm(fx / (fC * 0.22f) + sx, fy / (fC * 0.22f) + sx, 4, 2.f, 0.5f);
                        const float blotch = fbm(fx / (fC * 0.08f) + sx * 3.f, fy / (fC * 0.08f) + sx * 3.f,
                                                 3, 2.f, 0.5f);
                        float k = dark + 0.022f * grain + 0.016f * tone + 0.008f * blotch;
                        float rough = 0.93f - 0.05f * tone;
                        // The saw cut and its bitumen seam. Dark, but a SEAM and
                        // not a frame: a near-black ring all the way round made
                        // the quad read as a raised slab rather than a patch let
                        // into the surface, so it is lighter, thinner, and its
                        // depth varies along the cut the way a hand-run bead of
                        // sealant does.
                        const float dB = std::min(std::min(fx, fC - fx), std::min(fy, fC - fy));
                        const float seam = std::clamp((seamW - dB) / 1.5f + 0.5f, 0.f, 1.f) *
                                           (0.72f + 0.28f * (tone * 0.5f + 0.5f));
                        if (seam > 0.f) {
                            k += (0.145f - k) * seam;
                            rough += (0.62f - rough) * seam;
                        }
                        const size_t o = (static_cast<size_t>(y) * W + (v * C + x)) * 4;
                        m.albedo[o + 0] = toByte(k * 1.01f);
                        m.albedo[o + 1] = toByte(k);
                        m.albedo[o + 2] = toByte(k * 0.98f);
                        m.roughMetal[o + 0] = 0u;
                        m.roughMetal[o + 1] = toByte(rough);
                        m.roughMetal[o + 2] = 0u;// dielectric
                    }
                }
            }
            return m;
        }
        static constexpr int kPatchVariants = 4;

        // ── spawn helpers ────────────────────────────────────────────────────
        // First centerline sample position (with conformed elevation).
        [[nodiscard]] Vector3 startPoint() const {
            if (samples_.empty()) return Vector3::ZEROS();
            const Sample& s = samples_.front();
            return Vector3(s.pos.x, s.height, s.pos.z);
        }
        // Unit tangent at the start (XZ, Y=0).
        [[nodiscard]] Vector3 startForward() const {
            if (samples_.empty()) return Vector3(0.f, 0.f, 1.f);
            return samples_.front().tangent;
        }
        // Dense centerline polyline with conformed elevations (debug / markers).
        [[nodiscard]] const std::vector<Vector3>& centerlineSamples() const {
            // Lazily refresh the cached Vector3 view (positions + conformed Y).
            centerlineView_.resize(samples_.size());
            for (size_t i = 0; i < samples_.size(); ++i)
                centerlineView_[i] = Vector3(samples_[i].pos.x, samples_[i].height, samples_[i].pos.z);
            return centerlineView_;
        }

    private:
        // Per-centerline-sample frame.
        struct Sample {
            Vector3 pos{};      // XZ position (Y unused; height stored separately)
            float arcLength = 0.f;// cumulative arc length from the start
            float height = 0.f; // smoothed centerline elevation (after conformTo)
            Vector3 tangent{0.f, 0.f, 1.f};// unit forward (XZ)
            Vector3 right{1.f, 0.f, 0.f};  // unit right = tangent × up, normalised (XZ)
        };

        // ── centerline construction ──────────────────────────────────────────
        void buildCenterline() {
            samples_.clear();
            if (controlPoints_.empty()) return;

            if (controlPoints_.size() == 1) {
                Sample s;
                s.pos = controlPoints_[0];
                s.pos.y = 0.f;
                samples_.push_back(s);
                return;
            }

            // Flatten control points onto Y=0 for the curve (elevations come later).
            std::vector<Vector3> flat;
            flat.reserve(controlPoints_.size());
            for (const auto& cp : controlPoints_) flat.emplace_back(cp.x, 0.f, cp.z);

            CatmullRomCurve3 curve(flat, /*closed*/ false, CatmullRomCurve3::centripetal);

            const int sps = std::max(params_.samplesPerSegment, 1);
            const int total = static_cast<int>(controlPoints_.size() - 1) * sps + 1;

            samples_.reserve(static_cast<size_t>(total));
            Vector3 p{};
            for (int i = 0; i < total; ++i) {
                const float t = (total > 1) ? static_cast<float>(i) / static_cast<float>(total - 1) : 0.f;
                curve.getPoint(t, p);
                Sample s;
                s.pos = Vector3(p.x, 0.f, p.z);
                samples_.push_back(s);
            }

            // Cumulative arc length along the XZ polyline.
            samples_[0].arcLength = 0.f;
            for (size_t i = 1; i < samples_.size(); ++i) {
                const Vector3& a = samples_[i - 1].pos;
                const Vector3& b = samples_[i].pos;
                const float dx = b.x - a.x, dz = b.z - a.z;
                samples_[i].arcLength = samples_[i - 1].arcLength + std::sqrt(dx * dx + dz * dz);
            }

            // Unit tangent (central difference) and right vector (tangent × up).
            const Vector3 up{0.f, 1.f, 0.f};
            const size_t n = samples_.size();
            for (size_t i = 0; i < n; ++i) {
                const size_t im = (i == 0) ? 0 : i - 1;
                const size_t ip = (i + 1 < n) ? i + 1 : n - 1;
                Vector3 tan(samples_[ip].pos.x - samples_[im].pos.x, 0.f,
                            samples_[ip].pos.z - samples_[im].pos.z);
                if (tan.length() < 1e-6f) tan = Vector3(0.f, 0.f, 1.f);
                tan.normalize();
                samples_[i].tangent = tan;
                // right = tangent × up  (points to the +X side of a +Z heading).
                Vector3 right = tan.clone().cross(up);
                if (right.length() < 1e-6f) right = Vector3(1.f, 0.f, 0.f);
                right.normalize();
                samples_[i].right = right;
            }
        }

        // ── spatial hash for nearest-sample queries ──────────────────────────
        void buildSpatialHash() {
            cellSize_ = std::max(corridorHalfWidth(), 0.5f);
            hash_.clear();
            for (size_t i = 0; i < samples_.size(); ++i) {
                const auto key = cellKey(samples_[i].pos.x, samples_[i].pos.z);
                hash_[key].push_back(static_cast<int>(i));
            }
        }

        [[nodiscard]] long long cellKey(float x, float z) const {
            const int cx = static_cast<int>(std::floor(x / cellSize_));
            const int cz = static_cast<int>(std::floor(z / cellSize_));
            return packKey(cx, cz);
        }
        static long long packKey(int cx, int cz) {
            // Pack two 32-bit cell coords into one 64-bit key.
            return (static_cast<long long>(static_cast<unsigned int>(cx)) << 32) |
                   static_cast<unsigned int>(cz);
        }

        // Gather candidate sample indices from the 3×3 neighbour cells and return
        // the nearest-by-point index. Populates scratchCandidates_ for the caller
        // (distanceToCenter uses it for point-to-segment). Falls back to scanning
        // every sample if the local cells are empty (correctness over speed).
        [[nodiscard]] int nearestSampleIndex(float x, float z) const {
            scratchCandidates_.clear();
            if (samples_.empty()) return -1;

            const int cx = static_cast<int>(std::floor(x / cellSize_));
            const int cz = static_cast<int>(std::floor(z / cellSize_));
            for (int dz = -1; dz <= 1; ++dz)
                for (int dx = -1; dx <= 1; ++dx) {
                    auto it = hash_.find(packKey(cx + dx, cz + dz));
                    if (it == hash_.end()) continue;
                    for (int idx : it->second) scratchCandidates_.push_back(idx);
                }

            if (scratchCandidates_.empty()) {
                // No populated neighbour cells — scan all samples.
                scratchCandidates_.resize(samples_.size());
                for (size_t i = 0; i < samples_.size(); ++i)
                    scratchCandidates_[i] = static_cast<int>(i);
            }

            int best = -1;
            float bestSq = std::numeric_limits<float>::max();
            for (int idx : scratchCandidates_) {
                const auto& p = samples_[idx].pos;
                const float ddx = x - p.x, ddz = z - p.z;
                const float dSq = ddx * ddx + ddz * ddz;
                if (dSq < bestSq) {
                    bestSq = dSq;
                    best = idx;
                }
            }
            return best;
        }

        // Squared point-to-segment distance against segment (i0, i1) in XZ.
        [[nodiscard]] float distPointToSegmentSq(float x, float z, int i0, int i1) const {
            const Vector3& a = samples_[i0].pos;
            const Vector3& b = samples_[i1].pos;
            const float abx = b.x - a.x, abz = b.z - a.z;
            const float apx = x - a.x, apz = z - a.z;
            const float abLenSq = abx * abx + abz * abz;
            float t = (abLenSq > 1e-12f) ? (apx * abx + apz * abz) / abLenSq : 0.f;
            t = std::clamp(t, 0.f, 1.f);
            const float cx = a.x + t * abx, cz = a.z + t * abz;
            const float dx = x - cx, dz = z - cz;
            return dx * dx + dz * dz;
        }

        // ── banking ──────────────────────────────────────────────────────────
        // Signed camber roll (radians) at sample i, banked INTO the curve so the
        // outer edge lifts. Curvature is estimated from the turn of the tangent
        // between the neighbouring samples (the signed XZ cross product).
        [[nodiscard]] float bankingAngle(size_t i) const {
            const size_t n = samples_.size();
            if (n < 3 || i == 0 || i + 1 >= n) return 0.f;
            const Vector3& t0 = samples_[i - 1].tangent;
            const Vector3& t1 = samples_[i + 1].tangent;
            // Signed turn: cross.y of the two XZ tangents (>0 = left turn).
            const float cross = t0.z * t1.x - t0.x * t1.z;
            // Normalise the turn by an arc-length span so it reads as curvature.
            const float ds = std::max(samples_[i + 1].arcLength - samples_[i - 1].arcLength, 1e-3f);
            const float curvature = cross / ds;// signed (1/m)-ish
            const float t = std::clamp(curvature * params_.bankingCurvatureScale, -1.f, 1.f);
            // Sign: a left turn (cross>0) should lift the right (+lat) edge so the
            // road banks into the curve. lat*sin(bank) is the per-vertex camber, so
            // bank>0 lifts +lat. Left turn => outer edge is +lat => bank>0. Matches t.
            return t * params_.maxBanking;
        }

        // ── small numeric helpers (mirrors TerrainGenerator) ─────────────────
        static float lerp(float a, float b, float t) { return a + t * (b - a); }
        static unsigned char toByte(float v) {
            return static_cast<unsigned char>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
        }

        // ── value-noise / fBm for asphalt grain (deterministic, hash-based) ──
        // Self-contained integer hash so the generator needs no permutation table
        // and the texture is reproducible across runs.
        static float hash2(int x, int y) {
            unsigned int h = static_cast<unsigned int>(x) * 374761393u +
                             static_cast<unsigned int>(y) * 668265263u;
            h = (h ^ (h >> 13)) * 1274126177u;
            h ^= h >> 16;
            return static_cast<float>(h & 0xFFFFFFu) / static_cast<float>(0xFFFFFF) * 2.f - 1.f;
        }
        static float fade(float t) { return t * t * t * (t * (t * 6.f - 15.f) + 10.f); }
        // 2D value noise, output ~[-1, 1].
        static float noise2(float x, float y) {
            const int xi = static_cast<int>(std::floor(x));
            const int yi = static_cast<int>(std::floor(y));
            const float xf = x - static_cast<float>(xi);
            const float yf = y - static_cast<float>(yi);
            const float u = fade(xf), v = fade(yf);
            const float n00 = hash2(xi, yi);
            const float n10 = hash2(xi + 1, yi);
            const float n01 = hash2(xi, yi + 1);
            const float n11 = hash2(xi + 1, yi + 1);
            const float x1 = lerp(n00, n10, u);
            const float x2 = lerp(n01, n11, u);
            return lerp(x1, x2, v);
        }
        // Same value noise, but the Y lattice WRAPS every `period` cells. The
        // marking tile is shared and repeats every tileLength metres along the
        // road; a non-periodic field would draw a visible line across the
        // carriageway at every tile seam. Callers pick cell sizes that divide the
        // tile exactly (tileLength / cellSize == period, an integer).
        static float noise2p(float x, float y, int period) {
            const int P = std::max(period, 1);
            const int xi = static_cast<int>(std::floor(x));
            const int yi = static_cast<int>(std::floor(y));
            const float xf = x - static_cast<float>(xi);
            const float yf = y - static_cast<float>(yi);
            const float u = fade(xf), v = fade(yf);
            const int y0 = ((yi % P) + P) % P;
            const int y1 = (y0 + 1) % P;
            const float n00 = hash2(xi, y0);
            const float n10 = hash2(xi + 1, y0);
            const float n01 = hash2(xi, y1);
            const float n11 = hash2(xi + 1, y1);
            return lerp(lerp(n00, n10, u), lerp(n01, n11, u), v);
        }
        // fBm over noise2p. Octave i runs at lacunarity^i, so its wrap period
        // scales with it — integer as long as the caller's base period is.
        static float fbmP(float x, float y, int period, int oct, float lac, float gain) {
            float f = 1.f, a = 1.f, sum = 0.f, norm = 0.f;
            int p = std::max(period, 1);
            for (int i = 0; i < oct; ++i) {
                sum += a * noise2p(x * f, y * f, p);
                norm += a;
                f *= lac;
                p = static_cast<int>(std::lround(static_cast<double>(p) * static_cast<double>(lac)));
                a *= gain;
            }
            return norm > 0.f ? sum / norm : 0.f;
        }
        static float fbm(float x, float y, int oct, float lac, float gain) {
            float f = 1.f, a = 1.f, sum = 0.f, norm = 0.f;
            for (int i = 0; i < oct; ++i) {
                sum += a * noise2(x * f, y * f);
                norm += a;
                f *= lac;
                a *= gain;
            }
            return norm > 0.f ? sum / norm : 0.f;
        }

        RoadParams params_;
        std::vector<Vector3> controlPoints_;
        std::vector<Sample> samples_;
        bool conformed_ = false;

        // Spatial hash (cell == corridorHalfWidth so a query scans 3×3 cells).
        float cellSize_ = 1.f;
        std::unordered_map<long long, std::vector<int>> hash_;

        // Mutable scratch reused by the const query path (no per-call allocation
        // on the hot path; not part of the logical const state).
        mutable std::vector<int> scratchCandidates_;
        mutable std::vector<Vector3> centerlineView_;
    };

}// namespace threepp::road

#endif//THREEPP_EXTRAS_ROAD_ROADGENERATOR_HPP
