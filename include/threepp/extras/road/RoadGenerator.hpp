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
            return smoothstep(corridor, paved, d);
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
        [[nodiscard]] std::shared_ptr<BufferGeometry> buildSurface() const {
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

            std::vector<float> positions;
            std::vector<float> normals;
            std::vector<float> uvs;
            positions.reserve(n * 4 * 3);
            normals.reserve(n * 4 * 3);
            uvs.reserve(n * 4 * 2);

            const Vector3 up{0.f, 1.f, 0.f};
            for (size_t i = 0; i < n; ++i) {
                const Sample& s = samples_[i];
                // Banking: roll the cross-section about the tangent by an amount
                // proportional to curvature, clamped to ±maxBanking. Sign chosen so
                // the road banks into the curve (outer edge lifts).
                const float bank = bankingAngle(i);
                // Rotate the (right, up) basis about the tangent by `bank`. The
                // right vector tilts toward up; the per-vertex Y gets the camber.
                const float cb = std::cos(bank), sb = std::sin(bank);
                // Tilted lateral & vertical basis (in the plane spanned by right & up).
                const Vector3& right = s.right;
                const float vScale = s.arcLength / tile;

                for (int k = 0; k < 4; ++k) {
                    const float lat = off[k];
                    // Lateral displacement: right rotated about tangent by `bank`.
                    // right' = right*cos(bank) + up*sin(bank)  (a roll in the cross
                    // plane). The vertex sits at center + lat * right'.
                    const float px = s.pos.x + lat * (right.x * cb);
                    const float pz = s.pos.z + lat * (right.z * cb);
                    // Camber raises the outer edge: height contribution = lat*sin(bank)
                    // (lat sign distinguishes left/right of centerline).
                    float py = s.height + lat * sb;
                    if (paved[k]) py += raise;

                    positions.push_back(px);
                    positions.push_back(py);
                    positions.push_back(pz);

                    // Provisional normal (recomputed below); seed with up.
                    normals.push_back(0.f);
                    normals.push_back(1.f);
                    normals.push_back(0.f);

                    uvs.push_back(uCoord[k]);
                    uvs.push_back(vScale);
                }
            }

            // Index: 3 quads per segment, 2 triangles per quad. Cross-section of
            // sample i occupies vertices [i*4 .. i*4+3].
            std::vector<unsigned int> indices;
            indices.reserve((n - 1) * 3 * 6);
            for (size_t i = 0; i + 1 < n; ++i) {
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
        static float smoothstep(float e0, float e1, float x) {
            const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
            return t * t * (3.f - 2.f * t);
        }
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
