#include "threepp/objects/DisplacedMesh.hpp"

// kOceanCascade1Rot* — cascade-1 sample-domain rotation, shared with the GLSL
// samplers (water_displace.comp and the other ocean passes). Header is cross-language
// (pure constants + a POD struct); including it here keeps CPU/GPU parity
// literal-exact.
#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace threepp {

    DisplacedMesh::DisplacedMesh(const std::shared_ptr<BufferGeometry>& geometry,
                                 const std::shared_ptr<Material>& material)
        : Mesh(geometry, material) {}

    const std::string& DisplacedMesh::type() const {
        static const std::string typeName = "DisplacedMesh";
        return typeName;
    }

    namespace {
        // A wave-model switch's environment override: 1 / 0 forces it on /
        // off, unset or empty leaves the mesh's own field in charge (-1).
        int envSwitch(const char* name) {
            const char* e = std::getenv(name);
            if (!e || !*e) return -1;
            return e[0] == '0' ? 0 : 1;
        }
    }// namespace

    namespace {
        float smoothstepf(float e0, float e1, float x) {
            const float t = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f);
            return t * t * (3.f - 2.f * t);
        }
    }// namespace

    bool DisplacedMesh::cwmOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_CWM");
        return env < 0 ? params.cwm : env == 1;
    }

    bool DisplacedMesh::boundWavesOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_BOUND_WAVES");
        return env < 0 ? params.boundWaves : env == 1;
    }

    bool DisplacedMesh::waveReflectionsOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_WAVE_REFLECTIONS");
        return env < 0 ? params.waveReflections : env == 1;
    }

    bool DisplacedMesh::shortWaveSpectrumOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_SHORT_WAVE_SPECTRUM");
        return env < 0 ? params.shortWaveSpectrum : env == 1;
    }

    bool DisplacedMesh::hullContactOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_HULL_CONTACT");
        return env < 0 ? params.hullContact : env == 1;
    }

    float DisplacedMesh::longWaveHandoff(float tileSize0, uint32_t textureSize0, float tileSize1) {
        if (!(tileSize0 > 0.f && tileSize1 > 0.f)) return 0.f;
        // The large tile's grid cannot hold a wave shorter than eight of its
        // texels as anything but a few points (the B-spline read loses most
        // of it), so a coarse cascade 0 hands off higher up.
        const float texels = textureSize0 > 0u ? 8.f * tileSize0 / float(textureSize0) : 0.f;
        return std::min(std::max(tileSize1 / 6.f, texels), tileSize1);
    }

    float DisplacedMesh::peakWavelength(float windSpeed, float fetch) {
        // phillips_spectrum.comp: L = 1.1624764 U^2 / g (its Hs fix), times
        // (fetch / Ffd)^(2/3) for a young sea; the peak is at k L = 0.894.
        if (!(windSpeed > 0.f)) return 0.f;
        double L = 1.1624764 * double(windSpeed) * windSpeed / 9.81;
        if (fetch > 0.f) {
            const double ffd = 1609.0 * double(windSpeed) * windSpeed;
            L *= std::pow(std::clamp(double(fetch) / ffd, 0.005, 1.0), 2.0 / 3.0);
        }
        return static_cast<float>(2.0 * 3.14159265358979 * L / 0.894);
    }

    float DisplacedMesh::openSeaTile(float tileSize0, float windSpeed, float fetch,
                                     float swellHeight, float swellPeriod) {
        float peak = peakWavelength(windSpeed, fetch);
        if (swellHeight > 0.f && swellPeriod > 0.f)
            peak = std::max(peak, static_cast<float>(9.81 * swellPeriod * swellPeriod / (2.0 * 3.14159265358979)));
        if (!(peak > tileSize0 / 6.f)) return tileSize0;
        return 250.f * std::ceil(6.f * peak / 250.f);
    }

    float DisplacedMesh::effectiveChoppiness() const {
        return cwmOn() ? 1.f : params.choppiness;
    }

    float DisplacedMesh::effectiveCrestSharpness() const {
        static const float env = [] {
            const char* e = std::getenv("THREEPP_OCEAN_CREST_SHARPNESS");
            return (e && *e) ? static_cast<float>(std::atof(e)) : -1.f;
        }();
        if (!cwmOn()) return 0.f;
        return std::clamp(env >= 0.f ? env : params.crestSharpness, 0.f, 4.f);
    }

    namespace {

        struct Vec2f {
            float x = 0.f;
            float y = 0.f;
        };

        // Filtered, repeat-wrapped sample of an RG32F cascade field at tile
        // coordinates (u, v) = sample-domain XZ / tileSize. Texel i's value
        // sits at (i+0.5)/dim, as with GPU texture(). `bicubic` selects the
        // uniform cubic B-spline over the 4x4 neighbourhood, which is what
        // sampleBicubicR (water_displace.comp) and sampleBicubicRG
        // (ocean_cascade.glsl) evaluate with four bilinear taps; otherwise
        // bilinear. Returns the raw (unnormalized) R and G.
        Vec2f sampleField(const std::vector<float>& field, uint32_t dim,
                          float u, float v, bool bicubic) {
            const float fdim = float(dim);
            auto wrap = [fdim](float x) {
                float w = std::fmod(x, fdim);
                if (w < 0.f) w += fdim;
                return w;
            };
            const float fx = wrap(u * fdim - 0.5f);
            const float fz = wrap(v * fdim - 0.5f);
            const int ix = int(std::floor(fx));
            const int iz = int(std::floor(fz));
            const float tx = fx - float(ix);
            const float tz = fz - float(iz);

            auto texel = [&](int x, int z) {
                const int d = int(dim);
                x %= d;
                if (x < 0) x += d;
                z %= d;
                if (z < 0) z += d;
                const size_t idx = (size_t(z) * dim + size_t(x)) * 2u;
                return Vec2f{field[idx], field[idx + 1]};
            };

            float wx[4], wz[4];
            int first;
            int taps;
            if (bicubic) {
                auto bspline = [](float t, float w[4]) {
                    const float t2 = t * t;
                    const float t3 = t2 * t;
                    w[0] = (-t3 + 3.f * t2 - 3.f * t + 1.f) / 6.f;
                    w[1] = (3.f * t3 - 6.f * t2 + 4.f) / 6.f;
                    w[2] = (-3.f * t3 + 3.f * t2 + 3.f * t + 1.f) / 6.f;
                    w[3] = t3 / 6.f;
                };
                bspline(tx, wx);
                bspline(tz, wz);
                first = -1;
                taps = 4;
            } else {
                wx[0] = 1.f - tx;
                wx[1] = tx;
                wz[0] = 1.f - tz;
                wz[1] = tz;
                first = 0;
                taps = 2;
            }

            Vec2f r;
            for (int j = 0; j < taps; ++j) {
                for (int i = 0; i < taps; ++i) {
                    const Vec2f t = texel(ix + first + i, iz + first + j);
                    const float w = wx[i] * wz[j];
                    r.x += w * t.x;
                    r.y += w * t.y;
                }
            }
            return r;
        }

        bool hasField(const DisplacedMesh::CascadeField& cf, const std::vector<float>& field) {
            return cf.dim > 0 && cf.tileSize > 0.f &&
                   field.size() >= size_t(cf.dim) * cf.dim * 2u;
        }

        // Cascade i's sample-domain position for a world XZ. Cascade 1 is
        // sampled in a rotated domain (repeat-lattice break, see
        // vulkan_shared.h); mirror of oceanC1Domain().
        Vec2f cascadeDomain(uint32_t i, float worldX, float worldZ) {
            if (i != 1) return {worldX, worldZ};
            return {kOceanCascade1RotCos * worldX - kOceanCascade1RotSin * worldZ,
                    kOceanCascade1RotSin * worldX + kOceanCascade1RotCos * worldZ};
        }

        // Cascades 0 and 1 are B-spline reconstructed on the GPU, cascade 2
        // bilinear (see sampleDisplacement in water_displace.comp).
        bool cascadeBicubic(uint32_t i) {
            return i < 2;
        }

    }// namespace

    float DisplacedMesh::sampleHeight(float worldX, float worldZ,
                                      uint32_t cascadeMask) const {
        // Sticky opt-in: the Vulkan renderer only records the GPU→host height
        // copies (and the per-frame mirror memcpy) once something actually
        // queries the CPU wave field. First-ever call may return 0 until a
        // frame has recorded the copies and its fence has retired — the
        // mirror's fixed kFramesInFlight-frame latency (see the header).
        wantsHeightReadback = true;

        // World-space horizontal displacement D(q) of the selected cascades at
        // rest position q, choppiness applied. Under the Choppy Wave Model it
        // is the full displacement (choppiness 1); the shader's per-vertex
        // limiter, which lowers it where the Jacobian would fall under
        // kCwmMinJacobian, is not mirrored here (the audit reports where it
        // binds). Crest sharpness draws cascades 0 and 1 with 1 + s of it.
        // In the steep-crest mode the 1 + s is gated by the crests' own
        // steepness at the rest point (crestGateAt, below).
        const float chop = effectiveChoppiness();
        const float sharpness = effectiveCrestSharpness();
        // The wave climate's gains at a rest point (waveClimateGain), on every
        // field of its cascade, as water_displace.comp applies them, times
        // the gains over the shore's depth (shoreGain).
        const bool climate = hasWaveClimate() || !shoreDepth.data.empty();
        // The long cascades' turn over the bed (shoreBend): cascade i < 2 is
        // read at q + its departure, its displacement turned with its crests.
        // Taken once, at the world point: the rest point the iteration below
        // finds is within the waves' horizontal travel of it (a metre or so),
        // and the bend changes over many metres (it was five table reads at
        // every step of the iteration).
        const ShoreBend b = shoreDepth.data.empty() ? ShoreBend{} : shoreBend(worldX, worldZ);
        auto gains = [&](float qx, float qz) {
            std::array<float, 3> g = hasWaveClimate() ? waveClimateGain(qx, qz) : std::array<float, 3>{1.f, 1.f, 1.f};
            if (!shoreDepth.data.empty()) {
                const std::array<float, 3> s = shoreGainWith(qx, qz, b.amp);
                for (int c = 0; c < 3; ++c) g[c] *= s[c];
            }
            return g;
        };
        // The steep-crest gate at rest point q: water_displace.comp's
        // crestGate, the largest of cascades 0 and 1's
        // smoothstep(kCrestGateLow, kCrestGateHigh, k_m a).
        auto crestGateAt = [&](float qx, float qz) {
            float m = 0.f;
            const std::array<float, 3> g = climate ? gains(qx, qz) : std::array<float, 3>{1.f, 1.f, 1.f};
            for (uint32_t i = 0; i < 2; ++i) {
                if ((cascadeMask & (1u << i)) == 0u) continue;
                const CascadeField& cf = heightFields[i];
                if (!(cf.boundKm > 0.f) || !hasField(cf, cf.data) || !hasField(cf, cf.disp)) continue;
                const float invTile = 1.f / cf.tileSize;
                const Vec2f p = b.on ? cascadeDomain(i, qx + b.d[2 * i], qz + b.d[2 * i + 1]) : cascadeDomain(i, qx, qz);
                const Vec2f h = sampleField(cf.data, cf.dim, p.x * invTile, p.y * invTile, cascadeBicubic(i));
                const Vec2f v = sampleField(cf.disp, cf.dim, p.x * invTile, p.y * invTile, cascadeBicubic(i));
                const float s = invTile * params.waveScale * g[i];
                const float a = std::sqrt(h.x * h.x + v.x * v.x + v.y * v.y) * s;
                m = std::max(m, smoothstepf(kCrestGateLow[i], kCrestGateHigh[i], cf.boundKm * a));
            }
            return m;
        };
        auto displacement = [&](float qx, float qz) {
            Vec2f d;
            const std::array<float, 3> g = climate ? gains(qx, qz) : std::array<float, 3>{1.f, 1.f, 1.f};
            const float s01 = sharpness > 0.f ? sharpness * crestGateAt(qx, qz) : 0.f;
            for (uint32_t i = 0; i < 3; ++i) {
                if ((cascadeMask & (1u << i)) == 0u) continue;
                const CascadeField& cf = heightFields[i];
                if (!hasField(cf, cf.disp)) continue;
                const float invTile = 1.f / cf.tileSize;
                const bool  turn = i < 2u ? b.on : b.on2;
                const float dx = i < 2u ? b.d[2 * i] : b.d2[0], dz = i < 2u ? b.d[2 * i + 1] : b.d2[1];
                const Vec2f p = turn ? cascadeDomain(i, qx + dx, qz + dz) : cascadeDomain(i, qx, qz);
                Vec2f v = sampleField(cf.disp, cf.dim, p.x * invTile, p.y * invTile,
                                      cascadeBicubic(i));
                if (i == 1) {
                    // Domain vector back to world; mirror of oceanC1World().
                    v = {kOceanCascade1RotCos * v.x + kOceanCascade1RotSin * v.y,
                         -kOceanCascade1RotSin * v.x + kOceanCascade1RotCos * v.y};
                }
                if (turn) {
                    const float c = i < 2u ? b.turn[2 * i] : b.turn2[0], s = i < 2u ? b.turn[2 * i + 1] : b.turn2[1];
                    v = {c * v.x - s * v.y, s * v.x + c * v.y};
                }
                // Sharpness 0 keeps the historical sum, bit for bit.
                const float c = (i < 2 && s01 > 0.f) ? 1.f + s01 : 1.f;
                if (climate) v = {v.x * g[i], v.y * g[i]};
                d.x += v.x * invTile * c;
                d.y += v.y * invTile * c;
            }
            d.x *= chop;
            d.y *= chop;
            return d;
        };

        // Solve x = q + D(q) for the rest position q. The map q -> x - D(q)
        // contracts wherever the surface does not fold (Jacobian > 0), so
        // fixed-point iteration converges; near a fold it can oscillate, so
        // keep the iterate with the smallest residual |q + D(q) - x|, which
        // is the length of the step it produces. The error shrinks by the
        // displacement slope choppiness*a*k per step: a few steps at the
        // default choppiness, 14 at choppiness 1 on a steep (ak = 0.59) wave.
        float bestX = worldX;
        float bestZ = worldZ;
        if (chop != 0.f) {
            constexpr int kMaxIterations = 16;
            constexpr float kTolerance = 1e-3f;// metres
            float qx = worldX;
            float qz = worldZ;
            float bestResidual = std::numeric_limits<float>::max();
            for (int it = 0; it < kMaxIterations; ++it) {
                const Vec2f d = displacement(qx, qz);
                const float nx = worldX - d.x;
                const float nz = worldZ - d.y;
                const float residual = std::hypot(nx - qx, nz - qz);
                if (residual < bestResidual) {
                    bestResidual = residual;
                    bestX = qx;
                    bestZ = qz;
                }
                if (residual < kTolerance) break;
                qx = nx;
                qz = nz;
            }
        }

        float total = 0.f;
        float bound = 0.f;// second-order harmonic, already in world metres
        const std::array<float, 3> gq = climate ? gains(bestX, bestZ) : std::array<float, 3>{1.f, 1.f, 1.f};
        const ShoreBend& bq = b;
        for (uint32_t i = 0; i < 3; ++i) {
            if ((cascadeMask & (1u << i)) == 0u) continue;
            const CascadeField& cf = heightFields[i];
            if (!hasField(cf, cf.data)) continue;
            const float invTile = 1.f / cf.tileSize;
            const Vec2f p = (i < 2u ? bq.on : bq.on2)
                                    ? cascadeDomain(i, bestX + (i < 2u ? bq.d[2 * i] : bq.d2[0]),
                                                    bestZ + (i < 2u ? bq.d[2 * i + 1] : bq.d2[1]))
                                    : cascadeDomain(i, bestX, bestZ);
            Vec2f h = sampleField(cf.data, cf.dim, p.x * invTile, p.y * invTile,
                                  cascadeBicubic(i));
            if (climate) h.x *= gq[i];
            total += h.x * invTile;
            if (cf.boundKm > 0.f && hasField(cf, cf.disp)) {
                // eta2 = (k_m / 2)(eta^2 - |R[eta]|^2) of the drawn
                // (waveScale'd) first-order field, R[eta] = -D at choppiness
                // 1: water_displace.comp's boundHeight().
                Vec2f d = sampleField(cf.disp, cf.dim, p.x * invTile, p.y * invTile,
                                      cascadeBicubic(i));
                if (climate) d = {d.x * gq[i], d.y * gq[i]};
                const float s = invTile * params.waveScale;
                const float e = h.x * s, rx = d.x * s, rz = d.y * s;
                // Gated up to kSteepBoundGain in the steep-crest mode and
                // held at the Stokes cap of the band's local amplitude, as
                // water_displace.comp's boundHeight does.
                const float a   = std::sqrt(e * e + rx * rx + rz * rz);
                const float ke  = cf.boundKm * (sharpness > 0.f
                                                    ? 1.f + (kSteepBoundGain - 1.f) *
                                                                smoothstepf(kCrestGateLow[i], kCrestGateHigh[i], cf.boundKm * a)
                                                    : 1.f);
                const float cap = std::min(1.f, 2.f * kStokesHarmonicCap / std::max(ke * a, 1e-6f));
                bound += 0.5f * ke * cap * (e * e - rx * rx - rz * rz);
            }
        }
        return total * params.waveScale + bound;
    }

    void DisplacedMesh::setWaveClimate(std::vector<float> data, uint32_t width, uint32_t height,
                                       float originX, float originZ, float sizeX, float sizeZ) {
        if (width < 2u || height == 0u || data.size() != size_t(width) * height * 4u) {
            throw std::invalid_argument("DisplacedMesh::setWaveClimate: data must hold width x height x 4 floats");
        }
        if (!(sizeX > 0.f) || !(sizeZ > 0.f)) {
            throw std::invalid_argument("DisplacedMesh::setWaveClimate: the map's extent must be positive");
        }
        const uint64_t v = waveClimate_.version + 1u;
        waveClimate_ = WaveClimate{std::move(data), width, height, originX, originZ, sizeX, sizeZ, v};
    }

    void DisplacedMesh::clearWaveClimate() {
        const uint64_t v = waveClimate_.version + 1u;
        waveClimate_ = WaveClimate{};
        waveClimate_.version = v;
    }

    std::array<float, 3> DisplacedMesh::waveClimateGain(float x, float z) const {
        const WaveClimate& w = waveClimate_;
        if (w.width == 0u) return {1.f, 1.f, 1.f};
        // Texel centres at (i + 0.5) / width of the extent; clamp to the edge
        // texels as the shaders' reads do (ocean_cascade.glsl, oceanClimate).
        const float fx = std::clamp((x - w.originX) / w.sizeX * float(w.width) - 0.5f, 0.f, float(w.width - 1u));
        const float fz = std::clamp((z - w.originZ) / w.sizeZ * float(w.height) - 0.5f, 0.f, float(w.height - 1u));
        const uint32_t i0 = std::min(uint32_t(fx), w.width - 1u);
        const uint32_t j0 = std::min(uint32_t(fz), w.height - 1u);
        const uint32_t i1 = std::min(i0 + 1u, w.width - 1u);
        const uint32_t j1 = std::min(j0 + 1u, w.height - 1u);
        const float tx = fx - float(i0);
        const float tz = fz - float(j0);
        std::array<float, 3> g{};
        for (uint32_t c = 0; c < 3; ++c) {
            const auto at = [&](uint32_t i, uint32_t j) { return w.data[(size_t(j) * w.width + i) * 4u + c]; };
            const float a = at(i0, j0) + (at(i1, j0) - at(i0, j0)) * tx;
            const float b = at(i0, j1) + (at(i1, j1) - at(i0, j1)) * tx;
            g[c] = a + (b - a) * tz;
        }
        return g;
    }

    std::array<float, 3> DisplacedMesh::shoreGain(float x, float z) const {
        return shoreGainWith(x, z, shoreBend(x, z).amp);
    }

    std::array<float, 3> DisplacedMesh::shoreGainWith(float x, float z, const std::array<float, 3>& bendAmp) const {
        // ocean_shore_depth.glsl, line for line: its header's words, then the depths.
        constexpr size_t kHeaderWords = 20;
        const std::vector<float>& v = shoreDepth.data;
        if (v.size() < kHeaderWords) return {1.f, 1.f, 1.f};
        uint32_t res = 0;
        std::memcpy(&res, &v[16], sizeof(res));
        if (res < 2u || v.size() < kHeaderWords + 2u * size_t(res) * res) return {1.f, 1.f, 1.f};
        const float n = static_cast<float>(res);
        auto depthIn = [&](uint32_t k, float& wgt) {
            wgt = 0.f;
            const float* w = &v[k * 4u];
            if (w[3] < 0.5f) return 0.f;
            const float u  = (x - w[0]) / (w[2] * n);
            const float t  = (z - w[1]) / (w[2] * n);
            const float e  = std::min(std::min(u, 1.f - u), std::min(t, 1.f - t));
            const float s  = std::clamp(e * 16.f, 0.f, 1.f);
            wgt = s * s * (3.f - 2.f * s);
            if (wgt <= 0.f) return 0.f;
            const float fx = std::clamp(u * n - 0.5f, 0.f, n - 1.f);
            const float fz = std::clamp(t * n - 0.5f, 0.f, n - 1.f);
            const uint32_t i0 = std::min(static_cast<uint32_t>(fx), res - 2u);
            const uint32_t j0 = std::min(static_cast<uint32_t>(fz), res - 2u);
            const float tx = fx - float(i0), tz = fz - float(j0);
            const float* D = &v[kHeaderWords + size_t(k) * res * res];
            const size_t a = size_t(j0) * res + i0;
            const float lo = D[a] + (D[a + 1] - D[a]) * tx;
            const float hi = D[a + res] + (D[a + res + 1] - D[a + res]) * tx;
            return lo + (hi - lo) * tz;
        };
        const float* hs = &v[8];
        const float* km = &v[12];
        const std::array<float, 3> amp = bendAmp;
        auto gainAt = [&](float d) {
            std::array<float, 3> g = amp;
            if (d >= 60.f) return g;
            d = std::max(d, 0.f);
            constexpr float kGamma = 0.42f, kShare = 0.5f, kMax = 1.15f;
            float budget = kGamma * kGamma * d * d;
            for (int c = 2; c >= 0; --c) {
                float s = 1.f;
                if (km[c] > 0.f) {
                    const float kd = std::clamp(km[c] * d, 1e-4f, 10.f);
                    const float ks = 1.f / std::sqrt(std::tanh(kd) * (1.f + 2.f * kd / std::sinh(2.f * kd)));
                    s = std::min(1.f + (ks - 1.f) * kShare, kMax);
                }
                const float e    = hs[c] * hs[c] * s * s * (amp[c] * amp[c]);
                const float take = std::min(e, budget);
                budget -= take;
                g[c] = e > 1e-12f ? amp[c] * s * std::sqrt(take / e) : amp[c] * s;
            }
            return g;
        };
        float wN = 0.f, wF = 0.f;
        const float dN = depthIn(0u, wN);
        const float dF = depthIn(1u, wF);
        std::array<float, 3> g = amp;
        if (wF > 0.f) {
            const std::array<float, 3> f = gainAt(dF);
            for (int c = 0; c < 3; ++c) g[c] = amp[c] + (f[c] - amp[c]) * wF;
        }
        if (wN > 0.f) {
            const std::array<float, 3> nn = gainAt(dN);
            for (int c = 0; c < 3; ++c) g[c] = g[c] + (nn[c] - g[c]) * wN;
        }
        return g;
    }

    float DisplacedMesh::shoreDepthAt(float x, float z) const {
        constexpr size_t kHeaderWords = 20;
        const std::vector<float>& v = shoreDepth.data;
        if (v.size() < kHeaderWords) return -1.f;
        uint32_t res = 0;
        std::memcpy(&res, &v[16], sizeof(res));
        if (res < 2u || v.size() < kHeaderWords + 2u * size_t(res) * res) return -1.f;
        for (uint32_t k = 0; k < 2u; ++k) {
            const float* w = &v[k * 4u];
            if (w[3] < 0.5f) continue;
            const int i = static_cast<int>(std::floor((x - w[0]) / w[2]));
            const int j = static_cast<int>(std::floor((z - w[1]) / w[2]));
            if (i < 0 || j < 0 || i >= int(res) || j >= int(res)) continue;
            return v[kHeaderWords + (size_t(k) * res + size_t(j)) * res + size_t(i)];
        }
        return -1.f;
    }

    DisplacedMesh::ShoreBend DisplacedMesh::shoreBend(float x, float z) const {
        // ocean_shore_depth.glsl's oceanShoreBendAt and oceanShoreBend: the
        // bend's table after the depths, the near window's over the far
        // one's (over a quarter of its side), the far one's over the open sea
        // (over a sixteenth).
        ShoreBend out;
        constexpr size_t kHeaderWords = 20, kBendHeaderWords = 12;
        const std::vector<float>& v = shoreDepth.data;
        if (v.size() < kHeaderWords) return out;
        uint32_t res = 0, n = 0;
        std::memcpy(&res, &v[16], sizeof(res));
        std::memcpy(&n, &v[17], sizeof(n));
        const size_t base = kHeaderWords + 2u * size_t(res) * res;
        if (n < 2u || v.size() < base + kBendHeaderWords + 2u * size_t(n) * n * 4u) return out;
        const float* T = &v[base];
        if (T[3] < 0.5f && T[7] < 0.5f) return out;
        const float wx = T[8], wz = T[9];
        const float fn = static_cast<float>(n);
        // The chop's table (the near window's, half floats), after both windows' cells.
        uint32_t chopLaid = 0;
        std::memcpy(&chopLaid, &T[11], sizeof(chopLaid));
        const size_t chopBase = base + kBendHeaderWords + 2u * size_t(n) * n * 4u;
        const bool   chopOn   = chopLaid != 0u && v.size() >= chopBase + size_t(n) * n;
        auto halves = [&](size_t cell, float& lo, float& hi) {
            uint32_t u = 0;
            std::memcpy(&u, &v[chopBase + cell], sizeof(u));
            auto half = [](uint32_t h) {
                const uint32_t sign = (h >> 15u) & 1u, exp = (h >> 10u) & 31u, man = h & 1023u;
                float f;
                if (exp == 0u) f = std::ldexp(float(man), -24);
                else if (exp == 31u) f = man ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
                else f = std::ldexp(float(man | 1024u), int(exp) - 25);
                return sign ? -f : f;
            };
            lo = half(u & 0xffffu);
            hi = half(u >> 16u);
            return u;
        };
        // Window k's (dP0, dP1, A0, A1) and its weight.
        auto windowAt = [&](uint32_t k, float px, float pz, float blend, float* val) {
            val[0] = val[1] = 0.f;
            val[2] = val[3] = 1.f;
            const float* w = &T[k * 4u];
            if (w[3] < 0.5f) return 0.f;
            const float u = (px - w[0]) / (w[2] * fn);
            const float t = (pz - w[1]) / (w[2] * fn);
            const float e = std::min(std::min(u, 1.f - u), std::min(t, 1.f - t));
            const float s = std::clamp(e / blend, 0.f, 1.f);
            const float wgt = s * s * (3.f - 2.f * s);
            if (wgt <= 0.f) return 0.f;
            const float fx = std::clamp(u * fn - 0.5f, 0.f, fn - 1.f);
            const float fz = std::clamp(t * fn - 0.5f, 0.f, fn - 1.f);
            const uint32_t i0 = std::min(static_cast<uint32_t>(fx), n - 2u);
            const uint32_t j0 = std::min(static_cast<uint32_t>(fz), n - 2u);
            const float tx = fx - float(i0), tz = fz - float(j0);
            const float* C = &T[kBendHeaderWords + ((size_t(k) * n + j0) * n + i0) * 4u];
            const size_t r = size_t(n) * 4u;
            for (int c = 0; c < 4; ++c) {
                const float a = C[c], b = C[4 + c], cc = C[r + c], d = C[r + 4 + c];
                const float lo = a + (b - a) * tx, hi = cc + (d - cc) * tx;
                val[c] = (a == b && cc == d && a == cc) ? a : lo + (hi - lo) * tz;
            }
            return wgt;
        };
        auto at = [&](float px, float pz, float* P, float* A) {
            float vN[4], vF[4];
            const float wN = windowAt(0u, px, pz, 0.25f, vN);
            const float wF = windowAt(1u, px, pz, 1.f / 16.f, vF);
            for (int c = 0; c < 2; ++c) {
                P[c] = wF * vF[c];
                A[c] = 1.f + wF * (vF[2 + c] - 1.f);
                P[c] = P[c] + wN * (vN[c] - P[c]);
                A[c] = A[c] + wN * (vN[2 + c] - A[c]);
            }
            // The chop: the near window's alone (ocean_shore_depth.glsl's oceanShoreChopIn).
            P[2] = 0.f;
            A[2] = 1.f;
            if (chopOn && wN > 0.f) {
                const float* w = &T[0];
                const float fx = std::clamp((px - w[0]) / w[2] - 0.5f, 0.f, fn - 1.f);
                const float fz = std::clamp((pz - w[1]) / w[2] - 0.5f, 0.f, fn - 1.f);
                const uint32_t i0 = std::min(static_cast<uint32_t>(fx), n - 2u);
                const uint32_t j0 = std::min(static_cast<uint32_t>(fz), n - 2u);
                const float tx = fx - float(i0), tz = fz - float(j0);
                const size_t a = size_t(j0) * n + i0;
                float p00, a00, p10, a10, p01, a01, p11, a11;
                const uint32_t u00 = halves(a, p00, a00), u10 = halves(a + 1, p10, a10);
                const uint32_t u01 = halves(a + n, p01, a01), u11 = halves(a + n + 1, p11, a11);
                float vp = p00, va = a00;
                if (!(u00 == u10 && u01 == u11 && u00 == u01)) {
                    const float plo = p00 + (p10 - p00) * tx, phi = p01 + (p11 - p01) * tx;
                    const float alo = a00 + (a10 - a00) * tx, ahi = a01 + (a11 - a01) * tx;
                    vp = plo + (phi - plo) * tz;
                    va = alo + (ahi - alo) * tz;
                }
                P[2] = wN * vp;
                A[2] = 1.f + wN * (va - 1.f);
            }
            const float cF = T[7] > 0.5f ? T[6] : T[2];
            const float cN = T[3] > 0.5f ? T[2] : cF;
            return std::max(cF + (cN - cF) * wN, 0.05f);
        };
        float P[3], A[3], xp[3], xm[3], zp[3], zm[3], aT[3];
        const float h = at(x, z, P, A);
        out.amp = {A[0], A[1], A[2]};
        at(x + h, z, xp, aT);
        at(x - h, z, xm, aT);
        at(x, z + h, zp, aT);
        at(x, z - h, zm, aT);
        for (int c = 0; c < 3; ++c) {
            const bool any = P[c] != 0.f || xp[c] != 0.f || xm[c] != 0.f || zp[c] != 0.f || zm[c] != 0.f;
            if (c < 2) out.on = out.on || any;
            else       out.on2 = any;
        }
        for (int c = 0; c < 3; ++c) {
            if (!(c < 2 ? out.on : out.on2)) continue;
            // The waves' own heading here, grad P = w + grad dP, against the sea's.
            const float nx = wx + (xp[c] - xm[c]) / (2.f * h);
            const float nz = wz + (zp[c] - zm[c]) / (2.f * h);
            const float l  = std::hypot(nx, nz);
            const float tc = l > 1e-6f ? (nx * wx + nz * wz) / l : 1.f;
            const float ts = l > 1e-6f ? (wx * nz - wz * nx) / l : 0.f;
            if (c < 2) {
                out.d[2 * c]        = wx * P[c];
                out.d[2 * c + 1]    = wz * P[c];
                out.turn[2 * c]     = tc;
                out.turn[2 * c + 1] = ts;
            } else {
                out.d2    = {wx * P[c], wz * P[c]};
                out.turn2 = {tc, ts};
            }
        }
        return out;
    }

    bool DisplacedMesh::defaultRippleHeights() {
        static const bool on = [] {
            const char* v = std::getenv("THREEPP_WAKE_HEIGHTS");
            return !(v && v[0] == '0' && v[1] == '\0');
        }();
        return on;
    }

    bool DisplacedMesh::defaultRippleBreaking() {
        static const bool on = [] {
            const char* v = std::getenv("THREEPP_WAKE_BREAKING");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        return on;
    }

}// namespace threepp
