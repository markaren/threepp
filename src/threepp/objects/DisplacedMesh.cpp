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
        const float chop = effectiveChoppiness();
        const float chop01 = chop * (1.f + effectiveCrestSharpness());
        // The wave climate's gains at a rest point (waveClimateGain), on every
        // field of its cascade, as water_displace.comp applies them, times
        // the gains over the shore's depth (shoreGain).
        const bool climate = hasWaveClimate() || !shoreDepth.data.empty();
        auto gains = [&](float qx, float qz) {
            std::array<float, 3> g = hasWaveClimate() ? waveClimateGain(qx, qz) : std::array<float, 3>{1.f, 1.f, 1.f};
            if (!shoreDepth.data.empty()) {
                const std::array<float, 3> s = shoreGain(qx, qz);
                for (int c = 0; c < 3; ++c) g[c] *= s[c];
            }
            return g;
        };
        // The long cascades' turn over the bed (shoreBend): cascade i < 2 is
        // read at q + its departure, its displacement turned with its crests.
        const bool bent = !shoreDepth.data.empty();
        auto bendAt = [&](float qx, float qz) { return bent ? shoreBend(qx, qz) : ShoreBend{}; };
        auto displacement = [&](float qx, float qz) {
            Vec2f d;
            const std::array<float, 3> g = climate ? gains(qx, qz) : std::array<float, 3>{1.f, 1.f, 1.f};
            const ShoreBend b = bendAt(qx, qz);
            for (uint32_t i = 0; i < 3; ++i) {
                if ((cascadeMask & (1u << i)) == 0u) continue;
                const CascadeField& cf = heightFields[i];
                if (!hasField(cf, cf.disp)) continue;
                const float invTile = 1.f / cf.tileSize;
                const bool  turn = b.on && i < 2u;
                const Vec2f p = turn ? cascadeDomain(i, qx + b.d[2 * i], qz + b.d[2 * i + 1]) : cascadeDomain(i, qx, qz);
                Vec2f v = sampleField(cf.disp, cf.dim, p.x * invTile, p.y * invTile,
                                      cascadeBicubic(i));
                if (i == 1) {
                    // Domain vector back to world; mirror of oceanC1World().
                    v = {kOceanCascade1RotCos * v.x + kOceanCascade1RotSin * v.y,
                         -kOceanCascade1RotSin * v.x + kOceanCascade1RotCos * v.y};
                }
                if (turn) {
                    const float c = b.turn[2 * i], s = b.turn[2 * i + 1];
                    v = {c * v.x - s * v.y, s * v.x + c * v.y};
                }
                // Sharpness 0 keeps the historical sum, bit for bit.
                const float c = (chop01 == chop) ? 1.f : (i < 2 ? chop01 : chop) / chop;
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
        const ShoreBend bq = bendAt(bestX, bestZ);
        for (uint32_t i = 0; i < 3; ++i) {
            if ((cascadeMask & (1u << i)) == 0u) continue;
            const CascadeField& cf = heightFields[i];
            if (!hasField(cf, cf.data)) continue;
            const float invTile = 1.f / cf.tileSize;
            const Vec2f p = (bq.on && i < 2u) ? cascadeDomain(i, bestX + bq.d[2 * i], bestZ + bq.d[2 * i + 1])
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
                bound += 0.5f * cf.boundKm * (e * e - rx * rx - rz * rz);
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
        const ShoreBend bend = shoreBend(x, z);
        const std::array<float, 3> amp{bend.amp[0], bend.amp[1], 1.f};
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
            const float cF = T[7] > 0.5f ? T[6] : T[2];
            const float cN = T[3] > 0.5f ? T[2] : cF;
            return std::max(cF + (cN - cF) * wN, 0.05f);
        };
        float P[2], A[2], xp[2], xm[2], zp[2], zm[2], aT[2];
        const float h = at(x, z, P, A);
        out.amp = {A[0], A[1]};
        at(x + h, z, xp, aT);
        at(x - h, z, xm, aT);
        at(x, z + h, zp, aT);
        at(x, z - h, zm, aT);
        for (int c = 0; c < 2; ++c) {
            out.on = out.on || P[c] != 0.f || xp[c] != 0.f || xm[c] != 0.f || zp[c] != 0.f || zm[c] != 0.f;
        }
        if (!out.on) return out;
        for (int c = 0; c < 2; ++c) {
            out.d[2 * c]     = wx * P[c];
            out.d[2 * c + 1] = wz * P[c];
            // The waves' own heading here, grad P = w + grad dP, against the sea's.
            const float nx = wx + (xp[c] - xm[c]) / (2.f * h);
            const float nz = wz + (zp[c] - zm[c]) / (2.f * h);
            const float l  = std::hypot(nx, nz);
            out.turn[2 * c]     = l > 1e-6f ? (nx * wx + nz * wz) / l : 1.f;
            out.turn[2 * c + 1] = l > 1e-6f ? (wx * nz - wz * nx) / l : 0.f;
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
