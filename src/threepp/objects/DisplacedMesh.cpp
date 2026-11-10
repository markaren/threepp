#include "threepp/objects/DisplacedMesh.hpp"

// kOceanCascade1Rot* — cascade-1 sample-domain rotation, shared with the GLSL
// samplers (water_displace.comp / foam_world.comp). Header is cross-language
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

    namespace {
        // An ocean look switch's environment variable: exactly "0" is off,
        // "1", "on" or "true" is on; unset keeps `fallback`, and anything
        // else keeps it too, with a warning.
        bool envSwitch(const char* name, bool fallback) {
            const char* v = std::getenv(name);
            if (!v || !*v) return fallback;
            if (std::strcmp(v, "0") == 0) return false;
            if (std::strcmp(v, "1") == 0 || std::strcmp(v, "on") == 0 || std::strcmp(v, "true") == 0) return true;
            std::fprintf(stderr, "[DisplacedMesh] %s=%s not understood (0 is off; 1, on or true is on); it stays %s\n",
                         name, v, fallback ? "on" : "off");
            return fallback;
        }
    }// namespace

    bool defaultChopNormals() {
        static const bool value = envSwitch("THREEPP_OCEAN_CHOP_NORMALS", false);
        return value;
    }

    bool defaultMeasuredSpreading() {
        static const bool value = envSwitch("THREEPP_OCEAN_SPREAD", false);
        return value;
    }

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

    bool DisplacedMesh::breakingCrestsOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_BREAKING_CRESTS");
        return env < 0 ? params.breakingCrests : env == 1;
    }

    bool DisplacedMesh::crestFoamOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_CREST_FOAM");
        return env < 0 ? params.crestFoam : env == 1;
    }

    bool DisplacedMesh::waveReflectionsOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_WAVE_REFLECTIONS");
        return env < 0 ? params.waveReflections : env == 1;
    }

    bool DisplacedMesh::shortWavesOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_SHORT_WAVES");
        return env < 0 ? params.shortWaves : env == 1;
    }

    bool DisplacedMesh::detileOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_DETILE");
        return env < 0 ? params.detile : env == 1;
    }

    bool DisplacedMesh::shortWaveSpectrumOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_SHORT_WAVE_SPECTRUM");
        return env < 0 ? params.shortWaveSpectrum : env == 1;
    }

    bool DisplacedMesh::waveModulationOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_WAVE_MODULATION");
        return env < 0 ? params.waveModulation : env == 1;
    }

    bool DisplacedMesh::hullContactOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_HULL_CONTACT");
        return env < 0 ? params.hullContact : env == 1;
    }

    float DisplacedMesh::effectiveChoppiness() const {
        return cwmOn() ? 1.f : params.choppiness;
    }

    bool DisplacedMesh::foldFoamOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_FOLD_FOAM");
        return env < 0 ? params.foldFoam : env == 1;
    }

    bool DisplacedMesh::crestLineFoamOn() const {
        static const int env = envSwitch("THREEPP_OCEAN_CREST_LINE_FOAM");
        return foldFoamOn() && (env < 0 ? params.crestLineFoam : env == 1);
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

        // One vessel's wake height at a world XZ: the per-vessel block of
        // water_displace.comp, faded only inside this vessel's own footprint.
        float vesselWakeHeight(const DisplacedMesh::HullExclusion& hullExclusion,
                               const DisplacedMesh::VesselWake& wake,
                               float worldX, float worldZ) {
            // Two contributions: a per-frame bow bump (current pose only,
            // small) and the V-wedge ridge height that diverges from the
            // bow at ~20° half-angle. The V-wedge is taken as MAX over the
            // historical trail samples — same combiner as the shader, so
            // 60 samples don't stack into fountain-shaped pillars.
            if (!wake.enabled || !wake.analyticWaves || hullExclusion.halfLength <= 0.f) return 0.f;

            auto smoothstepF = [](float a, float b, float x) {
                const float t = std::clamp((x - a) / (b - a), 0.f, 1.f);
                return t * t * (3.f - 2.f * t);
            };

            // Hull margin: mirrors ocean_cascade.glsl oceanHullMargin (0.6 x half-beam
            // in [0.3, 2] m).
            const float hullMargin = std::clamp(0.6f * hullExclusion.halfBeam, 0.3f, 2.f);

            auto vWedgeAtPose = [&](float cx, float cz, float sinYaw, float cosYaw,
                                    float speed, float ageFade) -> float {
                const float spd  = std::abs(speed);
                const float gate = smoothstepF(0.5f, 1.5f, spd);
                if (gate <= 0.f) return 0.f;
                const float dx = worldX - cx;
                const float dz = worldZ - cz;
                const float lX = cosYaw * dx - sinYaw * dz;
                const float lZ = sinYaw * dx + cosYaw * dz;
                const float distAft = hullExclusion.halfLength - lZ;
                if (distAft <= 0.f || distAft > hullExclusion.halfLength * 6.f) return 0.f;
                // Per-pose hullFade — V-wedge only fires outside the hull
                // footprint, so a buoy passing close to the boat doesn't get
                // lifted by the wake formula under the hull itself.
                const float uHull = std::clamp(-lZ / hullExclusion.halfLength, -1.f, 1.f);
                float halfBeamAtLZ;
                if (uHull <= 0.f) {
                    halfBeamAtLZ = hullExclusion.halfBeam *
                                   std::pow(std::max(1.f - uHull * uHull, 0.f), 0.6f);
                } else {
                    halfBeamAtLZ = hullExclusion.halfBeam * (1.f - 0.25f * uHull * uHull);
                }
                const float hullEdgeX = std::abs(lX) - halfBeamAtLZ;
                const float hullEdgeZ = std::abs(lZ) - hullExclusion.halfLength;
                const float hullFade  = smoothstepF(0.f, hullMargin,
                                                    std::max(hullEdgeX, hullEdgeZ));
                if (hullFade <= 0.f) return 0.f;
                const float tanAV = 0.36f;
                const float expectedX = tanAV * distAft;
                const float dRidge = std::abs(std::abs(lX) - expectedX);
                const float sigmaR = 0.6f + 0.05f * distAft;
                const float ridge  = std::exp(-(dRidge * dRidge) / (sigmaR * sigmaR));
                const float alongDecay = std::exp(-distAft /
                                                  (hullExclusion.halfLength * 4.f));
                const float vAmp = std::clamp(0.016f * spd * spd + 0.06f * spd,
                                              0.f, 1.0f);
                return gate * vAmp * ridge * alongDecay * ageFade * hullFade;
            };

            // Current-pose bow bump (small, doesn't pile up across the trail).
            float h = 0.f;
            {
                const float spd  = std::abs(wake.forwardSpeed);
                const float gate = smoothstepF(0.5f, 1.5f, spd);
                if (gate > 0.f) {
                    const float dx = worldX - hullExclusion.centerX;
                    const float dz = worldZ - hullExclusion.centerZ;
                    const float lX = hullExclusion.cosYaw * dx - hullExclusion.sinYaw * dz;
                    const float lZ = hullExclusion.sinYaw * dx + hullExclusion.cosYaw * dz;
                    const float distFromBow = lZ - hullExclusion.halfLength;
                    const float bowR = 1.5f;
                    const float bowG = std::exp(-(distFromBow * distFromBow) /
                                                (bowR * bowR));
                    const float bowL = std::exp(-(lX * lX) /
                                                (hullExclusion.halfBeam *
                                                 hullExclusion.halfBeam));
                    const float bowAmp = std::clamp(0.012f * spd * spd, 0.f, 0.4f);
                    h += gate * bowAmp * bowG * bowL;
                }
            }

            // Historical V-wedge ridge — MAX over the trail samples.
            float vWedgeMax = 0.f;
            if (!wake.trail.empty()) {
                for (const auto& s : wake.trail) {
                    const float ageFade = std::exp(-s.age / 5.f);
                    vWedgeMax = std::max(vWedgeMax,
                                         vWedgeAtPose(s.worldX, s.worldZ,
                                                      s.sinYaw, s.cosYaw,
                                                      s.speed, ageFade));
                }
            } else {
                vWedgeMax = vWedgeAtPose(hullExclusion.centerX,
                                         hullExclusion.centerZ,
                                         hullExclusion.sinYaw,
                                         hullExclusion.cosYaw,
                                         wake.forwardSpeed, 1.f);
            }
            h += vWedgeMax;
            return h;
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
        // field of its cascade, as water_displace.comp applies them.
        const bool climate = hasWaveClimate();
        auto displacement = [&](float qx, float qz) {
            Vec2f d;
            const std::array<float, 3> g = climate ? waveClimateGain(qx, qz) : std::array<float, 3>{1.f, 1.f, 1.f};
            for (uint32_t i = 0; i < 3; ++i) {
                if ((cascadeMask & (1u << i)) == 0u) continue;
                const CascadeField& cf = heightFields[i];
                if (!hasField(cf, cf.disp)) continue;
                const float invTile = 1.f / cf.tileSize;
                const Vec2f p = cascadeDomain(i, qx, qz);
                Vec2f v = sampleField(cf.disp, cf.dim, p.x * invTile, p.y * invTile,
                                      cascadeBicubic(i));
                if (i == 1) {
                    // Domain vector back to world; mirror of oceanC1World().
                    v = {kOceanCascade1RotCos * v.x + kOceanCascade1RotSin * v.y,
                         -kOceanCascade1RotSin * v.x + kOceanCascade1RotCos * v.y};
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
        const std::array<float, 3> gq = climate ? waveClimateGain(bestX, bestZ) : std::array<float, 3>{1.f, 1.f, 1.f};
        for (uint32_t i = 0; i < 3; ++i) {
            if ((cascadeMask & (1u << i)) == 0u) continue;
            const CascadeField& cf = heightFields[i];
            if (!hasField(cf, cf.data)) continue;
            const float invTile = 1.f / cf.tileSize;
            const Vec2f p = cascadeDomain(i, bestX, bestZ);
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

    bool DisplacedMesh::defaultRippleHeights() {
        static const bool on = [] {
            const char* v = std::getenv("THREEPP_WAKE_HEIGHTS");
            return v && v[0] == '1' && v[1] == '\0';
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

    float DisplacedMesh::sampleWakeHeight(float worldX, float worldZ) const {
        // Mirrors water_displace.comp wake formulas closely enough that
        // floaters (buoys, etc.) bob through the rendered wake. Keep
        // in sync if the shader changes. Every vessel's wake adds; see
        // the header for why another vessel's footprint does not fade it.
        float h = 0.f;
        for (uint32_t i = 0; i < kMaxVessels; ++i)
            h += vesselWakeHeight(vesselHull(i), vesselWake(i), worldX, worldZ);
        return h;
    }

}// namespace threepp
