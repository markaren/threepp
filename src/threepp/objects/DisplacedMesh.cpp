#include "threepp/objects/DisplacedMesh.hpp"

// kOceanCascade1Rot* — cascade-1 sample-domain rotation, shared with the GLSL
// samplers (water_displace.comp / foam_world.comp). Header is cross-language
// (pure constants + a POD struct); including it here keeps CPU/GPU parity
// literal-exact.
#include "threepp/renderers/vulkan/shaders/vulkan_shared.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace threepp {

    DisplacedMesh::DisplacedMesh(const std::shared_ptr<BufferGeometry>& geometry,
                                 const std::shared_ptr<Material>& material)
        : Mesh(geometry, material) {}

    const std::string& DisplacedMesh::type() const {
        static const std::string typeName = "DisplacedMesh";
        return typeName;
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
        // sampleBicubicR/RG in water_displace.comp evaluate with four bilinear
        // taps; otherwise bilinear. Returns the raw (unnormalized) R and G.
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
            if (!wake.enabled || hullExclusion.halfLength <= 0.f) return 0.f;

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
        // rest position q, choppiness applied.
        auto displacement = [&](float qx, float qz) {
            Vec2f d;
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
                d.x += v.x * invTile;
                d.y += v.y * invTile;
            }
            d.x *= params.choppiness;
            d.y *= params.choppiness;
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
        if (params.choppiness != 0.f) {
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
        for (uint32_t i = 0; i < 3; ++i) {
            if ((cascadeMask & (1u << i)) == 0u) continue;
            const CascadeField& cf = heightFields[i];
            if (!hasField(cf, cf.data)) continue;
            const float invTile = 1.f / cf.tileSize;
            const Vec2f p = cascadeDomain(i, bestX, bestZ);
            total += sampleField(cf.data, cf.dim, p.x * invTile, p.y * invTile,
                                 cascadeBicubic(i)).x * invTile;
        }
        return total * params.waveScale;
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
