#include "threepp/objects/DisplacedMesh.hpp"

#include <algorithm>
#include <cmath>

namespace threepp {

    DisplacedMesh::DisplacedMesh(const std::shared_ptr<BufferGeometry>& geometry,
                                 const std::shared_ptr<Material>& material)
        : Mesh(geometry, material) {}

    std::string DisplacedMesh::type() const {
        return "DisplacedMesh";
    }

    namespace {

        float sampleCascade(const DisplacedMesh::CascadeField& cf,
                            float worldX, float worldZ) {
            if (cf.data.empty() || cf.dim == 0 || cf.tileSize <= 0.f)
                return 0.f;

            const uint32_t dim = cf.dim;
            const float invTile = 1.f / cf.tileSize;
            const float u = worldX * invTile;
            const float v = -worldZ * invTile;

            auto wrap = [dim](float x) {
                const float fdim = float(dim);
                float w = std::fmod(x, fdim);
                if (w < 0.f) w += fdim;
                return w;
            };
            const float fx = wrap(u * float(dim));
            const float fz = wrap(v * float(dim));

            const uint32_t x0 = uint32_t(std::floor(fx)) % dim;
            const uint32_t z0 = uint32_t(std::floor(fz)) % dim;
            const uint32_t x1 = (x0 + 1) % dim;
            const uint32_t z1 = (z0 + 1) % dim;
            const float tx = fx - std::floor(fx);
            const float tz = fz - std::floor(fz);

            auto h = [&](uint32_t x, uint32_t z) {
                return cf.data[(z * dim + x) * 2u + 0u];
            };
            const float h00 = h(x0, z0);
            const float h10 = h(x1, z0);
            const float h01 = h(x0, z1);
            const float h11 = h(x1, z1);
            const float a = h00 * (1.f - tx) + h10 * tx;
            const float b = h01 * (1.f - tx) + h11 * tx;

            return (a * (1.f - tz) + b * tz) * invTile;
        }

    }// namespace

    float DisplacedMesh::sampleHeight(float worldX, float worldZ,
                                      uint32_t cascadeMask) const {
        float total = 0.f;
        for (uint32_t i = 0; i < 3; ++i) {
            if ((cascadeMask & (1u << i)) != 0u)
                total += sampleCascade(heightFields[i], worldX, worldZ);
        }
        return total * params.waveScale;
    }

    float DisplacedMesh::sampleWakeHeight(float worldX, float worldZ) const {
        // Mirrors water_displace.comp wake formulas closely enough that
        // floaters (buoys, etc.) bob through the rendered wake. Keep
        // in sync if the shader changes.
        //
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
            const float hullFade  = smoothstepF(0.f, 2.f,
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

}// namespace threepp
